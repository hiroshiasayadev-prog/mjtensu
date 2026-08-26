from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sqlite3
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 64
DEFAULT_JP_TRAIN_PER_GROUP = 5000
DEFAULT_MANUAL_TRAIN_FRACTION = 0.80
DEFAULT_MANUAL_TRAIN_REPEAT = 20
GROUPS = tuple((suit, is_red) for suit in "mps" for is_red in (0, 1))

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE experiment_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE sample (
    sample_id               TEXT PRIMARY KEY,
    split                   TEXT NOT NULL CHECK (split IN ('train', 'jp_val', 'jp_test', 'manual_val')),
    source                  TEXT NOT NULL CHECK (source IN ('jp', 'manual')),
    source_partition        TEXT NOT NULL,
    suit                    TEXT NOT NULL CHECK (suit IN ('m', 'p', 's')),
    is_red                  INTEGER NOT NULL CHECK (is_red IN (0, 1)),
    source_label            TEXT NOT NULL,
    crop_id                 TEXT NOT NULL,
    image_size              INTEGER NOT NULL,
    image_rgb_u8            BLOB NOT NULL,
    train_repeat            INTEGER NOT NULL CHECK (train_repeat >= 1),
    original_width          INTEGER NOT NULL,
    original_height         INTEGER NOT NULL,
    source_image_path       TEXT NOT NULL,
    source_image_id         TEXT,
    source_annotation_id    TEXT NOT NULL,
    capture_id              TEXT,
    layout_id               TEXT,
    region                  TEXT,
    brightness              TEXT,
    shadow                  TEXT,
    annotation_angle_deg    REAL NOT NULL,
    expected_rotation_deg   INTEGER NOT NULL
);

CREATE INDEX idx_sample_split_red_suit
ON sample(split, is_red, suit);

CREATE INDEX idx_sample_split_source
ON sample(split, source);

CREATE INDEX idx_sample_capture
ON sample(capture_id);
"""

SOURCE_COLUMNS = (
    "crop_id",
    "source",
    "source_partition",
    "suit",
    "is_red",
    "source_label",
    "image_width",
    "image_height",
    "image_png",
    "source_image_path",
    "source_image_id",
    "source_annotation_id",
    "capture_id",
    "layout_id",
    "region",
    "brightness",
    "shadow",
    "annotation_angle_deg",
    "expected_rotation_deg",
)


@dataclass(frozen=True)
class SelectedRow:
    split: str
    values: tuple[Any, ...]
    train_repeat: int


@dataclass(frozen=True)
class PreparedRow:
    values: tuple[Any, ...]


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic 64x64 RGB binary red-five classifier dataset. "
            "JP train is balanced by suit and red state; JP valid/test are retained; "
            "manual captures are split at capture level."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--source-database",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/"
            "red_five_datasets/red_five_all.sqlite."
        ),
    )
    parser.add_argument(
        "--output-database",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/"
            "red_five_datasets/rgb64_binary_jp5000_seed42.sqlite."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--jp-train-per-group",
        type=int,
        default=DEFAULT_JP_TRAIN_PER_GROUP,
        help="JP train samples per (suit, is_red) group. Defaults to 5000.",
    )
    parser.add_argument(
        "--manual-train-fraction",
        type=float,
        default=DEFAULT_MANUAL_TRAIN_FRACTION,
    )
    parser.add_argument(
        "--manual-train-repeat",
        type=int,
        default=DEFAULT_MANUAL_TRAIN_REPEAT,
        help=(
            "Repeat hint stored on manual train rows for the later DataLoader. "
            "Rows are not physically duplicated. Defaults to 20."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="Parallel PNG decode/resize workers.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size < 16:
        raise ValueError("--image-size must be at least 16")
    if args.jp_train_per_group < 1:
        raise ValueError("--jp-train-per-group must be positive")
    if not 0.0 < args.manual_train_fraction < 1.0:
        raise ValueError("--manual-train-fraction must be between 0 and 1")
    if args.manual_train_repeat < 1:
        raise ValueError("--manual-train-repeat must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    repository_root = args.repository_root.resolve()
    source_database = (
        args.source_database.resolve()
        if args.source_database is not None
        else repository_root
        / ".local"
        / "recognition"
        / "red_five_datasets"
        / "red_five_all.sqlite"
    )
    output_database = (
        args.output_database.resolve()
        if args.output_database is not None
        else repository_root
        / ".local"
        / "recognition"
        / "red_five_datasets"
        / f"rgb{args.image_size}_binary_jp{args.jp_train_per_group}_seed{args.seed}.sqlite"
    )

    summary = build_red_five_classifier_dataset(
        source_database=source_database,
        output_database=output_database,
        seed=int(args.seed),
        image_size=int(args.image_size),
        jp_train_per_group=int(args.jp_train_per_group),
        manual_train_fraction=float(args.manual_train_fraction),
        manual_train_repeat=int(args.manual_train_repeat),
        workers=int(args.workers),
        force=bool(args.force),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_red_five_classifier_dataset(
    *,
    source_database: Path,
    output_database: Path,
    seed: int = DEFAULT_SEED,
    image_size: int = DEFAULT_IMAGE_SIZE,
    jp_train_per_group: int = DEFAULT_JP_TRAIN_PER_GROUP,
    manual_train_fraction: float = DEFAULT_MANUAL_TRAIN_FRACTION,
    manual_train_repeat: int = DEFAULT_MANUAL_TRAIN_REPEAT,
    workers: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    source_database = source_database.resolve()
    output_database = output_database.resolve()
    if not source_database.is_file():
        raise FileNotFoundError(source_database)
    if source_database == output_database:
        raise ValueError("Source and output database paths must differ")
    if output_database.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_database}. Use --force to replace it.")

    output_database.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_database_path(output_database)
    try:
        summary = _create_dataset(
            source_database=source_database,
            output_database=temporary_path,
            seed=seed,
            image_size=image_size,
            jp_train_per_group=jp_train_per_group,
            manual_train_fraction=manual_train_fraction,
            manual_train_repeat=manual_train_repeat,
            workers=workers,
        )
        remove_sqlite_sidecars(output_database)
        if output_database.exists():
            output_database.unlink()
        os.replace(temporary_path, output_database)
    except Exception:
        remove_sqlite_files(temporary_path)
        raise

    summary["database"] = str(output_database)
    summary["database_bytes"] = output_database.stat().st_size
    atomic_write_json(output_database.with_suffix(".summary.json"), summary)
    return summary


def _create_dataset(
    *,
    source_database: Path,
    output_database: Path,
    seed: int,
    image_size: int,
    jp_train_per_group: int,
    manual_train_fraction: float,
    manual_train_repeat: int,
    workers: int,
) -> dict[str, Any]:
    source = sqlite3.connect(sqlite_readonly_uri(source_database), uri=True, timeout=60)
    source.row_factory = sqlite3.Row
    target = sqlite3.connect(output_database, timeout=60)
    target.row_factory = sqlite3.Row
    try:
        validate_source_schema(source)
        target.execute("PRAGMA journal_mode = WAL")
        target.execute("PRAGMA synchronous = NORMAL")
        target.execute("PRAGMA temp_store = MEMORY")
        target.executescript(SCHEMA)

        manual_split = choose_manual_capture_split(
            source,
            seed=seed,
            train_fraction=manual_train_fraction,
        )
        selected: list[SelectedRow] = []
        selected.extend(
            select_jp_train_rows(
                source,
                samples_per_group=jp_train_per_group,
                seed=seed,
            )
        )
        selected.extend(select_jp_partition_rows(source, "valid", "jp_val"))
        selected.extend(select_jp_partition_rows(source, "test", "jp_test"))
        selected.extend(
            select_manual_rows(
                source,
                capture_split=manual_split,
                train_repeat=manual_train_repeat,
            )
        )

        print(
            f"[red-five-classifier-dataset] selected {len(selected)} rows; "
            f"preprocessing RGB {image_size}x{image_size} with {workers} workers"
        )
        insert_prepared_rows(
            target,
            selected,
            image_size=image_size,
            workers=workers,
        )

        source_stat = source_database.stat()
        metadata = {
            "schema_version": "1",
            "source_database": str(source_database),
            "source_database_size": str(source_stat.st_size),
            "source_database_mtime_ns": str(source_stat.st_mtime_ns),
            "seed": str(seed),
            "image_size": str(image_size),
            "jp_train_per_group": str(jp_train_per_group),
            "manual_train_fraction": repr(manual_train_fraction),
            "manual_train_repeat": str(manual_train_repeat),
            "preprocess": "rgb_aspect_preserving_letterbox_border_median_lanczos_u8",
            "target": "is_red_binary",
            "jp_split_policy": "train=balanced-reservoir;valid=all;jp-test=all",
            "manual_split_policy": "capture-level_stratified_by_brightness_shadow",
        }
        target.executemany(
            "INSERT INTO experiment_metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        target.commit()
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        counts = dataset_counts(target)
        return {
            "status": "completed",
            "database": str(output_database),
            "database_bytes": output_database.stat().st_size,
            "seed": seed,
            "image_size": image_size,
            "jp_train_per_group": jp_train_per_group,
            "manual_train_fraction": manual_train_fraction,
            "manual_train_repeat": manual_train_repeat,
            "manual_capture_split": {
                "train": sorted(k for k, v in manual_split.items() if v == "train"),
                "manual_val": sorted(k for k, v in manual_split.items() if v == "manual_val"),
            },
            **counts,
        }
    finally:
        target.close()
        source.close()


def validate_source_schema(connection: sqlite3.Connection) -> None:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sample'"
    ).fetchone()
    if table is None:
        raise ValueError("Source database has no sample table")
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(sample)")}
    missing = set(SOURCE_COLUMNS) - columns
    if missing:
        raise ValueError(f"Source sample table is missing columns: {sorted(missing)}")


def select_jp_train_rows(
    connection: sqlite3.Connection,
    *,
    samples_per_group: int,
    seed: int,
) -> list[SelectedRow]:
    reservoirs: dict[tuple[str, int], list[int]] = {group: [] for group in GROUPS}
    seen = {group: 0 for group in GROUPS}
    rngs = {
        group: random.Random(stable_seed(seed, f"jp-train:{group[0]}:{group[1]}"))
        for group in GROUPS
    }

    for row in connection.execute(
        """
        SELECT rowid, suit, is_red
        FROM sample
        WHERE source='jp' AND source_partition='train'
        ORDER BY rowid
        """
    ):
        group = (str(row["suit"]), int(row["is_red"]))
        if group not in reservoirs:
            continue
        count = seen[group]
        reservoir = reservoirs[group]
        if count < samples_per_group:
            reservoir.append(int(row["rowid"]))
        else:
            replacement = rngs[group].randrange(count + 1)
            if replacement < samples_per_group:
                reservoir[replacement] = int(row["rowid"])
        seen[group] = count + 1

    selected_rowids: list[int] = []
    for group in GROUPS:
        reservoir = reservoirs[group]
        if not reservoir:
            raise ValueError(f"No JP train samples for group {group}")
        if len(reservoir) < samples_per_group:
            print(
                f"[red-five-classifier-dataset/jp/train] {group}: "
                f"requested {samples_per_group}, available {len(reservoir)}"
            )
        else:
            print(
                f"[red-five-classifier-dataset/jp/train] {group}: "
                f"selected {len(reservoir)} of {seen[group]}"
            )
        selected_rowids.extend(reservoir)

    by_rowid = fetch_rows_by_rowid(connection, selected_rowids)
    return [
        SelectedRow(
            split="train",
            values=tuple(by_rowid[rowid][column] for column in SOURCE_COLUMNS),
            train_repeat=1,
        )
        for rowid in selected_rowids
    ]


def select_jp_partition_rows(
    connection: sqlite3.Connection,
    source_partition: str,
    split: str,
) -> list[SelectedRow]:
    rows = connection.execute(
        f"""
        SELECT {', '.join(SOURCE_COLUMNS)}
        FROM sample
        WHERE source='jp' AND source_partition=?
        ORDER BY crop_id
        """,
        (source_partition,),
    ).fetchall()
    return [
        SelectedRow(
            split=split,
            values=tuple(row[column] for column in SOURCE_COLUMNS),
            train_repeat=1,
        )
        for row in rows
    ]


def choose_manual_capture_split(
    connection: sqlite3.Connection,
    *,
    seed: int,
    train_fraction: float,
) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT capture_id,
               COALESCE(brightness, '') AS brightness,
               COALESCE(shadow, '') AS shadow,
               COUNT(*) AS sample_count
        FROM sample
        WHERE source='manual' AND capture_id IS NOT NULL
        GROUP BY capture_id, brightness, shadow
        ORDER BY capture_id
        """
    ).fetchall()
    if not rows:
        return {}

    strata: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        strata[(str(row["brightness"]), str(row["shadow"]))].append(str(row["capture_id"]))

    result: dict[str, str] = {}
    for stratum, capture_ids in sorted(strata.items()):
        ordered = sorted(set(capture_ids))
        rng = random.Random(stable_seed(seed, f"manual:{stratum!r}"))
        rng.shuffle(ordered)
        if len(ordered) == 1:
            validation_count = 0
        else:
            validation_count = max(1, int(round(len(ordered) * (1.0 - train_fraction))))
            validation_count = min(validation_count, len(ordered) - 1)
        validation = set(ordered[:validation_count])
        for capture_id in ordered:
            result[capture_id] = "manual_val" if capture_id in validation else "train"

    if result and not any(split == "manual_val" for split in result.values()):
        candidates = sorted(result)
        if len(candidates) > 1:
            holdout = candidates[stable_seed(seed, "manual-global-holdout") % len(candidates)]
            result[holdout] = "manual_val"
    return result


def select_manual_rows(
    connection: sqlite3.Connection,
    *,
    capture_split: dict[str, str],
    train_repeat: int,
) -> list[SelectedRow]:
    rows = connection.execute(
        f"""
        SELECT {', '.join(SOURCE_COLUMNS)}
        FROM sample
        WHERE source='manual'
        ORDER BY crop_id
        """
    ).fetchall()
    selected: list[SelectedRow] = []
    for row in rows:
        capture_id = row["capture_id"]
        if capture_id is None:
            raise ValueError(f"Manual crop {row['crop_id']} has no capture_id")
        split = capture_split.get(str(capture_id))
        if split is None:
            raise ValueError(f"No manual split assigned to capture {capture_id}")
        selected.append(
            SelectedRow(
                split=split,
                values=tuple(row[column] for column in SOURCE_COLUMNS),
                train_repeat=train_repeat if split == "train" else 1,
            )
        )
    return selected


def fetch_rows_by_rowid(
    connection: sqlite3.Connection,
    rowids: Sequence[int],
    *,
    chunk_size: int = 500,
) -> dict[int, sqlite3.Row]:
    result: dict[int, sqlite3.Row] = {}
    unique = sorted(set(int(rowid) for rowid in rowids))
    for start in range(0, len(unique), chunk_size):
        chunk = unique[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"""
            SELECT rowid AS _rowid, {', '.join(SOURCE_COLUMNS)}
            FROM sample
            WHERE rowid IN ({placeholders})
            """,
            chunk,
        ).fetchall()
        for row in rows:
            result[int(row["_rowid"])] = row
    missing = set(unique) - set(result)
    if missing:
        raise ValueError(f"Failed to fetch {len(missing)} selected source rows")
    return result


def insert_prepared_rows(
    connection: sqlite3.Connection,
    rows: Sequence[SelectedRow],
    *,
    image_size: int,
    workers: int,
    commit_interval: int = 1000,
) -> None:
    insert_sql = """
        INSERT INTO sample(
            sample_id, split, source, source_partition, suit, is_red, source_label,
            crop_id, image_size, image_rgb_u8, train_repeat,
            original_width, original_height, source_image_path, source_image_id,
            source_annotation_id, capture_id, layout_id, region, brightness, shadow,
            annotation_angle_deg, expected_rotation_deg
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def prepare(row: SelectedRow) -> PreparedRow:
        values = dict(zip(SOURCE_COLUMNS, row.values))
        rgb = preprocess_rgb_u8(bytes(values["image_png"]), image_size=image_size)
        return PreparedRow(
            values=(
                f"{row.split}:{values['crop_id']}",
                row.split,
                str(values["source"]),
                str(values["source_partition"]),
                str(values["suit"]),
                int(values["is_red"]),
                str(values["source_label"]),
                str(values["crop_id"]),
                image_size,
                rgb,
                row.train_repeat,
                int(values["image_width"]),
                int(values["image_height"]),
                str(values["source_image_path"]),
                None if values["source_image_id"] is None else str(values["source_image_id"]),
                str(values["source_annotation_id"]),
                None if values["capture_id"] is None else str(values["capture_id"]),
                None if values["layout_id"] is None else str(values["layout_id"]),
                None if values["region"] is None else str(values["region"]),
                None if values["brightness"] is None else str(values["brightness"]),
                None if values["shadow"] is None else str(values["shadow"]),
                float(values["annotation_angle_deg"]),
                int(values["expected_rotation_deg"]),
            )
        )

    batch: list[tuple[Any, ...]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for prepared in executor.map(prepare, rows, chunksize=32):
            batch.append(prepared.values)
            completed += 1
            if len(batch) >= commit_interval:
                connection.executemany(insert_sql, batch)
                connection.commit()
                batch.clear()
                print(f"[red-five-classifier-dataset] prepared {completed}/{len(rows)}")
    if batch:
        connection.executemany(insert_sql, batch)
        connection.commit()
    print(f"[red-five-classifier-dataset] prepared {completed}/{len(rows)}")


def preprocess_rgb_u8(image_png: bytes, *, image_size: int = DEFAULT_IMAGE_SIZE) -> bytes:
    with Image.open(io.BytesIO(image_png)) as source:
        source.load()
        image = source.convert("RGB")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid crop size: {image.size}")

    scale = min(image_size / width, image_size / height)
    resized_width = max(1, min(image_size, int(math.floor(width * scale + 0.5))))
    resized_height = max(1, min(image_size, int(math.floor(height * scale + 0.5))))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    pixels = np.asarray(image, dtype=np.uint8)
    if width == 1 or height == 1:
        border = pixels.reshape(-1, 3)
    else:
        border = np.concatenate(
            [pixels[0, :, :], pixels[-1, :, :], pixels[1:-1, 0, :], pixels[1:-1, -1, :]],
            axis=0,
        )
    fill = tuple(int(value) for value in np.median(border, axis=0))

    canvas = Image.new("RGB", (image_size, image_size), color=fill)
    offset_x = (image_size - resized_width) // 2
    offset_y = (image_size - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    raw = canvas.tobytes()
    expected = image_size * image_size * 3
    if len(raw) != expected:
        raise RuntimeError(f"Preprocessed RGB image has {len(raw)} bytes, expected {expected}")
    return raw


def dataset_counts(connection: sqlite3.Connection) -> dict[str, Any]:
    total = int(connection.execute("SELECT COUNT(*) FROM sample").fetchone()[0])
    by_split = {
        str(row["split"]): int(row["count"])
        for row in connection.execute(
            "SELECT split, COUNT(*) AS count FROM sample GROUP BY split ORDER BY split"
        )
    }
    by_split_source: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for row in connection.execute(
        """
        SELECT split, source, COUNT(*) AS count
        FROM sample GROUP BY split, source ORDER BY split, source
        """
    ):
        by_split_source[str(row["split"])][str(row["source"])] = int(row["count"])

    by_split_group: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for row in connection.execute(
        """
        SELECT split, suit, is_red, COUNT(*) AS count
        FROM sample GROUP BY split, suit, is_red ORDER BY split, suit, is_red
        """
    ):
        state = "red" if int(row["is_red"]) else "normal"
        by_split_group[str(row["split"])][f"{row['suit']}/{state}"] = int(row["count"])

    effective_train = int(
        connection.execute(
            "SELECT COALESCE(SUM(train_repeat), 0) FROM sample WHERE split='train'"
        ).fetchone()[0]
    )
    manual_conditions: defaultdict[str, int] = defaultdict(int)
    for row in connection.execute(
        """
        SELECT split, COALESCE(brightness, '') AS brightness,
               COALESCE(shadow, '') AS shadow, COUNT(*) AS count
        FROM sample
        WHERE source='manual'
        GROUP BY split, brightness, shadow
        ORDER BY split, brightness, shadow
        """
    ):
        key = f"{row['split']}|brightness={row['brightness']}|shadow={row['shadow']}"
        manual_conditions[key] = int(row["count"])

    return {
        "sample_count": total,
        "counts_by_split": by_split,
        "counts_by_split_and_source": dict(by_split_source),
        "counts_by_split_and_group": dict(by_split_group),
        "effective_train_samples_with_repeat": effective_train,
        "manual_condition_counts": dict(manual_conditions),
    }


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def sqlite_readonly_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def temporary_database_path(output_database: Path) -> Path:
    descriptor, path = tempfile.mkstemp(
        prefix=f".{output_database.name}.", suffix=".tmp.sqlite", dir=output_database.parent
    )
    os.close(descriptor)
    os.unlink(path)
    return Path(path)


def remove_sqlite_files(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    remove_sqlite_sidecars(path)


def remove_sqlite_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
