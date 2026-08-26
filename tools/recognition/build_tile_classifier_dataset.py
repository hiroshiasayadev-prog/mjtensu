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
DEFAULT_JP_TRAIN_PER_CLASS = 500
DEFAULT_JP_VALID_PER_CLASS = 200
DEFAULT_MANUAL_TRAIN_FRACTION = 0.80

BASE_LABELS = tuple(
    [f"{n}{s}" for s in "mps" for n in range(1, 10)]
    + ["east", "south", "west", "north", "white", "green", "red"]
)
BASE_LABEL_TO_INDEX = {label: index for index, label in enumerate(BASE_LABELS)}
RED_FIVE_TO_BASE = {
    "red5m": "5m",
    "red5p": "5p",
    "red5s": "5s",
}
BASE_TO_SOURCE_LABELS = {
    label: (label,) for label in BASE_LABELS
}
BASE_TO_SOURCE_LABELS.update(
    {
        "5m": ("5m", "red5m"),
        "5p": ("5p", "red5p"),
        "5s": ("5s", "red5s"),
    }
)

EXPERIMENT_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE experiment_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE sample (
    sample_id             TEXT PRIMARY KEY,
    split                 TEXT NOT NULL CHECK (split IN ('train', 'manual_val', 'jp_val')),
    source                TEXT NOT NULL CHECK (source IN ('jp', 'manual')),
    source_partition      TEXT NOT NULL,
    base_label            TEXT NOT NULL,
    class_index           INTEGER NOT NULL,
    original_label        TEXT NOT NULL,
    source_label          TEXT NOT NULL,
    quality_audit_decision TEXT,
    crop_id               TEXT NOT NULL,
    image_size            INTEGER NOT NULL,
    image_gray_u8         BLOB NOT NULL,
    original_width        INTEGER NOT NULL,
    original_height       INTEGER NOT NULL,
    source_image_path     TEXT NOT NULL,
    source_image_id       TEXT,
    source_annotation_id  TEXT NOT NULL,
    capture_id            TEXT,
    layout_id             TEXT,
    region                TEXT,
    brightness            TEXT,
    shadow                TEXT,
    annotation_angle_deg  REAL NOT NULL,
    expected_rotation_deg INTEGER NOT NULL
);

CREATE INDEX idx_sample_split_class
ON sample(split, class_index);

CREATE INDEX idx_sample_split_source_class
ON sample(split, source, class_index);

CREATE INDEX idx_sample_capture
ON sample(capture_id);
"""

SOURCE_COLUMNS = (
    "crop_id",
    "source",
    "source_partition",
    "tile_label",
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
class QualityReview:
    decision: str
    corrected_label: str | None


@dataclass(frozen=True)
class SelectedSourceRow:
    split: str
    values: tuple[Any, ...]
    effective_label: str
    quality_decision: str | None = None


@dataclass(frozen=True)
class PreparedSample:
    values: tuple[Any, ...]


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact, deterministic 34-class grayscale tile-classifier dataset. "
            "The persistent crop DB remains untouched. Red fives are merged into their "
            "base five class while the original label is preserved."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--source-database",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/"
            "tile_crop_dataset/dataset.sqlite."
        ),
    )
    parser.add_argument(
        "--output-database",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/"
            "tile_classifier_datasets/gray34_jp500_seed42.sqlite."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--jp-train-per-class",
        type=int,
        default=DEFAULT_JP_TRAIN_PER_CLASS,
    )
    parser.add_argument(
        "--jp-valid-per-class",
        type=int,
        default=DEFAULT_JP_VALID_PER_CLASS,
    )
    parser.add_argument(
        "--manual-train-fraction",
        type=float,
        default=DEFAULT_MANUAL_TRAIN_FRACTION,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="Parallel PNG decode/preprocess workers. SQLite writes remain serial.",
    )
    parser.add_argument(
        "--quality-audit-database",
        type=Path,
        help=(
            "Optional sidecar produced by review_tile_crop_label_audit.py. When omitted, "
            "<repository-root>/.local/recognition/tile_crop_dataset/quality_audit.sqlite "
            "is used automatically if it exists. label_error rows use corrected_label; "
            "unusable_crop/background rows are excluded; false_detection rows are kept."
        ),
    )
    parser.add_argument(
        "--ignore-quality-audit",
        action="store_true",
        help="Build without applying the default quality_audit.sqlite sidecar.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size < 16:
        raise ValueError("--image-size must be at least 16")
    if args.jp_train_per_class < 1:
        raise ValueError("--jp-train-per-class must be positive")
    if args.jp_valid_per_class < 1:
        raise ValueError("--jp-valid-per-class must be positive")
    if not 0.0 < args.manual_train_fraction < 1.0:
        raise ValueError("--manual-train-fraction must be between 0 and 1")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    repository_root = args.repository_root.resolve()
    source_database = (
        args.source_database.resolve()
        if args.source_database is not None
        else repository_root
        / ".local"
        / "recognition"
        / "tile_crop_dataset"
        / "dataset.sqlite"
    )
    output_database = (
        args.output_database.resolve()
        if args.output_database is not None
        else repository_root
        / ".local"
        / "recognition"
        / "tile_classifier_datasets"
        / f"gray34_jp{args.jp_train_per_class}_seed{args.seed}.sqlite"
    )
    default_quality_database = (
        repository_root
        / ".local"
        / "recognition"
        / "tile_crop_dataset"
        / "quality_audit.sqlite"
    )
    if args.ignore_quality_audit:
        quality_audit_database = None
    elif args.quality_audit_database is not None:
        quality_audit_database = args.quality_audit_database.resolve()
    else:
        quality_audit_database = (
            default_quality_database.resolve() if default_quality_database.is_file() else None
        )

    summary = build_classifier_dataset(
        source_database=source_database,
        output_database=output_database,
        seed=int(args.seed),
        image_size=int(args.image_size),
        jp_train_per_class=int(args.jp_train_per_class),
        jp_valid_per_class=int(args.jp_valid_per_class),
        manual_train_fraction=float(args.manual_train_fraction),
        workers=int(args.workers),
        quality_audit_database=quality_audit_database,
        force=bool(args.force),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_classifier_dataset(
    *,
    source_database: Path,
    output_database: Path,
    seed: int = DEFAULT_SEED,
    image_size: int = DEFAULT_IMAGE_SIZE,
    jp_train_per_class: int = DEFAULT_JP_TRAIN_PER_CLASS,
    jp_valid_per_class: int = DEFAULT_JP_VALID_PER_CLASS,
    manual_train_fraction: float = DEFAULT_MANUAL_TRAIN_FRACTION,
    workers: int = 1,
    quality_audit_database: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    source_database = source_database.resolve()
    output_database = output_database.resolve()
    if not source_database.is_file():
        raise FileNotFoundError(source_database)
    if source_database == output_database:
        raise ValueError("Source and output database paths must differ")
    quality_audit_database = (
        None if quality_audit_database is None else quality_audit_database.resolve()
    )
    if quality_audit_database is not None and not quality_audit_database.is_file():
        raise FileNotFoundError(quality_audit_database)
    if output_database.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_database}. Use --force only for this compact experiment DB."
        )

    output_database.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_database_path(output_database)
    try:
        summary = _create_classifier_dataset(
            source_database=source_database,
            output_database=temporary_path,
            seed=seed,
            image_size=image_size,
            jp_train_per_class=jp_train_per_class,
            jp_valid_per_class=jp_valid_per_class,
            manual_train_fraction=manual_train_fraction,
            workers=workers,
            quality_audit_database=quality_audit_database,
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
    summary_path = output_database.with_suffix(".summary.json")
    atomic_write_json(summary_path, summary)
    return summary


def _create_classifier_dataset(
    *,
    source_database: Path,
    output_database: Path,
    seed: int,
    image_size: int,
    jp_train_per_class: int,
    jp_valid_per_class: int,
    manual_train_fraction: float,
    workers: int,
    quality_audit_database: Path | None,
) -> dict[str, Any]:
    source = sqlite3.connect(sqlite_readonly_uri(source_database), uri=True, timeout=60)
    source.row_factory = sqlite3.Row
    target = sqlite3.connect(output_database, timeout=60)
    target.row_factory = sqlite3.Row
    try:
        target.execute("PRAGMA journal_mode = WAL")
        target.execute("PRAGMA synchronous = NORMAL")
        target.execute("PRAGMA temp_store = MEMORY")
        target.executescript(EXPERIMENT_SCHEMA)

        quality_reviews, quality_summary = load_quality_reviews(quality_audit_database)
        if quality_audit_database is not None:
            print(
                "[classifier-dataset/quality] "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in quality_summary["decision_counts"].items()
                )
            )

        manual_split = choose_manual_capture_split(
            source,
            seed=seed,
            train_fraction=manual_train_fraction,
        )
        selected = []
        selected.extend(
            select_jp_rows(
                source,
                source_partition="train",
                split="train",
                samples_per_class=jp_train_per_class,
                seed=seed,
                quality_reviews=quality_reviews,
            )
        )
        selected.extend(
            select_jp_rows(
                source,
                source_partition="valid",
                split="jp_val",
                samples_per_class=jp_valid_per_class,
                seed=seed + 10_000,
                quality_reviews=quality_reviews,
            )
        )
        selected.extend(
            select_manual_rows(
                source,
                manual_split,
                quality_reviews=quality_reviews,
            )
        )

        print(
            f"[classifier-dataset] selected {len(selected)} source crops; "
            f"preprocessing to gray {image_size}x{image_size} with {workers} workers"
        )
        insert_prepared_rows(
            target,
            selected,
            image_size=image_size,
            workers=workers,
        )

        source_stat = source_database.stat()
        metadata = {
            "schema_version": "2",
            "source_database": str(source_database),
            "source_database_size": str(source_stat.st_size),
            "source_database_mtime_ns": str(source_stat.st_mtime_ns),
            "seed": str(seed),
            "image_size": str(image_size),
            "jp_train_per_class": str(jp_train_per_class),
            "jp_valid_per_class": str(jp_valid_per_class),
            "manual_train_fraction": repr(manual_train_fraction),
            "base_labels": json.dumps(BASE_LABELS, ensure_ascii=False),
            "preprocess": "grayscale_aspect_preserving_letterbox_border_median_lanczos_u8",
            "red_five_policy": "red5m->5m,red5p->5p,red5s->5s",
            "quality_audit_database": "" if quality_audit_database is None else str(quality_audit_database),
            "quality_audit_policy": (
                "none"
                if quality_audit_database is None
                else "label_error=corrected_label;false_detection=keep;unusable_crop/background=exclude"
            ),
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
            "base_labels": list(BASE_LABELS),
            "jp_train_per_class": jp_train_per_class,
            "jp_valid_per_class": jp_valid_per_class,
            "manual_train_fraction": manual_train_fraction,
            "quality_audit": quality_summary,
            "selected_quality_audit_counts": dict(
                sorted(
                    {
                        decision: sum(
                            1 for row in selected if row.quality_decision == decision
                        )
                        for decision in (
                            "label_error",
                            "false_detection",
                            "unusable_crop",
                            "background",
                        )
                    }.items()
                )
            ),
            "manual_capture_split": {
                "train": sorted(
                    capture_id for capture_id, split in manual_split.items() if split == "train"
                ),
                "manual_val": sorted(
                    capture_id
                    for capture_id, split in manual_split.items()
                    if split == "manual_val"
                ),
            },
            **counts,
        }
    finally:
        target.close()
        source.close()


def base_label(tile_label: str) -> str:
    normalized = RED_FIVE_TO_BASE.get(tile_label, tile_label)
    if normalized not in BASE_LABEL_TO_INDEX:
        raise ValueError(f"Unsupported classifier tile label: {tile_label}")
    return normalized


def load_quality_reviews(
    database: Path | None,
) -> tuple[dict[str, QualityReview], dict[str, Any]]:
    decisions = ("label_error", "false_detection", "unusable_crop", "background")
    counts = {decision: 0 for decision in decisions}
    if database is None:
        return {}, {
            "database": None,
            "review_count": 0,
            "decision_counts": counts,
        }

    reviews: dict[str, QualityReview] = {}
    with sqlite3.connect(sqlite_readonly_uri(database), uri=True, timeout=60) as connection:
        connection.row_factory = sqlite3.Row
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'review'"
        ).fetchone()
        if table is None:
            raise ValueError(f"Quality audit database has no review table: {database}")
        rows = connection.execute(
            "SELECT crop_id, decision, corrected_label FROM review ORDER BY crop_id"
        ).fetchall()

    for row in rows:
        crop_id = str(row["crop_id"])
        decision = str(row["decision"])
        if decision not in counts:
            raise ValueError(f"Unsupported quality audit decision for {crop_id}: {decision}")
        corrected_label = (
            None if row["corrected_label"] is None else str(row["corrected_label"])
        )
        if decision == "label_error":
            if corrected_label is None:
                raise ValueError(f"label_error review has no corrected_label: {crop_id}")
            base_label(corrected_label)
        else:
            corrected_label = None
        reviews[crop_id] = QualityReview(
            decision=decision,
            corrected_label=corrected_label,
        )
        counts[decision] += 1

    return reviews, {
        "database": str(database),
        "review_count": len(reviews),
        "decision_counts": counts,
    }


def effective_quality_label(
    crop_id: str,
    source_label: str,
    quality_reviews: dict[str, QualityReview],
) -> tuple[str | None, str | None]:
    review = quality_reviews.get(crop_id)
    if review is None:
        return source_label, None
    if review.decision == "label_error":
        assert review.corrected_label is not None
        return review.corrected_label, review.decision
    if review.decision == "false_detection":
        return source_label, review.decision
    if review.decision in {"unusable_crop", "background"}:
        return None, review.decision
    raise ValueError(f"Unsupported quality audit decision: {review.decision}")


def choose_manual_capture_split(
    connection: sqlite3.Connection,
    *,
    seed: int,
    train_fraction: float,
) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT
            capture_id,
            COALESCE(brightness, '') AS brightness,
            COALESCE(shadow, '') AS shadow,
            COUNT(*) AS crop_count
        FROM tile_crop
        WHERE source = 'manual' AND capture_id IS NOT NULL
        GROUP BY capture_id, brightness, shadow
        ORDER BY capture_id
        """
    ).fetchall()
    if not rows:
        raise ValueError("Persistent crop database contains no manual captures")

    strata: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        strata[(str(row["brightness"]), str(row["shadow"]))].append(
            str(row["capture_id"])
        )

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

    if not any(split == "manual_val" for split in result.values()):
        # Extremely small synthetic/manual datasets can have one capture in each stratum.
        # Keep capture-level isolation and choose one deterministic holdout globally.
        candidates = sorted(result)
        if len(candidates) > 1:
            holdout = candidates[stable_seed(seed, "manual-global-holdout") % len(candidates)]
            result[holdout] = "manual_val"
    return result


def select_jp_rows(
    connection: sqlite3.Connection,
    *,
    source_partition: str,
    split: str,
    samples_per_class: int,
    seed: int,
    quality_reviews: dict[str, QualityReview],
) -> list[SelectedSourceRow]:
    reservoirs: dict[str, list[tuple[int, str, str | None]]] = {
        label: [] for label in BASE_LABELS
    }
    forced_quality: dict[str, list[tuple[int, str, str | None]]] = {
        label: [] for label in BASE_LABELS
    }
    seen = {label: 0 for label in BASE_LABELS}
    rngs = {
        label: random.Random(stable_seed(seed, f"jp:{source_partition}:{label}"))
        for label in BASE_LABELS
    }

    cursor = connection.execute(
        """
        SELECT rowid, crop_id, tile_label
        FROM tile_crop
        WHERE source = 'jp' AND source_partition = ?
        ORDER BY rowid
        """,
        (source_partition,),
    )
    for row in cursor:
        crop_id = str(row["crop_id"])
        source_label = str(row["tile_label"])
        effective_label, quality_decision = effective_quality_label(
            crop_id,
            source_label,
            quality_reviews,
        )
        if effective_label is None:
            continue
        try:
            label = base_label(effective_label)
        except ValueError:
            continue

        candidate = (int(row["rowid"]), effective_label, quality_decision)
        if split == "train" and quality_decision in {"label_error", "false_detection"}:
            # Human-reviewed classifier disagreements are valuable hard examples. Keep
            # them in addition to the normal JP-per-class reservoir instead of hoping
            # a 500/class random sample happens to select them again.
            forced_quality[label].append(candidate)
            continue

        class_seen = seen[label]
        reservoir = reservoirs[label]
        if class_seen < samples_per_class:
            reservoir.append(candidate)
        else:
            replacement = rngs[label].randrange(class_seen + 1)
            if replacement < samples_per_class:
                reservoir[replacement] = candidate
        seen[label] = class_seen + 1

    selected_entries: list[tuple[int, str, str | None]] = []
    for label in BASE_LABELS:
        reservoir = reservoirs[label]
        forced = forced_quality[label]
        available = seen[label]
        if available == 0 and not forced:
            raise ValueError(f"No JP {source_partition} crops found for base class {label}")
        if len(reservoir) < samples_per_class:
            print(
                f"[classifier-dataset/jp/{source_partition}] {label}: "
                f"requested {samples_per_class}, available {len(reservoir)}"
            )
        selected_entries.extend(reservoir)
        selected_entries.extend(forced)
        print(
            f"[classifier-dataset/jp/{source_partition}] {label}: "
            f"selected {len(reservoir)} of {available}"
            + (f" + forced_quality={len(forced)}" if forced else "")
        )

    by_rowid = fetch_rows_by_rowid(
        connection,
        [rowid for rowid, _, _ in selected_entries],
    )
    return [
        SelectedSourceRow(
            split=split,
            values=tuple(by_rowid[rowid][column] for column in SOURCE_COLUMNS),
            effective_label=effective_label,
            quality_decision=quality_decision,
        )
        for rowid, effective_label, quality_decision in selected_entries
    ]


def select_manual_rows(
    connection: sqlite3.Connection,
    capture_split: dict[str, str],
    *,
    quality_reviews: dict[str, QualityReview],
) -> list[SelectedSourceRow]:
    rows = connection.execute(
        f"""
        SELECT {', '.join(SOURCE_COLUMNS)}
        FROM tile_crop
        WHERE source = 'manual'
        ORDER BY crop_id
        """
    ).fetchall()
    selected: list[SelectedSourceRow] = []
    excluded = 0
    for row in rows:
        crop_id = str(row["crop_id"])
        source_label = str(row["tile_label"])
        effective_label, quality_decision = effective_quality_label(
            crop_id,
            source_label,
            quality_reviews,
        )
        if effective_label is None:
            excluded += 1
            continue
        # Fail fast for audited manual corrections that cannot feed the 34-class model.
        base_label(effective_label)

        capture_id = row["capture_id"]
        if capture_id is None:
            raise ValueError(f"Manual crop {crop_id} has no capture_id")
        split = capture_split.get(str(capture_id))
        if split is None:
            raise ValueError(f"No capture split assigned to {capture_id}")
        selected.append(
            SelectedSourceRow(
                split=split,
                values=tuple(row[column] for column in SOURCE_COLUMNS),
                effective_label=effective_label,
                quality_decision=quality_decision,
            )
        )
    print(
        "[classifier-dataset/manual] "
        + ", ".join(
            f"{split}={sum(1 for row in selected if row.split == split)}"
            for split in ("train", "manual_val")
        )
        + f", excluded_by_quality={excluded}"
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
            FROM tile_crop
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
    rows: Sequence[SelectedSourceRow],
    *,
    image_size: int,
    workers: int,
    commit_interval: int = 2000,
) -> None:
    insert_sql = """
        INSERT INTO sample(
            sample_id, split, source, source_partition, base_label, class_index,
            original_label, source_label, quality_audit_decision,
            crop_id, image_size, image_gray_u8,
            original_width, original_height, source_image_path, source_image_id,
            source_annotation_id, capture_id, layout_id, region, brightness, shadow,
            annotation_angle_deg, expected_rotation_deg
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """

    def prepare(row: SelectedSourceRow) -> PreparedSample:
        values = dict(zip(SOURCE_COLUMNS, row.values))
        source_label = str(values["tile_label"])
        original_label = row.effective_label
        merged_label = base_label(original_label)
        gray = preprocess_gray_u8(bytes(values["image_png"]), image_size=image_size)
        sample_id = f"{row.split}:{values['crop_id']}"
        return PreparedSample(
            values=(
                sample_id,
                row.split,
                str(values["source"]),
                str(values["source_partition"]),
                merged_label,
                BASE_LABEL_TO_INDEX[merged_label],
                original_label,
                source_label,
                row.quality_decision,
                str(values["crop_id"]),
                image_size,
                gray,
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

    prepared_batch: list[tuple[Any, ...]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for prepared in executor.map(prepare, rows, chunksize=32):
            prepared_batch.append(prepared.values)
            completed += 1
            if len(prepared_batch) >= commit_interval:
                connection.executemany(insert_sql, prepared_batch)
                connection.commit()
                prepared_batch.clear()
                print(f"[classifier-dataset] prepared {completed}/{len(rows)}")
    if prepared_batch:
        connection.executemany(insert_sql, prepared_batch)
        connection.commit()
    print(f"[classifier-dataset] prepared {completed}/{len(rows)}")


def preprocess_gray_u8(image_png: bytes, *, image_size: int = DEFAULT_IMAGE_SIZE) -> bytes:
    with Image.open(io.BytesIO(image_png)) as source:
        source.load()
        image = source.convert("L")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid crop size: {image.size}")

    scale = min(image_size / width, image_size / height)
    resized_width = max(1, min(image_size, int(math.floor(width * scale + 0.5))))
    resized_height = max(1, min(image_size, int(math.floor(height * scale + 0.5))))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.size == 0:
        fill = 127
    elif width == 1 or height == 1:
        fill = int(np.median(pixels))
    else:
        border = np.concatenate(
            [pixels[0, :], pixels[-1, :], pixels[1:-1, 0], pixels[1:-1, -1]]
        )
        fill = int(np.median(border))

    canvas = Image.new("L", (image_size, image_size), color=fill)
    offset_x = (image_size - resized_width) // 2
    offset_y = (image_size - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    raw = canvas.tobytes()
    expected = image_size * image_size
    if len(raw) != expected:
        raise RuntimeError(f"Preprocessed image has {len(raw)} bytes, expected {expected}")
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
        FROM sample
        GROUP BY split, source
        ORDER BY split, source
        """
    ):
        by_split_source[str(row["split"])][str(row["source"])] = int(row["count"])
    by_split_label: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for row in connection.execute(
        """
        SELECT split, base_label, COUNT(*) AS count
        FROM sample
        GROUP BY split, base_label
        ORDER BY split, class_index
        """
    ):
        by_split_label[str(row["split"])][str(row["base_label"])] = int(row["count"])
    manual_conditions: defaultdict[str, int] = defaultdict(int)
    for row in connection.execute(
        """
        SELECT split, COALESCE(brightness, '') AS brightness,
               COALESCE(shadow, '') AS shadow, COUNT(*) AS count
        FROM sample
        WHERE source = 'manual'
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
        "counts_by_split_and_label": dict(by_split_label),
        "manual_condition_counts": dict(manual_conditions),
    }


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def sqlite_readonly_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def temporary_database_path(output_database: Path) -> Path:
    descriptor, path = tempfile.mkstemp(
        prefix=f".{output_database.name}.",
        suffix=".tmp.sqlite",
        dir=output_database.parent,
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
