from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


RED_FIVE_LABELS = ("red5m", "red5p", "red5s")
NORMAL_FIVE_LABELS = ("5m", "5p", "5s")
SOURCE_LABELS = (
    "5m",
    "red5m",
    "5p",
    "red5p",
    "5s",
    "red5s",
)

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE dataset_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE sample (
    sample_id                TEXT PRIMARY KEY,
    crop_id                  TEXT NOT NULL UNIQUE,
    source                   TEXT NOT NULL CHECK (source IN ('jp', 'manual')),
    source_partition         TEXT NOT NULL,
    suit                     TEXT NOT NULL CHECK (suit IN ('m', 'p', 's')),
    is_red                   INTEGER NOT NULL CHECK (is_red IN (0, 1)),
    source_label             TEXT NOT NULL CHECK (
        source_label IN ('5m', 'red5m', '5p', 'red5p', '5s', 'red5s')
    ),
    raw_category_name        TEXT NOT NULL,
    raw_category_id          INTEGER,
    image_format             TEXT NOT NULL,
    image_width              INTEGER NOT NULL,
    image_height             INTEGER NOT NULL,
    image_png                BLOB NOT NULL,
    source_image_path        TEXT NOT NULL,
    source_image_id          TEXT,
    source_annotation_id     TEXT NOT NULL,
    bbox_json                TEXT NOT NULL,
    capture_id               TEXT,
    layout_id                TEXT,
    layout_ordinal           INTEGER,
    region                   TEXT,
    group_name               TEXT,
    group_ordinal            INTEGER,
    tile_ordinal             INTEGER,
    brightness               TEXT,
    shadow                   TEXT,
    annotation_angle_deg     REAL NOT NULL,
    expected_rotation_deg    INTEGER NOT NULL
);

CREATE INDEX idx_sample_red_suit
ON sample(is_red, suit);

CREATE INDEX idx_sample_source_partition
ON sample(source, source_partition);

CREATE INDEX idx_sample_source_label
ON sample(source, source_label);

CREATE INDEX idx_sample_capture
ON sample(capture_id);
"""

SOURCE_COLUMNS = (
    "crop_id",
    "source",
    "source_partition",
    "tile_label",
    "raw_category_name",
    "raw_category_id",
    "image_format",
    "image_width",
    "image_height",
    "image_png",
    "source_image_path",
    "source_image_id",
    "source_annotation_id",
    "bbox_json",
    "capture_id",
    "layout_id",
    "layout_ordinal",
    "region",
    "group_name",
    "group_ordinal",
    "tile_ordinal",
    "brightness",
    "shadow",
    "annotation_angle_deg",
    "expected_rotation_deg",
)

INSERT_SQL = """
INSERT INTO sample(
    sample_id,
    crop_id,
    source,
    source_partition,
    suit,
    is_red,
    source_label,
    raw_category_name,
    raw_category_id,
    image_format,
    image_width,
    image_height,
    image_png,
    source_image_path,
    source_image_id,
    source_annotation_id,
    bbox_json,
    capture_id,
    layout_id,
    layout_ordinal,
    region,
    group_name,
    group_ordinal,
    tile_ordinal,
    brightness,
    shadow,
    annotation_angle_deg,
    expected_rotation_deg
) VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
"""


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact RGB red-five dataset from the persistent tile crop DB. "
            "Only 5m/red5m, 5p/red5p, and 5s/red5s are copied; source crop metadata "
            "and lossless PNG bytes are preserved."
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
            "red_five_datasets/red_five_all.sqlite."
        ),
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("jp", "manual"),
        default=("jp", "manual"),
        help="Source families to include. Defaults to both jp and manual.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        / "red_five_datasets"
        / "red_five_all.sqlite"
    )

    summary = build_red_five_dataset(
        source_database=source_database,
        output_database=output_database,
        sources=tuple(args.sources),
        force=bool(args.force),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_red_five_dataset(
    *,
    source_database: Path,
    output_database: Path,
    sources: tuple[str, ...] = ("jp", "manual"),
    force: bool = False,
) -> dict[str, Any]:
    source_database = source_database.resolve()
    output_database = output_database.resolve()
    if not source_database.is_file():
        raise FileNotFoundError(source_database)
    if source_database == output_database:
        raise ValueError("Source and output database paths must differ")

    normalized_sources = tuple(dict.fromkeys(sources))
    if not normalized_sources:
        raise ValueError("At least one source must be selected")
    unsupported = set(normalized_sources) - {"jp", "manual"}
    if unsupported:
        raise ValueError(f"Unsupported sources: {sorted(unsupported)}")

    if output_database.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_database}. Use --force to replace it."
        )

    output_database.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_database_path(output_database)
    try:
        summary = _create_red_five_dataset(
            source_database=source_database,
            output_database=temporary_path,
            sources=normalized_sources,
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


def _create_red_five_dataset(
    *,
    source_database: Path,
    output_database: Path,
    sources: tuple[str, ...],
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

        placeholders = ",".join("?" for _ in sources)
        label_placeholders = ",".join("?" for _ in SOURCE_LABELS)
        cursor = source.execute(
            f"""
            SELECT {', '.join(SOURCE_COLUMNS)}
            FROM tile_crop
            WHERE source IN ({placeholders})
              AND tile_label IN ({label_placeholders})
            ORDER BY rowid
            """,
            (*sources, *SOURCE_LABELS),
        )

        copied = 0
        batch: list[tuple[Any, ...]] = []
        for row in cursor:
            batch.append(output_row(row))
            if len(batch) >= 5000:
                target.executemany(INSERT_SQL, batch)
                target.commit()
                copied += len(batch)
                batch.clear()
                print(f"[red-five-dataset] copied {copied} crops")
        if batch:
            target.executemany(INSERT_SQL, batch)
            target.commit()
            copied += len(batch)
        print(f"[red-five-dataset] copied {copied} crops")

        source_stat = source_database.stat()
        metadata = {
            "schema_version": "1",
            "source_database": str(source_database),
            "source_database_size": str(source_stat.st_size),
            "source_database_mtime_ns": str(source_stat.st_mtime_ns),
            "included_sources": json.dumps(sources),
            "included_source_labels": json.dumps(SOURCE_LABELS),
            "target": "red_five_binary_classification",
            "image_storage": "lossless_source_rgb_png",
        }
        target.executemany(
            "INSERT INTO dataset_metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        target.commit()
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        counts = dataset_counts(target)
        return {
            "status": "completed",
            "database": str(output_database),
            "database_bytes": output_database.stat().st_size,
            "sample_count": copied,
            "included_sources": list(sources),
            "included_source_labels": list(SOURCE_LABELS),
            **counts,
        }
    finally:
        target.close()
        source.close()


def validate_source_schema(connection: sqlite3.Connection) -> None:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'tile_crop'"
    ).fetchone()
    if table is None:
        raise ValueError("Source database has no tile_crop table")

    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(tile_crop)").fetchall()
    }
    missing = set(SOURCE_COLUMNS) - columns
    if missing:
        raise ValueError(f"Source tile_crop table is missing columns: {sorted(missing)}")


def output_row(row: sqlite3.Row) -> tuple[Any, ...]:
    source_label = str(row["tile_label"])
    if source_label not in SOURCE_LABELS:
        raise ValueError(f"Unexpected source label: {source_label}")
    suit = source_label[-1]
    is_red = 1 if source_label.startswith("red5") else 0
    crop_id = str(row["crop_id"])
    return (
        crop_id,
        crop_id,
        str(row["source"]),
        str(row["source_partition"]),
        suit,
        is_red,
        source_label,
        str(row["raw_category_name"]),
        row["raw_category_id"],
        str(row["image_format"]),
        int(row["image_width"]),
        int(row["image_height"]),
        bytes(row["image_png"]),
        str(row["source_image_path"]),
        None if row["source_image_id"] is None else str(row["source_image_id"]),
        str(row["source_annotation_id"]),
        str(row["bbox_json"]),
        None if row["capture_id"] is None else str(row["capture_id"]),
        None if row["layout_id"] is None else str(row["layout_id"]),
        None if row["layout_ordinal"] is None else int(row["layout_ordinal"]),
        None if row["region"] is None else str(row["region"]),
        None if row["group_name"] is None else str(row["group_name"]),
        None if row["group_ordinal"] is None else int(row["group_ordinal"]),
        None if row["tile_ordinal"] is None else int(row["tile_ordinal"]),
        None if row["brightness"] is None else str(row["brightness"]),
        None if row["shadow"] is None else str(row["shadow"]),
        float(row["annotation_angle_deg"]),
        int(row["expected_rotation_deg"]),
    )


def dataset_counts(connection: sqlite3.Connection) -> dict[str, Any]:
    by_label = {
        str(row["source_label"]): int(row["count"])
        for row in connection.execute(
            """
            SELECT source_label, COUNT(*) AS count
            FROM sample
            GROUP BY source_label
            ORDER BY source_label
            """
        )
    }
    by_red = {
        "red" if int(row["is_red"]) else "normal": int(row["count"])
        for row in connection.execute(
            """
            SELECT is_red, COUNT(*) AS count
            FROM sample
            GROUP BY is_red
            ORDER BY is_red
            """
        )
    }
    by_suit: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for row in connection.execute(
        """
        SELECT suit, is_red, COUNT(*) AS count
        FROM sample
        GROUP BY suit, is_red
        ORDER BY suit, is_red
        """
    ):
        by_suit[str(row["suit"])]["red" if int(row["is_red"]) else "normal"] = int(
            row["count"]
        )

    by_source_partition: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for row in connection.execute(
        """
        SELECT source, source_partition, source_label, COUNT(*) AS count
        FROM sample
        GROUP BY source, source_partition, source_label
        ORDER BY source, source_partition, source_label
        """
    ):
        key = f"{row['source']}/{row['source_partition']}"
        by_source_partition[key][str(row["source_label"])] = int(row["count"])

    return {
        "counts_by_source_label": by_label,
        "counts_by_red_state": by_red,
        "counts_by_suit_and_red_state": dict(by_suit),
        "counts_by_source_partition_and_label": dict(by_source_partition),
    }


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
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
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
