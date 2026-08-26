from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import tempfile
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SEED = 42
DEFAULT_JP_SAMPLES_PER_LABEL = 100

SAMPLE_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE sample_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE tile_crop (
    crop_id              TEXT PRIMARY KEY,
    source               TEXT NOT NULL CHECK (source IN ('jp', 'manual')),
    source_partition     TEXT NOT NULL,
    tile_label           TEXT NOT NULL,
    image_width          INTEGER NOT NULL,
    image_height         INTEGER NOT NULL,
    image_png            BLOB NOT NULL,
    source_image_path    TEXT NOT NULL,
    source_image_id      TEXT,
    capture_id           TEXT,
    layout_id            TEXT,
    region               TEXT,
    brightness           TEXT,
    shadow               TEXT
);

CREATE INDEX idx_sample_source_label
ON tile_crop(source, tile_label);

CREATE INDEX idx_sample_label_source
ON tile_crop(tile_label, source);
"""

SAMPLE_COLUMNS = (
    "crop_id",
    "source",
    "source_partition",
    "tile_label",
    "image_width",
    "image_height",
    "image_png",
    "source_image_path",
    "source_image_id",
    "capture_id",
    "layout_id",
    "region",
    "brightness",
    "shadow",
)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Create a compact deterministic color-trial dataset from the persistent "
            "tile-crop database. JP train is reservoir-sampled per label without "
            "ORDER BY RANDOM(); all manual crops are copied."
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
            "color_trials/sample_seed<seed>.sqlite."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--jp-samples-per-label",
        type=int,
        default=DEFAULT_JP_SAMPLES_PER_LABEL,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the compact sample even when its fingerprint is unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jp_samples_per_label < 1:
        raise ValueError("--jp-samples-per-label must be positive")

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
        / "color_trials"
        / f"sample_seed{args.seed}.sqlite"
    )

    result = build_sample_database(
        source_database=source_database,
        output_database=output_database,
        seed=int(args.seed),
        jp_samples_per_label=int(args.jp_samples_per_label),
        force=bool(args.force),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_sample_database(
    *,
    source_database: Path,
    output_database: Path,
    seed: int,
    jp_samples_per_label: int,
    force: bool = False,
) -> dict[str, Any]:
    source_database = source_database.resolve()
    output_database = output_database.resolve()
    if not source_database.is_file():
        raise FileNotFoundError(source_database)
    if source_database == output_database:
        raise ValueError("Source and output databases must be different files")
    if jp_samples_per_label < 1:
        raise ValueError("jp_samples_per_label must be positive")

    fingerprint = sample_fingerprint(
        source_database,
        seed=seed,
        jp_samples_per_label=jp_samples_per_label,
    )
    if not force and sample_is_current(output_database, fingerprint):
        summary = read_sample_summary(output_database)
        summary["action"] = "reused"
        return summary

    output_database.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_database_path(output_database)
    try:
        summary = create_sample_database(
            source_database=source_database,
            output_database=temporary_path,
            seed=seed,
            jp_samples_per_label=jp_samples_per_label,
            fingerprint=fingerprint,
        )
        remove_sqlite_sidecars(output_database)
        os.replace(temporary_path, output_database)
    except Exception:
        remove_sqlite_files(temporary_path)
        raise

    summary["database"] = str(output_database)
    summary["database_bytes"] = output_database.stat().st_size
    summary_path = output_database.with_suffix(".summary.json")
    atomic_write_json(summary_path, summary)
    return summary


def create_sample_database(
    *,
    source_database: Path,
    output_database: Path,
    seed: int,
    jp_samples_per_label: int,
    fingerprint: str,
) -> dict[str, Any]:
    remove_sqlite_files(output_database)
    source = sqlite3.connect(sqlite_readonly_uri(source_database), uri=True, timeout=60)
    source.row_factory = sqlite3.Row
    target = sqlite3.connect(output_database, timeout=60)
    target.row_factory = sqlite3.Row
    try:
        target.execute("PRAGMA journal_mode = WAL")
        target.execute("PRAGMA synchronous = NORMAL")
        target.execute("PRAGMA temp_store = MEMORY")
        target.executescript(SAMPLE_SCHEMA)

        selected_jp_rowids, available_counts = reservoir_sample_jp_train_rowids(
            source,
            samples_per_label=jp_samples_per_label,
            seed=seed,
        )
        selected_jp_count = copy_source_rows_by_rowid(
            source,
            target,
            rowids=selected_jp_rowids,
        )
        manual_count = copy_manual_rows(source, target)

        metadata = {
            "schema_version": "1",
            "source_database": str(source_database),
            "source_fingerprint": fingerprint,
            "seed": str(seed),
            "jp_samples_per_label": str(jp_samples_per_label),
        }
        target.executemany(
            "INSERT INTO sample_metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        target.commit()
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        counts = sample_counts(target)
        return {
            "status": "completed",
            "action": "built",
            "database": str(output_database),
            "database_bytes": output_database.stat().st_size,
            "seed": seed,
            "jp_samples_per_label": jp_samples_per_label,
            "crop_count": selected_jp_count + manual_count,
            "counts_by_source": counts["counts_by_source"],
            "counts_by_source_and_label": counts["counts_by_source_and_label"],
            "jp_available_train_counts": dict(sorted(available_counts.items())),
        }
    finally:
        target.close()
        source.close()


def reservoir_sample_jp_train_rowids(
    connection: sqlite3.Connection,
    *,
    samples_per_label: int,
    seed: int,
) -> tuple[list[int], dict[str, int]]:
    labels = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT DISTINCT tile_label
            FROM tile_crop INDEXED BY idx_tile_crop_source_label
            WHERE source = 'jp'
            ORDER BY tile_label
            """
        )
    ]
    if not labels:
        raise ValueError("Source database has no JP crops")

    train_rowid_range = contiguous_partition_rowid_range(
        connection,
        source="jp",
        partition="train",
    )
    if train_rowid_range is None:
        print(
            "[sample/jp] train rowids are not contiguous; using indexed label scans "
            "with partition verification"
        )
    else:
        print(
            f"[sample/jp] train rowids {train_rowid_range[0]}.."
            f"{train_rowid_range[1]} are contiguous; using index-only label scans"
        )

    selected: list[int] = []
    available_counts: dict[str, int] = {}
    for label in labels:
        label_seed = deterministic_label_seed(seed, label)
        random_source = random.Random(label_seed)
        reservoir: list[int] = []
        seen = 0
        cursor = jp_train_rowid_cursor(
            connection,
            label=label,
            contiguous_rowid_range=train_rowid_range,
        )
        for row in cursor:
            rowid = int(row[0])
            if seen < samples_per_label:
                reservoir.append(rowid)
            else:
                replacement = random_source.randrange(seen + 1)
                if replacement < samples_per_label:
                    reservoir[replacement] = rowid
            seen += 1

        if seen == 0:
            continue
        available_counts[label] = seen
        selected.extend(reservoir)
        print(f"[sample/jp] {label}: selected {len(reservoir)} of {seen}")

    return selected, available_counts


def contiguous_partition_rowid_range(
    connection: sqlite3.Connection,
    *,
    source: str,
    partition: str,
) -> tuple[int, int] | None:
    first = connection.execute(
        """
        SELECT rowid
        FROM tile_crop INDEXED BY idx_tile_crop_source_partition
        WHERE source = ? AND source_partition = ?
        ORDER BY rowid
        LIMIT 1
        """,
        (source, partition),
    ).fetchone()
    last = connection.execute(
        """
        SELECT rowid
        FROM tile_crop INDEXED BY idx_tile_crop_source_partition
        WHERE source = ? AND source_partition = ?
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (source, partition),
    ).fetchone()
    if first is None or last is None:
        return None

    minimum = int(first[0])
    maximum = int(last[0])
    count = source_partition_metadata_count(connection, source, partition)
    if count is None:
        count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM tile_crop INDEXED BY idx_tile_crop_source_partition
                WHERE source = ? AND source_partition = ?
                """,
                (source, partition),
            ).fetchone()[0]
        )
    if maximum - minimum + 1 != count:
        return None
    return minimum, maximum


def source_partition_metadata_count(
    connection: sqlite3.Connection,
    source: str,
    partition: str,
) -> int | None:
    key = f"source.{source}.{partition}.crop_count"
    try:
        row = connection.execute(
            "SELECT value FROM dataset_metadata WHERE key = ?",
            (key,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def jp_train_rowid_cursor(
    connection: sqlite3.Connection,
    *,
    label: str,
    contiguous_rowid_range: tuple[int, int] | None,
) -> sqlite3.Cursor:
    if contiguous_rowid_range is not None:
        minimum, maximum = contiguous_rowid_range
        return connection.execute(
            """
            SELECT rowid
            FROM tile_crop INDEXED BY idx_tile_crop_source_label
            WHERE source = 'jp'
              AND tile_label = ?
              AND rowid BETWEEN ? AND ?
            """,
            (label, minimum, maximum),
        )
    return connection.execute(
        """
        SELECT rowid
        FROM tile_crop INDEXED BY idx_tile_crop_source_label
        WHERE source = 'jp'
          AND tile_label = ?
          AND source_partition = 'train'
        """,
        (label,),
    )


def deterministic_label_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def copy_source_rows_by_rowid(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    *,
    rowids: Iterable[int],
    chunk_size: int = 500,
) -> int:
    rowid_list = sorted(set(int(rowid) for rowid in rowids))
    copied = 0
    insert_sql = sample_insert_sql()
    for start in range(0, len(rowid_list), chunk_size):
        chunk = rowid_list[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = source.execute(
            f"""
            SELECT {', '.join(SAMPLE_COLUMNS)}
            FROM tile_crop
            WHERE rowid IN ({placeholders})
            """,
            chunk,
        ).fetchall()
        target.executemany(insert_sql, (tuple(row) for row in rows))
        copied += len(rows)
    target.commit()
    return copied


def copy_manual_rows(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
) -> int:
    rows = source.execute(
        f"""
        SELECT {', '.join(SAMPLE_COLUMNS)}
        FROM tile_crop
        WHERE source = 'manual'
        ORDER BY crop_id
        """
    )
    copied = 0
    batch: list[tuple[Any, ...]] = []
    insert_sql = sample_insert_sql()
    for row in rows:
        batch.append(tuple(row))
        if len(batch) >= 500:
            target.executemany(insert_sql, batch)
            copied += len(batch)
            batch.clear()
    if batch:
        target.executemany(insert_sql, batch)
        copied += len(batch)
    target.commit()
    print(f"[sample/manual] copied {copied} crops")
    return copied


def sample_insert_sql() -> str:
    placeholders = ", ".join("?" for _ in SAMPLE_COLUMNS)
    return (
        f"INSERT INTO tile_crop({', '.join(SAMPLE_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )


def sample_counts(connection: sqlite3.Connection) -> dict[str, Any]:
    by_source = {
        str(row["source"]): int(row["crop_count"])
        for row in connection.execute(
            """
            SELECT source, COUNT(*) AS crop_count
            FROM tile_crop
            GROUP BY source
            ORDER BY source
            """
        )
    }
    by_source_and_label: dict[str, dict[str, int]] = defaultdict(dict)
    for row in connection.execute(
        """
        SELECT source, tile_label, COUNT(*) AS crop_count
        FROM tile_crop
        GROUP BY source, tile_label
        ORDER BY source, tile_label
        """
    ):
        by_source_and_label[str(row["source"])][str(row["tile_label"])] = int(
            row["crop_count"]
        )
    return {
        "counts_by_source": by_source,
        "counts_by_source_and_label": dict(by_source_and_label),
    }


def sample_fingerprint(
    source_database: Path,
    *,
    seed: int,
    jp_samples_per_label: int,
) -> str:
    stat = source_database.stat()
    digest = hashlib.sha256()
    for value in (
        str(source_database.resolve()),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        str(seed),
        str(jp_samples_per_label),
        "sample-schema-v1",
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def sample_is_current(output_database: Path, fingerprint: str) -> bool:
    if not output_database.is_file():
        return False
    try:
        with closing(
            sqlite3.connect(sqlite_readonly_uri(output_database), uri=True)
        ) as connection:
            row = connection.execute(
                "SELECT value FROM sample_metadata WHERE key = 'source_fingerprint'"
            ).fetchone()
            status = connection.execute("PRAGMA quick_check").fetchone()
    except sqlite3.DatabaseError:
        return False
    return (
        row is not None
        and str(row[0]) == fingerprint
        and status is not None
        and str(status[0]) == "ok"
    )


def read_sample_summary(output_database: Path) -> dict[str, Any]:
    with closing(
        sqlite3.connect(sqlite_readonly_uri(output_database), uri=True)
    ) as connection:
        connection.row_factory = sqlite3.Row
        counts = sample_counts(connection)
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM sample_metadata")
        }
        crop_count = int(connection.execute("SELECT COUNT(*) FROM tile_crop").fetchone()[0])
    return {
        "status": "completed",
        "database": str(output_database),
        "database_bytes": output_database.stat().st_size,
        "seed": int(metadata["seed"]),
        "jp_samples_per_label": int(metadata["jp_samples_per_label"]),
        "crop_count": crop_count,
        "counts_by_source": counts["counts_by_source"],
        "counts_by_source_and_label": counts["counts_by_source_and_label"],
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
    for candidate in (Path(f"{path}-wal"), Path(f"{path}-shm")):
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
