from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any

try:
    from .build_tile_classifier_dataset import BASE_LABELS, preprocess_gray_u8, sqlite_readonly_uri
    from .detector_duplicate_groups import DetectorCandidate, build_duplicate_plan
except ImportError:  # direct script execution
    from build_tile_classifier_dataset import BASE_LABELS, preprocess_gray_u8, sqlite_readonly_uri
    from detector_duplicate_groups import DetectorCandidate, build_duplicate_plan


INVALID_LABEL = "invalid"
# Legacy review reason from before detector-side duplicate suppression was separated.
# These bboxes are removed before classifier inference in production and therefore
# must never become gray35 invalid-class training examples.
EXCLUDED_INVALID_REASONS = frozenset({"duplicate_or_overlap"})
CLASS_LABELS = tuple(BASE_LABELS) + (INVALID_LABEL,)
LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_LABELS)}
SCHEMA_VERSION = "1"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE experiment_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE sample (
    sample_id               TEXT PRIMARY KEY,
    split                   TEXT NOT NULL CHECK (split IN ('train', 'manual_val', 'jp_val')),
    source                  TEXT NOT NULL,
    source_partition        TEXT NOT NULL,
    base_label              TEXT NOT NULL,
    class_index             INTEGER NOT NULL,
    original_label          TEXT NOT NULL,
    source_label            TEXT NOT NULL,
    quality_audit_decision  TEXT,
    crop_id                 TEXT NOT NULL,
    image_size              INTEGER NOT NULL,
    image_gray_u8           BLOB NOT NULL,
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
    expected_rotation_deg   INTEGER NOT NULL,
    detector_candidate_id   TEXT,
    detector_review_decision TEXT,
    invalid_reason          TEXT
);

CREATE INDEX idx_sample_split_class
ON sample(split, class_index);

CREATE INDEX idx_sample_split_source_class
ON sample(split, source, class_index);

CREATE INDEX idx_sample_capture
ON sample(capture_id);

CREATE UNIQUE INDEX idx_sample_detector_candidate
ON sample(detector_candidate_id)
WHERE detector_candidate_id IS NOT NULL;
"""

BASE_COPY_COLUMNS = (
    "sample_id",
    "split",
    "source",
    "source_partition",
    "base_label",
    "class_index",
    "original_label",
    "source_label",
    "quality_audit_decision",
    "crop_id",
    "image_size",
    "image_gray_u8",
    "original_width",
    "original_height",
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


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Build a gray35 compact classifier DB by copying the proven gray34 compact "
            "dataset and appending only human-reviewed NanoDet-derived crops. The 35th "
            "class is invalid; classifier predictions and detector/GT suggestions never "
            "become training labels automatically."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--base-database",
        type=Path,
        help="Defaults to .local/recognition/tile_classifier_datasets/gray34_jp500_seed42.sqlite.",
    )
    parser.add_argument(
        "--detector-database",
        type=Path,
        help="Defaults to .local/recognition/detector_crop_dataset/dataset.sqlite.",
    )
    parser.add_argument(
        "--review-database",
        type=Path,
        help=(
            "Optional explicit human-review sidecar. By default the detector run key is read "
            "from detector dataset.sqlite and reviews.<detector_run_key>.sqlite is used."
        ),
    )
    parser.add_argument(
        "--output-database",
        type=Path,
        help="Defaults to .local/recognition/tile_classifier_datasets/gray35_jp500_seed42.sqlite.",
    )
    parser.add_argument(
        "--fallback-manual-train-fraction",
        type=float,
        default=0.80,
        help=(
            "Capture-level split used only for reviewed detector captures not represented "
            "in the base compact DB. Defaults to 0.80."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.fallback_manual_train_fraction) < 1.0:
        raise ValueError("--fallback-manual-train-fraction must be between 0 and 1")

    repository_root = args.repository_root.resolve()
    default_detector_root = repository_root / ".local" / "recognition" / "detector_crop_dataset"
    classifier_root = repository_root / ".local" / "recognition" / "tile_classifier_datasets"
    base_database = (
        args.base_database.resolve()
        if args.base_database is not None
        else classifier_root / "gray34_jp500_seed42.sqlite"
    )
    detector_database = (
        args.detector_database.resolve()
        if args.detector_database is not None
        else default_detector_root / "dataset.sqlite"
    )
    if not detector_database.is_file():
        raise FileNotFoundError(detector_database)
    detector_root = detector_database.parent if args.detector_database is not None else default_detector_root
    detector_run_key = load_detector_run_key_from_path(detector_database)
    review_database = (
        args.review_database.resolve()
        if args.review_database is not None
        else detector_root / f"reviews.{detector_run_key}.sqlite"
    )
    output_database = (
        args.output_database.resolve()
        if args.output_database is not None
        else classifier_root / "gray35_jp500_seed42.sqlite"
    )

    summary = build_gray35_dataset(
        base_database=base_database,
        detector_database=detector_database,
        review_database=review_database,
        output_database=output_database,
        seed=int(args.seed),
        fallback_manual_train_fraction=float(args.fallback_manual_train_fraction),
        force=bool(args.force),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_gray35_dataset(
    *,
    base_database: Path,
    detector_database: Path,
    review_database: Path,
    output_database: Path,
    seed: int = 42,
    fallback_manual_train_fraction: float = 0.80,
    force: bool = False,
) -> dict[str, Any]:
    base_database = base_database.resolve()
    detector_database = detector_database.resolve()
    review_database = review_database.resolve()
    output_database = output_database.resolve()
    for path in (base_database, detector_database, review_database):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_database in (base_database, detector_database, review_database):
        raise ValueError("Output database must differ from all input databases")
    if output_database.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_database}. Use --force to replace it.")

    output_database.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_database_path(output_database)
    try:
        summary = _create_gray35_dataset(
            base_database=base_database,
            detector_database=detector_database,
            review_database=review_database,
            output_database=temporary,
            seed=seed,
            fallback_manual_train_fraction=fallback_manual_train_fraction,
        )
        replace_database(temporary, output_database)
    except Exception:
        remove_sqlite_files(temporary)
        raise

    summary["database"] = str(output_database)
    summary["database_bytes"] = output_database.stat().st_size
    summary_path = output_database.with_suffix(".summary.json")
    atomic_write_json(summary_path, summary)
    return summary


def _create_gray35_dataset(
    *,
    base_database: Path,
    detector_database: Path,
    review_database: Path,
    output_database: Path,
    seed: int,
    fallback_manual_train_fraction: float,
) -> dict[str, Any]:
    base = sqlite3.connect(sqlite_readonly_uri(base_database), uri=True, timeout=60)
    base.row_factory = sqlite3.Row
    detector = sqlite3.connect(sqlite_readonly_uri(detector_database), uri=True, timeout=60)
    detector.row_factory = sqlite3.Row
    reviews = sqlite3.connect(sqlite_readonly_uri(review_database), uri=True, timeout=60)
    reviews.row_factory = sqlite3.Row
    target = sqlite3.connect(output_database, timeout=60)
    target.row_factory = sqlite3.Row
    try:
        validate_base_database(base)
        detector_run_key = load_detector_run_key(detector)
        validate_review_database(reviews, expected_detector_run_key=detector_run_key)
        target.execute("PRAGMA journal_mode = WAL")
        target.execute("PRAGMA synchronous = NORMAL")
        target.execute("PRAGMA temp_store = MEMORY")
        target.executescript(SCHEMA)

        base_metadata = {
            str(row["key"]): str(row["value"])
            for row in base.execute("SELECT key, value FROM experiment_metadata")
        }
        image_size = int(base_metadata["image_size"])
        copied_count = copy_base_samples(base, target)
        capture_splits = load_base_capture_splits(base)

        reviewed_rows = load_reviewed_detector_rows(detector, reviews)
        if not reviewed_rows:
            raise ValueError(
                "No human-reviewed detector crops found. Review candidates before building gray35."
            )
        detector_counts: Counter[str] = Counter()
        detector_split_counts: Counter[str] = Counter()
        excluded_review_counts: Counter[str] = Counter()
        fallback_captures: dict[str, str] = {}
        insert_sql = """
            INSERT INTO sample(
                sample_id, split, source, source_partition, base_label, class_index,
                original_label, source_label, quality_audit_decision,
                crop_id, image_size, image_gray_u8,
                original_width, original_height, source_image_path, source_image_id,
                source_annotation_id, capture_id, layout_id, region, brightness, shadow,
                annotation_angle_deg, expected_rotation_deg,
                detector_candidate_id, detector_review_decision, invalid_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        prepared: list[tuple[Any, ...]] = []
        for row in reviewed_rows:
            decision = str(row["decision"])
            if decision == "valid":
                label = str(row["review_label"])
                if label not in LABEL_TO_INDEX or label == INVALID_LABEL:
                    raise ValueError(f"Unsupported reviewed valid label for {row['candidate_id']}: {label}")
                invalid_reason = None
            elif decision == "invalid":
                invalid_reason = str(row["invalid_reason"])
                if invalid_reason in EXCLUDED_INVALID_REASONS:
                    excluded_review_counts[invalid_reason] += 1
                    continue
                label = INVALID_LABEL
            else:
                raise ValueError(f"Unsupported detector review decision: {decision}")

            capture_id = str(row["capture_id"])
            split = capture_splits.get(capture_id)
            if split is None:
                split = fallback_capture_split(
                    capture_id,
                    seed=seed,
                    train_fraction=fallback_manual_train_fraction,
                )
                fallback_captures[capture_id] = split
            if split not in {"train", "manual_val"}:
                raise ValueError(f"Detector manual crop mapped to invalid split {split}: {capture_id}")

            gray = preprocess_gray_u8(bytes(row["image_png"]), image_size=image_size)
            candidate_id = str(row["candidate_id"])
            sample_id = f"{split}:detector:{candidate_id}"
            source_path = str(row["source_region_path"])
            prepared.append(
                (
                    sample_id,
                    split,
                    "detector_manual",
                    "capture",
                    label,
                    LABEL_TO_INDEX[label],
                    label,
                    label,
                    None,
                    candidate_id,
                    image_size,
                    gray,
                    int(row["crop_width"]),
                    int(row["crop_height"]),
                    source_path,
                    capture_id,
                    str(row["detection_index"]),
                    capture_id,
                    str(row["layout_id"]),
                    str(row["region"]),
                    str(row["brightness"]),
                    str(row["shadow"]),
                    0.0,
                    0,
                    candidate_id,
                    decision,
                    invalid_reason,
                )
            )
            detector_counts[label] += 1
            detector_split_counts[split] += 1

        if detector_counts[INVALID_LABEL] < 1:
            raise ValueError(
                "No human-reviewed invalid detector crops found; gray35 requires invalid examples."
            )
        target.executemany(insert_sql, prepared)

        metadata = dict(base_metadata)
        metadata.update(
            {
                "schema_version": SCHEMA_VERSION,
                "parent_gray34_database": str(base_database),
                "detector_candidate_database": str(detector_database),
                "detector_review_database": str(review_database),
                "detector_run_key": detector_run_key,
                "base_labels": json.dumps(CLASS_LABELS, ensure_ascii=False),
                "class_count": str(len(CLASS_LABELS)),
                "task": "gray35_base_tile_plus_invalid",
                "invalid_class": INVALID_LABEL,
                "invalid_label_policy": "human_review_only",
                "detector_prediction_label_policy": "never_used_as_training_truth",
                "detector_suggestion_label_policy": "never_used_as_training_truth",
                "fallback_manual_train_fraction": repr(fallback_manual_train_fraction),
                "fallback_split_seed": str(seed),
            }
        )
        target.executemany(
            "INSERT INTO experiment_metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        target.commit()
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        counts = dataset_counts(target)
        return {
            "status": "completed",
            "base_database": str(base_database),
            "detector_database": str(detector_database),
            "review_database": str(review_database),
            "image_size": image_size,
            "class_labels": list(CLASS_LABELS),
            "copied_gray34_samples": copied_count,
            "reviewed_detector_samples": len(prepared),
            "excluded_review_counts": dict(sorted(excluded_review_counts.items())),
            "reviewed_detector_counts_by_label": dict(sorted(detector_counts.items())),
            "reviewed_detector_counts_by_split": dict(sorted(detector_split_counts.items())),
            "fallback_capture_splits": dict(sorted(fallback_captures.items())),
            **counts,
        }
    finally:
        target.close()
        reviews.close()
        detector.close()
        base.close()


def validate_base_database(connection: sqlite3.Connection) -> None:
    metadata = {
        str(row["key"]): str(row["value"])
        for row in connection.execute("SELECT key, value FROM experiment_metadata")
    }
    labels = tuple(json.loads(metadata["base_labels"]))
    if labels != tuple(BASE_LABELS):
        raise ValueError(
            "Base compact DB must be the 34-class base-tile dataset in canonical class order"
        )
    if "image_size" not in metadata:
        raise ValueError("Base compact DB has no image_size metadata")


def load_detector_run_key(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT value FROM dataset_metadata WHERE key='detector_run_key'"
    ).fetchone()
    if row is None:
        raise ValueError("Detector candidate database has no detector_run_key metadata")
    return str(row[0])


def load_detector_run_key_from_path(database: Path) -> str:
    with closing(sqlite3.connect(sqlite_readonly_uri(database), uri=True, timeout=30)) as connection:
        return load_detector_run_key(connection)


def validate_review_database(
    connection: sqlite3.Connection,
    *,
    expected_detector_run_key: str,
) -> None:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='review'"
    ).fetchone()
    if table is None:
        raise ValueError("Detector review database has no review table")
    metadata_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_metadata'"
    ).fetchone()
    if metadata_table is None:
        raise ValueError("Detector review database has no review_metadata table")
    row = connection.execute(
        "SELECT value FROM review_metadata WHERE key='detector_run_key'"
    ).fetchone()
    if row is None:
        raise ValueError("Detector review database is not bound to a detector_run_key")
    if str(row[0]) != expected_detector_run_key:
        raise ValueError(
            "Detector review database belongs to another detector run: "
            f"stored={row[0]}, requested={expected_detector_run_key}"
        )


def copy_base_samples(source: sqlite3.Connection, target: sqlite3.Connection) -> int:
    columns = ", ".join(BASE_COPY_COLUMNS)
    placeholders = ", ".join("?" for _ in BASE_COPY_COLUMNS)
    insert_sql = f"""
        INSERT INTO sample(
            {columns}, detector_candidate_id, detector_review_decision, invalid_reason
        ) VALUES ({placeholders}, NULL, NULL, NULL)
    """
    count = 0
    batch: list[tuple[Any, ...]] = []
    for row in source.execute(f"SELECT {columns} FROM sample ORDER BY sample_id"):
        batch.append(tuple(row[column] for column in BASE_COPY_COLUMNS))
        count += 1
        if len(batch) >= 5000:
            target.executemany(insert_sql, batch)
            batch.clear()
    if batch:
        target.executemany(insert_sql, batch)
    return count


def load_base_capture_splits(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT capture_id, split, COUNT(*) AS sample_count
        FROM sample
        WHERE capture_id IS NOT NULL AND source = 'manual'
        GROUP BY capture_id, split
        ORDER BY capture_id, split
        """
    ).fetchall()
    by_capture: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_capture[str(row["capture_id"])].add(str(row["split"]))
    conflicts = {key: values for key, values in by_capture.items() if len(values) != 1}
    if conflicts:
        raise ValueError(f"Base compact DB has capture leakage across splits: {conflicts}")
    return {key: next(iter(values)) for key, values in by_capture.items()}


def load_reviewed_detector_rows(
    detector: sqlite3.Connection,
    reviews: sqlite3.Connection,
) -> list[dict[str, Any]]:
    review_rows = reviews.execute(
        """
        SELECT candidate_id, decision, label AS review_label, invalid_reason
        FROM review
        ORDER BY candidate_id
        """
    ).fetchall()
    all_reviews = {str(row["candidate_id"]): dict(row) for row in review_rows}
    if not all_reviews:
        return []

    candidate_rows = detector.execute(
        """
        SELECT candidate_id, capture_id, region, detection_index, detection_confidence,
               bbox_x, bbox_y, bbox_width, bbox_height
        FROM candidate
        ORDER BY capture_id, region, detection_index, candidate_id
        """
    ).fetchall()
    all_candidate_ids = {str(row["candidate_id"]) for row in candidate_rows}
    missing = sorted(set(all_reviews) - all_candidate_ids)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            f"Review DB contains {len(missing)} candidates absent from detector DB: "
            f"{preview}{suffix}. The review sidecar likely belongs to another detector dataset."
        )

    threshold_row = detector.execute(
        "SELECT value FROM dataset_metadata WHERE key='duplicate_overlap_threshold'"
    ).fetchone()
    threshold = 0.80 if threshold_row is None else float(threshold_row[0])
    plan = build_duplicate_plan(
        (
            DetectorCandidate(
                candidate_id=str(row["candidate_id"]),
                capture_id=str(row["capture_id"]),
                region=str(row["region"]),
                detection_index=int(row["detection_index"]),
                confidence=float(row["detection_confidence"]),
                bbox_x=float(row["bbox_x"]),
                bbox_y=float(row["bbox_y"]),
                bbox_width=float(row["bbox_width"]),
                bbox_height=float(row["bbox_height"]),
            )
            for row in candidate_rows
        ),
        threshold=threshold,
    )
    review_by_id = {
        candidate_id: review
        for candidate_id, review in all_reviews.items()
        if candidate_id in plan.winner_candidate_ids
    }

    result: list[dict[str, Any]] = []
    for row in detector.execute(
        """
        SELECT candidate_id, capture_id, layout_id, brightness, shadow, region,
               source_region_path, detection_index, crop_width, crop_height, image_png
        FROM candidate
        ORDER BY candidate_id
        """
    ):
        candidate_id = str(row["candidate_id"])
        if candidate_id not in plan.winner_candidate_ids:
            continue
        review = review_by_id.get(candidate_id)
        if review is None:
            continue
        values = dict(row)
        values.update(review)
        result.append(values)
    return result


def fallback_capture_split(
    capture_id: str,
    *,
    seed: int,
    train_fraction: float,
) -> str:
    digest = hashlib.sha256(f"{seed}\0detector-capture\0{capture_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "train" if value < train_fraction else "manual_val"


def dataset_counts(connection: sqlite3.Connection) -> dict[str, Any]:
    total = int(connection.execute("SELECT COUNT(*) FROM sample").fetchone()[0])
    by_split = {
        str(row["split"]): int(row["count"])
        for row in connection.execute(
            "SELECT split, COUNT(*) AS count FROM sample GROUP BY split ORDER BY split"
        )
    }
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
    by_source = {
        str(row["source"]): int(row["count"])
        for row in connection.execute(
            "SELECT source, COUNT(*) AS count FROM sample GROUP BY source ORDER BY source"
        )
    }
    return {
        "sample_count": total,
        "counts_by_split": by_split,
        "counts_by_split_and_label": dict(by_split_label),
        "counts_by_source": by_source,
    }


def temporary_database_path(output_database: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output_database.name}.", suffix=".tmp.sqlite", dir=output_database.parent
    )
    os.close(descriptor)
    os.unlink(name)
    return Path(name)


def replace_database(source: Path, destination: Path) -> None:
    remove_sqlite_files(destination)
    os.replace(source, destination)


def remove_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, payload: Any) -> None:
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
