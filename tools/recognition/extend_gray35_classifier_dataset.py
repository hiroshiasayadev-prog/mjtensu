from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

try:
    from .build_gray35_classifier_dataset import (
        CLASS_LABELS,
        INVALID_LABEL,
        EXCLUDED_INVALID_REASONS,
        LABEL_TO_INDEX,
        load_detector_run_key,
        load_reviewed_detector_rows,
        validate_review_database,
    )
    from .build_tile_classifier_dataset import preprocess_gray_u8, sqlite_readonly_uri
except ImportError:  # direct script execution
    from build_gray35_classifier_dataset import (
        CLASS_LABELS,
        INVALID_LABEL,
        EXCLUDED_INVALID_REASONS,
        LABEL_TO_INDEX,
        load_detector_run_key,
        load_reviewed_detector_rows,
        validate_review_database,
    )
    from build_tile_classifier_dataset import preprocess_gray_u8, sqlite_readonly_uri


BUILDER_VERSION = "1"


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Create a controlled gray35 extension by copying an existing gray35 compact DB "
            "unchanged and appending only human-reviewed detector crops from one additional "
            "detector/review dataset. Intended for clean v2->v3 ablations."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--base-database", type=Path, required=True)
    parser.add_argument("--detector-database", type=Path, required=True)
    parser.add_argument("--review-database", type=Path)
    parser.add_argument("--output-database", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector_database = args.detector_database.resolve()
    review_database = (
        args.review_database.resolve()
        if args.review_database is not None
        else default_review_database(detector_database)
    )
    summary = extend_gray35_dataset(
        base_database=args.base_database.resolve(),
        detector_database=detector_database,
        review_database=review_database,
        output_database=args.output_database.resolve(),
        force=bool(args.force),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def default_review_database(detector_database: Path) -> Path:
    with closing(
        sqlite3.connect(sqlite_readonly_uri(detector_database), uri=True, timeout=30)
    ) as connection:
        run_key = load_detector_run_key(connection)
    return detector_database.parent / f"reviews.{run_key}.sqlite"


def extend_gray35_dataset(
    *,
    base_database: Path,
    detector_database: Path,
    review_database: Path,
    output_database: Path,
    force: bool = False,
) -> dict[str, Any]:
    base_database = base_database.resolve()
    detector_database = detector_database.resolve()
    review_database = review_database.resolve()
    output_database = output_database.resolve()
    for path in (base_database, detector_database, review_database):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_database in {base_database, detector_database, review_database}:
        raise ValueError("Output database must differ from all input databases")
    if output_database.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_database}. Use --force to replace it.")

    output_database.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_database_path(output_database)
    try:
        summary = _create_extension(
            base_database=base_database,
            detector_database=detector_database,
            review_database=review_database,
            output_database=temporary,
        )
        replace_database(temporary, output_database)
    except Exception:
        remove_sqlite_files(temporary)
        raise

    summary["database"] = str(output_database)
    summary["database_bytes"] = output_database.stat().st_size
    summary_path = output_database.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _create_extension(
    *,
    base_database: Path,
    detector_database: Path,
    review_database: Path,
    output_database: Path,
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
        base_metadata = load_experiment_metadata(base)
        validate_gray35_base(base_metadata)
        image_size = int(base_metadata["image_size"])

        detector_metadata = load_dataset_metadata(detector)
        detector_run_key = load_detector_run_key(detector)
        validate_review_database(reviews, expected_detector_run_key=detector_run_key)
        source = detector_metadata.get("source")
        source_partition = detector_metadata.get("source_partition")
        if source != "jp":
            raise ValueError(
                "This controlled extender currently accepts only JP detector datasets; "
                f"found source={source!r}"
            )
        if source_partition != "train":
            raise ValueError(
                "Only jp/train may be appended to a training DB in this ablation; "
                f"found source_partition={source_partition!r}"
            )

        # SQLite backup copies the parent gray35 DB exactly, including all manual detector
        # samples and the frozen validation sets. v3 therefore differs from v2 only by the
        # reviewed JP rows appended below.
        base.backup(target)
        target.execute("PRAGMA journal_mode = WAL")
        target.execute("PRAGMA synchronous = NORMAL")

        reviewed_rows = load_reviewed_detector_rows(detector, reviews)
        if not reviewed_rows:
            raise ValueError("No human-reviewed JP detector crops found")

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
        counts_by_label: Counter[str] = Counter()
        counts_by_decision: Counter[str] = Counter()
        excluded_review_counts: Counter[str] = Counter()
        for row in reviewed_rows:
            decision = str(row["decision"])
            if decision == "valid":
                label = str(row["review_label"])
                if label not in LABEL_TO_INDEX or label == INVALID_LABEL:
                    raise ValueError(
                        f"Unsupported reviewed valid label for {row['candidate_id']}: {label}"
                    )
                invalid_reason = None
            elif decision == "invalid":
                invalid_reason = str(row["invalid_reason"])
                if invalid_reason in EXCLUDED_INVALID_REASONS:
                    excluded_review_counts[invalid_reason] += 1
                    continue
                label = INVALID_LABEL
            else:
                raise ValueError(f"Unsupported review decision: {decision}")

            candidate_id = str(row["candidate_id"])
            capture_id = str(row["capture_id"])
            gray = preprocess_gray_u8(bytes(row["image_png"]), image_size=image_size)
            prepared.append(
                (
                    f"train:detector:{candidate_id}",
                    "train",
                    "detector_jp",
                    "train",
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
                    str(row["source_region_path"]),
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
            counts_by_label[label] += 1
            counts_by_decision[decision] += 1

        if not prepared:
            raise ValueError("All reviewed JP crops were excluded; nothing to append")
        target.executemany(insert_sql, prepared)

        metadata_updates = {
            "extension_builder": "extend_gray35_classifier_dataset",
            "extension_builder_version": BUILDER_VERSION,
            "parent_gray35_database": str(base_database),
            "additional_detector_candidate_database": str(detector_database),
            "additional_detector_review_database": str(review_database),
            "additional_detector_run_key": detector_run_key,
            "additional_source": "jp",
            "additional_source_partition": "train",
            "additional_reviewed_sample_count": str(len(prepared)),
            "additional_reviewed_counts_by_label": json.dumps(
                dict(sorted(counts_by_label.items())), ensure_ascii=False
            ),
            "additional_reviewed_counts_by_decision": json.dumps(
                dict(sorted(counts_by_decision.items())), ensure_ascii=False
            ),
            "additional_excluded_review_counts": json.dumps(
                dict(sorted(excluded_review_counts.items())), ensure_ascii=False
            ),
        }
        target.executemany(
            """
            INSERT INTO experiment_metadata(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            metadata_updates.items(),
        )
        target.commit()
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        base_counts = count_samples(base)
        final_counts = count_samples(target)
        return {
            "status": "completed",
            "base_database": str(base_database),
            "detector_database": str(detector_database),
            "review_database": str(review_database),
            "detector_run_key": detector_run_key,
            "source": source,
            "source_partition": source_partition,
            "base_sample_count": base_counts["total"],
            "appended_reviewed_samples": len(prepared),
            "appended_counts_by_decision": dict(sorted(counts_by_decision.items())),
            "appended_counts_by_label": dict(sorted(counts_by_label.items())),
            "excluded_review_counts": dict(sorted(excluded_review_counts.items())),
            "final_sample_count": final_counts["total"],
            "counts_by_split_before": base_counts["by_split"],
            "counts_by_split_after": final_counts["by_split"],
        }
    finally:
        target.close()
        reviews.close()
        detector.close()
        base.close()


def load_experiment_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute("SELECT key, value FROM experiment_metadata").fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def load_dataset_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute("SELECT key, value FROM dataset_metadata").fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def validate_gray35_base(metadata: dict[str, str]) -> None:
    labels = tuple(json.loads(metadata.get("base_labels", "[]")))
    if labels != CLASS_LABELS:
        raise ValueError(
            "Base DB must be an existing gray35 compact dataset in canonical class order"
        )
    if int(metadata.get("class_count", "0")) != len(CLASS_LABELS):
        raise ValueError("Base DB class_count is not 35")
    if "image_size" not in metadata:
        raise ValueError("Base DB has no image_size metadata")


def count_samples(connection: sqlite3.Connection) -> dict[str, Any]:
    total = int(connection.execute("SELECT COUNT(*) FROM sample").fetchone()[0])
    by_split = {
        str(row[0]): int(row[1])
        for row in connection.execute(
            "SELECT split, COUNT(*) FROM sample GROUP BY split ORDER BY split"
        )
    }
    return {"total": total, "by_split": by_split}


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


if __name__ == "__main__":
    main()
