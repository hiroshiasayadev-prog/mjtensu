from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from build_tile_classifier_dataset import (
    BASE_LABELS,
    BASE_LABEL_TO_INDEX,
    SOURCE_COLUMNS,
    base_label,
    effective_quality_label,
    load_quality_reviews,
    preprocess_gray_u8,
    sqlite_readonly_uri,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--base-database", type=Path, required=True)
    p.add_argument(
        "--source-database",
        type=Path,
        default=repo / ".local/recognition/tile_crop_dataset/dataset.sqlite",
    )
    p.add_argument(
        "--quality-database",
        type=Path,
        default=repo / ".local/recognition/tile_crop_dataset/quality_audit.sqlite",
    )
    p.add_argument("--output-database", type=Path, required=True)
    p.add_argument("--target-per-class", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_per_class < 1:
        raise ValueError("--target-per-class must be positive")
    for path in (args.base_database, args.source_database, args.quality_database):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_database.exists() and not args.force:
        raise FileExistsError(args.output_database)

    args.output_database.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_database.with_suffix(args.output_database.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    shutil.copy2(args.base_database, tmp)

    reviews, quality_summary = load_quality_reviews(args.quality_database)
    source = sqlite3.connect(sqlite_readonly_uri(args.source_database), uri=True, timeout=60)
    source.row_factory = sqlite3.Row
    target = sqlite3.connect(tmp, timeout=60)
    target.row_factory = sqlite3.Row
    try:
        summary = augment(
            source=source,
            target=target,
            reviews=reviews,
            seed=args.seed,
            target_per_class=args.target_per_class,
            quality_summary=quality_summary,
        )
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        check = target.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            raise RuntimeError(f"sqlite integrity_check failed: {check}")
        target.execute("VACUUM")
    finally:
        target.close()
        source.close()

    args.output_database.unlink(missing_ok=True)
    tmp.replace(args.output_database)
    summary["database"] = str(args.output_database)
    summary["database_bytes"] = args.output_database.stat().st_size
    args.output_database.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def augment(
    *,
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    reviews,
    seed: int,
    target_per_class: int,
    quality_summary: dict,
) -> dict:
    current = {
        str(r["base_label"]): int(r["n"])
        for r in target.execute(
            "SELECT base_label, COUNT(*) n FROM sample "
            "WHERE split='manual_val' GROUP BY base_label"
        )
    }
    used_valid = {
        str(r["crop_id"])
        for r in target.execute(
            "SELECT crop_id FROM sample "
            "WHERE source='jp' AND source_partition='valid'"
        )
    }

    candidates: dict[str, list[tuple[int, str, str, str | None]]] = defaultdict(list)
    excluded_quality = 0
    for row in source.execute(
        "SELECT rowid, crop_id, tile_label FROM tile_crop "
        "WHERE source='jp' AND source_partition='valid' ORDER BY rowid"
    ):
        crop_id = str(row["crop_id"])
        if crop_id in used_valid:
            continue
        source_label = str(row["tile_label"])
        effective, decision = effective_quality_label(crop_id, source_label, reviews)
        if effective is None:
            excluded_quality += 1
            continue
        try:
            label = base_label(effective)
        except ValueError:
            continue
        candidates[label].append((int(row["rowid"]), crop_id, effective, decision))

    chosen: list[tuple[str, int, str, str, str | None]] = []
    deficits: dict[str, int] = {}
    available: dict[str, int] = {}
    for label in BASE_LABELS:
        have = current.get(label, 0)
        need = max(0, target_per_class - have)
        deficits[label] = need
        pool = candidates[label]
        available[label] = len(pool)
        if len(pool) < need:
            raise ValueError(f"{label}: need {need}, only {len(pool)} unused JP valid crops")
        rng = random.Random(stable_seed(seed, f"manual-jp-val:{label}"))
        rng.shuffle(pool)
        for rowid, crop_id, effective, decision in pool[:need]:
            chosen.append((label, rowid, crop_id, effective, decision))

    rowids = [rowid for _, rowid, _, _, _ in chosen]
    by_rowid: dict[int, sqlite3.Row] = {}
    for start in range(0, len(rowids), 500):
        chunk = rowids[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        for row in source.execute(
            f"SELECT rowid _rowid, {', '.join(SOURCE_COLUMNS)} "
            f"FROM tile_crop WHERE rowid IN ({placeholders})",
            chunk,
        ):
            by_rowid[int(row["_rowid"])] = row

    sql = """
        INSERT INTO sample(
            sample_id, split, source, source_partition, base_label, class_index,
            original_label, source_label, quality_audit_decision,
            crop_id, image_size, image_gray_u8,
            original_width, original_height, source_image_path, source_image_id,
            source_annotation_id, capture_id, layout_id, region, brightness, shadow,
            annotation_angle_deg, expected_rotation_deg,
            detector_candidate_id, detector_review_decision, invalid_reason, target
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    prepared = []
    for label, rowid, crop_id, effective, decision in chosen:
        row = by_rowid[rowid]
        gray = preprocess_gray_u8(bytes(row["image_png"]), image_size=64)
        source_label = str(row["tile_label"])
        prepared.append((
            f"manual_val:jp_extra:{crop_id}",
            "manual_val", "jp", "valid", label, BASE_LABEL_TO_INDEX[label],
            effective, source_label, decision,
            crop_id, 64, gray,
            int(row["image_width"]), int(row["image_height"]),
            str(row["source_image_path"]),
            None if row["source_image_id"] is None else str(row["source_image_id"]),
            str(row["source_annotation_id"]),
            None if row["capture_id"] is None else str(row["capture_id"]),
            None if row["layout_id"] is None else str(row["layout_id"]),
            None if row["region"] is None else str(row["region"]),
            None if row["brightness"] is None else str(row["brightness"]),
            None if row["shadow"] is None else str(row["shadow"]),
            float(row["annotation_angle_deg"]), int(row["expected_rotation_deg"]),
            None, None, None, label,
        ))
    target.executemany(sql, prepared)

    metadata = {
        "manual_jp_val_seed": str(seed),
        "manual_jp_val_target_per_valid_class": str(target_per_class),
        "manual_jp_val_source": "jp",
        "manual_jp_val_source_partition": "valid",
        "manual_jp_val_existing_jp_val_excluded": "true",
        "manual_jp_val_added_samples": str(len(prepared)),
    }
    for key, value in metadata.items():
        target.execute(
            "INSERT OR REPLACE INTO experiment_metadata(key,value) VALUES (?,?)",
            (key, value),
        )
    target.commit()

    final_counts = {
        str(r["base_label"]): int(r["n"])
        for r in target.execute(
            "SELECT base_label, COUNT(*) n FROM sample "
            "WHERE split='manual_val' GROUP BY base_label"
        )
    }
    for label in BASE_LABELS:
        if final_counts.get(label) != target_per_class:
            raise RuntimeError(f"{label}: final manual_val count={final_counts.get(label)}")

    overlap_jp_val = target.execute(
        """
        SELECT COUNT(*)
        FROM sample a
        JOIN sample b ON a.crop_id=b.crop_id
        WHERE a.split='manual_val' AND a.source='jp'
          AND b.split='jp_val'
        """
    ).fetchone()[0]
    overlap_train = target.execute(
        """
        SELECT COUNT(*)
        FROM sample a
        JOIN sample b ON a.crop_id=b.crop_id
        WHERE a.split='manual_val' AND a.source='jp'
          AND b.split='train'
        """
    ).fetchone()[0]
    if overlap_jp_val or overlap_train:
        raise RuntimeError(
            f"holdout leakage: jp_val={overlap_jp_val} train={overlap_train}"
        )

    split_counts = {
        str(r["split"]): int(r["n"])
        for r in target.execute(
            "SELECT split, COUNT(*) n FROM sample GROUP BY split ORDER BY split"
        )
    }
    return {
        "status": "completed",
        "seed": seed,
        "target_per_valid_class": target_per_class,
        "added_jp_valid": len(prepared),
        "excluded_existing_jp_valid": len(used_valid),
        "excluded_quality": excluded_quality,
        "quality_audit": quality_summary,
        "manual_val_before": current,
        "deficits_filled": deficits,
        "unused_jp_valid_available": available,
        "manual_val_after": final_counts,
        "invalid_after": final_counts.get("invalid", 0),
        "overlap_with_jp_val": int(overlap_jp_val),
        "overlap_with_train": int(overlap_train),
        "splits": split_counts,
    }


if __name__ == "__main__":
    main()
