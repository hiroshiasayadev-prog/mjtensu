from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from PIL import Image, ImageOps

if __package__:
    from .build_detector_crop_dataset import (
        GroundTruth,
        PreparedDetection,
        build_detector_run_key,
        encode_png,
        match_detection_to_ground_truth,
        repository_relative_or_absolute,
        sha256_file,
        suggest_state,
        suppress_near_duplicate_detections,
    )
    from .build_tile_classifier_dataset import BASE_LABELS
    from .build_tile_crop_dataset import normalize_jp_tile_label
    from .nanodet.evaluate_composite_onnx import INPUT_SIZE, decode_output, preprocess_image
else:
    from build_detector_crop_dataset import (
        GroundTruth,
        PreparedDetection,
        build_detector_run_key,
        encode_png,
        match_detection_to_ground_truth,
        repository_relative_or_absolute,
        sha256_file,
        suggest_state,
        suppress_near_duplicate_detections,
    )
    from build_tile_classifier_dataset import BASE_LABELS
    from build_tile_crop_dataset import normalize_jp_tile_label
    from nanodet.evaluate_composite_onnx import INPUT_SIZE, decode_output, preprocess_image


BUILDER_VERSION = "1"
JP_SPLITS = ("train", "valid", "test")
RED_FIVE_TO_BASE = {"red5m": "5m", "red5p": "5p", "red5s": "5s"}

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE dataset_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE candidate (
    candidate_id               TEXT PRIMARY KEY,
    source                     TEXT NOT NULL CHECK (source = 'jp'),
    capture_id                 TEXT NOT NULL,
    campaign_id                TEXT NOT NULL,
    layout_id                  TEXT NOT NULL,
    layout_ordinal             INTEGER NOT NULL,
    brightness                 TEXT NOT NULL,
    shadow                     TEXT NOT NULL,
    region                     TEXT NOT NULL,
    source_region_path         TEXT NOT NULL,
    source_composite_path      TEXT NOT NULL,
    detector_model_name        TEXT NOT NULL,
    detector_model_sha256      TEXT NOT NULL,
    detector_run_key           TEXT NOT NULL,
    detection_index            INTEGER NOT NULL,
    detection_confidence       REAL NOT NULL,
    bbox_x                     REAL NOT NULL,
    bbox_y                     REAL NOT NULL,
    bbox_width                 REAL NOT NULL,
    bbox_height                REAL NOT NULL,
    crop_width                 INTEGER NOT NULL,
    crop_height                INTEGER NOT NULL,
    image_png                  BLOB NOT NULL,
    suggested_state            TEXT NOT NULL CHECK (
        suggested_state IN ('single_gt', 'multi_gt', 'partial', 'background')
    ),
    suggested_label            TEXT,
    best_gt_id                 TEXT,
    best_gt_label              TEXT,
    best_iou                   REAL NOT NULL,
    best_gt_coverage           REAL NOT NULL,
    best_detection_coverage    REAL NOT NULL,
    substantial_gt_count       INTEGER NOT NULL,
    gt_json                    TEXT NOT NULL,
    created_at                 TEXT NOT NULL,
    UNIQUE(detector_run_key, capture_id, detection_index)
);

CREATE INDEX idx_candidate_state
ON candidate(suggested_state, detection_confidence DESC);

CREATE INDEX idx_candidate_capture
ON candidate(capture_id, region, detection_index);

CREATE INDEX idx_candidate_label
ON candidate(suggested_label, suggested_state);

CREATE TABLE postprocess_decision (
    candidate_id         TEXT PRIMARY KEY REFERENCES candidate(candidate_id) ON DELETE CASCADE,
    detector_run_key     TEXT NOT NULL,
    status               TEXT NOT NULL CHECK (status IN ('keep', 'remove')),
    reason               TEXT CHECK (reason IS NULL OR reason = 'duplicate'),
    winner_candidate_id  TEXT REFERENCES candidate(candidate_id),
    overlap_ratio        REAL,
    created_at           TEXT NOT NULL,
    CHECK (
        (status = 'keep' AND reason IS NULL AND winner_candidate_id IS NULL AND overlap_ratio IS NULL)
        OR
        (status = 'remove' AND reason = 'duplicate' AND winner_candidate_id IS NOT NULL AND overlap_ratio IS NOT NULL)
    )
);

CREATE INDEX idx_postprocess_decision_status
ON postprocess_decision(status, reason, candidate_id);

CREATE INDEX idx_postprocess_decision_winner
ON postprocess_decision(winner_candidate_id, overlap_ratio DESC);
"""


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run NanoDet over a deterministic sample of Mahjong-jp v2 images and persist "
            "raw detector crops plus COCO-GT geometry hints for gray35 hard-negative mining."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--jp-root", type=Path)
    parser.add_argument("--split", choices=JP_SPLITS, default="train")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-database", type=Path)
    parser.add_argument(
        "--image-limit",
        type=int,
        default=5000,
        help="Deterministic image sample size. 0 means all images. Defaults to 5000.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.60)
    parser.add_argument("--duplicate-overlap-threshold", type=float, default=0.80)
    parser.add_argument("--max-detections", type=int, default=200)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    repository_root = args.repository_root.resolve()
    jp_root = (args.jp_root or repository_root / "data" / "coco_mahjong_jp_v2").resolve()
    split = str(args.split)
    annotation_path = jp_root / split / "_annotations.coco.json"
    image_root = jp_root / split
    model_path = (
        args.model
        or repository_root
        / ".local"
        / "recognition"
        / "nanodet_runs"
        / "E1_plus_m_320_real_capture_ft10_l10_seed42"
        / "model_best"
        / "nanodet-plus-m-320-real-capture-ft10-l10.onnx"
    ).resolve()
    output_database = (
        args.output_database
        or repository_root
        / ".local"
        / "recognition"
        / "jp_detector_crop_dataset"
        / split
        / "dataset.sqlite"
    ).resolve()

    for path in (annotation_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    if output_database.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_database}. Use --force to replace it.")
    output_database.parent.mkdir(parents=True, exist_ok=True)

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required") from error

    available = set(ort.get_available_providers())
    if args.provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider requested but unavailable")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif args.provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )

    session = ort.InferenceSession(str(model_path), providers=providers)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("Expected one NanoDet ONNX input and one output")
    input_name = inputs[0].name
    output_name = outputs[0].name

    payload = load_json(annotation_path)
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        raise ValueError(f"Invalid COCO payload: {annotation_path}")
    category_names = {int(item["id"]): str(item["name"]) for item in categories}
    annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    selected_images = select_images(
        images,
        split=split,
        seed=int(args.seed),
        image_limit=int(args.image_limit),
    )

    model_sha256 = sha256_file(model_path)
    detector_run_key = build_detector_run_key(
        model_sha256=model_sha256,
        confidence_threshold=float(args.confidence_threshold),
        nms_iou_threshold=float(args.nms_iou_threshold),
        duplicate_overlap_threshold=float(args.duplicate_overlap_threshold),
        max_detections=int(args.max_detections),
    )
    now = datetime.now(timezone.utc).isoformat()
    selection_hash = hashlib.sha256(
        "\n".join(str(item["id"]) for item in selected_images).encode("utf-8")
    ).hexdigest()

    temporary = temporary_database_path(output_database)
    target: sqlite3.Connection | None = None
    try:
        target = sqlite3.connect(temporary, timeout=60)
        target.row_factory = sqlite3.Row
        target.execute("PRAGMA journal_mode = WAL")
        target.execute("PRAGMA synchronous = NORMAL")
        target.executescript(SCHEMA)
        metadata = {
            "schema_version": "3",
            "builder_version": BUILDER_VERSION,
            "source": "jp",
            "source_partition": split,
            "jp_root": str(jp_root),
            "annotations": str(annotation_path),
            "annotations_sha256": sha256_file(annotation_path),
            "selected_image_count": str(len(selected_images)),
            "source_image_count": str(len(images)),
            "selection_seed": str(int(args.seed)),
            "selection_hash": selection_hash,
            "detector_model": str(model_path),
            "detector_model_name": model_path.name,
            "detector_model_sha256": model_sha256,
            "detector_run_key": detector_run_key,
            "confidence_threshold": repr(float(args.confidence_threshold)),
            "nms_iou_threshold": repr(float(args.nms_iou_threshold)),
            "duplicate_overlap_threshold": repr(float(args.duplicate_overlap_threshold)),
            "max_detections": str(int(args.max_detections)),
            "providers": json.dumps(session.get_providers()),
            "created_at": now,
            "coordinate_policy": "NanoDet 320x320 output scaled back to original COCO image coordinates",
            "suggestion_policy": (
                "single_gt:best_gt_coverage>=0.70,best_detection_coverage>=0.60,substantial_gt_count=1;"
                "multi_gt:substantial_gt_count>=2 where substantial means gt_coverage>=0.30;"
                "background:best_gt_coverage<0.10;otherwise partial"
            ),
        }
        target.executemany(
            "INSERT INTO dataset_metadata(key, value) VALUES (?, ?)", metadata.items()
        )

        insert_candidate_sql = """
            INSERT INTO candidate(
                candidate_id, source, capture_id, campaign_id, layout_id, layout_ordinal,
                brightness, shadow, region, source_region_path, source_composite_path,
                detector_model_name, detector_model_sha256, detector_run_key, detection_index,
                detection_confidence, bbox_x, bbox_y, bbox_width, bbox_height,
                crop_width, crop_height, image_png,
                suggested_state, suggested_label, best_gt_id, best_gt_label,
                best_iou, best_gt_coverage, best_detection_coverage, substantial_gt_count,
                gt_json, created_at
            ) VALUES (?, 'jp', ?, ?, 'full_image', ?, 'unknown', 'unknown', 'full_image', ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        insert_postprocess_sql = """
            INSERT INTO postprocess_decision(
                candidate_id, detector_run_key, status, reason,
                winner_candidate_id, overlap_ratio, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        counts: Counter[str] = Counter()
        raw_candidate_count = 0
        for ordinal, image_record in enumerate(selected_images, start=1):
            image_id = int(image_record["id"])
            image_path = resolve_image_path(image_root, image_record)
            gts = build_ground_truths(
                annotations_by_image.get(image_id, ()),
                category_names=category_names,
            )

            tensor, source_image = preprocess_image(image_path)
            try:
                source_width, source_height = source_image.size
                raw_output = session.run([output_name], {input_name: tensor})[0]
                detections = decode_output(
                    raw_output,
                    confidence_threshold=float(args.confidence_threshold),
                    nms_iou_threshold=float(args.nms_iou_threshold),
                    max_detections=int(args.max_detections),
                )

                prepared = [
                    PreparedDetection(
                        detection_index=index,
                        confidence=float(detection.score),
                        region="full_image",
                        local_rect=scale_detection_rect(
                            detection.box.x1,
                            detection.box.y1,
                            detection.box.x2,
                            detection.box.y2,
                            source_width=source_width,
                            source_height=source_height,
                        ),
                    )
                    for index, detection in enumerate(detections)
                ]
                _kept, suppressions = suppress_near_duplicate_detections(
                    prepared,
                    threshold=float(args.duplicate_overlap_threshold),
                )
                suppression_by_removed_index = {
                    item.removed.detection_index: item for item in suppressions
                }
                counts["duplicate_suppressed"] += len(suppressions)

                source_path = repository_relative_or_absolute(image_path, repository_root)
                capture_id = f"jp:{split}:{image_id}"
                campaign_id = f"coco_mahjong_jp_v2:{split}"
                inserted_ids: dict[int, str] = {}
                gt_json = json.dumps(
                    [gt.json_value() for gt in gts],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for item in prepared:
                    crop = crop_rect(source_image, item.local_rect)
                    if crop is None:
                        counts["empty_crop"] += 1
                        continue
                    matches = match_detection_to_ground_truth(item.local_rect, gts)
                    suggestion = suggest_state(matches)
                    best = matches[0] if matches else None
                    suggested_label = None
                    if suggestion == "single_gt" and best is not None and best.ground_truth.label in BASE_LABELS:
                        suggested_label = best.ground_truth.label
                    substantial_count = sum(match.gt_coverage >= 0.30 for match in matches)
                    candidate_id = f"jp:{split}:{detector_run_key}:{image_id}:{item.detection_index}"
                    target.execute(
                        insert_candidate_sql,
                        (
                            candidate_id,
                            capture_id,
                            campaign_id,
                            ordinal,
                            source_path,
                            source_path,
                            model_path.name,
                            model_sha256,
                            detector_run_key,
                            item.detection_index,
                            item.confidence,
                            item.local_rect[0],
                            item.local_rect[1],
                            item.local_rect[2],
                            item.local_rect[3],
                            crop.width,
                            crop.height,
                            encode_png(crop),
                            suggestion,
                            suggested_label,
                            None if best is None else best.ground_truth.box_id,
                            None if best is None else best.ground_truth.label,
                            0.0 if best is None else best.iou,
                            0.0 if best is None else best.gt_coverage,
                            0.0 if best is None else best.detection_coverage,
                            substantial_count,
                            gt_json,
                            now,
                        ),
                    )
                    crop.close()
                    inserted_ids[item.detection_index] = candidate_id
                    raw_candidate_count += 1
                    counts[f"raw_{suggestion}"] += 1

                for detection_index, candidate_id in inserted_ids.items():
                    suppression = suppression_by_removed_index.get(detection_index)
                    if suppression is None:
                        target.execute(
                            insert_postprocess_sql,
                            (candidate_id, detector_run_key, "keep", None, None, None, now),
                        )
                        counts["postprocess_keep"] += 1
                    else:
                        winner_id = inserted_ids.get(suppression.winner.detection_index)
                        if winner_id is None:
                            raise ValueError(
                                f"Duplicate winner missing crop: image={image_id} detection={detection_index}"
                            )
                        target.execute(
                            insert_postprocess_sql,
                            (
                                candidate_id,
                                detector_run_key,
                                "remove",
                                "duplicate",
                                winner_id,
                                suppression.overlap_ratio,
                                now,
                            ),
                        )
                        counts["postprocess_remove_duplicate"] += 1
            finally:
                source_image.close()

            if ordinal % 100 == 0 or ordinal == len(selected_images):
                target.commit()
                print(
                    f"[jp-detector-crops] images={ordinal:,}/{len(selected_images):,} "
                    f"candidates={raw_candidate_count:,} states={dict(counts)}"
                )

        target.commit()
        decision_count = int(target.execute("SELECT COUNT(*) FROM postprocess_decision").fetchone()[0])
        keep_count = int(target.execute("SELECT COUNT(*) FROM postprocess_decision WHERE status='keep'").fetchone()[0])
        remove_count = int(target.execute("SELECT COUNT(*) FROM postprocess_decision WHERE status='remove'").fetchone()[0])
        if decision_count != raw_candidate_count or keep_count + remove_count != raw_candidate_count:
            raise ValueError(
                f"Postprocess coverage mismatch: raw={raw_candidate_count} decision={decision_count} "
                f"keep={keep_count} remove={remove_count}"
            )
        duplicate_group_count = int(
            target.execute(
                "SELECT COUNT(DISTINCT winner_candidate_id) FROM postprocess_decision "
                "WHERE status='remove' AND reason='duplicate'"
            ).fetchone()[0]
        )
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        target.close()
        target = None
        replace_database(temporary, output_database)
    except Exception:
        if target is not None:
            target.close()
        remove_sqlite_files(temporary)
        raise

    summary = {
        "status": "completed",
        "database": str(output_database),
        "source": "jp",
        "split": split,
        "source_image_count": len(images),
        "selected_image_count": len(selected_images),
        "candidate_count": raw_candidate_count,
        "classifier_candidate_count": keep_count,
        "removed_candidate_count": remove_count,
        "duplicate_group_count": duplicate_group_count,
        "counts": dict(sorted(counts.items())),
        "detector_model": str(model_path),
        "detector_run_key": detector_run_key,
        "providers": session.get_providers(),
        "next": {
            "review_database": str(output_database.parent / f"reviews.{detector_run_key}.sqlite"),
            "classifier_audit_database": str(output_database.parent / "classifier_audit.sqlite"),
        },
    }
    summary_path = output_database.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_args(args: argparse.Namespace) -> None:
    for name in ("confidence_threshold", "nms_iou_threshold", "duplicate_overlap_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")
    if int(args.max_detections) < 1:
        raise ValueError("--max-detections must be positive")
    if int(args.image_limit) < 0:
        raise ValueError("--image-limit must be non-negative")


def select_images(
    images: Sequence[dict[str, Any]],
    *,
    split: str,
    seed: int,
    image_limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        images,
        key=lambda image: hashlib.sha256(
            f"{seed}\0{split}\0{image['id']}\0{image.get('file_name', '')}".encode("utf-8")
        ).digest(),
    )
    if image_limit:
        ordered = ordered[:image_limit]
    return ordered


def build_ground_truths(
    annotations: Sequence[dict[str, Any]],
    *,
    category_names: dict[int, str],
) -> list[GroundTruth]:
    result: list[GroundTruth] = []
    for annotation in annotations:
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid COCO bbox: annotation={annotation.get('id')}")
        x, y, width, height = (float(value) for value in bbox)
        if width <= 0.0 or height <= 0.0 or not all(math.isfinite(v) for v in (x, y, width, height)):
            raise ValueError(f"Invalid COCO bbox values: annotation={annotation.get('id')} bbox={bbox}")
        category_id = int(annotation["category_id"])
        raw_label = category_names.get(category_id)
        if raw_label is None:
            raise ValueError(f"Unknown category id {category_id}")
        normalized = normalize_jp_tile_label(raw_label)
        label = raw_label if normalized is None else RED_FIVE_TO_BASE.get(normalized, normalized)
        result.append(
            GroundTruth(
                box_id=str(annotation["id"]),
                label=label,
                source_label=raw_label if normalized is None else normalized,
                center_x=x + width / 2.0,
                center_y=y + height / 2.0,
                width=width,
                height=height,
                angle_deg=0.0,
            )
        )
    return result


def resolve_image_path(image_root: Path, image_record: dict[str, Any]) -> Path:
    value = str(image_record["file_name"])
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe COCO file_name: {value}")
    path = image_root.joinpath(*pure.parts).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def scale_detection_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    source_width: int,
    source_height: int,
) -> tuple[float, float, float, float]:
    scale_x = float(source_width) / float(INPUT_SIZE)
    scale_y = float(source_height) / float(INPUT_SIZE)
    return (
        float(x1) * scale_x,
        float(y1) * scale_y,
        (float(x2) - float(x1)) * scale_x,
        (float(y2) - float(y1)) * scale_y,
    )


def crop_rect(image: Image.Image, rect: Sequence[float]) -> Image.Image | None:
    x, y, width, height = (float(value) for value in rect)
    left = max(0, math.floor(x))
    top = max(0, math.floor(y))
    right = min(image.width, math.ceil(x + width))
    bottom = min(image.height, math.ceil(y + height))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def temporary_database_path(output_database: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output_database.name}.", suffix=".tmp.sqlite", dir=output_database.parent
    )
    os.close(descriptor)
    os.unlink(name)
    return Path(name)


def remove_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def replace_database(source: Path, destination: Path) -> None:
    remove_sqlite_files(destination)
    os.replace(source, destination)


if __name__ == "__main__":
    main()
