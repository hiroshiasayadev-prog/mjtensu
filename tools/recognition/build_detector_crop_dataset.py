from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from PIL import Image

if __package__:
    from .build_tile_crop_dataset import assign_manual_boxes
    from .detector_duplicate_groups import DetectorCandidate, build_duplicate_plan
    from .nanodet.evaluate_composite_onnx import decode_output, preprocess_image
    from .nanodet.refresh_unannotated_capture_detections import region_at, validate_layout
else:  # direct script execution
    from build_tile_crop_dataset import assign_manual_boxes
    from detector_duplicate_groups import DetectorCandidate, build_duplicate_plan
    from nanodet.evaluate_composite_onnx import decode_output, preprocess_image
    from nanodet.refresh_unannotated_capture_detections import region_at, validate_layout


BUILDER_VERSION = "3"
REGIONS = ("completed_hand", "dora_indicators", "melds")
REGION_PATH_COLUMNS = {
    "completed_hand": "hand_crop_path",
    "dora_indicators": "dora_crop_path",
    "melds": "meld_crop_path",
}
RED_FIVE_TO_BASE = {"red5m": "5m", "red5p": "5p", "red5s": "5s"}

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE dataset_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE candidate (
    candidate_id               TEXT PRIMARY KEY,
    source                     TEXT NOT NULL CHECK (source = 'manual'),
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


@dataclass(frozen=True)
class GroundTruth:
    box_id: str
    label: str
    source_label: str
    center_x: float
    center_y: float
    width: float
    height: float
    angle_deg: float

    def json_value(self) -> dict[str, Any]:
        return {
            "id": self.box_id,
            "label": self.label,
            "sourceLabel": self.source_label,
            "centerX": self.center_x,
            "centerY": self.center_y,
            "width": self.width,
            "height": self.height,
            "angleDeg": self.angle_deg,
        }


@dataclass(frozen=True)
class Match:
    ground_truth: GroundTruth
    intersection_area: float
    iou: float
    gt_coverage: float
    detection_coverage: float


@dataclass(frozen=True)
class PreparedDetection:
    detection_index: int
    confidence: float
    region: str
    local_rect: tuple[float, float, float, float]


@dataclass(frozen=True)
class DuplicateSuppression:
    winner: PreparedDetection
    removed: PreparedDetection
    overlap_ratio: float


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run the current NanoDet model over completed manual captures and build an "
            "immutable detector-crop review dataset. Human annotation is used only to "
            "provide geometry/label suggestions; no candidate becomes training truth here."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--capture-database",
        type=Path,
        help="Defaults to .local/recognition/capture_dataset/dataset.sqlite.",
    )
    parser.add_argument(
        "--capture-storage-root",
        type=Path,
        help="Defaults to .local/recognition/capture_dataset.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help=(
            "NanoDet ONNX model. Defaults to the real-capture ft10/l10 model currently "
            "used by the capture refresh tooling."
        ),
    )
    parser.add_argument(
        "--output-database",
        type=Path,
        help="Defaults to .local/recognition/detector_crop_dataset/dataset.sqlite.",
    )
    parser.add_argument(
        "--campaigns",
        nargs="+",
        help="Optional campaign IDs. By default every capture with complete annotation is used.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.60)
    parser.add_argument(
        "--duplicate-overlap-threshold",
        type=float,
        default=0.80,
        help=(
            "After normal NanoDet NMS, suppress a lower-confidence bbox when intersection / "
            "min(areaA, areaB) reaches this ratio within the same capture region."
        ),
    )
    parser.add_argument("--max-detections", type=int, default=200)
    parser.add_argument(
        "--provider",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ONNX Runtime execution provider. auto prefers CUDA when available.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional capture limit for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    repository_root = args.repository_root.resolve()
    storage_root = (
        args.capture_storage_root.resolve()
        if args.capture_storage_root is not None
        else repository_root / ".local" / "recognition" / "capture_dataset"
    )
    capture_database = (
        args.capture_database.resolve()
        if args.capture_database is not None
        else storage_root / "dataset.sqlite"
    )
    model_path = (
        args.model.resolve()
        if args.model is not None
        else repository_root
        / ".local"
        / "recognition"
        / "nanodet_runs"
        / "E1_plus_m_320_real_capture_ft10_l10_seed42"
        / "model_best"
        / "nanodet-plus-m-320-real-capture-ft10-l10.onnx"
    )
    output_database = (
        args.output_database.resolve()
        if args.output_database is not None
        else repository_root
        / ".local"
        / "recognition"
        / "detector_crop_dataset"
        / "dataset.sqlite"
    )
    for path in (capture_database, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_database.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_database}. Use --force to rebuild it.")
    if args.force:
        assert_sqlite_replaceable(output_database)

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required to build detector crops") from error

    available = set(ort.get_available_providers())
    if args.provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider was requested but is unavailable")
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
    model_sha256 = sha256_file(model_path)
    detector_run_key = build_detector_run_key(
        model_sha256=model_sha256,
        confidence_threshold=float(args.confidence_threshold),
        nms_iou_threshold=float(args.nms_iou_threshold),
        duplicate_overlap_threshold=float(args.duplicate_overlap_threshold),
        max_detections=int(args.max_detections),
    )

    captures = load_completed_captures(
        capture_database,
        campaigns=None if args.campaigns is None else tuple(str(value) for value in args.campaigns),
        limit=int(args.limit),
    )
    if not captures:
        raise ValueError("No completed manual annotations matched the requested selection")

    output_database.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_database_path(output_database)
    counts: Counter[str] = Counter()
    candidate_count = 0
    target: sqlite3.Connection | None = None
    preserve_temporary = False
    try:
        target = sqlite3.connect(temporary, timeout=60)
        target.row_factory = sqlite3.Row
        target.execute("PRAGMA journal_mode = WAL")
        target.execute("PRAGMA synchronous = NORMAL")
        target.executescript(SCHEMA)
        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "schema_version": "3",
            "builder_version": BUILDER_VERSION,
            "capture_database": str(capture_database),
            "capture_storage_root": str(storage_root),
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
            "suggestion_policy": (
                "single_gt:best_gt_coverage>=0.70,best_detection_coverage>=0.60,substantial_gt_count=1;"
                "multi_gt:substantial_gt_count>=2 where substantial means gt_coverage>=0.30;"
                "background:best_gt_coverage<0.10;otherwise partial"
            ),
            "duplicate_suppression_policy": (
                "same capture+region; overlap=intersection/min(areaA,areaB); build connected "
                "components from overlap edges; keep one max-detector-confidence bbox per component; "
                f"edge threshold={float(args.duplicate_overlap_threshold):.6g}"
            ),
        }
        target.executemany(
            "INSERT INTO dataset_metadata(key, value) VALUES (?, ?)", metadata.items()
        )

        layout_cache: dict[str, dict[str, Any]] = {}
        insert_sql = """
            INSERT INTO candidate(
                candidate_id, source, capture_id, campaign_id, layout_id, layout_ordinal,
                brightness, shadow, region, source_region_path, source_composite_path,
                detector_model_name, detector_model_sha256, detector_run_key, detection_index,
                detection_confidence, bbox_x, bbox_y, bbox_width, bbox_height,
                crop_width, crop_height, image_png,
                suggested_state, suggested_label, best_gt_id, best_gt_label,
                best_iou, best_gt_coverage, best_detection_coverage, substantial_gt_count,
                gt_json, created_at
            ) VALUES (
                ?, 'manual', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """
        insert_postprocess_sql = """
            INSERT INTO postprocess_decision(
                candidate_id, detector_run_key, status, reason,
                winner_candidate_id, overlap_ratio, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        for ordinal, capture in enumerate(captures, start=1):
            campaign_id = str(capture["campaign_id"])
            layout = layout_for_campaign(repository_root, campaign_id, layout_cache)
            manifest = json.loads(str(capture["manifest_json"]))
            task = json.loads(str(capture["task_json"]))
            annotation = json.loads(str(capture["annotation_json"]))
            gt_by_region = ground_truth_by_region(task, annotation)
            enabled = {
                region: bool(manifest["regionRects"][region]["enabled"])
                for region in REGIONS
            }
            composite_path = resolve_storage_path(storage_root, str(capture["composite_path"]))
            tensor, source_composite = preprocess_image(composite_path)
            try:
                raw_output = session.run([output_name], {input_name: tensor})[0]
            finally:
                source_composite.close()
            detections = decode_output(
                raw_output,
                confidence_threshold=float(args.confidence_threshold),
                nms_iou_threshold=float(args.nms_iou_threshold),
                max_detections=int(args.max_detections),
            )

            prepared: list[PreparedDetection] = []
            region_paths: dict[str, Path] = {}
            for detection_index, detection in enumerate(detections):
                center_x = (float(detection.box.x1) + float(detection.box.x2)) / 2.0
                center_y = (float(detection.box.y1) + float(detection.box.y2)) / 2.0
                region = region_at(center_x, center_y, enabled, layout)
                if region == "invalid":
                    counts["outside_enabled_region"] += 1
                    continue
                source_path_value = capture[REGION_PATH_COLUMNS[region]]
                if source_path_value is None:
                    counts["missing_region_asset"] += 1
                    continue
                if region not in region_paths:
                    region_paths[region] = resolve_storage_path(storage_root, str(source_path_value))
                local = composite_detection_to_region_local(
                    detection.box.x1,
                    detection.box.y1,
                    detection.box.x2,
                    detection.box.y2,
                    destination=layout["regions"][region]["destination"],
                    source_rect=manifest["regionRects"][region]["pixel"],
                )
                if local[2] <= 0.0 or local[3] <= 0.0:
                    counts["empty_crop"] += 1
                    continue
                prepared.append(
                    PreparedDetection(
                        detection_index=detection_index,
                        confidence=float(detection.score),
                        region=region,
                        local_rect=local,
                    )
                )

            _kept, suppressions = suppress_near_duplicate_detections(
                prepared,
                threshold=float(args.duplicate_overlap_threshold),
            )
            suppression_by_removed_index = {
                suppression.removed.detection_index: suppression for suppression in suppressions
            }
            counts["duplicate_suppressed"] += len(suppressions)

            inserted_candidate_ids: dict[int, str] = {}
            region_images: dict[str, Image.Image] = {}
            try:
                # candidate is the immutable raw detector-crop table. Keep/remove is recorded
                # separately in postprocess_decision so removed detections remain auditable.
                for prepared_detection in prepared:
                    detection_index = prepared_detection.detection_index
                    region = prepared_detection.region
                    local = prepared_detection.local_rect
                    if region not in region_images:
                        region_images[region] = Image.open(region_paths[region]).convert("RGB")
                    crop = crop_local_rect(region_images[region], local)
                    if crop is None:
                        counts["empty_crop"] += 1
                        continue
                    matches = match_detection_to_ground_truth(local, gt_by_region[region])
                    suggestion = suggest_state(matches)
                    best = matches[0] if matches else None
                    suggested_label = (
                        best.ground_truth.label if suggestion == "single_gt" and best is not None else None
                    )
                    substantial_count = sum(match.gt_coverage >= 0.30 for match in matches)
                    candidate_id = (
                        f"manual:{detector_run_key}:{capture['capture_id']}:{detection_index}"
                    )
                    target.execute(
                        insert_sql,
                        (
                            candidate_id,
                            str(capture["capture_id"]),
                            campaign_id,
                            str(capture["layout_id"]),
                            int(capture["layout_ordinal"]),
                            str(capture["brightness"]),
                            str(capture["shadow"]),
                            region,
                            repository_relative_or_absolute(region_paths[region], repository_root),
                            repository_relative_or_absolute(composite_path, repository_root),
                            model_path.name,
                            model_sha256,
                            detector_run_key,
                            detection_index,
                            prepared_detection.confidence,
                            local[0],
                            local[1],
                            local[2],
                            local[3],
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
                            json.dumps(
                                [ground_truth.json_value() for ground_truth in gt_by_region[region]],
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            now,
                        ),
                    )
                    crop.close()
                    inserted_candidate_ids[detection_index] = candidate_id
                    candidate_count += 1
                    counts[f"raw_{suggestion}"] += 1
            finally:
                for image in region_images.values():
                    image.close()

            for detection_index, candidate_id in inserted_candidate_ids.items():
                suppression = suppression_by_removed_index.get(detection_index)
                if suppression is None:
                    target.execute(
                        insert_postprocess_sql,
                        (candidate_id, detector_run_key, "keep", None, None, None, now),
                    )
                    counts["postprocess_keep"] += 1
                    continue
                winner_candidate_id = inserted_candidate_ids.get(
                    suppression.winner.detection_index
                )
                if winner_candidate_id is None:
                    raise ValueError(
                        "Duplicate suppression winner has no candidate crop: "
                        f"capture={capture['capture_id']} "
                        f"winner_detection_index={suppression.winner.detection_index} "
                        f"removed_detection_index={detection_index}"
                    )
                target.execute(
                    insert_postprocess_sql,
                    (
                        candidate_id,
                        detector_run_key,
                        "remove",
                        "duplicate",
                        winner_candidate_id,
                        suppression.overlap_ratio,
                        now,
                    ),
                )
                counts["postprocess_remove_duplicate"] += 1

            if ordinal % 10 == 0 or ordinal == len(captures):
                target.commit()
                print(
                    f"[detector-crops] captures={ordinal}/{len(captures)} "
                    f"candidates={candidate_count} states={dict(counts)}"
                )

        target.commit()
        raw_candidate_count = int(target.execute("SELECT COUNT(*) FROM candidate").fetchone()[0])
        decision_count = int(
            target.execute("SELECT COUNT(*) FROM postprocess_decision").fetchone()[0]
        )
        keep_count = int(
            target.execute(
                "SELECT COUNT(*) FROM postprocess_decision WHERE status='keep'"
            ).fetchone()[0]
        )
        remove_count = int(
            target.execute(
                "SELECT COUNT(*) FROM postprocess_decision WHERE status='remove'"
            ).fetchone()[0]
        )
        duplicate_group_count = int(
            target.execute(
                """
                SELECT COUNT(DISTINCT winner_candidate_id)
                FROM postprocess_decision
                WHERE status='remove' AND reason='duplicate'
                """
            ).fetchone()[0]
        )
        bad_winner_count = int(
            target.execute(
                """
                SELECT COUNT(*)
                FROM postprocess_decision AS removed
                JOIN postprocess_decision AS winner
                  ON winner.candidate_id = removed.winner_candidate_id
                WHERE removed.status='remove' AND winner.status!='keep'
                """
            ).fetchone()[0]
        )
        if raw_candidate_count != decision_count or keep_count + remove_count != raw_candidate_count:
            raise ValueError(
                "Postprocess decision coverage mismatch: "
                f"candidate={raw_candidate_count} decision={decision_count} "
                f"keep={keep_count} remove={remove_count}"
            )
        if bad_winner_count:
            raise ValueError(
                f"Postprocess invariant violated: {bad_winner_count} removed candidates point "
                "to a winner that is not kept"
            )
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        target.close()
        target = None
        try:
            replace_database(temporary, output_database)
        except PermissionError as error:
            pending_database = output_database.with_name(
                f"{output_database.stem}.pending{output_database.suffix}"
            )
            remove_sqlite_files(pending_database)
            os.replace(temporary, pending_database)
            preserve_temporary = True
            raise PermissionError(
                "Detector crop generation completed, but the existing output database is "
                "locked by another process and could not be replaced. The completed new "
                f"database was preserved at: {pending_database}. Close the process holding "
                f"{output_database}, then replace it with the pending database instead of "
                "rerunning NanoDet."
            ) from error
    except Exception:
        if target is not None:
            target.close()
        if not preserve_temporary:
            remove_sqlite_files(temporary)
        raise

    summary = {
        "status": "completed",
        "database": str(output_database),
        "capture_count": len(captures),
        "candidate_count": raw_candidate_count,
        "classifier_candidate_count": keep_count,
        "removed_candidate_count": remove_count,
        "duplicate_group_count": duplicate_group_count,
        "counts": dict(sorted(counts.items())),
        "detector_model": str(model_path),
        "detector_model_sha256": model_sha256,
        "detector_run_key": detector_run_key,
        "providers": session.get_providers(),
        "next": {
            "review_database": str(
                output_database.parent / f"reviews.{detector_run_key}.sqlite"
            ),
            "classifier_audit_database": str(output_database.parent / "classifier_audit.sqlite"),
        },
    }
    summary_path = output_database.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= float(args.confidence_threshold) <= 1.0:
        raise ValueError("--confidence-threshold must be in [0,1]")
    if not 0.0 <= float(args.nms_iou_threshold) <= 1.0:
        raise ValueError("--nms-iou-threshold must be in [0,1]")
    if not 0.0 <= float(args.duplicate_overlap_threshold) <= 1.0:
        raise ValueError("--duplicate-overlap-threshold must be in [0,1]")
    if int(args.max_detections) < 1:
        raise ValueError("--max-detections must be positive")
    if int(args.limit) < 0:
        raise ValueError("--limit must be non-negative")


def load_completed_captures(
    database: Path,
    *,
    campaigns: Sequence[str] | None,
    limit: int,
) -> list[sqlite3.Row]:
    sql = """
        SELECT
            capture.id AS capture_id,
            capture.composite_path,
            capture.hand_crop_path,
            capture.dora_crop_path,
            capture.meld_crop_path,
            capture.manifest_json,
            capture_task.campaign_id,
            capture_task.layout_id,
            capture_task.layout_ordinal,
            capture_task.brightness,
            capture_task.shadow,
            capture_task.task_order,
            capture_task.task_json,
            capture_annotation.annotation_json
        FROM capture
        JOIN capture_task ON capture_task.id = capture.task_id
        JOIN capture_annotation ON capture_annotation.capture_id = capture.id
        WHERE capture_annotation.status = 'complete'
    """
    params: list[Any] = []
    if campaigns:
        placeholders = ",".join("?" for _ in campaigns)
        sql += f" AND capture_task.campaign_id IN ({placeholders})"
        params.extend(campaigns)
    sql += " ORDER BY capture_task.task_order, capture.id"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    with closing(
        sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=60)
    ) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(sql, params).fetchall()


def ground_truth_by_region(
    task: dict[str, Any], annotation: dict[str, Any]
) -> dict[str, list[GroundTruth]]:
    boxes = annotation.get("boxes")
    if not isinstance(boxes, dict):
        raise ValueError("Manual annotation document has no boxes object")
    result: dict[str, list[GroundTruth]] = {region: [] for region in REGIONS}
    for region in REGIONS:
        region_boxes = boxes.get(region)
        if not isinstance(region_boxes, list):
            raise ValueError(f"Manual annotation boxes.{region} is not an array")
        assigned = assign_manual_boxes(task, region, region_boxes)
        for item in assigned:
            box = item.box
            source_label = str(item.slot.tile_label)
            result[region].append(
                GroundTruth(
                    box_id=str(box["id"]),
                    label=RED_FIVE_TO_BASE.get(source_label, source_label),
                    source_label=source_label,
                    center_x=float(box["centerX"]),
                    center_y=float(box["centerY"]),
                    width=float(box["width"]),
                    height=float(box["height"]),
                    angle_deg=float(box["angleDeg"]),
                )
            )
    return result


def layout_for_campaign(
    repository_root: Path,
    campaign_id: str,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = "tile_catalog" if campaign_id.startswith("tile-catalog") else "capture"
    if key in cache:
        return cache[key]
    name = "tile_catalog_layout.v2.json" if key == "tile_catalog" else "capture_layout.v1.json"
    path = repository_root / "tools" / "recognition" / name
    with path.open("r", encoding="utf-8") as source:
        layout = json.load(source)
    if not isinstance(layout, dict):
        raise ValueError(f"Layout root must be an object: {path}")
    validate_layout(layout)
    cache[key] = layout
    return layout


def composite_detection_to_region_local(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    destination: dict[str, Any],
    source_rect: dict[str, Any],
) -> tuple[float, float, float, float]:
    dest_x = float(destination["x"])
    dest_y = float(destination["y"])
    dest_w = float(destination["width"])
    dest_h = float(destination["height"])
    left = max(float(x1), dest_x)
    top = max(float(y1), dest_y)
    right = min(float(x2), dest_x + dest_w)
    bottom = min(float(y2), dest_y + dest_h)
    if right <= left or bottom <= top:
        return (0.0, 0.0, 0.0, 0.0)
    scale_x = float(source_rect["width"]) / dest_w
    scale_y = float(source_rect["height"]) / dest_h
    return (
        (left - dest_x) * scale_x,
        (top - dest_y) * scale_y,
        (right - left) * scale_x,
        (bottom - top) * scale_y,
    )


def crop_local_rect(
    image: Image.Image, rect: Sequence[float]
) -> Image.Image | None:
    x, y, width, height = (float(value) for value in rect)
    left = max(0, math.floor(x))
    top = max(0, math.floor(y))
    right = min(image.width, math.ceil(x + width))
    bottom = min(image.height, math.ceil(y + height))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def rect_min_area_overlap(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Intersection area divided by the smaller rectangle area."""
    ax, ay, aw, ah = (float(value) for value in left)
    bx, by, bw, bh = (float(value) for value in right)
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    minimum_area = min(area_a, area_b)
    if minimum_area <= 0.0:
        return 0.0
    intersection_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return (intersection_width * intersection_height) / minimum_area


def suppress_near_duplicate_detections(
    detections: Sequence[PreparedDetection],
    *,
    threshold: float,
) -> tuple[list[PreparedDetection], list[DuplicateSuppression]]:
    """Compatibility wrapper around the shared duplicate-cluster implementation."""
    by_id = {
        f"{item.region}:{item.detection_index}": item
        for item in detections
    }
    if len(by_id) != len(detections):
        raise ValueError("Duplicate detection index within one region")
    plan = build_duplicate_plan(
        (
            DetectorCandidate(
                candidate_id=candidate_id,
                capture_id="current-capture",
                region=item.region,
                detection_index=item.detection_index,
                confidence=item.confidence,
                bbox_x=item.local_rect[0],
                bbox_y=item.local_rect[1],
                bbox_width=item.local_rect[2],
                bbox_height=item.local_rect[3],
            )
            for candidate_id, item in by_id.items()
        ),
        threshold=threshold,
    )
    kept = [by_id[candidate_id] for candidate_id in plan.winner_candidate_ids]
    suppressions: list[DuplicateSuppression] = []
    for cluster in plan.clusters:
        winner = by_id[cluster.winner.candidate_id]
        for loser in cluster.losers:
            suppressions.append(
                DuplicateSuppression(
                    winner=winner,
                    removed=by_id[loser.candidate.candidate_id],
                    overlap_ratio=loser.max_overlap_to_cluster,
                )
            )
    kept.sort(key=lambda item: item.detection_index)
    suppressions.sort(key=lambda item: item.removed.detection_index)
    return kept, suppressions


def match_detection_to_ground_truth(
    rect: Sequence[float],
    ground_truths: Sequence[GroundTruth],
) -> list[Match]:
    x, y, width, height = (float(value) for value in rect)
    detection_area = max(0.0, width) * max(0.0, height)
    if detection_area <= 0.0:
        return []
    matches: list[Match] = []
    for ground_truth in ground_truths:
        polygon = rotated_rectangle_polygon(ground_truth)
        gt_area = polygon_area(polygon)
        clipped = clip_polygon_to_rect(polygon, x, y, x + width, y + height)
        intersection = polygon_area(clipped)
        union = detection_area + gt_area - intersection
        matches.append(
            Match(
                ground_truth=ground_truth,
                intersection_area=intersection,
                iou=0.0 if union <= 0.0 else intersection / union,
                gt_coverage=0.0 if gt_area <= 0.0 else intersection / gt_area,
                detection_coverage=intersection / detection_area,
            )
        )
    matches.sort(
        key=lambda item: (item.gt_coverage, item.detection_coverage, item.iou),
        reverse=True,
    )
    return matches


def suggest_state(matches: Sequence[Match]) -> str:
    if not matches or matches[0].gt_coverage < 0.10:
        return "background"
    substantial_count = sum(match.gt_coverage >= 0.30 for match in matches)
    if substantial_count >= 2:
        return "multi_gt"
    best = matches[0]
    if (
        best.gt_coverage >= 0.70
        and best.detection_coverage >= 0.60
        and substantial_count == 1
    ):
        return "single_gt"
    return "partial"


def rotated_rectangle_polygon(box: GroundTruth) -> list[tuple[float, float]]:
    half_width = box.width / 2.0
    half_height = box.height / 2.0
    radians = math.radians(box.angle_deg)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    result: list[tuple[float, float]] = []
    for local_x, local_y in (
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ):
        result.append(
            (
                box.center_x + cosine * local_x - sine * local_y,
                box.center_y + sine * local_x + cosine * local_y,
            )
        )
    return result


def clip_polygon_to_rect(
    polygon: Sequence[tuple[float, float]],
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> list[tuple[float, float]]:
    result = list(polygon)
    result = clip_polygon(result, lambda p: p[0] >= left, lambda a, b: intersect_vertical(a, b, left))
    result = clip_polygon(result, lambda p: p[0] <= right, lambda a, b: intersect_vertical(a, b, right))
    result = clip_polygon(result, lambda p: p[1] >= top, lambda a, b: intersect_horizontal(a, b, top))
    result = clip_polygon(result, lambda p: p[1] <= bottom, lambda a, b: intersect_horizontal(a, b, bottom))
    return result


def clip_polygon(
    polygon: Sequence[tuple[float, float]],
    inside: Any,
    intersection: Any,
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    output: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = bool(inside(previous))
    for current in polygon:
        current_inside = bool(inside(current))
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def intersect_vertical(
    left: tuple[float, float], right: tuple[float, float], x: float
) -> tuple[float, float]:
    delta_x = right[0] - left[0]
    if abs(delta_x) < 1.0e-12:
        return (x, left[1])
    ratio = (x - left[0]) / delta_x
    return (x, left[1] + ratio * (right[1] - left[1]))


def intersect_horizontal(
    left: tuple[float, float], right: tuple[float, float], y: float
) -> tuple[float, float]:
    delta_y = right[1] - left[1]
    if abs(delta_y) < 1.0e-12:
        return (left[0], y)
    ratio = (y - left[1]) / delta_y
    return (left[0] + ratio * (right[0] - left[0]), y)


def polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    ) / 2.0


def encode_png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=1, optimize=False)
    return output.getvalue()


def resolve_storage_path(storage_root: Path, value: str) -> Path:
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe storage-relative path: {value}")
    path = storage_root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(storage_root.resolve())
    except ValueError as error:
        raise ValueError(f"Storage path escapes root: {value}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def repository_relative_or_absolute(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_detector_run_key(
    *,
    model_sha256: str,
    confidence_threshold: float,
    nms_iou_threshold: float,
    duplicate_overlap_threshold: float,
    max_detections: int,
) -> str:
    payload = (
        f"{model_sha256}|confidence={confidence_threshold:.12g}|"
        f"nms={nms_iou_threshold:.12g}|duplicate_overlap={duplicate_overlap_threshold:.12g}|"
        f"max={max_detections}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def assert_sqlite_replaceable(path: Path) -> None:
    """Fail before inference when Windows has an existing output SQLite file open."""
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if not candidate.exists():
            continue
        probe = candidate.with_name(f".{candidate.name}.replace-probe-{os.getpid()}")
        try:
            os.replace(candidate, probe)
        except PermissionError as error:
            raise PermissionError(
                f"Cannot replace {candidate}: another process has it open. Close the "
                "review server, SQLite viewer, or other process using the detector crop "
                "dataset before running with --force."
            ) from error
        try:
            os.replace(probe, candidate)
        except Exception:
            # Best effort restoration if the round-trip itself is interrupted.
            if probe.exists() and not candidate.exists():
                os.replace(probe, candidate)
            raise


def replace_database(source: Path, destination: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(f"{destination}{suffix}").unlink()
        except FileNotFoundError:
            pass
    os.replace(source, destination)


if __name__ == "__main__":
    main()
