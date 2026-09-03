from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

if __package__:
    from .build_rotated_detector_corpus import (
        REGION_KEYS,
        anisotropic_box_to_approx_obb,
        annotation_geometry_delta,
        canonicalize_obb,
        load_annotation_snapshot,
        parse_annotation_document,
    )
    from .rotated_box_rectifier import (
        CropWindow,
        RotatedBoxRectifier,
        decode_prediction,
        default_crop_window,
    )
else:
    repository_root_for_import = Path(__file__).resolve().parents[2]
    if str(repository_root_for_import) not in sys.path:
        sys.path.insert(0, str(repository_root_for_import))
    from tools.recognition.build_rotated_detector_corpus import (  # type: ignore[no-redef]
        REGION_KEYS,
        anisotropic_box_to_approx_obb,
        annotation_geometry_delta,
        canonicalize_obb,
        load_annotation_snapshot,
        parse_annotation_document,
    )
    from tools.recognition.rotated_box_rectifier import (  # type: ignore[no-redef]
        CropWindow,
        RotatedBoxRectifier,
        decode_prediction,
        default_crop_window,
    )


IMAGE_MEAN = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32).view(3, 1, 1)
IMAGE_STD = torch.tensor((0.25, 0.25, 0.25), dtype=torch.float32).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    capture_root = repository_root / ".local" / "recognition" / "capture_dataset"
    parser = argparse.ArgumentParser(
        description=(
            "Apply trained HBB-to-OBB rectifier predictions only to captures that have not "
            "been manually OBB-reviewed since the pre-review SQLite backup. Existing human "
            "OBB edits are preserved and tagged human_reviewed; generated suggestions are draft."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--database", type=Path, default=capture_root / "dataset.sqlite")
    parser.add_argument(
        "--baseline-database",
        type=Path,
        default=capture_root / "dataset.pre-obb-review.sqlite",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "rotated_box_rectifier_runs"
        / "rgb96_seed42"
        / "model_best.pt",
    )
    parser.add_argument("--campaign-id", default="initial-120")
    parser.add_argument("--context-scale", type=float, default=1.40)
    parser.add_argument("--geometry-epsilon", type=float, default=1.0e-4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--backup-path",
        type=Path,
        default=capture_root / "dataset.pre-rectifier-suggestions.sqlite",
    )
    parser.add_argument(
        "--refresh-model-suggestions",
        action="store_true",
        help="Replace existing model_suggested drafts. Human-reviewed captures are never replaced.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run selection and inference but do not update SQLite or create a backup.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    database = args.database.resolve()
    baseline_database = args.baseline_database.resolve()
    checkpoint = args.checkpoint.resolve()
    backup_path = args.backup_path.resolve()
    for path in (database, baseline_database, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if float(args.context_scale) <= 1.0:
        raise ValueError("--context-scale must be > 1")

    device = resolve_device(str(args.device))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    input_size = int(config["input_size"])
    model = RotatedBoxRectifier(input_size=input_size)
    model.load_state_dict(payload["model_state_dict"])
    model.eval().to(device)
    checkpoint_sha256 = sha256_file(checkpoint)

    baseline = load_annotation_snapshot(baseline_database)
    layout_path = repository_root / "tools" / "recognition" / "capture_layout.v1.json"
    with layout_path.open("r", encoding="utf-8") as source:
        layout = json.load(source)
    rows = load_campaign_rows(database, str(args.campaign_id))
    decisions: list[dict[str, Any]] = []
    writes: list[tuple[str, str, dict[str, Any]]] = []
    reviewed_backfills = 0
    suggestion_count = 0
    skipped_count = 0

    for row in rows:
        capture_id = str(row["capture_id"])
        current_document = (
            None
            if row["annotation_json"] is None
            else parse_annotation_document(str(row["annotation_json"]), capture_id)
        )
        current_review = review_state(current_document)
        if current_review == "human_reviewed":
            decisions.append({"capture_id": capture_id, "action": "skip_human_reviewed"})
            skipped_count += 1
            continue
        if current_review == "model_suggested" and not bool(args.refresh_model_suggestions):
            decisions.append({"capture_id": capture_id, "action": "skip_existing_model_suggestion"})
            skipped_count += 1
            continue

        baseline_row = baseline.get(capture_id)
        if (
            current_review != "model_suggested"
            and current_document is not None
            and baseline_row is not None
        ):
            baseline_document = parse_annotation_document(
                str(baseline_row["annotation_json"]), capture_id
            )
            changed_count, max_delta = annotation_geometry_delta(
                current_document,
                baseline_document,
                epsilon=float(args.geometry_epsilon),
            )
            if changed_count > 0:
                tagged = dict(current_document)
                tagged["review"] = {
                    "state": "human_reviewed",
                    "source": "pre_obb_backup_geometry_diff",
                    "reviewedAt": str(row["annotation_updated_at"] or now_iso()),
                    "changedBoxCount": changed_count,
                    "maxGeometryDelta": max_delta,
                }
                writes.append((capture_id, str(row["annotation_status"]), tagged))
                decisions.append(
                    {
                        "capture_id": capture_id,
                        "action": "backfill_human_reviewed",
                        "changed_box_count": changed_count,
                    }
                )
                reviewed_backfills += 1
                continue

        source_document, input_source = select_source_document(
            row,
            baseline_row=baseline_row,
        )
        if source_document is None:
            decisions.append({"capture_id": capture_id, "action": "skip_no_safe_hbb_source"})
            skipped_count += 1
            continue

        try:
            predicted_boxes = predict_document_boxes(
                source_document["boxes"],
                composite_path=database.parent.joinpath(*str(row["composite_path"]).replace("\\", "/").split("/")),
                manifest=json.loads(str(row["manifest_json"])),
                layout=layout,
                model=model,
                device=device,
                input_size=input_size,
                context_scale=float(args.context_scale),
            )
        except (FileNotFoundError, ValueError) as error:
            decisions.append(
                {
                    "capture_id": capture_id,
                    "action": "skip_inference_input_error",
                    "error": str(error),
                }
            )
            skipped_count += 1
            continue

        generated_at = now_iso()
        suggested_document = {
            "schemaVersion": 1,
            "captureId": capture_id,
            "boxes": predicted_boxes,
            "review": {
                "state": "model_suggested",
                "source": "rotated_box_rectifier",
                "generatedAt": generated_at,
                "modelSha256": checkpoint_sha256,
                "modelCheckpoint": repository_relative_or_absolute(repository_root, checkpoint),
                "checkpointEpoch": int(payload.get("epoch", -1)),
                "inputSource": input_source,
                "contextScale": float(args.context_scale),
                "inputSize": input_size,
            },
        }
        writes.append((capture_id, "draft", suggested_document))
        decisions.append(
            {
                "capture_id": capture_id,
                "action": "write_model_suggestion",
                "input_source": input_source,
                "box_count": sum(len(predicted_boxes[key]) for key in REGION_KEYS),
            }
        )
        suggestion_count += 1

    if not bool(args.dry_run):
        create_backup_once(database, backup_path)
        apply_writes(database, writes)

    result = {
        "status": "dry_run" if bool(args.dry_run) else "completed",
        "campaign_id": str(args.campaign_id),
        "database": str(database),
        "baseline_database": str(baseline_database),
        "backup": None if bool(args.dry_run) else str(backup_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "device": str(device),
        "capture_count": len(rows),
        "human_reviewed_backfilled": reviewed_backfills,
        "model_suggestions_written": suggestion_count,
        "skipped": skipped_count,
        "decisions": decisions,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def load_campaign_rows(database: Path, campaign_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                capture.id AS capture_id,
                capture.composite_path,
                capture.hand_crop_path,
                capture.dora_crop_path,
                capture.meld_crop_path,
                capture.manifest_json,
                capture_task.task_json,
                capture_task.task_order,
                capture_task.layout_id,
                capture_task.layout_ordinal,
                capture_annotation.status AS annotation_status,
                capture_annotation.annotation_json,
                capture_annotation.updated_at AS annotation_updated_at
            FROM capture
            JOIN capture_task ON capture_task.id = capture.task_id
            LEFT JOIN capture_annotation ON capture_annotation.capture_id = capture.id
            WHERE capture_task.campaign_id = ?
            ORDER BY capture_task.task_order, capture.id
            """,
            (campaign_id,),
        ).fetchall()
        detections = connection.execute(
            """
            SELECT
                detection.capture_id,
                detection.detection_index,
                detection.region,
                detection.original_x,
                detection.original_y,
                detection.original_width,
                detection.original_height
            FROM detection
            JOIN capture ON capture.id = detection.capture_id
            JOIN capture_task ON capture_task.id = capture.task_id
            WHERE capture_task.campaign_id = ?
            ORDER BY detection.capture_id, detection.detection_index
            """,
            (campaign_id,),
        ).fetchall()
    detections_by_capture: dict[str, list[dict[str, Any]]] = {}
    for detection in detections:
        detections_by_capture.setdefault(str(detection["capture_id"]), []).append(dict(detection))
    result = []
    for row in rows:
        item = dict(row)
        item["detections"] = detections_by_capture.get(str(row["capture_id"]), [])
        result.append(item)
    return result


def select_source_document(
    row: dict[str, Any],
    *,
    baseline_row: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    capture_id = str(row["capture_id"])
    current_document = (
        None
        if row["annotation_json"] is None
        else parse_annotation_document(str(row["annotation_json"]), capture_id)
    )
    if review_state(current_document) == "model_suggested" and baseline_row is not None:
        return (
            parse_annotation_document(str(baseline_row["annotation_json"]), capture_id),
            "baseline_annotation_hbb",
        )
    if current_document is not None and baseline_row is not None:
        return current_document, "existing_annotation_hbb"
    if row["annotation_json"] is not None and baseline_row is None:
        # Conservative: without the pre-review baseline we cannot prove that this annotation
        # was not manually changed, so never overwrite it automatically.
        return None, None

    task = json.loads(str(row["task_json"]))
    manifest = json.loads(str(row["manifest_json"]))
    boxes = detector_boxes_if_complete(
        row["detections"],
        task=task,
        manifest=manifest,
        capture_id=capture_id,
    )
    if boxes is None:
        return None, None
    return {"schemaVersion": 1, "captureId": capture_id, "boxes": boxes}, "detector_hbb"


def detector_boxes_if_complete(
    detections: Sequence[dict[str, Any]],
    *,
    task: dict[str, Any],
    manifest: dict[str, Any],
    capture_id: str,
) -> dict[str, list[dict[str, Any]]] | None:
    expected = expected_region_counts(task)
    result: dict[str, list[dict[str, Any]]] = {key: [] for key in REGION_KEYS}
    for detection in detections:
        region = str(detection["region"])
        if region not in result or detection["original_x"] is None:
            continue
        origin = manifest["regionRects"][region]["pixel"]
        x = float(detection["original_x"]) - float(origin["x"])
        y = float(detection["original_y"]) - float(origin["y"])
        width = float(detection["original_width"])
        height = float(detection["original_height"])
        sideways = width > height
        result[region].append(
            {
                "id": f"rectifier-{capture_id}-{int(detection['detection_index'])}",
                "centerX": x + width / 2.0,
                "centerY": y + height / 2.0,
                "width": height if sideways else width,
                "height": width if sideways else height,
                "angleDeg": 90.0 if sideways else 0.0,
            }
        )
    if any(len(result[key]) != expected[key] for key in REGION_KEYS):
        return None
    return result


def expected_region_counts(task: dict[str, Any]) -> dict[str, int]:
    return {
        "completed_hand": sum(1 for slot in task["hand"] if slot.get("face") == "front"),
        "dora_indicators": sum(
            1
            for row_key in ("visible", "ura")
            for slot in task["dora"][row_key]
            if slot.get("face") == "front"
        ),
        "melds": sum(
            1
            for meld in task["melds"]
            for slot in meld["tiles"]
            if slot.get("face") == "front"
        ),
    }


def predict_document_boxes(
    boxes_by_region: dict[str, Any],
    *,
    composite_path: Path,
    manifest: dict[str, Any],
    layout: dict[str, Any],
    model: RotatedBoxRectifier,
    device: torch.device,
    input_size: int,
    context_scale: float,
) -> dict[str, list[dict[str, Any]]]:
    if not composite_path.is_file():
        raise FileNotFoundError(composite_path)
    output: dict[str, list[dict[str, Any]]] = {key: [] for key in REGION_KEYS}
    with Image.open(composite_path) as opened:
        composite = opened.convert("RGB")
    if composite.size != (320, 320):
        raise ValueError(f"Expected 320x320 composite, found {composite.size}: {composite_path}")

    with torch.inference_mode():
        for region in REGION_KEYS:
            source_boxes = boxes_by_region[region]
            if not source_boxes:
                continue
            region_manifest = manifest["regionRects"][region]["pixel"]
            crop_width = max(1, math.floor(float(region_manifest["width"]) + 0.5))
            crop_height = max(1, math.floor(float(region_manifest["height"]) + 0.5))
            destination = layout["regions"][region]["destination"]
            scale_x = float(destination["width"]) / crop_width
            scale_y = float(destination["height"]) / crop_height

            samples: list[torch.Tensor] = []
            windows: list[CropWindow] = []
            for raw_box in source_boxes:
                source_composite_obb = canonicalize_obb(
                    anisotropic_box_to_approx_obb(
                        center_x=float(raw_box["centerX"]),
                        center_y=float(raw_box["centerY"]),
                        width=float(raw_box["width"]),
                        height=float(raw_box["height"]),
                        angle_deg=float(raw_box["angleDeg"]),
                        scale_x=scale_x,
                        scale_y=scale_y,
                        offset_x=float(destination["x"]),
                        offset_y=float(destination["y"]),
                    )
                )
                window = default_crop_window(source_composite_obb, context_scale=context_scale)
                crop = sample_window(composite, window, input_size)
                array = np.asarray(crop, dtype=np.uint8).copy()
                crop.close()
                tensor = torch.from_numpy(array).permute(2, 0, 1).float().mul_(1.0 / 255.0)
                tensor = tensor.sub(IMAGE_MEAN).div(IMAGE_STD)
                samples.append(tensor)
                windows.append(window)

            batch = torch.stack(samples).to(device)
            predictions = model(batch).detach().float().cpu()
            for raw_box, prediction, window in zip(source_boxes, predictions, windows, strict=True):
                composite_obb = decode_prediction(prediction.tolist(), window)
                local_obb = canonicalize_obb(
                    anisotropic_box_to_approx_obb(
                        center_x=float(composite_obb[0]) - float(destination["x"]),
                        center_y=float(composite_obb[1]) - float(destination["y"]),
                        width=float(composite_obb[2]),
                        height=float(composite_obb[3]),
                        angle_deg=float(composite_obb[4]),
                        scale_x=1.0 / scale_x,
                        scale_y=1.0 / scale_y,
                        offset_x=0.0,
                        offset_y=0.0,
                    )
                )
                output[region].append(
                    {
                        "id": str(raw_box["id"]),
                        "centerX": local_obb[0],
                        "centerY": local_obb[1],
                        "width": local_obb[2],
                        "height": local_obb[3],
                        "angleDeg": local_obb[4],
                    }
                )
    composite.close()
    return output


def sample_window(image: Image.Image, window: CropWindow, input_size: int) -> Image.Image:
    return image.transform(
        (input_size, input_size),
        Image.Transform.EXTENT,
        (window.left, window.top, window.right, window.bottom),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def review_state(document: dict[str, Any] | None) -> str | None:
    if document is None:
        return None
    review = document.get("review")
    if not isinstance(review, dict):
        return None
    state = review.get("state")
    return str(state) if state in {"model_suggested", "human_reviewed"} else None


def create_backup_once(database: Path, backup_path: Path) -> None:
    if backup_path.exists():
        return
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)


def apply_writes(database: Path, writes: Sequence[tuple[str, str, dict[str, Any]]]) -> None:
    with sqlite3.connect(database, timeout=30) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        now = now_iso()
        for capture_id, status, document in writes:
            connection.execute(
                """
                INSERT INTO capture_annotation(
                    capture_id, status, schema_version, annotation_json, updated_at
                ) VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(capture_id) DO UPDATE SET
                    status = excluded.status,
                    schema_version = excluded.schema_version,
                    annotation_json = excluded.annotation_json,
                    updated_at = excluded.updated_at
                """,
                (
                    capture_id,
                    status,
                    json.dumps(document, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
        connection.commit()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative_or_absolute(repository_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return str(path)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
