from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

if __package__:
    from .evaluate_composite_onnx import decode_output, preprocess_image
else:  # direct script execution
    from evaluate_composite_onnx import decode_output, preprocess_image

from tools.recognition.capture_dataset_api.database import CaptureDatabase


REGION_KEYS = ("completed_hand", "dora_indicators", "melds")


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Re-run NanoDet over saved 320x320 capture composites. By default only "
            "captures with no annotation row are updated; draft candidate sets can be "
            "included explicitly without changing their saved annotation document."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--campaign-id", default="initial-120")
    parser.add_argument(
        "--from-layout",
        type=int,
        default=1,
        help="First one-based layout number eligible for refresh (default: 1).",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--layout",
        type=Path,
        help=(
            "Detector composite layout JSON. Defaults to tile_catalog_layout.v2.json "
            "for tile-catalog campaigns and capture_layout.v1.json otherwise."
        ),
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.60)
    parser.add_argument("--max-detections", type=int, default=200)
    parser.add_argument(
        "--include-drafts",
        action="store_true",
        help=(
            "Also refresh the underlying detector candidates for draft captures. "
            "The draft annotation document itself is not modified."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference and print the plan without modifying SQLite.",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Do not create a consistent SQLite backup before writing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    storage_root = (
        args.storage_root.resolve()
        if args.storage_root is not None
        else repository_root / ".local" / "recognition" / "capture_dataset"
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
    database_path = storage_root / "dataset.sqlite"
    layout_path = (
        args.layout.resolve()
        if args.layout is not None
        else repository_root
        / "tools"
        / "recognition"
        / (
            "tile_catalog_layout.v2.json"
            if str(args.campaign_id).startswith("tile-catalog")
            else "capture_layout.v1.json"
        )
    )

    validate_arguments(args, database_path=database_path, model_path=model_path, layout_path=layout_path)
    layout = load_json(layout_path)
    validate_layout(layout)

    database = CaptureDatabase(database_path)
    include_drafts = bool(
        args.include_drafts or str(args.campaign_id).startswith("tile-catalog")
    )
    captures = database.unannotated_captures(
        args.campaign_id,
        minimum_layout_ordinal=args.from_layout - 1,
        include_drafts=include_drafts,
    )
    if not captures:
        print(
            json.dumps(
                {
                    "status": "nothing-to-refresh",
                    "campaignId": args.campaign_id,
                    "fromLayout": args.from_layout,
                    "reason": "No captures without an annotation row matched the selection.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required. Install project dependencies before running this command."
        ) from error

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            f"Expected one ONNX input and output, found {len(inputs)} and {len(outputs)}"
        )
    input_name = inputs[0].name
    output_name = outputs[0].name
    model_sha256 = sha256_file(model_path)
    model_name = model_path.name

    prepared: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for ordinal, capture in enumerate(captures, start=1):
        image_path = resolve_storage_path(storage_root, capture["compositePath"])
        tensor, source_image = preprocess_image(image_path)
        try:
            raw_output = session.run([output_name], {input_name: tensor})[0]
        finally:
            source_image.close()
        decoded = decode_output(
            raw_output,
            confidence_threshold=float(args.confidence_threshold),
            nms_iou_threshold=float(args.nms_iou_threshold),
            max_detections=int(args.max_detections),
        )
        mapped = map_detections(decoded, capture["manifest"], layout)
        prepared.append((capture, mapped))
        region_counts = {
            region: sum(1 for detection in mapped if detection["region"] == region)
            for region in (*REGION_KEYS, "invalid")
        }
        print(
            f"[{ordinal:02d}/{len(captures):02d}] "
            f"layout={capture['layoutOrdinal'] + 1:02d} "
            f"{capture['environment']['brightness']}/{capture['environment']['shadow']} "
            f"detections={len(mapped)} regions={region_counts}"
        )

    backup_path: Path | None = None
    if not args.dry_run and not args.skip_backup:
        backup_path = create_database_backup(database_path, storage_root / "backups")

    updated = 0
    skipped_became_annotated = 0
    if not args.dry_run:
        for capture, detections in prepared:
            replaced = database.replace_unannotated_detections(
                capture["captureId"],
                detections,
                model_sha256=model_sha256,
                model_name=model_name,
                confidence_threshold=float(args.confidence_threshold),
                nms_iou_threshold=float(args.nms_iou_threshold),
                provider="onnxruntime-cpu-refresh",
                allow_draft=include_drafts,
            )
            if replaced:
                updated += 1
            else:
                skipped_became_annotated += 1

    summary = {
        "status": "dry-run" if args.dry_run else "completed",
        "campaignId": args.campaign_id,
        "fromLayout": args.from_layout,
        "includeDrafts": include_drafts,
        "model": str(model_path),
        "modelSha256": model_sha256,
        "confidenceThreshold": float(args.confidence_threshold),
        "nmsIouThreshold": float(args.nms_iou_threshold),
        "selectedCaptures": len(prepared),
        "updatedCaptures": 0 if args.dry_run else updated,
        "skippedBecauseAnnotationAppeared": (
            0 if args.dry_run else skipped_became_annotated
        ),
        "totalDetections": sum(len(detections) for _capture, detections in prepared),
        "backup": None if backup_path is None else str(backup_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def validate_arguments(
    args: argparse.Namespace,
    *,
    database_path: Path,
    model_path: Path,
    layout_path: Path,
) -> None:
    if args.from_layout < 1:
        raise ValueError("--from-layout must be at least 1")
    if not 0.0 <= float(args.confidence_threshold) <= 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")
    if not 0.0 <= float(args.nms_iou_threshold) <= 1.0:
        raise ValueError("--nms-iou-threshold must be between 0 and 1")
    if args.max_detections <= 0:
        raise ValueError("--max-detections must be positive")
    for path in (database_path, model_path, layout_path):
        if not path.is_file():
            raise FileNotFoundError(path)


def map_detections(
    decoded: list[Any],
    manifest: dict[str, Any],
    layout: dict[str, Any],
) -> list[dict[str, Any]]:
    enabled = {
        region: bool(manifest["regionRects"][region]["enabled"])
        for region in REGION_KEYS
    }
    result: list[dict[str, Any]] = []
    for detection_index, detection in enumerate(decoded):
        composite = {
            "x": float(detection.box.x1),
            "y": float(detection.box.y1),
            "width": float(detection.box.x2 - detection.box.x1),
            "height": float(detection.box.y2 - detection.box.y1),
        }
        center_x = composite["x"] + composite["width"] / 2
        center_y = composite["y"] + composite["height"] / 2
        region = region_at(center_x, center_y, enabled, layout)
        if region == "invalid":
            original = None
            preview = None
        else:
            destination = layout["regions"][region]["destination"]
            source = manifest["regionRects"][region]["pixel"]
            original = composite_rect_to_source(composite, destination, source)
            preview = source_rect_to_preview(original, manifest["preview"])
        result.append(
            {
                "detectionIndex": detection_index,
                "region": region,
                "confidence": float(detection.score),
                "composite": composite,
                "original": original,
                "preview": preview,
            }
        )
    return result


def region_at(
    x: float,
    y: float,
    enabled: dict[str, bool],
    layout: dict[str, Any],
) -> str:
    for region in REGION_KEYS:
        if not enabled[region]:
            continue
        destination = layout["regions"][region]["destination"]
        if (
            float(destination["x"]) <= x < float(destination["x"] + destination["width"])
            and float(destination["y"]) <= y < float(destination["y"] + destination["height"])
        ):
            return region
    return "invalid"


def composite_rect_to_source(
    composite: dict[str, float],
    destination: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, float]:
    left = max(composite["x"], float(destination["x"]))
    top = max(composite["y"], float(destination["y"]))
    right = min(
        composite["x"] + composite["width"],
        float(destination["x"] + destination["width"]),
    )
    bottom = min(
        composite["y"] + composite["height"],
        float(destination["y"] + destination["height"]),
    )
    scale_x = float(source["width"]) / float(destination["width"])
    scale_y = float(source["height"]) / float(destination["height"])
    return {
        "x": float(source["x"]) + (left - float(destination["x"])) * scale_x,
        "y": float(source["y"]) + (top - float(destination["y"])) * scale_y,
        "width": max(0.0, right - left) * scale_x,
        "height": max(0.0, bottom - top) * scale_y,
    }


def source_rect_to_preview(
    source: dict[str, float],
    preview_contract: dict[str, Any],
) -> dict[str, float]:
    video = preview_contract["videoElement"]
    scale = float(preview_contract["sourceToDisplayScale"])
    return {
        "x": (
            float(video["x"])
            + float(preview_contract["sourceDisplayOffsetX"])
            + source["x"] * scale
        ),
        "y": (
            float(video["y"])
            + float(preview_contract["sourceDisplayOffsetY"])
            + source["y"] * scale
        ),
        "width": source["width"] * scale,
        "height": source["height"] * scale,
    }


def create_database_backup(database_path: Path, backup_directory: Path) -> Path:
    backup_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_directory / f"dataset-before-detection-refresh-{timestamp}.sqlite"
    with sqlite3.connect(database_path) as source, sqlite3.connect(backup_path) as destination:
        source.backup(destination)
    return backup_path


def resolve_storage_path(storage_root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe storage-relative path: {relative_path}")
    path = storage_root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(storage_root.resolve())
    except ValueError as error:
        raise ValueError(f"Capture image escapes storage root: {relative_path}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_layout(layout: dict[str, Any]) -> None:
    composite = layout.get("composite")
    if not isinstance(composite, dict):
        raise ValueError("Capture layout has no composite object")
    if int(composite.get("width", 0)) != 320 or int(composite.get("height", 0)) != 320:
        raise ValueError("Capture layout composite must be 320x320")
    regions = layout.get("regions")
    if not isinstance(regions, dict) or set(regions) != set(REGION_KEYS):
        raise ValueError("Capture layout regions do not match the detector contract")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
