from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


CATEGORY = {
    "id": 1,
    "name": "mahjong_tile",
    "supercategory": "mahjong_tile",
}
REGION_KEYS = ("completed_hand", "dora_indicators", "melds")
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    capture_root = repository_root / ".local" / "recognition" / "capture_dataset"
    parser = argparse.ArgumentParser(
        description=(
            "Build a layout-disjoint human-OBB train/validation corpus from capture "
            "annotations changed since the pre-OBB-review SQLite backup."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--current-database",
        type=Path,
        default=capture_root / "dataset.sqlite",
    )
    parser.add_argument(
        "--baseline-database",
        type=Path,
        default=capture_root / "dataset.pre-obb-review.sqlite",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repository_root / ".local" / "recognition" / "rotated_detector_corpus",
    )
    parser.add_argument(
        "--campaign-id",
        action="append",
        default=[],
        help="Optional campaign filter. May be repeated. Defaults to all campaigns.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-drafts",
        action="store_true",
        help="Include changed draft annotations. By default only complete annotations are used.",
    )
    parser.add_argument(
        "--include-new-annotations",
        action="store_true",
        help="Also include current annotations absent from the baseline backup.",
    )
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help=(
            "Use every annotation whose document review.state is human_reviewed instead of "
            "selecting only geometry changed since the pre-OBB-review backup."
        ),
    )
    parser.add_argument(
        "--geometry-epsilon",
        type=float,
        default=1.0e-4,
        help="Minimum numeric box-geometry delta treated as a manual edit.",
    )
    parser.add_argument(
        "--allow-capture-split",
        action="store_true",
        help=(
            "If only one campaign/layout group is available, fall back to a capture-level split. "
            "This is useful for smoke tests but leaks layout geometry into validation."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    current_database = args.current_database.resolve()
    baseline_database = args.baseline_database.resolve()
    output_directory = args.output_directory.resolve()

    validate_args(args)
    for path in (current_database, baseline_database):
        if not path.is_file():
            raise FileNotFoundError(path)
    ensure_inside_repository(repository_root, output_directory)
    if output_directory.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists; pass --overwrite to replace it: {output_directory}"
            )
        shutil.rmtree(output_directory)

    current_rows = load_capture_rows(
        current_database,
        campaign_ids=frozenset(str(value) for value in args.campaign_id),
    )
    baseline_annotations = load_annotation_snapshot(baseline_database)
    if bool(args.reviewed_only):
        selected_rows, diff_summary = select_reviewed_rows(current_rows)
        selection_kind = "annotation_review_state_human_reviewed"
    else:
        selected_rows, diff_summary = select_changed_rows(
            current_rows,
            baseline_annotations,
            include_drafts=bool(args.include_drafts),
            include_new=bool(args.include_new_annotations),
            geometry_epsilon=float(args.geometry_epsilon),
        )
        selection_kind = "annotation_geometry_changed_since_backup"
    if not selected_rows:
        raise ValueError(
            "No human OBB annotations matched the requested selection. "
            "For the reviewed corpus, complete OBB review first and pass --reviewed-only."
        )

    payload = rows_to_rotated_coco(
        selected_rows,
        repository_root=repository_root,
        capture_root=current_database.parent,
    )
    train_payload, val_payload, split = split_payload(
        payload,
        train_fraction=float(args.train_fraction),
        seed=int(args.seed),
        allow_capture_split=bool(args.allow_capture_split),
    )

    annotations_directory = output_directory / "annotations"
    all_path = annotations_directory / "human_all.json"
    train_path = annotations_directory / "train.json"
    val_path = annotations_directory / "val.json"
    atomic_write_json(all_path, payload, compact=True)
    atomic_write_json(train_path, train_payload, compact=True)
    atomic_write_json(val_path, val_payload, compact=True)

    provenance = {
        "artifact": "rotated_detector_human_obb_corpus",
        "schema_version": SCHEMA_VERSION,
        "repository_root": str(repository_root),
        "current_database": str(current_database),
        "baseline_database": str(baseline_database),
        "selection": {
            "kind": selection_kind,
            "campaign_ids": list(args.campaign_id),
            "include_drafts": bool(args.include_drafts),
            "include_new_annotations": bool(args.include_new_annotations),
            "geometry_epsilon": float(args.geometry_epsilon),
            **diff_summary,
        },
        "split": split,
        "counts": {
            "human_all": coco_counts(payload),
            "train": coco_counts(train_payload),
            "val": coco_counts(val_payload),
        },
        "outputs": {
            "all": str(all_path),
            "train": str(train_path),
            "val": str(val_path),
        },
        "contract": {
            "input_size": [320, 320],
            "category_count": 1,
            "obb": "[center_x, center_y, width, height, angle_deg]",
            "angle_period_deg": 180,
            "coordinates": "320x320 fixed semantic composite pixels",
            "validation_policy": "layout-disjoint unless explicit capture-split fallback is enabled",
        },
    }
    provenance_path = output_directory / "provenance.json"
    atomic_write_json(provenance_path, provenance, compact=False)

    print(
        json.dumps(
            {
                "status": "completed",
                "output_directory": str(output_directory),
                "selected_captures": diff_summary["selected_capture_count"],
                "counts": provenance["counts"],
                "split": split,
                "provenance": str(provenance_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < float(args.train_fraction) < 1.0:
        raise ValueError("--train-fraction must be strictly between zero and one")
    if not math.isfinite(float(args.geometry_epsilon)) or float(args.geometry_epsilon) < 0.0:
        raise ValueError("--geometry-epsilon must be finite and non-negative")


def ensure_inside_repository(repository_root: Path, output_directory: Path) -> None:
    try:
        relative = output_directory.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(f"Output must be inside repository root: {output_directory}") from error
    if not relative.parts:
        raise ValueError("Output directory must not be the repository root")


def load_capture_rows(
    database: Path,
    *,
    campaign_ids: frozenset[str],
) -> list[dict[str, Any]]:
    sql = """
        SELECT
            capture.id AS capture_id,
            capture.composite_path,
            capture.manifest_json,
            capture_task.campaign_id,
            capture_task.layout_id,
            capture_task.layout_ordinal,
            capture_task.environment_ordinal,
            capture_task.brightness,
            capture_task.shadow,
            capture_task.task_order,
            capture_annotation.status AS annotation_status,
            capture_annotation.annotation_json,
            capture_annotation.updated_at
        FROM capture
        JOIN capture_task ON capture_task.id = capture.task_id
        JOIN capture_annotation ON capture_annotation.capture_id = capture.id
        ORDER BY capture_task.task_order, capture.id
    """
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute(sql).fetchall()]
    if campaign_ids:
        rows = [row for row in rows if str(row["campaign_id"]) in campaign_ids]
    return rows


def load_annotation_snapshot(database: Path) -> dict[str, dict[str, Any]]:
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT capture_id, status, annotation_json, updated_at
            FROM capture_annotation
            """
        ).fetchall()
    return {
        str(row["capture_id"]): {
            "status": str(row["status"]),
            "annotation_json": str(row["annotation_json"]),
            "updated_at": str(row["updated_at"]),
        }
        for row in rows
    }


def select_changed_rows(
    rows: Sequence[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    *,
    include_drafts: bool,
    include_new: bool,
    geometry_epsilon: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    skipped_status = 0
    unchanged = 0
    missing_baseline = 0
    changed_box_counts: list[int] = []
    max_geometry_deltas: list[float] = []

    for row in rows:
        status = str(row["annotation_status"])
        if status != "complete" and not (include_drafts and status == "draft"):
            skipped_status += 1
            continue
        capture_id = str(row["capture_id"])
        baseline_row = baseline.get(capture_id)
        if baseline_row is None:
            missing_baseline += 1
            if include_new:
                selected.append(row)
                current_document = parse_annotation_document(str(row["annotation_json"]), capture_id)
                changed_box_counts.append(annotation_box_count(current_document))
                max_geometry_deltas.append(float("inf"))
            continue

        current_document = parse_annotation_document(str(row["annotation_json"]), capture_id)
        baseline_document = parse_annotation_document(
            str(baseline_row["annotation_json"]), capture_id
        )
        changed_count, max_delta = annotation_geometry_delta(
            current_document,
            baseline_document,
            epsilon=geometry_epsilon,
        )
        if changed_count == 0:
            unchanged += 1
            continue
        selected.append(row)
        changed_box_counts.append(changed_count)
        max_geometry_deltas.append(max_delta)

    return selected, {
        "current_annotation_count": len(rows),
        "selected_capture_count": len(selected),
        "skipped_status_count": skipped_status,
        "unchanged_capture_count": unchanged,
        "missing_baseline_count": missing_baseline,
        "changed_box_count_total": sum(changed_box_counts),
        "changed_box_count_per_capture": summarize_numbers(changed_box_counts),
        "finite_max_geometry_delta": summarize_numbers(
            [value for value in max_geometry_deltas if math.isfinite(value)]
        ),
        "selected_capture_ids": [str(row["capture_id"]) for row in selected],
    }


def select_reviewed_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    non_reviewed = 0
    draft_reviewed = 0
    reviewed_box_counts: list[int] = []

    for row in rows:
        capture_id = str(row["capture_id"])
        document = parse_annotation_document(str(row["annotation_json"]), capture_id)
        review = document.get("review")
        if not isinstance(review, dict) or review.get("state") != "human_reviewed":
            non_reviewed += 1
            continue
        if str(row["annotation_status"]) == "draft":
            draft_reviewed += 1
        selected.append(row)
        reviewed_box_counts.append(annotation_box_count(document))

    return selected, {
        "current_annotation_count": len(rows),
        "selected_capture_count": len(selected),
        "non_reviewed_capture_count": non_reviewed,
        "draft_reviewed_capture_count": draft_reviewed,
        "reviewed_box_count_total": sum(reviewed_box_counts),
        "reviewed_box_count_per_capture": summarize_numbers(reviewed_box_counts),
        "selected_capture_ids": [str(row["capture_id"]) for row in selected],
    }


def parse_annotation_document(raw: str, capture_id: str) -> dict[str, Any]:
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError(f"Capture {capture_id} annotation root is not an object")
    boxes = document.get("boxes")
    if not isinstance(boxes, dict) or set(boxes) != set(REGION_KEYS):
        raise ValueError(f"Capture {capture_id} annotation has invalid region boxes")
    for region in REGION_KEYS:
        if not isinstance(boxes[region], list):
            raise ValueError(f"Capture {capture_id} boxes.{region} is not an array")
    return document


def annotation_box_count(document: dict[str, Any]) -> int:
    return sum(len(document["boxes"][region]) for region in REGION_KEYS)


def annotation_geometry_delta(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    epsilon: float,
) -> tuple[int, float]:
    current_boxes = boxes_by_id(current)
    baseline_boxes = boxes_by_id(baseline)
    changed_ids = current_boxes.keys() ^ baseline_boxes.keys()
    changed_count = len(changed_ids)
    max_delta = float("inf") if changed_ids else 0.0

    for box_id in current_boxes.keys() & baseline_boxes.keys():
        current_box = current_boxes[box_id]
        baseline_box = baseline_boxes[box_id]
        deltas = [
            abs(float(current_box[key]) - float(baseline_box[key]))
            for key in ("centerX", "centerY", "width", "height", "angleDeg")
        ]
        box_delta = max(deltas)
        if box_delta > epsilon:
            changed_count += 1
            max_delta = max(max_delta, box_delta)
    return changed_count, max_delta


def boxes_by_id(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for region in REGION_KEYS:
        for raw_box in document["boxes"][region]:
            if not isinstance(raw_box, dict):
                raise ValueError(f"Non-object annotation box in {region}")
            box_id = str(raw_box.get("id", ""))
            if not box_id or box_id in result:
                raise ValueError(f"Invalid or duplicate annotation box id: {box_id!r}")
            result[box_id] = raw_box
    return result


def rows_to_rotated_coco(
    rows: Sequence[dict[str, Any]],
    *,
    repository_root: Path,
    capture_root: Path,
) -> dict[str, Any]:
    layout = load_json(repository_root / "tools" / "recognition" / "capture_layout.v1.json")
    validate_layout(layout)
    payload = empty_coco("Human-reviewed OBB seed corpus")
    next_image_id = 1
    next_annotation_id = 1

    for row in rows:
        capture_id = str(row["capture_id"])
        composite_relative = safe_relative_path(str(row["composite_path"]))
        composite_path = capture_root / composite_relative
        if not composite_path.is_file():
            raise FileNotFoundError(composite_path)
        file_name = repository_relative_path(repository_root, composite_path)
        manifest = json.loads(str(row["manifest_json"]))
        document = parse_annotation_document(str(row["annotation_json"]), capture_id)

        image_record = {
            "id": next_image_id,
            "file_name": file_name,
            "width": 320,
            "height": 320,
            "capture_id": capture_id,
            "campaign_id": str(row["campaign_id"]),
            "layout_id": str(row["layout_id"]),
            "layout_ordinal": int(row["layout_ordinal"]),
            "environment_ordinal": int(row["environment_ordinal"]),
            "brightness": str(row["brightness"]),
            "shadow": str(row["shadow"]),
            "annotation_status": str(row["annotation_status"]),
            "annotation_updated_at": str(row["updated_at"]),
            "dataset_origin": "human_obb_review",
            "split_group": f"{row['campaign_id']}::{row['layout_id']}",
        }
        payload["images"].append(image_record)

        for region in REGION_KEYS:
            region_manifest = manifest["regionRects"][region]["pixel"]
            crop_width = max(1, math.floor(float(region_manifest["width"]) + 0.5))
            crop_height = max(1, math.floor(float(region_manifest["height"]) + 0.5))
            destination = layout["regions"][region]["destination"]
            scale_x = float(destination["width"]) / crop_width
            scale_y = float(destination["height"]) / crop_height
            for box in document["boxes"][region]:
                obb = canonicalize_obb(
                    anisotropic_box_to_approx_obb(
                        center_x=float(box["centerX"]),
                        center_y=float(box["centerY"]),
                        width=float(box["width"]),
                        height=float(box["height"]),
                        angle_deg=float(box["angleDeg"]),
                        scale_x=scale_x,
                        scale_y=scale_y,
                        offset_x=float(destination["x"]),
                        offset_y=float(destination["y"]),
                    )
                )
                validate_obb(obb, capture_id=capture_id, region=region)
                polygon = obb_corners(obb)
                if not polygon_inside_canvas(polygon, 320, 320):
                    raise ValueError(
                        f"Capture {capture_id}/{region} box {box.get('id')} escapes 320 composite"
                    )
                hbb = polygon_hbb(polygon)
                payload["annotations"].append(
                    {
                        "id": next_annotation_id,
                        "image_id": next_image_id,
                        "category_id": 1,
                        "bbox": hbb,
                        "obb": obb,
                        "segmentation": [[coordinate for point in polygon for coordinate in point]],
                        "area": obb[2] * obb[3],
                        "iscrowd": 0,
                        "region": region,
                        "annotation_box_id": str(box["id"]),
                        "annotation_source": "human_obb_review",
                    }
                )
                next_annotation_id += 1
        next_image_id += 1

    if not payload["images"] or not payload["annotations"]:
        raise ValueError("Generated human OBB corpus is empty")
    return payload


def split_payload(
    payload: dict[str, Any],
    *,
    train_fraction: float,
    seed: int,
    allow_capture_split: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for image in payload["images"]:
        groups[str(image["split_group"])].append(int(image["id"]))

    rng = random.Random(seed)
    group_names = sorted(groups)
    split_unit = "campaign_layout"
    if len(group_names) >= 2:
        rng.shuffle(group_names)
        train_group_count = max(
            1,
            min(len(group_names) - 1, round(len(group_names) * train_fraction)),
        )
        train_groups = frozenset(group_names[:train_group_count])
        train_ids = {
            image_id for group in train_groups for image_id in groups[group]
        }
        val_ids = {
            image_id
            for group in group_names[train_group_count:]
            for image_id in groups[group]
        }
    else:
        if not allow_capture_split:
            raise ValueError(
                "Only one campaign/layout group is available. Review at least one more layout "
                "for a leakage-resistant validation set, or pass --allow-capture-split for a smoke test."
            )
        split_unit = "capture_fallback"
        image_ids = sorted(int(image["id"]) for image in payload["images"])
        if len(image_ids) < 2:
            raise ValueError("At least two reviewed captures are required for train/val")
        rng.shuffle(image_ids)
        train_count = max(1, min(len(image_ids) - 1, round(len(image_ids) * train_fraction)))
        train_ids = set(image_ids[:train_count])
        val_ids = set(image_ids[train_count:])
        train_groups = frozenset(
            str(image["capture_id"])
            for image in payload["images"]
            if int(image["id"]) in train_ids
        )
        group_names = [
            str(image["capture_id"])
            for image in payload["images"]
            if int(image["id"]) in val_ids
        ]

    train = subset_coco(payload, train_ids, description="Human OBB training partition")
    val = subset_coco(payload, val_ids, description="Human OBB validation partition")
    return train, val, {
        "seed": seed,
        "train_fraction": train_fraction,
        "unit": split_unit,
        "train_groups": sorted(train_groups),
        "val_groups": sorted(group_names[len(train_groups):])
        if split_unit == "campaign_layout"
        else sorted(group_names),
        "train_image_count": len(train["images"]),
        "val_image_count": len(val["images"]),
    }


def subset_coco(
    payload: dict[str, Any],
    image_ids: Iterable[int],
    *,
    description: str,
) -> dict[str, Any]:
    selected = set(int(value) for value in image_ids)
    images = [dict(image) for image in payload["images"] if int(image["id"]) in selected]
    annotations = [
        dict(annotation)
        for annotation in payload["annotations"]
        if int(annotation["image_id"]) in selected
    ]
    if not images or not annotations:
        raise ValueError(f"Empty split: {description}")
    result = empty_coco(description)
    result["images"] = images
    result["annotations"] = annotations
    return result


def validate_layout(layout: dict[str, Any]) -> None:
    if layout.get("composite") != {
        "width": 320,
        "height": 320,
        "paddingRgb": [0, 0, 0],
    }:
        raise ValueError("Unexpected 320x320 capture layout contract")
    regions = layout.get("regions")
    if not isinstance(regions, dict) or set(regions) != set(REGION_KEYS):
        raise ValueError("Unexpected semantic region set")


def anisotropic_box_to_approx_obb(
    *,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    angle_deg: float,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> list[float]:
    """Map a crop-space rotated rectangle into the composite as a rectangle approximation.

    The capture crop can differ by a pixel from the fixed-layout destination aspect ratio,
    so the exact affine image mapping is very slightly anisotropic. That technically maps
    a rotated rectangle to a parallelogram. For mjtensu we intentionally keep the target
    representation rectangular: transform the two local axes independently, preserve their
    transformed lengths, and use the transformed width-axis direction as the OBB angle.
    """
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    width_axis_x = scale_x * cosine
    width_axis_y = scale_y * sine
    height_axis_x = -scale_x * sine
    height_axis_y = scale_y * cosine
    transformed_width_scale = math.hypot(width_axis_x, width_axis_y)
    transformed_height_scale = math.hypot(height_axis_x, height_axis_y)
    transformed_angle = math.degrees(math.atan2(width_axis_y, width_axis_x))
    return [
        offset_x + center_x * scale_x,
        offset_y + center_y * scale_y,
        width * transformed_width_scale,
        height * transformed_height_scale,
        normalize_angle_180(transformed_angle),
    ]


def canonicalize_obb(obb: Sequence[float]) -> list[float]:
    """Use one geometric representation: short side is width, long side is height.

    Rotated rectangles have the equivalence (w,h,theta) == (h,w,theta+90deg).
    Normalizing that ambiguity is essential for angle regression because sin(2theta),
    cos(2theta) changes sign under a 90-degree representation swap.
    """
    cx, cy, width, height, angle_deg = (float(value) for value in obb)
    if width > height:
        width, height = height, width
        angle_deg += 90.0
    return [cx, cy, width, height, normalize_angle_180(angle_deg)]


def validate_obb(obb: Sequence[float], *, capture_id: str, region: str) -> None:
    if len(obb) != 5 or not all(math.isfinite(float(value)) for value in obb):
        raise ValueError(f"Invalid OBB in {capture_id}/{region}: {obb}")
    if float(obb[2]) <= 1.0 or float(obb[3]) <= 1.0:
        raise ValueError(f"Non-positive OBB size in {capture_id}/{region}: {obb}")


def obb_corners(obb: Sequence[float]) -> list[tuple[float, float]]:
    cx, cy, width, height, angle_deg = (float(value) for value in obb)
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    half_width = width / 2.0
    half_height = height / 2.0
    result = []
    for local_x, local_y in (
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ):
        result.append(
            (
                cx + local_x * cosine - local_y * sine,
                cy + local_x * sine + local_y * cosine,
            )
        )
    return result


def polygon_hbb(polygon: Sequence[tuple[float, float]]) -> list[float]:
    left = min(x for x, _y in polygon)
    top = min(y for _x, y in polygon)
    right = max(x for x, _y in polygon)
    bottom = max(y for _x, y in polygon)
    return [left, top, right - left, bottom - top]


def polygon_inside_canvas(
    polygon: Sequence[tuple[float, float]], width: int, height: int
) -> bool:
    return all(
        -1.0e-4 <= x <= width + 1.0e-4 and -1.0e-4 <= y <= height + 1.0e-4
        for x, y in polygon
    )


def normalize_angle_180(angle_deg: float) -> float:
    angle = (angle_deg + 90.0) % 180.0 - 90.0
    if angle >= 90.0:
        angle -= 180.0
    return angle


def empty_coco(description: str) -> dict[str, Any]:
    return {
        "info": {
            "description": description,
            "rotated_bbox_schema": "cx_cy_w_h_angle_deg",
            "rotated_bbox_angle_period_deg": 180,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [CATEGORY],
    }


def coco_counts(payload: dict[str, Any]) -> dict[str, int]:
    return {
        "images": len(payload["images"]),
        "annotations": len(payload["annotations"]),
    }


def summarize_numbers(values: Sequence[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": sum(numbers) / len(numbers),
    }


def safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe relative path: {value}")
    return Path(*pure.parts)


def repository_relative_path(repository_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(f"Image must be inside repository root: {resolved}") from error
    return PurePosixPath(relative.as_posix()).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def atomic_write_json(path: Path, payload: Any, *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            if compact:
                json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
            else:
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
    raise SystemExit(main())
