from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


CATEGORY = {
    "id": 1,
    "name": "mahjong_tile",
    "supercategory": "mahjong_tile",
}
REGION_KEYS = ("completed_hand", "dora_indicators", "melds")


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Build a NanoDet fine-tuning COCO dataset from completed capture annotations. "
            "The split is performed by layout so four environment variants of one layout "
            "never cross train and validation."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--campaign-id", default="initial-120")
    parser.add_argument(
        "--layout-count",
        type=int,
        default=10,
        help="Use layout ordinals [0, layout-count). Every selected layout must be fully annotated.",
    )
    parser.add_argument("--train-layout-fraction", type=float, default=0.8)
    parser.add_argument("--real-repeat", type=int, default=20)
    parser.add_argument("--base-replay-images", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--skip-image-existence-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    storage_root = (
        args.storage_root.resolve()
        if args.storage_root is not None
        else repository_root / ".local" / "recognition" / "capture_dataset"
    )
    output_directory = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else repository_root / ".local" / "recognition" / "nanodet_capture_finetune_dataset"
    )
    database_path = storage_root / "dataset.sqlite"
    layout_path = repository_root / "tools" / "recognition" / "capture_layout.v1.json"
    old_dataset_root = (
        repository_root / ".local" / "recognition" / "nanodet_composite_augmented_dataset"
    )
    old_train_path = old_dataset_root / "annotations" / "instances_train.json"
    old_composite_train_path = (
        old_dataset_root / "annotations" / "instances_composite_train.json"
    )
    old_composite_val_path = (
        old_dataset_root / "annotations" / "instances_composite_val.json"
    )

    if args.layout_count < 2:
        raise ValueError("layout-count must be at least two")
    if not 0 < args.train_layout_fraction < 1:
        raise ValueError("train-layout-fraction must be strictly between zero and one")
    if args.real_repeat < 1:
        raise ValueError("real-repeat must be positive")
    if args.base_replay_images < 0:
        raise ValueError("base-replay-images must not be negative")
    for path in (
        database_path,
        layout_path,
        old_train_path,
        old_composite_train_path,
        old_composite_val_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    layout = load_json(layout_path)
    validate_capture_layout(layout)
    captures = load_completed_captures(
        database_path,
        campaign_id=args.campaign_id,
        layout_count=args.layout_count,
    )
    layout_ids = sorted(
        {str(capture["layout_id"]) for capture in captures},
        key=lambda layout_id: min(
            int(capture["layout_ordinal"])
            for capture in captures
            if capture["layout_id"] == layout_id
        ),
    )
    train_layout_ids, val_layout_ids = split_layouts(
        layout_ids,
        train_fraction=float(args.train_layout_fraction),
        seed=int(args.seed),
    )

    check_images = not args.skip_image_existence_check
    real_train_unique = captures_to_coco(
        captures,
        included_layout_ids=train_layout_ids,
        repository_root=repository_root,
        storage_root=storage_root,
        layout=layout,
        description="Annotated real capture training images",
        check_images=check_images,
    )
    real_val = captures_to_coco(
        captures,
        included_layout_ids=val_layout_ids,
        repository_root=repository_root,
        storage_root=storage_root,
        layout=layout,
        description="Held-out annotated real capture validation images",
        check_images=check_images,
    )
    real_train_repeated = repeat_coco_images(real_train_unique, args.real_repeat)

    old_train = load_coco(old_train_path)
    old_composite_train = load_coco(old_composite_train_path)
    base_replay = sample_dataset_origin(
        old_train,
        dataset_origin="base_train",
        image_count=args.base_replay_images,
        seed=args.seed,
    )

    merged_train = merge_coco_payloads(
        [
            ("real_capture", real_train_repeated),
            ("composite_replay", old_composite_train),
            ("base_replay", base_replay),
        ],
        description="NanoDet real-capture fine-tuning train dataset",
    )

    annotations_directory = output_directory / "annotations"
    paths = {
        "train": annotations_directory / "instances_train.json",
        "real_train": annotations_directory / "instances_real_train.json",
        "real_val": annotations_directory / "instances_real_val.json",
    }
    payloads = {
        "train": merged_train,
        "real_train": real_train_unique,
        "real_val": real_val,
    }
    for name, path in paths.items():
        atomic_write_json(path, payloads[name], compact=True)

    provenance = {
        "artifact": "nanodet_capture_finetune_dataset",
        "campaign_id": args.campaign_id,
        "repository_root": str(repository_root),
        "storage_root": str(storage_root),
        "layout_count": args.layout_count,
        "seed": args.seed,
        "split_unit": "layout_id",
        "train_layout_fraction": args.train_layout_fraction,
        "train_layout_ids": sorted(train_layout_ids),
        "val_layout_ids": sorted(val_layout_ids),
        "real_repeat": args.real_repeat,
        "base_replay_images_requested": args.base_replay_images,
        "counts": {
            "completed_captures": len(captures),
            "real_train_unique": coco_counts(real_train_unique),
            "real_train_repeated": coco_counts(real_train_repeated),
            "real_val": coco_counts(real_val),
            "composite_replay": coco_counts(old_composite_train),
            "base_replay": coco_counts(base_replay),
            "merged_train": coco_counts(merged_train),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "existing_regression_validation": str(old_composite_val_path),
        "images_copied": False,
    }
    provenance_path = output_directory / "provenance.json"
    atomic_write_json(provenance_path, provenance, compact=False)

    print(
        json.dumps(
            {
                "status": "completed",
                "output_directory": str(output_directory),
                "train_layout_ids": provenance["train_layout_ids"],
                "val_layout_ids": provenance["val_layout_ids"],
                "counts": provenance["counts"],
                "provenance": str(provenance_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_completed_captures(
    database_path: Path,
    *,
    campaign_id: str,
    layout_count: int,
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        layout_rows = connection.execute(
            """
            SELECT
                capture_task.layout_id,
                capture_task.layout_ordinal,
                COUNT(*) AS task_count,
                SUM(CASE WHEN capture.id IS NOT NULL THEN 1 ELSE 0 END) AS capture_count,
                SUM(CASE WHEN capture_annotation.status = 'complete' THEN 1 ELSE 0 END) AS complete_count
            FROM capture_task
            LEFT JOIN capture ON capture.task_id = capture_task.id
            LEFT JOIN capture_annotation ON capture_annotation.capture_id = capture.id
            WHERE capture_task.campaign_id = ? AND capture_task.layout_ordinal < ?
            GROUP BY capture_task.layout_id, capture_task.layout_ordinal
            ORDER BY capture_task.layout_ordinal
            """,
            (campaign_id, layout_count),
        ).fetchall()
        if len(layout_rows) != layout_count:
            raise ValueError(
                f"Expected {layout_count} selected layouts, found {len(layout_rows)} in {campaign_id}"
            )
        incomplete = [
            {
                "layout_id": str(row["layout_id"]),
                "task_count": int(row["task_count"]),
                "capture_count": int(row["capture_count"] or 0),
                "complete_count": int(row["complete_count"] or 0),
            }
            for row in layout_rows
            if int(row["task_count"]) != int(row["capture_count"] or 0)
            or int(row["task_count"]) != int(row["complete_count"] or 0)
        ]
        if incomplete:
            raise ValueError(
                "Every selected layout must have all environment captures annotated complete: "
                + json.dumps(incomplete, ensure_ascii=False)
            )

        rows = connection.execute(
            """
            SELECT
                capture.id AS capture_id,
                capture.composite_path,
                capture.manifest_json,
                capture_task.layout_id,
                capture_task.layout_ordinal,
                capture_task.environment_ordinal,
                capture_task.brightness,
                capture_task.shadow,
                capture_annotation.annotation_json
            FROM capture
            JOIN capture_task ON capture_task.id = capture.task_id
            JOIN capture_annotation ON capture_annotation.capture_id = capture.id
            WHERE
                capture_task.campaign_id = ?
                AND capture_task.layout_ordinal < ?
                AND capture_annotation.status = 'complete'
            ORDER BY capture_task.layout_ordinal, capture_task.environment_ordinal
            """,
            (campaign_id, layout_count),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def split_layouts(
    layout_ids: list[str],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[frozenset[str], frozenset[str]]:
    if len(layout_ids) < 2:
        raise ValueError("At least two layouts are required")
    shuffled = list(layout_ids)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * train_fraction)))
    train = frozenset(shuffled[:train_count])
    val = frozenset(shuffled[train_count:])
    if train & val or train | val != frozenset(layout_ids):
        raise AssertionError("Invalid layout split")
    return train, val


def captures_to_coco(
    captures: list[dict[str, Any]],
    *,
    included_layout_ids: frozenset[str],
    repository_root: Path,
    storage_root: Path,
    layout: dict[str, Any],
    description: str,
    check_images: bool,
) -> dict[str, Any]:
    payload = empty_coco(description)
    image_id = 1
    annotation_id = 1
    storage_relative = storage_root.relative_to(repository_root)

    for capture in captures:
        if str(capture["layout_id"]) not in included_layout_ids:
            continue
        relative_path = safe_relative_path(str(capture["composite_path"]))
        image_path = storage_root / relative_path
        if check_images and not image_path.is_file():
            raise FileNotFoundError(image_path)
        file_name = str(PurePosixPath(storage_relative.as_posix()) / relative_path.as_posix())
        payload["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(layout["composite"]["width"]),
                "height": int(layout["composite"]["height"]),
                "capture_id": str(capture["capture_id"]),
                "layout_id": str(capture["layout_id"]),
                "layout_ordinal": int(capture["layout_ordinal"]),
                "environment": {
                    "brightness": str(capture["brightness"]),
                    "shadow": str(capture["shadow"]),
                },
                "dataset_origin": "real_capture",
            }
        )
        manifest = json.loads(capture["manifest_json"])
        annotation = json.loads(capture["annotation_json"])
        boxes_by_region = annotation.get("boxes")
        if not isinstance(boxes_by_region, dict):
            raise ValueError(f"Capture {capture['capture_id']} annotation has no boxes")
        for region_key in REGION_KEYS:
            region_boxes = boxes_by_region.get(region_key)
            if not isinstance(region_boxes, list):
                raise ValueError(
                    f"Capture {capture['capture_id']} boxes.{region_key} is not an array"
                )
            pixel = manifest["regionRects"][region_key]["pixel"]
            crop_width = max(1, math.floor(float(pixel["width"]) + 0.5))
            crop_height = max(1, math.floor(float(pixel["height"]) + 0.5))
            destination = layout["regions"][region_key]["destination"]
            for box in region_boxes:
                polygon = rotated_box_to_composite_polygon(
                    box,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    destination=destination,
                )
                bbox = polygon_bbox(
                    polygon,
                    width=int(layout["composite"]["width"]),
                    height=int(layout["composite"]["height"]),
                )
                payload["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": CATEGORY["id"],
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "segmentation": [[coordinate for point in polygon for coordinate in point]],
                        "annotation_box_id": str(box["id"]),
                        "region": region_key,
                    }
                )
                annotation_id += 1
        image_id += 1
    if not payload["images"] or not payload["annotations"]:
        raise ValueError(f"Empty COCO partition: {description}")
    return payload


def rotated_box_to_composite_polygon(
    box: dict[str, Any],
    *,
    crop_width: int,
    crop_height: int,
    destination: dict[str, Any],
) -> list[list[float]]:
    center_x = float(box["centerX"])
    center_y = float(box["centerY"])
    half_width = float(box["width"]) / 2
    half_height = float(box["height"]) / 2
    angle = math.radians(float(box["angleDeg"]))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    scale_x = float(destination["width"]) / crop_width
    scale_y = float(destination["height"]) / crop_height
    polygon: list[list[float]] = []
    for local_x, local_y in (
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ):
        crop_x = center_x + local_x * cosine - local_y * sine
        crop_y = center_y + local_x * sine + local_y * cosine
        polygon.append(
            [
                float(destination["x"]) + crop_x * scale_x,
                float(destination["y"]) + crop_y * scale_y,
            ]
        )
    return polygon


def polygon_bbox(
    polygon: list[list[float]],
    *,
    width: int,
    height: int,
) -> list[float]:
    left = max(0.0, min(point[0] for point in polygon))
    top = max(0.0, min(point[1] for point in polygon))
    right = min(float(width), max(point[0] for point in polygon))
    bottom = min(float(height), max(point[1] for point in polygon))
    box_width = right - left
    box_height = bottom - top
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Generated non-positive bbox from polygon: {polygon}")
    return [left, top, box_width, box_height]


def repeat_coco_images(payload: dict[str, Any], repeat: int) -> dict[str, Any]:
    if repeat == 1:
        return json.loads(json.dumps(payload))
    annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    repeated = empty_coco(f"{payload['info']['description']} repeated {repeat} times")
    next_image_id = 1
    next_annotation_id = 1
    for repeat_index in range(repeat):
        for source_image in payload["images"]:
            source_image_id = int(source_image["id"])
            image = dict(source_image)
            image["id"] = next_image_id
            image["repeat_index"] = repeat_index
            repeated["images"].append(image)
            for source_annotation in annotations_by_image[source_image_id]:
                annotation = dict(source_annotation)
                annotation["id"] = next_annotation_id
                annotation["image_id"] = next_image_id
                repeated["annotations"].append(annotation)
                next_annotation_id += 1
            next_image_id += 1
    return repeated


def sample_dataset_origin(
    payload: dict[str, Any],
    *,
    dataset_origin: str,
    image_count: int,
    seed: int,
) -> dict[str, Any]:
    candidates = [
        image
        for image in payload["images"]
        if image.get("dataset_origin") == dataset_origin
    ]
    if image_count > len(candidates):
        raise ValueError(
            f"Requested {image_count} {dataset_origin} replay images; only {len(candidates)} exist"
        )
    selected = random.Random(seed).sample(candidates, image_count)
    selected_ids = {int(image["id"]) for image in selected}
    result = empty_coco(f"Sampled {dataset_origin} replay")
    result["images"] = [dict(image) for image in selected]
    result["annotations"] = [
        dict(annotation)
        for annotation in payload["annotations"]
        if int(annotation["image_id"]) in selected_ids
    ]
    return result


def merge_coco_payloads(
    sources: Iterable[tuple[str, dict[str, Any]]],
    *,
    description: str,
) -> dict[str, Any]:
    result = empty_coco(description)
    next_image_id = 1
    next_annotation_id = 1
    for source_name, payload in sources:
        annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        for source_image in payload["images"]:
            old_image_id = int(source_image["id"])
            image = {key: value for key, value in source_image.items() if key != "id"}
            image["id"] = next_image_id
            image["fine_tune_source"] = source_name
            result["images"].append(image)
            for source_annotation in annotations_by_image[old_image_id]:
                annotation = {
                    key: value
                    for key, value in source_annotation.items()
                    if key not in {"id", "image_id", "category_id"}
                }
                annotation["id"] = next_annotation_id
                annotation["image_id"] = next_image_id
                annotation["category_id"] = CATEGORY["id"]
                result["annotations"].append(annotation)
                next_annotation_id += 1
            next_image_id += 1
    return result


def load_coco(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"Invalid COCO field {key}: {path}")
    categories = payload["categories"]
    if len(categories) != 1 or categories[0].get("name") != CATEGORY["name"]:
        raise ValueError(f"Unexpected COCO category: {path}")
    return payload


def empty_coco(description: str) -> dict[str, Any]:
    return {
        "info": {"description": description},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [CATEGORY],
    }


def validate_capture_layout(layout: dict[str, Any]) -> None:
    if layout.get("composite") != {
        "width": 320,
        "height": 320,
        "paddingRgb": [0, 0, 0],
    }:
        raise ValueError("Unexpected capture composite contract")
    if set(layout.get("regions", {})) != set(REGION_KEYS):
        raise ValueError("Unexpected capture layout regions")


def safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe storage-relative path: {value}")
    return Path(*pure.parts)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def coco_counts(payload: dict[str, Any]) -> dict[str, int]:
    return {
        "images": len(payload["images"]),
        "annotations": len(payload["annotations"]),
    }


def atomic_write_json(path: Path, payload: Any, *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
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
    main()
