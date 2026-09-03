from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from PIL import Image, ImageDraw

if __package__:
    from .build_nanodet_region_rotation_augmented_dataset import (
        RegionRect,
        plan_region_transform,
        scale_rotate_point,
        transform_region_image,
    )
    from .build_rotated_detector_corpus import (
        CATEGORY,
        REGION_KEYS,
        empty_coco,
        normalize_angle_180,
        obb_corners,
        polygon_hbb,
    )
else:
    _repository_root_for_import = Path(__file__).resolve().parents[2]
    if str(_repository_root_for_import) not in sys.path:
        sys.path.insert(0, str(_repository_root_for_import))
    from tools.recognition.build_nanodet_region_rotation_augmented_dataset import (  # type: ignore[no-redef]
        RegionRect,
        plan_region_transform,
        scale_rotate_point,
        transform_region_image,
    )
    from tools.recognition.build_rotated_detector_corpus import (  # type: ignore[no-redef]
        CATEGORY,
        REGION_KEYS,
        empty_coco,
        normalize_angle_180,
        obb_corners,
        polygon_hbb,
    )


_EPSILON = 1.0e-5


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    corpus_root = repository_root / ".local" / "recognition" / "rotated_detector_corpus"
    parser = argparse.ArgumentParser(
        description=(
            "Generate OBB-preserving fixed-region rotation augmentation. Each semantic "
            "region is transformed independently, and the image plus every OBB receive "
            "the exact same scale/rotation/translation. Angles are resampled or shrunk "
            "until all GT corners remain visible."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=corpus_root / "annotations" / "train.json",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=repository_root / "tools" / "recognition" / "capture_layout.v1.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "rotated_detector_augmented_dataset",
    )
    parser.add_argument("--copies-per-image", type=int, default=12)
    parser.add_argument("--max-rotation-deg", type=float, default=45.0)
    parser.add_argument(
        "--max-shrink-fraction",
        type=float,
        default=0.20,
        help=(
            "Maximum angle-dependent uniform shrink used to keep all OBB corners visible. "
            "0.20 means scale may reach 0.80 at the maximum rotation."
        ),
    )
    parser.add_argument("--max-resamples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-originals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include untouched human OBB training images in the generated train manifest.",
    )
    parser.add_argument("--preflight-count", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    annotations_path = args.annotations.resolve()
    layout_path = args.layout.resolve()
    output_directory = args.output_directory.resolve()

    validate_args(args)
    for path in (annotations_path, layout_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    ensure_inside_repository(repository_root, output_directory)
    if output_directory.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists; pass --overwrite to replace it: {output_directory}"
            )
        shutil.rmtree(output_directory)

    source = load_rotated_coco(annotations_path)
    layout = load_json(layout_path)
    regions = load_regions(layout)
    by_image = annotations_by_image(source)

    images_directory = output_directory / "images"
    images_directory.mkdir(parents=True, exist_ok=True)
    output = empty_coco("Human OBB originals plus OBB-preserving rotation augmentation")
    next_image_id = 1
    next_annotation_id = 1

    if bool(args.include_originals):
        next_image_id, next_annotation_id = append_originals(
            source,
            output,
            next_image_id=next_image_id,
            next_annotation_id=next_annotation_id,
        )

    preflight: list[tuple[Image.Image, Image.Image, str]] = []
    transform_records: list[dict[str, Any]] = []
    region_transform_count = 0
    resample_count = 0
    shrink_count = 0
    applied_angles: list[float] = []
    applied_scales: list[float] = []

    for source_image in sorted(source["images"], key=lambda item: int(item["id"])):
        source_image_id = int(source_image["id"])
        image_path = resolve_repository_image(repository_root, str(source_image["file_name"]))
        with Image.open(image_path) as opened:
            source_rgb = opened.convert("RGB")
        if source_rgb.size != (320, 320):
            raise ValueError(f"Expected 320x320 source image, found {source_rgb.size}: {image_path}")
        source_annotations = sorted(
            by_image[source_image_id], key=lambda annotation: int(annotation["id"])
        )
        grouped = group_annotations_by_region(source_annotations, regions)

        for copy_index in range(int(args.copies_per_image)):
            augmented = source_rgb.copy()
            generated_annotations: list[dict[str, Any]] = []
            image_record: dict[str, Any] = {
                "source_image_id": source_image_id,
                "copy_index": copy_index,
                "regions": {},
            }

            for region_key in REGION_KEYS:
                region_annotations = grouped.get(region_key, [])
                if not region_annotations:
                    continue
                region = regions[region_key]
                local_polygons = [
                    tuple((x - region.x, y - region.y) for x, y in obb_corners(annotation["obb"]))
                    for annotation in region_annotations
                ]
                plan = plan_region_transform(
                    local_polygons,
                    region_width=region.width,
                    region_height=region.height,
                    seed=int(args.seed),
                    sample_key=(source_image.get("file_name"), copy_index, region_key, "obb"),
                    max_rotation_deg=float(args.max_rotation_deg),
                    max_shrink_fraction=float(args.max_shrink_fraction),
                    max_resamples=int(args.max_resamples),
                )
                transform = plan.transform
                crop = augmented.crop((region.x, region.y, region.right, region.bottom))
                transformed_crop = transform_region_image(crop, transform)
                augmented.paste(transformed_crop, (region.x, region.y))
                crop.close()
                transformed_crop.close()

                region_transform_count += 1
                resample_count += int(transform.resample_count)
                shrink_count += int(transform.scale < 1.0 - 1.0e-6)
                applied_angles.append(float(transform.angle_deg))
                applied_scales.append(float(transform.scale))
                image_record["regions"][region_key] = {
                    "angle_deg": float(transform.angle_deg),
                    "scale": float(transform.scale),
                    "translation": [float(transform.translate_x), float(transform.translate_y)],
                    "resample_count": int(transform.resample_count),
                }

                for source_annotation in region_annotations:
                    generated_annotations.append(
                        transform_annotation(
                            source_annotation,
                            region=region,
                            transform=transform,
                            image_id=next_image_id,
                            annotation_id=next_annotation_id,
                        )
                    )
                    next_annotation_id += 1

            if len(generated_annotations) != len(source_annotations):
                raise AssertionError(
                    f"Image {source_image_id}: transformed {len(generated_annotations)} of "
                    f"{len(source_annotations)} annotations"
                )

            output_name = generated_image_name(source_image, source_image_id, copy_index)
            output_path = images_directory / output_name
            augmented.save(output_path, format="PNG", optimize=False)
            generated_image = {
                key: value
                for key, value in source_image.items()
                if key not in {"id", "file_name", "dataset_origin"}
            }
            generated_image.update(
                {
                    "id": next_image_id,
                    "file_name": repository_relative_path(repository_root, output_path),
                    "dataset_origin": "human_obb_rotation_augmented",
                    "augmentation_source_image_id": source_image_id,
                    "augmentation_copy_index": copy_index,
                    "augmentation_seed": int(args.seed),
                    "region_transforms": image_record["regions"],
                }
            )
            output["images"].append(generated_image)
            output["annotations"].extend(generated_annotations)
            transform_records.append(image_record)

            if len(preflight) < int(args.preflight_count):
                preflight.append(
                    (
                        render_overlay(source_rgb, source_annotations),
                        render_overlay(augmented, generated_annotations),
                        transform_label(image_record),
                    )
                )
            augmented.close()
            next_image_id += 1
        source_rgb.close()

    annotations_directory = output_directory / "annotations"
    train_path = annotations_directory / "train.json"
    atomic_write_json(train_path, output, compact=True)
    write_jsonl(output_directory / "transforms.jsonl", transform_records)
    if preflight:
        preflight_directory = output_directory / "preflight"
        preflight_directory.mkdir(parents=True, exist_ok=True)
        sheet = render_contact_sheet(preflight)
        sheet.save(preflight_directory / "contact_sheet.jpg", format="JPEG", quality=92)
        sheet.close()
        for original, augmented, _label in preflight:
            original.close()
            augmented.close()

    provenance = {
        "artifact": "rotated_detector_obb_augmentation",
        "repository_root": str(repository_root),
        "source_annotations": str(annotations_path),
        "layout": str(layout_path),
        "seed": int(args.seed),
        "copies_per_image": int(args.copies_per_image),
        "include_originals": bool(args.include_originals),
        "max_rotation_deg": float(args.max_rotation_deg),
        "max_shrink_fraction": float(args.max_shrink_fraction),
        "max_resamples": int(args.max_resamples),
        "input": coco_counts(source),
        "output": coco_counts(output),
        "region_transform_count": region_transform_count,
        "resample_count_total": resample_count,
        "shrunken_region_transform_count": shrink_count,
        "angle_deg": stats(applied_angles),
        "scale": stats(applied_scales),
        "contract": {
            "fixed_composite": [320, 320],
            "transform_scope": "semantic_region_content_only",
            "obb_transform": "exact uniform scale + rotation + translation",
            "box_clipping": False,
            "angle_period_deg": 180,
            "fit_policy": "translate if possible; angle-dependent shrink and resampling otherwise",
        },
        "outputs": {
            "train": str(train_path),
            "transforms": str(output_directory / "transforms.jsonl"),
            "preflight": str(output_directory / "preflight" / "contact_sheet.jpg"),
        },
    }
    atomic_write_json(output_directory / "provenance.json", provenance, compact=False)
    print(json.dumps({"status": "completed", **provenance}, ensure_ascii=False, indent=2))
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if int(args.copies_per_image) < 1:
        raise ValueError("--copies-per-image must be positive")
    if not 0.0 <= float(args.max_rotation_deg) <= 90.0:
        raise ValueError("--max-rotation-deg must be within [0,90]")
    if not 0.0 <= float(args.max_shrink_fraction) < 1.0:
        raise ValueError("--max-shrink-fraction must be within [0,1)")
    if int(args.max_resamples) < 0:
        raise ValueError("--max-resamples must be non-negative")
    if int(args.preflight_count) < 0:
        raise ValueError("--preflight-count must be non-negative")


def ensure_inside_repository(repository_root: Path, output_directory: Path) -> None:
    try:
        relative = output_directory.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(f"Output must be inside repository root: {output_directory}") from error
    if not relative.parts:
        raise ValueError("Output directory must not be repository root")


def load_rotated_coco(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"Invalid COCO field {key}: {path}")
    categories = payload["categories"]
    if len(categories) != 1 or categories[0].get("name") != CATEGORY["name"]:
        raise ValueError(f"Unexpected category contract: {path}")
    image_ids = {int(image["id"]) for image in payload["images"]}
    for annotation in payload["annotations"]:
        if int(annotation["image_id"]) not in image_ids:
            raise ValueError(f"Annotation references missing image: {annotation.get('id')}")
        validate_obb(annotation.get("obb"), annotation_id=annotation.get("id"))
    return payload


def load_regions(layout: dict[str, Any]) -> dict[str, RegionRect]:
    composite = layout.get("composite")
    if not isinstance(composite, dict) or int(composite["width"]) != 320 or int(composite["height"]) != 320:
        raise ValueError("Expected 320x320 capture layout")
    source = layout.get("regions")
    if not isinstance(source, dict) or set(source) != set(REGION_KEYS):
        raise ValueError("Unexpected semantic region set")
    result: dict[str, RegionRect] = {}
    for key in REGION_KEYS:
        destination = source[key]["destination"]
        result[key] = RegionRect(
            key=key,
            x=int(destination["x"]),
            y=int(destination["y"]),
            width=int(destination["width"]),
            height=int(destination["height"]),
        )
    return result


def annotations_by_image(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        result[int(annotation["image_id"])].append(annotation)
    return dict(result)


def group_annotations_by_region(
    annotations: Sequence[dict[str, Any]],
    regions: dict[str, RegionRect],
) -> dict[str, list[dict[str, Any]]]:
    result: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        region = str(annotation.get("region", ""))
        if region not in regions:
            raise ValueError(f"Annotation {annotation.get('id')} has invalid region {region!r}")
        if not corners_inside_region(obb_corners(annotation["obb"]), regions[region]):
            raise ValueError(f"Annotation {annotation.get('id')} escapes region {region}")
        result[region].append(annotation)
    return dict(result)


def append_originals(
    source: dict[str, Any],
    output: dict[str, Any],
    *,
    next_image_id: int,
    next_annotation_id: int,
) -> tuple[int, int]:
    by_image = annotations_by_image(source)
    for source_image in sorted(source["images"], key=lambda item: int(item["id"])):
        source_id = int(source_image["id"])
        image = dict(source_image)
        image["id"] = next_image_id
        image["augmentation_source_image_id"] = source_id
        image["augmentation_copy_index"] = -1
        output["images"].append(image)
        for source_annotation in by_image[source_id]:
            annotation = dict(source_annotation)
            annotation["id"] = next_annotation_id
            annotation["image_id"] = next_image_id
            annotation["source_annotation_id"] = int(source_annotation["id"])
            output["annotations"].append(annotation)
            next_annotation_id += 1
        next_image_id += 1
    return next_image_id, next_annotation_id


def transform_annotation(
    source: dict[str, Any],
    *,
    region: RegionRect,
    transform: Any,
    image_id: int,
    annotation_id: int,
) -> dict[str, Any]:
    cx, cy, width, height, angle_deg = (float(value) for value in source["obb"])
    local_center = (cx - region.x, cy - region.y)
    transformed_center = scale_rotate_point(
        local_center[0],
        local_center[1],
        center_x=float(transform.center_x),
        center_y=float(transform.center_y),
        angle_deg=float(transform.angle_deg),
        scale=float(transform.scale),
    )
    transformed_obb = [
        transformed_center[0] + float(transform.translate_x) + region.x,
        transformed_center[1] + float(transform.translate_y) + region.y,
        width * float(transform.scale),
        height * float(transform.scale),
        normalize_angle_180(angle_deg + float(transform.angle_deg)),
    ]
    polygon = obb_corners(transformed_obb)
    if not corners_inside_region(polygon, region):
        raise AssertionError(
            f"Transformed OBB escaped {region.key}: source={source['obb']} transformed={transformed_obb}"
        )
    generated = {
        key: value
        for key, value in source.items()
        if key
        not in {
            "id",
            "image_id",
            "bbox",
            "obb",
            "segmentation",
            "area",
            "augmentation",
        }
    }
    generated.update(
        {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": polygon_hbb(polygon),
            "obb": transformed_obb,
            "segmentation": [[coordinate for point in polygon for coordinate in point]],
            "area": transformed_obb[2] * transformed_obb[3],
            "iscrowd": int(source.get("iscrowd", 0)),
            "source_annotation_id": int(source["id"]),
            "augmentation": {
                "angle_deg": float(transform.angle_deg),
                "scale": float(transform.scale),
                "translation": [float(transform.translate_x), float(transform.translate_y)],
            },
        }
    )
    return generated


def corners_inside_region(
    corners: Sequence[tuple[float, float]], region: RegionRect
) -> bool:
    return all(
        region.x - _EPSILON <= x <= region.right + _EPSILON
        and region.y - _EPSILON <= y <= region.bottom + _EPSILON
        for x, y in corners
    )


def validate_obb(value: Any, *, annotation_id: Any) -> None:
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError(f"Annotation {annotation_id} has no valid obb")
    values = [float(item) for item in value]
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"Annotation {annotation_id} has non-finite obb")
    if values[2] <= 1.0 or values[3] <= 1.0:
        raise ValueError(f"Annotation {annotation_id} has non-positive OBB size")


def resolve_repository_image(repository_root: Path, file_name: str) -> Path:
    pure = PurePosixPath(file_name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe repository-relative path: {file_name}")
    path = repository_root.joinpath(*pure.parts)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def repository_relative_path(repository_root: Path, path: Path) -> str:
    return PurePosixPath(path.resolve().relative_to(repository_root).as_posix()).as_posix()


def generated_image_name(
    source_image: dict[str, Any], source_image_id: int, copy_index: int
) -> str:
    source_name = Path(str(source_image.get("file_name", "image"))).stem
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in source_name
    )
    return f"{safe_name}__src{source_image_id:06d}__obbrot{copy_index:02d}.png"


def render_overlay(image: Image.Image, annotations: Sequence[dict[str, Any]]) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    for annotation in annotations:
        polygon = obb_corners(annotation["obb"])
        draw.line([*polygon, polygon[0]], fill=(0, 255, 0), width=2)
        cx, cy, _width, _height, _angle = (float(value) for value in annotation["obb"])
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(255, 220, 0))
    return rendered


def render_contact_sheet(
    pairs: Sequence[tuple[Image.Image, Image.Image, str]]
) -> Image.Image:
    pair_width = 640
    pair_height = 348
    columns = 2 if len(pairs) > 1 else 1
    rows = math.ceil(len(pairs) / columns)
    sheet = Image.new("RGB", (pair_width * columns, pair_height * rows), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    for index, (original, augmented, label) in enumerate(pairs):
        column = index % columns
        row = index // columns
        x = column * pair_width
        y = row * pair_height
        sheet.paste(original, (x, y))
        sheet.paste(augmented, (x + 320, y))
        draw.text((x + 4, y + 323), f"human -> aug | {label}", fill=(255, 255, 255))
    return sheet


def transform_label(record: dict[str, Any]) -> str:
    parts = []
    for key in REGION_KEYS:
        item = record["regions"].get(key)
        if item is None:
            continue
        parts.append(
            f"{key}:{float(item['angle_deg']):+.1f}deg "
            f"s={float(item['scale']):.3f} r={int(item['resample_count'])}"
        )
    return " | ".join(parts)


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def coco_counts(payload: dict[str, Any]) -> dict[str, int]:
    return {"images": len(payload["images"]), "annotations": len(payload["annotations"])}


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
