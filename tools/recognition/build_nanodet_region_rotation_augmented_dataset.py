from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw


CATEGORY = {
    "id": 1,
    "name": "mahjong_tile",
    "supercategory": "mahjong_tile",
}
REGION_KEYS = ("completed_hand", "dora_indicators", "melds")
_EPSILON = 1e-6


@dataclass(frozen=True)
class RegionRect:
    key: str
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class RegionTransform:
    angle_deg: float
    scale: float
    center_x: float
    center_y: float
    translate_x: float
    translate_y: float
    resample_count: int


@dataclass(frozen=True)
class TransformPlan:
    transform: RegionTransform
    transformed_polygons: tuple[tuple[tuple[float, float], ...], ...]


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed-layout NanoDet augmentation images by scaling and rotating "
            "photographic content independently inside each semantic capture region. "
            "The 320x320 three-region layout never moves. GT polygon corners receive "
            "the identical affine transform and the detector bbox becomes their "
            "enclosing AABB."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--annotations",
        type=Path,
        help=(
            "Input fixed-layout COCO annotations. Defaults to the unique real-capture "
            "training partition from the current NanoDet fine-tune dataset."
        ),
    )
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--copies-per-image", type=int, default=1)
    parser.add_argument(
        "--max-rotation-deg",
        type=float,
        default=12.0,
        help="Sample each populated semantic region uniformly from [-value,+value].",
    )
    parser.add_argument(
        "--max-shrink-fraction",
        type=float,
        default=0.10,
        help=(
            "Quadratic angle-dependent shrink at |angle| == max rotation. "
            "Default 0.10 means scale 1.0 at 0deg and 0.90 at max rotation."
        ),
    )
    parser.add_argument(
        "--max-resamples",
        type=int,
        default=32,
        help="Retry the angle when the rotated GT union cannot fit inside its region.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-images",
        type=int,
        help="Optional deterministic prefix limit for preflight/debug generation.",
    )
    parser.add_argument(
        "--preflight-count",
        type=int,
        default=12,
        help="Number of source/augmented pairs rendered into preflight/contact_sheet.jpg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    annotations_path = (
        args.annotations.resolve()
        if args.annotations is not None
        else repository_root
        / ".local"
        / "recognition"
        / "nanodet_capture_finetune_dataset"
        / "annotations"
        / "instances_real_train.json"
    )
    layout_path = (
        args.layout.resolve()
        if args.layout is not None
        else repository_root / "tools" / "recognition" / "capture_layout.v1.json"
    )
    output_directory = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else repository_root
        / ".local"
        / "recognition"
        / "nanodet_region_rotation_augmented_dataset"
    )

    validate_cli_args(args)
    ensure_output_is_inside_repository(repository_root, output_directory)
    if not annotations_path.is_file():
        raise FileNotFoundError(annotations_path)
    if not layout_path.is_file():
        raise FileNotFoundError(layout_path)

    if output_directory.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists; pass --overwrite to replace it: {output_directory}"
            )
        shutil.rmtree(output_directory)

    payload = load_coco(annotations_path)
    layout = load_json(layout_path)
    regions, composite_size = load_layout(layout)
    selected_images = sorted(payload["images"], key=lambda image: int(image["id"]))
    if args.limit_images is not None:
        selected_images = selected_images[: args.limit_images]
    if not selected_images:
        raise ValueError("No input images selected")

    annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    output_payload = empty_coco(
        "Fixed-layout semantic-region rotation augmentation for NanoDet"
    )
    images_directory = output_directory / "images"
    images_directory.mkdir(parents=True, exist_ok=True)

    next_image_id = 1
    next_annotation_id = 1
    transform_records: list[dict[str, Any]] = []
    preflight_pairs: list[tuple[Image.Image, Image.Image, str]] = []
    translation_count = 0
    resample_total = 0
    region_transform_count = 0
    applied_angles: list[float] = []
    applied_scales: list[float] = []

    for source_image in selected_images:
        source_image_id = int(source_image["id"])
        source_path = repository_image_path(repository_root, source_image.get("file_name"))
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with Image.open(source_path) as opened:
            source_rgb = opened.convert("RGB")
        if source_rgb.size != composite_size:
            raise ValueError(
                f"Image {source_image_id} has size {source_rgb.size}; expected {composite_size}"
            )

        source_annotations = sorted(
            annotations_by_image[source_image_id], key=lambda annotation: int(annotation["id"])
        )
        grouped = group_annotations_by_region(source_annotations, regions)

        for copy_index in range(args.copies_per_image):
            augmented = source_rgb.copy()
            generated_annotations: list[dict[str, Any]] = []
            image_transform_record: dict[str, Any] = {
                "source_image_id": source_image_id,
                "copy_index": copy_index,
                "regions": {},
            }

            for region_key in REGION_KEYS:
                region = regions[region_key]
                region_annotations = grouped.get(region_key, [])
                if not region_annotations:
                    continue

                polygons_global = [
                    annotation_polygon(annotation) for annotation in region_annotations
                ]
                polygons_local = [
                    tuple((x - region.x, y - region.y) for x, y in polygon)
                    for polygon in polygons_global
                ]
                plan = plan_region_transform(
                    polygons_local,
                    region_width=region.width,
                    region_height=region.height,
                    seed=args.seed,
                    sample_key=(source_image.get("file_name"), copy_index, region_key),
                    max_rotation_deg=args.max_rotation_deg,
                    max_shrink_fraction=args.max_shrink_fraction,
                    max_resamples=args.max_resamples,
                )
                transform = plan.transform
                region_crop = augmented.crop(
                    (region.x, region.y, region.right, region.bottom)
                )
                transformed_crop = transform_region_image(region_crop, transform)
                augmented.paste(transformed_crop, (region.x, region.y))

                if abs(transform.translate_x) > _EPSILON or abs(transform.translate_y) > _EPSILON:
                    translation_count += 1
                resample_total += transform.resample_count
                region_transform_count += 1
                applied_angles.append(transform.angle_deg)
                applied_scales.append(transform.scale)
                image_transform_record["regions"][region_key] = {
                    "angle_deg": transform.angle_deg,
                    "scale": transform.scale,
                    "center": [transform.center_x, transform.center_y],
                    "translation": [transform.translate_x, transform.translate_y],
                    "resample_count": transform.resample_count,
                }

                for source_annotation, transformed_local in zip(
                    region_annotations, plan.transformed_polygons, strict=True
                ):
                    transformed_global = tuple(
                        (x + region.x, y + region.y) for x, y in transformed_local
                    )
                    assert_polygon_inside_region(transformed_global, region)
                    generated_annotation = transformed_annotation(
                        source_annotation,
                        transformed_global,
                        image_id=next_image_id,
                        annotation_id=next_annotation_id,
                        region_key=region_key,
                        transform=transform,
                    )
                    generated_annotations.append(generated_annotation)
                    next_annotation_id += 1

            # Preserve annotations in empty/untransformed regions only if grouping found them;
            # every annotation is expected to belong to one populated region above.
            if len(generated_annotations) != len(source_annotations):
                raise AssertionError(
                    f"Image {source_image_id}: transformed {len(generated_annotations)} of "
                    f"{len(source_annotations)} annotations"
                )

            output_name = generated_image_name(source_image, source_image_id, copy_index)
            output_path = images_directory / output_name
            augmented.save(output_path, format="PNG", optimize=False)
            relative_output_name = repository_relative_path(repository_root, output_path)

            generated_image = {
                key: value
                for key, value in source_image.items()
                if key not in {"id", "file_name"}
            }
            generated_image.update(
                {
                    "id": next_image_id,
                    "file_name": relative_output_name,
                    "source_image_id": source_image_id,
                    "region_rotation_copy_index": copy_index,
                    "region_rotation_seed": args.seed,
                    "region_transforms": image_transform_record["regions"],
                    "dataset_origin": "region_rotation_augmented",
                }
            )
            output_payload["images"].append(generated_image)
            output_payload["annotations"].extend(generated_annotations)
            transform_records.append(image_transform_record)

            if len(preflight_pairs) < args.preflight_count:
                original_overlay = render_overlay(source_rgb, source_annotations, regions)
                augmented_overlay = render_overlay(augmented, generated_annotations, regions)
                label = transform_label(image_transform_record)
                preflight_pairs.append((original_overlay, augmented_overlay, label))

            next_image_id += 1

    annotations_directory = output_directory / "annotations"
    annotations_path_out = annotations_directory / "instances_train.json"
    atomic_write_json(annotations_path_out, output_payload, compact=True)
    write_transform_jsonl(output_directory / "transforms.jsonl", transform_records)
    if preflight_pairs:
        preflight_directory = output_directory / "preflight"
        preflight_directory.mkdir(parents=True, exist_ok=True)
        render_contact_sheet(preflight_pairs).save(
            preflight_directory / "contact_sheet.jpg",
            format="JPEG",
            quality=92,
        )

    provenance = {
        "artifact": "nanodet_region_rotation_augmented_dataset",
        "repository_root": str(repository_root),
        "source_annotations": str(annotations_path),
        "layout": str(layout_path),
        "layout_id": layout.get("id"),
        "fixed_composite_size": list(composite_size),
        "fixed_region_destinations": {
            key: {
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height,
            }
            for key, region in regions.items()
        },
        "contract": {
            "transform_scope": "semantic_region_content_only",
            "bbox_after_rotation": "axis_aligned_enclosure_of_scaled_rotated_polygon",
            "angle_scale_policy": "quadratic_shrink_toward_max_rotation",
            "fit_policy": "minimal_whole_region_translation_then_resample_if_impossible",
            "box_clipping": False,
            "border_sampling": "black_fill_then_bilinear_affine",
        },
        "seed": args.seed,
        "copies_per_image": args.copies_per_image,
        "max_rotation_deg": args.max_rotation_deg,
        "max_shrink_fraction": args.max_shrink_fraction,
        "max_resamples": args.max_resamples,
        "input_images_selected": len(selected_images),
        "output_images": len(output_payload["images"]),
        "output_annotations": len(output_payload["annotations"]),
        "region_transform_count": region_transform_count,
        "translation_to_fit_count": translation_count,
        "resample_count_total": resample_total,
        "applied_angle_deg": angle_stats(applied_angles),
        "applied_scale": value_stats(applied_scales),
        "outputs": {
            "annotations": str(annotations_path_out),
            "transforms": str(output_directory / "transforms.jsonl"),
            "contact_sheet": str(output_directory / "preflight" / "contact_sheet.jpg"),
        },
    }
    provenance_path = output_directory / "provenance.json"
    atomic_write_json(provenance_path, provenance, compact=False)

    print(
        json.dumps(
            {
                "status": "completed",
                "output_directory": str(output_directory),
                "images": provenance["output_images"],
                "annotations": provenance["output_annotations"],
                "region_transforms": region_transform_count,
                "translation_to_fit": translation_count,
                "resamples": resample_total,
                "angle_deg": provenance["applied_angle_deg"],
                "scale": provenance["applied_scale"],
                "preflight": provenance["outputs"]["contact_sheet"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.copies_per_image < 1:
        raise ValueError("--copies-per-image must be positive")
    if not math.isfinite(args.max_rotation_deg) or not 0.0 <= args.max_rotation_deg <= 45.0:
        raise ValueError("--max-rotation-deg must be finite and within [0,45]")
    if (
        not math.isfinite(args.max_shrink_fraction)
        or not 0.0 <= args.max_shrink_fraction < 1.0
    ):
        raise ValueError("--max-shrink-fraction must be finite and within [0,1)")
    if args.max_resamples < 0:
        raise ValueError("--max-resamples must not be negative")
    if args.limit_images is not None and args.limit_images < 1:
        raise ValueError("--limit-images must be positive")
    if args.preflight_count < 0:
        raise ValueError("--preflight-count must not be negative")


def ensure_output_is_inside_repository(repository_root: Path, output_directory: Path) -> None:
    try:
        relative = output_directory.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(
            f"Output directory must be inside repository root: {output_directory}"
        ) from error
    if not relative.parts:
        raise ValueError("Output directory must not be the repository root")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_coco(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"Invalid COCO field {key}: {path}")
    categories = payload["categories"]
    if len(categories) != 1 or categories[0].get("name") != CATEGORY["name"]:
        raise ValueError(f"Unexpected COCO category: {path}")
    image_ids = {int(image["id"]) for image in payload["images"]}
    if len(image_ids) != len(payload["images"]):
        raise ValueError(f"Duplicate image IDs: {path}")
    for annotation in payload["annotations"]:
        if int(annotation["image_id"]) not in image_ids:
            raise ValueError(
                f"Annotation {annotation.get('id')} references missing image "
                f"{annotation.get('image_id')}"
            )
        annotation_polygon(annotation)
    return payload


def load_layout(layout: dict[str, Any]) -> tuple[dict[str, RegionRect], tuple[int, int]]:
    composite = layout.get("composite")
    if not isinstance(composite, dict):
        raise ValueError("Layout has no composite object")
    width = int(composite["width"])
    height = int(composite["height"])
    if width != 320 or height != 320:
        raise ValueError(f"Expected 320x320 deployment composite, got {width}x{height}")
    regions_document = layout.get("regions")
    if not isinstance(regions_document, dict) or set(regions_document) != set(REGION_KEYS):
        raise ValueError("Unexpected semantic region set")
    regions: dict[str, RegionRect] = {}
    for key in REGION_KEYS:
        destination = regions_document[key]["destination"]
        region = RegionRect(
            key=key,
            x=int(destination["x"]),
            y=int(destination["y"]),
            width=int(destination["width"]),
            height=int(destination["height"]),
        )
        if (
            region.x < 0
            or region.y < 0
            or region.width <= 0
            or region.height <= 0
            or region.right > width
            or region.bottom > height
        ):
            raise ValueError(f"Invalid destination for {key}: {destination}")
        regions[key] = region
    return regions, (width, height)


def empty_coco(description: str) -> dict[str, Any]:
    return {
        "info": {"description": description},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [CATEGORY],
    }


def repository_image_path(repository_root: Path, file_name: Any) -> Path:
    if not isinstance(file_name, str) or not file_name:
        raise ValueError(f"Invalid image file_name: {file_name!r}")
    normalized = PurePosixPath(file_name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Image file_name must be repository-relative: {file_name!r}")
    return repository_root.joinpath(*normalized.parts)


def repository_relative_path(repository_root: Path, path: Path) -> str:
    return PurePosixPath(path.resolve().relative_to(repository_root).as_posix()).as_posix()


def generated_image_name(
    source_image: dict[str, Any], source_image_id: int, copy_index: int
) -> str:
    source_name = Path(str(source_image.get("file_name", "image"))).stem
    safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in source_name)
    return f"{safe_name}__src{source_image_id:06d}__rot{copy_index:02d}.png"


def group_annotations_by_region(
    annotations: Sequence[dict[str, Any]], regions: dict[str, RegionRect]
) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        explicit = annotation.get("region")
        if explicit is not None:
            if explicit not in regions:
                raise ValueError(
                    f"Annotation {annotation.get('id')} has unknown region {explicit!r}"
                )
            region_key = str(explicit)
        else:
            polygon = annotation_polygon(annotation)
            candidates = [
                key
                for key, region in regions.items()
                if polygon_inside_region(polygon, region)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Annotation {annotation.get('id')} cannot be assigned to exactly one "
                    f"fixed semantic region: candidates={candidates}"
                )
            region_key = candidates[0]
        polygon = annotation_polygon(annotation)
        if not polygon_inside_region(polygon, regions[region_key]):
            raise ValueError(
                f"Annotation {annotation.get('id')} is not fully inside region {region_key}"
            )
        grouped[region_key].append(annotation)
    return dict(grouped)


def annotation_polygon(annotation: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    segmentation = annotation.get("segmentation")
    if (
        isinstance(segmentation, list)
        and segmentation
        and isinstance(segmentation[0], list)
        and len(segmentation[0]) >= 8
        and len(segmentation[0]) % 2 == 0
    ):
        values = segmentation[0]
        try:
            return tuple(
                (float(values[index]), float(values[index + 1]))
                for index in range(0, len(values), 2)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Annotation {annotation.get('id')} has invalid segmentation"
            ) from error

    bbox = annotation.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(
            f"Annotation {annotation.get('id')} has neither polygon segmentation nor bbox"
        )
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Annotation {annotation.get('id')} has invalid bbox") from error
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"Annotation {annotation.get('id')} has non-positive bbox")
    return (
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    )


def polygon_inside_region(
    polygon: Sequence[tuple[float, float]], region: RegionRect
) -> bool:
    return all(
        region.x - _EPSILON <= x <= region.right + _EPSILON
        and region.y - _EPSILON <= y <= region.bottom + _EPSILON
        for x, y in polygon
    )


def assert_polygon_inside_region(
    polygon: Sequence[tuple[float, float]], region: RegionRect
) -> None:
    if not polygon_inside_region(polygon, region):
        raise AssertionError(f"Transformed polygon escaped {region.key}: {polygon}")


def plan_region_transform(
    polygons: Sequence[Sequence[tuple[float, float]]],
    *,
    region_width: int,
    region_height: int,
    seed: int,
    sample_key: Sequence[Any],
    max_rotation_deg: float,
    max_shrink_fraction: float = 0.10,
    max_resamples: int = 32,
) -> TransformPlan:
    if not polygons:
        raise ValueError("At least one polygon is required")
    points = [point for polygon in polygons for point in polygon]
    min_x, min_y, max_x, max_y = point_bounds(points)
    if (
        min_x < -_EPSILON
        or min_y < -_EPSILON
        or max_x > region_width + _EPSILON
        or max_y > region_height + _EPSILON
    ):
        raise ValueError("Source GT union is already outside the semantic region")
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    for attempt in range(max_resamples + 1):
        angle_deg = deterministic_uniform(
            -max_rotation_deg,
            max_rotation_deg,
            seed=seed,
            sample_key=(*sample_key, attempt),
        )
        scale = angle_dependent_scale(
            angle_deg,
            max_rotation_deg=max_rotation_deg,
            max_shrink_fraction=max_shrink_fraction,
        )
        rotated = tuple(
            tuple(
                scale_rotate_point(
                    x,
                    y,
                    center_x=center_x,
                    center_y=center_y,
                    angle_deg=angle_deg,
                    scale=scale,
                )
                for x, y in polygon
            )
            for polygon in polygons
        )
        translation = minimal_translation_to_fit(
            [point for polygon in rotated for point in polygon],
            width=region_width,
            height=region_height,
        )
        if translation is None:
            continue
        translate_x, translate_y = translation
        transformed = tuple(
            tuple((x + translate_x, y + translate_y) for x, y in polygon)
            for polygon in rotated
        )
        return TransformPlan(
            transform=RegionTransform(
                angle_deg=angle_deg,
                scale=scale,
                center_x=center_x,
                center_y=center_y,
                translate_x=translate_x,
                translate_y=translate_y,
                resample_count=attempt,
            ),
            transformed_polygons=transformed,
        )

    # Angle 0 is always a valid final fallback when the source labels were valid.
    translation = minimal_translation_to_fit(points, width=region_width, height=region_height)
    if translation is None:
        raise AssertionError("Valid source GT union unexpectedly cannot fit at zero rotation")
    translate_x, translate_y = translation
    return TransformPlan(
        transform=RegionTransform(
            angle_deg=0.0,
            scale=1.0,
            center_x=center_x,
            center_y=center_y,
            translate_x=translate_x,
            translate_y=translate_y,
            resample_count=max_resamples + 1,
        ),
        transformed_polygons=tuple(
            tuple((x + translate_x, y + translate_y) for x, y in polygon)
            for polygon in polygons
        ),
    )


def deterministic_uniform(
    low: float, high: float, *, seed: int, sample_key: Sequence[Any]
) -> float:
    if low == high:
        return float(low)
    material = json.dumps(
        [seed, *sample_key], ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    numerator = int.from_bytes(digest[:8], "big")
    unit = numerator / float((1 << 64) - 1)
    return low + (high - low) * unit


def angle_dependent_scale(
    angle_deg: float,
    *,
    max_rotation_deg: float,
    max_shrink_fraction: float,
) -> float:
    if max_rotation_deg <= _EPSILON or max_shrink_fraction <= _EPSILON:
        return 1.0
    normalized = min(abs(angle_deg) / max_rotation_deg, 1.0)
    return 1.0 - max_shrink_fraction * normalized * normalized


def scale_rotate_point(
    x: float,
    y: float,
    *,
    center_x: float,
    center_y: float,
    angle_deg: float,
    scale: float,
) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    relative_x = scale * (x - center_x)
    relative_y = scale * (y - center_y)
    return (
        center_x + cosine * relative_x - sine * relative_y,
        center_y + sine * relative_x + cosine * relative_y,
    )


def rotate_point(
    x: float,
    y: float,
    *,
    center_x: float,
    center_y: float,
    angle_deg: float,
) -> tuple[float, float]:
    return scale_rotate_point(
        x,
        y,
        center_x=center_x,
        center_y=center_y,
        angle_deg=angle_deg,
        scale=1.0,
    )


def point_bounds(
    points: Iterable[tuple[float, float]],
) -> tuple[float, float, float, float]:
    materialized = list(points)
    if not materialized:
        raise ValueError("No points")
    return (
        min(x for x, _y in materialized),
        min(y for _x, y in materialized),
        max(x for x, _y in materialized),
        max(y for _x, y in materialized),
    )


def minimal_translation_to_fit(
    points: Sequence[tuple[float, float]], *, width: int, height: int
) -> tuple[float, float] | None:
    min_x, min_y, max_x, max_y = point_bounds(points)
    x_low = -min_x
    x_high = float(width) - max_x
    y_low = -min_y
    y_high = float(height) - max_y
    if x_low > x_high + _EPSILON or y_low > y_high + _EPSILON:
        return None
    return (
        closest_to_zero(x_low, x_high),
        closest_to_zero(y_low, y_high),
    )


def closest_to_zero(low: float, high: float) -> float:
    if low <= 0.0 <= high:
        return 0.0
    if low > 0.0:
        return low
    return high


def transform_region_image(image: Image.Image, transform: RegionTransform) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size

    radians = math.radians(transform.angle_deg)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    inverse_scale = 1.0 / transform.scale
    cx = transform.center_x
    cy = transform.center_y
    dx = transform.translate_x
    dy = transform.translate_y

    # Pillow AFFINE expects inverse mapping: output pixel -> source pixel.
    # Forward geometry is isotropic scale + rotation around the GT-union center,
    # followed by the whole-region fit translation. Out-of-bounds source pixels
    # are intentionally filled black rather than mirrored/repeated.
    affine = (
        inverse_scale * cosine,
        inverse_scale * sine,
        cx - inverse_scale * cosine * (dx + cx) - inverse_scale * sine * (dy + cy),
        -inverse_scale * sine,
        inverse_scale * cosine,
        cy + inverse_scale * sine * (dx + cx) - inverse_scale * cosine * (dy + cy),
    )
    return image.transform(
        (width, height),
        Image.Transform.AFFINE,
        affine,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def transformed_annotation(
    source: dict[str, Any],
    polygon: Sequence[tuple[float, float]],
    *,
    image_id: int,
    annotation_id: int,
    region_key: str,
    transform: RegionTransform,
) -> dict[str, Any]:
    min_x, min_y, max_x, max_y = point_bounds(polygon)
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"Transformed annotation {source.get('id')} has invalid bbox")
    generated = {
        key: value
        for key, value in source.items()
        if key
        not in {
            "id",
            "image_id",
            "category_id",
            "bbox",
            "area",
            "segmentation",
            "region_rotation",
        }
    }
    generated.update(
        {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": CATEGORY["id"],
            "bbox": [min_x, min_y, width, height],
            "area": width * height,
            "iscrowd": int(source.get("iscrowd", 0)),
            "segmentation": [[coordinate for point in polygon for coordinate in point]],
            "region": region_key,
            "source_annotation_id": int(source["id"]),
            "region_rotation": {
                "angle_deg": transform.angle_deg,
                "scale": transform.scale,
                "translation": [transform.translate_x, transform.translate_y],
            },
        }
    )
    return generated


def render_overlay(
    image: Image.Image,
    annotations: Sequence[dict[str, Any]],
    regions: dict[str, RegionRect],
) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    for region in regions.values():
        draw.rectangle(
            (region.x, region.y, region.right - 1, region.bottom - 1),
            outline=(255, 255, 255),
            width=1,
        )
    for annotation in annotations:
        polygon = annotation_polygon(annotation)
        draw.line([*polygon, polygon[0]], fill=(0, 255, 0), width=2)
        bbox = annotation.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, width, height = (float(value) for value in bbox)
            draw.rectangle((x, y, x + width, y + height), outline=(255, 215, 0), width=1)
    return rendered


def transform_label(record: dict[str, Any]) -> str:
    pieces = []
    for key in REGION_KEYS:
        region = record["regions"].get(key)
        if region is None:
            continue
        pieces.append(
            f"{key}: {region['angle_deg']:+.1f}deg s={region['scale']:.3f} "
            f"t=({region['translation'][0]:+.1f},{region['translation'][1]:+.1f})"
        )
    return " | ".join(pieces)


def render_contact_sheet(
    pairs: Sequence[tuple[Image.Image, Image.Image, str]]
) -> Image.Image:
    pair_width = 640
    pair_height = 348
    columns = 2 if len(pairs) > 1 else 1
    rows = math.ceil(len(pairs) / columns)
    sheet = Image.new("RGB", (pair_width * columns, pair_height * rows), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    for index, (original, augmented, label) in enumerate(pairs):
        column = index % columns
        row = index // columns
        x = column * pair_width
        y = row * pair_height
        sheet.paste(original, (x, y))
        sheet.paste(augmented, (x + 320, y))
        draw.text((x + 4, y + 322), f"orig -> aug | {label}", fill=(255, 255, 255))
    return sheet


def write_transform_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
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


def angle_stats(angles: Sequence[float]) -> dict[str, float | None]:
    if not angles:
        return {"min": None, "max": None, "mean": None, "mean_abs": None}
    return {
        "min": min(angles),
        "max": max(angles),
        "mean": sum(angles) / len(angles),
        "mean_abs": sum(abs(angle) for angle in angles) / len(angles),
    }


def value_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


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
    main()
