from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from PIL import Image

from .geometry import (
    annotation_matches_crop,
    has_exact_aspect,
    retained_area_ratio,
    transform_bbox,
)
from .layout import COMPOSITE_SIZE, PADDING_RGB, REGION_SPECS
from .models import (
    Rect,
    RegionCompositeStats,
    RegionSelection,
    TransformedAnnotation,
)


DEFAULT_MIN_RETAINED_AREA_RATIO = 0.6


@dataclass(frozen=True)
class CompositeResult:
    image: Image.Image
    annotations: list[TransformedAnnotation]
    stats_by_region: dict[str, RegionCompositeStats]


def compose_capture_image(
    source_image: Image.Image,
    selections: Mapping[str, RegionSelection],
    source_annotations: Iterable[dict[str, Any]],
    *,
    annotation_selection_policy: str = "center",
    min_retained_area_ratio: float = DEFAULT_MIN_RETAINED_AREA_RATIO,
) -> CompositeResult:
    if not selections:
        raise ValueError("At least one capture region must be selected")
    _validate_min_retained_area_ratio(min_retained_area_ratio)

    source_rgb = source_image.convert("RGB")
    source_size = source_rgb.size
    image_bounds = Rect(0, 0, *source_size)
    composite = Image.new("RGB", COMPOSITE_SIZE, PADDING_RGB)

    for region_key, selection in selections.items():
        _validate_selection(region_key, selection, image_bounds, source_size)
        spec = REGION_SPECS[region_key]
        crop_image = source_rgb.crop(selection.crop.to_pillow_box())
        rotated_image = _rotate_image_clockwise(
            crop_image,
            selection.rotation_clockwise,
        )
        destination_width = int(spec.destination.width)
        destination_height = int(spec.destination.height)
        resized_image = rotated_image.resize(
            (destination_width, destination_height),
            resample=Image.Resampling.LANCZOS,
        )
        composite.paste(
            resized_image,
            (int(spec.destination.x), int(spec.destination.y)),
        )

    transformed_annotations, stats_by_region = transform_capture_annotations(
        source_size,
        selections,
        source_annotations,
        annotation_selection_policy=annotation_selection_policy,
        min_retained_area_ratio=min_retained_area_ratio,
    )
    return CompositeResult(
        image=composite,
        annotations=transformed_annotations,
        stats_by_region=stats_by_region,
    )


def transform_capture_annotations(
    source_size: tuple[int, int],
    selections: Mapping[str, RegionSelection],
    source_annotations: Iterable[dict[str, Any]],
    *,
    annotation_selection_policy: str = "center",
    min_retained_area_ratio: float = DEFAULT_MIN_RETAINED_AREA_RATIO,
) -> tuple[list[TransformedAnnotation], dict[str, RegionCompositeStats]]:
    if not selections:
        raise ValueError("At least one capture region must be selected")
    _validate_min_retained_area_ratio(min_retained_area_ratio)

    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid source image size: {source_size}")
    image_bounds = Rect(0, 0, source_width, source_height)
    annotations = list(source_annotations)
    transformed_annotations: list[TransformedAnnotation] = []
    stats_by_region: dict[str, RegionCompositeStats] = {}

    for region_key, selection in selections.items():
        _validate_selection(region_key, selection, image_bounds, source_size)
        spec = REGION_SPECS[region_key]
        retained = 0
        clipped_count = 0

        for annotation in annotations:
            annotation_id = _annotation_id(annotation)
            source_bbox = Rect.from_coco_bbox(
                annotation.get("bbox"),
                context=f"annotation {annotation_id}",
            )
            visible_bbox = source_bbox.intersection(image_bounds)
            if visible_bbox is None:
                continue
            if not annotation_matches_crop(
                visible_bbox,
                selection.crop,
                annotation_selection_policy,
            ):
                continue
            clipped_to_crop = visible_bbox.intersection(selection.crop)
            if clipped_to_crop is None:
                continue
            if retained_area_ratio(source_bbox, selection.crop) <= min_retained_area_ratio:
                continue

            output_bbox = transform_bbox(
                visible_bbox,
                selection.crop,
                selection.rotation_clockwise,
                spec.destination,
            )
            if output_bbox is None or output_bbox.area <= 0:
                continue
            if clipped_to_crop != visible_bbox:
                clipped_count += 1
            transformed_annotations.append(
                TransformedAnnotation(
                    source_annotation_id=annotation_id,
                    region_key=region_key,
                    bbox=output_bbox,
                    iscrowd=int(annotation.get("iscrowd", 0)),
                )
            )
            retained += 1

        stats_by_region[region_key] = RegionCompositeStats(
            retained_annotations=retained,
            clipped_annotations=clipped_count,
        )

    return transformed_annotations, stats_by_region


def _validate_selection(
    region_key: str,
    selection: RegionSelection,
    image_bounds: Rect,
    source_size: tuple[int, int],
) -> None:
    if region_key not in REGION_SPECS:
        raise ValueError(f"Unknown capture region: {region_key}")
    if selection.region_key != region_key:
        raise ValueError(
            f"Selection key mismatch: mapping={region_key}, "
            f"selection={selection.region_key}"
        )
    if not image_bounds.contains_rect(selection.crop):
        raise ValueError(
            f"Selection is outside source image bounds: {selection.crop}, "
            f"image={source_size}"
        )
    spec = REGION_SPECS[region_key]
    expected_aspect = spec.source_aspect_for_rotation(selection.rotation_clockwise)
    if not has_exact_aspect(selection.crop, expected_aspect):
        raise ValueError(
            f"Selection does not match required source aspect {expected_aspect}: "
            f"{selection.crop}"
        )


def _validate_min_retained_area_ratio(value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(
            "min_retained_area_ratio must be at least zero and less than one: "
            f"{value}"
        )


def _rotate_image_clockwise(image: Image.Image, rotation_clockwise: int) -> Image.Image:
    if rotation_clockwise == 0:
        return image
    if rotation_clockwise == 90:
        return image.transpose(Image.Transpose.ROTATE_270)
    if rotation_clockwise == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if rotation_clockwise == 270:
        return image.transpose(Image.Transpose.ROTATE_90)
    raise ValueError(f"Unsupported clockwise rotation: {rotation_clockwise}")


def _annotation_id(annotation: dict[str, Any]) -> int:
    try:
        return int(annotation["id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid annotation id: {annotation.get('id')!r}") from error
