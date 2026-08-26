from __future__ import annotations

import math

from .models import Rect


ANNOTATION_SELECTION_POLICIES = ("center", "contained", "intersect")


def constrain_drag_rect(
    anchor: tuple[int, int],
    pointer: tuple[int, int],
    image_size: tuple[int, int],
    aspect: tuple[int, int],
) -> Rect | None:
    """Return an in-image integer rectangle with an exact integer aspect ratio.

    The anchor remains one corner. Width and height are integer multiples of the
    requested aspect components, which guarantees uniform destination resizing.
    """

    image_width, image_height = image_size
    aspect_width, aspect_height = aspect
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size: {image_size}")
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"Invalid aspect ratio: {aspect}")

    anchor_x = min(max(int(anchor[0]), 0), image_width)
    anchor_y = min(max(int(anchor[1]), 0), image_height)
    pointer_x = min(max(int(pointer[0]), 0), image_width)
    pointer_y = min(max(int(pointer[1]), 0), image_height)

    direction_x = 1 if pointer_x >= anchor_x else -1
    direction_y = 1 if pointer_y >= anchor_y else -1
    available_width = image_width - anchor_x if direction_x > 0 else anchor_x
    available_height = image_height - anchor_y if direction_y > 0 else anchor_y
    maximum_multiplier = min(
        available_width // aspect_width,
        available_height // aspect_height,
    )
    if maximum_multiplier < 1:
        return None

    requested_multiplier = max(
        abs(pointer_x - anchor_x) / aspect_width,
        abs(pointer_y - anchor_y) / aspect_height,
    )
    multiplier = max(1, int(round(requested_multiplier)))
    multiplier = min(multiplier, maximum_multiplier)

    width = aspect_width * multiplier
    height = aspect_height * multiplier
    end_x = anchor_x + direction_x * width
    end_y = anchor_y + direction_y * height
    left = min(anchor_x, end_x)
    top = min(anchor_y, end_y)
    return Rect(left, top, width, height)


def annotation_matches_crop(bbox: Rect, crop: Rect, policy: str) -> bool:
    if policy not in ANNOTATION_SELECTION_POLICIES:
        raise ValueError(f"Unsupported annotation selection policy: {policy}")
    if policy == "center":
        return crop.contains_point(*bbox.center)
    if policy == "contained":
        return crop.contains_rect(bbox)
    return bbox.intersection(crop) is not None


def retained_area_ratio(bbox: Rect, crop: Rect) -> float:
    if bbox.area <= 0.0:
        return 0.0
    intersection = bbox.intersection(crop)
    if intersection is None:
        return 0.0
    return intersection.area / bbox.area


def rotated_size(
    width: float,
    height: float,
    rotation_clockwise: int,
) -> tuple[float, float]:
    _validate_rotation(rotation_clockwise)
    if rotation_clockwise in {90, 270}:
        return (height, width)
    return (width, height)


def rotate_point_clockwise(
    x: float,
    y: float,
    source_width: float,
    source_height: float,
    rotation_clockwise: int,
) -> tuple[float, float]:
    _validate_rotation(rotation_clockwise)
    if rotation_clockwise == 0:
        return (x, y)
    if rotation_clockwise == 90:
        return (source_height - y, x)
    if rotation_clockwise == 180:
        return (source_width - x, source_height - y)
    return (y, source_width - x)


def transform_bbox(
    bbox: Rect,
    crop: Rect,
    rotation_clockwise: int,
    destination: Rect,
) -> Rect | None:
    clipped = bbox.intersection(crop)
    if clipped is None:
        return None

    local_left = clipped.x - crop.x
    local_top = clipped.y - crop.y
    local_right = clipped.right - crop.x
    local_bottom = clipped.bottom - crop.y
    rotated_points = [
        rotate_point_clockwise(
            x,
            y,
            crop.width,
            crop.height,
            rotation_clockwise,
        )
        for x, y in (
            (local_left, local_top),
            (local_right, local_top),
            (local_right, local_bottom),
            (local_left, local_bottom),
        )
    ]

    rotated_width, rotated_height = rotated_size(
        crop.width,
        crop.height,
        rotation_clockwise,
    )
    scale_x = destination.width / rotated_width
    scale_y = destination.height / rotated_height
    if not math.isclose(scale_x, scale_y, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "Selection and destination do not permit uniform resizing: "
            f"crop={crop}, rotation={rotation_clockwise}, "
            f"destination={destination}, scales=({scale_x}, {scale_y})"
        )

    destination_points = [
        (
            destination.x + point_x * scale_x,
            destination.y + point_y * scale_y,
        )
        for point_x, point_y in rotated_points
    ]
    left = min(point[0] for point in destination_points)
    top = min(point[1] for point in destination_points)
    right = max(point[0] for point in destination_points)
    bottom = max(point[1] for point in destination_points)
    transformed = Rect(left, top, right - left, bottom - top)
    return transformed.intersection(destination)


def has_exact_aspect(rect: Rect, aspect: tuple[int, int]) -> bool:
    aspect_width, aspect_height = aspect
    return math.isclose(
        rect.width * aspect_height,
        rect.height * aspect_width,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def _validate_rotation(rotation_clockwise: int) -> None:
    if rotation_clockwise not in {0, 90, 180, 270}:
        raise ValueError(f"Unsupported clockwise rotation: {rotation_clockwise}")
