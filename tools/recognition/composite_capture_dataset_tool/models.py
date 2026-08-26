from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError(f"Rectangle dimensions must be non-negative: {self}")

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_point(self, x: float, y: float) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom

    def contains_rect(self, other: "Rect") -> bool:
        return (
            self.x <= other.x
            and self.y <= other.y
            and other.right <= self.right
            and other.bottom <= self.bottom
        )

    def intersection(self, other: "Rect") -> "Rect | None":
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return Rect(left, top, right - left, bottom - top)

    def to_coco_bbox(self, *, precision: int = 6) -> list[float]:
        return [
            _rounded_float(self.x, precision),
            _rounded_float(self.y, precision),
            _rounded_float(self.width, precision),
            _rounded_float(self.height, precision),
        ]

    def to_pillow_box(self) -> tuple[int, int, int, int]:
        values = (self.x, self.y, self.right, self.bottom)
        if not all(isclose(value, round(value), abs_tol=1e-9) for value in values):
            raise ValueError(f"Pillow crop rectangle must use integer edges: {self}")
        return tuple(int(round(value)) for value in values)  # type: ignore[return-value]

    @classmethod
    def from_coco_bbox(cls, bbox: Any, *, context: str) -> "Rect":
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"Invalid COCO bbox in {context}: {bbox!r}")
        try:
            x, y, width, height = (float(value) for value in bbox)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Non-numeric COCO bbox in {context}: {bbox!r}") from error
        if width <= 0 or height <= 0:
            raise ValueError(f"Non-positive COCO bbox in {context}: {bbox!r}")
        return cls(x, y, width, height)


@dataclass(frozen=True)
class RegionSelection:
    region_key: str
    crop: Rect
    rotation_clockwise: int = 0

    def __post_init__(self) -> None:
        if self.rotation_clockwise not in {0, 90, 180, 270}:
            raise ValueError(
                "rotation_clockwise must be one of 0, 90, 180, or 270: "
                f"{self.rotation_clockwise}"
            )
        if self.crop.width <= 0 or self.crop.height <= 0:
            raise ValueError(f"Selection crop must be non-empty: {self.crop}")
        self.crop.to_pillow_box()


@dataclass(frozen=True)
class TransformedAnnotation:
    source_annotation_id: int
    region_key: str
    bbox: Rect
    iscrowd: int


@dataclass(frozen=True)
class RegionCompositeStats:
    retained_annotations: int
    clipped_annotations: int


@dataclass(frozen=True)
class SavedComposite:
    image_id: int
    annotation_count: int
    image_path: str
    annotations_path: str


def _rounded_float(value: float, precision: int) -> float:
    rounded = round(float(value), precision)
    if rounded == -0.0:
        return 0.0
    return rounded
