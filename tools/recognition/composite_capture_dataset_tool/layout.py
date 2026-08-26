from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Rect


_LAYOUT_PATH = Path(__file__).resolve().parents[1] / "capture_layout.v1.json"
_LAYOUT_DOCUMENT: dict[str, Any] = json.loads(_LAYOUT_PATH.read_text(encoding="utf-8"))

_COMPOSITE = _LAYOUT_DOCUMENT["composite"]
COMPOSITE_SIZE = (int(_COMPOSITE["width"]), int(_COMPOSITE["height"]))
PADDING_RGB = tuple(int(value) for value in _COMPOSITE["paddingRgb"])
LAYOUT_ID = str(_LAYOUT_DOCUMENT["id"])
TARGET_CATEGORY = {
    "id": 1,
    "name": "mahjong_tile",
    "supercategory": "mahjong_tile",
}


@dataclass(frozen=True)
class RegionSpec:
    key: str
    label: str
    destination: Rect
    source_aspect_width: int
    source_aspect_height: int

    def source_aspect_for_rotation(self, rotation_clockwise: int) -> tuple[int, int]:
        if rotation_clockwise not in {0, 90, 180, 270}:
            raise ValueError(f"Unsupported rotation: {rotation_clockwise}")
        if rotation_clockwise in {90, 270}:
            return (self.source_aspect_height, self.source_aspect_width)
        return (self.source_aspect_width, self.source_aspect_height)


def _region_spec(key: str) -> RegionSpec:
    document = _LAYOUT_DOCUMENT["regions"][key]
    destination = document["destination"]
    source_aspect = document["sourceAspect"]
    return RegionSpec(
        key=key,
        label=str(document["label"]),
        destination=Rect(
            int(destination["x"]),
            int(destination["y"]),
            int(destination["width"]),
            int(destination["height"]),
        ),
        source_aspect_width=int(source_aspect[0]),
        source_aspect_height=int(source_aspect[1]),
    )


REGION_SPECS: dict[str, RegionSpec] = {
    key: _region_spec(key)
    for key in ("completed_hand", "dora_indicators", "melds")
}
