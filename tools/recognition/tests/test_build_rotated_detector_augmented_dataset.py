from __future__ import annotations

from tools.recognition.build_nanodet_region_rotation_augmented_dataset import (
    RegionRect,
    RegionTransform,
)
from tools.recognition.build_rotated_detector_augmented_dataset import transform_annotation


def test_transform_annotation_rotates_center_size_and_angle_together() -> None:
    source = {
        "id": 7,
        "image_id": 1,
        "category_id": 1,
        "obb": [50.0, 40.0, 20.0, 30.0, 10.0],
        "region": "completed_hand",
        "iscrowd": 0,
    }
    region = RegionRect("completed_hand", x=0, y=0, width=100, height=100)
    transform = RegionTransform(
        angle_deg=20.0,
        scale=0.9,
        center_x=50.0,
        center_y=50.0,
        translate_x=3.0,
        translate_y=-2.0,
        resample_count=0,
    )
    generated = transform_annotation(
        source,
        region=region,
        transform=transform,
        image_id=2,
        annotation_id=8,
    )
    assert generated["obb"][2] == 18.0
    assert generated["obb"][3] == 27.0
    assert generated["obb"][4] == 30.0
    assert generated["image_id"] == 2
    assert generated["id"] == 8
