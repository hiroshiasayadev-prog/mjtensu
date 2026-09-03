from __future__ import annotations

import json

from tools.recognition.build_rotated_detector_corpus import (
    anisotropic_box_to_approx_obb,
    annotation_geometry_delta,
    canonicalize_obb,
    empty_coco,
    normalize_angle_180,
    obb_corners,
    select_reviewed_rows,
    split_payload,
)


def document(angle: float, width: float = 20.0) -> dict:
    return {
        "schemaVersion": 1,
        "captureId": "cap_test",
        "boxes": {
            "completed_hand": [
                {
                    "id": "box-1",
                    "centerX": 30.0,
                    "centerY": 40.0,
                    "width": width,
                    "height": 30.0,
                    "angleDeg": angle,
                }
            ],
            "dora_indicators": [],
            "melds": [],
        },
    }


def test_annotation_geometry_delta_detects_manual_obb_edit() -> None:
    changed_count, max_delta = annotation_geometry_delta(
        document(12.5, 18.0),
        document(0.0, 20.0),
        epsilon=1.0e-4,
    )
    assert changed_count == 1
    assert max_delta == 12.5


def test_annotation_geometry_delta_ignores_numeric_noise() -> None:
    changed_count, max_delta = annotation_geometry_delta(
        document(0.00001),
        document(0.0),
        epsilon=1.0e-4,
    )
    assert changed_count == 0
    assert max_delta == 0.0


def test_angle_normalization_has_180_degree_period() -> None:
    assert normalize_angle_180(100.0) == -80.0
    assert normalize_angle_180(-100.0) == 80.0
    assert normalize_angle_180(180.0) == 0.0


def test_obb_corners_stay_rectangular_after_rotation() -> None:
    corners = obb_corners((100.0, 100.0, 20.0, 40.0, 30.0))
    assert len(corners) == 4
    assert abs(sum(point[0] for point in corners) / 4.0 - 100.0) < 1.0e-6
    assert abs(sum(point[1] for point in corners) / 4.0 - 100.0) < 1.0e-6


def test_anisotropic_crop_mapping_stays_representable_as_approximate_obb() -> None:
    obb = anisotropic_box_to_approx_obb(
        center_x=100.0,
        center_y=50.0,
        width=20.0,
        height=40.0,
        angle_deg=30.0,
        scale_x=0.29310344827586204,
        scale_y=0.2926829268292683,
        offset_x=7.0,
        offset_y=0.0,
    )
    assert abs(obb[0] - 36.310344827586206) < 1.0e-9
    assert abs(obb[1] - 14.634146341463415) < 1.0e-9
    assert 5.84 < obb[2] < 5.87
    assert 11.70 < obb[3] < 11.73
    assert 29.9 < obb[4] < 30.1


def test_canonicalize_obb_swaps_equivalent_long_side_representation() -> None:
    canonical = canonicalize_obb((100.0, 100.0, 30.0, 20.0, -90.0))
    assert canonical == [100.0, 100.0, 20.0, 30.0, 0.0]


def test_canonicalize_obb_keeps_short_side_as_width_and_angle_in_half_turn_range() -> None:
    canonical = canonicalize_obb((100.0, 100.0, 42.0, 24.0, 10.0))
    assert canonical[2] == 24.0
    assert canonical[3] == 42.0
    assert canonical[4] == -80.0
    assert canonical[2] <= canonical[3]
    assert -90.0 <= canonical[4] < 90.0


def test_select_reviewed_rows_uses_explicit_human_review_state() -> None:
    reviewed = {
        "capture_id": "cap-reviewed",
        "annotation_status": "complete",
        "annotation_json": json.dumps(
            {**document(0.0), "captureId": "cap-reviewed", "review": {"state": "human_reviewed"}}
        ),
    }
    suggested = {
        "capture_id": "cap-suggested",
        "annotation_status": "draft",
        "annotation_json": json.dumps(
            {**document(0.0), "captureId": "cap-suggested", "review": {"state": "model_suggested"}}
        ),
    }
    selected, summary = select_reviewed_rows([reviewed, suggested])
    assert [row["capture_id"] for row in selected] == ["cap-reviewed"]
    assert summary["selected_capture_count"] == 1
    assert summary["non_reviewed_capture_count"] == 1


def test_split_payload_keeps_layout_groups_disjoint() -> None:
    payload = empty_coco("test")
    payload["images"] = [
        {"id": 1, "split_group": "campaign::layout-a"},
        {"id": 2, "split_group": "campaign::layout-a"},
        {"id": 3, "split_group": "campaign::layout-b"},
        {"id": 4, "split_group": "campaign::layout-b"},
    ]
    payload["annotations"] = [
        {"id": image_id, "image_id": image_id, "category_id": 1}
        for image_id in range(1, 5)
    ]
    train, val, split = split_payload(
        payload,
        train_fraction=0.5,
        seed=42,
        allow_capture_split=False,
    )
    train_groups = {image["split_group"] for image in train["images"]}
    val_groups = {image["split_group"] for image in val["images"]}
    assert split["unit"] == "campaign_layout"
    assert train_groups.isdisjoint(val_groups)
