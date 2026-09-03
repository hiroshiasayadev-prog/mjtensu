from __future__ import annotations

import json

from tools.recognition.apply_rotated_box_rectifier_suggestions import (
    detector_boxes_if_complete,
    expected_region_counts,
    review_state,
    select_source_document,
)


def task() -> dict:
    return {
        "hand": [
            {"face": "front"},
            {"face": "front"},
        ],
        "dora": {
            "visible": [{"face": "front"}],
            "ura": [{"face": "back"}],
        },
        "melds": [
            {
                "tiles": [
                    {"face": "front"},
                    {"face": "front"},
                    {"face": "back"},
                ]
            }
        ],
    }


def manifest() -> dict:
    return {
        "regionRects": {
            "completed_hand": {"pixel": {"x": 10.0, "y": 20.0}},
            "dora_indicators": {"pixel": {"x": 10.0, "y": 100.0}},
            "melds": {"pixel": {"x": 200.0, "y": 300.0}},
        }
    }


def annotation_document(*, review: dict | None = None) -> dict:
    document = {
        "schemaVersion": 1,
        "captureId": "cap-test",
        "boxes": {
            "completed_hand": [],
            "dora_indicators": [],
            "melds": [],
        },
    }
    if review is not None:
        document["review"] = review
    return document


def test_review_state_distinguishes_model_and_human() -> None:
    assert review_state(None) is None
    assert review_state(annotation_document()) is None
    assert review_state(
        annotation_document(review={"state": "model_suggested"})
    ) == "model_suggested"
    assert review_state(
        annotation_document(review={"state": "human_reviewed"})
    ) == "human_reviewed"
    assert review_state(annotation_document(review={"state": "other"})) is None


def test_expected_region_counts_only_include_visible_front_tiles() -> None:
    assert expected_region_counts(task()) == {
        "completed_hand": 2,
        "dora_indicators": 1,
        "melds": 2,
    }


def test_detector_fallback_requires_exact_expected_counts() -> None:
    detections = [
        {
            "detection_index": 0,
            "region": "completed_hand",
            "original_x": 20.0,
            "original_y": 30.0,
            "original_width": 20.0,
            "original_height": 30.0,
        },
        {
            "detection_index": 1,
            "region": "completed_hand",
            "original_x": 50.0,
            "original_y": 30.0,
            "original_width": 20.0,
            "original_height": 30.0,
        },
        {
            "detection_index": 2,
            "region": "dora_indicators",
            "original_x": 20.0,
            "original_y": 110.0,
            "original_width": 20.0,
            "original_height": 30.0,
        },
        {
            "detection_index": 3,
            "region": "melds",
            "original_x": 210.0,
            "original_y": 310.0,
            "original_width": 30.0,
            "original_height": 20.0,
        },
        {
            "detection_index": 4,
            "region": "melds",
            "original_x": 250.0,
            "original_y": 310.0,
            "original_width": 20.0,
            "original_height": 30.0,
        },
    ]
    boxes = detector_boxes_if_complete(
        detections,
        task=task(),
        manifest=manifest(),
        capture_id="cap-test",
    )
    assert boxes is not None
    assert [len(boxes[key]) for key in ("completed_hand", "dora_indicators", "melds")] == [2, 1, 2]
    sideways = boxes["melds"][0]
    assert sideways["width"] == 20.0
    assert sideways["height"] == 30.0
    assert sideways["angleDeg"] == 90.0

    assert detector_boxes_if_complete(
        detections[:-1],
        task=task(),
        manifest=manifest(),
        capture_id="cap-test",
    ) is None


def test_refreshing_model_suggestion_reuses_baseline_hbb_not_model_geometry() -> None:
    baseline_document = annotation_document()
    model_document = annotation_document(review={"state": "model_suggested"})
    row = {
        "capture_id": "cap-test",
        "annotation_json": json.dumps(model_document),
        "task_json": "{}",
        "manifest_json": "{}",
        "detections": [],
    }
    document, source = select_source_document(
        row,
        baseline_row={"annotation_json": json.dumps(baseline_document)},
    )
    assert document == baseline_document
    assert source == "baseline_annotation_hbb"


def test_existing_annotation_without_baseline_is_never_overwritten() -> None:
    row = {
        "capture_id": "cap-test",
        "annotation_json": '{"schemaVersion":1,"captureId":"cap-test","boxes":{"completed_hand":[],"dora_indicators":[],"melds":[]}}',
        "task_json": "{}",
        "manifest_json": "{}",
        "detections": [],
    }
    document, source = select_source_document(row, baseline_row=None)
    assert document is None
    assert source is None
