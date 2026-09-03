from __future__ import annotations

import math

import torch

from tools.recognition.rotated_box_rectifier import (
    RotatedBoxRectifier,
    count_parameters,
    decode_prediction,
    default_crop_window,
    encode_target,
    hbb_from_obb,
    rotated_overlap_metrics,
    window_contains_obb,
)


def test_rectifier_forward_shape_and_size() -> None:
    model = RotatedBoxRectifier(input_size=96).eval()
    with torch.inference_mode():
        output = model(torch.zeros(4, 3, 96, 96))
    assert output.shape == (4, 6)
    assert count_parameters(model) < 500_000


def test_hbb_encloses_rotated_rectangle() -> None:
    obb = (100.0, 80.0, 24.0, 40.0, 33.0)
    left, top, width, height = hbb_from_obb(obb)
    assert width > 24.0
    assert height > 40.0
    assert left < 100.0 < left + width
    assert top < 80.0 < top + height


def test_default_crop_window_keeps_all_obb_corners_visible() -> None:
    obb = (100.0, 80.0, 24.0, 40.0, -42.0)
    window = default_crop_window(obb, context_scale=1.40)
    assert window_contains_obb(window, obb)


def test_target_encode_decode_round_trip() -> None:
    obb = (101.25, 79.5, 23.0, 41.0, 37.0)
    window = default_crop_window(obb, context_scale=1.40)
    encoded = encode_target(obb, window)
    decoded = decode_prediction(encoded.tolist(), window)
    iou, coverage, purity = rotated_overlap_metrics(decoded, obb)
    assert math.isclose(iou, 1.0, rel_tol=1.0e-5, abs_tol=1.0e-5)
    assert math.isclose(coverage, 1.0, rel_tol=1.0e-5, abs_tol=1.0e-5)
    assert math.isclose(purity, 1.0, rel_tol=1.0e-5, abs_tol=1.0e-5)


def test_equivalent_wide_representation_decodes_to_canonical_box() -> None:
    obb = (100.0, 100.0, 40.0, 24.0, -90.0)
    window = default_crop_window(obb, context_scale=1.40)
    decoded = decode_prediction(encode_target(obb, window).tolist(), window)
    assert decoded[2] <= decoded[3]
    iou, _coverage, _purity = rotated_overlap_metrics(decoded, obb)
    assert math.isclose(iou, 1.0, rel_tol=1.0e-5, abs_tol=1.0e-5)
