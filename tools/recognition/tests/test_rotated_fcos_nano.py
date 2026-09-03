from __future__ import annotations

import math

import torch

from tools.recognition.rotated_fcos_nano import (
    RotatedFCOSNano,
    build_targets,
    canonicalize_obb,
    count_parameters,
    make_points,
    rotated_iou,
)


def test_rotated_fcos_nano_forward_shapes_and_size() -> None:
    model = RotatedFCOSNano(
        backbone="shufflenet_v2_x0_5",
        fpn_channels=64,
        head_convs=2,
        head_normalization="group",
        pretrained_backbone=False,
    ).eval()
    with torch.inference_mode():
        outputs = model(torch.zeros(2, 3, 320, 320))
    assert [tuple(output.shape) for output in outputs] == [
        (2, 8, 40, 40),
        (2, 8, 20, 20),
        (2, 8, 10, 10),
    ]
    assert count_parameters(model) < 2_000_000


def test_shared_head_supports_group_norm_without_batch_norm_running_stats() -> None:
    model = RotatedFCOSNano(
        fpn_channels=64,
        head_convs=2,
        head_normalization="group",
        pretrained_backbone=False,
    )
    head_modules = list(model.head.modules())
    assert any(isinstance(module, torch.nn.GroupNorm) for module in head_modules)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in head_modules)
    assert model.config()["head_normalization"] == "group"


def test_legacy_batch_normalized_head_remains_available() -> None:
    model = RotatedFCOSNano(
        fpn_channels=64,
        head_convs=2,
        head_normalization="batch",
        pretrained_backbone=False,
    )
    assert any(isinstance(module, torch.nn.BatchNorm2d) for module in model.head.modules())
    assert model.config()["head_normalization"] == "batch"


def test_target_assignment_produces_positive_angle_targets() -> None:
    model = RotatedFCOSNano(pretrained_backbone=False).eval()
    with torch.inference_mode():
        outputs = model(torch.zeros(1, 3, 320, 320))
    points, strides, levels = make_points(outputs)
    boxes = torch.tensor([[100.0, 80.0, 24.0, 40.0, 30.0]], dtype=torch.float32)
    objectness, centerness, regression = build_targets(
        boxes,
        points,
        strides,
        levels,
    )
    positive = objectness > 0.5
    assert positive.any()
    assert torch.all(centerness[positive] > 0)
    expected_sin = math.sin(math.radians(60.0))
    expected_cos = math.cos(math.radians(60.0))
    assert torch.allclose(
        regression[positive, 4],
        torch.full_like(regression[positive, 4], expected_sin),
        atol=1.0e-5,
    )
    assert torch.allclose(
        regression[positive, 5],
        torch.full_like(regression[positive, 5], expected_cos),
        atol=1.0e-5,
    )


def test_canonicalize_obb_preserves_equivalent_geometry() -> None:
    canonical = canonicalize_obb((100.0, 100.0, 30.0, 20.0, -90.0))
    assert canonical == (100.0, 100.0, 20.0, 30.0, 0.0)
    assert math.isclose(
        rotated_iou((100.0, 100.0, 30.0, 20.0, -90.0), canonical),
        1.0,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )


def test_rotated_iou_handles_identical_and_disjoint_boxes() -> None:
    box = (100.0, 100.0, 30.0, 50.0, 27.0)
    assert math.isclose(rotated_iou(box, box), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
    assert rotated_iou(box, (250.0, 250.0, 30.0, 50.0, 27.0)) == 0.0


def test_rotated_iou_is_angle_periodic_over_180_degrees() -> None:
    first = (100.0, 100.0, 30.0, 50.0, -35.0)
    second = (100.0, 100.0, 30.0, 50.0, 145.0)
    assert math.isclose(rotated_iou(first, second), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
