from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mldb_v2.src.catalog.architecture_build import _load_architecture_build


ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
ARCH = "tile-classifier/tile-mobilenet-v3-small-1x-gray35-v1"


def test_build_shape_and_mobile_operators() -> None:
    build = _load_architecture_build(ROOT, ARCH)
    model = build().eval()
    with torch.no_grad():
        output = model(torch.randn(2, 1, 64, 64))
    assert tuple(output.shape) == (2, 35)

    depthwise = [
        module
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
        and module.groups == module.in_channels
        and module.in_channels > 1
    ]
    assert len(depthwise) == 11
    assert any(isinstance(module, nn.Hardswish) for module in model.modules())
    assert any(isinstance(module, nn.Hardsigmoid) for module in model.modules())


def test_width_and_batchnorm_contract() -> None:
    build = _load_architecture_build(ROOT, ARCH)
    model = build()
    assert model.width_mult == 1.0
    batch_norms = [
        module for module in model.modules() if isinstance(module, nn.BatchNorm2d)
    ]
    assert batch_norms
    assert all(module.eps == 0.001 for module in batch_norms)
    assert all(module.momentum == 0.01 for module in batch_norms)
