from pathlib import Path

import torch
from torch import nn

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
ARCH = "tile-classifier/tile-c8-gray35-narrow-stage234-dw-m2-m2-m2-v1"


def test_architecture_build_contract() -> None:
    model = _load_architecture_build(ROOT, ARCH)()
    assert tuple(model(torch.zeros(2, 1, 64, 64)).shape) == (2, 35)
    assert model.group_size == 8
    assert model.field_counts == (4, 8, 16, 32)
    assert model.tensor_channels == (32, 64, 128, 256)
    assert model.stage_multipliers == {2: 2, 3: 2, 4: 2}


def test_export_signature() -> None:
    model = _load_architecture_build(ROOT, ARCH)()
    convs = [
        module
        for module in model.equivariant_backbone.eval().export().modules()
        if isinstance(module, nn.Conv2d)
    ]
    assert [
        (module.kernel_size, module.groups, module.in_channels, module.out_channels)
        for module in convs
    ] == [
        ((5, 5), 1, 1, 32),
        ((3, 3), 4, 32, 64),
        ((1, 1), 1, 64, 64),
        ((3, 3), 8, 64, 128),
        ((1, 1), 1, 128, 128),
        ((3, 3), 16, 128, 256),
        ((1, 1), 1, 256, 256),
    ]
