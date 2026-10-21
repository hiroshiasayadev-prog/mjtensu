from pathlib import Path

import torch
from torch import nn

from mldb_v2.src.catalog.architecture_build import _load_architecture_build


ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
ARCHITECTURE_ID = "tile-classifier/tile-c8-gray35-narrow-late2-dw3-pw1-v1"


def test_architecture_build_contract_and_grouped_export() -> None:
    model = _load_architecture_build(ROOT, ARCHITECTURE_ID)()
    assert tuple(model(torch.zeros(2, 1, 64, 64)).shape) == (2, 35)
    assert model.group_size == 8
    assert model.field_counts == (4, 8, 16, 32)
    assert model.tensor_channels == (32, 64, 128, 256)

    exported = model.equivariant_backbone.eval().export()
    convs = [module for module in exported.modules() if isinstance(module, nn.Conv2d)]
    assert [
        (module.kernel_size, module.groups, module.in_channels, module.out_channels)
        for module in convs
    ] == [
        ((5, 5), 1, 1, 32),
        ((3, 3), 1, 32, 64),
        ((3, 3), 8, 64, 64),
        ((1, 1), 1, 64, 128),
        ((3, 3), 16, 128, 128),
        ((1, 1), 1, 128, 256),
    ]
