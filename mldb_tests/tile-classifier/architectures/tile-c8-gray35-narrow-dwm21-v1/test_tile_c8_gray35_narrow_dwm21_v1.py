from pathlib import Path

import torch
from torch import nn

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
ARCH = "tile-classifier/tile-c8-gray35-narrow-dwm21-v1"


def test_build_and_export_signature() -> None:
    model = _load_architecture_build(ROOT, ARCH)()
    assert tuple(model(torch.zeros(2, 1, 64, 64)).shape) == (2, 35)
    assert model.stage_multipliers == {3: 2, 4: 1}
    convs = [m for m in model.equivariant_backbone.eval().export().modules() if isinstance(m, nn.Conv2d)]
    assert [(m.kernel_size, m.groups, m.in_channels, m.out_channels) for m in convs] == [((5, 5), 1, 1, 32), ((3, 3), 1, 32, 64), ((3, 3), 8, 64, 128), ((1, 1), 1, 128, 128), ((3, 3), 16, 128, 128), ((1, 1), 1, 128, 256)]
