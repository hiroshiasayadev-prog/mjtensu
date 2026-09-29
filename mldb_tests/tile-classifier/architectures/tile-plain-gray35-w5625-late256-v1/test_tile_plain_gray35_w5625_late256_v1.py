from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    model = _load_architecture_build(ROOT, "tile-classifier/tile-plain-gray35-w5625-late256-v1")()
    assert tuple(model(torch.zeros(2, 1, 64, 64)).shape) == (2, 35)
    convs = [module for module in model.features if isinstance(module, torch.nn.Conv2d)]
    assert [module.out_channels for module in convs[:4]] == [18, 36, 72, 108]
    assert convs[-1].kernel_size == (1, 1)
    assert convs[-1].out_channels == 256
