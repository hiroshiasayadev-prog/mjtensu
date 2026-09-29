from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    model = _load_architecture_build(ROOT, "tile-classifier/tile-plain-gray35-w500-late256-stage34-1x1-v1")()
    assert tuple(model(torch.zeros(2, 1, 64, 64)).shape) == (2, 35)
    convs = [module for module in model.features if isinstance(module, torch.nn.Conv2d)]
    assert [module.kernel_size for module in convs[:4]] == [(5, 5), (3, 3), (1, 1), (1, 1)]
    assert [module.out_channels for module in convs[:4]] == [16, 32, 64, 96]
    assert convs[-1].kernel_size == (1, 1)
    assert convs[-1].out_channels == 256
