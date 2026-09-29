from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    model = _load_architecture_build(ROOT, "tile-classifier/tile-plain-gray35-w500-late256-stage2-dw3-pw1-v1")()
    assert tuple(model(torch.zeros(2, 1, 64, 64)).shape) == (2, 35)
    convs = [module for module in model.features if isinstance(module, torch.nn.Conv2d)]
    assert [(c.in_channels, c.out_channels, c.kernel_size, c.groups) for c in convs] == [
        (1, 16, (5, 5), 1),
        (16, 16, (3, 3), 16),
        (16, 32, (1, 1), 1),
        (32, 64, (3, 3), 1),
        (64, 96, (3, 3), 1),
        (96, 256, (1, 1), 1),
    ]
    pools = [module for module in model.features if isinstance(module, torch.nn.MaxPool2d)]
    assert len(pools) == 3
