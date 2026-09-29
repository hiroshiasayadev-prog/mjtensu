from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    model = _load_architecture_build(ROOT, "tile-classifier/tile-plain-gray35-w500-late256-late-dw3-pw1-v1")()
    assert tuple(model(torch.zeros(2, 1, 64, 64)).shape) == (2, 35)
    convs = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]
    actual = [(m.in_channels, m.out_channels, m.kernel_size, m.groups) for m in convs]
    expected = [(1,16,(5,5),1),(16,32,(3,3),1),(32,64,(3,3),1),(64,96,(3,3),1),(96,96,(3,3),96),(96,256,(1,1),1)]
    assert actual == expected
    assert len([m for m in model.features if isinstance(m, torch.nn.MaxPool2d)]) == 3
