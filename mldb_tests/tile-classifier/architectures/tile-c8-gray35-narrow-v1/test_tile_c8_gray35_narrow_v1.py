from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    model = _load_architecture_build(ROOT, "tile-classifier/tile-c8-gray35-narrow-v1")()
    assert tuple(model(torch.zeros(2, 1, 64, 64)).shape) == (2, 35)
    assert model.group_size == 8
    assert model.field_counts == (4, 8, 16, 32)
    assert model.tensor_channels == (32, 64, 128, 256)
