from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    model = _load_architecture_build(ROOT, "tile-classifier/tile-c4-gray35-c8n-width-v1")()
    assert tuple(model(torch.zeros(2, 1, 64, 64)).shape) == (2, 35)
    assert model.group_size == 4
    assert model.field_counts == (8, 16, 32, 64)
    assert model.tensor_channels == (32, 64, 128, 256)
