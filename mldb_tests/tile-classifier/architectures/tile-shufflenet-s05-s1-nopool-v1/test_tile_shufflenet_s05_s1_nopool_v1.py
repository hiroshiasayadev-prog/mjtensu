from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
ID = "tile-classifier/tile-shufflenet-s05-s1-nopool-v1"


def test_architecture_build_and_spatial_contract() -> None:
    model = _load_architecture_build(ROOT, ID)()
    image = torch.zeros(2, 1, 64, 64)
    with torch.no_grad():
        features = model.features(image)
        logits = model(image)
    assert tuple(features.shape[-2:]) == (8, 8)
    assert tuple(logits.shape) == (2, 35)