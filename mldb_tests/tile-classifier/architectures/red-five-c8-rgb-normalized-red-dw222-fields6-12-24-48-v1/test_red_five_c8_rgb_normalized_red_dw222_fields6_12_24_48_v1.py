from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    model = _load_architecture_build(ROOT, "tile-classifier/red-five-c8-rgb-normalized-red-dw222-fields6-12-24-48-v1")()
    assert tuple(model(torch.full((2, 3, 64, 64), 0.5)).shape) == (2, 2)
    assert model.feature_kind == "normalized-red"
    assert model.field_counts == (6, 12, 24, 48)
    assert model.stage_multipliers == {2: 2, 3: 2, 4: 2}
    assert tuple(model.input_mean.shape) == (4,)
    assert tuple(model.input_std.shape) == (4,)
