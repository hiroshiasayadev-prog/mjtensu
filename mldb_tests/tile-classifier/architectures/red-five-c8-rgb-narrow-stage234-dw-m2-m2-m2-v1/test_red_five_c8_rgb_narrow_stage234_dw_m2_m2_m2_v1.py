from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    model = _load_architecture_build(
        ROOT, "tile-classifier/red-five-c8-rgb-narrow-stage234-dw-m2-m2-m2-v1"
    )()
    assert tuple(model(torch.zeros(2, 3, 64, 64)).shape) == (2, 2)
    assert model.group_size == 8
    assert model.field_counts == (4, 8, 16, 32)
    assert model.tensor_channels == (32, 64, 128, 256)
    assert model.stage_multipliers == {2: 2, 3: 2, 4: 2}


def test_architecture_rejects_non_rgb() -> None:
    model = _load_architecture_build(
        ROOT, "tile-classifier/red-five-c8-rgb-narrow-stage234-dw-m2-m2-m2-v1"
    )()
    try:
        model(torch.zeros(1, 1, 64, 64))
    except ValueError as error:
        assert "[N,3,H,W]" in str(error)
    else:
        raise AssertionError("expected RGB shape rejection")
