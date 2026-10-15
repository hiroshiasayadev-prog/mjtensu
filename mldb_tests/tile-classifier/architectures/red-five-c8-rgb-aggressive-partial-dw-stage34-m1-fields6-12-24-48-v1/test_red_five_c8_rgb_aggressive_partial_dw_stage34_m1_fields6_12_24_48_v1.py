from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_contract() -> None:
    model = _load_architecture_build(ROOT, "tile-classifier/red-five-c8-rgb-aggressive-partial-dw-stage34-m1-fields6-12-24-48-v1")()
    assert model.field_counts == (6, 12, 24, 48)
    assert model.stage_strides == (1, 2, 2, 2)
    assert model.dw_stages == (3, 4)
    assert model.dw_multiplier == 1
    assert sum(p.numel() for p in model.parameters()) == 22052
    assert tuple(model(torch.zeros((2, 3, 64, 64))).shape) == (2, 2)


def test_final_spatial_size_is_4x4() -> None:
    model = _load_architecture_build(ROOT, "tile-classifier/red-five-c8-rgb-aggressive-partial-dw-stage34-m1-fields6-12-24-48-v1")()
    with torch.inference_mode():
        x = model._enn.GeometricTensor(torch.zeros((1, 3, 64, 64)), model.input_type)
        y = model.equivariant_backbone(x).tensor
    assert tuple(y.shape[-2:]) == (4, 4)
