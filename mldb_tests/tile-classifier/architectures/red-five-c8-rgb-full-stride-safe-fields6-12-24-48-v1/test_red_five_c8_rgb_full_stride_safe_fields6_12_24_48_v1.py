from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    model = _load_architecture_build(ROOT, "tile-classifier/red-five-c8-rgb-full-stride-safe-fields6-12-24-48-v1")()
    assert tuple(model(torch.zeros((2, 3, 64, 64))).shape) == (2, 2)
    assert model.field_counts == (6, 12, 24, 48)
    assert model.stage_strides == (1, 2, 2, 1)
    assert sum(p.numel() for p in model.parameters()) == 77852


def test_final_spatial_size() -> None:
    model = _load_architecture_build(ROOT, "tile-classifier/red-five-c8-rgb-full-stride-safe-fields6-12-24-48-v1")()
    with torch.inference_mode():
        x = model._enn.GeometricTensor(torch.zeros((1, 3, 64, 64)), model.input_type)
        y = model.equivariant_backbone(x).tensor
    assert tuple(y.shape[-2:]) == (8, 8)
