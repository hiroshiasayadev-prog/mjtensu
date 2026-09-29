from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    build = _load_architecture_build(ROOT, "tile-classifier/tile-plain-gray35-threequarter-late-expand-v1")
    model = build()
    output = model(torch.zeros(2, 1, 64, 64))
    assert tuple(output.shape) == (2, 35)
    one_by_one = [module for module in model.features if isinstance(module, torch.nn.Conv2d) and module.kernel_size == (1, 1)]
    assert len(one_by_one) == 1
    assert one_by_one[0].in_channels == 144
    assert one_by_one[0].out_channels == 192
