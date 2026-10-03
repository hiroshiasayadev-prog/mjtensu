from pathlib import Path

import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    build = _load_architecture_build(
        ROOT,
        "tile-classifier/tile-plain-gray35-w500-late256-late-dw3-pw1-down3-dw-v1",
    )
    model = build()
    assert tuple(model(torch.zeros(2, 1, 64, 64)).shape) == (2, 35)

    stride2_depthwise = [
        module
        for module in model.features
        if isinstance(module, torch.nn.Conv2d)
        and module.kernel_size == (3, 3)
        and module.stride == (2, 2)
        and module.groups == 64
    ]
    assert len(stride2_depthwise) == 1
