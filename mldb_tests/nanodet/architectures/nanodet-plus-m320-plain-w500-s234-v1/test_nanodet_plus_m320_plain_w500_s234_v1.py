from pathlib import Path
import torch
from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_plain_backbone_ghostpan_s234_output():
    torch.set_num_threads(2)
    build = _load_architecture_build(ROOT, "nanodet/nanodet-plus-m320-plain-w500-s234-v1")
    model = build().eval()
    assert tuple(model.head.strides) == (2, 4, 8, 16)
    assert tuple(model.aux_head.strides) == tuple(model.head.strides)
    with torch.no_grad():
        outputs = model(torch.zeros((1,3,320,320)))
    assert tuple(outputs.shape) == (1,34000,33)
