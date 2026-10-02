from pathlib import Path
import torch

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
ARCHITECTURE_ID = "nanodet/nanodet-plus-m320-ghostpan-residual-v1"


def test_architecture_build_and_output_contract() -> None:
    build = _load_architecture_build(ROOT, ARCHITECTURE_ID)
    model = build().eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 3, 320, 320), dtype=torch.float32))
    assert tuple(output.shape) == (1, 2125, 33)
