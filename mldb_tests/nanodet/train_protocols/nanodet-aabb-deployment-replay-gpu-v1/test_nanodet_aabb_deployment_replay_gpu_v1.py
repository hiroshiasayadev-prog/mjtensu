from pathlib import Path
from mldb_v2.src.training.train_interface import _load_train_callable
ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
def test_train_callable_contract() -> None:
    assert callable(_load_train_callable(ROOT, "nanodet/nanodet-aabb-deployment-replay-gpu-v1"))
