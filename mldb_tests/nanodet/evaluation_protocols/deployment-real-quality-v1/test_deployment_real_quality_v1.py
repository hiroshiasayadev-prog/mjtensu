from pathlib import Path
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable
ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
def test_evaluate_callable_contract() -> None:
    assert callable(_load_evaluation_callable(ROOT, "nanodet/deployment-real-quality-v1"))
