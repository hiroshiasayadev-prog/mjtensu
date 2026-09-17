from pathlib import Path

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_evaluation_callable_contract() -> None:
    evaluate = _load_evaluation_callable(
        ROOT,
        "tile-classifier/tile-shape-angle-robustness-v2",
    )
    assert callable(evaluate)
