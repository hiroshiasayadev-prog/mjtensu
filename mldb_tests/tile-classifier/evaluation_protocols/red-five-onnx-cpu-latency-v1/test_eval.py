from pathlib import Path
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable
ROOT=Path(__file__).resolve().parents[4]/'mldb_data'
def test_callable_is_loadable(): assert callable(_load_evaluation_callable(ROOT,'tile-classifier/red-five-onnx-cpu-latency-v1'))
