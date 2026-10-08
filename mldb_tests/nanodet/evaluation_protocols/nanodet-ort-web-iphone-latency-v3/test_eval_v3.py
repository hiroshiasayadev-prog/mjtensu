from pathlib import Path

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_callable_is_loadable():
    assert callable(_load_evaluation_callable(ROOT, "nanodet/nanodet-ort-web-iphone-latency-v3"))


def test_detector_onnx_only_input_and_dynamic_dense_positions():
    source = (ROOT / "nanodet/evaluation_protocols/nanodet-ort-web-iphone-latency-v3.py").read_text()
    assert "import torch" not in source
    assert "torch.onnx.export" not in source
    assert 'context.inputs["onnx_model"].data' in source
    assert "const n=dims[1]" in source
