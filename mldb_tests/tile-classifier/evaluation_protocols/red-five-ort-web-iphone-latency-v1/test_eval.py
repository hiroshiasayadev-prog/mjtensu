from pathlib import Path
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable
ROOT = Path(__file__).resolve().parents[4] / "mldb_data"

def test_callable_is_loadable():
    assert callable(_load_evaluation_callable(ROOT, "tile-classifier/red-five-ort-web-iphone-latency-v1"))

def test_protocol_is_rgb_two_class_onnx_only():
    source=(ROOT / "tile-classifier/evaluation_protocols/red-five-ort-web-iphone-latency-v1.py").read_text()
    assert "import torch" not in source
    assert "torch.onnx.export" not in source
    assert "const IMAGE_CHANNELS = 3;" in source
    assert "const CLASS_COUNT = 2;" in source
    assert 'context.inputs["onnx_model"].data' in source
