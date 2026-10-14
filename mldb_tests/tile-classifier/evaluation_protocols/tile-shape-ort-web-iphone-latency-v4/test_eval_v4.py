from pathlib import Path

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_callable_is_loadable():
    assert callable(_load_evaluation_callable(ROOT, "tile-classifier/tile-shape-ort-web-iphone-latency-v4"))


def test_protocol_never_imports_torch_or_exports_onnx():
    source = (ROOT / "tile-classifier/evaluation_protocols/tile-shape-ort-web-iphone-latency-v4.py").read_text()
    assert "import torch" not in source
    assert "torch.onnx.export" not in source
    assert 'context.inputs["onnx_model"].data' in source
    assert "health.get(\"tunnel_ready\")" not in source
