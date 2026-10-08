"""ONNX-artifact-only iPhone protocol contract tests, without physical hardware."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2] / "mldb_data"


def _module(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("onnx_only_protocol_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "torch" not in module.__dict__
    return module


def _context(tmp_path: Path, parameters: dict):
    return SimpleNamespace(
        model=None,
        onnx_input=b"immutable ONNX fixture",
        onnx_source_evaluation_result="demo/completed-export-v1",
        parameters=parameters, work_dir=tmp_path,
        inputs={"onnx_model": SimpleNamespace(data=b"immutable ONNX fixture", source_evaluation_result="demo/completed-export-v1")},
    )


def _browser(blocks: int, runs: int, warmup: int, batch: int, *, detector=False):
    parity = (
        {"dims": [1, 2100, 33], "indices": list(range(10)), "values": [0.5] * 10}
        if detector else {"dims": [4, 35], "logits": [0.0] * 140}
    )
    return {
        "ok": True,
        "parity": parity,
        "benchmark": {
            "batch_size": batch, "warmup_runs": warmup,
            "runs_per_block": runs, "measure_blocks": blocks,
            "total_measure_runs": runs * blocks,
            "samples_ms": [1.0 + i * 0.01 for i in range(blocks)],
            "block_totals_ms": [runs * (1.0 + i * 0.01) for i in range(blocks)],
        },
        "environment": {"provider": "wasm-simd", "num_threads": 1, "wasm_proxy": False},
    }


def test_classifier_iphone_eval_uses_only_preexported_bytes(tmp_path, monkeypatch):
    path = "tile-classifier/evaluation_protocols/tile-shape-ort-web-iphone-latency-v4.py"
    module = _module(path)
    parameters = {
        "batch_size": 1, "warmup_runs": 25,
        "runs_per_block": 100, "measure_blocks": 10,
        "source_onnx_evaluation_result": "demo/completed-export-v1",
    }
    htmls = []
    def fake_runner(document):
        htmls.append(document)
        return _browser(10, 100, 25, 1), {"udid": "test-device"}
    monkeypatch.setattr(module, "_run_browser_job", fake_runner)
    result = module.evaluate(_context(tmp_path, parameters))
    assert htmls and "const MODEL_B64" in htmls[0]
    assert "__MODEL_B64__" not in htmls[0]
    assert result.metrics["latency_p50_ms"] > 0
    report = json.loads(result.artifacts["latency_report"].read_text())
    assert report["onnx_sha256"] == hashlib.sha256(b"immutable ONNX fixture").hexdigest()
    assert report["source_evaluation_result"] == "demo/completed-export-v1"


def test_detector_iphone_eval_uses_only_preexported_bytes(tmp_path, monkeypatch):
    path = "nanodet/evaluation_protocols/nanodet-ort-web-iphone-latency-v3.py"
    module = _module(path)
    parameters = {
        "warmup_runs": 5, "runs_per_block": 10,
        "measure_blocks": 10,
        "source_onnx_evaluation_result": "demo/completed-export-v1",
    }
    htmls = []
    def fake_runner(document):
        htmls.append(document)
        return _browser(10, 10, 5, 1, detector=True), {"udid": "test-device"}
    monkeypatch.setattr(module, "_run_page", fake_runner)
    result = module.evaluate(_context(tmp_path, parameters))
    assert htmls and "const n=dims[1]" in htmls[0]
    assert "__POSITIONS__" not in htmls[0]
    report = json.loads(result.artifacts["latency_report"].read_text())
    assert report["dense_positions"] == 2100
    assert report["onnx_bytes"] == len(b"immutable ONNX fixture")


@pytest.mark.parametrize("relative", [
    "tile-classifier/evaluation_protocols/tile-shape-ort-web-iphone-latency-v4",
    "nanodet/evaluation_protocols/nanodet-ort-web-iphone-latency-v3",
    "nanodet/evaluation_protocols/nanodet-onnx-export-v1",
])
def test_new_sealed_protocol_implementation_hash_matches(relative):
    py_path = ROOT / (relative + ".py")
    document = json.loads((ROOT / (relative + ".yaml")).read_text())
    assert document["implementation"]["sha256"] == hashlib.sha256(py_path.read_bytes()).hexdigest()
