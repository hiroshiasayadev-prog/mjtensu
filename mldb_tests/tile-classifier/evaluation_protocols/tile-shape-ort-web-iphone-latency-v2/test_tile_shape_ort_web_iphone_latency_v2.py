from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable


ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
PROTOCOL_ID = "tile-classifier/tile-shape-ort-web-iphone-latency-v2"
IMPLEMENTATION = (
    ROOT
    / "tile-classifier"
    / "evaluation_protocols"
    / "tile-shape-ort-web-iphone-latency-v2.py"
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 35)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.conv(images)).flatten(1))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "tile_shape_ort_web_iphone_latency_v2",
        IMPLEMENTATION,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _context(model: nn.Module, tmp_path: Path):
    return SimpleNamespace(
        parameters={
            "batch_size": 1,
            "warmup_runs": 1,
            "runs_per_block": 10,
            "measure_blocks": 10,
        },
        model=SimpleNamespace(module=model),
        work_dir=tmp_path,
    )


def test_evaluation_callable_contract() -> None:
    evaluate = _load_evaluation_callable(ROOT, PROTOCOL_ID)
    assert callable(evaluate)


def test_benchmark_document_pins_runtime_and_auto_runs() -> None:
    module = _load_module()
    document = module._benchmark_document(
        model_bytes=b"onnx-bytes",
        parity_input=module._parity_input(),
        batch_size=4,
        warmup_runs=7,
        runs_per_block=13,
        measure_blocks=11,
    )

    assert "onnxruntime-web@1.27.0" in document
    assert 'ort.env.wasm.numThreads = 1' in document
    assert 'ort.env.wasm.simd = true' in document
    assert 'graphOptimizationLevel: "all"' in document
    assert 'executionMode: "sequential"' in document
    assert "const BATCH_SIZE = 4;" in document
    assert "const WARMUP_RUNS = 7;" in document
    assert "const RUNS_PER_BLOCK = 13;" in document
    assert "const MEASURE_BLOCKS = 11;" in document
    assert "void main();" in document


def test_browser_samples_reject_mismatched_contract() -> None:
    module = _load_module()
    with pytest.raises(RuntimeError, match="batch_size mismatch"):
        module._browser_samples(
            {
                "ok": True,
                "benchmark": {
                    "batch_size": 4,
                    "warmup_runs": 1,
                    "runs_per_block": 10,
                    "measure_blocks": 10,
                    "total_measure_runs": 100,
                    "block_totals_ms": [10.0] * 10,
                    "samples_ms": [1.0] * 10,
                },
            },
            batch_size=1,
            warmup_runs=1,
            runs_per_block=10,
            measure_blocks=10,
        )


def test_direct_export_browser_result_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("onnx")
    module = _load_module()
    model = TinyClassifier().eval()
    with torch.no_grad():
        parity_logits = model(module._parity_input()).flatten().tolist()

    captured: dict[str, str] = {}

    def fake_browser_job(document: str):
        captured["document"] = document
        return (
            {
                "ok": True,
                "parity": {
                    "output_name": "logits",
                    "dims": [4, 35],
                    "logits": parity_logits,
                },
                "benchmark": {
                    "batch_size": 1,
                    "warmup_runs": 1,
                    "runs_per_block": 10,
                    "measure_blocks": 10,
                    "total_measure_runs": 100,
                    "block_totals_ms": [
                        10.0, 20.0, 30.0, 40.0, 50.0,
                        60.0, 70.0, 80.0, 90.0, 100.0,
                    ],
                    "samples_ms": [
                        1.0, 2.0, 3.0, 4.0, 5.0,
                        6.0, 7.0, 8.0, 9.0, 10.0,
                    ],
                },
                "environment": {
                    "provider": "wasm-simd",
                    "num_threads": 1,
                    "wasm_proxy": False,
                    "graph_optimization_level": "all",
                    "execution_mode": "sequential",
                    "onnxruntime_web_version_expected": "1.27.0",
                    "onnxruntime_web_version_reported": "1.27.0",
                    "hardware_concurrency": 4,
                    "cross_origin_isolated": False,
                    "secure_context": False,
                    "user_agent": "test-safari",
                },
            },
            {
                "ok": True,
                "udid": "private-device-id",
            },
        )

    monkeypatch.setattr(module, "_run_browser_job", fake_browser_job)
    candidate = module.evaluate(_context(model, tmp_path))

    assert candidate.metrics == {
        "latency_p50_ms": 5.5,
        "latency_p95_ms": 10.0,
        "latency_mean_ms": 5.5,
    }
    assert candidate.artifacts["onnx_model"].is_file()
    report = json.loads(
        candidate.artifacts["latency_report"].read_text(encoding="utf-8")
    )
    assert report["export"]["mode"] == "direct-torch-export"
    assert report["export"]["torch_parity"]["allclose"] is True
    assert report["export"]["browser_parity"]["allclose"] is True
    assert report["benchmark"]["runs_per_block"] == 10
    assert report["benchmark"]["measure_blocks"] == 10
    assert report["benchmark"]["total_measure_runs"] == 100
    assert report["benchmark"]["block_totals_ms"] == [
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
    ]
    assert report["runtime"]["onnxruntime_web_version"] == "1.27.0"
    assert report["environment"]["runner_device_id_sha256"] == hashlib.sha256(
        b"private-device-id"
    ).hexdigest()
    assert "private-device-id" not in json.dumps(report)
    assert "onnxruntime-web@1.27.0" in captured["document"]
