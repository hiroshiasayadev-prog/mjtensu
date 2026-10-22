from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable


ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
PROTOCOL_ID = "mldb-smoke/c8-narrow-late2-fieldwise-dw-pw-iphone-smoke-v1"


def _context(tmp_path: Path):
    return SimpleNamespace(
        parameters={
            "batch_size": 1,
            "warmup_runs": 1,
            "runs_per_block": 10,
            "measure_blocks": 10,
        },
        model=SimpleNamespace(module=None),
        work_dir=tmp_path,
    )


def test_evaluation_callable_contract() -> None:
    assert callable(_load_evaluation_callable(ROOT, PROTOCOL_ID))


def test_iphone_protocol_uses_v3_semantics_without_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("escnn")
    pytest.importorskip("onnx")
    evaluate = _load_evaluation_callable(ROOT, PROTOCOL_ID)
    helper = evaluate.__globals__["evaluate_iphone"].__globals__

    source = helper["build_prototype"]().eval()
    with torch.no_grad():
        parity_logits = source(helper["_parity_input"]()).flatten().tolist()

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
                    "block_totals_ms": [10.0 * value for value in range(1, 11)],
                    "samples_ms": [float(value) for value in range(1, 11)],
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
            {"ok": True, "udid": "private-device-id"},
        )

    monkeypatch.setitem(helper, "_run_browser_job", fake_browser_job)
    candidate = evaluate(_context(tmp_path))
    report = json.loads(candidate.artifacts["smoke_report"].read_text(encoding="utf-8"))

    assert candidate.metrics == {
        "latency_p50_ms": 5.5,
        "latency_p95_ms": 10.0,
        "latency_mean_ms": 5.5,
    }
    assert report["export"]["browser_parity"]["allclose"] is True
    expected_signature = [
        {"kernel": [5, 5], "group": 1},
        {"kernel": [3, 3], "group": 1},
        {"kernel": [3, 3], "group": 8},
        {"kernel": [1, 1], "group": 1},
        {"kernel": [3, 3], "group": 16},
        {"kernel": [1, 1], "group": 1},
    ]
    assert report["export"]["conv_kernel_group_signature"] == {
        "expected": expected_signature,
        "torch_export": expected_signature,
        "onnx": expected_signature,
    }
    assert "onnxruntime-web@1.27.0" in captured["document"]
    assert "ort.env.wasm.numThreads = 1" in captured["document"]
    assert "ort.env.wasm.simd = true" in captured["document"]
    assert 'graphOptimizationLevel: "all"' in captured["document"]
    assert 'executionMode: "sequential"' in captured["document"]
