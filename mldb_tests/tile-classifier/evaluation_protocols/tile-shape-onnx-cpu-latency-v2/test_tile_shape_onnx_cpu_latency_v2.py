from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mldb_v2.src.catalog.architecture_build import _load_architecture_build
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable


ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
PROTOCOL_ID = "tile-classifier/tile-shape-onnx-cpu-latency-v2"


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 35)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.conv(images)).flatten(1))


def _context(model: nn.Module, tmp_path: Path):
    return SimpleNamespace(
        parameters={
            "batch_size": 1,
            "warmup_runs": 1,
            "measure_runs": 10,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
        },
        model=SimpleNamespace(module=model),
        work_dir=tmp_path,
    )


def _report(candidate) -> dict[str, object]:
    return json.loads(
        candidate.artifacts["latency_report"].read_text(encoding="utf-8")
    )


def test_evaluation_callable_contract() -> None:
    evaluate = _load_evaluation_callable(ROOT, PROTOCOL_ID)
    assert callable(evaluate)


def test_direct_torch_export_and_benchmark(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    evaluate = _load_evaluation_callable(ROOT, PROTOCOL_ID)
    candidate = evaluate(_context(TinyClassifier(), tmp_path))

    assert set(candidate.metrics) == {
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_mean_ms",
    }
    assert all(float(value) > 0.0 for value in candidate.metrics.values())
    report = _report(candidate)
    assert report["export"]["mode"] == "direct-torch-export"
    assert report["export"]["torch_parity"]["allclose"] is True
    assert report["export"]["onnx_parity"]["allclose"] is True
    assert report["environment"]["active_providers"] == ["CPUExecutionProvider"]


def test_c8_tensor_export_and_benchmark(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("escnn")
    build = _load_architecture_build(
        ROOT,
        "tile-classifier/tile-c8-gray35-v1",
    )
    evaluate = _load_evaluation_callable(ROOT, PROTOCOL_ID)
    candidate = evaluate(_context(build(), tmp_path))

    assert candidate.artifacts["onnx_model"].is_file()
    report = _report(candidate)
    assert report["export"]["mode"] == "c8-equivariant-tensor-export"
    assert report["export"]["torch_parity"]["allclose"] is True
    assert report["export"]["torch_parity"]["prediction_mismatches"] == 0
    assert report["export"]["onnx_parity"]["allclose"] is True
    assert report["export"]["onnx_parity"]["prediction_mismatches"] == 0
    assert report["benchmark"]["measure_runs"] == 10
