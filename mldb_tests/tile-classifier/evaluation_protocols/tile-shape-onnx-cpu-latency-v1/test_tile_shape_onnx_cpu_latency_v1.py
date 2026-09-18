from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 35)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.conv(images)).flatten(1))


def test_evaluation_callable_contract() -> None:
    evaluate = _load_evaluation_callable(
        ROOT,
        "tile-classifier/tile-shape-onnx-cpu-latency-v1",
    )
    assert callable(evaluate)


def test_cpu_latency_evaluator_exports_and_benchmarks(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    evaluate = _load_evaluation_callable(
        ROOT,
        "tile-classifier/tile-shape-onnx-cpu-latency-v1",
    )
    context = SimpleNamespace(
        parameters={
            "batch_size": 1,
            "warmup_runs": 1,
            "measure_runs": 10,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
        },
        model=SimpleNamespace(module=TinyClassifier()),
        work_dir=tmp_path,
    )

    candidate = evaluate(context)

    assert set(candidate.metrics) == {"latency_p50_ms", "latency_p95_ms", "latency_mean_ms"}
    assert all(float(value) > 0.0 for value in candidate.metrics.values())
    assert candidate.artifacts["onnx_model"].is_file()
    report = json.loads(candidate.artifacts["latency_report"].read_text(encoding="utf-8"))
    assert report["environment"]["active_providers"] == ["CPUExecutionProvider"]
    assert report["benchmark"]["measure_runs"] == 10
    assert len(report["benchmark"]["samples_ms"]) == 10
    assert report["input"]["shape"] == [1, 1, 64, 64]
