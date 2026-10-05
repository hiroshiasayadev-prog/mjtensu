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
PROTOCOL_ID = "tile-classifier/tile-shape-onnx-cpu-latency-v3"


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 35)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.conv(images)).flatten(1))


class _FakeExportableBackbone(nn.Module):
    def __init__(self, *, group_size: int, fields: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, group_size * fields, 1, bias=False)
        self.out_type = SimpleNamespace(size=group_size * fields)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.conv(images)

    def export(self) -> nn.Module:
        return self


class _FakeGroupPool(nn.Module):
    def __init__(self, *, group_size: int, fields: int) -> None:
        super().__init__()
        self.group_size = group_size
        self.fields = fields
        self.in_type = SimpleNamespace(size=group_size * fields)
        self.out_type = SimpleNamespace(size=fields)

    def forward(self, features: torch.Tensor):
        batch, _channels, height, width = features.shape
        pooled = features.reshape(
            batch, self.fields, self.group_size, height, width
        ).amax(dim=2)
        return SimpleNamespace(tensor=pooled)


class TinyCyclicClassifier(nn.Module):
    def __init__(self, group_size: int = 4, fields: int = 2) -> None:
        super().__init__()
        self.field_counts = (fields,)
        self.equivariant_backbone = _FakeExportableBackbone(
            group_size=group_size, fields=fields
        )
        self.group_pool = _FakeGroupPool(group_size=group_size, fields=fields)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(fields, 35))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.equivariant_backbone(images)
        invariant = self.group_pool(features).tensor
        return self.classifier(self.spatial_pool(invariant))


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
    assert report["export"]["mode"] == "cyclic-equivariant-tensor-export"
    assert report["export"]["torch_parity"]["allclose"] is True
    assert report["export"]["torch_parity"]["prediction_mismatches"] == 0
    assert report["export"]["onnx_parity"]["allclose"] is True
    assert report["export"]["onnx_parity"]["prediction_mismatches"] == 0
    assert report["benchmark"]["measure_runs"] == 10


def test_c4_style_tensor_export_and_benchmark(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    evaluate = _load_evaluation_callable(ROOT, PROTOCOL_ID)
    candidate = evaluate(_context(TinyCyclicClassifier(group_size=4), tmp_path))

    report = _report(candidate)
    assert report["export"]["mode"] == "cyclic-equivariant-tensor-export"
    assert report["export"]["torch_parity"]["allclose"] is True
    assert report["export"]["onnx_parity"]["allclose"] is True
