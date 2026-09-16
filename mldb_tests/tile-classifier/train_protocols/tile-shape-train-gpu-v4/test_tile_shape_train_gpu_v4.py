from __future__ import annotations

import copy
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from mldb_v2.src.training.train_interface import (
    MaterializedCorpus,
    TrainContext,
    _load_train_callable,
)

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
V3 = "tile-classifier/tile-shape-train-gpu-v3"
V4 = "tile-classifier/tile-shape-train-gpu-v4"


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def report_scalar(
        self, *, group: str, series: str, value: int | float, step: int
    ) -> None:
        self.events.append(
            {"group": group, "series": series, "value": value, "step": step}
        )


class FailingTelemetry:
    def report_scalar(
        self, *, group: str, series: str, value: int | float, step: int
    ) -> None:
        raise RuntimeError("telemetry rejected")


def _write_corpus(root: Path, count: int = 5) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / "dataset.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sample (sample_id INTEGER PRIMARY KEY, split TEXT, "
            "image_gray_u8 BLOB, class_index INTEGER)"
        )
        for index in range(count):
            image = bytes([20 + index * 30]) * (64 * 64)
            connection.execute(
                "INSERT INTO sample VALUES (?, 'train', ?, ?)",
                (index + 1, image, index % 5),
            )


def _model() -> nn.Module:
    return nn.Sequential(nn.AvgPool2d(64), nn.Flatten(), nn.Linear(1, 35))


def _parameters(*, epochs: int) -> dict[str, object]:
    return {
        "epochs": epochs,
        "batch_size": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "rotation_augment_deg": 0.0,
        "perspective_augment": 0.0,
        "shear_augment": 0.0,
        "stretch_augment": 0.0,
        "projective_augment_probability": 0.0,
        "amp": False,
        "tf32": False,
        "cache_device": "cpu",
        "cache_vram_fraction": 0.5,
    }


def _context(
    root: Path, model: nn.Module, telemetry: Any, *, epochs: int
) -> TrainContext:
    work_dir = root / "work"
    work_dir.mkdir(exist_ok=True)
    return TrainContext(
        task=cast(Any, None),
        corpus=MaterializedCorpus(definition=cast(Any, None), root=root),
        architecture=cast(Any, None),
        model=model,
        seed=42,
        parameters=cast(Any, _parameters(epochs=epochs)),
        telemetry=telemetry,
        work_dir=work_dir,
    )


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.mark.parametrize("epochs", [1, 3])
def test_reports_one_sample_weighted_loss_per_completed_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, epochs: int
) -> None:
    _force_cpu(monkeypatch)
    _write_corpus(tmp_path)
    reporter = RecordingTelemetry()
    context = _context(tmp_path, _model(), reporter, epochs=epochs)
    returned = _load_train_callable(ROOT, V4)(context)

    assert returned is context.model
    assert len(reporter.events) == epochs
    assert [event["step"] for event in reporter.events] == list(range(1, epochs + 1))
    for event in reporter.events:
        assert event["group"] == "optimization"
        assert event["series"] == "cross_entropy_loss"
        assert type(event["value"]) is float
        assert math.isfinite(cast(float, event["value"]))


def test_epoch_loss_is_sample_weighted_not_batch_weighted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_cpu(monkeypatch)
    _write_corpus(tmp_path, count=5)
    reporter = RecordingTelemetry()
    train = _load_train_callable(ROOT, V4)
    module = sys.modules[train.__module__]

    def fake_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        value = 5.0 if target.shape[0] == 1 else 2.0
        return logits.sum() * 0.0 + value

    monkeypatch.setattr(module.F, "cross_entropy", fake_cross_entropy)
    train(_context(tmp_path, _model(), reporter, epochs=1))

    assert len(reporter.events) == 1
    assert reporter.events[0]["value"] == pytest.approx(2.6)


def test_telemetry_failure_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_cpu(monkeypatch)
    _write_corpus(tmp_path)
    context = _context(tmp_path, _model(), FailingTelemetry(), epochs=1)

    with pytest.raises(RuntimeError, match="telemetry rejected"):
        _load_train_callable(ROOT, V4)(context)


def test_telemetry_does_not_change_learned_state_vs_v3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_cpu(monkeypatch)
    _write_corpus(tmp_path)
    torch.manual_seed(1234)
    base = _model()
    v3_model = copy.deepcopy(base)
    v4_model = copy.deepcopy(base)

    v3 = _load_train_callable(ROOT, V3)
    v4 = _load_train_callable(ROOT, V4)
    v3(_context(tmp_path, v3_model, RecordingTelemetry(), epochs=2))
    reporter = RecordingTelemetry()
    v4(_context(tmp_path, v4_model, reporter, epochs=2))

    assert len(reporter.events) == 2
    for name, tensor in v3_model.state_dict().items():
        assert torch.equal(tensor, v4_model.state_dict()[name]), name
