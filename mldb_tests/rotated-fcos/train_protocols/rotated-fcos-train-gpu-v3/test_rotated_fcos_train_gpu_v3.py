from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from mldb_v2.src.training.train_interface import _load_train_callable

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
PROTOCOL_ID = "rotated-fcos/rotated-fcos-train-gpu-v3"


class _RecordingTelemetry:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[dict[str, object]] = []
        self.fail = fail

    def report_scalar(self, *, group: str, series: str, value: int | float, step: int) -> None:
        if self.fail:
            raise ValueError("telemetry rejected")
        self.events.append({"group": group, "series": series, "value": value, "step": step})


class _StateValue:
    def __init__(self, value: int) -> None:
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return _StateValue(self.value)


class _Backbone:
    def parameters(self):
        return ()

    def eval(self) -> None:
        return None


class _Model:
    def __init__(self) -> None:
        self.backbone = _Backbone()
        self.marker = 0
        self.loaded_marker: int | None = None

    def to(self, _device):
        return self

    def parameters(self):
        return ()

    def train(self):
        return self

    def state_dict(self):
        return {"marker": _StateValue(self.marker)}

    def load_state_dict(self, state) -> None:
        self.loaded_marker = state["marker"].value


class _EmptyLoader:
    def __len__(self) -> int:
        return 1

    def __iter__(self):
        return iter(())


class _Optimizer:
    def __init__(self, *_args, **_kwargs) -> None:
        pass


def _run_train(monkeypatch: pytest.MonkeyPatch, keys, *, epochs: int | None = None, patience: int = 0, reporter=None):
    train = _load_train_callable(ROOT, PROTOCOL_ID)
    globals_ = train.__globals__
    monkeypatch.setitem(globals_, "DetectionDataset", lambda *_args, **_kwargs: object())
    monkeypatch.setitem(globals_, "DataLoader", lambda *_args, **_kwargs: _EmptyLoader())
    monkeypatch.setitem(globals_, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(globals_["torch"].cuda, "is_available", lambda: True)
    monkeypatch.setattr(globals_["torch"].backends.cudnn, "benchmark", False)
    monkeypatch.setattr(globals_["torch"].optim, "AdamW", _Optimizer)
    monkeypatch.setattr(globals_["torch"].optim.lr_scheduler, "LambdaLR", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(globals_["torch"].cuda.amp, "GradScaler", lambda **_kwargs: object())

    model = _Model()
    values = iter(keys)
    calls = {"count": 0}

    def fake_validation_key(current_model, _loader, **_kwargs):
        calls["count"] += 1
        current_model.marker = calls["count"]
        return next(values)

    monkeypatch.setitem(globals_, "validation_key", fake_validation_key)
    telemetry = reporter or _RecordingTelemetry()
    parameters = {
        "train_split": "train",
        "validation_split": "val",
        "batch_size": 1,
        "workers": 0,
        "pretrained_backbone": False,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "epochs": epochs if epochs is not None else len(keys),
        "warmup_epochs": 0.0,
        "amp": False,
        "freeze_backbone_epochs": 0,
        "early_stop_patience": patience,
        "center_radius": 1.5,
        "score_threshold": 0.2,
        "nms_iou_threshold": 0.45,
        "match_iou_threshold": 0.5,
        "max_detections": 64,
    }
    context = SimpleNamespace(
        parameters=parameters,
        seed=42,
        model=model,
        corpus=SimpleNamespace(root=Path(".")),
        telemetry=telemetry,
    )
    result = train(context)
    return telemetry, result, calls["count"]


def test_train_callable_contract() -> None:
    assert callable(_load_train_callable(ROOT, PROTOCOL_ID))


def test_one_completed_epoch_reports_exact_validation_series_and_positive_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry, _model, count = _run_train(monkeypatch, [(0.6, 0.7, 0.8, -1.25)])
    assert count == 1
    assert telemetry.events == [
        {"group": "validation", "series": "f1", "value": 0.6, "step": 1},
        {"group": "validation", "series": "recall", "value": 0.7, "step": 1},
        {"group": "validation", "series": "mean_iou", "value": 0.8, "step": 1},
        {"group": "validation", "series": "loss", "value": 1.25, "step": 1},
    ]
    for event in telemetry.events:
        assert math.isfinite(float(event["value"]))
        assert float(event["value"]) >= 0.0


def test_each_completed_epoch_reports_once_per_series_and_preserves_best_key_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        (0.4, 0.8, 0.7, -1.5),
        (0.5, 0.2, 0.1, -3.0),
        (0.5, 0.2, 0.1, -2.0),
    ]
    telemetry, model, count = _run_train(monkeypatch, keys)
    assert count == 3
    for series in ("f1", "recall", "mean_iou", "loss"):
        events = [event for event in telemetry.events if event["series"] == series]
        assert [event["step"] for event in events] == [1, 2, 3]
        assert len(events) == 3
    assert [event["value"] for event in telemetry.events if event["series"] == "loss"] == [1.5, 3.0, 2.0]
    assert model.loaded_marker == 3


def test_early_stop_reports_stopping_epoch_without_phantom_future_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        (0.5, 0.5, 0.5, -1.0),
        (0.4, 0.9, 0.9, -0.5),
        (0.9, 0.9, 0.9, -0.1),
    ]
    telemetry, model, count = _run_train(monkeypatch, keys, epochs=5, patience=1)
    assert count == 2
    assert sorted({event["step"] for event in telemetry.events}) == [1, 2]
    assert len(telemetry.events) == 8
    assert model.loaded_marker == 1


def test_telemetry_failure_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    reporter = _RecordingTelemetry(fail=True)
    with pytest.raises(ValueError, match="telemetry rejected"):
        _run_train(monkeypatch, [(0.6, 0.7, 0.8, -1.25)], reporter=reporter)
