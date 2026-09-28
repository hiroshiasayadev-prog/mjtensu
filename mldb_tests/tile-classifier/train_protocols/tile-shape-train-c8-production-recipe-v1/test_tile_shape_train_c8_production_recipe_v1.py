from __future__ import annotations

import json
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


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "mldb_data"
PROTOCOL = "tile-classifier/tile-shape-train-c8-production-recipe-v1"
ARCHITECTURE = "tile-classifier/tile-c8-gray35-v1"


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def report_scalar(
        self, *, group: str, series: str, value: int | float, step: int
    ) -> None:
        self.events.append(
            {"group": group, "series": series, "value": value, "step": step}
        )


class TinyModel(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(float(value)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((images.shape[0], 35), device=images.device)
        return logits + self.marker * 0.0


def _write_corpus(root: Path) -> None:
    with sqlite3.connect(root / "dataset.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sample (sample_id TEXT PRIMARY KEY, split TEXT, "
            "image_gray_u8 BLOB, class_index INTEGER)"
        )
        for split, count in (("train", 4), ("manual_val", 2), ("jp_val", 2)):
            for index in range(count):
                image = bytes([30 + index * 20]) * (64 * 64)
                connection.execute(
                    "INSERT INTO sample VALUES (?, ?, ?, ?)",
                    (f"{split}-{index}", split, image, index % 2),
                )


def _parameters() -> dict[str, object]:
    return {
        "epochs": 4,
        "batch_size": 2,
        "eval_batch_size": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "rotation_augment_deg": 22.5,
        "eval_angles": [0.0, 15.0, 30.0, 45.0],
        "angle_eval_every": 3,
        "amp": False,
        "tf32": False,
        "cache_device": "cpu",
        "cache_vram_fraction": 0.25,
    }


def _context(root: Path, telemetry: RecordingTelemetry) -> TrainContext:
    work = root / "work"
    work.mkdir()
    return TrainContext(
        task=cast(Any, None),
        corpus=MaterializedCorpus(definition=cast(Any, None), root=root),
        architecture=cast(Any, {"id": ARCHITECTURE}),
        model=TinyModel(),
        seed=42,
        parameters=cast(Any, _parameters()),
        telemetry=telemetry,
        work_dir=work,
    )


def test_recipe_defaults_match_recovered_production_conditions() -> None:
    definition = json.loads(
        (
            DATA
            / "tile-classifier"
            / "train_protocols"
            / "tile-shape-train-c8-production-recipe-v1.yaml"
        ).read_text(encoding="utf-8")
    )
    defaults = {
        key: value["default"] for key, value in definition["parameters"].items()
    }
    assert defaults["epochs"] == 50
    assert defaults["batch_size"] == 512
    assert defaults["eval_batch_size"] == 4096
    assert defaults["learning_rate"] == pytest.approx(0.001)
    assert defaults["weight_decay"] == pytest.approx(0.0001)
    assert defaults["rotation_augment_deg"] == pytest.approx(22.5)
    assert defaults["eval_angles"] == [0.0, 15.0, 30.0, 45.0]
    assert defaults["angle_eval_every"] == 5
    assert defaults["amp"] is True
    assert defaults["tf32"] is True
    assert "perspective_augment" not in defaults
    assert "shear_augment" not in defaults
    assert "stretch_augment" not in defaults


def test_seed_precedes_rebuild_and_only_full_sweeps_select_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_corpus(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train = _load_train_callable(DATA, PROTOCOL)
    module = sys.modules[train.__module__]

    initial_values: list[float] = []

    def fake_build() -> nn.Module:
        value = float(torch.rand(1).item())
        initial_values.append(value)
        return TinyModel(value)

    def fake_train_epoch(model, split, *, epoch: int, **kwargs):
        del split, kwargs
        model.marker.data.fill_(float(epoch))
        return {"loss": 1.0 / epoch, "accuracy": 0.5}

    full_scores = {1: 0.40, 3: 0.60, 4: 0.55}

    def fake_evaluate(model, splits, *, angles, **kwargs):
        del splits, kwargs
        epoch = int(round(float(model.marker.detach().item())))
        if len(angles) == 1:
            score = 0.99
        else:
            score = full_scores[epoch]
        angle_payload = {module._angle_key(float(angle)): score for angle in angles}
        return {
            "manual_val": {"angles": dict(angle_payload)},
            "jp_val": {"angles": dict(angle_payload)},
        }

    monkeypatch.setattr(module, "build_c8_tile_shape_classifier", fake_build)
    monkeypatch.setattr(module, "_train_one_epoch", fake_train_epoch)
    monkeypatch.setattr(module, "_evaluate_all", fake_evaluate)

    torch.manual_seed(42)
    expected_initial = float(torch.rand(1).item())
    telemetry = RecordingTelemetry()
    returned = train(_context(tmp_path, telemetry))

    assert initial_values == pytest.approx([expected_initial])
    assert isinstance(returned, TinyModel)
    assert returned.marker.item() == pytest.approx(3.0)

    checkpoint_steps = [
        event["step"]
        for event in telemetry.events
        if event["series"] == "checkpoint_manual_angle_mean"
    ]
    assert checkpoint_steps == [1, 3, 4]
