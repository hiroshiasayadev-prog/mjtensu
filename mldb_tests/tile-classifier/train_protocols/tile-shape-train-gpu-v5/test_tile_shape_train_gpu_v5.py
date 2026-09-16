from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from mldb_v2.src.training.train_interface import MaterializedCorpus, TrainContext, _load_train_callable

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
V5 = "tile-classifier/tile-shape-train-gpu-v5"


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def report_scalar(self, *, group: str, series: str, value: int | float, step: int) -> None:
        self.events.append({"group": group, "series": series, "value": value, "step": step})


def _write_corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / "dataset.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sample (sample_id TEXT PRIMARY KEY, split TEXT, image_gray_u8 BLOB, "
            "class_index INTEGER, original_width INTEGER, original_height INTEGER)"
        )
        for split, count in (("train", 10), ("manual_val", 6)):
            for index in range(count):
                image = bytes([20 + (index % 5) * 30]) * (64 * 64)
                connection.execute(
                    "INSERT INTO sample VALUES (?, ?, ?, ?, ?, ?)",
                    (f"{split}-{index}", split, image, index % 3 + 4, 42 + index % 3, 64),
                )


def _model() -> nn.Module:
    return nn.Sequential(nn.AvgPool2d(64), nn.Flatten(), nn.Linear(1, 35))


def _parameters(recipe: str = "inv013-mix-heavy-v1") -> dict[str, object]:
    return {
        "epochs": 1,
        "batch_size": 5,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "rotation_augment_deg": 180.0,
        "augmentation_recipe": recipe,
        "checkpoint_eval_every": 1,
        "amp": False,
        "tf32": False,
        "cache_device": "cpu",
        "cache_vram_fraction": 0.5,
    }


def _context(root: Path, telemetry: RecordingTelemetry, recipe: str = "inv013-mix-heavy-v1") -> TrainContext:
    (root / "work").mkdir(exist_ok=True)
    return TrainContext(
        task=cast(Any, None),
        corpus=MaterializedCorpus(definition=cast(Any, None), root=root),
        architecture=cast(Any, None),
        model=_model(),
        seed=42,
        parameters=cast(Any, _parameters(recipe)),
        telemetry=telemetry,
        work_dir=root / "work",
    )


def test_random360_range_and_recipe_are_deterministic() -> None:
    train = _load_train_callable(ROOT, V5)
    module = sys.modules[train.__module__]
    sample_ids = ["a", "b", "c", "d"]
    first = module._deterministic_angles(sample_ids, seed=42, epoch=3, maximum=180.0)
    second = module._deterministic_angles(sample_ids, seed=42, epoch=3, maximum=180.0)
    assert (first == second).all()
    assert (first >= -180.0).all() and (first < 180.0).all()
    choices1 = module._deterministic_choices(sample_ids, recipe="inv013-mix-heavy-v1", seed=42, epoch=3)
    choices2 = module._deterministic_choices(sample_ids, recipe="inv013-mix-heavy-v1", seed=42, epoch=3)
    assert (choices1 == choices2).all()


def test_v5_trains_with_mix_heavy_and_reports_observability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _write_corpus(tmp_path)
    reporter = RecordingTelemetry()
    context = _context(tmp_path, reporter)
    returned = _load_train_callable(ROOT, V5)(context)
    assert returned is context.model
    observed = {(event["group"], event["series"]) for event in reporter.events}
    assert ("optimization", "cross_entropy_loss") in observed
    assert ("validation", "checkpoint_angle_mean") in observed
    assert ("validation", "accuracy_0deg") in observed
    for branch in ("original", "a0-random360", "a1-anisotropic-affine", "a2-perspective", "a3-perspective-recrop"):
        assert ("augmentation_mix", branch) in observed


def test_rotation_bound_rejects_above_180(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _write_corpus(tmp_path)
    train = _load_train_callable(ROOT, V5)
    module = sys.modules[train.__module__]
    parameters = _parameters()
    parameters["rotation_augment_deg"] = 180.1
    with pytest.raises(ValueError, match=r"\[0,180\]"):
        module._validate_parameters_v5(parameters)
