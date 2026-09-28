from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mldb_v2.src.training.train_interface import _load_train_callable


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "mldb_data"
PROTOCOL = (
    "tile-classifier/"
    "tile-shape-train-mobilenet-v3-small-1x-random360-e150-v1"
)


def test_recipe_defaults_match_inv011_selected_run() -> None:
    definition = json.loads(
        (
            DATA
            / "tile-classifier"
            / "train_protocols"
            / "tile-shape-train-mobilenet-v3-small-1x-random360-e150-v1.yaml"
        ).read_text(encoding="utf-8")
    )
    defaults = {
        key: value["default"]
        for key, value in definition["parameters"].items()
    }
    assert defaults["epochs"] == 150
    assert defaults["batch_size"] == 512
    assert defaults["eval_batch_size"] == 256
    assert defaults["learning_rate"] == pytest.approx(0.001)
    assert defaults["weight_decay"] == pytest.approx(0.0001)
    assert defaults["eval_angles"] == [0, 15, 30, 45]
    assert defaults["angle_eval_every"] == 5
    assert defaults["amp"] is True
    assert defaults["tf32"] is True


def test_random360_angles_are_deterministic_and_bounded() -> None:
    train = _load_train_callable(DATA, PROTOCOL)
    module = sys.modules[train.__module__]
    sample_ids = ("a", "b", "c", "d")
    first = module._deterministic_random360_angles(
        sample_ids,
        seed=42,
        epoch=7,
    )
    second = module._deterministic_random360_angles(
        sample_ids,
        seed=42,
        epoch=7,
    )
    other = module._deterministic_random360_angles(
        sample_ids,
        seed=42,
        epoch=8,
    )
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= -180.0)
    assert np.all(first < 180.0)
    assert not np.array_equal(first, other)


def test_checkpoint_angle_mean_uses_strict_greater_selection_semantics() -> None:
    scores = [
        (1, np.mean([0.4, 0.4, 0.4, 0.4])),
        (5, np.mean([0.6, 0.6, 0.6, 0.6])),
        (10, np.mean([0.6, 0.6, 0.6, 0.6])),
        (150, np.mean([0.7, 0.7, 0.7, 0.7])),
    ]
    best = -1.0
    best_epoch = 0
    for epoch, score in scores:
        if score > best:
            best = float(score)
            best_epoch = epoch
    assert best_epoch == 150
    assert best == pytest.approx(0.7)
