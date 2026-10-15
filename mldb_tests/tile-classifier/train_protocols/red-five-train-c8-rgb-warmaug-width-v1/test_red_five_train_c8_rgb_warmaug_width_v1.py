from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from mldb_v2.src.training.train_interface import MaterializedCorpus, TrainContext, _load_train_callable

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "mldb_data"
PROTOCOL = "tile-classifier/red-five-train-c8-rgb-warmaug-width-v1"
ARCHITECTURE = "tile-classifier/red-five-c8-rgb-dw222-fields6-12-24-48-v1"


class Telemetry:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
    def report_scalar(self, *, group: str, series: str, value: int | float, step: int) -> None:
        self.events.append({"group": group, "series": series, "value": value, "step": step})


class TinyModel(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(float(value)))
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((images.shape[0], 2), device=images.device)
        return logits + self.marker * 0.0


def write_corpus(root: Path) -> None:
    with sqlite3.connect(root / "dataset.sqlite") as c:
        c.execute("CREATE TABLE sample (sample_id TEXT PRIMARY KEY, split TEXT, image_rgb_u8 BLOB, is_red INTEGER, train_repeat INTEGER)")
        for split, count in (("train", 6), ("jp_val", 4)):
            for index in range(count):
                pixel = bytes([30 + index, 80 + index, 120 + index]) * (64 * 64)
                c.execute("INSERT INTO sample VALUES (?,?,?,?,?)", (f"{split}-{index}", split, pixel, index % 2, 2 if split == "train" and index == 0 else 1))


def parameters() -> dict[str, object]:
    return {
        "epochs": 4, "batch_size": 2, "eval_batch_size": 2,
        "learning_rate": 0.001, "weight_decay": 0.0001,
        "rotation_augment_deg": 22.5, "eval_angles": [0.0, 15.0, 30.0, 45.0],
        "angle_eval_every": 3, "warm_augment_prob": 0.5,
        "warm_strength_min": 0.1, "warm_strength_max": 1.0,
        "warm_red_gain_max": 1.5, "warm_green_gain_max": 1.08,
        "warm_blue_gain_min": 0.45, "warm_exposure_min": 0.65,
        "warm_exposure_max": 1.15, "amp": False, "tf32": False,
        "cache_device": "cpu", "cache_vram_fraction": 0.25,
    }


def context(root: Path, telemetry: Telemetry) -> TrainContext:
    return TrainContext(
        task=cast(Any, None), corpus=MaterializedCorpus(definition=cast(Any, None), root=root),
        architecture=cast(Any, {"id": ARCHITECTURE}), model=TinyModel(), seed=42,
        parameters=cast(Any, parameters()), telemetry=telemetry, work_dir=root / "work",
    )


def test_defaults_match_selected_red_five_recipe() -> None:
    d=json.loads((DATA / "tile-classifier/train_protocols/red-five-train-c8-rgb-warmaug-width-v1.yaml").read_text())
    defaults={k:v["default"] for k,v in d["parameters"].items()}
    assert defaults["epochs"] == 50
    assert defaults["batch_size"] == 1024
    assert defaults["eval_batch_size"] == 4096
    assert defaults["rotation_augment_deg"] == pytest.approx(22.5)
    assert defaults["eval_angles"] == [0.0,15.0,30.0,45.0]
    assert defaults["warm_augment_prob"] == pytest.approx(0.5)
    assert defaults["warm_red_gain_max"] == pytest.approx(1.5)
    assert defaults["warm_blue_gain_min"] == pytest.approx(0.45)
    assert defaults["warm_exposure_min"] == pytest.approx(0.65)
    assert defaults["warm_exposure_max"] == pytest.approx(1.15)


def test_seed_precedes_rebuild_and_only_full_angle_sweeps_select(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_corpus(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train=_load_train_callable(DATA, PROTOCOL)
    module=sys.modules[train.__module__]
    initial=[]
    def fake_build(*, fields):
        assert tuple(fields) == (6, 12, 24, 48)
        value=float(torch.rand(1).item()); initial.append(value); return TinyModel(value)
    def fake_train_epoch(model, split, *, epoch: int, **kwargs):
        del split, kwargs
        model.marker.data.fill_(float(epoch))
        return {"loss":1.0/epoch,"accuracy":0.5,"warm_augmented_fraction":0.5}
    full_scores={1:0.40,3:0.60,4:0.55}
    def fake_evaluate(model, split, *, angle: float, **kwargs):
        del split, angle, kwargs
        epoch=int(round(float(model.marker.detach().item())))
        score=full_scores.get(epoch,0.99)
        return {"accuracy":score,"balanced_accuracy":score,"red_recall":score,"normal_specificity":score}
    monkeypatch.setattr(module,"build_c8_rgb_stage234_dw222_width_classifier",fake_build)
    monkeypatch.setattr(module,"_train_epoch",fake_train_epoch)
    monkeypatch.setattr(module,"_evaluate",fake_evaluate)
    torch.manual_seed(42); expected=float(torch.rand(1).item())
    telemetry=Telemetry(); returned=train(context(tmp_path,telemetry))
    assert initial == pytest.approx([expected])
    assert isinstance(returned,TinyModel)
    assert returned.marker.item() == pytest.approx(3.0)
    steps=[e["step"] for e in telemetry.events if e["series"]=="checkpoint_jp_angle_mean_balanced_accuracy"]
    assert steps == [1,3,4]
