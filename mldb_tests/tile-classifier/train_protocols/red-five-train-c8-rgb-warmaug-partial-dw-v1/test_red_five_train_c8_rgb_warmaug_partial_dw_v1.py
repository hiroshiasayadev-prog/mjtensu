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
PROTOCOL = "tile-classifier/red-five-train-c8-rgb-warmaug-partial-dw-v1"
ARCH = "tile-classifier/red-five-c8-rgb-aggressive-partial-dw-stage34-m2-fields6-12-24-48-v1"


class Telemetry:
    def report_scalar(self, *, group, series, value, step):
        pass


class TinyModel(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(float(value)))
        self.field_counts = (6, 12, 24, 48)
        self.stage_strides = (1, 2, 2, 2)
        self.dw_stages = (3, 4)
        self.dw_multiplier = 2
    def forward(self, images):
        return torch.zeros((images.shape[0], 2), device=images.device) + self.marker * 0.0


def _write_corpus(root: Path) -> None:
    with sqlite3.connect(root / "dataset.sqlite") as c:
        c.execute("CREATE TABLE sample (sample_id TEXT PRIMARY KEY, split TEXT, image_rgb_u8 BLOB, is_red INTEGER, train_repeat INTEGER)")
        for split,count in (("train",6),("jp_val",4)):
            for i in range(count):
                c.execute("INSERT INTO sample VALUES (?,?,?,?,?)",(f"{split}-{i}",split,bytes([30+i,80+i,120+i])*(64*64),i%2,1))


def _params():
    return {"epochs":4,"batch_size":2,"eval_batch_size":2,"learning_rate":0.001,"weight_decay":0.0001,"rotation_augment_deg":22.5,"eval_angles":[0.0,15.0,30.0,45.0],"angle_eval_every":3,"warm_augment_prob":0.5,"warm_strength_min":0.1,"warm_strength_max":1.0,"warm_red_gain_max":1.5,"warm_green_gain_max":1.08,"warm_blue_gain_min":0.45,"warm_exposure_min":0.65,"warm_exposure_max":1.15,"amp":False,"tf32":False,"cache_device":"cpu","cache_vram_fraction":0.25}


def test_defaults_are_batch256() -> None:
    d=json.loads((DATA / "tile-classifier/train_protocols/red-five-train-c8-rgb-warmaug-partial-dw-v1.yaml").read_text())
    assert d["parameters"]["batch_size"]["default"] == 256
    assert d["parameters"]["eval_batch_size"]["default"] == 1024


def test_stage34_m2_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_corpus(tmp_path)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    train=_load_train_callable(DATA,PROTOCOL)
    module=sys.modules[train.__module__]
    built=[]
    def fake_build(*, fields, dw_stages, dw_multiplier):
        assert tuple(fields)==(6,12,24,48)
        assert tuple(dw_stages)==(3,4)
        assert dw_multiplier==2
        m=TinyModel(float(torch.rand(1))); built.append(m); return m
    def fake_train_epoch(model, split, *, epoch, **kwargs):
        model.marker.data.fill_(float(epoch)); return {"loss":1/epoch,"accuracy":0.5,"warm_augmented_fraction":0.5}
    scores={1:0.4,3:0.6,4:0.55}
    def fake_eval(model, split, *, angle, **kwargs):
        score=scores.get(int(round(float(model.marker))),0.99)
        return {"accuracy":score,"balanced_accuracy":score,"red_recall":score,"normal_specificity":score}
    monkeypatch.setattr(module,"build_c8_rgb_aggressive_partial_dw_classifier",fake_build)
    monkeypatch.setattr(module,"_train_epoch",fake_train_epoch)
    monkeypatch.setattr(module,"_evaluate",fake_eval)
    ctx=TrainContext(task=cast(Any,None),corpus=MaterializedCorpus(definition=cast(Any,None),root=tmp_path),architecture=cast(Any,{"id":ARCH}),model=TinyModel(),seed=42,parameters=cast(Any,_params()),telemetry=Telemetry(),work_dir=tmp_path/"work")
    returned=train(ctx)
    assert returned.marker.item()==pytest.approx(3.0)
    assert len(built)==1
