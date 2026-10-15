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
PROTOCOL = "tile-classifier/red-five-train-c8-rgb-warmaug-full-stride-v1"


class Telemetry:
    def __init__(self) -> None: self.events=[]
    def report_scalar(self, *, group, series, value, step): self.events.append((group,series,value,step))


class TinyModel(nn.Module):
    def __init__(self, value: float = 0.0, *, strides=None) -> None:
        super().__init__(); self.marker=nn.Parameter(torch.tensor(float(value))); self.field_counts=(6,12,24,48)
        if strides is not None: self.stage_strides=tuple(strides)
    def forward(self, images): return torch.zeros((images.shape[0],2),device=images.device)+self.marker*0.0


def write_corpus(root: Path) -> None:
    with sqlite3.connect(root / "dataset.sqlite") as c:
        c.execute("CREATE TABLE sample (sample_id TEXT PRIMARY KEY, split TEXT, image_rgb_u8 BLOB, is_red INTEGER, train_repeat INTEGER)")
        for split,count in (("train",6),("jp_val",4)):
            for i in range(count):
                c.execute("INSERT INTO sample VALUES (?,?,?,?,?)",(f"{split}-{i}",split,bytes([30+i,80+i,120+i])*(64*64),i%2,2 if split=="train" and i==0 else 1))


def params():
    return {"epochs":4,"batch_size":2,"eval_batch_size":2,"learning_rate":0.001,"weight_decay":0.0001,"rotation_augment_deg":22.5,"eval_angles":[0.0,15.0,30.0,45.0],"angle_eval_every":3,"warm_augment_prob":0.5,"warm_strength_min":0.1,"warm_strength_max":1.0,"warm_red_gain_max":1.5,"warm_green_gain_max":1.08,"warm_blue_gain_min":0.45,"warm_exposure_min":0.65,"warm_exposure_max":1.15,"amp":False,"tf32":False,"cache_device":"cpu","cache_vram_fraction":0.25}


def run_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, architecture: str, expected_kind: str, expected_strides):
    write_corpus(tmp_path); monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    train=_load_train_callable(DATA,PROTOCOL); module=sys.modules[train.__module__]
    built=[]
    def fake_full(*, fields):
        assert expected_kind=="baseline" and tuple(fields)==(6,12,24,48); m=TinyModel(float(torch.rand(1))); built.append(m); return m
    def fake_stride(*, fields, stage_strides):
        assert expected_kind=="stride" and tuple(fields)==(6,12,24,48) and tuple(stage_strides)==expected_strides; m=TinyModel(float(torch.rand(1)),strides=stage_strides); built.append(m); return m
    def fake_train_epoch(model, split, *, epoch, **kwargs): model.marker.data.fill_(float(epoch)); return {"loss":1/epoch,"accuracy":0.5,"warm_augmented_fraction":0.5}
    scores={1:0.4,3:0.6,4:0.55}
    def fake_eval(model, split, *, angle, **kwargs):
        score=scores.get(int(round(float(model.marker))),0.99); return {"accuracy":score,"balanced_accuracy":score,"red_recall":score,"normal_specificity":score}
    monkeypatch.setattr(module,"build_c8_rgb_full_width_classifier",fake_full); monkeypatch.setattr(module,"build_c8_rgb_full_stride_classifier",fake_stride); monkeypatch.setattr(module,"_train_epoch",fake_train_epoch); monkeypatch.setattr(module,"_evaluate",fake_eval)
    ctx=TrainContext(task=cast(Any,None),corpus=MaterializedCorpus(definition=cast(Any,None),root=tmp_path),architecture=cast(Any,{"id":architecture}),model=TinyModel(strides=expected_strides if expected_kind=="stride" else None),seed=42,parameters=cast(Any,params()),telemetry=Telemetry(),work_dir=tmp_path/"work")
    returned=train(ctx); assert returned.marker.item()==pytest.approx(3.0); assert len(built)==1


def test_defaults_match_red_five_recipe() -> None:
    d=json.loads((DATA / "tile-classifier/train_protocols/red-five-train-c8-rgb-warmaug-full-stride-v1.yaml").read_text())
    assert d["parameters"]["epochs"]["default"]==50
    assert d["parameters"]["batch_size"]["default"]==128
    assert d["parameters"]["eval_batch_size"]["default"]==1024


def test_baseline_dispatch(tmp_path, monkeypatch):
    run_case(tmp_path,monkeypatch,"tile-classifier/red-five-c8-rgb-full-fields6-12-24-48-v1","baseline",None)


def test_safe_stride_dispatch(tmp_path, monkeypatch):
    run_case(tmp_path,monkeypatch,"tile-classifier/red-five-c8-rgb-full-stride-safe-fields6-12-24-48-v1","stride",(1,2,2,1))
