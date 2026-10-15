from __future__ import annotations

import io, json, sqlite3
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable


class ColorModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.feature_kind="exr"; self.register_buffer("input_mean",torch.zeros(4)); self.register_buffer("input_std",torch.ones(4))
    def forward(self, inputs):
        score=inputs[:,0].mean((1,2))-inputs[:,1].mean((1,2)); return torch.stack((-score,score),dim=1)


def png(rgb):
    image=Image.new("RGB",(32,48),rgb); b=io.BytesIO(); image.save(b,format="PNG"); return b.getvalue()


def test_v3_preserves_full_audited_views(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False); corpus=tmp_path/"corpus"; corpus.mkdir()
    with sqlite3.connect(corpus/"dataset.sqlite") as c:
        c.execute("CREATE TABLE sample (sample_id TEXT PRIMARY KEY,crop_id TEXT,source TEXT,source_partition TEXT,suit TEXT,is_red INTEGER,source_label TEXT,image_png BLOB,source_image_path TEXT,source_annotation_id TEXT,capture_id TEXT,brightness TEXT,shadow TEXT)")
        rows=[("normal","normal","jp","test","m",0,"5m",png((120,120,120))),("red","red","jp","test","m",1,"red5m",png((230,40,40))),("correct","correct","jp","train","m",0,"5m",png((230,40,40))),("exclude","exclude","jp","train","s",1,"red5s",png((230,40,40))),("manual-normal","manual-normal","manual","capture","p",0,"5p",png((120,120,120))),("manual-red","manual-red","manual","capture","p",1,"red5p",png((230,40,40))),("false-red","false-red","jp","valid","s",0,"5s",png((230,40,40)))]
        c.executemany("INSERT INTO sample VALUES (?,?,?,?,?,?,?,?,?,'ann',NULL,'normal','none')",[(a,b,c1,d,e,f,g,h,'fixture.png') for a,b,c1,d,e,f,g,h in rows])
    with sqlite3.connect(corpus/"warm.sqlite") as c:
        c.execute("CREATE TABLE sample (crop_id TEXT PRIMARY KEY,tile_label TEXT,image_png BLOB,capture_id TEXT,brightness TEXT,shadow TEXT,source_image_path TEXT)")
        c.executemany("INSERT INTO sample VALUES (?,?,?,'warm','warm','none','warm.png')",[("warm-normal","5m",png((120,120,120))),("warm-red","red5m",png((230,40,40)))])
    (corpus/"audit.json").write_text(json.dumps({"corrected":{"correct":{"target":1,"suit":"p"},"exclude":{"target":0,"suit":"m"}},"specialist_excluded":{"exclude":{"reason":"physically non-five"}}}))
    evaluate=_load_evaluation_callable("mldb_data","tile-classifier/red-five-binary-diagnostic-v3")
    candidate=evaluate(SimpleNamespace(parameters={"batch_size":3},corpus=SimpleNamespace(root=corpus),model=SimpleNamespace(module=ColorModel()),work_dir=tmp_path/"work"))
    assert candidate.metrics["specialist_five_error_count"]==1
    assert candidate.metrics["manual_accuracy"]==1.0 and candidate.metrics["warm_accuracy"]==1.0
    report=json.loads(candidate.artifacts["diagnostic_report"].read_text()); assert report["input_preprocessing"]["feature_kind"]=="exr"
