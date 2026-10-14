from __future__ import annotations
import hashlib, io, json, sqlite3
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable

class ColorRule(torch.nn.Module):
    def forward(self, x):
        score = x[:,0].mean((1,2)) - x[:,1].mean((1,2))
        return torch.stack((-score, score), 1)

class Telemetry:
    def __init__(self): self.events=[]
    def report_scalar(self, *, group, series, value, step):
        self.events.append((group,series,float(value),step))

def rgb(target):
    v=np.array([220,60,120] if target else [60,220,120],dtype=np.uint8)
    return np.broadcast_to(v,(64,64,3)).copy()

def png(target):
    out=io.BytesIO(); Image.fromarray(rgb(target),mode="RGB").save(out,format="PNG"); return out.getvalue()

def make_corpus(root):
    with sqlite3.connect(root/"dataset.sqlite") as c:
        c.execute("""CREATE TABLE sample (sample_id TEXT PRIMARY KEY, split TEXT, source TEXT, source_partition TEXT, suit TEXT, is_red INTEGER, source_label TEXT, crop_id TEXT, image_size INTEGER, image_rgb_u8 BLOB, train_repeat INTEGER, original_width INTEGER, original_height INTEGER, source_image_path TEXT, source_image_id TEXT, source_annotation_id TEXT, capture_id TEXT, layout_id TEXT, region TEXT, brightness TEXT, shadow TEXT, annotation_angle_deg REAL, expected_rotation_deg INTEGER)""")
        serial=0
        for split in ("train","jp_val","jp_test","manual_val"):
            for target in (0,1):
                for suit in "mps":
                    serial+=1; label=f"red5{suit}" if target else f"5{suit}"
                    source="manual" if split=="manual_val" else "jp"
                    part="capture" if split=="manual_val" else split.replace("jp_","")
                    c.execute("INSERT INTO sample VALUES ("+",".join("?"*23)+")",(f"{split}-{serial}",split,source,part,suit,target,label,f"crop-{serial}",64,rgb(target).tobytes(),1,64,64,f"{split}.png",None,str(serial),f"cap-{serial}" if split=="manual_val" else None,None,"hand","normal","none",0.0,0))
    with sqlite3.connect(root/"warm_holdout.sqlite") as c:
        c.execute("CREATE TABLE sample (crop_id TEXT PRIMARY KEY, tile_label TEXT, image_png BLOB, capture_id TEXT, brightness TEXT, shadow TEXT, source_image_path TEXT)")
        for target in (0,1):
            for suit in "mps":
                label=f"red5{suit}" if target else f"5{suit}"
                c.execute("INSERT INTO sample VALUES (?,?,?,?,?,?,?)",(f"warm-{label}",label,png(target),f"warm-{suit}","warm","none",f"{label}.png"))

def test_red_five_binary_diagnostic(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    corpus=tmp_path/"corpus"; corpus.mkdir(); make_corpus(corpus)
    telemetry=Telemetry()
    evaluate=_load_evaluation_callable("mldb_data","tile-classifier/red-five-binary-diagnostic-v1")
    candidate=evaluate(SimpleNamespace(parameters={"batch_size":8,"eval_angles":[0.0,15.0]},corpus=SimpleNamespace(root=corpus),model=SimpleNamespace(module=ColorRule()),telemetry=telemetry,work_dir=tmp_path/"work"))
    for name,value in candidate.metrics.items():
        assert value==0 if name.endswith("_count_0deg") else float(value)==1.0
    assert set(candidate.artifacts)=={"population_summary_table","angle_summary_table","confusion_matrix","error_details","diagnostic_report"}
    report=json.loads(candidate.artifacts["diagnostic_report"].read_text())
    assert report["standard_sample_count"]==18 and report["warm_holdout_sample_count"]==6
    assert len(telemetry.events)==4

def test_protocol_hash_and_contract():
    root=Path(__file__).parents[4]
    y=root/"mldb_data/tile-classifier/evaluation_protocols/red-five-binary-diagnostic-v1.yaml"
    p=y.with_suffix(".py"); protocol=json.loads(y.read_text())
    assert protocol["task"]=="tile-classifier/red-five-binary-classification-v1"
    assert protocol["implementation"]["sha256"]==hashlib.sha256(p.read_bytes()).hexdigest()
    for key in ("standard_fp_count_0deg","standard_fn_count_0deg","warm_error_count_0deg"):
        assert protocol["metrics"][key]["preference"]=="lower"
