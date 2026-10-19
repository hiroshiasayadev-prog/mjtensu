from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable

LABELS=tuple([f"{n}{s}" for s in "mps" for n in range(1,10)]+["east","south","west","north","white","green","red","invalid"])
VALID=LABELS[:-1]

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.pool=torch.nn.AdaptiveAvgPool2d(1); self.head=torch.nn.Linear(1,len(LABELS))
    def forward(self,x): return self.head(self.pool(x).flatten(1))

def make_db(path: Path):
    with sqlite3.connect(path) as c:
        c.execute("create table experiment_metadata(key text primary key,value text not null)")
        c.execute("create table sample(sample_id text primary key, split text, source text, source_partition text, base_label text, class_index integer, image_gray_u8 blob, source_image_path text)")
        c.execute("insert into experiment_metadata values('base_labels',?)",(json.dumps(LABELS),))
        # training rows for normalization
        for i in range(4):
            arr=np.full((64,64),30+i,dtype=np.uint8)
            c.execute("insert into sample values(?,?,?,?,?,?,?,?)",(f'train-{i}','train','test','train','1m',0,arr.tobytes(),f'train-{i}.png'))
        for ci,label in enumerate(VALID):
            for j in range(200):
                arr=np.full((64,64),(ci*7+j)%256,dtype=np.uint8)
                c.execute("insert into sample values(?,?,?,?,?,?,?,?)",(f'{label}-{j}','jp_val','jp','valid',label,ci,arr.tobytes(),f'{label}-{j}.png'))

def test_protocol_balanced_population(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    root=tmp_path/'corpus'; root.mkdir(); make_db(root/'dataset.sqlite')
    evaluate=_load_evaluation_callable('mldb_data','tile-classifier/tile-shape-jp-val-balanced-class-recall-v1')
    context=SimpleNamespace(parameters={'batch_size':256,'recall_gate':0.8},task={'target':{'labels':list(LABELS)}},corpus=SimpleNamespace(root=root),model=SimpleNamespace(module=Tiny()),work_dir=tmp_path/'work')
    candidate=evaluate(context)
    assert set(candidate.metrics)=={'accuracy','macro_recall','worst_class_recall','class_recall_below_gate_count','invalid_prediction_rate'}
    report=json.loads(candidate.artifacts['report'].read_text())
    assert report['population']=={'split':'jp_val','classes':34,'per_class':200,'included':6800}
    assert candidate.artifacts['class_recall_table'].exists()
    assert candidate.artifacts['confusion_matrix'].exists()
    assert candidate.artifacts['per_sample_predictions'].exists()
