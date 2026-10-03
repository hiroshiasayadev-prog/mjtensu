from __future__ import annotations
import gzip, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable

LABELS = (
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
    "east","south","west","north","white","green","red","invalid",
)

class Model(torch.nn.Module):
    def forward(self, x):
        out=torch.zeros((x.shape[0],len(LABELS)),device=x.device)
        target=(x[:,0,0,0]*255).round().long().clamp(0,len(LABELS)-2)
        out.scatter_(1,target[:,None],8.0)
        return out

class Telemetry:
    def __init__(self): self.events=[]
    def report_scalar(self, **kwargs): self.events.append(kwargs)

def test_all_real_crop_protocol_streams_population_and_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    root=tmp_path/"corpus"; root.mkdir()
    images=np.stack([np.full((64,64),i,dtype=np.uint8) for i in list(range(34))+[0]])
    labels=np.asarray(list(range(34))+[0],dtype=np.uint8)
    n=len(labels)
    np.savez_compressed(
        root/"shard-0000.npz",
        images=images,
        labels=labels,
        widths=np.asarray([42 if i%2==0 else 56 for i in range(n)],dtype=np.uint16),
        heights=np.asarray([56 if i%2==0 else 42 for i in range(n)],dtype=np.uint16),
        source_codes=np.asarray([0]*34+[1],dtype=np.uint8),
        partition_codes=np.asarray([0]*34+[3],dtype=np.uint8),
        annotation_angles_deg=np.asarray([0.0]*34+[90.0],dtype=np.float32),
        expected_rotations_deg=np.asarray([0]*34+[90],dtype=np.int16),
        corrected_flags=np.zeros(n,dtype=np.uint8),
    )
    with gzip.open(root/"shard-0000.jsonl.gz","wt",encoding="utf-8") as f:
        for i,label in enumerate(LABELS[:-1]):
            f.write(json.dumps({
                "crop_id":f"jp:train:{i}","source":"jp","source_partition":"train",
                "base_label":label,"source_label":label,"corrected":False,
                "source_image_path":"fixture.jpg","source_image_id":"1",
                "source_annotation_id":str(i),"bbox":[0,0,42,56],
                "capture_id":None,"layout_id":None,"region":None,
                "brightness":None,"shadow":None,
            })+"\n")
        f.write(json.dumps({
            "crop_id":"manual:0","source":"manual","source_partition":"capture",
            "base_label":"1m","source_label":"1m","corrected":False,
            "source_image_path":"manual.jpg","source_image_id":"m1",
            "source_annotation_id":"m0","bbox":[0,0,56,42],
            "capture_id":"cap","layout_id":None,"region":None,
            "brightness":None,"shadow":None,
        })+"\n")
    (root/"index.json").write_text(json.dumps({
        "schema":"mjtensu.recognition/tile-all-real-crops/v1",
        "labels":list(LABELS),"valid_labels":list(LABELS[:-1]),
        "image_shape":[1,64,64],"preprocess":"fixture",
        "normalization":{"mean":0.0,"std":1.0,"source_corpus":"fixture"},
        "counts":{"included":n},
    }))
    telemetry=Telemetry()
    evaluate=_load_evaluation_callable("mldb_data","tile-classifier/tile-shape-all-real-crop-recall-v1")
    candidate=evaluate(SimpleNamespace(
        parameters={"batch_size":16,"slice_min_samples":1},
        corpus=SimpleNamespace(root=root),
        model=SimpleNamespace(module=Model()),
        telemetry=telemetry,
        work_dir=tmp_path/"work",
    ))
    assert candidate.metrics["accuracy"] == 1.0
    assert candidate.metrics["macro_recall"] == 1.0
    assert candidate.metrics["worst_class_recall"] == 1.0
    assert candidate.metrics["invalid_prediction_rate"] == 0.0
    assert candidate.metrics["jp_accuracy"] == 1.0
    assert set(candidate.artifacts) == {
        "class_recall_table","source_recall_table","class_orientation_recall_table",
        "confusion_matrix","class_orientation_recall_heatmap",
        "high_confidence_error_contact_sheet","error_inventory",
        "per_sample_predictions","report",
    }
    assert all(p.is_file() for p in candidate.artifacts.values())
    assert telemetry.events and telemetry.events[0]["step"] == 0

def test_protocol_declares_failure_inventory_artifacts():
    p=Path(__file__).parents[4]/"mldb_data/tile-classifier/evaluation_protocols/tile-shape-all-real-crop-recall-v1.yaml"
    d=json.loads(p.read_text())
    assert d["status"] in {"draft","sealed"}
    assert d["artifacts"]["per_sample_predictions"]["format"] == "jsonl.gz"
    assert d["metrics"]["worst_class_orientation_recall"]["preference"] == "higher"
