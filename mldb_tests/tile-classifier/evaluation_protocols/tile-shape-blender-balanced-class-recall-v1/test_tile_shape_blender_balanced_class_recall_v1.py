from __future__ import annotations

import gzip
import json
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
        out = torch.zeros((x.shape[0], len(LABELS)), device=x.device)
        target = (x[:, 0, 0, 0] * 255).round().long().clamp(0, len(LABELS) - 2)
        out.scatter_(1, target[:, None], 8.0)
        return out


class Telemetry:
    def __init__(self):
        self.events = []

    def report_scalar(self, **kwargs):
        self.events.append(kwargs)


def test_balanced_protocol_requires_and_scores_exact_100_per_class(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    root = tmp_path / "corpus"
    root.mkdir()
    labels = np.repeat(np.arange(34, dtype=np.int64), 100)
    images = np.stack([np.full((64, 64), int(label), dtype=np.uint8) for label in labels])
    np.savez_compressed(
        root / "shard-0000.npz",
        images=images,
        labels=labels,
        widths=np.full(len(labels), 40, dtype=np.int64),
        heights=np.full(len(labels), 56, dtype=np.int64),
    )
    with gzip.open(root / "shard-0000.jsonl.gz", "wt", encoding="utf-8") as handle:
        for i, target in enumerate(labels.tolist()):
            handle.write(json.dumps({
                "crop_id": f"blender:{i}",
                "source_annotation_id": str(i),
                "source_image_path": "synthetic.png",
                "region": "completed_hand",
                "tile_identity": LABELS[target],
            }) + "\n")
    (root / "index.json").write_text(json.dumps({
        "schema": "mjtensu.recognition/blender-balanced-classifier-corpus-index/v1",
        "labels": list(LABELS),
        "true_labels": list(LABELS[:-1]),
        "counts": {"included": 3400, "per_class": 100, "classes": 34},
        "normalization": {"mean": 0.0, "std": 1.0, "source_corpus": "fixture", "source_split": "train"},
        "preprocess": "fixture",
        "selection": {"source": "fixture"},
    }))
    telemetry = Telemetry()
    evaluate = _load_evaluation_callable(
        "mldb_data", "tile-classifier/tile-shape-blender-balanced-class-recall-v1"
    )
    candidate = evaluate(SimpleNamespace(
        parameters={"batch_size": 512, "recall_gate": 0.8},
        corpus=SimpleNamespace(root=root),
        model=SimpleNamespace(module=Model()),
        telemetry=telemetry,
        work_dir=tmp_path / "work",
    ))
    assert candidate.metrics == {
        "accuracy": 1.0,
        "macro_recall": 1.0,
        "worst_class_recall": 1.0,
        "class_recall_below_gate_count": 0,
        "invalid_prediction_rate": 0.0,
    }
    assert set(candidate.artifacts) == {
        "class_recall_table", "confusion_matrix", "high_confidence_error_contact_sheet",
        "error_inventory", "per_sample_predictions", "report",
    }
    assert all(path.is_file() for path in candidate.artifacts.values())
    assert telemetry.events and telemetry.events[0]["step"] == 0


def test_protocol_declares_80pct_gate_metric():
    path = Path(__file__).parents[4] / "mldb_data/tile-classifier/evaluation_protocols/tile-shape-blender-balanced-class-recall-v1.yaml"
    doc = json.loads(path.read_text())
    assert doc["parameters"]["recall_gate"]["default"] == 0.8
    assert doc["metrics"]["worst_class_recall"]["preference"] == "higher"
    assert doc["metrics"]["class_recall_below_gate_count"]["preference"] == "lower"
