from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable

LABELS = tuple(
    [f"{n}{s}" for s in "mps" for n in range(1, 10)]
    + ["east", "south", "west", "north", "white", "green", "red", "invalid"]
)
VALID = LABELS[:-1]


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(1, len(LABELS))

    def forward(self, x):
        return self.head(self.pool(x).flatten(1))


def make_db(path: Path) -> None:
    with sqlite3.connect(path) as c:
        c.execute(
            "create table experiment_metadata(key text primary key,value text not null)"
        )
        c.execute(
            "create table sample("
            "sample_id text primary key, split text, source text, "
            "source_partition text, base_label text, class_index integer, "
            "image_gray_u8 blob, source_image_path text)"
        )
        c.execute(
            "insert into experiment_metadata values('base_labels',?)",
            (json.dumps(LABELS),),
        )
        for i in range(4):
            arr = np.full((64, 64), 30 + i, dtype=np.uint8)
            c.execute(
                "insert into sample values(?,?,?,?,?,?,?,?)",
                (
                    f"train-{i}",
                    "train",
                    "test",
                    "train",
                    "1m",
                    0,
                    arr.tobytes(),
                    f"train-{i}.png",
                ),
            )
        for ci, label in enumerate(VALID):
            for j in range(100):
                arr = np.full((64, 64), (ci * 7 + j) % 256, dtype=np.uint8)
                c.execute(
                    "insert into sample values(?,?,?,?,?,?,?,?)",
                    (
                        f"{label}-{j}",
                        "manual_val",
                        "mixed",
                        "holdout",
                        label,
                        ci,
                        arr.tobytes(),
                        f"{label}-{j}.png",
                    ),
                )
        for j in range(32):
            arr = np.full((64, 64), (200 + j) % 256, dtype=np.uint8)
            c.execute(
                "insert into sample values(?,?,?,?,?,?,?,?)",
                (
                    f"invalid-{j}",
                    "manual_val",
                    "manual",
                    "capture",
                    "invalid",
                    len(VALID),
                    arr.tobytes(),
                    f"invalid-{j}.png",
                ),
            )


def test_protocol_balanced_valid_and_separate_invalid(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    root = tmp_path / "corpus"
    root.mkdir()
    make_db(root / "dataset.sqlite")
    evaluate = _load_evaluation_callable(
        "mldb_data",
        "tile-classifier/tile-shape-manual-jp-balanced-class-diagnostic-v1",
    )
    context = SimpleNamespace(
        parameters={"batch_size": 256, "recall_gate": 0.8},
        task={"target": {"labels": list(LABELS)}},
        corpus=SimpleNamespace(root=root),
        model=SimpleNamespace(module=Tiny()),
        work_dir=tmp_path / "work",
    )
    candidate = evaluate(context)
    assert set(candidate.metrics) == {
        "valid_accuracy",
        "valid_macro_recall",
        "valid_worst_class_recall",
        "valid_bottom5_class_recall_mean",
        "valid_class_recall_below_gate_count",
        "valid_to_invalid_rate",
        "valid_macro_precision",
        "valid_macro_f1",
        "invalid_recall",
        "invalid_precision",
    }
    report = json.loads(candidate.artifacts["report"].read_text())
    assert report["population"] == {
        "split": "manual_val",
        "valid_classes": 34,
        "valid_per_class": 100,
        "invalid_support": 32,
        "included": 3432,
    }
    assert len(report["bottom5_classes"]) == 5
    assert candidate.artifacts["class_metrics_table"].exists()
    assert candidate.artifacts["confusion_matrix"].exists()
    assert candidate.artifacts["per_sample_predictions"].exists()
