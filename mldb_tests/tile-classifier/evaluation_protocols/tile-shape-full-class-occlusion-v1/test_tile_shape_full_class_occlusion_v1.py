from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable


LABELS = (
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "east", "south", "west", "north", "white", "green", "red", "invalid",
)


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 12, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(2),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Linear(12, len(LABELS))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(inputs)).flatten(1))


def _create_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE experiment_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            """CREATE TABLE sample (
            sample_id TEXT PRIMARY KEY, split TEXT, base_label TEXT, class_index INTEGER,
            image_gray_u8 BLOB, original_width INTEGER, original_height INTEGER,
            source TEXT, capture_id TEXT, layout_id TEXT, region TEXT, source_image_path TEXT)"""
        )
        connection.execute(
            "INSERT INTO experiment_metadata VALUES (?, ?)",
            ("base_labels", json.dumps(list(LABELS))),
        )
        for index, value in enumerate((70, 110, 150, 190)):
            image = np.full((64, 64), value, dtype=np.uint8)
            connection.execute(
                "INSERT INTO sample VALUES (?, 'train', '1m', 0, ?, 64, 64, 'fixture', NULL, NULL, NULL, 'train.png')",
                (f"train-{index}", image.tobytes()),
            )
        for class_index, label in enumerate(LABELS):
            image = np.full((64, 64), 210, dtype=np.uint8)
            x0 = 4 + (class_index * 5) % 48
            y0 = 4 + (class_index * 7) % 48
            image[y0:y0 + 10, x0:x0 + 10] = 25 + (class_index % 6) * 20
            image[20:24, 8:56] = 50 + class_index
            connection.execute(
                "INSERT INTO sample VALUES (?, 'manual_val', ?, ?, ?, 42, 58, 'fixture', ?, 'layout', 'hand', 'manual.png')",
                (f"{label}-0", label, class_index, image.tobytes(), f"capture-{label}"),
            )


def test_full_class_occlusion_covers_every_class_and_patch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)
    corpus_root = tmp_path / "corpus"
    work_dir = tmp_path / "work"
    corpus_root.mkdir()
    _create_database(corpus_root / "dataset.sqlite")
    evaluate = _load_evaluation_callable(
        "mldb_data",
        "tile-classifier/tile-shape-full-class-occlusion-v1",
    )
    candidate = evaluate(SimpleNamespace(
        parameters={"split": "manual_val", "batch_size": 256, "occlusion_patch": 8},
        corpus=SimpleNamespace(root=corpus_root),
        model=SimpleNamespace(module=_TinyClassifier()),
        work_dir=work_dir,
    ))

    assert set(candidate.metrics) == {
        "baseline_accuracy",
        "baseline_true_margin_mean",
        "occlusion_peak_drop_mean",
        "occlusion_top_patch_share_mean",
        "occlusion_effective_patch_count_mean",
        "occlusion_positive_drop_mean",
    }
    assert all(math.isfinite(float(value)) for value in candidate.metrics.values())
    assert set(candidate.artifacts) == {
        "class_summary_table",
        "class_patch_table",
        "class_occlusion_contact_sheet",
        "per_sample_details",
        "report_json",
    }

    with candidate.artifacts["class_summary_table"].open(encoding="utf-8", newline="") as handle:
        class_rows = list(csv.DictReader(handle))
    assert [row["label"] for row in class_rows] == list(LABELS)
    assert all(int(row["sample_count"]) == 1 for row in class_rows)

    with candidate.artifacts["class_patch_table"].open(encoding="utf-8", newline="") as handle:
        patch_rows = list(csv.DictReader(handle))
    assert len(patch_rows) == len(LABELS) * 64
    assert {(int(row["grid_y"]), int(row["grid_x"])) for row in patch_rows if row["label"] == "1m"} == {
        (y, x) for y in range(8) for x in range(8)
    }

    with Image.open(candidate.artifacts["class_occlusion_contact_sheet"]) as image:
        assert image.size == (2100, 950)

    report = json.loads(candidate.artifacts["report_json"].read_text(encoding="utf-8"))
    assert report["sample_count"] == len(LABELS)
    assert report["occlusion_patch"] == 8
    assert len(report["class_summary"]) == len(LABELS)
    assert "true-class logit minus" in report["notes"]["margin"]
