from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable


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
        self.classifier = torch.nn.Linear(12, 35)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(inputs)).flatten(1)
        return self.classifier(features)


class _Telemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, float, int]] = []

    def report_scalar(self, *, group: str, series: str, value: int | float, step: int) -> None:
        self.events.append((group, series, float(value), step))


def _create_database(path: Path) -> None:
    labels = ["5m", "6m", "7m"] + [f"other-{index}" for index in range(32)]
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE experiment_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            """CREATE TABLE sample (
            sample_id TEXT PRIMARY KEY, split TEXT, base_label TEXT, class_index INTEGER,
            image_gray_u8 BLOB, original_width INTEGER, original_height INTEGER,
            source TEXT, capture_id TEXT, layout_id TEXT, region TEXT, source_image_path TEXT)"""
        )
        connection.execute("INSERT INTO experiment_metadata VALUES (?, ?)", ("base_labels", json.dumps(labels)))
        for index in range(6):
            image = np.full((64, 64), 96 + index * 8, dtype=np.uint8)
            connection.execute(
                "INSERT INTO sample VALUES (?, 'train', '5m', 0, ?, 64, 64, 'fixture', NULL, NULL, NULL, 'train.png')",
                (f"train-{index}", image.tobytes()),
            )
        for class_index, label in enumerate(("5m", "6m", "7m")):
            for variant in range(3):
                image = np.full((64, 64), 235, dtype=np.uint8)
                x0 = 12 + class_index * 12
                image[12:52, x0 : x0 + 5] = 30 + variant * 5
                image[36:42, 10 + variant : 54 - variant] = 70 + class_index * 20
                connection.execute(
                    "INSERT INTO sample VALUES (?, 'manual_val', ?, ?, ?, 40, 58, 'fixture', ?, 'layout', 'hand', 'manual.png')",
                    (f"{label}-{variant}", label, class_index, image.tobytes(), f"capture-{label}-{variant}"),
                )


def test_manzu_diagnostic_produces_rich_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    corpus_root = tmp_path / "corpus"
    work_dir = tmp_path / "work"
    corpus_root.mkdir()
    _create_database(corpus_root / "dataset.sqlite")
    telemetry = _Telemetry()
    evaluate = _load_evaluation_callable("mldb_data", "tile-classifier/tile-shape-manzu-diagnostic-v3")
    context = SimpleNamespace(
        parameters={"split": "manual_val", "batch_size": 64, "occlusion_patch": 8},
        corpus=SimpleNamespace(root=corpus_root),
        model=SimpleNamespace(module=_TinyClassifier()),
        telemetry=telemetry,
        work_dir=work_dir,
    )

    candidate = evaluate(context)

    assert candidate.metrics["front_accuracy"] >= 0.0
    assert candidate.metrics["occlusion_effective_patch_count_mean"] >= 0.0
    assert all(math.isfinite(float(value)) for value in candidate.metrics.values())
    assert set(candidate.artifacts) == {
        "summary_table", "perturbation_table", "robustness_plot", "confusion_plot",
        "layer_separation_table", "layer_separation_plot", "contact_sheet",
        "per_sample_details", "report_json", "report_html",
    }
    assert all(path.is_file() and path.is_relative_to(work_dir) for path in candidate.artifacts.values())
    report = json.loads(candidate.artifacts["report_json"].read_text(encoding="utf-8"))
    assert report["sample_count"] == 9
    assert len(report["representatives"]) == 9
    assert report["representation"]
    assert any(group == "manzu/occlusion" for group, _series, _value, _step in telemetry.events)
    assert any(group == "manzu/representation" for group, _series, _value, _step in telemetry.events)
