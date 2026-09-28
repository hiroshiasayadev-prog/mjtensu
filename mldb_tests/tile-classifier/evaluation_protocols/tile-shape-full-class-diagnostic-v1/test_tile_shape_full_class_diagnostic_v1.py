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
CONDITION_COUNT = 15


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
        features = self.pool(self.features(inputs)).flatten(1)
        return self.classifier(features)


class _Telemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, float, int]] = []

    def report_scalar(
        self,
        *,
        group: str,
        series: str,
        value: int | float,
        step: int,
    ) -> None:
        self.events.append((group, series, float(value), step))


def _create_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE experiment_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
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
        for index in range(8):
            image = np.full((64, 64), 100 + index * 5, dtype=np.uint8)
            connection.execute(
                "INSERT INTO sample VALUES (?, 'train', '1m', 0, ?, 42, 58, 'fixture', NULL, NULL, NULL, 'train.png')",
                (f"train-{index}", image.tobytes()),
            )
        for class_index, label in enumerate(LABELS):
            for variant in range(2):
                image = np.full(
                    (64, 64),
                    205 - (class_index % 7) * 7,
                    dtype=np.uint8,
                )
                x0 = 4 + (class_index * 3 + variant * 5) % 48
                y0 = 6 + (class_index * 5 + variant * 7) % 46
                image[y0 : y0 + 8, x0 : x0 + 8] = 25 + (class_index % 5) * 20
                image[16 + variant : 20 + variant, 8:56] = 70 + class_index
                connection.execute(
                    "INSERT INTO sample VALUES (?, 'manual_val', ?, ?, ?, 42, 58, 'fixture', ?, 'layout', 'hand', 'manual.png')",
                    (
                        f"{label}-{variant}",
                        label,
                        class_index,
                        image.tobytes(),
                        f"capture-{label}-{variant}",
                    ),
                )


def _run_protocol(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)
    corpus_root = tmp_path / "corpus"
    work_dir = tmp_path / "work"
    corpus_root.mkdir()
    _create_database(corpus_root / "dataset.sqlite")
    telemetry = _Telemetry()
    evaluate = _load_evaluation_callable(
        "mldb_data",
        "tile-classifier/tile-shape-full-class-diagnostic-v1",
    )
    context = SimpleNamespace(
        parameters={"batch_size": 32},
        corpus=SimpleNamespace(root=corpus_root),
        model=SimpleNamespace(module=_TinyClassifier()),
        telemetry=telemetry,
        work_dir=work_dir,
    )
    candidate = evaluate(context)
    return candidate, telemetry, work_dir


def test_full_class_diagnostic_covers_all_classes_and_separates_plot_scales(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate, telemetry, work_dir = _run_protocol(tmp_path, monkeypatch)

    assert set(candidate.metrics) == {
        "front_accuracy",
        "front_macro_recall",
        "front_worst_class_recall",
        "mean_condition_accuracy",
        "worst_condition_accuracy",
        "mean_condition_macro_recall",
        "worst_condition_macro_recall",
        "worst_class_condition_recall",
        "front_true_margin_mean",
        "mean_condition_true_margin",
        "true_6m_to_5m_or_7m_confusion_max",
    }
    assert all(math.isfinite(float(value)) for value in candidate.metrics.values())
    for key in (
        "front_accuracy",
        "front_macro_recall",
        "front_worst_class_recall",
        "mean_condition_accuracy",
        "worst_condition_accuracy",
        "mean_condition_macro_recall",
        "worst_condition_macro_recall",
        "worst_class_condition_recall",
        "true_6m_to_5m_or_7m_confusion_max",
    ):
        assert 0.0 <= float(candidate.metrics[key]) <= 1.0

    assert set(candidate.artifacts) == {
        "condition_summary_table",
        "per_class_condition_table",
        "front_confusion_matrix",
        "robustness_accuracy_recall_plot",
        "robustness_true_margin_plot",
        "class_condition_recall_heatmap",
        "class_error_contact_sheet",
        "manzu_focus_table",
        "per_sample_details",
        "diagnostic_report",
    }
    assert all(
        path.is_file() and path.is_relative_to(work_dir)
        for path in candidate.artifacts.values()
    )

    with candidate.artifacts["condition_summary_table"].open(
        encoding="utf-8",
        newline="",
    ) as handle:
        condition_rows = list(csv.DictReader(handle))
    assert len(condition_rows) == CONDITION_COUNT
    assert condition_rows[0]["condition"] == "front-facing"

    with candidate.artifacts["per_class_condition_table"].open(
        encoding="utf-8",
        newline="",
    ) as handle:
        class_rows = list(csv.DictReader(handle))
    assert len(class_rows) == CONDITION_COUNT * len(LABELS)
    for condition_index in range(CONDITION_COUNT):
        condition_slice = [
            row
            for row in class_rows
            if int(row["condition_index"]) == condition_index
        ]
        assert [row["label"] for row in condition_slice] == list(LABELS)

    confusion = json.loads(
        candidate.artifacts["front_confusion_matrix"].read_text(encoding="utf-8")
    )
    confusion_trace = confusion["data"][0]
    assert confusion_trace["x"] == list(LABELS)
    assert confusion_trace["y"] == list(LABELS)
    assert len(confusion_trace["z"]) == len(LABELS)
    assert all(len(row) == len(LABELS) for row in confusion_trace["z"])
    assert sum(sum(row) for row in confusion_trace["z"]) == len(LABELS) * 2

    rates = json.loads(
        candidate.artifacts["robustness_accuracy_recall_plot"].read_text(
            encoding="utf-8"
        )
    )
    rate_names = [trace["name"] for trace in rates["data"]]
    assert rate_names == [
        "overall accuracy",
        "macro recall (35 classes)",
        "worst-class recall",
    ]
    assert "yaxis2" not in rates["layout"]
    assert all(
        0.0 <= float(value) <= 1.0
        for trace in rates["data"]
        for value in trace["y"]
    )

    margin = json.loads(
        candidate.artifacts["robustness_true_margin_plot"].read_text(
            encoding="utf-8"
        )
    )
    assert [trace["name"] for trace in margin["data"]] == [
        "mean true-class logit margin"
    ]
    assert all(
        token not in margin["data"][0]["name"].lower()
        for token in ("accuracy", "recall", "error")
    )
    assert "model-dependent scale" in margin["layout"]["title"]

    heatmap = json.loads(
        candidate.artifacts["class_condition_recall_heatmap"].read_text(
            encoding="utf-8"
        )
    )
    heatmap_trace = heatmap["data"][0]
    assert heatmap_trace["y"] == list(LABELS)
    assert len(heatmap_trace["x"]) == CONDITION_COUNT
    assert len(heatmap_trace["z"]) == len(LABELS)
    assert all(len(row) == CONDITION_COUNT for row in heatmap_trace["z"])

    with Image.open(candidate.artifacts["class_error_contact_sheet"]) as image:
        assert image.size == (1550, 1260)

    with candidate.artifacts["manzu_focus_table"].open(
        encoding="utf-8",
        newline="",
    ) as handle:
        manzu_rows = list(csv.DictReader(handle))
    assert len(manzu_rows) == CONDITION_COUNT
    assert "true_6m_to_5m_or_7m_confusion_rate" in manzu_rows[0]

    report = json.loads(
        candidate.artifacts["diagnostic_report"].read_text(encoding="utf-8")
    )
    assert report["split"] == "manual_val"
    assert report["labels"] == list(LABELS)
    assert report["sample_count"] == len(LABELS) * 2
    assert report["condition_order"][0] == "front-facing"
    assert len(report["condition_order"]) == CONDITION_COUNT
    assert [
        item["label"]
        for item in report["class_error_inspection_selection"]
    ] == list(LABELS)
    assert "true 6m samples predicted as 5m or 7m" in report["manzu_focus"]["semantics"]
    assert "within-model" in report["notes"]["true_margin"]

    robustness_events = [
        event
        for event in telemetry.events
        if event[0] == "classifier/full_class_robustness"
    ]
    manzu_events = [
        event
        for event in telemetry.events
        if event[0] == "classifier/manzu_focus"
    ]
    assert len(robustness_events) == CONDITION_COUNT * 3
    assert len(manzu_events) == CONDITION_COUNT
    assert sorted({step for _group, _series, _value, step in manzu_events}) == list(
        range(CONDITION_COUNT)
    )


def test_protocol_declares_selectable_visuals_and_neutral_margin_metrics() -> None:
    protocol_path = (
        Path(__file__).parents[4]
        / "mldb_data"
        / "tile-classifier"
        / "evaluation_protocols"
        / "tile-shape-full-class-diagnostic-v1.yaml"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    assert protocol["id"] == "tile-classifier/tile-shape-full-class-diagnostic-v1"
    assert protocol["status"] in {"draft", "sealed"}
    assert protocol["metrics"]["front_true_margin_mean"]["preference"] == "neutral"
    assert (
        protocol["metrics"]["mean_condition_true_margin"]["preference"]
        == "neutral"
    )
    for artifact in (
        "front_confusion_matrix",
        "robustness_accuracy_recall_plot",
        "robustness_true_margin_plot",
        "class_condition_recall_heatmap",
        "class_error_contact_sheet",
    ):
        assert protocol["artifacts"][artifact]["study_view"] == "select"
