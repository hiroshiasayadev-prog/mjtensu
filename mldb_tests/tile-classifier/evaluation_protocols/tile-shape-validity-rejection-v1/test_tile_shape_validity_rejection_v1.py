from __future__ import annotations

import csv
import importlib.util
import json
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable


REPO_ROOT = Path(__file__).parents[4]
PROTOCOL_PY = (
    REPO_ROOT
    / "mldb_data"
    / "tile-classifier"
    / "evaluation_protocols"
    / "tile-shape-validity-rejection-v1.py"
)
PROTOCOL_YAML = PROTOCOL_PY.with_suffix(".yaml")


def _load_protocol_module():
    spec = importlib.util.spec_from_file_location(
        "tile_shape_validity_rejection_v1",
        PROTOCOL_PY,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_protocol_module()
LABELS = MODULE.EXPECTED_LABELS


def _row(sample_id: str, label: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "base_label": label,
        "class_index": LABELS.index(label),
    }


def _logit_row(best_label: str, invalid_logit: float = -5.0) -> torch.Tensor:
    logits = torch.full((len(LABELS),), -10.0)
    logits[LABELS.index(best_label)] = 5.0
    logits[LABELS.index("invalid")] = invalid_logit
    return logits


def test_binary_collapse_counts_tile_identity_errors_as_accepted() -> None:
    rows = [
        _row("valid-wrong-tile", "1m"),
        _row("valid-rejected", "2m"),
        _row("invalid-rejected", "invalid"),
        _row("invalid-accepted", "invalid"),
    ]
    logits = torch.stack(
        [
            _logit_row("2m"),
            _logit_row("1m", invalid_logit=8.0),
            _logit_row("1m", invalid_logit=8.0),
            _logit_row("3m", invalid_logit=0.0),
        ]
    )

    summary = MODULE._summarize_rejection(logits, rows, LABELS)

    assert [row["outcome"] for row in summary["details"]] == [
        "true_accept",
        "false_reject",
        "true_reject",
        "false_accept",
    ]
    assert summary["valid_accept_count"] == 1
    assert summary["valid_false_reject_count"] == 1
    assert summary["invalid_reject_count"] == 1
    assert summary["invalid_false_accept_count"] == 1
    assert summary["valid_accept_recall"] == 0.5
    assert summary["valid_false_reject_rate"] == 0.5
    assert summary["invalid_reject_recall"] == 0.5
    assert summary["invalid_false_accept_rate"] == 0.5
    assert summary["balanced_accuracy"] == 0.5


def test_group_balanced_metric_is_independent_of_raw_group_counts() -> None:
    small_rows = [
        _row("v-a", "1m"),
        _row("v-r", "2m"),
        _row("i-r", "invalid"),
        _row("i-a", "invalid"),
    ]
    small_logits = torch.stack(
        [
            _logit_row("1m"),
            _logit_row("2m", invalid_logit=8.0),
            _logit_row("1m", invalid_logit=8.0),
            _logit_row("1m"),
        ]
    )
    small = MODULE._summarize_rejection(small_logits, small_rows, LABELS)

    large_rows: list[dict[str, object]] = []
    large_logits: list[torch.Tensor] = []
    for index in range(10):
        large_rows.append(_row(f"v-a-{index}", "1m"))
        large_logits.append(_logit_row("1m"))
        large_rows.append(_row(f"v-r-{index}", "2m"))
        large_logits.append(_logit_row("2m", invalid_logit=8.0))
    large_rows.extend([_row("i-r", "invalid"), _row("i-a", "invalid")])
    large_logits.extend(
        [
            _logit_row("1m", invalid_logit=8.0),
            _logit_row("1m"),
        ]
    )
    large = MODULE._summarize_rejection(
        torch.stack(large_logits),
        large_rows,
        LABELS,
    )

    assert small["valid_count"] == 2
    assert large["valid_count"] == 20
    assert small["invalid_count"] == large["invalid_count"] == 2
    assert small["balanced_accuracy"] == large["balanced_accuracy"] == 0.5


def test_rejection_score_auc_is_threshold_free_rank_measure() -> None:
    assert MODULE._roc_auc(
        torch.tensor([2.0, 3.0]),
        torch.tensor([-2.0, -1.0]),
    ) == 1.0
    assert MODULE._roc_auc(
        torch.tensor([-2.0, -1.0]),
        torch.tensor([2.0, 3.0]),
    ) == 0.0
    assert MODULE._roc_auc(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
    ) == 0.5


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Linear(1, len(LABELS))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(inputs).flatten(1))


class _Telemetry:
    def report_scalar(self, **_kwargs) -> None:
        raise AssertionError("Protocol should rely on terminal metric projection")


def _create_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE experiment_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            """CREATE TABLE sample (
            sample_id TEXT PRIMARY KEY, split TEXT NOT NULL, source TEXT NOT NULL,
            base_label TEXT NOT NULL, class_index INTEGER NOT NULL,
            image_gray_u8 BLOB NOT NULL, original_width INTEGER NOT NULL,
            original_height INTEGER NOT NULL, source_image_path TEXT NOT NULL,
            capture_id TEXT, layout_id TEXT, region TEXT,
            detector_review_decision TEXT, invalid_reason TEXT)"""
        )
        connection.execute(
            "INSERT INTO experiment_metadata VALUES (?, ?)",
            ("base_labels", json.dumps(list(LABELS))),
        )
        for index in range(4):
            image = np.full((64, 64), 80 + index, dtype=np.uint8)
            connection.execute(
                """INSERT INTO sample VALUES (
                ?, 'train', 'manual', '1m', 0, ?, 64, 64, 'train.png',
                NULL, NULL, NULL, NULL, NULL)""",
                (f"train-{index}", image.tobytes()),
            )
        for class_index, label in enumerate(LABELS):
            image = np.full((64, 64), 100 + class_index, dtype=np.uint8)
            if label == "invalid":
                source = "detector_manual"
                review = "invalid"
                reason = "background"
            else:
                source = "manual"
                review = None
                reason = None
            connection.execute(
                """INSERT INTO sample VALUES (
                ?, 'manual_val', ?, ?, ?, ?, 64, 64, 'manual.png',
                ?, 'layout', 'hand', ?, ?)""",
                (
                    f"manual-{label}",
                    source,
                    label,
                    class_index,
                    image.tobytes(),
                    f"capture-{label}",
                    review,
                    reason,
                ),
            )


def _run_protocol(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)
    corpus_root = tmp_path / "corpus"
    work_dir = tmp_path / "work"
    corpus_root.mkdir()
    _create_database(corpus_root / "dataset.sqlite")
    evaluate = _load_evaluation_callable(
        "mldb_data",
        "tile-classifier/tile-shape-validity-rejection-v1",
    )
    context = SimpleNamespace(
        parameters={"batch_size": 16},
        corpus=SimpleNamespace(root=corpus_root),
        model=SimpleNamespace(module=_TinyClassifier()),
        telemetry=_Telemetry(),
        work_dir=work_dir,
    )
    return evaluate(context), work_dir


def test_artifacts_are_binary_auditable_and_cover_all_valid_classes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate, work_dir = _run_protocol(tmp_path, monkeypatch)

    assert set(candidate.metrics) == {
        "balanced_accuracy",
        "valid_accept_recall",
        "valid_false_reject_rate",
        "invalid_reject_recall",
        "invalid_false_accept_rate",
        "roc_auc",
    }
    assert all(math.isfinite(float(value)) for value in candidate.metrics.values())
    assert set(candidate.artifacts) == {
        "rejection_summary_table",
        "binary_confusion_matrix",
        "per_valid_class_false_reject_table",
        "per_valid_class_false_reject_plot",
        "error_inspection_contact_sheet",
        "per_sample_details",
        "rejection_roc_curve",
        "rejection_report",
    }
    assert all(
        path.is_file() and path.is_relative_to(work_dir)
        for path in candidate.artifacts.values()
    )

    confusion = json.loads(
        candidate.artifacts["binary_confusion_matrix"].read_text(encoding="utf-8")
    )
    trace = confusion["data"][0]
    assert trace["x"] == ["accept / valid", "reject / invalid"]
    assert trace["y"] == ["true valid", "true invalid"]
    assert len(trace["z"]) == 2
    assert all(len(row) == 2 for row in trace["z"])
    assert all(abs(sum(row) - 1.0) < 1.0e-12 for row in trace["z"])
    assert trace["customdata"][0][0] + trace["customdata"][0][1] == 34
    assert trace["customdata"][1][0] + trace["customdata"][1][1] == 1

    with candidate.artifacts["per_valid_class_false_reject_table"].open(
        encoding="utf-8",
        newline="",
    ) as handle:
        per_class = list(csv.DictReader(handle))
    assert [row["label"] for row in per_class] == list(LABELS[:-1])
    assert len(per_class) == 34

    per_class_plot = json.loads(
        candidate.artifacts["per_valid_class_false_reject_plot"].read_text(
            encoding="utf-8"
        )
    )
    assert per_class_plot["data"][0]["x"] == list(LABELS[:-1])

    with Image.open(candidate.artifacts["error_inspection_contact_sheet"]) as image:
        assert image.width > 0 and image.height > 0

    details = [
        json.loads(line)
        for line in candidate.artifacts["per_sample_details"]
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(details) == 35
    assert {
        "true_group",
        "decision",
        "outcome",
        "argmax_label",
        "best_valid_label",
        "invalid_logit",
        "best_valid_logit",
        "rejection_score",
    }.issubset(details[0])

    report = json.loads(
        candidate.artifacts["rejection_report"].read_text(encoding="utf-8")
    )
    assert report["sample_population"]["sample_count"] == 35
    assert report["sample_population"]["valid_count"] == 34
    assert report["sample_population"]["invalid_count"] == 1
    assert report["sample_population"]["perturbations"] == "none"
    assert report["score_semantics"]["formula"] == (
        "invalid_logit - max(valid_class_logits)"
    )
    assert report["score_semantics"]["runtime_boundary"] == "reject when score >= 0"


def test_protocol_declares_selectable_comparison_visuals() -> None:
    protocol = json.loads(PROTOCOL_YAML.read_text(encoding="utf-8"))
    assert protocol["id"] == "tile-classifier/tile-shape-validity-rejection-v1"
    assert protocol["status"] in {"draft", "sealed"}
    assert protocol["metrics"]["balanced_accuracy"]["preference"] == "higher"
    assert protocol["metrics"]["valid_false_reject_rate"]["preference"] == "lower"
    assert protocol["metrics"]["invalid_false_accept_rate"]["preference"] == "lower"
    for artifact in (
        "binary_confusion_matrix",
        "per_valid_class_false_reject_plot",
        "error_inspection_contact_sheet",
        "rejection_roc_curve",
    ):
        assert protocol["artifacts"][artifact]["study_view"] == "select"
