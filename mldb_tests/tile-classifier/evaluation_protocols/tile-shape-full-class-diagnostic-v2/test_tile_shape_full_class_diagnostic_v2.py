from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable


LABELS = (
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "east", "south", "west", "north", "white", "green", "red", "invalid",
)
CONDITION_COUNT = 15
VALID_SUPPORT = 100
INVALID_SUPPORT = 32


class _NoopClassifier(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (inputs.shape[0], len(LABELS)),
            dtype=inputs.dtype,
            device=inputs.device,
        )


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


def _create_balanced_database(path: Path) -> None:
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
        for index in range(4):
            image = np.full((64, 64), 100 + index, dtype=np.uint8)
            connection.execute(
                "INSERT INTO sample VALUES (?, 'train', '1m', 0, ?, 42, 58, 'fixture', NULL, NULL, NULL, 'train.png')",
                (f"train-{index}", image.tobytes()),
            )
        for class_index, label in enumerate(LABELS):
            support = INVALID_SUPPORT if label == "invalid" else VALID_SUPPORT
            for sample_index in range(support):
                image = np.full(
                    (64, 64),
                    40 + ((class_index * 3 + sample_index) % 180),
                    dtype=np.uint8,
                )
                connection.execute(
                    "INSERT INTO sample VALUES (?, 'manual_val', ?, ?, ?, 42, 58, 'fixture', ?, 'layout', 'hand', 'manual.png')",
                    (
                        f"{label}-{sample_index:03d}",
                        label,
                        class_index,
                        image.tobytes(),
                        f"capture-{label}-{sample_index:03d}",
                    ),
                )


def _run_protocol(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    corpus_root = tmp_path / "corpus"
    work_dir = tmp_path / "work"
    corpus_root.mkdir()
    _create_balanced_database(corpus_root / "dataset.sqlite")

    evaluate = _load_evaluation_callable(
        "mldb_data",
        "tile-classifier/tile-shape-full-class-diagnostic-v2",
    )

    # v1 already exercises the geometry implementation itself. Keep this v2 test
    # focused on balanced-population semantics and the new per-class Study table.
    monkeypatch.setitem(
        evaluate.__globals__,
        "_apply_geometry_case",
        lambda images, rows, case: images,
    )
    monkeypatch.setitem(
        evaluate.__globals__,
        "_shift_images",
        lambda images, dx, dy: images,
    )
    monkeypatch.setitem(
        evaluate.__globals__,
        "_infer",
        lambda model, images_01, **kwargs: torch.zeros(
            (images_01.shape[0], len(LABELS)), dtype=torch.float32
        ),
    )

    telemetry = _Telemetry()
    candidate = evaluate(
        SimpleNamespace(
            parameters={"batch_size": 512},
            corpus=SimpleNamespace(root=corpus_root),
            model=SimpleNamespace(module=_NoopClassifier()),
            telemetry=telemetry,
            work_dir=work_dir,
        )
    )
    return candidate, telemetry, work_dir


def test_balanced_full_class_v2_writes_selectable_per_class_robustness_table(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate, telemetry, work_dir = _run_protocol(tmp_path, monkeypatch)

    assert "mean_condition_accuracy" in candidate.metrics
    assert "worst_condition_accuracy" in candidate.metrics
    assert "per_class_robustness_table" in candidate.artifacts
    assert all(path.is_file() for path in candidate.artifacts.values())

    with candidate.artifacts["per_class_robustness_table"].open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(LABELS)
    assert [row["label"] for row in rows] == list(LABELS)
    assert rows[0]["sample_count"] == str(VALID_SUPPORT)
    assert rows[-1]["sample_count"] == str(INVALID_SUPPORT)
    assert rows[-1]["kind"] == "invalid"
    assert all(row["worst_condition"] == "front-facing" for row in rows)
    assert set(rows[0]) == {
        "class_index",
        "label",
        "kind",
        "sample_count",
        "front_recall",
        "mean_condition_recall",
        "worst_condition_recall",
        "worst_condition",
        "worst_top_wrong_label",
        "worst_top_wrong_count",
        "worst_top_wrong_rate",
    }

    report = json.loads(
        candidate.artifacts["diagnostic_report"].read_text(encoding="utf-8")
    )
    assert report["balanced_manual_val"] == {
        "valid_class_support": VALID_SUPPORT,
        "invalid_support": INVALID_SUPPORT,
        "valid_class_count": 34,
    }
    assert len(report["condition_order"]) == CONDITION_COUNT
    assert len(report["per_class_robustness"]) == len(LABELS)

    robustness_events = [
        event
        for event in telemetry.events
        if event[0] == "classifier/full_class_robustness"
    ]
    assert len(robustness_events) == CONDITION_COUNT * 3

    protocol = json.loads(
        (
            Path(__file__).parents[4]
            / "mldb_data"
            / "tile-classifier"
            / "evaluation_protocols"
            / "tile-shape-full-class-diagnostic-v2.yaml"
        ).read_text(encoding="utf-8")
    )
    table = protocol["artifacts"]["per_class_robustness_table"]
    assert table["format"] == "csv"
    assert table["study_view"] == "select"


def test_balanced_full_class_v2_rejects_wrong_class_support(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    database = corpus_root / "dataset.sqlite"
    _create_balanced_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM sample WHERE sample_id = ?",
            ("1m-099",),
        )

    evaluate = _load_evaluation_callable(
        "mldb_data",
        "tile-classifier/tile-shape-full-class-diagnostic-v2",
    )
    context = SimpleNamespace(
        parameters={"batch_size": 512},
        corpus=SimpleNamespace(root=corpus_root),
        model=SimpleNamespace(module=_NoopClassifier()),
        telemetry=_Telemetry(),
        work_dir=tmp_path / "work",
    )

    import pytest

    with pytest.raises(ValueError, match="exactly 100 samples"):
        evaluate(context)
