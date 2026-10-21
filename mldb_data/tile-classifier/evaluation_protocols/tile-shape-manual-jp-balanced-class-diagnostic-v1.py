from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

EXPECTED_LABELS = (
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
    "east","south","west","north","white","green","red","invalid",
)
VALID_LABELS = EXPECTED_LABELS[:-1]
INVALID_INDEX = len(VALID_LABELS)
SPLIT = "manual_val"
IMAGE_SIZE = 64
EXPECTED_VALID_PER_CLASS = 100
EXPECTED_INVALID = 32


def _load(database: Path) -> tuple[list[dict[str, Any]], float, float]:
    with sqlite3.connect(database) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM experiment_metadata"))
        labels = tuple(str(v) for v in json.loads(metadata["base_labels"]))
        if labels != EXPECTED_LABELS:
            raise ValueError("Corpus labels do not match 35-class task")

        pixel_sum = 0.0
        pixel_sq_sum = 0.0
        pixel_count = 0
        for (payload,) in connection.execute(
            "SELECT image_gray_u8 FROM sample WHERE split='train'"
        ):
            values = np.frombuffer(payload, dtype=np.uint8).astype(np.float64) / 255.0
            pixel_sum += float(values.sum())
            pixel_sq_sum += float(np.square(values).sum())
            pixel_count += int(values.size)
        if pixel_count == 0:
            raise ValueError("Corpus train split is empty")
        mean = pixel_sum / pixel_count
        variance = max(pixel_sq_sum / pixel_count - mean * mean, 0.0)
        std = max(math.sqrt(variance), 1.0 / 255.0)

        raw_rows = connection.execute(
            """
            SELECT sample_id, base_label, class_index, image_gray_u8,
                   source, source_partition, source_image_path
            FROM sample
            WHERE split=?
            ORDER BY class_index, sample_id
            """,
            (SPLIT,),
        ).fetchall()

    counts = {label: 0 for label in EXPECTED_LABELS}
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        class_index = int(raw[2])
        if not (0 <= class_index < len(EXPECTED_LABELS)):
            raise ValueError(f"manual_val class index out of range: {class_index}")
        label = str(raw[1])
        if EXPECTED_LABELS[class_index] != label:
            raise ValueError(
                f"manual_val label/index mismatch: {label!r} vs {class_index}"
            )
        image = np.frombuffer(raw[3], dtype=np.uint8).copy()
        if image.size != IMAGE_SIZE * IMAGE_SIZE:
            raise ValueError("manual_val image payload has wrong size")
        counts[label] += 1
        rows.append(
            {
                "sample_id": str(raw[0]),
                "label": label,
                "class_index": class_index,
                "image": image.reshape(IMAGE_SIZE, IMAGE_SIZE),
                "source": str(raw[4]),
                "source_partition": str(raw[5]),
                "source_image_path": str(raw[6]),
            }
        )

    wrong_valid = {
        label: counts[label]
        for label in VALID_LABELS
        if counts[label] != EXPECTED_VALID_PER_CLASS
    }
    if wrong_valid:
        raise ValueError(
            f"manual_val must contain exactly {EXPECTED_VALID_PER_CLASS} samples "
            f"for each valid class: {wrong_valid}"
        )
    if counts["invalid"] != EXPECTED_INVALID:
        raise ValueError(
            f"manual_val invalid support must be {EXPECTED_INVALID}, "
            f"got {counts['invalid']}"
        )
    expected_total = len(VALID_LABELS) * EXPECTED_VALID_PER_CLASS + EXPECTED_INVALID
    if len(rows) != expected_total:
        raise ValueError(
            f"manual_val total must be {expected_total}, got {len(rows)}"
        )
    return rows, mean, std


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate(context):
    batch_size = int(context.parameters["batch_size"])
    recall_gate = float(context.parameters["recall_gate"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not (0.0 <= recall_gate <= 1.0):
        raise ValueError("recall_gate must be within [0,1]")
    if tuple(context.task["target"]["labels"]) != EXPECTED_LABELS:
        raise ValueError("Task labels do not match protocol labels")

    rows, mean, std = _load(context.corpus.root / "dataset.sqlite")
    images = (
        torch.from_numpy(np.stack([row["image"] for row in rows]))
        .float()
        .unsqueeze(1)
        .mul_(1.0 / 255.0)
    )
    targets = torch.tensor(
        [int(row["class_index"]) for row in rows], dtype=torch.long
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = context.model.module.to(device)
    model.eval()
    predictions: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(
                device, non_blocking=device.type == "cuda"
            )
            logits = model(batch.sub(mean).div(std))
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 2
                or logits.shape[1] != len(EXPECTED_LABELS)
            ):
                raise ValueError(
                    f"classifier logits shape invalid: {tuple(logits.shape)}"
                )
            probs = torch.softmax(logits.float(), dim=1)
            confidence, pred = probs.max(dim=1)
            predictions.append(pred.cpu())
            confidences.append(confidence.cpu())

    predicted = torch.cat(predictions)
    confidence = torch.cat(confidences)
    confusion = np.zeros(
        (len(EXPECTED_LABELS), len(EXPECTED_LABELS)), dtype=np.int64
    )
    per_sample: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        target = int(targets[i])
        pred = int(predicted[i])
        confusion[target, pred] += 1
        per_sample.append(
            {
                "sample_id": row["sample_id"],
                "true_label": EXPECTED_LABELS[target],
                "predicted_label": EXPECTED_LABELS[pred],
                "confidence": float(confidence[i]),
                "correct": bool(pred == target),
                "source": row["source"],
                "source_partition": row["source_partition"],
                "source_image_path": row["source_image_path"],
            }
        )

    class_rows: list[dict[str, Any]] = []
    valid_recalls: list[float] = []
    valid_precisions: list[float] = []
    valid_f1s: list[float] = []
    for idx, label in enumerate(EXPECTED_LABELS):
        support = int(confusion[idx].sum())
        predicted_count = int(confusion[:, idx].sum())
        correct = int(confusion[idx, idx])
        recall = _safe_div(correct, support)
        precision = _safe_div(correct, predicted_count)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        wrong = confusion[idx].copy()
        wrong[idx] = 0
        dominant = int(wrong.argmax()) if int(wrong.sum()) else idx
        if idx < INVALID_INDEX:
            valid_recalls.append(recall)
            valid_precisions.append(precision)
            valid_f1s.append(f1)
        class_rows.append(
            {
                "class_index": idx,
                "label": label,
                "sample_count": support,
                "predicted_count": predicted_count,
                "correct_count": correct,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "passes_recall_gate": recall >= recall_gate,
                "error_count": support - correct,
                "dominant_wrong_label": (
                    EXPECTED_LABELS[dominant] if int(wrong.sum()) else ""
                ),
                "dominant_wrong_count": (
                    int(wrong[dominant]) if int(wrong.sum()) else 0
                ),
                "invalid_prediction_count": (
                    int(confusion[idx, INVALID_INDEX])
                    if idx < INVALID_INDEX
                    else 0
                ),
            }
        )

    valid_total = int(confusion[:INVALID_INDEX, :].sum())
    valid_correct = int(
        sum(confusion[i, i] for i in range(INVALID_INDEX))
    )
    valid_to_invalid = int(confusion[:INVALID_INDEX, INVALID_INDEX].sum())
    bottom5 = sorted(valid_recalls)[:5]
    invalid_support = int(confusion[INVALID_INDEX].sum())
    invalid_correct = int(confusion[INVALID_INDEX, INVALID_INDEX])
    invalid_predicted = int(confusion[:, INVALID_INDEX].sum())

    metrics: dict[str, int | float] = {
        "valid_accuracy": _safe_div(valid_correct, valid_total),
        "valid_macro_recall": float(np.mean(valid_recalls)),
        "valid_worst_class_recall": float(min(valid_recalls)),
        "valid_bottom5_class_recall_mean": float(np.mean(bottom5)),
        "valid_class_recall_below_gate_count": int(
            sum(recall < recall_gate for recall in valid_recalls)
        ),
        "valid_to_invalid_rate": _safe_div(valid_to_invalid, valid_total),
        "valid_macro_precision": float(np.mean(valid_precisions)),
        "valid_macro_f1": float(np.mean(valid_f1s)),
        "invalid_recall": _safe_div(invalid_correct, invalid_support),
        "invalid_precision": _safe_div(invalid_correct, invalid_predicted),
    }


    work = context.work_dir
    work.mkdir(parents=True, exist_ok=True)
    class_path = work / "manual-jp-balanced-class-metrics.csv"
    confusion_path = work / "manual-jp-balanced-confusion.plotly.json"
    predictions_path = work / "manual-jp-balanced-predictions.jsonl"
    report_path = work / "manual-jp-balanced-report.json"

    _write_csv(class_path, class_rows)
    confusion_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "type": "heatmap",
                        "x": list(EXPECTED_LABELS),
                        "y": list(EXPECTED_LABELS),
                        "z": confusion.tolist(),
                        "colorscale": "Blues",
                        "hovertemplate": (
                            "GT=%{y}<br>pred=%{x}<br>count=%{z}<extra></extra>"
                        ),
                    }
                ],
                "layout": {
                    "title": "Manual + unused JP validation 35-class confusion",
                    "xaxis": {"title": "Predicted class"},
                    "yaxis": {"title": "True class", "autorange": "reversed"},
                    "height": 900,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in per_sample
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(
            {
                "schema": (
                    "mjtensu.recognition/"
                    "manual-jp-balanced-class-diagnostic-report/v1"
                ),
                "population": {
                    "split": SPLIT,
                    "valid_classes": len(VALID_LABELS),
                    "valid_per_class": EXPECTED_VALID_PER_CLASS,
                    "invalid_support": EXPECTED_INVALID,
                    "included": len(rows),
                },
                "normalization": {"mean": mean, "std": std},
                "recall_gate": recall_gate,
                "metrics": metrics,
                "bottom5_classes": [
                    row["label"]
                    for row in sorted(
                        class_rows[:INVALID_INDEX],
                        key=lambda row: (float(row["recall"]), str(row["label"])),
                    )[:5]
                ],
                "per_class": class_rows,
                "notes": {
                    "valid_metrics": (
                        "Computed across the 34 valid tile classes only. "
                        "Each valid class has exactly 100 GT samples."
                    ),
                    "invalid": (
                        "Invalid remains a separate 32-sample manual holdout. "
                        "Its recall/precision are not mixed into valid-class macro metrics."
                    ),
                    "valid_to_invalid": (
                        "Fraction of valid-tile GT samples rejected into the "
                        "invalid output."
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return EvaluationCandidate(
        metrics=metrics,
        artifacts={
            "class_metrics_table": class_path,
            "confusion_matrix": confusion_path,
            "per_sample_predictions": predictions_path,
            "report": report_path,
        },
    )
