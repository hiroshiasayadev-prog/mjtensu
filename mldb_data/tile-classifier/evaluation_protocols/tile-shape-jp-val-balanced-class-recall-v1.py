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
SPLIT = "jp_val"
IMAGE_SIZE = 64
EXPECTED_PER_CLASS = 200


def _load(database: Path) -> tuple[list[dict[str, Any]], float, float]:
    with sqlite3.connect(database) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM experiment_metadata"))
        labels = tuple(str(v) for v in json.loads(metadata["base_labels"]))
        if labels != EXPECTED_LABELS:
            raise ValueError("Corpus labels do not match 35-class task")

        pixel_sum = 0.0
        pixel_sq_sum = 0.0
        pixel_count = 0
        for (payload,) in connection.execute("SELECT image_gray_u8 FROM sample WHERE split='train'"):
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

    counts = {label: 0 for label in VALID_LABELS}
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        class_index = int(raw[2])
        if not (0 <= class_index < len(VALID_LABELS)):
            raise ValueError(f"jp_val contains non-known class index {class_index}")
        label = str(raw[1])
        if VALID_LABELS[class_index] != label:
            raise ValueError(f"jp_val label/index mismatch: {label} vs {class_index}")
        image = np.frombuffer(raw[3], dtype=np.uint8).copy()
        if image.size != IMAGE_SIZE * IMAGE_SIZE:
            raise ValueError("jp_val image payload has wrong size")
        counts[label] += 1
        rows.append({
            "sample_id": str(raw[0]),
            "label": label,
            "class_index": class_index,
            "image": image.reshape(IMAGE_SIZE, IMAGE_SIZE),
            "source": str(raw[4]),
            "source_partition": str(raw[5]),
            "source_image_path": str(raw[6]),
        })
    wrong_counts = {label: count for label, count in counts.items() if count != EXPECTED_PER_CLASS}
    if wrong_counts:
        raise ValueError(f"jp_val must contain exactly {EXPECTED_PER_CLASS} samples/class: {wrong_counts}")
    if len(rows) != len(VALID_LABELS) * EXPECTED_PER_CLASS:
        raise ValueError("jp_val total population is not 34x200")
    return rows, mean, std


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(context):
    batch_size = int(context.parameters["batch_size"])
    recall_gate = float(context.parameters["recall_gate"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not (0.0 <= recall_gate <= 1.0):
        raise ValueError("recall_gate must be within [0,1]")
    task_labels = tuple(context.task["target"]["labels"])
    if task_labels != EXPECTED_LABELS:
        raise ValueError("Task labels do not match protocol labels")

    rows, mean, std = _load(context.corpus.root / "dataset.sqlite")
    images = torch.from_numpy(np.stack([row["image"] for row in rows])).float().unsqueeze(1).mul_(1.0 / 255.0)
    targets = torch.tensor([int(row["class_index"]) for row in rows], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = context.model.module.to(device)
    model.eval()
    predictions: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].to(device, non_blocking=device.type == "cuda")
            logits = model(batch.sub(mean).div(std))
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != len(EXPECTED_LABELS):
                raise ValueError(f"classifier logits shape invalid: {tuple(logits.shape)}")
            probs = torch.softmax(logits.float(), dim=1)
            confidence, pred = probs.max(dim=1)
            predictions.append(pred.cpu())
            confidences.append(confidence.cpu())
    predicted = torch.cat(predictions)
    confidence = torch.cat(confidences)

    confusion = np.zeros((len(VALID_LABELS), len(EXPECTED_LABELS)), dtype=np.int64)
    per_sample: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        target = int(targets[i])
        pred = int(predicted[i])
        confusion[target, pred] += 1
        per_sample.append({
            "sample_id": row["sample_id"],
            "true_label": VALID_LABELS[target],
            "predicted_label": EXPECTED_LABELS[pred],
            "confidence": float(confidence[i]),
            "correct": pred == target,
            "source": row["source"],
            "source_partition": row["source_partition"],
            "source_image_path": row["source_image_path"],
        })

    class_rows: list[dict[str, Any]] = []
    recalls: list[float] = []
    for idx, label in enumerate(VALID_LABELS):
        count = int(confusion[idx].sum())
        correct = int(confusion[idx, idx])
        recall = correct / count
        recalls.append(recall)
        wrong = confusion[idx].copy(); wrong[idx] = 0
        dominant = int(wrong.argmax()) if int(wrong.sum()) else idx
        class_rows.append({
            "class_index": idx,
            "label": label,
            "sample_count": count,
            "correct_count": correct,
            "recall": recall,
            "passes_gate": recall >= recall_gate,
            "error_count": count - correct,
            "dominant_wrong_label": EXPECTED_LABELS[dominant] if int(wrong.sum()) else "",
            "dominant_wrong_count": int(wrong[dominant]) if int(wrong.sum()) else 0,
            "invalid_prediction_count": int(confusion[idx, len(VALID_LABELS)]),
        })

    total = len(rows)
    correct_total = int(sum(int(confusion[i, i]) for i in range(len(VALID_LABELS))))
    invalid_predictions = int(confusion[:, len(VALID_LABELS)].sum())
    metrics: dict[str, int | float] = {
        "accuracy": correct_total / total,
        "macro_recall": float(sum(recalls) / len(recalls)),
        "worst_class_recall": float(min(recalls)),
        "class_recall_below_gate_count": int(sum(r < recall_gate for r in recalls)),
        "invalid_prediction_rate": invalid_predictions / total,
    }

    work = context.work_dir
    work.mkdir(parents=True, exist_ok=True)
    table_path = work / "jp-val-balanced-class-recall.csv"
    confusion_path = work / "jp-val-balanced-confusion.plotly.json"
    predictions_path = work / "jp-val-balanced-predictions.jsonl"
    report_path = work / "jp-val-balanced-report.json"
    _write_csv(table_path, class_rows)
    confusion_path.write_text(json.dumps({
        "data":[{"type":"heatmap","x":list(EXPECTED_LABELS),"y":list(VALID_LABELS),"z":confusion.tolist(),"colorscale":"Blues"}],
        "layout":{"title":"JP validation balanced 34-class confusion counts","xaxis":{"title":"Predicted class"},"yaxis":{"title":"True class","autorange":"reversed"},"height":900},
    },ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")
    predictions_path.write_text("".join(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n" for row in per_sample),encoding="utf-8")
    report_path.write_text(json.dumps({
        "schema":"mjtensu.recognition/jp-val-balanced-class-recall-report/v1",
        "population":{"split":SPLIT,"classes":len(VALID_LABELS),"per_class":EXPECTED_PER_CLASS,"included":total},
        "normalization":{"mean":mean,"std":std},
        "recall_gate":recall_gate,
        "metrics":metrics,
        "notes":{
            "purpose":"balanced known-tile classification feasibility gate",
            "domain":"held-out JP tile-image validation split from the classifier corpus",
            "invalid":"invalid has no GT rows and remains a prediction-only error destination",
        },
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    return EvaluationCandidate(metrics=metrics,artifacts={
        "class_recall_table":table_path,
        "confusion_matrix":confusion_path,
        "per_sample_predictions":predictions_path,
        "report":report_path,
    })
