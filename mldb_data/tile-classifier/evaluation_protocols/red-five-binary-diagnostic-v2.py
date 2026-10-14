from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

IMAGE_SIZE = 64
WORKERS = 12


@dataclass
class Counter:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, target: int, prediction: int) -> None:
        if target == 1 and prediction == 1:
            self.tp += 1
        elif target == 0 and prediction == 0:
            self.tn += 1
        elif target == 0 and prediction == 1:
            self.fp += 1
        elif target == 1 and prediction == 0:
            self.fn += 1
        else:
            raise ValueError("binary target/prediction required")

    def summary(self) -> dict[str, int | float]:
        red = self.tp + self.fn
        normal = self.tn + self.fp
        count = red + normal
        recall = self.tp / red if red else 0.0
        specificity = self.tn / normal if normal else 0.0
        accuracy = (self.tp + self.tn) / count if count else 0.0
        return {
            "count": count,
            "errors": self.fp + self.fn,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "accuracy": accuracy,
            "balanced_accuracy": 0.5 * (recall + specificity) if red and normal else accuracy,
            "red_recall": recall,
            "normal_specificity": specificity,
            "false_red_rate": self.fp / normal if normal else 0.0,
            "false_normal_rate": self.fn / red if red else 0.0,
        }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _preprocess(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as source:
        source.load()
        image = source.convert("RGB")
    width, height = image.size
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = max(1, min(IMAGE_SIZE, int(math.floor(width * scale + 0.5))))
    resized_height = max(1, min(IMAGE_SIZE, int(math.floor(height * scale + 0.5))))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.uint8)
    if width == 1 or height == 1:
        border = pixels.reshape(-1, 3)
    else:
        border = np.concatenate((pixels[0], pixels[-1], pixels[1:-1, 0], pixels[1:-1, -1]), axis=0)
    fill = tuple(int(value) for value in np.median(border, axis=0))
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), fill)
    canvas.paste(resized, ((IMAGE_SIZE - resized_width) // 2, (IMAGE_SIZE - resized_height) // 2))
    return np.asarray(canvas, dtype=np.uint8).copy()


def _infer(model, images_u8, mean, std, device):
    images = torch.from_numpy(images_u8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean_tensor = images.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = images.new_tensor(std).view(1, 3, 1, 1)
    with torch.inference_mode():
        logits = model((images - mean_tensor) / std_tensor)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"expected [N,2] logits, got {tuple(logits.shape)}")
    return logits.softmax(1)[:, 1].cpu().numpy(), logits.argmax(1).cpu().numpy()


def _load_rows(database: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            "SELECT sample_id,crop_id,source,source_partition,suit,is_red,source_label,image_png,source_image_path,source_annotation_id,capture_id,brightness,shadow FROM sample ORDER BY sample_id"
        ).fetchall()
    finally:
        connection.close()


def _load_warm_rows(database: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            "SELECT crop_id,tile_label,image_png,capture_id,brightness,shadow,source_image_path FROM sample ORDER BY crop_id"
        ).fetchall()
    finally:
        connection.close()


def _counter_row(scope: str, key: str, counter: Counter) -> dict[str, Any]:
    return {"scope": scope, "key": key, **counter.summary()}


def _write_group_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("scope","key","count","errors","tp","tn","fp","fn","accuracy","balanced_accuracy","red_recall","normal_specificity","false_red_rate","false_normal_rate")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _confusion(counter: Counter) -> dict[str, Any]:
    labels = ["normal-five", "red-five"]
    return {"data":[{"type":"heatmap","x":labels,"y":labels,"z":[[counter.tn,counter.fp],[counter.fn,counter.tp]],"colorscale":"Blues"}],"layout":{"title":"Specialist five-scope red-five confusion counts","xaxis":{"title":"predicted class"},"yaxis":{"title":"true class","autorange":"reversed"}}}


def evaluate(context):
    batch_size = int(context.parameters["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    root = context.corpus.root
    dataset = root / "dataset.sqlite"
    warm_dataset = root / "warm.sqlite"
    if not dataset.is_file() or not warm_dataset.is_file():
        raise FileNotFoundError("red-five corpus requires dataset.sqlite and warm.sqlite")

    normalization = _load_json(root / "normalization.json")
    mean = tuple(float(value) for value in normalization["mean"])
    std = tuple(float(value) for value in normalization["std"])
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("RGB normalization must contain three positive std values")

    audit = _load_json(root / "audit.json")
    corrected = audit.get("corrected", {})
    specialist_excluded = audit.get("specialist_excluded", {})
    if not isinstance(corrected, dict) or not isinstance(specialist_excluded, dict):
        raise ValueError("audit mappings are invalid")
    if not set(specialist_excluded) <= set(corrected):
        raise ValueError("specialist_excluded must be a subset of corrected")

    rows = _load_rows(dataset)
    source_counter = Counter()
    physical_counter = Counter()
    specialist_counter = Counter()
    manual_counter = Counter()
    by_source: dict[str, Counter] = {}
    by_suit = {suit: Counter() for suit in "mps"}
    physical_errors: list[dict[str, Any]] = []
    specialist_errors: list[dict[str, Any]] = []
    seen_audit: set[str] = set()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = context.model.module.to(device)
    model.eval()
    workers = max(1, min(WORKERS, batch_size))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            images = np.stack(list(executor.map(lambda row: _preprocess(row["image_png"]), batch_rows)))
            probabilities, predictions = _infer(model, images, mean, std, device)
            for row, probability, prediction_value in zip(batch_rows, probabilities, predictions):
                sample_id = str(row["sample_id"])
                source_target = int(row["is_red"])
                prediction = int(prediction_value)
                source_counter.add(source_target, prediction)
                source_key = f"{row['source']}/{row['source_partition']}"
                by_source.setdefault(source_key, Counter()).add(source_target, prediction)
                if str(row["source"]) == "manual":
                    manual_counter.add(source_target, prediction)

                target = source_target
                audited_suit = str(row["suit"])
                if sample_id in corrected:
                    seen_audit.add(sample_id)
                    target = int(corrected[sample_id]["target"])
                    audited_suit = str(corrected[sample_id]["suit"])

                physical_counter.add(target, prediction)
                detail = {
                    "sample_id": sample_id,
                    "source": str(row["source"]),
                    "source_partition": str(row["source_partition"]),
                    "source_label": str(row["source_label"]),
                    "source_target": source_target,
                    "audited_target": target,
                    "audited_suit": audited_suit,
                    "prediction": prediction,
                    "red_probability": float(probability),
                    "source_image_path": str(row["source_image_path"]),
                    "source_annotation_id": str(row["source_annotation_id"]),
                }
                if target != prediction:
                    physical_errors.append(detail)
                if sample_id in specialist_excluded:
                    seen_audit.add(sample_id)
                    continue
                specialist_counter.add(target, prediction)
                by_suit[audited_suit].add(target, prediction)
                if target != prediction:
                    specialist_errors.append(detail)

    missing_audit = (set(corrected) | set(specialist_excluded)) - seen_audit
    if missing_audit:
        raise ValueError(f"audit references missing samples: {sorted(missing_audit)}")

    warm_counter = Counter()
    warm_details: list[dict[str, Any]] = []
    warm_rows = _load_warm_rows(warm_dataset)
    if not warm_rows:
        raise ValueError("warm.sqlite is empty")
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(warm_rows))) as executor:
        warm_images = np.stack(list(executor.map(lambda row: _preprocess(row["image_png"]), warm_rows)))
    warm_probabilities, warm_predictions = _infer(model, warm_images, mean, std, device)
    for row, probability, prediction_value in zip(warm_rows, warm_probabilities, warm_predictions):
        tile_label = str(row["tile_label"])
        target = 1 if tile_label.startswith("red5") else 0
        prediction = int(prediction_value)
        warm_counter.add(target, prediction)
        warm_details.append({
            "crop_id": str(row["crop_id"]),
            "tile_label": tile_label,
            "target": target,
            "prediction": prediction,
            "red_probability": float(probability),
            "source_image_path": str(row["source_image_path"]),
        })

    source = source_counter.summary()
    physical = physical_counter.summary()
    specialist = specialist_counter.summary()
    manual = manual_counter.summary()
    warm = warm_counter.summary()
    metrics: dict[str, int | float] = {
        "specialist_five_accuracy": float(specialist["accuracy"]),
        "specialist_five_balanced_accuracy": float(specialist["balanced_accuracy"]),
        "specialist_five_red_recall": float(specialist["red_recall"]),
        "specialist_five_normal_specificity": float(specialist["normal_specificity"]),
        "specialist_five_false_red_rate": float(specialist["false_red_rate"]),
        "specialist_five_false_normal_rate": float(specialist["false_normal_rate"]),
        "specialist_five_error_count": int(specialist["errors"]),
        "historical_audited_physical_accuracy": float(physical["accuracy"]),
        "historical_audited_physical_error_count": int(physical["errors"]),
        "manual_accuracy": float(manual["accuracy"]),
        "warm_accuracy": float(warm["accuracy"]),
        "source_label_accuracy": float(source["accuracy"]),
    }

    work = context.work_dir
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / "red-five-diagnostic-report.json"
    group_path = work / "red-five-group-summary.csv"
    specialist_errors_path = work / "red-five-specialist-errors.jsonl"
    physical_errors_path = work / "red-five-historical-audited-errors.jsonl"
    confusion_path = work / "red-five-specialist-confusion.plotly.json"
    warm_path = work / "red-five-warm-holdout.jsonl"

    groups = [
        _counter_row("overall", "source-label", source_counter),
        _counter_row("overall", "historical-audited-physical", physical_counter),
        _counter_row("overall", "specialist-five-scope", specialist_counter),
        _counter_row("overall", "manual-capture", manual_counter),
        _counter_row("overall", "warm-holdout", warm_counter),
    ]
    groups.extend(_counter_row("source_partition", key, value) for key, value in sorted(by_source.items()))
    groups.extend(_counter_row("specialist_suit", key, value) for key, value in sorted(by_suit.items()))
    _write_group_summary(group_path, groups)

    with specialist_errors_path.open("w", encoding="utf-8") as handle:
        for row in specialist_errors:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with physical_errors_path.open("w", encoding="utf-8") as handle:
        for row in physical_errors:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with warm_path.open("w", encoding="utf-8") as handle:
        for row in warm_details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    confusion_path.write_text(json.dumps(_confusion(specialist_counter), ensure_ascii=False) + "\n", encoding="utf-8")

    report = {
        "schema": "mjtensu.recognition/red-five-binary-diagnostic-report/v2",
        "task": "tile-classifier/red-five-binary-classification-v1",
        "normalization": {"mean": list(mean), "std": list(std)},
        "metrics": metrics,
        "source_label": source,
        "historical_audited_physical": physical,
        "specialist_five_scope": specialist,
        "manual_capture": manual,
        "warm_holdout": warm,
        "audit": {
            "corrected_samples": sorted(corrected),
            "specialist_excluded_non_five_samples": sorted(specialist_excluded),
        },
        "groups": groups,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return EvaluationCandidate(
        metrics=metrics,
        artifacts={
            "diagnostic_report": report_path,
            "group_summary": group_path,
            "specialist_errors": specialist_errors_path,
            "historical_audited_errors": physical_errors_path,
            "specialist_confusion": confusion_path,
            "warm_holdout_details": warm_path,
        },
    )
