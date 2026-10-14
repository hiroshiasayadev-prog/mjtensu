from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate


IMAGE_SIZE = 64
STANDARD_SPLITS = ("jp_val", "jp_test", "manual_val")
SOURCE_LABELS = ("5m", "red5m", "5p", "red5p", "5s", "red5s")


def _preprocess_rgb_u8(image_png: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_png)) as source:
        source.load()
        image = source.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid crop size: {image.size}")
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = max(1, min(IMAGE_SIZE, int(math.floor(width * scale + 0.5))))
    resized_height = max(1, min(IMAGE_SIZE, int(math.floor(height * scale + 0.5))))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.uint8)
    if width == 1 or height == 1:
        border = pixels.reshape(-1, 3)
    else:
        border = np.concatenate(
            [pixels[0], pixels[-1], pixels[1:-1, 0], pixels[1:-1, -1]],
            axis=0,
        )
    fill = tuple(int(value) for value in np.median(border, axis=0))
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=fill)
    canvas.paste(resized, ((IMAGE_SIZE - resized_width) // 2, (IMAGE_SIZE - resized_height) // 2))
    return np.asarray(canvas, dtype=np.uint8).copy()


def _load_training_stats(database: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    weighted_pixels = 0.0
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT image_rgb_u8, train_repeat FROM sample WHERE split='train'"
        )
        for payload, repeat in rows:
            image = np.frombuffer(payload, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
            values = image.astype(np.float64) * (1.0 / 255.0)
            weight = int(repeat)
            channel_sum += values.sum(axis=(0, 1)) * weight
            channel_square_sum += np.square(values).sum(axis=(0, 1)) * weight
            weighted_pixels += IMAGE_SIZE * IMAGE_SIZE * weight
    if weighted_pixels <= 0:
        raise ValueError("red-five corpus train split is empty")
    mean = channel_sum / weighted_pixels
    variance = np.maximum(channel_square_sum / weighted_pixels - mean * mean, 0.0)
    std = np.maximum(np.sqrt(variance), 1.0 / 255.0)
    return tuple(float(v) for v in mean), tuple(float(v) for v in std)


def _load_standard_rows(database: Path) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in STANDARD_SPLITS)
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            f"""
            SELECT sample_id, split, source, source_partition, suit, is_red,
                   source_label, image_rgb_u8, capture_id, brightness, shadow,
                   source_image_path
            FROM sample
            WHERE split IN ({placeholders})
            ORDER BY split, suit, is_red, sample_id
            """,
            STANDARD_SPLITS,
        ).fetchall()
    result = []
    for row in rows:
        source_label = str(row[6])
        if source_label not in SOURCE_LABELS:
            raise ValueError(f"unexpected red-five source label: {source_label}")
        result.append(
            {
                "sample_id": str(row[0]),
                "population": str(row[1]),
                "source": str(row[2]),
                "source_partition": str(row[3]),
                "suit": str(row[4]),
                "target": int(row[5]),
                "source_label": source_label,
                "image": np.frombuffer(row[7], dtype=np.uint8).copy().reshape(IMAGE_SIZE, IMAGE_SIZE, 3),
                "capture_id": None if row[8] is None else str(row[8]),
                "brightness": None if row[9] is None else str(row[9]),
                "shadow": None if row[10] is None else str(row[10]),
                "source_image_path": str(row[11]),
            }
        )
    if not result:
        raise ValueError("red-five standard evaluation population is empty")
    return result


def _load_warm_rows(database: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT crop_id, tile_label, image_png, capture_id, brightness, shadow, source_image_path
            FROM sample
            ORDER BY crop_id
            """
        ).fetchall()
    result = []
    for row in rows:
        label = str(row[1])
        if label not in SOURCE_LABELS:
            raise ValueError(f"unexpected warm red-five label: {label}")
        result.append(
            {
                "sample_id": str(row[0]),
                "population": "warm_holdout",
                "source": "manual",
                "source_partition": "warm_holdout",
                "suit": label[-1],
                "target": int(label.startswith("red5")),
                "source_label": label,
                "image": _preprocess_rgb_u8(row[2]),
                "capture_id": None if row[3] is None else str(row[3]),
                "brightness": None if row[4] is None else str(row[4]),
                "shadow": None if row[5] is None else str(row[5]),
                "source_image_path": str(row[6]),
            }
        )
    if not result:
        raise ValueError("red-five warm holdout is empty")
    return result


def _rotate_batch(images: torch.Tensor, angle_deg: float) -> torch.Tensor:
    if abs(angle_deg) < 1.0e-12:
        return images
    radians = images.new_full((images.shape[0],), float(angle_deg) * math.pi / 180.0)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    theta = torch.zeros((images.shape[0], 2, 3), device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _infer(
    model: torch.nn.Module,
    rows: Sequence[dict[str, Any]],
    *,
    mean: Sequence[float],
    std: Sequence[float],
    batch_size: int,
    angle_deg: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    predictions: list[np.ndarray] = []
    red_probabilities: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = np.stack([row["image"] for row in batch_rows])
        tensor = torch.from_numpy(images).to(device=device).permute(0, 3, 1, 2).float().mul_(1.0 / 255.0)
        tensor = _rotate_batch(tensor, angle_deg)
        tensor = tensor.sub(mean_tensor).div(std_tensor)
        with torch.inference_mode():
            logits = model(tensor)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"red-five model must emit [N,2] logits, got {tuple(logits.shape)}")
        probabilities = logits.softmax(dim=1)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        red_probabilities.append(probabilities[:, 1].cpu().numpy())
    return np.concatenate(predictions), np.concatenate(red_probabilities)


def _summary(targets: np.ndarray, predictions: np.ndarray) -> dict[str, int | float]:
    tp = int(np.sum((targets == 1) & (predictions == 1)))
    tn = int(np.sum((targets == 0) & (predictions == 0)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    count = tp + tn + fp + fn
    red_count = tp + fn
    normal_count = tn + fp
    accuracy = (tp + tn) / count if count else 0.0
    red_recall = tp / red_count if red_count else 0.0
    normal_specificity = tn / normal_count if normal_count else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * red_recall / (precision + red_recall) if precision + red_recall else 0.0
    balanced = 0.5 * (red_recall + normal_specificity) if red_count and normal_count else accuracy
    return {
        "count": count,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "red_precision": precision,
        "red_recall": red_recall,
        "normal_specificity": normal_specificity,
        "red_f1": f1,
    }


def _subset(rows: Sequence[dict[str, Any]], indices: Iterable[int]) -> list[dict[str, Any]]:
    return [rows[index] for index in indices]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(context):
    batch_size = int(context.parameters["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    angles = [float(value) for value in context.parameters["eval_angles"]]
    if not angles:
        raise ValueError("eval_angles must not be empty")

    dataset = context.corpus.root / "dataset.sqlite"
    warm_database = context.corpus.root / "warm_holdout.sqlite"
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if not warm_database.is_file():
        raise FileNotFoundError(warm_database)

    mean, std = _load_training_stats(dataset)
    standard_rows = _load_standard_rows(dataset)
    warm_rows = _load_warm_rows(warm_database)
    targets = np.asarray([row["target"] for row in standard_rows], dtype=np.int64)
    warm_targets = np.asarray([row["target"] for row in warm_rows], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = context.model.module.to(device)
    model.eval()

    zero_index = min(range(len(angles)), key=lambda index: abs(angles[index]))
    predictions_by_angle: list[np.ndarray] = []
    probabilities_by_angle: list[np.ndarray] = []
    angle_rows: list[dict[str, Any]] = []
    for angle_index, angle in enumerate(angles):
        predictions, red_probability = _infer(
            model,
            standard_rows,
            mean=mean,
            std=std,
            batch_size=batch_size,
            angle_deg=angle,
            device=device,
        )
        predictions_by_angle.append(predictions)
        probabilities_by_angle.append(red_probability)
        summary = _summary(targets, predictions)
        angle_rows.append({"angle_index": angle_index, "angle_deg": angle, **summary})
        context.telemetry.report_scalar(
            group="red-five/angle-robustness",
            series="balanced_accuracy",
            value=float(summary["balanced_accuracy"]),
            step=angle_index,
        )
        context.telemetry.report_scalar(
            group="red-five/angle-robustness",
            series="red_recall",
            value=float(summary["red_recall"]),
            step=angle_index,
        )

    zero_predictions = predictions_by_angle[zero_index]
    zero_probabilities = probabilities_by_angle[zero_index]
    standard_summary = _summary(targets, zero_predictions)

    population_rows: list[dict[str, Any]] = []
    population_summaries: dict[str, dict[str, int | float]] = {}
    for population in STANDARD_SPLITS:
        indices = [index for index, row in enumerate(standard_rows) if row["population"] == population]
        subset_targets = targets[indices]
        subset_predictions = zero_predictions[indices]
        summary = _summary(subset_targets, subset_predictions)
        population_summaries[population] = summary
        population_rows.append({"population": population, **summary})
    for suit in ("m", "p", "s"):
        indices = [index for index, row in enumerate(standard_rows) if row["suit"] == suit]
        summary = _summary(targets[indices], zero_predictions[indices])
        population_summaries[f"suit_{suit}"] = summary
        population_rows.append({"population": f"suit_{suit}", **summary})

    warm_predictions, warm_probabilities = _infer(
        model,
        warm_rows,
        mean=mean,
        std=std,
        batch_size=batch_size,
        angle_deg=0.0,
        device=device,
    )
    warm_summary = _summary(warm_targets, warm_predictions)
    population_summaries["warm_holdout"] = warm_summary
    population_rows.append({"population": "warm_holdout", **warm_summary})

    error_rows: list[dict[str, Any]] = []
    for row, target, prediction, probability in zip(
        standard_rows, targets, zero_predictions, zero_probabilities, strict=True
    ):
        if int(target) != int(prediction):
            error_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "population": row["population"],
                    "suit": row["suit"],
                    "source_label": row["source_label"],
                    "target": int(target),
                    "prediction": int(prediction),
                    "red_probability": float(probability),
                    "capture_id": row["capture_id"],
                    "brightness": row["brightness"],
                    "shadow": row["shadow"],
                    "source_image_path": row["source_image_path"],
                }
            )
    for row, target, prediction, probability in zip(
        warm_rows, warm_targets, warm_predictions, warm_probabilities, strict=True
    ):
        if int(target) != int(prediction):
            error_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "population": "warm_holdout",
                    "suit": row["suit"],
                    "source_label": row["source_label"],
                    "target": int(target),
                    "prediction": int(prediction),
                    "red_probability": float(probability),
                    "capture_id": row["capture_id"],
                    "brightness": row["brightness"],
                    "shadow": row["shadow"],
                    "source_image_path": row["source_image_path"],
                }
            )

    angle_balanced = [float(row["balanced_accuracy"]) for row in angle_rows]
    suit_balanced = [
        float(population_summaries[f"suit_{suit}"]["balanced_accuracy"])
        for suit in ("m", "p", "s")
    ]
    metrics: dict[str, int | float] = {
        "standard_accuracy_0deg": float(standard_summary["accuracy"]),
        "standard_balanced_accuracy_0deg": float(standard_summary["balanced_accuracy"]),
        "standard_red_precision_0deg": float(standard_summary["red_precision"]),
        "standard_red_recall_0deg": float(standard_summary["red_recall"]),
        "standard_normal_specificity_0deg": float(standard_summary["normal_specificity"]),
        "standard_red_f1_0deg": float(standard_summary["red_f1"]),
        "standard_fp_count_0deg": int(standard_summary["fp"]),
        "standard_fn_count_0deg": int(standard_summary["fn"]),
        "standard_angle_mean_balanced_accuracy": float(np.mean(angle_balanced)),
        "standard_worst_angle_balanced_accuracy": float(min(angle_balanced)),
        "worst_suit_balanced_accuracy_0deg": float(min(suit_balanced)),
        "manual_accuracy_0deg": float(population_summaries["manual_val"]["accuracy"]),
        "warm_accuracy_0deg": float(warm_summary["accuracy"]),
        "warm_error_count_0deg": int(warm_summary["fp"]) + int(warm_summary["fn"]),
    }

    work_dir = context.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    population_path = work_dir / "red-five-population-summary.csv"
    angle_path = work_dir / "red-five-angle-summary.csv"
    confusion_path = work_dir / "red-five-confusion.plotly.json"
    errors_path = work_dir / "red-five-errors.jsonl"
    report_path = work_dir / "red-five-diagnostic-report.json"

    summary_fields = (
        "population", "count", "tp", "tn", "fp", "fn", "accuracy",
        "balanced_accuracy", "red_precision", "red_recall",
        "normal_specificity", "red_f1",
    )
    _write_csv(population_path, summary_fields, population_rows)
    _write_csv(
        angle_path,
        ("angle_index", "angle_deg", "count", "tp", "tn", "fp", "fn", "accuracy",
         "balanced_accuracy", "red_precision", "red_recall", "normal_specificity", "red_f1"),
        angle_rows,
    )
    confusion = {
        "data": [{
            "type": "heatmap",
            "x": ["normal-five", "red-five"],
            "y": ["normal-five", "red-five"],
            "z": [
                [int(standard_summary["tn"]), int(standard_summary["fp"])],
                [int(standard_summary["fn"]), int(standard_summary["tp"])],
            ],
            "hovertemplate": "true=%{y}<br>predicted=%{x}<br>count=%{z}<extra></extra>",
        }],
        "layout": {
            "title": "Red-five standard evaluation confusion counts at 0 degrees",
            "xaxis": {"title": "predicted"},
            "yaxis": {"title": "true", "autorange": "reversed"},
        },
    }
    confusion_path.write_text(json.dumps(confusion, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    with errors_path.open("w", encoding="utf-8") as handle:
        for row in error_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema": "mjtensu.recognition/red-five-binary-diagnostic-report/v1",
        "normalization": {"mean": mean, "std": std, "source": "weighted train split"},
        "standard_splits": list(STANDARD_SPLITS),
        "standard_sample_count": len(standard_rows),
        "warm_holdout_sample_count": len(warm_rows),
        "eval_angles": angles,
        "zero_angle_index": zero_index,
        "metrics": metrics,
        "population_summary": population_rows,
        "angle_summary": angle_rows,
        "error_count": len(error_rows),
        "notes": {
            "standard_scope": "Uses only jp_val, jp_test, and manual_val from the compact RGB64 corpus; train rows are excluded.",
            "label_audit": "The six known historical source-label defects documented in PRODUCT-INV-RECOGNITION-006 are absent from this compact evaluation population.",
            "warm_holdout": "The 24 real warm-light crops remain external to training and checkpoint selection.",
            "telemetry": "Per-angle balanced accuracy and red recall are emitted at zero-based authored eval_angles index; terminal metrics are projected by MLDB.",
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return EvaluationCandidate(
        metrics=metrics,
        artifacts={
            "population_summary_table": population_path,
            "angle_summary_table": angle_path,
            "confusion_matrix": confusion_path,
            "error_details": errors_path,
            "diagnostic_report": report_path,
        },
    )
