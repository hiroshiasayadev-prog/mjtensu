from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate


EXPECTED_LABELS = (
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "east", "south", "west", "north", "white", "green", "red", "invalid",
)
IMAGE_SIZE = 64


def _load_metadata_and_normalization(database: Path) -> tuple[list[str], float, float]:
    with sqlite3.connect(database) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM experiment_metadata"))
        labels = [str(value) for value in json.loads(metadata["base_labels"])]
        pixel_sum = 0.0
        pixel_sq_sum = 0.0
        pixel_count = 0
        for (payload,) in connection.execute("SELECT image_gray_u8 FROM sample WHERE split='train'"):
            values = np.frombuffer(payload, dtype=np.uint8).astype(np.float64) / 255.0
            pixel_sum += float(values.sum())
            pixel_sq_sum += float(np.square(values).sum())
            pixel_count += int(values.size)
    if labels != list(EXPECTED_LABELS):
        raise ValueError(f"unexpected classifier labels: {labels}")
    if pixel_count == 0:
        raise ValueError("Corpus train split is empty")
    mean = pixel_sum / pixel_count
    variance = max(pixel_sq_sum / pixel_count - mean * mean, 0.0)
    return labels, mean, max(math.sqrt(variance), 1.0 / 255.0)


def _load_rows(database: Path, split: str, labels: Sequence[str]) -> list[dict[str, Any]]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT sample_id, base_label, class_index, image_gray_u8, source,
                   capture_id, layout_id, region, source_image_path
            FROM sample
            WHERE split=?
            ORDER BY class_index, sample_id
            """,
            (split,),
        ).fetchall()
    result: list[dict[str, Any]] = []
    counts = {label: 0 for label in labels}
    for row in rows:
        class_index = int(row[2])
        label = str(row[1])
        if class_index < 0 or class_index >= len(labels) or labels[class_index] != label:
            raise ValueError(f"split label/index mismatch: label={label!r} index={class_index}")
        counts[label] += 1
        result.append({
            "sample_id": str(row[0]),
            "base_label": label,
            "class_index": class_index,
            "image": np.frombuffer(row[3], dtype=np.uint8).copy().reshape(IMAGE_SIZE, IMAGE_SIZE),
            "source": str(row[4]),
            "capture_id": None if row[5] is None else str(row[5]),
            "layout_id": None if row[6] is None else str(row[6]),
            "region": None if row[7] is None else str(row[7]),
            "source_image_path": str(row[8]),
        })
    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"split must cover every classifier class; missing={missing}")
    return result


def _infer(
    model,
    images_01: torch.Tensor,
    *,
    mean: float,
    std: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, images_01.shape[0], batch_size):
            batch = images_01[start : start + batch_size].to(device)
            batch = (batch - mean) / std
            outputs.append(model(batch).detach().float().cpu())
    return torch.cat(outputs, dim=0)


def _true_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    true_values = logits.gather(1, targets[:, None]).squeeze(1)
    masked = logits.clone()
    masked[torch.arange(logits.shape[0]), targets] = -torch.inf
    return true_values - masked.max(dim=1).values


def _occlusion_result(
    model,
    image_01: torch.Tensor,
    target_index: int,
    baseline_margin: float,
    *,
    mean: float,
    std: float,
    patch: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    variants: list[torch.Tensor] = []
    cells: list[tuple[int, int]] = []
    for y in range(0, IMAGE_SIZE, patch):
        for x in range(0, IMAGE_SIZE, patch):
            variant = image_01.clone()
            variant[:, y : y + patch, x : x + patch] = mean
            variants.append(variant)
            cells.append((y // patch, x // patch))
    logits = _infer(
        model,
        torch.stack(variants),
        mean=mean,
        std=std,
        batch_size=batch_size,
        device=device,
    )
    targets = torch.full((len(variants),), target_index, dtype=torch.long)
    masked_margins = _true_margin(logits, targets).numpy()
    drops = baseline_margin - masked_margins
    positive = np.maximum(drops, 0.0)
    total = float(positive.sum())
    square_sum = float(np.square(positive).sum())
    grid_size = IMAGE_SIZE // patch
    heat = np.zeros((grid_size, grid_size), dtype=np.float32)
    for (gy, gx), drop in zip(cells, drops, strict=True):
        heat[gy, gx] = float(drop)
    return heat, {
        "top_patch_share": 0.0 if total <= 1.0e-12 else float(positive.max() / total),
        "effective_patch_count": 0.0 if square_sum <= 1.0e-12 else float(total * total / square_sum),
        "peak_drop": float(max(float(positive.max(initial=0.0)), 0.0)),
        "positive_drop_mean": float(positive.mean()),
    }


def _overlay(image_u8: np.ndarray, heat_grid: np.ndarray, patch: int) -> Image.Image:
    heat = np.repeat(np.repeat(heat_grid, patch, axis=0), patch, axis=1)
    base = np.repeat(image_u8[:, :, None], 3, axis=2).astype(np.float32)
    scale = max(float(np.max(np.abs(heat))), 1.0e-6)
    strength = np.clip(np.abs(heat) / scale, 0.0, 1.0)[:, :, None]
    positive = np.zeros_like(base)
    positive[:, :, 0] = 255.0
    negative = np.zeros_like(base)
    negative[:, :, 2] = 255.0
    color = np.where(heat[:, :, None] >= 0, positive, negative)
    mixed = base * (1.0 - 0.60 * strength) + color * (0.60 * strength)
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), mode="RGB")


def _contact_sheet(
    path: Path,
    class_rows: Sequence[dict[str, Any]],
    class_images: dict[str, np.ndarray],
    class_heats: dict[str, np.ndarray],
    patch: int,
) -> None:
    columns = 7
    rows_count = math.ceil(len(class_rows) / columns)
    cell_width, cell_height = 300, 190
    sheet = Image.new("RGB", (columns * cell_width, rows_count * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for position, summary in enumerate(class_rows):
        row_index, column = divmod(position, columns)
        x0, y0 = column * cell_width, row_index * cell_height
        label = str(summary["label"])
        source = Image.fromarray(class_images[label], mode="L").convert("RGB").resize((112, 112), Image.Resampling.NEAREST)
        overlay = _overlay(class_images[label], class_heats[label], patch).resize((112, 112), Image.Resampling.NEAREST)
        sheet.paste(source, (x0 + 8, y0 + 45))
        sheet.paste(overlay, (x0 + 132, y0 + 45))
        draw.text((x0 + 8, y0 + 8), f"{label}  n={summary['sample_count']}", fill="black")
        draw.text(
            (x0 + 8, y0 + 24),
            f"margin={summary['baseline_true_margin_mean']:.2f} peak={summary['occlusion_peak_drop_mean']:.2f}",
            fill="black",
        )
        draw.text((x0 + 132, y0 + 160), "red: hide hurts", fill="black")
    sheet.save(path)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(context):
    parameters = context.parameters
    split = str(parameters["split"])
    batch_size = int(parameters["batch_size"])
    patch = int(parameters["occlusion_patch"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if patch < 1 or IMAGE_SIZE % patch != 0:
        raise ValueError("occlusion_patch must divide 64 exactly")

    database = context.corpus.root / "dataset.sqlite"
    labels, mean, std = _load_metadata_and_normalization(database)
    rows = _load_rows(database, split, labels)
    images = torch.from_numpy(np.stack([row["image"] for row in rows])).float().unsqueeze(1).mul(1.0 / 255.0)
    targets = torch.tensor([row["class_index"] for row in rows], dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = context.model.module.to(device)
    model.eval()

    baseline_logits = _infer(model, images, mean=mean, std=std, batch_size=batch_size, device=device)
    baseline_margins = _true_margin(baseline_logits, targets)
    baseline_predictions = baseline_logits.argmax(dim=1)

    heatmaps: list[np.ndarray] = []
    stats_rows: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        heat, stats = _occlusion_result(
            model,
            images[index],
            int(row["class_index"]),
            float(baseline_margins[index]),
            mean=mean,
            std=std,
            patch=patch,
            batch_size=batch_size,
            device=device,
        )
        heatmaps.append(heat)
        stats_rows.append(stats)

    class_rows: list[dict[str, Any]] = []
    class_images: dict[str, np.ndarray] = {}
    class_heats: dict[str, np.ndarray] = {}
    patch_rows: list[dict[str, Any]] = []
    for class_index, label in enumerate(labels):
        indices = [index for index, row in enumerate(rows) if row["class_index"] == class_index]
        class_margin = baseline_margins[indices].numpy()
        class_accuracy = float((baseline_predictions[indices] == targets[indices]).float().mean())
        class_heat = np.mean(np.stack([heatmaps[index] for index in indices]), axis=0)
        class_heats[label] = class_heat
        class_images[label] = np.mean(np.stack([rows[index]["image"] for index in indices]), axis=0).round().astype(np.uint8)
        summary = {
            "class_index": class_index,
            "label": label,
            "sample_count": len(indices),
            "baseline_accuracy": class_accuracy,
            "baseline_true_margin_mean": float(np.mean(class_margin)),
            "occlusion_peak_drop_mean": float(np.mean([stats_rows[index]["peak_drop"] for index in indices])),
            "occlusion_top_patch_share_mean": float(np.mean([stats_rows[index]["top_patch_share"] for index in indices])),
            "occlusion_effective_patch_count_mean": float(np.mean([stats_rows[index]["effective_patch_count"] for index in indices])),
            "occlusion_positive_drop_mean": float(np.mean([stats_rows[index]["positive_drop_mean"] for index in indices])),
        }
        class_rows.append(summary)
        for gy in range(class_heat.shape[0]):
            for gx in range(class_heat.shape[1]):
                patch_rows.append({
                    "class_index": class_index,
                    "label": label,
                    "grid_y": gy,
                    "grid_x": gx,
                    "y0": gy * patch,
                    "x0": gx * patch,
                    "mean_true_margin_drop": float(class_heat[gy, gx]),
                })

    per_sample = []
    for index, row in enumerate(rows):
        per_sample.append({
            "sample_id": row["sample_id"],
            "true_label": row["base_label"],
            "prediction": labels[int(baseline_predictions[index])],
            "baseline_true_margin": float(baseline_margins[index]),
            **stats_rows[index],
        })

    aggregate = {
        "baseline_accuracy": float((baseline_predictions == targets).float().mean()),
        "baseline_true_margin_mean": float(baseline_margins.mean()),
        "occlusion_peak_drop_mean": float(np.mean([row["peak_drop"] for row in stats_rows])),
        "occlusion_top_patch_share_mean": float(np.mean([row["top_patch_share"] for row in stats_rows])),
        "occlusion_effective_patch_count_mean": float(np.mean([row["effective_patch_count"] for row in stats_rows])),
        "occlusion_positive_drop_mean": float(np.mean([row["positive_drop_mean"] for row in stats_rows])),
    }

    work_dir = context.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    class_summary_path = work_dir / "full-class-occlusion-summary.csv"
    patch_table_path = work_dir / "full-class-occlusion-patches.csv"
    contact_sheet_path = work_dir / "full-class-occlusion-contact-sheet.png"
    per_sample_path = work_dir / "full-class-occlusion-samples.jsonl"
    report_path = work_dir / "full-class-occlusion-report.json"

    _write_csv(
        class_summary_path,
        (
            "class_index", "label", "sample_count", "baseline_accuracy",
            "baseline_true_margin_mean", "occlusion_peak_drop_mean",
            "occlusion_top_patch_share_mean", "occlusion_effective_patch_count_mean",
            "occlusion_positive_drop_mean",
        ),
        class_rows,
    )
    _write_csv(
        patch_table_path,
        ("class_index", "label", "grid_y", "grid_x", "y0", "x0", "mean_true_margin_drop"),
        patch_rows,
    )
    _contact_sheet(contact_sheet_path, class_rows, class_images, class_heats, patch)
    with per_sample_path.open("w", encoding="utf-8") as handle:
        for row in per_sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema": "mjtensu.recognition/full-class-occlusion-report/v1",
        "split": split,
        "sample_count": len(rows),
        "labels": labels,
        "occlusion_patch": patch,
        "normalization": {"mean": mean, "std": std},
        "metrics": aggregate,
        "class_summary": class_rows,
        "notes": {
            "margin": "true-class logit minus the strongest competing-class logit",
            "occlusion": "Each patch is replaced with the training-set grayscale mean. Positive/red margin drop means hiding that region hurts evidence for the true class; negative/blue means hiding it helps.",
            "aggregation": "Heatmaps are averaged over every sample of each class in the requested split. The contact sheet uses the class-average crop and class-average margin-drop heatmap.",
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return EvaluationCandidate(
        metrics=aggregate,
        artifacts={
            "class_summary_table": class_summary_path,
            "class_patch_table": patch_table_path,
            "class_occlusion_contact_sheet": contact_sheet_path,
            "per_sample_details": per_sample_path,
            "report_json": report_path,
        },
    )
