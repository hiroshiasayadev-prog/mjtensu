from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path

import torch
import torch.nn.functional as F

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate


def _load_split(database: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT image_gray_u8, class_index FROM sample WHERE split = ? ORDER BY sample_id",
            (split,),
        ).fetchall()
    if not rows:
        raise ValueError(f"Corpus split is empty: {split}")
    images = torch.stack([
        torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone().reshape(1, 64, 64)
        for payload, _class_index in rows
    ])
    labels = torch.tensor([int(class_index) for _payload, class_index in rows], dtype=torch.long)
    return images, labels


def _normalization(images_u8: torch.Tensor) -> tuple[float, float]:
    values = images_u8.float().mul(1.0 / 255.0)
    mean = float(values.mean())
    std = max(float(values.std(unbiased=False)), 1.0 / 255.0)
    return mean, std


def _resolve_cache_device(requested: str, *, image_bytes: int, fraction: float) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("cache_device must be one of: auto, cpu, cuda")
    if not 0.0 < fraction < 0.8:
        raise ValueError("cache_vram_fraction must be between 0 and 0.8")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cache_device=cuda requires CUDA")
        return "cuda"
    if not torch.cuda.is_available():
        return "cpu"
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return "cuda" if image_bytes <= int(free_bytes * fraction) else "cpu"


def _cache_pair(images: torch.Tensor, labels: torch.Tensor, *, device, cache_device):
    if cache_device == "cuda":
        return images.to(device), labels.to(device)
    if device.type == "cuda":
        return images.pin_memory(), labels.pin_memory()
    return images, labels


def _slice_cached(images, labels, start: int, end: int, *, device):
    batch_images = images[start:end]
    batch_labels = labels[start:end]
    if images.device.type == "cuda":
        return batch_images, batch_labels
    return (
        batch_images.to(device, non_blocking=True),
        batch_labels.to(device, non_blocking=True),
    )


def _rotate(images: torch.Tensor, degrees: float) -> torch.Tensor:
    if abs(degrees) < 1.0e-12:
        return images
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    theta = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0]],
        dtype=images.dtype,
        device=images.device,
    ).unsqueeze(0).repeat(images.shape[0], 1, 1)
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _accuracy(
    model,
    images,
    labels,
    *,
    angle: float,
    mean: float,
    std: float,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, images.shape[0], batch_size):
            batch_u8, target = _slice_cached(
                images,
                labels,
                start,
                start + batch_size,
                device=device,
            )
            batch = batch_u8.float().mul(1.0 / 255.0)
            batch = _rotate(batch, angle)
            batch = batch.sub(mean).div(std)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(batch)
            correct += int((logits.argmax(dim=1) == target).sum().item())
            total += int(target.numel())
    return correct / max(1, total)


def _write_angle_artifacts(work_dir: Path, angles, manual, jp) -> tuple[Path, Path]:
    table_path = work_dir / "angle-robustness.csv"
    plot_path = work_dir / "angle-robustness.plotly.json"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("angle_index", "angle_deg", "manual_accuracy", "jp_accuracy"),
        )
        writer.writeheader()
        for index, angle in enumerate(angles):
            writer.writerow(
                {
                    "angle_index": index,
                    "angle_deg": angle,
                    "manual_accuracy": manual[angle],
                    "jp_accuracy": jp[angle],
                }
            )
    figure = {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "manual_val",
                "x": list(angles),
                "y": [manual[angle] for angle in angles],
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "jp_val",
                "x": list(angles),
                "y": [jp[angle] for angle in angles],
            },
        ],
        "layout": {
            "title": "Classifier accuracy by rotation angle",
            "xaxis": {"title": "angle (deg)"},
            "yaxis": {"title": "accuracy", "range": [0, 1]},
        },
    }
    plot_path.write_text(json.dumps(figure, ensure_ascii=False) + "\n", encoding="utf-8")
    return table_path, plot_path


def evaluate(context):
    parameters = context.parameters
    angles = tuple(float(value) for value in parameters["eval_angles"])
    if not angles or not any(abs(angle) < 1.0e-9 for angle in angles):
        raise ValueError("eval_angles must include 0 degrees")
    batch_size = int(parameters["batch_size"])
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(parameters["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(parameters["tf32"])

    train_images, _train_labels = _load_split(context.corpus.root / "dataset.sqlite", "train")
    manual_images, manual_labels = _load_split(context.corpus.root / "dataset.sqlite", "manual_val")
    jp_images, jp_labels = _load_split(context.corpus.root / "dataset.sqlite", "jp_val")
    mean, std = _normalization(train_images)
    image_bytes = int(train_images.numel() + manual_images.numel() + jp_images.numel())
    cache_device = _resolve_cache_device(
        str(parameters["cache_device"]),
        image_bytes=image_bytes,
        fraction=float(parameters["cache_vram_fraction"]),
    )
    manual_images, manual_labels = _cache_pair(
        manual_images,
        manual_labels,
        device=device,
        cache_device=cache_device,
    )
    jp_images, jp_labels = _cache_pair(
        jp_images,
        jp_labels,
        device=device,
        cache_device=cache_device,
    )
    model = context.model.module.to(device)
    amp_enabled = bool(parameters["amp"]) and device.type == "cuda"

    manual = {
        angle: _accuracy(
            model,
            manual_images,
            manual_labels,
            angle=angle,
            mean=mean,
            std=std,
            device=device,
            batch_size=batch_size,
            amp_enabled=amp_enabled,
        )
        for angle in angles
    }
    jp = {
        angle: _accuracy(
            model,
            jp_images,
            jp_labels,
            angle=angle,
            mean=mean,
            std=std,
            device=device,
            batch_size=batch_size,
            amp_enabled=amp_enabled,
        )
        for angle in angles
    }
    for angle_index, angle in enumerate(angles):
        context.telemetry.report_scalar(
            group="angle-sweep",
            series="manual_accuracy",
            value=float(manual[angle]),
            step=angle_index,
        )
        context.telemetry.report_scalar(
            group="angle-sweep",
            series="jp_accuracy",
            value=float(jp[angle]),
            step=angle_index,
        )

    table_path, plot_path = _write_angle_artifacts(context.work_dir, angles, manual, jp)
    zero_angle = min(angles, key=lambda value: abs(value))
    metrics = {
        "manual_accuracy_0deg": float(manual[zero_angle]),
        "jp_accuracy_0deg": float(jp[zero_angle]),
        "manual_angle_mean": float(sum(manual.values()) / len(manual)),
        "jp_angle_mean": float(sum(jp.values()) / len(jp)),
    }
    return EvaluationCandidate(
        metrics=metrics,
        artifacts={
            "angle_table": table_path,
            "angle_plot": plot_path,
        },
    )
