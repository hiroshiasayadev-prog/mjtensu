from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import torch
import torch.nn.functional as F

from mldb.src.evaluation.interface import EvaluationResult
from mldb.src.model.loading import load_model


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


def _train_normalization(database: Path) -> tuple[float, float]:
    images, _labels = _load_split(database, "train")
    values = images.float().mul_(1.0 / 255.0)
    return float(values.mean()), max(float(values.std(unbiased=False)), 1.0e-6)


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
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _accuracy(model, images, labels, *, angle, mean, std, device, batch_size, amp_enabled):
    correct = 0
    total = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, images.shape[0], batch_size):
            batch = images[start:start + batch_size].float().mul(1.0 / 255.0).to(device)
            target = labels[start:start + batch_size].to(device)
            batch = (batch - mean) / std
            batch = _rotate(batch, angle)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(batch)
            correct += int((logits.argmax(dim=1) == target).sum().item())
            total += int(target.numel())
    return correct / total


def evaluate(context):
    parameters = context.parameters
    angles = tuple(float(value) for value in parameters["eval_angles"])
    if 0.0 not in angles:
        raise ValueError("eval_angles must include 0 degrees because the declared result includes 0deg accuracy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(parameters["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(parameters["tf32"])
    model = load_model(context.model).to(device)
    mean, std = _train_normalization(context.corpus.artifact_path)
    manual_images, manual_labels = _load_split(context.corpus.artifact_path, "manual_val")
    jp_images, jp_labels = _load_split(context.corpus.artifact_path, "jp_val")
    batch_size = int(parameters["batch_size"])
    amp_enabled = bool(parameters["amp"]) and device.type == "cuda"

    manual = {
        angle: _accuracy(model, manual_images, manual_labels, angle=angle, mean=mean, std=std,
                         device=device, batch_size=batch_size, amp_enabled=amp_enabled)
        for angle in angles
    }
    jp = {
        angle: _accuracy(model, jp_images, jp_labels, angle=angle, mean=mean, std=std,
                         device=device, batch_size=batch_size, amp_enabled=amp_enabled)
        for angle in angles
    }
    metrics = {
        "manual_accuracy_0deg": float(manual[0.0]),
        "jp_accuracy_0deg": float(jp[0.0]),
        "manual_angle_mean": float(sum(manual.values()) / len(manual)),
        "jp_angle_mean": float(sum(jp.values()) / len(jp)),
    }
    return EvaluationResult(metrics=metrics, artifacts={}, unavailable_outputs=())
