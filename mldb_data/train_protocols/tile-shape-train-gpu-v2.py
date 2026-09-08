from __future__ import annotations

import math
import random
import sqlite3
from pathlib import Path

import torch
import torch.nn.functional as F

from mldb.src.runtime.executable_loader import load_architecture_build


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _normalize(images: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    values = images.float().mul_(1.0 / 255.0)
    mean = float(values.mean())
    std = max(float(values.std(unbiased=False)), 1.0e-6)
    return values, mean, std


def _rotate_batch(images: torch.Tensor, max_degrees: float) -> torch.Tensor:
    if max_degrees <= 0.0:
        return images
    batch = images.shape[0]
    angles = (torch.rand(batch, device=images.device) * 2.0 - 1.0) * math.radians(max_degrees)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    theta = torch.zeros(batch, 2, 3, device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def train(context):
    parameters = context.parameters
    seed = int(context.seed)
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(parameters["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(parameters["tf32"])

    uint8_images, labels = _load_split(context.corpus.artifact_path, "train")
    images, mean, std = _normalize(uint8_images)
    model = load_architecture_build(context.architecture)().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    epochs = int(parameters["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_enabled = bool(parameters["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    batch_size = int(parameters["batch_size"])
    max_rotation = float(parameters["rotation_augment_deg"])

    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(images.shape[0], generator=generator)
        model.train()
        for start in range(0, order.numel(), batch_size):
            indices = order[start : start + batch_size]
            batch = images[indices].to(device, non_blocking=True)
            target = labels[indices].to(device, non_blocking=True)
            batch = (batch - mean) / std
            batch = _rotate_batch(batch, max_rotation)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(batch)
                loss = F.cross_entropy(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
    return model
