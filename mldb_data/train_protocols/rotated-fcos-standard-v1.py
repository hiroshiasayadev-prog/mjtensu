from __future__ import annotations

import json
import math
import random
import sqlite3
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from mldb.src.runtime.executable_loader import load_architecture_build

STRIDES = (8, 16, 32)
DEFAULT_SIZE_RANGES = ((0.0, 64.0), (48.0, 128.0), (96.0, 1.0e8))
IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)

def make_points(
    outputs: Sequence[torch.Tensor],
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(outputs) != len(STRIDES):
        raise ValueError(f"Expected {len(STRIDES)} output levels, found {len(outputs)}")
    point_chunks: list[torch.Tensor] = []
    stride_chunks: list[torch.Tensor] = []
    level_chunks: list[torch.Tensor] = []
    for level, (output, stride) in enumerate(zip(outputs, STRIDES, strict=True)):
        height, width = output.shape[-2:]
        target_device = output.device if device is None else device
        ys = (torch.arange(height, device=target_device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(width, device=target_device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        point_chunks.append(points)
        stride_chunks.append(
            torch.full((points.shape[0],), float(stride), device=target_device)
        )
        level_chunks.append(
            torch.full((points.shape[0],), level, device=target_device, dtype=torch.long)
        )
    return (
        torch.cat(point_chunks, dim=0),
        torch.cat(stride_chunks, dim=0),
        torch.cat(level_chunks, dim=0),
    )

def flatten_outputs(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    chunks = [output.permute(0, 2, 3, 1).reshape(output.shape[0], -1, 8) for output in outputs]
    return torch.cat(chunks, dim=1)

def build_targets(
    boxes: torch.Tensor,
    points: torch.Tensor,
    strides: torch.Tensor,
    levels: torch.Tensor,
    *,
    center_radius: float = 1.5,
    size_ranges: Sequence[tuple[float, float]] = DEFAULT_SIZE_RANGES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    point_count = points.shape[0]
    device = points.device
    objectness = torch.zeros(point_count, device=device)
    centerness = torch.zeros(point_count, device=device)
    regression = torch.zeros(point_count, 6, device=device)
    if boxes.numel() == 0:
        return objectness, centerness, regression
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError(f"Expected [N,5] OBB tensor, found {tuple(boxes.shape)}")

    boxes = boxes.to(device=device, dtype=torch.float32)
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    widths = boxes[:, 2].clamp_min(1.0e-4)
    heights = boxes[:, 3].clamp_min(1.0e-4)
    angles = torch.deg2rad(boxes[:, 4])
    cosine = torch.cos(angles)
    sine = torch.sin(angles)

    dx = points[:, 0, None] - cx[None, :]
    dy = points[:, 1, None] - cy[None, :]
    local_x = dx * cosine[None, :] + dy * sine[None, :]
    local_y = -dx * sine[None, :] + dy * cosine[None, :]
    half_width = widths[None, :] / 2.0
    half_height = heights[None, :] / 2.0
    inside = (local_x.abs() <= half_width) & (local_y.abs() <= half_height)

    radius = center_radius * strides[:, None]
    center_inside = (
        (local_x.abs() <= torch.minimum(half_width, radius))
        & (local_y.abs() <= torch.minimum(half_height, radius))
    )
    max_size = torch.maximum(widths, heights)
    ranges = torch.tensor(size_ranges, device=device, dtype=torch.float32)
    range_low = ranges[levels, 0][:, None]
    range_high = ranges[levels, 1][:, None]
    level_match = (max_size[None, :] >= range_low) & (max_size[None, :] <= range_high)
    candidates = inside & center_inside & level_match

    areas = (widths * heights)[None, :].expand(point_count, -1).clone()
    areas[~candidates] = float("inf")
    assigned_area, assigned_index = areas.min(dim=1)
    positive = torch.isfinite(assigned_area)
    if not positive.any():
        return objectness, centerness, regression

    positive_indices = positive.nonzero(as_tuple=False).squeeze(1)
    gt_indices = assigned_index[positive]
    objectness[positive] = 1.0

    selected_local_x = local_x[positive_indices, gt_indices]
    selected_local_y = local_y[positive_indices, gt_indices]
    selected_width = widths[gt_indices]
    selected_height = heights[gt_indices]
    selected_stride = strides[positive]
    left = selected_width / 2.0 + selected_local_x
    right = selected_width / 2.0 - selected_local_x
    top = selected_height / 2.0 + selected_local_y
    bottom = selected_height / 2.0 - selected_local_y
    lr = torch.minimum(left, right) / torch.maximum(left, right).clamp_min(1.0e-6)
    tb = torch.minimum(top, bottom) / torch.maximum(top, bottom).clamp_min(1.0e-6)
    centerness[positive] = torch.sqrt((lr * tb).clamp_min(0.0))

    selected_cx = cx[gt_indices]
    selected_cy = cy[gt_indices]
    selected_angle = angles[gt_indices]
    regression[positive, 0] = (selected_cx - points[positive, 0]) / selected_stride
    regression[positive, 1] = (selected_cy - points[positive, 1]) / selected_stride
    regression[positive, 2] = torch.log(selected_width / selected_stride)
    regression[positive, 3] = torch.log(selected_height / selected_stride)
    regression[positive, 4] = torch.sin(2.0 * selected_angle)
    regression[positive, 5] = torch.cos(2.0 * selected_angle)
    return objectness, centerness, regression

def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    binary_cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * (1.0 - p_t).pow(gamma) * binary_cross_entropy

def compute_loss(
    outputs: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    *,
    center_radius: float = 1.5,
) -> dict[str, torch.Tensor]:
    predictions = flatten_outputs(outputs)
    points, strides, levels = make_points(outputs)
    batch_size, point_count, _channels = predictions.shape
    if len(targets) != batch_size:
        raise ValueError(f"Expected {batch_size} target tensors, found {len(targets)}")

    objectness_targets = torch.zeros(batch_size, point_count, device=predictions.device)
    centerness_targets = torch.zeros(batch_size, point_count, device=predictions.device)
    regression_targets = torch.zeros(batch_size, point_count, 6, device=predictions.device)
    for batch_index, boxes in enumerate(targets):
        obj, ctr, reg = build_targets(
            boxes,
            points,
            strides,
            levels,
            center_radius=center_radius,
        )
        objectness_targets[batch_index] = obj
        centerness_targets[batch_index] = ctr
        regression_targets[batch_index] = reg

    positive = objectness_targets > 0.5
    positive_count = positive.sum().clamp_min(1).to(predictions.dtype)
    objectness_loss = sigmoid_focal_loss(
        predictions[..., 0], objectness_targets
    ).sum() / positive_count

    if positive.any():
        centerness_loss = F.binary_cross_entropy_with_logits(
            predictions[..., 1][positive],
            centerness_targets[positive],
            reduction="sum",
        ) / positive_count
        weights = centerness_targets[positive].clamp_min(0.05)
        predicted_regression = predictions[..., 2:][positive]
        target_regression = regression_targets[positive]
        center_size = F.smooth_l1_loss(
            predicted_regression[:, :4],
            target_regression[:, :4],
            reduction="none",
            beta=0.25,
        ).mean(dim=1)
        center_size_loss = (center_size * weights).sum() / weights.sum().clamp_min(1.0)

        predicted_angle = F.normalize(predicted_regression[:, 4:6], dim=1, eps=1.0e-6)
        target_angle = F.normalize(target_regression[:, 4:6], dim=1, eps=1.0e-6)
        angle_distance = 1.0 - (predicted_angle * target_angle).sum(dim=1)
        angle_loss = (angle_distance * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        zero = predictions.sum() * 0.0
        centerness_loss = zero
        center_size_loss = zero
        angle_loss = zero

    total = objectness_loss + centerness_loss + 2.0 * center_size_loss + angle_loss
    return {
        "total": total,
        "objectness": objectness_loss,
        "centerness": centerness_loss,
        "center_size": center_size_loss,
        "angle": angle_loss,
        "positive_count": positive_count.detach(),
    }

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_training_split(database: Path) -> tuple[torch.Tensor, list[torch.Tensor]]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT image_rgb_u8, annotations_json FROM sample WHERE split = 'train' ORDER BY sample_id"
        ).fetchall()
    if not rows:
        raise ValueError("Corpus train split is empty")
    images: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for payload, annotations_raw in rows:
        image = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone().reshape(3, 320, 320)
        annotations = json.loads(annotations_raw)
        boxes = torch.tensor(
            [[float(value) for value in item["obb"]] for item in annotations],
            dtype=torch.float32,
        )
        if boxes.numel() == 0:
            boxes = torch.empty((0, 5), dtype=torch.float32)
        images.append(image)
        targets.append(boxes)
    return torch.stack(images), targets

def _lr_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(0.05, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(context):
    parameters = context.parameters
    seed = int(context.seed)
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(parameters["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(parameters["tf32"])
    images_u8, targets = _load_training_split(context.corpus.artifact_path)
    model = load_architecture_build(context.architecture)().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    epochs = int(parameters["epochs"])
    batch_size = int(parameters["batch_size"])
    steps_per_epoch = max(1, math.ceil(images_u8.shape[0] / batch_size))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, round(float(parameters["warmup_epochs"]) * steps_per_epoch))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_multiplier(
            step, total_steps=total_steps, warmup_steps=warmup_steps
        ),
    )
    amp_enabled = bool(parameters["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    center_radius = float(parameters["center_radius"])
    gradient_clip = float(parameters["gradient_clip"])
    mean = IMAGE_MEAN.to(device)
    std = IMAGE_STD.to(device)

    for epoch in range(epochs):
        order = torch.randperm(images_u8.shape[0], generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        for start in range(0, order.numel(), batch_size):
            indices = order[start : start + batch_size]
            batch = images_u8[indices].to(device, non_blocking=True).float().mul_(1.0 / 255.0)
            batch = (batch - mean) / std
            batch_targets = [targets[int(index)].to(device) for index in indices.tolist()]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(batch)
                losses = compute_loss(outputs, batch_targets, center_radius=center_radius)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
    return model
