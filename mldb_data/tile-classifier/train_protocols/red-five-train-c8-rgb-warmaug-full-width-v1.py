from __future__ import annotations

import math
import random
import sqlite3
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..lib.c8_rgb_full_width_classifier_v1 import (
    build_c8_rgb_full_width_classifier,
)


SUPPORTED_ARCHITECTURES = {
    "tile-classifier/red-five-c8-rgb-full-fields6-12-24-48-v1": (6, 12, 24, 48),
    "tile-classifier/red-five-c8-rgb-full-fields8-16-32-64-v1": (8, 16, 32, 64),
}
IMAGE_SIZE = 64


@dataclass
class _Split:
    images_u8: torch.Tensor
    labels: torch.Tensor
    train_repeat: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.labels.shape[0])

    @property
    def effective_count(self) -> int:
        return int(self.train_repeat.sum().item())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_cuda(tf32: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def _load_split(database, split_name: str) -> _Split:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT image_rgb_u8, is_red, train_repeat FROM sample WHERE split=? ORDER BY sample_id",
            (split_name,),
        ).fetchall()
    if not rows:
        raise ValueError(f"Corpus split is empty: {split_name}")
    expected = IMAGE_SIZE * IMAGE_SIZE * 3
    images = np.empty((len(rows), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    labels = np.empty((len(rows),), dtype=np.int64)
    repeats = np.empty((len(rows),), dtype=np.int64)
    for index, (payload, target, repeat) in enumerate(rows):
        raw = bytes(payload)
        if len(raw) != expected:
            raise ValueError(f"{split_name} sample has {len(raw)} RGB bytes, expected {expected}")
        target = int(target)
        repeat = int(repeat)
        if target not in (0, 1):
            raise ValueError(f"is_red must be 0 or 1, got {target}")
        if repeat < 1:
            raise ValueError("train_repeat must be positive")
        images[index] = np.frombuffer(raw, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
        labels[index] = target
        repeats[index] = repeat
    return _Split(torch.from_numpy(images), torch.from_numpy(labels), torch.from_numpy(repeats))


def _normalization(split: _Split) -> tuple[tuple[float, ...], tuple[float, ...]]:
    images = split.images_u8.numpy().astype(np.float64) * (1.0 / 255.0)
    weights = split.train_repeat.numpy().astype(np.float64)
    per_sample_sum = images.sum(axis=(1, 2))
    per_sample_sq = np.square(images).sum(axis=(1, 2))
    pixels = IMAGE_SIZE * IMAGE_SIZE
    weighted_pixels = float(weights.sum() * pixels)
    mean = (per_sample_sum * weights[:, None]).sum(axis=0) / weighted_pixels
    sq_mean = (per_sample_sq * weights[:, None]).sum(axis=0) / weighted_pixels
    std = np.sqrt(np.maximum(sq_mean - np.square(mean), 0.0))
    std = np.maximum(std, 1.0 / 255.0)
    return tuple(float(v) for v in mean), tuple(float(v) for v in std)


def _resolve_cache_device(requested: str, *, image_bytes: int, fraction: float) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("cache_device must be one of auto/cpu/cuda")
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
    free_bytes, _ = torch.cuda.mem_get_info()
    return "cuda" if image_bytes <= int(free_bytes * fraction) else "cpu"


def _cache(split: _Split, *, device: torch.device, cache_device: str) -> _Split:
    if cache_device == "cuda":
        split.images_u8 = split.images_u8.to(device=device)
        split.labels = split.labels.to(device=device)
        split.train_repeat = split.train_repeat.to(device=device)
    elif device.type == "cuda":
        split.images_u8 = split.images_u8.pin_memory()
        split.labels = split.labels.pin_memory()
        split.train_repeat = split.train_repeat.pin_memory()
    return split


def _fetch(split: _Split, indices: np.ndarray, *, device: torch.device):
    index_cpu = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    if split.images_u8.device.type == "cuda":
        idx = index_cpu.to(device=device)
        return split.images_u8.index_select(0, idx), split.labels.index_select(0, idx)
    return (
        split.images_u8.index_select(0, index_cpu).to(device=device, non_blocking=True),
        split.labels.index_select(0, index_cpu).to(device=device, non_blocking=True),
    )


def _rotate(images: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    radians = angles_deg.float() * (math.pi / 180.0)
    cosine, sine = torch.cos(radians), torch.sin(radians)
    theta = torch.zeros((images.shape[0], 2, 3), device=images.device, dtype=torch.float32)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _normalize(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = images.new_tensor(mean).view(1, 3, 1, 1)
    std_t = images.new_tensor(std).view(1, 3, 1, 1)
    return images.sub(mean_t).div(std_t)


def _warm_augment(
    images_u8: torch.Tensor,
    *,
    probability: float,
    strength_min: float,
    strength_max: float,
    red_gain_max: float,
    green_gain_max: float,
    blue_gain_min: float,
    exposure_min: float,
    exposure_max: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    rgb = images_u8.float().mul_(1.0 / 255.0)
    count = int(rgb.shape[0])
    if count == 0 or probability <= 0.0:
        return rgb, 0
    selected = torch.rand((count,), device=rgb.device, generator=generator) < probability
    selected_count = int(selected.sum().item())
    if selected_count == 0:
        return rgb, 0
    strength = strength_min + torch.rand((count,), device=rgb.device, generator=generator) * (strength_max - strength_min)
    strength = torch.where(selected, strength, torch.zeros_like(strength))
    gains = torch.stack(
        (
            1.0 + strength * (red_gain_max - 1.0),
            1.0 + strength * (green_gain_max - 1.0),
            1.0 - strength * (1.0 - blue_gain_min),
        ),
        dim=1,
    ).view(count, 1, 1, 3)
    log_min, log_max = math.log(exposure_min), math.log(exposure_max)
    exposure = torch.exp(log_min + torch.rand((count,), device=rgb.device, generator=generator) * (log_max - log_min))
    exposure = torch.where(selected, exposure, torch.ones_like(exposure))
    return rgb.mul(gains).mul(exposure.view(count, 1, 1, 1)).clamp_(0.0, 1.0), selected_count


def _binary_summary(targets: torch.Tensor, predictions: torch.Tensor) -> dict[str, float]:
    targets = targets.long()
    predictions = predictions.long()
    tp = int(((targets == 1) & (predictions == 1)).sum().item())
    tn = int(((targets == 0) & (predictions == 0)).sum().item())
    fp = int(((targets == 0) & (predictions == 1)).sum().item())
    fn = int(((targets == 1) & (predictions == 0)).sum().item())
    red = tp + fn
    normal = tn + fp
    total = red + normal
    recall = tp / red if red else 0.0
    specificity = tn / normal if normal else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity) if red and normal else ((tp + tn) / total if total else 0.0),
        "red_recall": recall,
        "normal_specificity": specificity,
    }


def _evaluate(model, split: _Split, *, angle: float, batch_size: int, device: torch.device, mean, std, amp: bool):
    model.eval()
    target_chunks, prediction_chunks = [], []
    with torch.inference_mode():
        for start in range(0, split.count, batch_size):
            idx = np.arange(start, min(split.count, start + batch_size), dtype=np.int64)
            rgb, targets = _fetch(split, idx, device=device)
            images = rgb.permute(0, 3, 1, 2).float().mul_(1.0 / 255.0)
            if abs(angle) > 1.0e-12:
                images = _rotate(images, torch.full((images.shape[0],), angle, device=device))
            images = _normalize(images, mean, std)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
            target_chunks.append(targets.detach().cpu())
            prediction_chunks.append(logits.argmax(dim=1).detach().cpu())
    return _binary_summary(torch.cat(target_chunks), torch.cat(prediction_chunks))


def _train_epoch(model, split: _Split, *, optimizer, scaler, batch_size, device, mean, std, rotation_deg, amp, epoch, seed, warm):
    model.train()
    base_indices = np.arange(split.count, dtype=np.int64)
    repeats = split.train_repeat.detach().cpu().numpy().astype(np.int64)
    indices = np.repeat(base_indices, repeats)
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    rng.shuffle(indices)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + epoch * 7_919 + 313)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    warm_count = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        rgb_u8, targets = _fetch(split, batch_indices, device=device)
        rgb, augmented = _warm_augment(rgb_u8, generator=generator, **warm)
        images = rgb.permute(0, 3, 1, 2)
        if rotation_deg > 0.0:
            angles = torch.rand((images.shape[0],), device=device, generator=generator) * (2.0 * rotation_deg) - rotation_deg
            images = _rotate(images, angles)
        images = _normalize(images, mean, std)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count = int(targets.shape[0])
        total_loss += float(loss.detach().item()) * count
        total_correct += int((logits.detach().argmax(1) == targets).sum().item())
        total_count += count
        warm_count += augmented
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "warm_augmented_fraction": warm_count / max(total_count, 1),
    }


def _validate_parameters(p) -> None:
    if int(p["epochs"]) < 1: raise ValueError("epochs must be positive")
    if int(p["batch_size"]) < 2 or int(p["eval_batch_size"]) < 2: raise ValueError("batch sizes must be at least 2")
    if float(p["learning_rate"]) <= 0: raise ValueError("learning_rate must be positive")
    if float(p["weight_decay"]) < 0: raise ValueError("weight_decay must be non-negative")
    if not 0 <= float(p["rotation_augment_deg"]) <= 45: raise ValueError("rotation_augment_deg must be in [0,45]")
    angles = tuple(float(v) for v in p["eval_angles"])
    if not angles or not any(abs(v) < 1e-12 for v in angles): raise ValueError("eval_angles must include 0")
    if int(p["angle_eval_every"]) < 1: raise ValueError("angle_eval_every must be positive")
    if not 0 <= float(p["warm_augment_prob"]) <= 1: raise ValueError("warm_augment_prob must be in [0,1]")
    if not 0 <= float(p["warm_strength_min"]) <= float(p["warm_strength_max"]) <= 1: raise ValueError("invalid warm strength range")
    if float(p["warm_red_gain_max"]) < 1: raise ValueError("warm_red_gain_max must be >=1")
    if float(p["warm_green_gain_max"]) <= 0: raise ValueError("warm_green_gain_max must be positive")
    if not 0 < float(p["warm_blue_gain_min"]) <= 1: raise ValueError("warm_blue_gain_min must be in (0,1]")
    if not 0 < float(p["warm_exposure_min"]) <= float(p["warm_exposure_max"]): raise ValueError("invalid warm exposure range")
    _resolve_cache_device(str(p["cache_device"]), image_bytes=0, fraction=float(p["cache_vram_fraction"]))


def train(context):
    p = context.parameters
    _validate_parameters(p)
    architecture_id = context.architecture.get("id")
    fields = SUPPORTED_ARCHITECTURES.get(architecture_id)
    if fields is None:
        raise ValueError(f"unsupported red-five full-convolution width architecture: {architecture_id}")
    observed_fields = tuple(int(v) for v in getattr(context.model, "field_counts", fields))
    if observed_fields != fields:
        raise ValueError(f"architecture field-count mismatch: expected {fields}, got {observed_fields}")
    seed = int(context.seed)
    _seed_everything(seed)
    _configure_cuda(bool(p["tf32"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    database = context.corpus.root / "dataset.sqlite"
    train_split = _load_split(database, "train")
    jp_val = _load_split(database, "jp_val")
    mean, std = _normalization(train_split)
    total_bytes = int(train_split.images_u8.numel() + jp_val.images_u8.numel())
    cache_device = _resolve_cache_device(str(p["cache_device"]), image_bytes=total_bytes, fraction=float(p["cache_vram_fraction"]))
    _cache(train_split, device=device, cache_device=cache_device)
    _cache(jp_val, device=device, cache_device=cache_device)

    model = build_c8_rgb_full_width_classifier(fields=fields).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(p["learning_rate"]), weight_decay=float(p["weight_decay"]))
    epochs = int(p["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(p["learning_rate"]) * 0.05)
    amp = bool(p["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    angles = tuple(float(v) for v in p["eval_angles"])
    every = int(p["angle_eval_every"])
    warm = {
        "probability": float(p["warm_augment_prob"]),
        "strength_min": float(p["warm_strength_min"]),
        "strength_max": float(p["warm_strength_max"]),
        "red_gain_max": float(p["warm_red_gain_max"]),
        "green_gain_max": float(p["warm_green_gain_max"]),
        "blue_gain_min": float(p["warm_blue_gain_min"]),
        "exposure_min": float(p["warm_exposure_min"]),
        "exposure_max": float(p["warm_exposure_max"]),
    }
    best_score = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_metrics = _train_epoch(
            model, train_split, optimizer=optimizer, scaler=scaler,
            batch_size=int(p["batch_size"]), device=device, mean=mean, std=std,
            rotation_deg=float(p["rotation_augment_deg"]), amp=amp,
            epoch=epoch, seed=seed, warm=warm,
        )
        full = epoch == 1 or epoch == epochs or epoch % every == 0
        epoch_angles = angles if full else (0.0,)
        validation = {
            angle: _evaluate(model, jp_val, angle=angle, batch_size=int(p["eval_batch_size"]), device=device, mean=mean, std=std, amp=amp)
            for angle in epoch_angles
        }
        scheduler.step()
        context.telemetry.report_scalar(group="optimization", series="cross_entropy_loss", value=float(train_metrics["loss"]), step=epoch)
        context.telemetry.report_scalar(group="optimization", series="warm_augmented_fraction", value=float(train_metrics["warm_augmented_fraction"]), step=epoch)
        context.telemetry.report_scalar(group="validation", series="jp_balanced_accuracy_0deg", value=float(validation[0.0]["balanced_accuracy"]), step=epoch)
        if full:
            score = float(np.mean([validation[angle]["balanced_accuracy"] for angle in angles]))
            context.telemetry.report_scalar(group="validation", series="checkpoint_jp_angle_mean_balanced_accuracy", value=score, step=epoch)
            if score > best_score:
                best_score = score
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint candidate")
    model.load_state_dict(best_state, strict=True)
    return model
