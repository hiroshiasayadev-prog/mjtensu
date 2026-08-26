from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from red_five_classifier import (
        DEFAULT_C8_FIELDS,
        INPUT_CHANNELS,
        SUPPORTED_INPUT_MODES,
        build_model,
        describe_model,
        normalize_input_mode,
    )
except ModuleNotFoundError:  # package import path used by unit tests
    from tools.recognition.red_five_classifier import (
        DEFAULT_C8_FIELDS,
        INPUT_CHANNELS,
        SUPPORTED_INPUT_MODES,
        build_model,
        describe_model,
        normalize_input_mode,
    )


DEFAULT_SEED = 42
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 1024
DEFAULT_EVAL_BATCH_SIZE = 4096
DEFAULT_LEARNING_RATE = 1.0e-3
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_ROTATION_AUGMENT_DEG = 22.5
DEFAULT_EVAL_ANGLES = (0.0, 15.0, 30.0, 45.0)


@dataclass
class SplitCache:
    name: str
    images_u8: torch.Tensor
    labels: torch.Tensor
    train_repeat: np.ndarray
    suit: np.ndarray
    source: np.ndarray
    brightness: np.ndarray
    shadow: np.ndarray
    sample_ids: list[str]

    @property
    def count(self) -> int:
        return int(self.labels.shape[0])

    @property
    def effective_count(self) -> int:
        return int(self.train_repeat.sum())


@dataclass
class TrainingCache:
    splits: dict[str, SplitCache]
    image_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    cache_device: str
    image_bytes: int
    input_mode: str


@dataclass
class EvaluationResult:
    loss: float
    metrics: dict[str, Any]
    predictions: np.ndarray
    targets: np.ndarray


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Train the C8 binary red-five classifier from the RGB64 experiment DB. "
            "Input representation is selectable: RGB, Cr only, or Y+Cr."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-mode", choices=SUPPORTED_INPUT_MODES, required=True)
    parser.add_argument("--c8-fields", type=int, nargs="+", default=list(DEFAULT_C8_FIELDS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--rotation-augment-deg",
        type=float,
        default=DEFAULT_ROTATION_AUGMENT_DEG,
        help=(
            "Uniform residual rotation augmentation. With C8, ±22.5 degrees fills the "
            "gaps between the eight 45-degree group elements. Defaults to 22.5."
        ),
    )
    parser.add_argument(
        "--eval-angles",
        type=float,
        nargs="+",
        default=list(DEFAULT_EVAL_ANGLES),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cache-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--cache-vram-fraction", type=float, default=0.25)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--angle-eval-every", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script")

    repository_root = args.repository_root.resolve()
    database = args.database.resolve()
    output_dir = args.output_dir.resolve()
    input_mode = normalize_input_mode(args.input_mode)
    prepare_output_directory(output_dir, overwrite=bool(args.overwrite))

    seed_everything(int(args.seed))
    configure_cuda(tf32=bool(args.tf32))
    device = torch.device("cuda")

    cache = load_training_cache(
        database,
        input_mode=input_mode,
        device=device,
        cache_device=str(args.cache_device),
        cache_vram_fraction=float(args.cache_vram_fraction),
    )
    model = build_model(
        input_mode,
        c8_fields=tuple(args.c8_fields),
    ).to(device)
    model_description = describe_model(model, input_mode)

    config = {
        "database": str(database),
        "output_dir": str(output_dir),
        "input_mode": input_mode,
        "input_channels": INPUT_CHANNELS[input_mode],
        "color_transform": color_transform_description(input_mode),
        "model": asdict(model_description),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "rotation_augment_deg": float(args.rotation_augment_deg),
        "eval_angles": [float(value) for value in args.eval_angles],
        "angle_eval_every": int(args.angle_eval_every),
        "best_checkpoint_metric": "jp_val_angle_mean_balanced_accuracy",
        "manual_val_policy": "external_holdout_not_used_for_checkpoint_selection",
        "seed": int(args.seed),
        "amp": bool(args.amp),
        "tf32": bool(args.tf32),
        "cache_device": cache.cache_device,
        "cache_image_bytes": cache.image_bytes,
        "image_size": cache.image_size,
        "normalization": {
            "mean": list(cache.mean),
            "std": list(cache.std),
            "weighted_by_train_repeat": True,
        },
        "train_unique_samples": cache.splits["train"].count,
        "train_effective_samples": cache.splits["train"].effective_count,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    atomic_write_json(output_dir / "config.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.epochs),
        eta_min=float(args.learning_rate) * 0.05,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    history_path = output_dir / "history.jsonl"
    best_score = -1.0
    best_epoch = 0
    train_split = cache.splits["train"]
    full_eval_angles = tuple(float(value) for value in args.eval_angles)

    for epoch in range(1, int(args.epochs) + 1):
        torch.cuda.reset_peak_memory_stats()
        train_metrics = train_one_epoch(
            model,
            train_split,
            optimizer=optimizer,
            scaler=scaler,
            batch_size=int(args.batch_size),
            device=device,
            cache=cache,
            rotation_augment_deg=float(args.rotation_augment_deg),
            amp=bool(args.amp),
            epoch=epoch,
            seed=int(args.seed),
        )

        if (
            epoch == 1
            or epoch == int(args.epochs)
            or epoch % int(args.angle_eval_every) == 0
        ):
            epoch_eval_angles = full_eval_angles
        else:
            epoch_eval_angles = (0.0,)

        validation = evaluate_selected_splits(
            model,
            cache,
            split_names=("jp_val", "manual_val"),
            device=device,
            batch_size=int(args.eval_batch_size),
            angles=epoch_eval_angles,
            amp=bool(args.amp),
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        primary = robust_jp_validation_score(
            validation,
            required_angles=full_eval_angles,
        )
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": validation,
            "primary_score": primary,
            "primary_score_kind": "jp_val_angle_mean_balanced_accuracy",
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        append_json_line(history_path, record)
        print_epoch_summary(record)

        if primary is not None and primary > best_score:
            best_score = primary
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                config=config,
                metrics=record,
            )

        if args.checkpoint_every > 0 and epoch % int(args.checkpoint_every) == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                config=config,
                metrics=record,
            )

    save_checkpoint(
        output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=int(args.epochs),
        config=config,
        metrics=record,
    )

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_evaluation = evaluate_selected_splits(
        model,
        cache,
        split_names=("jp_val", "jp_test", "manual_val"),
        device=device,
        batch_size=int(args.eval_batch_size),
        angles=full_eval_angles,
        amp=bool(args.amp),
    )
    summary = {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_primary_score": best_score,
        "best_checkpoint_selection": "jp_val only; manual_val is external holdout",
        "best_evaluation": final_evaluation,
        "config": config,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 2 or args.eval_batch_size < 2:
        raise ValueError("batch sizes must be at least 2")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must not be negative")
    if not 0.0 <= args.rotation_augment_deg <= 22.5:
        raise ValueError("--rotation-augment-deg must be in [0,22.5] for C8 residual augmentation")
    if not args.eval_angles:
        raise ValueError("--eval-angles must not be empty")
    if not any(abs(float(angle)) < 1.0e-9 for angle in args.eval_angles):
        raise ValueError("--eval-angles must include 0")
    if not 0.0 < args.cache_vram_fraction < 0.8:
        raise ValueError("--cache-vram-fraction must be between 0 and 0.8")
    if args.angle_eval_every < 1:
        raise ValueError("--angle-eval-every must be positive")


def configure_cuda(*, tf32: bool) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Run output directory is not empty: {path}. Use --overwrite to replace run outputs."
        )
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in ("history.jsonl", "summary.json", "config.json", "best.pt", "last.pt"):
            candidate = path / name
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        checkpoint_dir = path / "checkpoints"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)


def load_training_cache(
    database: Path,
    *,
    input_mode: str,
    device: torch.device,
    cache_device: str,
    cache_vram_fraction: float,
) -> TrainingCache:
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM experiment_metadata")
        }
        image_size = int(metadata["image_size"])
        splits = {
            split_name: load_split(connection, split_name, image_size=image_size)
            for split_name in ("train", "jp_val", "jp_test", "manual_val")
        }
    finally:
        connection.close()

    mean, std = compute_input_statistics(splits["train"], input_mode=input_mode)
    total_image_bytes = sum(split.images_u8.numel() for split in splits.values())
    resolved = resolve_cache_device(
        cache_device,
        image_bytes=total_image_bytes,
        cache_vram_fraction=cache_vram_fraction,
    )
    if resolved == "cuda":
        print(f"[cache] uploading {total_image_bytes / (1024**2):.1f} MiB RGB uint8 cache to VRAM")
        for split in splits.values():
            split.images_u8 = split.images_u8.to(device=device, non_blocking=False)
            split.labels = split.labels.to(device=device, non_blocking=False)
        torch.cuda.synchronize()
    else:
        print(f"[cache] keeping {total_image_bytes / (1024**2):.1f} MiB RGB uint8 cache in RAM")
        for split in splits.values():
            split.images_u8 = split.images_u8.pin_memory()
            split.labels = split.labels.pin_memory()

    for name, split in splits.items():
        print(
            f"[cache] {name}: unique={split.count} effective={split.effective_count}"
        )
    print(
        f"[cache] input_mode={input_mode} mean={list(mean)} std={list(std)}"
    )
    return TrainingCache(
        splits=splits,
        image_size=image_size,
        mean=mean,
        std=std,
        cache_device=resolved,
        image_bytes=total_image_bytes,
        input_mode=input_mode,
    )


def load_split(
    connection: sqlite3.Connection,
    split_name: str,
    *,
    image_size: int,
) -> SplitCache:
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM sample WHERE split = ?",
            (split_name,),
        ).fetchone()[0]
    )
    if count < 1:
        raise ValueError(f"Dataset split is empty: {split_name}")

    images = np.empty((count, image_size, image_size, 3), dtype=np.uint8)
    labels = np.empty((count,), dtype=np.int64)
    repeats = np.empty((count,), dtype=np.int64)
    suit = np.empty((count,), dtype=object)
    source = np.empty((count,), dtype=object)
    brightness = np.empty((count,), dtype=object)
    shadow = np.empty((count,), dtype=object)
    sample_ids: list[str] = []

    rows = connection.execute(
        """
        SELECT sample_id, is_red, image_rgb_u8, train_repeat, suit, source,
               COALESCE(brightness, '') AS brightness,
               COALESCE(shadow, '') AS shadow
        FROM sample
        WHERE split = ?
        ORDER BY sample_id
        """,
        (split_name,),
    )
    expected_bytes = image_size * image_size * 3
    for index, row in enumerate(rows):
        raw = bytes(row["image_rgb_u8"])
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{row['sample_id']} has {len(raw)} RGB bytes, expected {expected_bytes}"
            )
        images[index] = np.frombuffer(raw, dtype=np.uint8).reshape(image_size, image_size, 3)
        labels[index] = int(row["is_red"])
        repeats[index] = int(row["train_repeat"])
        suit[index] = str(row["suit"])
        source[index] = str(row["source"])
        brightness[index] = str(row["brightness"])
        shadow[index] = str(row["shadow"])
        sample_ids.append(str(row["sample_id"]))

    if np.any(repeats < 1):
        raise ValueError(f"Split {split_name} contains train_repeat < 1")
    return SplitCache(
        name=split_name,
        images_u8=torch.from_numpy(images),
        labels=torch.from_numpy(labels),
        train_repeat=repeats,
        suit=suit,
        source=source,
        brightness=brightness,
        shadow=shadow,
        sample_ids=sample_ids,
    )


def compute_input_statistics(
    split: SplitCache,
    *,
    input_mode: str,
    block_size: int = 256,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if split.images_u8.device.type != "cpu":
        raise ValueError("Input statistics must be computed before moving cache to CUDA")
    images = split.images_u8.numpy()
    channel_count = INPUT_CHANNELS[input_mode]
    total_weighted_pixels = 0.0
    channel_sum = np.zeros((channel_count,), dtype=np.float64)
    channel_square_sum = np.zeros((channel_count,), dtype=np.float64)

    for start in range(0, split.count, block_size):
        end = min(split.count, start + block_size)
        represented = rgb_u8_to_input_numpy(images[start:end], input_mode=input_mode)
        weights = split.train_repeat[start:end].astype(np.float64)
        pixel_count = represented.shape[1] * represented.shape[2]
        per_sample_sum = represented.sum(axis=(1, 2), dtype=np.float64)
        per_sample_square_sum = np.square(represented, dtype=np.float64).sum(
            axis=(1, 2), dtype=np.float64
        )
        channel_sum += (per_sample_sum * weights[:, None]).sum(axis=0)
        channel_square_sum += (per_sample_square_sum * weights[:, None]).sum(axis=0)
        total_weighted_pixels += float(weights.sum() * pixel_count)

    mean = channel_sum / total_weighted_pixels
    variance = np.maximum(channel_square_sum / total_weighted_pixels - mean * mean, 0.0)
    std = np.sqrt(variance)
    std = np.maximum(std, 1.0 / 255.0)
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


def rgb_u8_to_input_numpy(images_u8: np.ndarray, *, input_mode: str) -> np.ndarray:
    mode = normalize_input_mode(input_mode)
    if images_u8.ndim != 4 or images_u8.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB array, got {images_u8.shape}")
    rgb = images_u8.astype(np.float32) * (1.0 / 255.0)
    if mode == "rgb":
        return rgb
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    cr = np.clip(cr, 0.0, 1.0)
    if mode == "cr":
        return cr[..., None]
    return np.stack((y, cr), axis=-1)


def rgb_u8_to_input_torch(images_u8: torch.Tensor, *, input_mode: str) -> torch.Tensor:
    mode = normalize_input_mode(input_mode)
    if images_u8.ndim != 4 or images_u8.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB tensor, got {tuple(images_u8.shape)}")
    rgb = images_u8.permute(0, 3, 1, 2).float().mul_(1.0 / 255.0)
    if mode == "rgb":
        return rgb
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    cr = cr.clamp_(0.0, 1.0)
    if mode == "cr":
        return cr
    return torch.cat((y, cr), dim=1)


def color_transform_description(input_mode: str) -> str:
    mode = normalize_input_mode(input_mode)
    if mode == "rgb":
        return "RGB channels scaled to [0,1]"
    if mode == "cr":
        return "BT.601-like Cr = 0.5R - 0.418688G - 0.081312B + 0.5"
    return (
        "BT.601-like Y+Cr: Y=0.299R+0.587G+0.114B; "
        "Cr=0.5R-0.418688G-0.081312B+0.5"
    )


def resolve_cache_device(
    requested: str,
    *,
    image_bytes: int,
    cache_vram_fraction: float,
) -> str:
    if requested == "cuda":
        return "cuda"
    if requested == "cpu":
        return "cpu"
    free_bytes, _ = torch.cuda.mem_get_info()
    budget = int(free_bytes * cache_vram_fraction)
    print(
        f"[cache] free VRAM={free_bytes / (1024**3):.2f} GiB, "
        f"auto budget={budget / (1024**3):.2f} GiB"
    )
    return "cuda" if image_bytes <= budget else "cpu"


def train_one_epoch(
    model: nn.Module,
    split: SplitCache,
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    batch_size: int,
    device: torch.device,
    cache: TrainingCache,
    rotation_augment_deg: float,
    amp: bool,
    epoch: int,
    seed: int,
) -> dict[str, Any]:
    model.train()
    repeated_indices = np.repeat(np.arange(split.count, dtype=np.int64), split.train_repeat)
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    rng.shuffle(repeated_indices)

    total_loss_gpu = torch.zeros((), device=device, dtype=torch.float64)
    total_correct_gpu = torch.zeros((), device=device, dtype=torch.int64)
    total_count = 0
    step_count = 0

    torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, len(repeated_indices), batch_size):
        indices = repeated_indices[start : start + batch_size]
        rgb_u8, targets = fetch_batch(split, indices, device=device)
        images = rgb_u8_to_input_torch(rgb_u8, input_mode=cache.input_mode)
        if rotation_augment_deg > 0.0:
            angles = torch.empty(
                (images.shape[0],), device=device, dtype=torch.float32
            ).uniform_(-rotation_augment_deg, rotation_augment_deg)
            images = rotate_batch(images, angles)
        images = normalize_tensor(images, mean=cache.mean, std=cache.std)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_count = int(targets.shape[0])
        total_loss_gpu.add_(loss.detach().to(torch.float64) * batch_count)
        total_correct_gpu.add_((logits.detach().argmax(dim=1) == targets).sum())
        total_count += batch_count
        step_count += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    total_loss = float(total_loss_gpu.item())
    total_correct = int(total_correct_gpu.item())
    return {
        "loss": total_loss / max(1, total_count),
        "accuracy": total_correct / max(1, total_count),
        "correct": total_correct,
        "count": total_count,
        "optimizer_steps": step_count,
        "seconds": elapsed,
        "samples_per_second": total_count / max(elapsed, 1.0e-9),
    }


def fetch_batch(
    split: SplitCache,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    index_cpu = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    if split.images_u8.device.type == "cuda":
        index_device = index_cpu.to(device=device, non_blocking=False)
        return (
            split.images_u8.index_select(0, index_device),
            split.labels.index_select(0, index_device),
        )
    images = split.images_u8.index_select(0, index_cpu)
    targets = split.labels.index_select(0, index_cpu)
    return (
        images.to(device=device, non_blocking=True),
        targets.to(device=device, non_blocking=True),
    )


def normalize_tensor(
    images: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    mean_tensor = images.new_tensor(mean).view(1, -1, 1, 1)
    std_tensor = images.new_tensor(std).view(1, -1, 1, 1)
    return images.sub(mean_tensor).div(std_tensor)


def rotate_batch(images: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected NCHW images, got {tuple(images.shape)}")
    if angles_deg.ndim != 1 or angles_deg.shape[0] != images.shape[0]:
        raise ValueError("angles_deg must contain one angle per image")
    radians = angles_deg.to(dtype=torch.float32) * (math.pi / 180.0)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    theta = torch.zeros(
        (images.shape[0], 2, 3),
        device=images.device,
        dtype=torch.float32,
    )
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def evaluate_selected_splits(
    model: nn.Module,
    cache: TrainingCache,
    *,
    split_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    angles: Sequence[float],
    amp: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_name in split_names:
        split = cache.splits[split_name]
        angle_payload: dict[str, Any] = {}
        for angle in angles:
            evaluation = evaluate_split(
                model,
                split,
                device=device,
                batch_size=batch_size,
                cache=cache,
                angle_deg=float(angle),
                amp=amp,
            )
            payload = {
                "loss": evaluation.loss,
                "overall": evaluation.metrics,
                "by_suit": grouped_binary_metrics(
                    split.suit,
                    predictions=evaluation.predictions,
                    targets=evaluation.targets,
                ),
            }
            if split_name == "manual_val":
                condition_keys = np.asarray(
                    [
                        f"brightness={brightness}|shadow={shadow}"
                        for brightness, shadow in zip(split.brightness, split.shadow)
                    ],
                    dtype=object,
                )
                payload["by_condition"] = grouped_binary_metrics(
                    condition_keys,
                    predictions=evaluation.predictions,
                    targets=evaluation.targets,
                )
            angle_payload[angle_key(float(angle))] = payload
        result[split_name] = {"angles": angle_payload}
    return result


def evaluate_split(
    model: nn.Module,
    split: SplitCache,
    *,
    device: torch.device,
    batch_size: int,
    cache: TrainingCache,
    angle_deg: float,
    amp: bool,
) -> EvaluationResult:
    model.eval()
    total_loss_gpu = torch.zeros((), device=device, dtype=torch.float64)
    predictions_gpu: list[torch.Tensor] = []
    targets_gpu: list[torch.Tensor] = []
    total_count = 0

    with torch.inference_mode():
        for start in range(0, split.count, batch_size):
            indices = np.arange(start, min(split.count, start + batch_size), dtype=np.int64)
            rgb_u8, targets = fetch_batch(split, indices, device=device)
            images = rgb_u8_to_input_torch(rgb_u8, input_mode=cache.input_mode)
            if abs(angle_deg) > 1.0e-9:
                angles = torch.full(
                    (images.shape[0],),
                    angle_deg,
                    device=device,
                    dtype=torch.float32,
                )
                images = rotate_batch(images, angles)
            images = normalize_tensor(images, mean=cache.mean, std=cache.std)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
                loss_sum = F.cross_entropy(logits, targets, reduction="sum")
            total_loss_gpu.add_(loss_sum.detach().to(torch.float64))
            predictions_gpu.append(logits.argmax(dim=1))
            targets_gpu.append(targets)
            total_count += int(targets.shape[0])

    predictions = torch.cat(predictions_gpu).cpu().numpy()
    targets = torch.cat(targets_gpu).cpu().numpy()
    return EvaluationResult(
        loss=float(total_loss_gpu.item()) / max(1, total_count),
        metrics=binary_metrics(predictions, targets),
        predictions=predictions,
        targets=targets,
    )


def binary_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    tp = int(((predictions == 1) & (targets == 1)).sum())
    tn = int(((predictions == 0) & (targets == 0)).sum())
    fp = int(((predictions == 1) & (targets == 0)).sum())
    fn = int(((predictions == 0) & (targets == 1)).sum())
    normal = tn + fp
    red = tp + fn
    count = normal + red
    recall = tp / red if red else 0.0
    specificity = tn / normal if normal else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (tp + tn) / count if count else 0.0
    balanced = 0.5 * (recall + specificity) if normal and red else accuracy
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sample_count": count,
        "normal_count": normal,
        "red_count": red,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def grouped_binary_metrics(
    groups: np.ndarray,
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in sorted(set(str(item) for item in groups)):
        mask = np.asarray([str(item) == value for item in groups], dtype=bool)
        result[value] = binary_metrics(predictions[mask], targets[mask])
    return result


def robust_jp_validation_score(
    validation: dict[str, Any],
    *,
    required_angles: Sequence[float],
) -> float | None:
    angles = validation.get("jp_val", {}).get("angles", {})
    keys = [angle_key(float(angle)) for angle in required_angles]
    if not keys or any(key not in angles for key in keys):
        return None
    return float(
        np.mean(
            [float(angles[key]["overall"]["balanced_accuracy"]) for key in keys]
        )
    )


def angle_key(angle: float) -> str:
    value = float(angle)
    if value.is_integer():
        return f"{int(value)}deg"
    return f"{value:g}deg"


def print_epoch_summary(record: dict[str, Any]) -> None:
    train = record["train"]
    jp0 = record["validation"]["jp_val"]["angles"][angle_key(0.0)]["overall"]
    manual0 = record["validation"]["manual_val"]["angles"][angle_key(0.0)]["overall"]
    angle_parts = []
    for key, value in record["validation"]["manual_val"]["angles"].items():
        angle_parts.append(
            f"manual@{key}={value['overall']['balanced_accuracy']:.4f}"
        )
    print(
        f"epoch={int(record['epoch']):03d} "
        f"train_loss={train['loss']:.5f} train_acc={train['accuracy']:.4f} "
        f"jp_bal={jp0['balanced_accuracy']:.4f} "
        f"manual_bal={manual0['balanced_accuracy']:.4f} "
        f"manual_recall={manual0['recall']:.4f} "
        f"samples/s={train['samples_per_second']:.1f} "
        f"peak_vram={record['cuda_peak_allocated_bytes'] / (1024**3):.2f}GiB "
        + " ".join(angle_parts)
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        temporary,
    )
    os.replace(temporary, path)


def append_json_line(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        output.write("\n")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as error:
        print(f"CUDA OOM: {error}", file=sys.stderr)
        raise SystemExit(75) from error
