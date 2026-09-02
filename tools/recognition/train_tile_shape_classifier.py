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

from tile_shape_classifier import DEFAULT_C8_FIELDS, build_model, describe_model
from classifier_geometric_augmentation import projective_augment_batch


DEFAULT_SEED = 42
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LEARNING_RATE = 1.0e-3
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_ROTATION_AUGMENT_DEG = 0.0
DEFAULT_EVAL_ANGLES = (0.0, 15.0, 30.0, 45.0)


@dataclass
class SplitCache:
    name: str
    images_u8: torch.Tensor
    labels: torch.Tensor
    source: np.ndarray
    brightness: np.ndarray
    shadow: np.ndarray
    region: np.ndarray
    sample_ids: list[str]

    @property
    def count(self) -> int:
        return int(self.labels.shape[0])


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    macro_accuracy: float
    count: int
    correct: int
    per_class_accuracy: list[float | None]
    confusion_matrix: list[list[int]]
    predictions: np.ndarray
    targets: np.ndarray


@dataclass
class TrainingCache:
    splits: dict[str, SplitCache]
    image_size: int
    class_labels: tuple[str, ...]
    mean: float
    std: float
    cache_device: str
    image_bytes: int


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Train the gray64 tile-shape classifier using the class list stored in the "
            "compact SQLite DB. The DB is read once at startup; images are kept as uint8 "
            "tensors in RAM or VRAM for the entire run. No per-sample SQLite/PIL work "
            "occurs in the hot loop."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("c8", "plain"), default="c8")
    parser.add_argument("--c8-fields", type=int, nargs="+", default=list(DEFAULT_C8_FIELDS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--rotation-augment-deg",
        type=float,
        default=DEFAULT_ROTATION_AUGMENT_DEG,
        help="Uniform residual rotation range. B experiment uses 22.5; A uses 0.",
    )
    parser.add_argument(
        "--perspective-augment",
        type=float,
        default=0.0,
        help="Maximum normalized projective strength per axis. 0 disables; try 0.08 for mild camera tilt.",
    )
    parser.add_argument(
        "--shear-augment",
        type=float,
        default=0.0,
        help="Maximum normalized x/y shear. 0 disables; try 0.08.",
    )
    parser.add_argument(
        "--stretch-augment",
        type=float,
        default=0.0,
        help="Maximum independent x/y scale deviation. 0 disables; try 0.12.",
    )
    parser.add_argument(
        "--projective-augment-probability",
        type=float,
        default=0.0,
        help="Probability that a training sample receives projective/shear/stretch augmentation.",
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
        help=(
            "Where uint8 image caches live after startup. auto uses VRAM when the full "
            "dataset cache is comfortably below the configured fraction of free VRAM."
        ),
    )
    parser.add_argument(
        "--cache-vram-fraction",
        type=float,
        default=0.25,
        help="Maximum fraction of currently free VRAM used by auto image caching.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision. Enabled by default.",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow TF32 for float32 CUDA matmul/convolution paths.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--angle-eval-every",
        type=int,
        default=5,
        help=(
            "Evaluate the full configured angle sweep every N epochs. best.pt is selected "
            "only from these full-sweep epochs using mean manual accuracy across angles; "
            "0 degree validation still runs every epoch for monitoring."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing run directory. Known run outputs are replaced.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Linux training script")

    repository_root = args.repository_root.resolve()
    database = args.database.resolve()
    output_dir = args.output_dir.resolve()
    prepare_output_directory(output_dir, overwrite=bool(args.overwrite))

    seed_everything(int(args.seed))
    configure_cuda(tf32=bool(args.tf32))
    device = torch.device("cuda")

    cache = load_training_cache(
        database,
        device=device,
        cache_device=str(args.cache_device),
        cache_vram_fraction=float(args.cache_vram_fraction),
    )
    class_count = len(cache.class_labels)
    if class_count not in (34, 35):
        raise ValueError(
            f"Expected the canonical 34-class base task or 35-class base+invalid task, "
            f"found {class_count} classes"
        )

    model = build_model(
        args.model,
        class_count=class_count,
        c8_fields=tuple(args.c8_fields),
    ).to(device)
    model_description = describe_model(model, args.model)

    config = {
        "database": str(database),
        "output_dir": str(output_dir),
        "model": asdict(model_description),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "rotation_augment_deg": float(args.rotation_augment_deg),
        "perspective_augment": float(args.perspective_augment),
        "shear_augment": float(args.shear_augment),
        "stretch_augment": float(args.stretch_augment),
        "projective_augment_probability": float(args.projective_augment_probability),
        "eval_angles": [float(value) for value in args.eval_angles],
        "angle_eval_every": int(args.angle_eval_every),
        "best_checkpoint_metric": "manual_angle_mean_full_sweep_only",
        "seed": int(args.seed),
        "amp": bool(args.amp),
        "tf32": bool(args.tf32),
        "cache_device": cache.cache_device,
        "cache_image_bytes": cache.image_bytes,
        "image_size": cache.image_size,
        "class_labels": list(cache.class_labels),
        "normalization": {"mean": cache.mean, "std": cache.std},
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
    best_metrics: dict[str, Any] | None = None
    train_split = cache.splits["train"]

    for epoch in range(1, int(args.epochs) + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_metrics = train_one_epoch(
            model,
            train_split,
            optimizer=optimizer,
            scaler=scaler,
            batch_size=int(args.batch_size),
            device=device,
            mean=cache.mean,
            std=cache.std,
            rotation_augment_deg=float(args.rotation_augment_deg),
            perspective_augment=float(args.perspective_augment),
            shear_augment=float(args.shear_augment),
            stretch_augment=float(args.stretch_augment),
            projective_augment_probability=float(args.projective_augment_probability),
            amp=bool(args.amp),
            epoch=epoch,
            seed=int(args.seed),
        )

        full_eval_angles = tuple(float(value) for value in args.eval_angles)
        if (
            epoch == 1
            or epoch == int(args.epochs)
            or epoch % int(args.angle_eval_every) == 0
        ):
            epoch_eval_angles = full_eval_angles
        else:
            epoch_eval_angles = (0.0,)
        validation = evaluate_all(
            model,
            cache,
            device=device,
            batch_size=int(args.eval_batch_size),
            angles=epoch_eval_angles,
            amp=bool(args.amp),
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        primary = robust_manual_validation_score(
            validation,
            required_angles=full_eval_angles,
        )
        epoch_record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": validation,
            "primary_score": primary,
            "primary_score_kind": "manual_angle_mean",
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        append_json_line(history_path, epoch_record)
        print_epoch_summary(epoch_record)

        # best.pt represents the best rotation-robust checkpoint.  Only epochs
        # with the complete configured angle sweep are eligible; otherwise a
        # 0-degree-only epoch could incorrectly outrank a robust checkpoint.
        if primary is not None and primary > best_score:
            best_score = primary
            best_epoch = epoch
            best_metrics = epoch_record
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                config=config,
                metrics=epoch_record,
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
                metrics=epoch_record,
            )

    # The final epoch always runs the full requested angle sweep, so reuse it
    # instead of paying for an identical validation pass again.
    last_validation = validation
    save_checkpoint(
        output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=int(args.epochs),
        config=config,
        metrics={"validation": last_validation},
    )
    summary = {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_primary_score": best_score,
        "best_metrics": best_metrics,
        "last_validation": last_validation,
        "config": config,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(
        f"completed: best_epoch={best_epoch} primary={best_score:.6f} "
        f"output={output_dir}"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 2 or args.eval_batch_size < 2:
        raise ValueError("batch sizes must be at least 2")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must not be negative")
    if not 0.0 <= args.rotation_augment_deg <= 45.0:
        raise ValueError("--rotation-augment-deg must be in [0,45]")
    if not 0.0 <= args.perspective_augment <= 0.25:
        raise ValueError("--perspective-augment must be in [0,0.25]")
    if not 0.0 <= args.shear_augment <= 0.25:
        raise ValueError("--shear-augment must be in [0,0.25]")
    if not 0.0 <= args.stretch_augment <= 0.30:
        raise ValueError("--stretch-augment must be in [0,0.30]")
    if not 0.0 <= args.projective_augment_probability <= 1.0:
        raise ValueError("--projective-augment-probability must be in [0,1]")
    if not args.eval_angles:
        raise ValueError("--eval-angles must not be empty")
    if not any(abs(float(angle)) < 1.0e-9 for angle in args.eval_angles):
        raise ValueError("--eval-angles must include 0 for checkpoint selection")
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
            f"Run output directory is not empty: {path}. Use a new directory or --overwrite."
        )
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in ("history.jsonl", "summary.json", "config.json", "best.pt", "last.pt"):
            candidate = path / name
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        checkpoint_directory = path / "checkpoints"
        if checkpoint_directory.exists():
            shutil.rmtree(checkpoint_directory)


def load_training_cache(
    database: Path,
    *,
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
        class_labels = tuple(json.loads(metadata["base_labels"]))
        splits: dict[str, SplitCache] = {}
        total_image_bytes = 0
        for split_name in ("train", "manual_val", "jp_val"):
            split = load_split(connection, split_name, image_size=image_size)
            splits[split_name] = split
            total_image_bytes += split.images_u8.numel()
    finally:
        connection.close()

    train_np = splits["train"].images_u8.numpy()
    mean = float(train_np.mean(dtype=np.float64) / 255.0)
    std = float(train_np.std(dtype=np.float64) / 255.0)
    std = max(std, 1.0 / 255.0)

    resolved_cache_device = resolve_cache_device(
        cache_device,
        image_bytes=total_image_bytes,
        cache_vram_fraction=cache_vram_fraction,
    )
    if resolved_cache_device == "cuda":
        print(
            f"[cache] uploading {total_image_bytes / (1024**2):.1f} MiB uint8 image cache to VRAM"
        )
        for split in splits.values():
            split.images_u8 = split.images_u8.to(device=device, non_blocking=False)
            split.labels = split.labels.to(device=device, non_blocking=False)
        torch.cuda.synchronize()
    else:
        print(
            f"[cache] keeping {total_image_bytes / (1024**2):.1f} MiB uint8 image cache in RAM"
        )
        # Pinning the compact cache makes the fallback H2D copy path asynchronous.
        for split in splits.values():
            split.images_u8 = split.images_u8.pin_memory()
            split.labels = split.labels.pin_memory()

    for name, split in splits.items():
        print(f"[cache] {name}: {split.count} images")
    print(f"[cache] global train normalization mean={mean:.6f} std={std:.6f}")
    return TrainingCache(
        splits=splits,
        image_size=image_size,
        class_labels=class_labels,
        mean=mean,
        std=std,
        cache_device=resolved_cache_device,
        image_bytes=total_image_bytes,
    )


def load_split(
    connection: sqlite3.Connection,
    split_name: str,
    *,
    image_size: int,
) -> SplitCache:
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM sample WHERE split = ?", (split_name,)
        ).fetchone()[0]
    )
    if count < 1:
        raise ValueError(f"Dataset split is empty: {split_name}")

    images = np.empty((count, image_size, image_size), dtype=np.uint8)
    labels = np.empty((count,), dtype=np.int64)
    source = np.empty((count,), dtype=object)
    brightness = np.empty((count,), dtype=object)
    shadow = np.empty((count,), dtype=object)
    region = np.empty((count,), dtype=object)
    sample_ids: list[str] = []

    rows = connection.execute(
        """
        SELECT sample_id, class_index, image_gray_u8, source,
               COALESCE(brightness, '') AS brightness,
               COALESCE(shadow, '') AS shadow,
               COALESCE(region, '') AS region
        FROM sample
        WHERE split = ?
        ORDER BY sample_id
        """,
        (split_name,),
    )
    expected_bytes = image_size * image_size
    for index, row in enumerate(rows):
        raw = bytes(row["image_gray_u8"])
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{row['sample_id']} has {len(raw)} gray bytes, expected {expected_bytes}"
            )
        images[index] = np.frombuffer(raw, dtype=np.uint8).reshape(image_size, image_size)
        labels[index] = int(row["class_index"])
        source[index] = str(row["source"])
        brightness[index] = str(row["brightness"])
        shadow[index] = str(row["shadow"])
        region[index] = str(row["region"])
        sample_ids.append(str(row["sample_id"]))

    return SplitCache(
        name=split_name,
        images_u8=torch.from_numpy(images),
        labels=torch.from_numpy(labels),
        source=source,
        brightness=brightness,
        shadow=shadow,
        region=region,
        sample_ids=sample_ids,
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
        f"auto cache budget={budget / (1024**3):.2f} GiB"
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
    mean: float,
    std: float,
    rotation_augment_deg: float,
    perspective_augment: float,
    shear_augment: float,
    stretch_augment: float,
    projective_augment_probability: float,
    amp: bool,
    epoch: int,
    seed: int,
) -> dict[str, Any]:
    model.train()
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    order = rng.permutation(split.count)
    total_loss_gpu = torch.zeros((), device=device, dtype=torch.float64)
    total_correct_gpu = torch.zeros((), device=device, dtype=torch.int64)
    total_count = 0
    step_count = 0

    torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, split.count, batch_size):
        batch_indices = order[start : start + batch_size]
        images, targets = fetch_batch(split, batch_indices, device=device)
        images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
        if rotation_augment_deg > 0.0:
            angles = torch.empty(
                (images.shape[0],), device=device, dtype=torch.float32
            ).uniform_(-rotation_augment_deg, rotation_augment_deg)
            images = rotate_batch(images, angles)
        if projective_augment_probability > 0.0 and (
            perspective_augment > 0.0 or shear_augment > 0.0 or stretch_augment > 0.0
        ):
            images = projective_augment_batch(
                images,
                max_perspective=perspective_augment,
                max_shear=shear_augment,
                max_stretch=stretch_augment,
                probability=projective_augment_probability,
            )
        images = images.sub(mean).div(std)

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


def evaluate_all(
    model: nn.Module,
    cache: TrainingCache,
    *,
    device: torch.device,
    batch_size: int,
    angles: Sequence[float],
    amp: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_name in ("manual_val", "jp_val"):
        split = cache.splits[split_name]
        angle_results: dict[str, Any] = {}
        angle_zero_result: EvaluationResult | None = None
        for angle in angles:
            evaluation = evaluate_split(
                model,
                split,
                device=device,
                batch_size=batch_size,
                mean=cache.mean,
                std=cache.std,
                angle_deg=float(angle),
                class_count=len(cache.class_labels),
                amp=amp,
            )
            if abs(float(angle)) < 1.0e-9:
                angle_zero_result = evaluation
            angle_results[angle_key(angle)] = evaluation_to_json(evaluation)
        payload: dict[str, Any] = {"angles": angle_results}
        if split_name == "manual_val" and angle_zero_result is not None:
            payload["conditions_at_0deg"] = manual_condition_metrics(
                split,
                predictions=angle_zero_result.predictions,
                targets=angle_zero_result.targets,
            )
        result[split_name] = payload
    return result


def evaluate_split(
    model: nn.Module,
    split: SplitCache,
    *,
    device: torch.device,
    batch_size: int,
    mean: float,
    std: float,
    angle_deg: float,
    class_count: int,
    amp: bool,
) -> EvaluationResult:
    model.eval()
    total_loss_gpu = torch.zeros((), device=device, dtype=torch.float64)
    total_count = 0
    predictions_gpu: list[torch.Tensor] = []
    targets_gpu: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, split.count, batch_size):
            indices = np.arange(start, min(split.count, start + batch_size), dtype=np.int64)
            images, targets = fetch_batch(split, indices, device=device)
            images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
            if abs(angle_deg) > 1.0e-9:
                angles = torch.full(
                    (images.shape[0],),
                    angle_deg,
                    device=device,
                    dtype=torch.float32,
                )
                images = rotate_batch(images, angles)
            images = images.sub(mean).div(std)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
                loss_sum = F.cross_entropy(logits, targets, reduction="sum")
            prediction = logits.argmax(dim=1)
            total_loss_gpu.add_(loss_sum.detach().to(torch.float64))
            total_count += int(targets.shape[0])
            predictions_gpu.append(prediction)
            targets_gpu.append(targets)

    predicted = torch.cat(predictions_gpu).cpu().numpy()
    targets_np = torch.cat(targets_gpu).cpu().numpy()
    total_loss = float(total_loss_gpu.item())
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (targets_np, predicted), 1)
    correct = int(np.trace(confusion))
    per_class: list[float | None] = []
    for class_index in range(class_count):
        class_total = int(confusion[class_index].sum())
        per_class.append(
            None
            if class_total == 0
            else float(confusion[class_index, class_index] / class_total)
        )
    present = [value for value in per_class if value is not None]
    return EvaluationResult(
        loss=total_loss / max(1, total_count),
        accuracy=correct / max(1, total_count),
        macro_accuracy=float(np.mean(present)) if present else 0.0,
        count=total_count,
        correct=correct,
        per_class_accuracy=per_class,
        confusion_matrix=confusion.tolist(),
        predictions=predicted,
        targets=targets_np,
    )


def evaluation_to_json(result: EvaluationResult) -> dict[str, Any]:
    return {
        "loss": result.loss,
        "accuracy": result.accuracy,
        "macro_accuracy": result.macro_accuracy,
        "count": result.count,
        "correct": result.correct,
        "per_class_accuracy": result.per_class_accuracy,
        "confusion_matrix": result.confusion_matrix,
    }


def manual_condition_metrics(
    split: SplitCache,
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    if len(predictions) != split.count:
        raise ValueError("Prediction count does not match manual validation split")
    groups: dict[str, np.ndarray] = {}
    for field_name, values in (
        ("brightness", split.brightness),
        ("shadow", split.shadow),
        ("region", split.region),
    ):
        for value in sorted(set(str(item) for item in values)):
            mask = np.asarray([str(item) == value for item in values], dtype=bool)
            groups[f"{field_name}={value}"] = mask

    dark_mask = np.asarray(
        ["dark" in str(value).lower() for value in split.brightness], dtype=bool
    )
    if dark_mask.any():
        groups["brightness_contains_dark"] = dark_mask

    metrics: dict[str, Any] = {}
    for key, mask in groups.items():
        count = int(mask.sum())
        if count == 0:
            continue
        correct = int((predictions[mask] == targets[mask]).sum())
        metrics[key] = {
            "count": count,
            "correct": correct,
            "accuracy": correct / count,
        }
    return metrics


def robust_manual_validation_score(
    validation: dict[str, Any],
    *,
    required_angles: Sequence[float],
) -> float | None:
    """Return mean manual accuracy across the complete configured angle sweep.

    Non-full-evaluation epochs return None and are deliberately ineligible for
    best.pt selection.  This prevents a perfect 0-degree score from hiding a
    weaker arbitrary-angle checkpoint.
    """
    manual_angles = validation.get("manual_val", {}).get("angles", {})
    keys = [angle_key(float(angle)) for angle in required_angles]
    if not keys or any(key not in manual_angles for key in keys):
        return None
    return float(np.mean([float(manual_angles[key]["accuracy"]) for key in keys]))


def angle_key(angle: float) -> str:
    value = float(angle)
    if value.is_integer():
        return f"{int(value)}deg"
    return f"{value:g}deg"


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


def print_epoch_summary(record: dict[str, Any]) -> None:
    epoch = int(record["epoch"])
    train = record["train"]
    manual = record["validation"]["manual_val"]["angles"][angle_key(0.0)]
    jp = record["validation"]["jp_val"]["angles"][angle_key(0.0)]
    angle_parts = []
    for key, value in record["validation"]["manual_val"]["angles"].items():
        angle_parts.append(f"manual@{key}={value['accuracy']:.4f}")
    print(
        f"epoch={epoch:03d} "
        f"train_loss={train['loss']:.5f} train_acc={train['accuracy']:.4f} "
        f"manual={manual['accuracy']:.4f} jp={jp['accuracy']:.4f} "
        f"samples/s={train['samples_per_second']:.1f} "
        f"peak_vram={record['cuda_peak_allocated_bytes'] / (1024**3):.2f}GiB "
        + " ".join(angle_parts)
    )


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
        # EX_TEMPFAIL-style distinct code lets the sweep skip an oversized batch
        # without hiding real model/environment failures behind the same status.
        raise SystemExit(75) from error
