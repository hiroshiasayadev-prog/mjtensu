from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import train_red_five_classifier as base
    from red_five_classifier import (
        DEFAULT_C8_FIELDS,
        INPUT_CHANNELS,
        SUPPORTED_INPUT_MODES,
        build_model,
        describe_model,
        normalize_input_mode,
    )
except ModuleNotFoundError:  # package import path used by tests/tools
    from tools.recognition import train_red_five_classifier as base
    from tools.recognition.red_five_classifier import (
        DEFAULT_C8_FIELDS,
        INPUT_CHANNELS,
        SUPPORTED_INPUT_MODES,
        build_model,
        describe_model,
        normalize_input_mode,
    )


DEFAULT_WARM_AUGMENT_PROB = 0.50
DEFAULT_WARM_STRENGTH_MIN = 0.10
DEFAULT_WARM_STRENGTH_MAX = 1.00
DEFAULT_WARM_RED_GAIN_MAX = 1.50
DEFAULT_WARM_GREEN_GAIN_MAX = 1.08
DEFAULT_WARM_BLUE_GAIN_MIN = 0.45
DEFAULT_WARM_EXPOSURE_MIN = 0.65
DEFAULT_WARM_EXPOSURE_MAX = 1.15


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Train the C8 red-five classifier from scratch with stochastic warm-light "
            "white-balance augmentation applied in RGB before RGB/Cr/Y+Cr conversion. "
            "The real warm-light capture set is intentionally not used for training."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-mode", choices=SUPPORTED_INPUT_MODES, required=True)
    parser.add_argument("--c8-fields", type=int, nargs="+", default=list(DEFAULT_C8_FIELDS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--rotation-augment-deg", type=float, default=22.5)
    parser.add_argument(
        "--eval-angles",
        type=float,
        nargs="+",
        default=[0.0, 15.0, 30.0, 45.0],
    )
    parser.add_argument("--seed", type=int, default=42)
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

    parser.add_argument("--warm-augment-prob", type=float, default=DEFAULT_WARM_AUGMENT_PROB)
    parser.add_argument("--warm-strength-min", type=float, default=DEFAULT_WARM_STRENGTH_MIN)
    parser.add_argument("--warm-strength-max", type=float, default=DEFAULT_WARM_STRENGTH_MAX)
    parser.add_argument("--warm-red-gain-max", type=float, default=DEFAULT_WARM_RED_GAIN_MAX)
    parser.add_argument("--warm-green-gain-max", type=float, default=DEFAULT_WARM_GREEN_GAIN_MAX)
    parser.add_argument("--warm-blue-gain-min", type=float, default=DEFAULT_WARM_BLUE_GAIN_MIN)
    parser.add_argument("--warm-exposure-min", type=float, default=DEFAULT_WARM_EXPOSURE_MIN)
    parser.add_argument("--warm-exposure-max", type=float, default=DEFAULT_WARM_EXPOSURE_MAX)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    base.validate_args(args)
    if not 0.0 <= args.warm_augment_prob <= 1.0:
        raise ValueError("--warm-augment-prob must be in [0,1]")
    if not 0.0 <= args.warm_strength_min <= args.warm_strength_max <= 1.0:
        raise ValueError("warm strength must satisfy 0 <= min <= max <= 1")
    if args.warm_red_gain_max < 1.0:
        raise ValueError("--warm-red-gain-max must be >= 1")
    if args.warm_green_gain_max <= 0.0:
        raise ValueError("--warm-green-gain-max must be positive")
    if not 0.0 < args.warm_blue_gain_min <= 1.0:
        raise ValueError("--warm-blue-gain-min must be in (0,1]")
    if not 0.0 < args.warm_exposure_min <= args.warm_exposure_max:
        raise ValueError("warm exposure must satisfy 0 < min <= max")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script")

    repository_root = args.repository_root.resolve()
    database = args.database.resolve()
    output_dir = args.output_dir.resolve()
    input_mode = normalize_input_mode(args.input_mode)

    base.prepare_output_directory(output_dir, overwrite=bool(args.overwrite))
    base.seed_everything(int(args.seed))
    base.configure_cuda(tf32=bool(args.tf32))
    device = torch.device("cuda")

    cache = base.load_training_cache(
        database,
        input_mode=input_mode,
        device=device,
        cache_device=str(args.cache_device),
        cache_vram_fraction=float(args.cache_vram_fraction),
    )
    model = build_model(input_mode, c8_fields=tuple(args.c8_fields)).to(device)
    model_description = describe_model(model, input_mode)

    warm_config = {
        "kind": "per_sample_rgb_white_balance_before_input_conversion",
        "probability": float(args.warm_augment_prob),
        "strength_min": float(args.warm_strength_min),
        "strength_max": float(args.warm_strength_max),
        "red_gain_at_strength_1": float(args.warm_red_gain_max),
        "green_gain_at_strength_1": float(args.warm_green_gain_max),
        "blue_gain_at_strength_1": float(args.warm_blue_gain_min),
        "exposure_min": float(args.warm_exposure_min),
        "exposure_max": float(args.warm_exposure_max),
        "real_warm_capture_policy": "external_holdout_not_used_for_training_or_checkpoint_selection",
    }

    config = {
        "database": str(database),
        "output_dir": str(output_dir),
        "input_mode": input_mode,
        "input_channels": INPUT_CHANNELS[input_mode],
        "color_transform": base.color_transform_description(input_mode),
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
        "warm_augmentation": warm_config,
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
            "computed_from_unaugmented_training_rgb": True,
        },
        "train_unique_samples": cache.splits["train"].count,
        "train_effective_samples": cache.splits["train"].effective_count,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    base.atomic_write_json(output_dir / "config.json", config)
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
    record: dict[str, Any] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        torch.cuda.reset_peak_memory_stats()
        train_metrics = train_one_epoch_warm(
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
            warm_augment_prob=float(args.warm_augment_prob),
            warm_strength_min=float(args.warm_strength_min),
            warm_strength_max=float(args.warm_strength_max),
            warm_red_gain_max=float(args.warm_red_gain_max),
            warm_green_gain_max=float(args.warm_green_gain_max),
            warm_blue_gain_min=float(args.warm_blue_gain_min),
            warm_exposure_min=float(args.warm_exposure_min),
            warm_exposure_max=float(args.warm_exposure_max),
        )

        if (
            epoch == 1
            or epoch == int(args.epochs)
            or epoch % int(args.angle_eval_every) == 0
        ):
            epoch_eval_angles = full_eval_angles
        else:
            epoch_eval_angles = (0.0,)

        validation = base.evaluate_selected_splits(
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

        primary = base.robust_jp_validation_score(
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
        base.append_json_line(history_path, record)
        base.print_epoch_summary(record)

        if primary is not None and primary > best_score:
            best_score = primary
            best_epoch = epoch
            base.save_checkpoint(
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
            base.save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                config=config,
                metrics=record,
            )

    if record is None:
        raise RuntimeError("No training epoch was executed")

    base.save_checkpoint(
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
    final_evaluation = base.evaluate_selected_splits(
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
        "best_checkpoint_selection": "jp_val only; manual_val and real warm captures are external holdouts",
        "best_evaluation": final_evaluation,
        "config": config,
    }
    base.atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def train_one_epoch_warm(
    model: torch.nn.Module,
    split: base.SplitCache,
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    batch_size: int,
    device: torch.device,
    cache: base.TrainingCache,
    rotation_augment_deg: float,
    amp: bool,
    epoch: int,
    seed: int,
    warm_augment_prob: float,
    warm_strength_min: float,
    warm_strength_max: float,
    warm_red_gain_max: float,
    warm_green_gain_max: float,
    warm_blue_gain_min: float,
    warm_exposure_min: float,
    warm_exposure_max: float,
) -> dict[str, Any]:
    model.train()
    repeated_indices = np.repeat(np.arange(split.count, dtype=np.int64), split.train_repeat)
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    rng.shuffle(repeated_indices)

    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(seed + epoch * 7_919 + 313)

    total_loss_gpu = torch.zeros((), device=device, dtype=torch.float64)
    total_correct_gpu = torch.zeros((), device=device, dtype=torch.int64)
    total_count = 0
    total_warm_augmented = 0
    step_count = 0

    torch.cuda.synchronize()
    started = time.perf_counter()

    for start in range(0, len(repeated_indices), batch_size):
        indices = repeated_indices[start : start + batch_size]
        rgb_u8, targets = base.fetch_batch(split, indices, device=device)

        rgb, warm_count = augment_warm_rgb(
            rgb_u8,
            probability=warm_augment_prob,
            strength_min=warm_strength_min,
            strength_max=warm_strength_max,
            red_gain_max=warm_red_gain_max,
            green_gain_max=warm_green_gain_max,
            blue_gain_min=warm_blue_gain_min,
            exposure_min=warm_exposure_min,
            exposure_max=warm_exposure_max,
            generator=torch_generator,
        )
        images = rgb_float_to_input_torch(rgb, input_mode=cache.input_mode)

        if rotation_augment_deg > 0.0:
            angles = (
                torch.rand(
                    (images.shape[0],),
                    device=device,
                    dtype=torch.float32,
                    generator=torch_generator,
                )
                * (2.0 * rotation_augment_deg)
                - rotation_augment_deg
            )
            images = base.rotate_batch(images, angles)

        images = base.normalize_tensor(images, mean=cache.mean, std=cache.std)

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
        total_warm_augmented += warm_count
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
        "warm_augmented_count": total_warm_augmented,
        "warm_augmented_fraction": total_warm_augmented / max(1, total_count),
        "optimizer_steps": step_count,
        "seconds": elapsed,
        "samples_per_second": total_count / max(elapsed, 1.0e-9),
    }


def augment_warm_rgb(
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
    if images_u8.ndim != 4 or images_u8.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB uint8 tensor, got {tuple(images_u8.shape)}")

    rgb = images_u8.float().mul_(1.0 / 255.0)
    count = rgb.shape[0]
    if probability <= 0.0 or count == 0:
        return rgb, 0

    selected = torch.rand(
        (count,),
        device=rgb.device,
        generator=generator,
    ) < probability
    selected_count = int(selected.sum().item())
    if selected_count == 0:
        return rgb, 0

    strength = torch.rand(
        (count,),
        device=rgb.device,
        generator=generator,
    )
    strength = strength_min + strength * (strength_max - strength_min)
    strength = torch.where(selected, strength, torch.zeros_like(strength))

    red_gain = 1.0 + strength * (red_gain_max - 1.0)
    green_gain = 1.0 + strength * (green_gain_max - 1.0)
    blue_gain = 1.0 - strength * (1.0 - blue_gain_min)
    gains = torch.stack((red_gain, green_gain, blue_gain), dim=1).view(count, 1, 1, 3)

    if abs(exposure_min - exposure_max) < 1.0e-12:
        exposure = torch.full(
            (count,), exposure_min, device=rgb.device, dtype=torch.float32
        )
    else:
        log_min = math.log(exposure_min)
        log_max = math.log(exposure_max)
        exposure = torch.exp(
            log_min
            + torch.rand(
                (count,),
                device=rgb.device,
                generator=generator,
            )
            * (log_max - log_min)
        )
    exposure = torch.where(selected, exposure, torch.ones_like(exposure))

    rgb = rgb.mul(gains).mul(exposure.view(count, 1, 1, 1)).clamp_(0.0, 1.0)
    return rgb, selected_count


def rgb_float_to_input_torch(rgb_nhwc: torch.Tensor, *, input_mode: str) -> torch.Tensor:
    mode = normalize_input_mode(input_mode)
    if rgb_nhwc.ndim != 4 or rgb_nhwc.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB float tensor, got {tuple(rgb_nhwc.shape)}")

    rgb = rgb_nhwc.permute(0, 3, 1, 2)
    if mode == "rgb":
        return rgb

    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (0.5 * r - 0.418688 * g - 0.081312 * b + 0.5).clamp(0.0, 1.0)
    if mode == "cr":
        return cr
    return torch.cat((y, cr), dim=1)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as error:
        print(f"CUDA OOM: {error}")
        raise SystemExit(75) from error
