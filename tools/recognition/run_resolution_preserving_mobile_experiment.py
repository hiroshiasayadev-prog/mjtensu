from __future__ import annotations

"""Run PRODUCT-INV-RECOGNITION-012 end to end.

INV-012 isolates two architectural factors after the INV-011 live regression:
late feature-map resolution (8x8 vs 4x4) and terminal same-resolution depth (1..3
96-channel MobileNetV3 blocks).  Training/dense-angle/export settings stay aligned
with INV-011, while a deterministic crop-perturbation proxy and confusion analysis
are added so promotion is not decided from dense-angle accuracy alone.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from resolution_preserving_mobile_models import (
        build_resolution_preserving_mobile_classifier,
        describe_resolution_preserving_mobile_classifier,
    )
    from run_mobile_classifier_experiment import (
        analyze_mobile_onnx_graph,
        load_mobile_checkpoint_model,
        smoke_onnx_dynamic_batch,
    )
    from run_rotation_classifier_experiment import (
        CHECKPOINT_ANGLES,
        DENSE_ANGLES,
        angle_key,
        append_json_line,
        assert_v3_contract,
        atomic_write_json,
        benchmark_onnx_cpu,
        compare_pytorch_onnx,
        configure_cuda,
        dense_evaluation,
        environment_info,
        evaluate_angles_with_oom_fallback,
        export_custom_model,
        fetch_batch,
        load_cache,
        load_checkpoint_model,
        read_json_if_exists,
        read_last_json_line,
        save_checkpoint,
        seed_everything,
        train_one_epoch,
    )
except ModuleNotFoundError:  # package-style import used by tests
    from tools.recognition.resolution_preserving_mobile_models import (
        build_resolution_preserving_mobile_classifier,
        describe_resolution_preserving_mobile_classifier,
    )
    from tools.recognition.run_mobile_classifier_experiment import (
        analyze_mobile_onnx_graph,
        load_mobile_checkpoint_model,
        smoke_onnx_dynamic_batch,
    )
    from tools.recognition.run_rotation_classifier_experiment import (
        CHECKPOINT_ANGLES,
        DENSE_ANGLES,
        angle_key,
        append_json_line,
        assert_v3_contract,
        atomic_write_json,
        benchmark_onnx_cpu,
        compare_pytorch_onnx,
        configure_cuda,
        dense_evaluation,
        environment_info,
        evaluate_angles_with_oom_fallback,
        export_custom_model,
        fetch_batch,
        load_cache,
        load_checkpoint_model,
        read_json_if_exists,
        read_last_json_line,
        save_checkpoint,
        seed_everything,
        train_one_epoch,
    )


EXPERIMENT_IMPLEMENTATION_VERSION = "inv012-resolution-preserving-mobile-v1"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 150
DEFAULT_EFFECTIVE_BATCH = 512
DEFAULT_EVAL_BATCH = 256
DEFAULT_LR = 1.0e-3
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_OPSET = 16

PLAIN_E150_ACCURACY_REFERENCE = {
    "manual_mean_accuracy": 0.94743,
    "manual_worst_accuracy": 0.94000,
    "jp_mean_accuracy": 0.99959,
    "jp_worst_accuracy": 0.99794,
    "source": "PRODUCT-INV-RECOGNITION-008",
}
STANDARD_MOBILE_ACCURACY_REFERENCE = {
    "manual_mean_accuracy": 0.9487153,
    "manual_worst_accuracy": 0.9311111,
    "jp_mean_accuracy": 0.9994784,
    "jp_worst_accuracy": 0.9989706,
    "source": "PRODUCT-INV-RECOGNITION-011",
}


@dataclass(frozen=True)
class Condition:
    name: str
    final_feature_resolution: int
    late_repeats: int


CONDITIONS: tuple[Condition, ...] = tuple(
    Condition(
        name=f"mobile-tile-f{resolution}-r{repeats}",
        final_feature_resolution=resolution,
        late_repeats=repeats,
    )
    for resolution in (8, 4)
    for repeats in (1, 2, 3)
)


@dataclass(frozen=True)
class RobustnessPerturbation:
    name: str
    shift_x_px: float = 0.0
    shift_y_px: float = 0.0
    content_scale: float = 1.0
    blur_kernel: int = 1


ROBUSTNESS_PERTURBATIONS: tuple[RobustnessPerturbation, ...] = (
    RobustnessPerturbation("identity"),
    RobustnessPerturbation("shift-x-minus-2px", shift_x_px=-2.0),
    RobustnessPerturbation("shift-x-plus-2px", shift_x_px=2.0),
    RobustnessPerturbation("shift-y-minus-2px", shift_y_px=-2.0),
    RobustnessPerturbation("shift-y-plus-2px", shift_y_px=2.0),
    RobustnessPerturbation("scale-0p94", content_scale=0.94),
    RobustnessPerturbation("scale-1p06", content_scale=1.06),
    RobustnessPerturbation("blur-3x3", blur_kernel=3),
)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run INV-012 resolution-preserving MobileNet tile-classifier comparison."
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--database",
        type=Path,
        default=(
            repository_root
            / ".local"
            / "recognition"
            / "tile_classifier_datasets"
            / "gray35_jp500_seed42_v3_jp189.sqlite"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            repository_root
            / ".local"
            / "recognition"
            / "resolution_preserving_mobile_experiment"
        ),
    )
    parser.add_argument(
        "--plain-reference-checkpoint",
        type=Path,
        default=(
            repository_root
            / ".local"
            / "recognition"
            / "plain_random360_epoch_sweep"
            / "e150"
            / "plain-random360"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--standard-mobile-reference-checkpoint",
        type=Path,
        default=(
            repository_root
            / ".local"
            / "recognition"
            / "mobile_classifier_experiment"
            / "mobilenet-v3-small-1.0x"
            / "training"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--plain-reference-onnx",
        type=Path,
        default=(
            repository_root
            / "vendor"
            / "recognition-models"
            / "tile-plain-gray35-random360-e150.onnx"
        ),
    )
    parser.add_argument(
        "--standard-mobile-reference-onnx",
        type=Path,
        default=(
            repository_root
            / "vendor"
            / "recognition-models"
            / "tile-mobilenet-v3-small-1.0x-random360-e150.onnx"
        ),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--effective-batch-size", type=int, default=DEFAULT_EFFECTIVE_BATCH)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH)
    parser.add_argument("--robustness-batch-size", type=int, default=DEFAULT_EVAL_BATCH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--angle-eval-every", type=int, default=5)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--benchmark-batch-size", type=int, default=16)
    parser.add_argument("--benchmark-warmup", type=int, default=100)
    parser.add_argument("--benchmark-runs", type=int, default=1000)
    parser.add_argument(
        "--cache-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=[condition.name for condition in CONDITIONS],
        help="Optional subset. Default: all six INV-012 candidates.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--overwrite-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def selected_conditions(args: argparse.Namespace) -> list[Condition]:
    if not args.conditions:
        return list(CONDITIONS)
    requested = set(str(value) for value in args.conditions)
    return [condition for condition in CONDITIONS if condition.name in requested]


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.effective_batch_size < 2 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.robustness_batch_size < 1:
        raise ValueError("--robustness-batch-size must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("invalid optimizer settings")
    if args.angle_eval_every < 1:
        raise ValueError("--angle-eval-every must be positive")
    if args.opset < 16:
        raise ValueError("INV-012 requires ONNX opset >= 16")
    if args.benchmark_batch_size < 1 or args.benchmark_warmup < 0 or args.benchmark_runs < 1:
        raise ValueError("invalid benchmark settings")


def require_runtime_dependencies() -> None:
    missing: list[str] = []
    for package_name in ("onnx", "onnxruntime"):
        try:
            __import__(package_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        raise RuntimeError(
            "Missing INV-012 dependencies: "
            + ", ".join(missing)
            + ". Install them into the existing classifier CUDA environment."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_runtime_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for INV-012 training/evaluation")

    repository_root = args.repository_root.resolve()
    database = args.database.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configure_cuda(tf32=not bool(args.no_tf32))
    seed_everything(int(args.seed))
    cache = load_cache(database, cache_device=str(args.cache_device))
    assert_v3_contract(cache)
    conditions = selected_conditions(args)

    baselines = build_baselines(args, cache=cache)
    atomic_write_json(output_root / "baselines.json", baselines)
    atomic_write_json(
        output_root / "manifest.json",
        {
            "status": "in_progress",
            "investigation": "PRODUCT-INV-RECOGNITION-012",
            "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
            "repository_root": str(repository_root),
            "database": str(database),
            "output_root": str(output_root),
            "conditions": [asdict(condition) for condition in conditions],
            "baselines": {
                name: {
                    "checkpoint": baseline.get("checkpoint"),
                    "onnx": baseline.get("onnx"),
                    "evaluation_status": baseline.get("evaluation_status"),
                }
                for name, baseline in baselines.items()
            },
            "training": {
                "epochs": int(args.epochs),
                "effective_batch_size": int(args.effective_batch_size),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "augmentation": "random360",
                "seed": int(args.seed),
                "amp": not bool(args.no_amp),
                "tf32": not bool(args.no_tf32),
                "checkpoint_angles": list(CHECKPOINT_ANGLES),
                "dense_angles": list(DENSE_ANGLES),
            },
            "robustness_proxy": {
                "split": "manual_val",
                "perturbations": [asdict(value) for value in ROBUSTNESS_PERTURBATIONS],
                "limitation": (
                    "Deterministic perturbations are applied to cached classifier crops. "
                    "They approximate detector-crop variation but are not a replacement for "
                    "a reviewed real detector-crop holdout."
                ),
            },
            "dataset": {
                "image_size": cache.image_size,
                "class_labels": list(cache.class_labels),
                "normalization": {"mean": cache.mean, "std": cache.std},
                "splits": {name: split.count for name, split in cache.splits.items()},
                "cache_device": cache.cache_device,
            },
            "environment": environment_info(),
        },
    )

    results: dict[str, Any] = {}
    for condition in conditions:
        run_dir = output_root / condition.name
        result_path = run_dir / "result.json"
        prior = read_json_if_exists(result_path)
        if prior_result_is_reusable(condition, prior) and not bool(args.overwrite_completed):
            print(f"[resume] skip completed {condition.name}", flush=True)
            results[condition.name] = prior
            write_summary(output_root, conditions, results, baselines=baselines)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(f"\n===== {condition.name} =====", flush=True)
        try:
            result = run_condition(condition, run_dir=run_dir, cache=cache, args=args)
            result["status"] = "completed"
            result["implementation_version"] = EXPERIMENT_IMPLEMENTATION_VERSION
            result["elapsed_seconds"] = time.perf_counter() - started
        except Exception as error:
            result = {
                "status": "failed",
                "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
                "condition": asdict(condition),
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            print(result["traceback"], file=sys.stderr, flush=True)
        atomic_write_json(result_path, result)
        results[condition.name] = result
        write_summary(output_root, conditions, results, baselines=baselines)
        if result["status"] != "completed" and bool(args.fail_fast):
            raise RuntimeError(f"Condition failed: {condition.name}: {result.get('error')}")

    summary = write_summary(
        output_root,
        conditions,
        results,
        baselines=baselines,
        final=True,
    )
    manifest = read_json_if_exists(output_root / "manifest.json") or {}
    manifest["status"] = summary["status"]
    atomic_write_json(output_root / "manifest.json", manifest)
    print("\n===== INV-012 experiment finished =====", flush=True)
    print(json.dumps(summary["status_counts"], ensure_ascii=False), flush=True)
    print(f"summary: {output_root / 'summary.json'}", flush=True)


def prior_result_is_reusable(
    condition: Condition,
    prior: dict[str, Any] | None,
) -> bool:
    return bool(
        prior is not None
        and prior.get("status") == "completed"
        and prior.get("implementation_version") == EXPERIMENT_IMPLEMENTATION_VERSION
        and prior.get("condition") == asdict(condition)
    )


def run_condition(
    condition: Condition,
    *,
    run_dir: Path,
    cache: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    recovered = recover_completed_training(
        condition,
        run_dir=run_dir,
        cache=cache,
        args=args,
    )
    if recovered is None:
        checkpoint_path, training, model = train_with_oom_fallback(
            condition,
            run_dir=run_dir,
            cache=cache,
            args=args,
        )
    else:
        checkpoint_path, training, model = recovered

    dense_accuracy = dense_evaluation(
        model,
        cache,
        batch_size=int(args.eval_batch_size),
        angles=DENSE_ANGLES,
        amp=False,
    )
    atomic_write_json(run_dir / "dense_evaluation.json", dense_accuracy)
    confusion = confusion_analysis_from_dense(
        dense_accuracy,
        class_labels=cache.class_labels,
        split_name="manual_val",
    )
    atomic_write_json(run_dir / "confusion_analysis.json", confusion)

    robustness = evaluate_robustness_with_oom_fallback(
        model,
        cache.splits["manual_val"],
        class_labels=cache.class_labels,
        batch_size=int(args.robustness_batch_size),
        mean=cache.mean,
        std=cache.std,
    )
    atomic_write_json(run_dir / "robustness_proxy.json", robustness)

    deployment = deploy_and_benchmark(
        condition,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        cache=cache,
        args=args,
    )
    return {
        "condition": asdict(condition),
        "training": training,
        "accuracy": dense_accuracy,
        "confusion": confusion,
        "robustness": robustness,
        "deployment": deployment,
    }


def train_with_oom_fallback(
    condition: Condition,
    *,
    run_dir: Path,
    cache: Any,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], nn.Module]:
    effective_batch = int(args.effective_batch_size)
    candidates: list[int] = []
    value = effective_batch
    while value >= 16:
        candidates.append(value)
        value //= 2
    last_error: str | None = None
    for microbatch in candidates:
        training_dir = run_dir / "training"
        try:
            if training_dir.exists():
                shutil.rmtree(training_dir)
            training_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[train] {condition.name} effective_batch={effective_batch} "
                f"microbatch={microbatch}",
                flush=True,
            )
            return train_condition(
                condition,
                output_dir=training_dir,
                cache=cache,
                args=args,
                microbatch=microbatch,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower():
                raise
            last_error = str(error)
            print(
                f"[oom] {condition.name} microbatch={microbatch}; retry smaller batch",
                file=sys.stderr,
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(
        f"{condition.name} could not train at any physical microbatch: {last_error}"
    )


def train_condition(
    condition: Condition,
    *,
    output_dir: Path,
    cache: Any,
    args: argparse.Namespace,
    microbatch: int,
) -> tuple[Path, dict[str, Any], nn.Module]:
    seed_everything(int(args.seed))
    device = torch.device("cuda")
    model = build_resolution_preserving_mobile_classifier(
        condition.name,
        class_count=len(cache.class_labels),
    ).to(device)
    description = describe_resolution_preserving_mobile_classifier(model, condition.name)
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
    amp = not bool(args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history_path = output_dir / "history.jsonl"
    best_path = output_dir / "best.pt"
    best_score = -1.0
    best_epoch = 0
    started = time.perf_counter()

    config = {
        "condition": asdict(condition),
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "database": str(args.database.resolve()),
        "model": {
            "name": description.name,
            "family": description.family,
            "parameter_count": description.parameter_count,
            "trainable_parameter_count": description.trainable_parameter_count,
            **description.details,
        },
        "image_size": cache.image_size,
        "class_labels": list(cache.class_labels),
        "normalization": {"mean": cache.mean, "std": cache.std},
        "epochs": int(args.epochs),
        "effective_batch_size": int(args.effective_batch_size),
        "physical_microbatch": int(microbatch),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "augmentation": "random360",
        "seed": int(args.seed),
        "amp": amp,
        "tf32": not bool(args.no_tf32),
        "checkpoint_angles": list(CHECKPOINT_ANGLES),
    }
    atomic_write_json(output_dir / "config.json", config)

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(
            model,
            cache.splits["train"],
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cache=cache,
            effective_batch_size=int(args.effective_batch_size),
            microbatch=microbatch,
            augmentation="random360",
            epoch=epoch,
            seed=int(args.seed),
            amp=amp,
        )
        full_sweep = (
            epoch == 1
            or epoch == int(args.epochs)
            or epoch % int(args.angle_eval_every) == 0
        )
        angles = CHECKPOINT_ANGLES if full_sweep else (0.0,)
        manual_validation = evaluate_angles_with_oom_fallback(
            model,
            cache.splits["manual_val"],
            angles=angles,
            batch_size=int(args.eval_batch_size),
            mean=cache.mean,
            std=cache.std,
            amp=False,
        )
        score = None
        if full_sweep:
            score = float(
                np.mean(
                    [
                        manual_validation[angle_key(angle)]["accuracy"]
                        for angle in CHECKPOINT_ANGLES
                    ]
                )
            )
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "manual_validation": manual_validation,
            "checkpoint_score": score,
        }
        append_json_line(history_path, record)
        scheduler.step()
        angle_text = " ".join(
            f"manual@{key}={value['accuracy']:.5f}"
            for key, value in manual_validation.items()
        )
        print(
            f"epoch={epoch:03d} loss={train_metrics['loss']:.5f} "
            f"acc={train_metrics['accuracy']:.5f} "
            f"samples/s={train_metrics['samples_per_second']:.1f} {angle_text}",
            flush=True,
        )
        if score is not None and score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                best_path,
                model=model,
                epoch=epoch,
                config=config,
                metrics=record,
            )

    if not best_path.is_file():
        raise RuntimeError(f"No best checkpoint produced for {condition.name}")
    best_model = load_candidate_checkpoint_model(
        best_path,
        model_name=condition.name,
        class_count=len(cache.class_labels),
        device=device,
    )
    result = {
        "kind": "trained",
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_checkpoint_score": best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "effective_batch_size": int(args.effective_batch_size),
        "physical_microbatch": int(microbatch),
        "parameter_count": description.parameter_count,
        "trainable_parameter_count": description.trainable_parameter_count,
        "model_details": description.details,
    }
    atomic_write_json(output_dir / "summary.json", result)
    return best_path, result, best_model


def recover_completed_training(
    condition: Condition,
    *,
    run_dir: Path,
    cache: Any,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], nn.Module] | None:
    training_dir = run_dir / "training"
    best_path = training_dir / "best.pt"
    config_path = training_dir / "config.json"
    history_path = training_dir / "history.jsonl"
    if not (best_path.is_file() and config_path.is_file() and history_path.is_file()):
        return None
    config = read_json_if_exists(config_path)
    last_record = read_last_json_line(history_path)
    if config is None or last_record is None:
        return None
    expected = {
        "condition": asdict(condition),
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "database": str(args.database.resolve()),
        "image_size": int(cache.image_size),
        "class_labels": list(cache.class_labels),
        "epochs": int(args.epochs),
        "effective_batch_size": int(args.effective_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "augmentation": "random360",
        "seed": int(args.seed),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            return None
    if config.get("normalization") != {"mean": cache.mean, "std": cache.std}:
        return None
    if int(last_record.get("epoch", -1)) != int(args.epochs):
        return None

    checkpoint = torch.load(best_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return None
    metrics = checkpoint.get("metrics")
    best_score = None
    if isinstance(metrics, dict) and metrics.get("checkpoint_score") is not None:
        best_score = float(metrics["checkpoint_score"])
    model = load_candidate_checkpoint_model(
        best_path,
        model_name=condition.name,
        class_count=len(cache.class_labels),
        device="cuda",
    )
    model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    result = {
        "kind": "recovered_completed_training",
        "checkpoint": str(best_path),
        "best_epoch": int(checkpoint.get("epoch", 0)),
        "best_checkpoint_score": best_score,
        "effective_batch_size": int(config["effective_batch_size"]),
        "physical_microbatch": int(config.get("physical_microbatch", 0)),
        "parameter_count": model_config.get("parameter_count"),
        "trainable_parameter_count": model_config.get("trainable_parameter_count"),
        "model_details": model_config,
    }
    atomic_write_json(training_dir / "summary.json", result)
    print(f"[resume] recovered completed training for {condition.name}", flush=True)
    return best_path, result, model


def load_candidate_checkpoint_model(
    checkpoint_path: Path,
    *,
    model_name: str,
    class_count: int,
    device: str | torch.device = "cpu",
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state_dict"), dict
    ):
        raise ValueError(f"Invalid INV-012 checkpoint: {checkpoint_path}")
    model = build_resolution_preserving_mobile_classifier(
        model_name,
        class_count=class_count,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(torch.device(device))
    model.eval()
    return model


def apply_robustness_perturbation(
    images: torch.Tensor,
    perturbation: RobustnessPerturbation,
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(f"Expected [N,1,H,W], got {tuple(images.shape)}")
    output = images
    if (
        abs(perturbation.shift_x_px) > 1.0e-12
        or abs(perturbation.shift_y_px) > 1.0e-12
        or abs(perturbation.content_scale - 1.0) > 1.0e-12
    ):
        batch, _channels, height, width = output.shape
        scale = float(perturbation.content_scale)
        if scale <= 0.0:
            raise ValueError("content_scale must be positive")
        theta = torch.zeros((batch, 2, 3), device=output.device, dtype=torch.float32)
        inverse_scale = 1.0 / scale
        theta[:, 0, 0] = inverse_scale
        theta[:, 1, 1] = inverse_scale
        theta[:, 0, 2] = -2.0 * float(perturbation.shift_x_px) / (float(width) * scale)
        theta[:, 1, 2] = -2.0 * float(perturbation.shift_y_px) / (float(height) * scale)
        grid = F.affine_grid(theta, output.shape, align_corners=False)
        output = F.grid_sample(
            output,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
    if perturbation.blur_kernel > 1:
        kernel = int(perturbation.blur_kernel)
        if kernel % 2 != 1:
            raise ValueError("blur_kernel must be odd")
        pad = kernel // 2
        output = F.avg_pool2d(
            F.pad(output, (pad, pad, pad, pad), mode="replicate"),
            kernel_size=kernel,
            stride=1,
        )
    return output


def evaluate_robustness_with_oom_fallback(
    model: nn.Module,
    split: Any,
    *,
    class_labels: Sequence[str],
    batch_size: int,
    mean: float,
    std: float,
) -> dict[str, Any]:
    candidate = int(batch_size)
    while candidate >= 8:
        try:
            return evaluate_robustness(
                model,
                split,
                class_labels=class_labels,
                batch_size=candidate,
                mean=mean,
                std=std,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower():
                raise
            gc.collect()
            torch.cuda.empty_cache()
            candidate //= 2
    raise RuntimeError(f"Robustness evaluation for {split.name} does not fit at batch=8")


def evaluate_robustness(
    model: nn.Module,
    split: Any,
    *,
    class_labels: Sequence[str],
    batch_size: int,
    mean: float,
    std: float,
) -> dict[str, Any]:
    model.eval()
    device = torch.device("cuda")
    perturbation_results: dict[str, Any] = {}
    with torch.inference_mode():
        for perturbation in ROBUSTNESS_PERTURBATIONS:
            confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
            total_count = 0
            for start in range(0, split.count, batch_size):
                indices = np.arange(start, min(split.count, start + batch_size), dtype=np.int64)
                images, targets = fetch_batch(split, indices, device=device)
                images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
                images = apply_robustness_perturbation(images, perturbation)
                images = images.sub(mean).div(std)
                logits = model(images)
                predictions = logits.argmax(dim=1).detach().cpu().numpy()
                target_np = targets.detach().cpu().numpy()
                np.add.at(confusion, (target_np, predictions), 1)
                total_count += int(targets.shape[0])
            correct = int(np.trace(confusion))
            perturbation_results[perturbation.name] = {
                "perturbation": asdict(perturbation),
                "count": total_count,
                "correct": correct,
                "errors": total_count - correct,
                "accuracy": correct / max(total_count, 1),
                "confusion": summarize_confusion(confusion, class_labels),
            }

    accuracies = [float(value["accuracy"]) for value in perturbation_results.values()]
    names = list(perturbation_results)
    worst_index = int(np.argmin(accuracies))
    return {
        "split": split.name,
        "proxy_kind": "deterministic_cached-crop_geometry_and_blur",
        "limitation": (
            "This applies bounded perturbations after the cached classifier-crop boundary. "
            "It measures sensitivity to plausible geometry/resampling changes but does not "
            "reproduce the complete detector->crop extraction distribution."
        ),
        "batch_size": batch_size,
        "conditions": perturbation_results,
        "summary": {
            "mean_accuracy": float(np.mean(accuracies)),
            "worst_accuracy": accuracies[worst_index],
            "worst_condition": names[worst_index],
            "identity_accuracy": float(perturbation_results["identity"]["accuracy"]),
        },
    }


def confusion_analysis_from_dense(
    dense_accuracy: dict[str, Any],
    *,
    class_labels: Sequence[str],
    split_name: str,
) -> dict[str, Any]:
    split = dense_accuracy["splits"][split_name]
    zero = split["angles"][angle_key(0.0)]
    confusion = np.asarray(zero["confusion_matrix"], dtype=np.int64)
    return {
        "split": split_name,
        "angle_deg": 0.0,
        **summarize_confusion(confusion, class_labels),
    }


def summarize_confusion(
    confusion: np.ndarray,
    class_labels: Sequence[str],
) -> dict[str, Any]:
    labels = tuple(str(value) for value in class_labels)
    if confusion.shape != (len(labels), len(labels)):
        raise ValueError(
            f"Confusion shape {confusion.shape} does not match {len(labels)} labels"
        )
    label_to_index = {label: index for index, label in enumerate(labels)}

    focus_labels = tuple(label for label in ("2m", "6m", "7m") if label in label_to_index)
    focus_pairs: list[dict[str, Any]] = []
    for true_label in focus_labels:
        true_index = label_to_index[true_label]
        true_total = int(confusion[true_index].sum())
        for predicted_label in focus_labels:
            if predicted_label == true_label:
                continue
            predicted_index = label_to_index[predicted_label]
            count = int(confusion[true_index, predicted_index])
            focus_pairs.append(
                {
                    "true": true_label,
                    "predicted": predicted_label,
                    "count": count,
                    "rate_given_true": count / max(true_total, 1),
                }
            )

    within_suit: list[dict[str, Any]] = []
    for true_index, true_label in enumerate(labels):
        if not _is_numbered_suit_label(true_label):
            continue
        true_total = int(confusion[true_index].sum())
        for predicted_index, predicted_label in enumerate(labels):
            if true_label == predicted_label:
                continue
            if not _is_numbered_suit_label(predicted_label):
                continue
            if true_label[-1] != predicted_label[-1]:
                continue
            count = int(confusion[true_index, predicted_index])
            if count <= 0:
                continue
            within_suit.append(
                {
                    "true": true_label,
                    "predicted": predicted_label,
                    "count": count,
                    "rate_given_true": count / max(true_total, 1),
                }
            )
    within_suit.sort(
        key=lambda value: (int(value["count"]), float(value["rate_given_true"])),
        reverse=True,
    )

    invalid = None
    if "invalid" in label_to_index:
        invalid_index = label_to_index["invalid"]
        invalid_total = int(confusion[invalid_index].sum())
        invalid_to_tile = int(confusion[invalid_index].sum() - confusion[invalid_index, invalid_index])
        tile_to_invalid = int(confusion[:, invalid_index].sum() - confusion[invalid_index, invalid_index])
        tile_total = int(confusion.sum() - invalid_total)
        invalid = {
            "invalid_to_tile_count": invalid_to_tile,
            "invalid_to_tile_rate": invalid_to_tile / max(invalid_total, 1),
            "tile_to_invalid_count": tile_to_invalid,
            "tile_to_invalid_rate": tile_to_invalid / max(tile_total, 1),
        }

    return {
        "labels": list(labels),
        "focus_2m_6m_7m": focus_pairs,
        "worst_within_suit_pairs": within_suit[:12],
        "invalid_background": invalid,
    }


def _is_numbered_suit_label(label: str) -> bool:
    return len(label) == 2 and label[0] in "123456789" and label[1] in "mps"


def deploy_and_benchmark(
    condition: Condition,
    *,
    checkpoint_path: Path,
    run_dir: Path,
    cache: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    deployment_dir = run_dir / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = deployment_dir / f"{condition.name}.onnx"
    cpu_model = load_candidate_checkpoint_model(
        checkpoint_path,
        model_name=condition.name,
        class_count=len(cache.class_labels),
        device="cpu",
    )
    description = describe_resolution_preserving_mobile_classifier(cpu_model, condition.name)
    export_info = export_custom_model(
        cpu_model,
        onnx_path=onnx_path,
        image_size=cache.image_size,
        opset=int(args.opset),
    )
    parity = compare_pytorch_onnx(
        cpu_model,
        onnx_path,
        cache=cache,
        sample_count=16,
        seed=int(args.seed),
    )
    if parity["prediction_mismatches"] != 0 or not parity["allclose"]:
        raise RuntimeError(f"ONNX parity failed for {condition.name}: {parity}")

    dynamic_batch_smoke = smoke_onnx_dynamic_batch(
        onnx_path,
        image_size=cache.image_size,
        batch_sizes=(1, 16, 24),
        seed=int(args.seed),
    )
    graph = analyze_mobile_onnx_graph(
        onnx_path,
        batch_size=int(args.benchmark_batch_size),
    )
    graph["final_feature_resolution"] = description.final_feature_resolution
    graph["final_feature_channels"] = description.details["last_conv_channels"]
    graph["late_repeats"] = description.late_repeats
    benchmark = benchmark_onnx_cpu(
        onnx_path,
        batch_size=int(args.benchmark_batch_size),
        image_size=cache.image_size,
        warmup=int(args.benchmark_warmup),
        runs=int(args.benchmark_runs),
        seed=int(args.seed),
    )
    deployment = {
        "status": "completed",
        "onnx": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "export": export_info,
        "parity": parity,
        "dynamic_batch_smoke": dynamic_batch_smoke,
        "graph": graph,
        "benchmark": benchmark,
    }
    atomic_write_json(deployment_dir / "summary.json", deployment)
    return deployment


def build_baselines(args: argparse.Namespace, *, cache: Any) -> dict[str, Any]:
    baselines = {
        "plain-random360-e150": build_one_baseline(
            name="plain-random360-e150",
            accuracy_reference=PLAIN_E150_ACCURACY_REFERENCE,
            checkpoint=args.plain_reference_checkpoint.resolve(),
            onnx_path=args.plain_reference_onnx.resolve(),
            load_model=lambda path: load_checkpoint_model(
                path,
                architecture="plain",
                class_count=len(cache.class_labels),
                image_size=cache.image_size,
                device="cuda",
            ),
            cache=cache,
            args=args,
        ),
        "mobilenet-v3-small-1.0x-standard": build_one_baseline(
            name="mobilenet-v3-small-1.0x-standard",
            accuracy_reference=STANDARD_MOBILE_ACCURACY_REFERENCE,
            checkpoint=args.standard_mobile_reference_checkpoint.resolve(),
            onnx_path=args.standard_mobile_reference_onnx.resolve(),
            load_model=lambda path: load_mobile_checkpoint_model(
                path,
                model_name="mobilenet-v3-small-1.0x",
                class_count=len(cache.class_labels),
                device="cuda",
            ),
            cache=cache,
            args=args,
        ),
    }
    return baselines


def build_one_baseline(
    *,
    name: str,
    accuracy_reference: dict[str, Any],
    checkpoint: Path,
    onnx_path: Path,
    load_model: Any,
    cache: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "accuracy_reference": dict(accuracy_reference),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "onnx": str(onnx_path),
        "onnx_exists": onnx_path.is_file(),
    }
    if checkpoint.is_file():
        model = load_model(checkpoint)
        try:
            zero = evaluate_angles_with_oom_fallback(
                model,
                cache.splits["manual_val"],
                angles=(0.0,),
                batch_size=int(args.eval_batch_size),
                mean=cache.mean,
                std=cache.std,
                amp=False,
            )
            confusion = np.asarray(
                zero[angle_key(0.0)]["confusion_matrix"], dtype=np.int64
            )
            robustness = evaluate_robustness_with_oom_fallback(
                model,
                cache.splits["manual_val"],
                class_labels=cache.class_labels,
                batch_size=int(args.robustness_batch_size),
                mean=cache.mean,
                std=cache.std,
            )
            result["evaluation_status"] = "completed"
            result["zero_degree_accuracy"] = float(zero[angle_key(0.0)]["accuracy"])
            result["confusion"] = summarize_confusion(confusion, cache.class_labels)
            result["robustness"] = robustness
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    else:
        result["evaluation_status"] = "checkpoint_unavailable"

    if onnx_path.is_file():
        try:
            graph = analyze_mobile_onnx_graph(
                onnx_path,
                batch_size=int(args.benchmark_batch_size),
            )
            benchmark = benchmark_onnx_cpu(
                onnx_path,
                batch_size=int(args.benchmark_batch_size),
                image_size=cache.image_size,
                warmup=int(args.benchmark_warmup),
                runs=int(args.benchmark_runs),
                seed=int(args.seed),
            )
            result["deployment"] = {
                "status": "completed",
                "onnx_bytes": onnx_path.stat().st_size,
                "graph": graph,
                "benchmark": benchmark,
            }
        except Exception as error:
            result["deployment"] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    else:
        result["deployment"] = {"status": "onnx_unavailable"}
    return result


def write_summary(
    output_root: Path,
    conditions: Sequence[Condition],
    results: dict[str, Any],
    *,
    baselines: dict[str, Any],
    final: bool = False,
) -> dict[str, Any]:
    status_counts = Counter(
        str(result.get("status", "unknown")) for result in results.values()
    )
    pending = [condition.name for condition in conditions if condition.name not in results]
    rows: list[dict[str, Any]] = []

    for baseline_name, baseline in baselines.items():
        accuracy = baseline.get("accuracy_reference", {})
        robustness = baseline.get("robustness", {}).get("summary", {})
        deployment = baseline.get("deployment", {})
        graph = deployment.get("graph", {}) if isinstance(deployment, dict) else {}
        benchmark = deployment.get("benchmark", {}) if isinstance(deployment, dict) else {}
        rows.append(
            {
                "condition": baseline_name,
                "kind": "baseline",
                "final_feature_resolution": 8 if baseline_name.startswith("plain") else 2,
                "late_repeats": None,
                "manual_mean_accuracy": accuracy.get("manual_mean_accuracy"),
                "manual_worst_accuracy": accuracy.get("manual_worst_accuracy"),
                "robustness_mean_accuracy": robustness.get("mean_accuracy"),
                "robustness_worst_accuracy": robustness.get("worst_accuracy"),
                "robustness_worst_condition": robustness.get("worst_condition"),
                "onnx_bytes": deployment.get("onnx_bytes") if isinstance(deployment, dict) else None,
                "known_macs_per_sample_estimate": graph.get("known_macs_per_sample_estimate"),
                "ort_cpu_median_ms_batch": benchmark.get("median_ms_per_batch"),
            }
        )

    for condition in conditions:
        result = results.get(condition.name)
        if not result or result.get("status") != "completed":
            continue
        split = result.get("accuracy", {}).get("splits", {}).get("manual_val", {})
        dense = split.get("summary", {})
        robustness = result.get("robustness", {}).get("summary", {})
        deployment = result.get("deployment", {})
        graph = deployment.get("graph", {})
        benchmark = deployment.get("benchmark", {})
        rows.append(
            {
                "condition": condition.name,
                "kind": "candidate",
                "final_feature_resolution": condition.final_feature_resolution,
                "late_repeats": condition.late_repeats,
                "manual_mean_accuracy": dense.get("mean_accuracy"),
                "manual_worst_accuracy": dense.get("worst_accuracy"),
                "robustness_mean_accuracy": robustness.get("mean_accuracy"),
                "robustness_worst_accuracy": robustness.get("worst_accuracy"),
                "robustness_worst_condition": robustness.get("worst_condition"),
                "onnx_bytes": deployment.get("onnx_bytes"),
                "known_macs_per_sample_estimate": graph.get("known_macs_per_sample_estimate"),
                "depthwise_conv_count": graph.get("depthwise_conv_count"),
                "pointwise_conv_count": graph.get("pointwise_conv_count"),
                "ort_cpu_median_ms_batch": benchmark.get("median_ms_per_batch"),
            }
        )

    status = "completed" if final and not pending else "in_progress"
    if final and status_counts.get("failed", 0):
        status = "completed_with_failures"
    summary = {
        "status": status,
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "status_counts": dict(status_counts),
        "pending": pending,
        "comparison_rows": rows,
        "results": {name: result.get("status") for name, result in results.items()},
        "baseline_status": {
            name: baseline.get("evaluation_status") for name, baseline in baselines.items()
        },
    }
    atomic_write_json(output_root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    main()
