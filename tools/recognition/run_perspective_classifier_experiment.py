from __future__ import annotations

"""Run PRODUCT-INV-RECOGNITION-013 end to end.

INV-013 is a controlled 2x4 architecture/augmentation experiment:

    Plain  x A0/A1/A2/A3
    f8-r1  x A0/A1/A2/A3

A0 reuses the accepted 150-epoch random360 checkpoints by default when available.
A1/A2/A3 retrain each architecture under identical optimizer/data/checkpoint-selection
settings.  Every successful condition is evaluated on the historical dense in-plane
angle sweep, an independent deterministic perspective sweep, an optional reviewed real
crop SQLite holdout, and ONNX deployment/parity/CPU latency gates.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from perspective_classifier_augmentation import (
        AUGMENTATION_SPECS,
        PERSPECTIVE_EVALUATION_CASES,
        PerspectiveAugmentationSpec,
        apply_evaluation_case,
        apply_training_geometry,
        deterministic_geometry_units,
    )
    from resolution_preserving_mobile_models import (
        build_resolution_preserving_mobile_classifier,
        describe_resolution_preserving_mobile_classifier,
    )
    from rotation_classifier_experiment_models import (
        build_experiment_model,
        describe_experiment_model,
    )
    from run_mobile_classifier_experiment import (
        analyze_mobile_onnx_graph,
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
        deterministic_random360_angles,
        environment_info,
        evaluate_angles_with_oom_fallback,
        export_custom_model,
        fetch_batch,
        load_cache,
        read_json_if_exists,
        read_last_json_line,
        save_checkpoint,
        seed_everything,
    )
except ModuleNotFoundError:  # package-style imports used by tests
    from tools.recognition.perspective_classifier_augmentation import (
        AUGMENTATION_SPECS,
        PERSPECTIVE_EVALUATION_CASES,
        PerspectiveAugmentationSpec,
        apply_evaluation_case,
        apply_training_geometry,
        deterministic_geometry_units,
    )
    from tools.recognition.resolution_preserving_mobile_models import (
        build_resolution_preserving_mobile_classifier,
        describe_resolution_preserving_mobile_classifier,
    )
    from tools.recognition.rotation_classifier_experiment_models import (
        build_experiment_model,
        describe_experiment_model,
    )
    from tools.recognition.run_mobile_classifier_experiment import (
        analyze_mobile_onnx_graph,
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
        deterministic_random360_angles,
        environment_info,
        evaluate_angles_with_oom_fallback,
        export_custom_model,
        fetch_batch,
        load_cache,
        read_json_if_exists,
        read_last_json_line,
        save_checkpoint,
        seed_everything,
    )


EXPERIMENT_IMPLEMENTATION_VERSION = "inv013-perspective-augmentation-v1"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 150
DEFAULT_EFFECTIVE_BATCH = 512
DEFAULT_EVAL_BATCH = 256
DEFAULT_LR = 1.0e-3
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_OPSET = 16
ARCHITECTURES = ("plain", "f8-r1")
AUGMENTATIONS = (
    "a0-random360",
    "a1-anisotropic-affine",
    "a2-perspective",
    "a3-perspective-recrop",
)


@dataclass(frozen=True)
class Condition:
    name: str
    architecture: str
    augmentation: str


CONDITIONS: tuple[Condition, ...] = tuple(
    Condition(
        name=f"{architecture}-{augmentation}",
        architecture=architecture,
        augmentation=augmentation,
    )
    for architecture in ARCHITECTURES
    for augmentation in AUGMENTATIONS
)


@dataclass
class HoldoutSplit:
    name: str
    images_u8: torch.Tensor
    labels: torch.Tensor
    sample_ids: list[str]

    @property
    def count(self) -> int:
        return int(self.labels.shape[0])


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run INV-013 perspective-aware Plain vs f8-r1 classifier experiment."
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
            / "perspective_classifier_experiment"
        ),
    )
    parser.add_argument(
        "--plain-a0-checkpoint",
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
        "--f8-r1-a0-checkpoint",
        type=Path,
        default=(
            repository_root
            / ".local"
            / "recognition"
            / "resolution_preserving_mobile_experiment"
            / "mobile-tile-f8-r1"
            / "training"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--real-holdout-database",
        type=Path,
        help=(
            "Optional reviewed real detector-crop SQLite database. Expected sample rows "
            "contain sample_id, class_index and production-preprocessed 64x64 image_gray_u8."
        ),
    )
    parser.add_argument(
        "--real-holdout-split",
        type=str,
        help="Optional split name inside --real-holdout-database; default evaluates every row.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--effective-batch-size", type=int, default=DEFAULT_EFFECTIVE_BATCH)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH)
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
        help="Optional subset. Default: all eight Plain/f8-r1 x A0/A1/A2/A3 conditions.",
    )
    parser.add_argument(
        "--retrain-a0",
        action="store_true",
        help="Retrain A0 instead of reusing INV-008/INV-012 150-epoch checkpoints.",
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
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("invalid optimizer settings")
    if args.angle_eval_every < 1:
        raise ValueError("--angle-eval-every must be positive")
    if args.opset < 16:
        raise ValueError("INV-013 requires ONNX opset >= 16")
    if args.benchmark_batch_size < 1 or args.benchmark_warmup < 0 or args.benchmark_runs < 1:
        raise ValueError("invalid benchmark settings")
    if args.real_holdout_split and args.real_holdout_database is None:
        raise ValueError("--real-holdout-split requires --real-holdout-database")


def require_runtime_dependencies() -> None:
    missing: list[str] = []
    for package_name in ("onnx", "onnxruntime"):
        try:
            __import__(package_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        raise RuntimeError(
            "Missing INV-013 dependencies: "
            + ", ".join(missing)
            + ". Install them into the existing classifier CUDA environment."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_runtime_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for INV-013 training/evaluation")

    repository_root = args.repository_root.resolve()
    database = args.database.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configure_cuda(tf32=not bool(args.no_tf32))
    seed_everything(int(args.seed))
    cache = load_cache(database, cache_device=str(args.cache_device))
    assert_v3_contract(cache)
    conditions = selected_conditions(args)
    real_holdout = load_optional_real_holdout(args, cache=cache)

    manifest = {
        "status": "in_progress",
        "investigation": "PRODUCT-INV-RECOGNITION-013",
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "repository_root": str(repository_root),
        "database": str(database),
        "output_root": str(output_root),
        "conditions": [asdict(condition) for condition in conditions],
        "augmentation_specs": {
            name: asdict(AUGMENTATION_SPECS[name]) for name in AUGMENTATIONS
        },
        "perspective_evaluation_cases": [
            asdict(case) for case in PERSPECTIVE_EVALUATION_CASES
        ],
        "a0_reference_checkpoints": {
            "plain": str(args.plain_a0_checkpoint.resolve()),
            "f8-r1": str(args.f8_r1_a0_checkpoint.resolve()),
            "retrain_a0": bool(args.retrain_a0),
        },
        "training": {
            "epochs": int(args.epochs),
            "effective_batch_size": int(args.effective_batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "amp": not bool(args.no_amp),
            "tf32": not bool(args.no_tf32),
            "checkpoint_angles": list(CHECKPOINT_ANGLES),
            "dense_angles": list(DENSE_ANGLES),
            "geometry_stream": "inv013-shared-geometry",
        },
        "dataset": {
            "image_size": cache.image_size,
            "class_labels": list(cache.class_labels),
            "normalization": {"mean": cache.mean, "std": cache.std},
            "splits": {name: split.count for name, split in cache.splits.items()},
            "cache_device": cache.cache_device,
        },
        "real_holdout": real_holdout_manifest(args, real_holdout),
        "environment": environment_info(),
    }
    atomic_write_json(output_root / "manifest.json", manifest)

    results: dict[str, Any] = {}
    for condition in conditions:
        run_dir = output_root / condition.name
        result_path = run_dir / "result.json"
        prior = read_json_if_exists(result_path)
        if prior_result_is_reusable(condition, prior) and not bool(args.overwrite_completed):
            print(f"[resume] skip completed {condition.name}", flush=True)
            results[condition.name] = prior
            write_summary(output_root, conditions, results)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(f"\n===== {condition.name} =====", flush=True)
        try:
            result = run_condition(
                condition,
                run_dir=run_dir,
                cache=cache,
                real_holdout=real_holdout,
                args=args,
            )
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
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        atomic_write_json(result_path, result)
        results[condition.name] = result
        write_summary(output_root, conditions, results)
        if result["status"] != "completed" and bool(args.fail_fast):
            raise RuntimeError(f"Condition failed: {condition.name}: {result.get('error')}")

    summary = write_summary(output_root, conditions, results, final=True)
    manifest = read_json_if_exists(output_root / "manifest.json") or {}
    manifest["status"] = summary["status"]
    atomic_write_json(output_root / "manifest.json", manifest)
    print("\n===== INV-013 experiment finished =====", flush=True)
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
    real_holdout: HoldoutSplit | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reference = reusable_a0_checkpoint(condition, args, cache=cache)
    if reference is not None and not bool(args.retrain_a0):
        checkpoint_path = reference
        model = load_condition_checkpoint(
            checkpoint_path,
            architecture=condition.architecture,
            class_count=len(cache.class_labels),
            image_size=cache.image_size,
            device="cuda",
        )
        training: dict[str, Any] = {
            "kind": "existing_a0_reference",
            "checkpoint": str(checkpoint_path),
            "source_investigation": (
                "PRODUCT-INV-RECOGNITION-008"
                if condition.architecture == "plain"
                else "PRODUCT-INV-RECOGNITION-012"
            ),
        }
    else:
        recovered = recover_completed_training(
            condition, run_dir=run_dir, cache=cache, args=args
        )
        if recovered is None:
            checkpoint_path, training, model = train_with_oom_fallback(
                condition, run_dir=run_dir, cache=cache, args=args
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

    perspective = evaluate_perspective_with_oom_fallback(
        model,
        cache.splits["manual_val"],
        class_labels=cache.class_labels,
        batch_size=int(args.eval_batch_size),
        mean=cache.mean,
        std=cache.std,
    )
    atomic_write_json(run_dir / "perspective_evaluation.json", perspective)

    if real_holdout is None:
        holdout_result: dict[str, Any] = {"status": "not_provided"}
    else:
        holdout_result = evaluate_real_holdout_with_oom_fallback(
            model,
            real_holdout,
            class_labels=cache.class_labels,
            batch_size=int(args.eval_batch_size),
            mean=cache.mean,
            std=cache.std,
        )
    atomic_write_json(run_dir / "real_holdout_evaluation.json", holdout_result)

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
        "perspective": perspective,
        "real_holdout": holdout_result,
        "deployment": deployment,
    }


def reusable_a0_checkpoint(
    condition: Condition,
    args: argparse.Namespace,
    *,
    cache: Any,
) -> Path | None:
    if condition.augmentation != "a0-random360":
        return None
    path = (
        args.plain_a0_checkpoint.resolve()
        if condition.architecture == "plain"
        else args.f8_r1_a0_checkpoint.resolve()
    )
    if not path.is_file():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(checkpoint, dict):
        return None
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        return None
    recorded_augmentation = config.get("augmentation")
    if isinstance(recorded_augmentation, dict):
        recorded_augmentation = recorded_augmentation.get("name")
    expected = {
        "image_size": int(cache.image_size),
        "class_labels": list(cache.class_labels),
        "epochs": int(args.epochs),
        "effective_batch_size": int(args.effective_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            return None
    if recorded_augmentation != "random360":
        return None
    if config.get("normalization") != {"mean": cache.mean, "std": cache.std}:
        return None
    recorded_database = config.get("database")
    if recorded_database is not None:
        try:
            if Path(str(recorded_database)).resolve() != args.database.resolve():
                return None
        except OSError:
            return None
    return path


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
    model = build_condition_model(
        condition.architecture,
        class_count=len(cache.class_labels),
        image_size=cache.image_size,
    ).to(device)
    description = describe_condition_model(model, condition.architecture)
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
    spec = AUGMENTATION_SPECS[condition.augmentation]

    config = {
        "condition": asdict(condition),
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "database": str(args.database.resolve()),
        "model": description,
        "image_size": cache.image_size,
        "class_labels": list(cache.class_labels),
        "normalization": {"mean": cache.mean, "std": cache.std},
        "epochs": int(args.epochs),
        "effective_batch_size": int(args.effective_batch_size),
        "physical_microbatch": int(microbatch),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "augmentation": asdict(spec),
        "seed": int(args.seed),
        "amp": amp,
        "tf32": not bool(args.no_tf32),
        "checkpoint_angles": list(CHECKPOINT_ANGLES),
        "geometry_stream": "inv013-shared-geometry",
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
            spec=spec,
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
    best_model = load_condition_checkpoint(
        best_path,
        architecture=condition.architecture,
        class_count=len(cache.class_labels),
        image_size=cache.image_size,
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
        "model": description,
    }
    atomic_write_json(output_dir / "summary.json", result)
    return best_path, result, best_model


def train_one_epoch(
    model: nn.Module,
    split: Any,
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cache: Any,
    effective_batch_size: int,
    microbatch: int,
    spec: PerspectiveAugmentationSpec,
    epoch: int,
    seed: int,
    amp: bool,
) -> dict[str, Any]:
    model.train()
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    order = rng.permutation(split.count)
    angles = deterministic_random360_angles(split.sample_ids, seed=seed, epoch=epoch)
    units = (
        None
        if spec.name == "a0-random360"
        else deterministic_geometry_units(
            split.sample_ids,
            seed=seed,
            epoch=epoch,
            stream="inv013-shared-geometry",
        )
    )
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    optimizer_steps = 0
    torch.cuda.synchronize()
    started = time.perf_counter()

    for effective_start in range(0, split.count, effective_batch_size):
        effective_indices = order[effective_start : effective_start + effective_batch_size]
        effective_count = len(effective_indices)
        optimizer.zero_grad(set_to_none=True)
        for micro_start in range(0, effective_count, microbatch):
            batch_indices = effective_indices[micro_start : micro_start + microbatch]
            images, targets = fetch_batch(split, batch_indices, device=device)
            images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
            batch_angles = torch.from_numpy(angles[batch_indices]).to(
                device=device, dtype=torch.float32
            )
            if units is None:
                batch_units = torch.zeros(
                    (len(batch_indices), 12), device=device, dtype=torch.float32
                )
            else:
                batch_units = torch.from_numpy(units[batch_indices]).to(
                    device=device, dtype=torch.float32
                )
            images = apply_training_geometry(
                images,
                spec=spec,
                angles_deg=batch_angles,
                units=batch_units,
            )
            images = images.sub(cache.mean).div(cache.std)

            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
                raw_loss = F.cross_entropy(logits, targets)
                weighted_loss = raw_loss * (
                    float(len(batch_indices)) / float(effective_count)
                )
            scaler.scale(weighted_loss).backward()
            total_loss += float(raw_loss.detach().item()) * len(batch_indices)
            total_correct += int(
                (logits.detach().argmax(dim=1) == targets).sum().item()
            )
            total_count += len(batch_indices)
        scaler.step(optimizer)
        scaler.update()
        optimizer_steps += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "correct": total_correct,
        "count": total_count,
        "optimizer_steps": optimizer_steps,
        "seconds": elapsed,
        "samples_per_second": total_count / max(elapsed, 1.0e-9),
    }


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
        "augmentation": asdict(AUGMENTATION_SPECS[condition.augmentation]),
        "seed": int(args.seed),
        "geometry_stream": "inv013-shared-geometry",
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
    model = load_condition_checkpoint(
        best_path,
        architecture=condition.architecture,
        class_count=len(cache.class_labels),
        image_size=cache.image_size,
        device="cuda",
    )
    result = {
        "kind": "recovered_completed_training",
        "checkpoint": str(best_path),
        "best_epoch": int(checkpoint.get("epoch", 0)),
        "best_checkpoint_score": best_score,
        "effective_batch_size": int(config["effective_batch_size"]),
        "physical_microbatch": int(config.get("physical_microbatch", 0)),
        "model": config.get("model", {}),
    }
    atomic_write_json(training_dir / "summary.json", result)
    print(f"[resume] recovered completed training for {condition.name}", flush=True)
    return best_path, result, model


def build_condition_model(
    architecture: str,
    *,
    class_count: int,
    image_size: int,
) -> nn.Module:
    if architecture == "plain":
        return build_experiment_model(
            "plain", class_count=class_count, image_size=image_size
        )
    if architecture == "f8-r1":
        return build_resolution_preserving_mobile_classifier(
            "mobile-tile-f8-r1", class_count=class_count
        )
    raise ValueError(f"Unsupported INV-013 architecture: {architecture}")


def describe_condition_model(model: nn.Module, architecture: str) -> dict[str, Any]:
    if architecture == "plain":
        description = describe_experiment_model(model, "plain")
        return {
            "architecture": architecture,
            "family": "plain",
            "parameter_count": description.parameter_count,
            "trainable_parameter_count": description.trainable_parameter_count,
            **description.details,
        }
    if architecture == "f8-r1":
        description = describe_resolution_preserving_mobile_classifier(
            model, "mobile-tile-f8-r1"  # type: ignore[arg-type]
        )
        return {
            "architecture": architecture,
            "family": description.family,
            "parameter_count": description.parameter_count,
            "trainable_parameter_count": description.trainable_parameter_count,
            **description.details,
        }
    raise ValueError(f"Unsupported INV-013 architecture: {architecture}")


def load_condition_checkpoint(
    checkpoint_path: Path,
    *,
    architecture: str,
    class_count: int,
    image_size: int,
    device: str | torch.device,
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state_dict"), dict
    ):
        raise ValueError(f"Invalid INV-013 classifier checkpoint: {checkpoint_path}")
    model = build_condition_model(
        architecture, class_count=class_count, image_size=image_size
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(torch.device(device))
    model.eval()
    return model


def evaluate_perspective_with_oom_fallback(
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
            return evaluate_perspective(
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
    raise RuntimeError("Perspective evaluation does not fit at batch=8")


def evaluate_perspective(
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
    results: dict[str, Any] = {}
    with torch.inference_mode():
        for case in PERSPECTIVE_EVALUATION_CASES:
            confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
            total_loss = 0.0
            total_count = 0
            for start in range(0, split.count, batch_size):
                indices = np.arange(
                    start, min(split.count, start + batch_size), dtype=np.int64
                )
                images, targets = fetch_batch(split, indices, device=device)
                images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
                images = apply_evaluation_case(images, case)
                images = images.sub(mean).div(std)
                logits = model(images)
                loss = F.cross_entropy(logits, targets, reduction="sum")
                prediction = logits.argmax(dim=1).detach().cpu().numpy()
                target_np = targets.detach().cpu().numpy()
                np.add.at(confusion, (target_np, prediction), 1)
                total_loss += float(loss.detach().item())
                total_count += int(targets.shape[0])
            correct = int(np.trace(confusion))
            results[case.name] = {
                "case": asdict(case),
                "count": total_count,
                "correct": correct,
                "errors": total_count - correct,
                "loss": total_loss / max(total_count, 1),
                "accuracy": correct / max(total_count, 1),
                "confusion": summarize_confusion(confusion, class_labels),
            }

    names = list(results)
    accuracies = [float(results[name]["accuracy"]) for name in names]
    front = float(results["front-facing"]["accuracy"])
    oblique_names = [name for name in names if name != "front-facing"]
    oblique_accuracies = [float(results[name]["accuracy"]) for name in oblique_names]
    worst_index = int(np.argmin(accuracies))
    oblique_worst_index = int(np.argmin(oblique_accuracies))
    six_rates = [
        float(results[name]["confusion"]["six_m_to_5m_or_7m_rate"] or 0.0)
        for name in names
    ]
    six_worst_index = int(np.argmax(six_rates))
    return {
        "split": split.name,
        "batch_size": batch_size,
        "cases": results,
        "summary": {
            "mean_accuracy": float(np.mean(accuracies)),
            "worst_accuracy": accuracies[worst_index],
            "worst_case": names[worst_index],
            "front_accuracy": front,
            "oblique_mean_accuracy": float(np.mean(oblique_accuracies)),
            "oblique_worst_accuracy": oblique_accuracies[oblique_worst_index],
            "oblique_worst_case": oblique_names[oblique_worst_index],
            "front_to_oblique_mean_delta": front - float(np.mean(oblique_accuracies)),
            "six_m_to_5m_or_7m_worst_rate": six_rates[six_worst_index],
            "six_m_worst_case": names[six_worst_index],
        },
    }


def summarize_confusion(
    confusion: np.ndarray,
    class_labels: Sequence[str],
) -> dict[str, Any]:
    labels = tuple(str(value) for value in class_labels)
    if confusion.shape != (len(labels), len(labels)):
        raise ValueError("Confusion matrix does not match label count")
    label_to_index = {label: index for index, label in enumerate(labels)}
    focus = tuple(label for label in ("5m", "6m", "7m") if label in label_to_index)
    focus_pairs: list[dict[str, Any]] = []
    for true_label in focus:
        true_index = label_to_index[true_label]
        total = int(confusion[true_index].sum())
        for predicted_label in focus:
            if predicted_label == true_label:
                continue
            count = int(confusion[true_index, label_to_index[predicted_label]])
            focus_pairs.append(
                {
                    "true": true_label,
                    "predicted": predicted_label,
                    "count": count,
                    "rate_given_true": count / max(total, 1),
                }
            )

    within_suit: list[dict[str, Any]] = []
    for true_index, true_label in enumerate(labels):
        if not _is_numbered_suit_label(true_label):
            continue
        total = int(confusion[true_index].sum())
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
                    "rate_given_true": count / max(total, 1),
                }
            )
    within_suit.sort(
        key=lambda row: (int(row["count"]), float(row["rate_given_true"])),
        reverse=True,
    )

    six_count: int | None = None
    six_rate: float | None = None
    if all(label in label_to_index for label in ("5m", "6m", "7m")):
        six_index = label_to_index["6m"]
        six_total = int(confusion[six_index].sum())
        six_count = int(
            confusion[six_index, label_to_index["5m"]]
            + confusion[six_index, label_to_index["7m"]]
        )
        six_rate = six_count / max(six_total, 1)

    return {
        "labels": list(labels),
        "focus_5m_6m_7m": focus_pairs,
        "six_m_to_5m_or_7m_count": six_count,
        "six_m_to_5m_or_7m_rate": six_rate,
        "worst_within_suit_pairs": within_suit[:12],
    }


def _is_numbered_suit_label(label: str) -> bool:
    return len(label) == 2 and label[0] in "123456789" and label[1] in "mps"


def evaluate_real_holdout_with_oom_fallback(
    model: nn.Module,
    split: HoldoutSplit,
    *,
    class_labels: Sequence[str],
    batch_size: int,
    mean: float,
    std: float,
) -> dict[str, Any]:
    candidate = int(batch_size)
    while candidate >= 8:
        try:
            return evaluate_real_holdout(
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
    raise RuntimeError("Real-holdout evaluation does not fit at batch=8")


def evaluate_real_holdout(
    model: nn.Module,
    split: HoldoutSplit,
    *,
    class_labels: Sequence[str],
    batch_size: int,
    mean: float,
    std: float,
) -> dict[str, Any]:
    model.eval()
    device = torch.device("cuda")
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
    total_loss = 0.0
    total_count = 0
    with torch.inference_mode():
        for start in range(0, split.count, batch_size):
            indices = np.arange(
                start, min(split.count, start + batch_size), dtype=np.int64
            )
            images, targets = fetch_batch(split, indices, device=device)
            images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
            images = images.sub(mean).div(std)
            logits = model(images)
            loss = F.cross_entropy(logits, targets, reduction="sum")
            prediction = logits.argmax(dim=1).detach().cpu().numpy()
            target_np = targets.detach().cpu().numpy()
            np.add.at(confusion, (target_np, prediction), 1)
            total_loss += float(loss.detach().item())
            total_count += int(targets.shape[0])
    correct = int(np.trace(confusion))
    return {
        "status": "completed",
        "split": split.name,
        "count": total_count,
        "correct": correct,
        "errors": total_count - correct,
        "accuracy": correct / max(total_count, 1),
        "loss": total_loss / max(total_count, 1),
        "confusion": summarize_confusion(confusion, class_labels),
    }


def load_optional_real_holdout(args: argparse.Namespace, *, cache: Any) -> HoldoutSplit | None:
    if args.real_holdout_database is None:
        return None
    path = args.real_holdout_database.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        table_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "sample" not in table_names:
            raise ValueError("Real holdout database must contain a sample table")
        if "experiment_metadata" in table_names:
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    "SELECT key, value FROM experiment_metadata"
                )
            }
            if "image_size" in metadata and int(metadata["image_size"]) != int(cache.image_size):
                raise ValueError("Real holdout image_size does not match gray64 contract")
            if "base_labels" in metadata:
                labels = tuple(str(value) for value in json.loads(metadata["base_labels"]))
                if labels != tuple(cache.class_labels):
                    raise ValueError("Real holdout class-label order does not match classifier contract")

        if args.real_holdout_split:
            rows = list(
                connection.execute(
                    """
                    SELECT sample_id, class_index, image_gray_u8
                    FROM sample
                    WHERE split=?
                    ORDER BY sample_id
                    """,
                    (str(args.real_holdout_split),),
                )
            )
            name = str(args.real_holdout_split)
        else:
            rows = list(
                connection.execute(
                    """
                    SELECT sample_id, class_index, image_gray_u8
                    FROM sample
                    ORDER BY sample_id
                    """
                )
            )
            name = "all"
    finally:
        connection.close()

    if not rows:
        raise ValueError("Real holdout database selected zero rows")
    image_size = int(cache.image_size)
    expected = image_size * image_size
    images = np.empty((len(rows), image_size, image_size), dtype=np.uint8)
    labels = np.empty((len(rows),), dtype=np.int64)
    sample_ids: list[str] = []
    for index, row in enumerate(rows):
        raw = bytes(row["image_gray_u8"])
        if len(raw) != expected:
            raise ValueError(
                f"Real holdout {row['sample_id']} has {len(raw)} bytes; expected {expected}"
            )
        class_index = int(row["class_index"])
        if class_index < 0 or class_index >= len(cache.class_labels):
            raise ValueError(f"Real holdout class_index out of range: {class_index}")
        images[index] = np.frombuffer(raw, dtype=np.uint8).reshape(image_size, image_size)
        labels[index] = class_index
        sample_ids.append(str(row["sample_id"]))
    return HoldoutSplit(
        name=name,
        images_u8=torch.from_numpy(images).pin_memory(),
        labels=torch.from_numpy(labels).pin_memory(),
        sample_ids=sample_ids,
    )


def real_holdout_manifest(
    args: argparse.Namespace, holdout: HoldoutSplit | None
) -> dict[str, Any]:
    if holdout is None:
        return {"status": "not_provided"}
    return {
        "status": "loaded",
        "database": str(args.real_holdout_database.resolve()),
        "split": holdout.name,
        "count": holdout.count,
        "contract": "production-preprocessed-gray64-u8",
    }


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
    cpu_model = load_condition_checkpoint(
        checkpoint_path,
        architecture=condition.architecture,
        class_count=len(cache.class_labels),
        image_size=cache.image_size,
        device="cpu",
    )
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
        onnx_path, batch_size=int(args.benchmark_batch_size)
    )
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


def write_summary(
    output_root: Path,
    conditions: Sequence[Condition],
    results: dict[str, Any],
    *,
    final: bool = False,
) -> dict[str, Any]:
    status_counts = Counter(
        str(result.get("status", "unknown")) for result in results.values()
    )
    pending = [condition.name for condition in conditions if condition.name not in results]
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        result = results.get(condition.name)
        if not result or result.get("status") != "completed":
            continue
        manual = (
            result.get("accuracy", {})
            .get("splits", {})
            .get("manual_val", {})
            .get("summary", {})
        )
        perspective = result.get("perspective", {}).get("summary", {})
        holdout = result.get("real_holdout", {})
        deployment = result.get("deployment", {})
        benchmark = deployment.get("benchmark", {})
        rows.append(
            {
                "condition": condition.name,
                "architecture": condition.architecture,
                "augmentation": condition.augmentation,
                "training_kind": result.get("training", {}).get("kind"),
                "manual_mean_accuracy": manual.get("mean_accuracy"),
                "manual_worst_accuracy": manual.get("worst_accuracy"),
                "manual_worst_angle_deg": manual.get("worst_angle_deg"),
                "perspective_mean_accuracy": perspective.get("mean_accuracy"),
                "perspective_worst_accuracy": perspective.get("worst_accuracy"),
                "perspective_worst_case": perspective.get("worst_case"),
                "front_accuracy": perspective.get("front_accuracy"),
                "oblique_mean_accuracy": perspective.get("oblique_mean_accuracy"),
                "oblique_worst_accuracy": perspective.get("oblique_worst_accuracy"),
                "front_to_oblique_mean_delta": perspective.get("front_to_oblique_mean_delta"),
                "six_m_to_5m_or_7m_worst_rate": perspective.get(
                    "six_m_to_5m_or_7m_worst_rate"
                ),
                "six_m_worst_case": perspective.get("six_m_worst_case"),
                "real_holdout_status": holdout.get("status"),
                "real_holdout_accuracy": holdout.get("accuracy"),
                "real_holdout_six_m_to_5m_or_7m_rate": holdout.get(
                    "confusion", {}
                ).get("six_m_to_5m_or_7m_rate"),
                "onnx_bytes": deployment.get("onnx_bytes"),
                "ort_cpu_median_ms_batch": benchmark.get("median_ms_per_batch"),
                "ort_cpu_p95_ms_batch": benchmark.get("p95_ms_per_batch"),
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
    }
    atomic_write_json(output_root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    main()
