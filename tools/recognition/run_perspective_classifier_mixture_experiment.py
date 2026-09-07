from __future__ import annotations

"""Run the INV-013 follow-up mixture-training sweep.

This runner answers a different question from run_perspective_classifier_experiment.py.
The first INV-013 runner trained A0/A1/A2/A3 as isolated augmentation conditions. This
follow-up keeps the canonical crop in the training distribution and samples one of:

    Original / A0 random360 / A1 affine / A2 perspective / A3 perspective+recrop

for every sample at every epoch. Three augmentation-strength recipes are compared under
identical optimization settings for Plain and f8-r1 (six new training runs total).

A3 is corrected for the compact DB's existing 64x64 letterbox. The runner reconstructs
the actual pre-letterbox content rectangle from sample.original_width/original_height
and uses that rectangle when deriving the synthetic detector bbox. Existing INV-013 v1
code remains reproducible because perspective_classifier_augmentation only uses this
corrected behavior when content extents are supplied explicitly.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import gc
import json
import math
import sqlite3
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
    from perspective_classifier_augmentation import (
        AUGMENTATION_SPECS,
        GEOMETRY_UNIT_COUNT,
        PERSPECTIVE_EVALUATION_CASES,
        apply_evaluation_case,
        apply_training_geometry,
        deterministic_geometry_units,
    )
    from run_perspective_classifier_experiment import (
        build_condition_model,
        deploy_and_benchmark,
        describe_condition_model,
        evaluate_real_holdout_with_oom_fallback,
        load_condition_checkpoint,
        load_optional_real_holdout,
        real_holdout_manifest,
        require_runtime_dependencies,
        summarize_confusion,
    )
    from run_rotation_classifier_experiment import (
        CHECKPOINT_ANGLES,
        DENSE_ANGLES,
        angle_key,
        append_json_line,
        assert_v3_contract,
        atomic_write_json,
        configure_cuda,
        dense_evaluation,
        deterministic_random360_angles,
        environment_info,
        evaluate_angles_with_oom_fallback,
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
        GEOMETRY_UNIT_COUNT,
        PERSPECTIVE_EVALUATION_CASES,
        apply_evaluation_case,
        apply_training_geometry,
        deterministic_geometry_units,
    )
    from tools.recognition.run_perspective_classifier_experiment import (
        build_condition_model,
        deploy_and_benchmark,
        describe_condition_model,
        evaluate_real_holdout_with_oom_fallback,
        load_condition_checkpoint,
        load_optional_real_holdout,
        real_holdout_manifest,
        require_runtime_dependencies,
        summarize_confusion,
    )
    from tools.recognition.run_rotation_classifier_experiment import (
        CHECKPOINT_ANGLES,
        DENSE_ANGLES,
        angle_key,
        append_json_line,
        assert_v3_contract,
        atomic_write_json,
        configure_cuda,
        dense_evaluation,
        deterministic_random360_angles,
        environment_info,
        evaluate_angles_with_oom_fallback,
        fetch_batch,
        load_cache,
        read_json_if_exists,
        read_last_json_line,
        save_checkpoint,
        seed_everything,
    )


EXPERIMENT_IMPLEMENTATION_VERSION = "inv013-perspective-mixture-v1"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 150
DEFAULT_EFFECTIVE_BATCH = 512
DEFAULT_EVAL_BATCH = 256
DEFAULT_LR = 1.0e-3
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_OPSET = 16
ARCHITECTURES = ("plain", "f8-r1")
BRANCHES = (
    "original",
    "a0-random360",
    "a1-anisotropic-affine",
    "a2-perspective",
    "a3-perspective-recrop",
)
BRANCH_TO_INDEX = {name: index for index, name in enumerate(BRANCHES)}
CHOICE_STREAM = "inv013-mixture-choice"
GEOMETRY_STREAM = "inv013-shared-geometry"
A3_CONTENT_POLICY = "pre-letterbox-content-from-original-width-height"


@dataclass(frozen=True)
class MixtureRecipe:
    name: str
    original: float
    a0_random360: float
    a1_affine: float
    a2_perspective: float
    a3_perspective_recrop: float

    def weights(self) -> tuple[float, ...]:
        return (
            self.original,
            self.a0_random360,
            self.a1_affine,
            self.a2_perspective,
            self.a3_perspective_recrop,
        )

    def by_branch(self) -> dict[str, float]:
        return dict(zip(BRANCHES, self.weights(), strict=True))


RECIPES: tuple[MixtureRecipe, ...] = (
    MixtureRecipe(
        name="mix-light",
        original=0.30,
        a0_random360=0.25,
        a1_affine=0.15,
        a2_perspective=0.20,
        a3_perspective_recrop=0.10,
    ),
    MixtureRecipe(
        name="mix-mid",
        original=0.20,
        a0_random360=0.20,
        a1_affine=0.15,
        a2_perspective=0.30,
        a3_perspective_recrop=0.15,
    ),
    MixtureRecipe(
        name="mix-heavy",
        original=0.10,
        a0_random360=0.15,
        a1_affine=0.15,
        a2_perspective=0.35,
        a3_perspective_recrop=0.25,
    ),
)
RECIPE_BY_NAME = {recipe.name: recipe for recipe in RECIPES}


@dataclass(frozen=True)
class Condition:
    name: str
    architecture: str
    recipe: str


CONDITIONS: tuple[Condition, ...] = tuple(
    Condition(
        name=f"{architecture}-{recipe.name}",
        architecture=architecture,
        recipe=recipe.name,
    )
    for architecture in ARCHITECTURES
    for recipe in RECIPES
)


@dataclass(frozen=True)
class SplitContentGeometry:
    extent_x: np.ndarray
    extent_y: np.ndarray


@dataclass(frozen=True)
class ContentGeometry:
    splits: dict[str, SplitContentGeometry]
    policy: str = A3_CONTENT_POLICY


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Train Plain and f8-r1 with Original/A0/A1/A2/A3 stochastic mixture "
            "recipes and corrected detector-style A3 recropping."
        )
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
            / "perspective_classifier_mixture_experiment"
        ),
    )
    parser.add_argument(
        "--real-holdout-database",
        type=Path,
        help="Optional production-preprocessed reviewed gray64 holdout database.",
    )
    parser.add_argument("--real-holdout-split", type=str)
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
        help="Optional subset; default runs all six Plain/f8-r1 x light/mid/heavy conditions.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--overwrite-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def selected_conditions(args: argparse.Namespace) -> list[Condition]:
    if not args.conditions:
        return list(CONDITIONS)
    selected = set(str(value) for value in args.conditions)
    return [condition for condition in CONDITIONS if condition.name in selected]


def validate_args(args: argparse.Namespace) -> None:
    for recipe in RECIPES:
        validate_recipe(recipe)
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.effective_batch_size < 2 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("invalid optimizer settings")
    if args.angle_eval_every < 1:
        raise ValueError("--angle-eval-every must be positive")
    if args.opset < 16:
        raise ValueError("mixture experiment requires ONNX opset >= 16")
    if args.benchmark_batch_size < 1 or args.benchmark_warmup < 0 or args.benchmark_runs < 1:
        raise ValueError("invalid benchmark settings")
    if args.real_holdout_split and args.real_holdout_database is None:
        raise ValueError("--real-holdout-split requires --real-holdout-database")


def validate_recipe(recipe: MixtureRecipe) -> None:
    weights = recipe.weights()
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"Negative mixture weight in {recipe.name}: {weights}")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"Mixture weights must sum to one for {recipe.name}: {weights}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_runtime_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for perspective mixture training/evaluation")

    repository_root = args.repository_root.resolve()
    database = args.database.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configure_cuda(tf32=not bool(args.no_tf32))
    seed_everything(int(args.seed))
    cache = load_cache(database, cache_device=str(args.cache_device))
    assert_v3_contract(cache)
    content_geometry = load_content_geometry(database, cache=cache)
    conditions = selected_conditions(args)
    real_holdout = load_optional_real_holdout(args, cache=cache)

    manifest = {
        "status": "in_progress",
        "investigation": "PRODUCT-INV-RECOGNITION-013",
        "experiment": "perspective-mixture-followup",
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "repository_root": str(repository_root),
        "database": str(database),
        "output_root": str(output_root),
        "conditions": [asdict(condition) for condition in conditions],
        "recipes": {recipe.name: recipe.by_branch() for recipe in RECIPES},
        "augmentation_specs": {
            name: asdict(AUGMENTATION_SPECS[name])
            for name in BRANCHES
            if name != "original"
        },
        "a3_content_policy": A3_CONTENT_POLICY,
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
            "choice_stream": CHOICE_STREAM,
            "geometry_stream": GEOMETRY_STREAM,
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
                content_geometry=content_geometry,
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
    print("\n===== perspective mixture experiment finished =====", flush=True)
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
    content_geometry: ContentGeometry,
    real_holdout: Any,
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
            content_geometry=content_geometry,
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

    perspective = evaluate_perspective_corrected_with_oom_fallback(
        model,
        cache.splits["manual_val"],
        geometry=content_geometry.splits["manual_val"],
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
        "recipe": RECIPE_BY_NAME[condition.recipe].by_branch(),
        "training": training,
        "accuracy": dense_accuracy,
        "perspective": perspective,
        "real_holdout": holdout_result,
        "deployment": deployment,
    }


def train_with_oom_fallback(
    condition: Condition,
    *,
    run_dir: Path,
    cache: Any,
    content_geometry: ContentGeometry,
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
                content_geometry=content_geometry,
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
    content_geometry: ContentGeometry,
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
    recipe = RECIPE_BY_NAME[condition.recipe]

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
        "recipe": recipe.by_branch(),
        "seed": int(args.seed),
        "amp": amp,
        "tf32": not bool(args.no_tf32),
        "checkpoint_angles": list(CHECKPOINT_ANGLES),
        "choice_stream": CHOICE_STREAM,
        "geometry_stream": GEOMETRY_STREAM,
        "a3_content_policy": A3_CONTENT_POLICY,
    }
    atomic_write_json(output_dir / "config.json", config)

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(
            model,
            cache.splits["train"],
            geometry=content_geometry.splits["train"],
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cache=cache,
            effective_batch_size=int(args.effective_batch_size),
            microbatch=microbatch,
            recipe=recipe,
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
        mix_text = " ".join(
            f"{name}={train_metrics['branch_counts'][name]}" for name in BRANCHES
        )
        print(
            f"epoch={epoch:03d} loss={train_metrics['loss']:.5f} "
            f"acc={train_metrics['accuracy']:.5f} "
            f"samples/s={train_metrics['samples_per_second']:.1f} "
            f"{mix_text} {angle_text}",
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
        "kind": "trained_mixture",
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_checkpoint_score": best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "effective_batch_size": int(args.effective_batch_size),
        "physical_microbatch": int(microbatch),
        "recipe": recipe.by_branch(),
        "model": description,
    }
    atomic_write_json(output_dir / "summary.json", result)
    return best_path, result, best_model


def train_one_epoch(
    model: nn.Module,
    split: Any,
    *,
    geometry: SplitContentGeometry,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cache: Any,
    effective_batch_size: int,
    microbatch: int,
    recipe: MixtureRecipe,
    epoch: int,
    seed: int,
    amp: bool,
) -> dict[str, Any]:
    model.train()
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    order = rng.permutation(split.count)
    angles = deterministic_random360_angles(split.sample_ids, seed=seed, epoch=epoch)
    units = deterministic_geometry_units(
        split.sample_ids,
        seed=seed,
        epoch=epoch,
        stream=GEOMETRY_STREAM,
    )
    choices = deterministic_recipe_choices(
        split.sample_ids,
        recipe=recipe,
        seed=seed,
        epoch=epoch,
    )
    branch_histogram = np.bincount(choices, minlength=len(BRANCHES))

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
            batch_units = torch.from_numpy(units[batch_indices]).to(
                device=device, dtype=torch.float32
            )
            batch_choices = torch.from_numpy(choices[batch_indices]).to(
                device=device, dtype=torch.long
            )
            extent_x = torch.from_numpy(geometry.extent_x[batch_indices]).to(
                device=device, dtype=torch.float32
            )
            extent_y = torch.from_numpy(geometry.extent_y[batch_indices]).to(
                device=device, dtype=torch.float32
            )
            images = apply_mixed_training_geometry(
                images,
                choices=batch_choices,
                angles_deg=batch_angles,
                units=batch_units,
                content_extent_x=extent_x,
                content_extent_y=extent_y,
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
        "branch_counts": {
            name: int(branch_histogram[index]) for index, name in enumerate(BRANCHES)
        },
    }


def deterministic_recipe_choices(
    sample_ids: Sequence[str],
    *,
    recipe: MixtureRecipe,
    seed: int,
    epoch: int,
) -> np.ndarray:
    validate_recipe(recipe)
    units = deterministic_geometry_units(
        sample_ids,
        seed=seed,
        epoch=epoch,
        stream=CHOICE_STREAM,
        unit_count=1,
    )[:, 0]
    thresholds = np.cumsum(np.asarray(recipe.weights(), dtype=np.float64))
    choices = np.searchsorted(thresholds, units.astype(np.float64), side="right")
    return np.minimum(choices, len(BRANCHES) - 1).astype(np.int64, copy=False)


def apply_mixed_training_geometry(
    images: torch.Tensor,
    *,
    choices: torch.Tensor,
    angles_deg: torch.Tensor,
    units: torch.Tensor,
    content_extent_x: torch.Tensor,
    content_extent_y: torch.Tensor,
) -> torch.Tensor:
    batch = int(images.shape[0])
    if choices.shape != (batch,):
        raise ValueError("choices must have shape [N]")
    if angles_deg.shape != (batch,):
        raise ValueError("angles_deg must have shape [N]")
    if units.shape != (batch, GEOMETRY_UNIT_COUNT):
        raise ValueError(f"units must have shape [N,{GEOMETRY_UNIT_COUNT}]")
    if content_extent_x.shape != (batch,) or content_extent_y.shape != (batch,):
        raise ValueError("content extents must have shape [N]")

    observed = images.clone()
    for branch_name in BRANCHES[1:]:
        branch_index = BRANCH_TO_INDEX[branch_name]
        selected = torch.nonzero(choices == branch_index, as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        branch_images = images.index_select(0, selected)
        kwargs: dict[str, Any] = {}
        if branch_name == "a3-perspective-recrop":
            kwargs["content_extent_x"] = content_extent_x.index_select(0, selected)
            kwargs["content_extent_y"] = content_extent_y.index_select(0, selected)
        transformed = apply_training_geometry(
            branch_images,
            spec=AUGMENTATION_SPECS[branch_name],
            angles_deg=angles_deg.index_select(0, selected),
            units=units.index_select(0, selected),
            **kwargs,
        )
        observed.index_copy_(0, selected, transformed)
    return observed


def load_content_geometry(database: Path, *, cache: Any) -> ContentGeometry:
    path = database.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT sample_id, original_width, original_height FROM sample ORDER BY sample_id"
        ).fetchall()
    finally:
        connection.close()
    dimensions = {
        str(row["sample_id"]): (int(row["original_width"]), int(row["original_height"]))
        for row in rows
    }
    result: dict[str, SplitContentGeometry] = {}
    for split_name, split in cache.splits.items():
        extent_x = np.empty((split.count,), dtype=np.float32)
        extent_y = np.empty((split.count,), dtype=np.float32)
        for index, sample_id in enumerate(split.sample_ids):
            try:
                width, height = dimensions[sample_id]
            except KeyError as error:
                raise ValueError(f"Missing original dimensions for {sample_id}") from error
            x, y = preletterbox_content_extent(
                width,
                height,
                image_size=int(cache.image_size),
            )
            extent_x[index] = np.float32(x)
            extent_y[index] = np.float32(y)
        result[split_name] = SplitContentGeometry(extent_x=extent_x, extent_y=extent_y)
    return ContentGeometry(splits=result)


def preletterbox_content_extent(
    original_width: int,
    original_height: int,
    *,
    image_size: int,
) -> tuple[float, float]:
    if original_width <= 0 or original_height <= 0 or image_size <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(image_size / original_width, image_size / original_height)
    resized_width = max(
        1,
        min(image_size, int(math.floor(original_width * scale + 0.5))),
    )
    resized_height = max(
        1,
        min(image_size, int(math.floor(original_height * scale + 0.5))),
    )
    return resized_width / image_size, resized_height / image_size


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
        "recipe": RECIPE_BY_NAME[condition.recipe].by_branch(),
        "seed": int(args.seed),
        "choice_stream": CHOICE_STREAM,
        "geometry_stream": GEOMETRY_STREAM,
        "a3_content_policy": A3_CONTENT_POLICY,
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
        "kind": "recovered_completed_mixture_training",
        "checkpoint": str(best_path),
        "best_epoch": int(checkpoint.get("epoch", 0)),
        "best_checkpoint_score": best_score,
        "effective_batch_size": int(config["effective_batch_size"]),
        "physical_microbatch": int(config.get("physical_microbatch", 0)),
        "recipe": config.get("recipe", {}),
        "model": config.get("model", {}),
    }
    atomic_write_json(training_dir / "summary.json", result)
    print(f"[resume] recovered completed training for {condition.name}", flush=True)
    return best_path, result, model


def evaluate_perspective_corrected_with_oom_fallback(
    model: nn.Module,
    split: Any,
    *,
    geometry: SplitContentGeometry,
    class_labels: Sequence[str],
    batch_size: int,
    mean: float,
    std: float,
) -> dict[str, Any]:
    candidate = int(batch_size)
    while candidate >= 8:
        try:
            return evaluate_perspective_corrected(
                model,
                split,
                geometry=geometry,
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
    raise RuntimeError("Corrected perspective evaluation does not fit at batch=8")


def evaluate_perspective_corrected(
    model: nn.Module,
    split: Any,
    *,
    geometry: SplitContentGeometry,
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
                    start,
                    min(split.count, start + batch_size),
                    dtype=np.int64,
                )
                images, targets = fetch_batch(split, indices, device=device)
                images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
                extent_x = torch.from_numpy(geometry.extent_x[indices]).to(
                    device=device, dtype=torch.float32
                )
                extent_y = torch.from_numpy(geometry.extent_y[indices]).to(
                    device=device, dtype=torch.float32
                )
                images = apply_evaluation_case(
                    images,
                    case,
                    content_extent_x=extent_x,
                    content_extent_y=extent_y,
                )
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
        "a3_content_policy": A3_CONTENT_POLICY,
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
                "recipe": condition.recipe,
                "weights": RECIPE_BY_NAME[condition.recipe].by_branch(),
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
        "a3_content_policy": A3_CONTENT_POLICY,
        "status_counts": dict(status_counts),
        "pending": pending,
        "comparison_rows": rows,
        "results": {name: result.get("status") for name, result in results.items()},
    }
    atomic_write_json(output_root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    main()
