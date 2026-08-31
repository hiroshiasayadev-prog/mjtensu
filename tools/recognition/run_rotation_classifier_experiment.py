from __future__ import annotations

"""Run PRODUCT-INV-RECOGNITION-007 end to end.

One invocation trains/evaluates the requested matrix, exports every successful model to
ONNX, checks PyTorch/ORT parity, records ONNX graph statistics, and benchmarks ORT CPU
batch-16 inference.  Conditions are isolated: a failed model is recorded and the runner
continues, and completed conditions are skipped on a later resume.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from rotation_classifier_experiment_models import (
        build_experiment_model,
        describe_experiment_model,
    )
except ModuleNotFoundError:  # package-style import used by tests
    from tools.recognition.rotation_classifier_experiment_models import (
        build_experiment_model,
        describe_experiment_model,
    )


EXPERIMENT_IMPLEMENTATION_VERSION = "inv007-fidelity-v2"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 50
DEFAULT_EFFECTIVE_BATCH = 512
DEFAULT_EVAL_BATCH = 256
DEFAULT_LR = 1.0e-3
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_OPSET = 16
CHECKPOINT_ANGLES = (0.0, 15.0, 30.0, 45.0)
DENSE_ANGLES = tuple(index * 5.625 for index in range(64))
EXPECTED_V3_MEAN = 0.6815832403977466
EXPECTED_V3_STD = 0.2725553681973969


@dataclass(frozen=True)
class Condition:
    name: str
    architecture: str
    augmentation: str
    production_reference: bool = False


CONDITIONS: tuple[Condition, ...] = (
    Condition("c8-noaug", "c8", "none"),
    Condition("c8-production", "c8", "production-rot22p5", production_reference=True),
    Condition("plain-noaug", "plain", "none"),
    Condition("plain-random360", "plain", "random360"),
    Condition("roteqnet-noaug", "roteqnet", "none"),
    Condition("roteqnet-random360", "roteqnet", "random360"),
    Condition("riccnn-noaug", "riccnn", "none"),
    Condition("riccnn-random360", "riccnn", "random360"),
    Condition("sconv-noaug", "sconv", "none"),
    Condition("sconv-random360", "sconv", "random360"),
)


@dataclass
class SplitCache:
    name: str
    images_u8: torch.Tensor
    labels: torch.Tensor
    sample_ids: list[str]

    @property
    def count(self) -> int:
        return int(self.labels.shape[0])


@dataclass
class ExperimentCache:
    splits: dict[str, SplitCache]
    image_size: int
    class_labels: tuple[str, ...]
    mean: float
    std: float
    cache_device: str


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    classifier_root = repository_root / ".local" / "recognition" / "tile_classifier_runs"
    production_root = classifier_root / "gray64_c8_rot22p5_bs512_gray35_v3_jp189_seed42"
    parser = argparse.ArgumentParser(
        description="Run the INV-007 rotation-classifier architecture/augmentation comparison."
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
        default=repository_root / ".local" / "recognition" / "rotation_classifier_experiment",
    )
    parser.add_argument(
        "--production-c8-checkpoint",
        type=Path,
        default=production_root / "best.pt",
    )
    parser.add_argument(
        "--production-c8-onnx",
        type=Path,
        default=production_root / "tile-c8-gray35-v3-jp189.onnx",
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
        "--cache-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=[condition.name for condition in CONDITIONS],
        help="Optional subset. Default: all ten conditions.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument(
        "--overwrite-completed",
        action="store_true",
        help="Rerun conditions whose result.json already says status=completed.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed condition instead of recording it and continuing.",
    )
    return parser.parse_args()


def selected_conditions(args: argparse.Namespace) -> list[Condition]:
    if not args.conditions:
        return list(CONDITIONS)
    selected = set(str(value) for value in args.conditions)
    return [condition for condition in CONDITIONS if condition.name in selected]


def prior_result_is_reusable(
    condition: Condition,
    prior: dict[str, Any] | None,
) -> bool:
    if prior is None or prior.get("status") != "completed":
        return False
    if prior.get("condition") != asdict(condition):
        return False
    version = prior.get("implementation_version")
    if version == EXPERIMENT_IMPLEMENTATION_VERSION:
        return True
    # The fidelity-v2 changes affect only the three research architectures. C8 and
    # Plain reuse the accepted repository implementations, so successful v1 results
    # for those families remain valid and should not be recomputed unnecessarily.
    return condition.architecture in {"c8", "plain"}


def main() -> None:
    args = parse_args()
    validate_args(args)
    repository_root = args.repository_root.resolve()
    database = args.database.resolve()
    output_root = args.output_root.resolve()
    production_checkpoint = args.production_c8_checkpoint.resolve()
    production_onnx = args.production_c8_onnx.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    require_runtime_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for INV-007 training/evaluation")
    configure_cuda(tf32=not bool(args.no_tf32))
    seed_everything(int(args.seed))

    cache = load_cache(database, cache_device=str(args.cache_device))
    assert_v3_contract(cache)
    conditions = selected_conditions(args)
    run_manifest = {
        "status": "in_progress",
        "investigation": "PRODUCT-INV-RECOGNITION-007",
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "repository_root": str(repository_root),
        "database": str(database),
        "output_root": str(output_root),
        "production_c8_checkpoint": str(production_checkpoint),
        "production_c8_onnx": str(production_onnx),
        "conditions": [asdict(condition) for condition in conditions],
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
        },
        "dataset": {
            "image_size": cache.image_size,
            "class_labels": list(cache.class_labels),
            "normalization": {"mean": cache.mean, "std": cache.std},
            "splits": {name: split.count for name, split in cache.splits.items()},
            "cache_device": cache.cache_device,
        },
        "environment": environment_info(),
    }
    atomic_write_json(output_root / "manifest.json", run_manifest)

    all_results: dict[str, Any] = {}
    for condition in conditions:
        run_dir = output_root / condition.name
        result_path = run_dir / "result.json"
        prior = read_json_if_exists(result_path)
        if (
            prior_result_is_reusable(condition, prior)
            and not bool(args.overwrite_completed)
        ):
            print(f"[resume] skip completed {condition.name}", flush=True)
            all_results[condition.name] = prior
            write_aggregate_summary(output_root, conditions, all_results)
            continue

        print(f"\n===== {condition.name} =====", flush=True)
        started = time.perf_counter()
        try:
            # Preserve failed/interrupted run artifacts so a completed training phase can
            # be recovered without paying another 50 epochs merely because a later
            # checkpoint-load/export/evaluation step failed.
            run_dir.mkdir(parents=True, exist_ok=True)
            result = run_condition(
                condition,
                run_dir=run_dir,
                cache=cache,
                args=args,
                repository_root=repository_root,
                production_checkpoint=production_checkpoint,
                production_onnx=production_onnx,
            )
            result["status"] = "completed"
            result["implementation_version"] = EXPERIMENT_IMPLEMENTATION_VERSION
            result["elapsed_seconds"] = time.perf_counter() - started
        except Exception as error:  # deliberate per-condition isolation for overnight runs
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
        all_results[condition.name] = result
        write_aggregate_summary(output_root, conditions, all_results)
        if result["status"] != "completed" and bool(args.fail_fast):
            raise RuntimeError(f"Condition failed: {condition.name}: {result.get('error')}")

    summary = write_aggregate_summary(output_root, conditions, all_results, final=True)
    print("\n===== experiment finished =====", flush=True)
    print(json.dumps(summary["status_counts"], ensure_ascii=False), flush=True)
    print(f"summary: {output_root / 'summary.json'}", flush=True)


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
        raise ValueError("INV-007 custom sampling models require ONNX opset >= 16")
    if args.benchmark_batch_size < 1 or args.benchmark_runs < 1:
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
            "Missing INV-007 dependencies: "
            + ", ".join(missing)
            + ". Install them into the existing classifier environment without replacing its CUDA torch wheel."
        )


def configure_cuda(*, tf32: bool) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sqlite_readonly_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def load_cache(database: Path, *, cache_device: str) -> ExperimentCache:
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(sqlite_readonly_uri(database), uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM experiment_metadata")
        }
        image_size = int(metadata["image_size"])
        class_labels = tuple(str(value) for value in json.loads(metadata["base_labels"]))
        splits = {
            split_name: load_split(connection, split_name, image_size=image_size)
            for split_name in ("train", "manual_val", "jp_val")
        }
    finally:
        connection.close()

    train_np = splits["train"].images_u8.numpy()
    mean = float(train_np.mean(dtype=np.float64) / 255.0)
    std = max(float(train_np.std(dtype=np.float64) / 255.0), 1.0 / 255.0)
    resolved_cache = cache_device
    if resolved_cache == "auto":
        total_bytes = sum(split.images_u8.numel() for split in splits.values())
        free_bytes, _ = torch.cuda.mem_get_info()
        resolved_cache = "cuda" if total_bytes < int(free_bytes * 0.10) else "cpu"

    if resolved_cache == "cuda":
        for split in splits.values():
            split.images_u8 = split.images_u8.cuda(non_blocking=False)
            split.labels = split.labels.cuda(non_blocking=False)
        torch.cuda.synchronize()
    else:
        for split in splits.values():
            split.images_u8 = split.images_u8.pin_memory()
            split.labels = split.labels.pin_memory()

    print(
        f"[dataset] mean={mean:.12f} std={std:.12f} cache={resolved_cache} "
        + " ".join(f"{name}={split.count}" for name, split in splits.items()),
        flush=True,
    )
    return ExperimentCache(
        splits=splits,
        image_size=image_size,
        class_labels=class_labels,
        mean=mean,
        std=std,
        cache_device=resolved_cache,
    )


def load_split(
    connection: sqlite3.Connection,
    split_name: str,
    *,
    image_size: int,
) -> SplitCache:
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM sample WHERE split=?", (split_name,)
        ).fetchone()[0]
    )
    images = np.empty((count, image_size, image_size), dtype=np.uint8)
    labels = np.empty((count,), dtype=np.int64)
    sample_ids: list[str] = []
    expected = image_size * image_size
    rows = connection.execute(
        """
        SELECT sample_id, class_index, image_gray_u8
        FROM sample
        WHERE split=?
        ORDER BY sample_id
        """,
        (split_name,),
    )
    for index, row in enumerate(rows):
        raw = bytes(row["image_gray_u8"])
        if len(raw) != expected:
            raise ValueError(f"{row['sample_id']} has {len(raw)} bytes, expected {expected}")
        images[index] = np.frombuffer(raw, dtype=np.uint8).reshape(image_size, image_size)
        labels[index] = int(row["class_index"])
        sample_ids.append(str(row["sample_id"]))
    return SplitCache(
        name=split_name,
        images_u8=torch.from_numpy(images),
        labels=torch.from_numpy(labels),
        sample_ids=sample_ids,
    )


def assert_v3_contract(cache: ExperimentCache) -> None:
    expected_counts = {"train": 19593, "manual_val": 450, "jp_val": 6800}
    observed_counts = {name: split.count for name, split in cache.splits.items()}
    if observed_counts != expected_counts:
        raise ValueError(
            f"INV-007 requires the frozen v3 split counts {expected_counts}; found {observed_counts}"
        )
    if len(cache.class_labels) != 35 or cache.class_labels[-1] != "invalid":
        raise ValueError("INV-007 requires the canonical 35-class v3 label contract")
    if abs(cache.mean - EXPECTED_V3_MEAN) > 1.0e-9 or abs(cache.std - EXPECTED_V3_STD) > 1.0e-9:
        raise ValueError(
            "Frozen v3 normalization changed: "
            f"observed mean/std={cache.mean}/{cache.std}, "
            f"expected {EXPECTED_V3_MEAN}/{EXPECTED_V3_STD}"
        )


def run_condition(
    condition: Condition,
    *,
    run_dir: Path,
    cache: ExperimentCache,
    args: argparse.Namespace,
    repository_root: Path,
    production_checkpoint: Path,
    production_onnx: Path,
) -> dict[str, Any]:
    if condition.production_reference:
        if not production_checkpoint.is_file():
            raise FileNotFoundError(
                f"Production C8 checkpoint is required for dense evaluation: {production_checkpoint}"
            )
        checkpoint_path = production_checkpoint
        training_result: dict[str, Any] = {
            "kind": "existing_production_reference",
            "checkpoint": str(checkpoint_path),
        }
        model = load_checkpoint_model(
            checkpoint_path,
            architecture="c8",
            class_count=len(cache.class_labels),
            image_size=cache.image_size,
            device="cuda",
        )
    else:
        recovered = recover_completed_training(
            condition,
            run_dir=run_dir,
            cache=cache,
            args=args,
        )
        if recovered is not None:
            checkpoint_path, training_result, model = recovered
        else:
            checkpoint_path, training_result, model = train_with_oom_fallback(
                condition,
                run_dir=run_dir,
                cache=cache,
                args=args,
            )

    model.eval()
    dense_accuracy = dense_evaluation(
        model,
        cache,
        batch_size=int(args.eval_batch_size),
        angles=DENSE_ANGLES,
        amp=False,
    )
    atomic_write_json(run_dir / "dense_evaluation.json", dense_accuracy)

    deployment: dict[str, Any]
    try:
        deployment = deploy_and_benchmark(
            condition,
            model=model,
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            cache=cache,
            args=args,
            repository_root=repository_root,
            production_onnx=production_onnx,
        )
    except Exception as error:
        deployment = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        print(deployment["traceback"], file=sys.stderr, flush=True)

    return {
        "condition": asdict(condition),
        "training": training_result,
        "accuracy": dense_accuracy,
        "deployment": deployment,
    }


def recover_completed_training(
    condition: Condition,
    *,
    run_dir: Path,
    cache: ExperimentCache,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], nn.Module] | None:
    """Recover a finished C8/Plain training phase after a later runner failure.

    The first INV-007 runner could finish all 50 epochs and then fail while rebuilding
    the best C8 checkpoint for evaluation.  Retaining that checkpoint is safe for the
    unchanged accepted C8/Plain implementations, but old experimental RotEqNet/RIC/SConv
    artifacts are deliberately not recovered because their fidelity is being reviewed.
    """
    if condition.architecture not in {"c8", "plain"}:
        return None
    training_dir = run_dir / "training"
    best_path = training_dir / "best.pt"
    config_path = training_dir / "config.json"
    history_path = training_dir / "history.jsonl"
    if not (best_path.is_file() and config_path.is_file() and history_path.is_file()):
        return None
    config = read_json_if_exists(config_path)
    if config is None:
        return None
    expected_condition = asdict(condition)
    if config.get("condition") != expected_condition:
        return None
    try:
        recorded_database = Path(str(config.get("database", ""))).resolve()
    except OSError:
        return None
    if recorded_database != args.database.resolve():
        return None
    if int(config.get("image_size", -1)) != int(cache.image_size):
        return None
    if tuple(config.get("class_labels", ())) != tuple(cache.class_labels):
        return None
    expected_scalars = {
        "epochs": int(args.epochs),
        "effective_batch_size": int(args.effective_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "augmentation": condition.augmentation,
        "seed": int(args.seed),
    }
    for key, expected in expected_scalars.items():
        if config.get(key) != expected:
            return None
    normalization = config.get("normalization")
    if not isinstance(normalization, dict):
        return None
    if (
        float(normalization.get("mean", float("nan"))) != float(cache.mean)
        or float(normalization.get("std", float("nan"))) != float(cache.std)
    ):
        return None

    last_record = read_last_json_line(history_path)
    if last_record is None or int(last_record.get("epoch", -1)) != int(args.epochs):
        return None
    checkpoint = torch.load(best_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return None
    best_epoch = int(checkpoint.get("epoch", 0))
    metrics = checkpoint.get("metrics")
    best_score = None
    if isinstance(metrics, dict) and metrics.get("checkpoint_score") is not None:
        best_score = float(metrics["checkpoint_score"])
    model = load_checkpoint_model(
        best_path,
        architecture=condition.architecture,
        class_count=len(cache.class_labels),
        image_size=cache.image_size,
        device="cuda",
    )
    result = {
        "kind": "recovered_completed_training",
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_checkpoint_score": best_score,
        "effective_batch_size": int(config["effective_batch_size"]),
        "physical_microbatch": int(config.get("physical_microbatch", 0)),
        "parameter_count": config.get("model", {}).get("parameter_count"),
        "trainable_parameter_count": config.get("model", {}).get(
            "trainable_parameter_count"
        ),
        "model_details": config.get("model", {}),
        "recovered_after_post_training_failure": True,
    }
    atomic_write_json(training_dir / "summary.json", result)
    print(
        f"[resume] recovered finished training for {condition.name} from {best_path}",
        flush=True,
    )
    return best_path, result, model


def train_with_oom_fallback(
    condition: Condition,
    *,
    run_dir: Path,
    cache: ExperimentCache,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], nn.Module]:
    effective_batch = int(args.effective_batch_size)
    candidates = []
    value = effective_batch
    while value >= 16:
        candidates.append(value)
        value //= 2
    last_error_message: str | None = None
    for microbatch in candidates:
        try:
            attempt_dir = run_dir / "training"
            if attempt_dir.exists():
                shutil.rmtree(attempt_dir)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[train] {condition.name} effective_batch={effective_batch} microbatch={microbatch}",
                flush=True,
            )
            checkpoint, result, model = train_condition(
                condition,
                output_dir=attempt_dir,
                cache=cache,
                args=args,
                microbatch=microbatch,
            )
            return checkpoint, result, model
        except torch.cuda.OutOfMemoryError as error:
            last_error_message = str(error)
            print(
                f"[oom] {condition.name} microbatch={microbatch}; retrying smaller physical batch",
                file=sys.stderr,
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            last_error_message = str(error)
            print(
                f"[oom] {condition.name} microbatch={microbatch}; retrying smaller physical batch",
                file=sys.stderr,
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
    if last_error_message is not None:
        raise RuntimeError(
            f"{condition.name} could not train even at microbatch={candidates[-1]}: "
            f"{last_error_message}"
        )
    raise RuntimeError("No training microbatch candidates were generated")


def train_condition(
    condition: Condition,
    *,
    output_dir: Path,
    cache: ExperimentCache,
    args: argparse.Namespace,
    microbatch: int,
) -> tuple[Path, dict[str, Any], nn.Module]:
    seed_everything(int(args.seed))
    device = torch.device("cuda")
    model = build_experiment_model(
        condition.architecture,
        class_count=len(cache.class_labels),
        image_size=cache.image_size,
    ).to(device)
    description = describe_experiment_model(model, condition.architecture)
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
    train_split = cache.splits["train"]
    history_path = output_dir / "history.jsonl"
    best_score = -1.0
    best_epoch = 0
    best_path = output_dir / "best.pt"
    started = time.perf_counter()

    config = {
        "condition": asdict(condition),
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "database": str(args.database.resolve()),
        "model": {
            "name": condition.architecture,
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
        "augmentation": condition.augmentation,
        "seed": int(args.seed),
        "amp": amp,
        "tf32": not bool(args.no_tf32),
        "checkpoint_angles": list(CHECKPOINT_ANGLES),
    }
    atomic_write_json(output_dir / "config.json", config)

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(
            model,
            train_split,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cache=cache,
            effective_batch_size=int(args.effective_batch_size),
            microbatch=microbatch,
            augmentation=condition.augmentation,
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
                np.mean([manual_validation[angle_key(angle)]["accuracy"] for angle in CHECKPOINT_ANGLES])
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
            f"manual@{key}={value['accuracy']:.5f}" for key, value in manual_validation.items()
        )
        print(
            f"epoch={epoch:03d} loss={train_metrics['loss']:.5f} "
            f"acc={train_metrics['accuracy']:.5f} samples/s={train_metrics['samples_per_second']:.1f} "
            f"{angle_text}",
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
        raise RuntimeError(f"No best checkpoint was produced for {condition.name}")
    elapsed = time.perf_counter() - started
    best_model = load_checkpoint_model(
        best_path,
        architecture=condition.architecture,
        class_count=len(cache.class_labels),
        image_size=cache.image_size,
        device=device,
    )
    training_result = {
        "kind": "trained",
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_checkpoint_score": best_score,
        "elapsed_seconds": elapsed,
        "effective_batch_size": int(args.effective_batch_size),
        "physical_microbatch": microbatch,
        "parameter_count": description.parameter_count,
        "trainable_parameter_count": description.trainable_parameter_count,
        "model_details": description.details,
    }
    atomic_write_json(output_dir / "summary.json", training_result)
    return best_path, training_result, best_model


def train_one_epoch(
    model: nn.Module,
    split: SplitCache,
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cache: ExperimentCache,
    effective_batch_size: int,
    microbatch: int,
    augmentation: str,
    epoch: int,
    seed: int,
    amp: bool,
) -> dict[str, Any]:
    model.train()
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    order = rng.permutation(split.count)
    angles = (
        deterministic_random360_angles(split.sample_ids, seed=seed, epoch=epoch)
        if augmentation == "random360"
        else None
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
            if augmentation == "random360":
                if angles is None:  # pragma: no cover
                    raise RuntimeError("random360 angle corpus is missing")
                batch_angles = torch.from_numpy(angles[batch_indices]).to(
                    device=device, dtype=torch.float32
                )
                images = rotate_batch(images, batch_angles)
            elif augmentation != "none":
                raise ValueError(f"Unsupported training augmentation: {augmentation}")
            images = images.sub(cache.mean).div(cache.std)

            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
                raw_loss = F.cross_entropy(logits, targets)
                weighted_loss = raw_loss * (float(len(batch_indices)) / float(effective_count))
            scaler.scale(weighted_loss).backward()
            total_loss += float(raw_loss.detach().item()) * len(batch_indices)
            total_correct += int((logits.detach().argmax(dim=1) == targets).sum().item())
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


def deterministic_random360_angles(
    sample_ids: Sequence[str],
    *,
    seed: int,
    epoch: int,
) -> np.ndarray:
    result = np.empty((len(sample_ids),), dtype=np.float32)
    prefix = f"{seed}\0{epoch}\0".encode("utf-8")
    denominator = float(2**64)
    for index, sample_id in enumerate(sample_ids):
        digest = hashlib.sha256(prefix + sample_id.encode("utf-8")).digest()
        unit = int.from_bytes(digest[:8], "big") / denominator
        result[index] = np.float32(-180.0 + 360.0 * unit)
    return result


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
    labels = split.labels.index_select(0, index_cpu)
    return (
        images.to(device=device, non_blocking=True),
        labels.to(device=device, non_blocking=True),
    )


def rotate_batch(images: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    radians = angles_deg.to(dtype=torch.float32) * (math.pi / 180.0)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    theta = torch.zeros((images.shape[0], 2, 3), device=images.device, dtype=torch.float32)
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


def evaluate_angles_with_oom_fallback(
    model: nn.Module,
    split: SplitCache,
    *,
    angles: Sequence[float],
    batch_size: int,
    mean: float,
    std: float,
    amp: bool,
) -> dict[str, Any]:
    candidate = int(batch_size)
    while candidate >= 8:
        try:
            return evaluate_angles(
                model,
                split,
                angles=angles,
                batch_size=candidate,
                mean=mean,
                std=std,
                amp=amp,
            )
        except torch.cuda.OutOfMemoryError:
            print(
                f"[eval-oom] {split.name} batch={candidate}; retrying {candidate // 2}",
                file=sys.stderr,
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
            candidate //= 2
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            print(
                f"[eval-oom] {split.name} batch={candidate}; retrying {candidate // 2}",
                file=sys.stderr,
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
            candidate //= 2
    raise RuntimeError(f"Evaluation for {split.name} does not fit even at batch=8")


def evaluate_angles(
    model: nn.Module,
    split: SplitCache,
    *,
    angles: Sequence[float],
    batch_size: int,
    mean: float,
    std: float,
    amp: bool,
) -> dict[str, Any]:
    return {
        angle_key(angle): evaluate_split(
            model,
            split,
            angle=float(angle),
            batch_size=batch_size,
            mean=mean,
            std=std,
            amp=amp,
        )
        for angle in angles
    }


def evaluate_split(
    model: nn.Module,
    split: SplitCache,
    *,
    angle: float,
    batch_size: int,
    mean: float,
    std: float,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    confusion: np.ndarray | None = None
    total_loss = 0.0
    total_count = 0
    device = torch.device("cuda")
    with torch.inference_mode():
        for start in range(0, split.count, batch_size):
            indices = np.arange(start, min(split.count, start + batch_size), dtype=np.int64)
            images, targets = fetch_batch(split, indices, device=device)
            images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
            if abs(angle) > 1.0e-12:
                angle_tensor = torch.full(
                    (images.shape[0],), angle, device=device, dtype=torch.float32
                )
                images = rotate_batch(images, angle_tensor)
            images = images.sub(mean).div(std)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
                loss = F.cross_entropy(logits, targets, reduction="sum")
            if confusion is None:
                class_count = int(logits.shape[1])
                confusion = np.zeros((class_count, class_count), dtype=np.int64)
            prediction = logits.argmax(dim=1).detach().cpu().numpy()
            target_np = targets.detach().cpu().numpy()
            np.add.at(confusion, (target_np, prediction), 1)
            total_loss += float(loss.detach().item())
            total_count += int(targets.shape[0])
    if confusion is None:  # pragma: no cover - frozen validation splits are non-empty
        raise ValueError(f"Evaluation split is empty: {split.name}")
    class_count = int(confusion.shape[0])
    correct = int(np.trace(confusion))
    per_class: list[float | None] = []
    for class_index in range(class_count):
        class_total = int(confusion[class_index].sum())
        per_class.append(
            None if class_total == 0 else float(confusion[class_index, class_index] / class_total)
        )
    present = [value for value in per_class if value is not None]
    return {
        "angle_deg": angle,
        "count": total_count,
        "correct": correct,
        "errors": total_count - correct,
        "loss": total_loss / max(total_count, 1),
        "accuracy": correct / max(total_count, 1),
        "macro_accuracy": float(np.mean(present)) if present else 0.0,
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def dense_evaluation(
    model: nn.Module,
    cache: ExperimentCache,
    *,
    batch_size: int,
    angles: Sequence[float],
    amp: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"angle_grid_deg": list(angles), "splits": {}}
    for split_name in ("manual_val", "jp_val"):
        print(f"[dense-eval] {split_name} {len(angles)} angles", flush=True)
        angle_results = evaluate_angles_with_oom_fallback(
            model,
            cache.splits[split_name],
            angles=angles,
            batch_size=batch_size,
            mean=cache.mean,
            std=cache.std,
            amp=amp,
        )
        accuracies = [float(angle_results[angle_key(angle)]["accuracy"]) for angle in angles]
        worst_index = int(np.argmin(accuracies))
        best_index = int(np.argmax(accuracies))
        total_errors = int(sum(angle_results[angle_key(angle)]["errors"] for angle in angles))
        payload["splits"][split_name] = {
            "angles": angle_results,
            "summary": {
                "mean_accuracy": float(np.mean(accuracies)),
                "worst_accuracy": accuracies[worst_index],
                "worst_angle_deg": float(angles[worst_index]),
                "best_accuracy": accuracies[best_index],
                "best_angle_deg": float(angles[best_index]),
                "std_accuracy": float(np.std(accuracies)),
                "zero_degree_accuracy": float(angle_results[angle_key(0.0)]["accuracy"]),
                "total_errors": total_errors,
            },
        }
        summary = payload["splits"][split_name]["summary"]
        print(
            f"[dense-eval] {split_name} mean={summary['mean_accuracy']:.7f} "
            f"worst={summary['worst_accuracy']:.7f}@{summary['worst_angle_deg']:g}",
            flush=True,
        )
    return payload


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    architecture: str,
    class_count: int,
    image_size: int,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load a checkpoint onto its final device before entering eval mode.

    This ordering is required for escnn C8 models. R2Conv.train(False) expands its
    steerable basis immediately; in a process that has already trained a CUDA C8 model,
    escnn may reuse cached basis-expansion tensors on CUDA. Calling eval() while the new
    checkpoint weights are still on CPU therefore mixes CPU weights with CUDA basis
    tensors. Moving the whole module to the requested device first keeps the expansion
    device-consistent.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"Invalid classifier checkpoint: {checkpoint_path}")
    target_device = torch.device(device)
    model = build_experiment_model(
        architecture,
        class_count=class_count,
        image_size=image_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(target_device)
    model.eval()
    return model


def deploy_and_benchmark(
    condition: Condition,
    *,
    model: nn.Module,
    checkpoint_path: Path,
    run_dir: Path,
    cache: ExperimentCache,
    args: argparse.Namespace,
    repository_root: Path,
    production_onnx: Path,
) -> dict[str, Any]:
    deployment_dir = run_dir / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = deployment_dir / f"{condition.name}.onnx"

    if condition.production_reference and production_onnx.is_file():
        shutil.copy2(production_onnx, onnx_path)
        export_info: dict[str, Any] = {
            "kind": "copied_existing_production_onnx",
            "source": str(production_onnx),
        }
        # Do not move the live CUDA C8 instance back to CPU after dense evaluation.
        # Build a fresh CPU copy so escnn basis-expansion state is established on the
        # same device as the checkpoint weights before eval/parity.
        source = load_checkpoint_model(
            checkpoint_path,
            architecture="c8",
            class_count=len(cache.class_labels),
            image_size=cache.image_size,
            device="cpu",
        )
        parity = compare_pytorch_onnx(
            source,
            onnx_path,
            cache=cache,
            sample_count=16,
            seed=int(args.seed),
        )
    elif condition.architecture == "c8":
        export_info = export_c8_checkpoint(
            checkpoint_path,
            onnx_path=onnx_path,
            repository_root=repository_root,
            opset=int(args.opset),
        )
        source = load_checkpoint_model(
            checkpoint_path,
            architecture="c8",
            class_count=len(cache.class_labels),
            image_size=cache.image_size,
            device="cpu",
        )
        parity = compare_pytorch_onnx(
            source,
            onnx_path,
            cache=cache,
            sample_count=16,
            seed=int(args.seed),
        )
    else:
        export_info = export_custom_model(
            model.cpu(),
            onnx_path=onnx_path,
            image_size=cache.image_size,
            opset=int(args.opset),
        )
        parity = compare_pytorch_onnx(
            model.cpu(),
            onnx_path,
            cache=cache,
            sample_count=16,
            seed=int(args.seed),
        )

    if parity["prediction_mismatches"] != 0 or not parity["allclose"]:
        raise RuntimeError(f"ONNX parity failed for {condition.name}: {parity}")
    graph = analyze_onnx_graph(onnx_path, batch_size=int(args.benchmark_batch_size))
    benchmark = shared_architecture_benchmark(
        condition.architecture,
        onnx_path=onnx_path,
        graph=graph,
        output_root=run_dir.parent,
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
        "graph": graph,
        "benchmark": benchmark,
    }
    atomic_write_json(deployment_dir / "summary.json", deployment)
    return deployment


def export_c8_checkpoint(
    checkpoint_path: Path,
    *,
    onnx_path: Path,
    repository_root: Path,
    opset: int,
) -> dict[str, Any]:
    exporter = repository_root / "tools" / "recognition" / "export_c8_classifiers_onnx.py"
    metadata_path = onnx_path.with_suffix(onnx_path.suffix + ".metadata.json")
    command = [
        sys.executable,
        str(exporter),
        "--checkpoint",
        str(checkpoint_path),
        "--kind",
        "tile-shape",
        "--output",
        str(onnx_path),
        "--metadata-output",
        str(metadata_path),
        "--opset",
        str(opset),
        "--overwrite",
    ]
    process = subprocess.run(command, cwd=repository_root, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"C8 ONNX exporter failed with code {process.returncode}")
    return {
        "kind": "existing_c8_exporter",
        "command": command,
        "metadata": str(metadata_path),
    }


def export_custom_model(
    model: nn.Module,
    *,
    onnx_path: Path,
    image_size: int,
    opset: int,
) -> dict[str, Any]:
    import onnx

    model.eval()
    example = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    )
    checked = onnx.load(str(onnx_path))
    onnx.checker.check_model(checked)
    return {"kind": "torch.onnx.export", "opset": opset}


def parity_input(cache: ExperimentCache, *, sample_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train = cache.splits["train"]
    indices = rng.choice(train.count, size=sample_count, replace=False)
    if train.images_u8.device.type == "cuda":
        image_u8 = train.images_u8.index_select(
            0, torch.from_numpy(indices.astype(np.int64)).cuda()
        ).cpu()
    else:
        image_u8 = train.images_u8.index_select(0, torch.from_numpy(indices.astype(np.int64)))
    images = image_u8.numpy().astype(np.float32)[:, None, :, :] / 255.0
    return (images - np.float32(cache.mean)) / np.float32(cache.std)


def compare_pytorch_onnx(
    model: nn.Module,
    onnx_path: Path,
    *,
    cache: ExperimentCache,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    inputs = parity_input(cache, sample_count=sample_count, seed=seed)
    model.eval()
    with torch.inference_mode():
        expected = model(torch.from_numpy(inputs)).detach().cpu().numpy().astype(np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    observed = np.asarray(session.run([output_name], {input_name: inputs})[0], dtype=np.float32)
    difference = np.abs(expected - observed)
    return {
        "sample_count": sample_count,
        "allclose": bool(np.allclose(expected, observed, atol=1.0e-4, rtol=1.0e-4)),
        "atol": 1.0e-4,
        "rtol": 1.0e-4,
        "max_abs_error": float(difference.max()) if difference.size else 0.0,
        "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
        "prediction_mismatches": int(
            np.count_nonzero(expected.argmax(axis=1) != observed.argmax(axis=1))
        ),
    }


def analyze_onnx_graph(onnx_path: Path, *, batch_size: int) -> dict[str, Any]:
    import onnx
    from onnx import shape_inference

    model = onnx.load(str(onnx_path))
    inferred = shape_inference.infer_shapes(model)
    histogram = Counter(node.op_type for node in inferred.graph.node)
    initializer_shapes = {
        initializer.name: tuple(int(value) for value in initializer.dims)
        for initializer in inferred.graph.initializer
    }
    shapes = collect_onnx_shapes(inferred)
    conv_macs = 0
    gemm_macs = 0
    counted_conv_nodes = 0
    for node in inferred.graph.node:
        if node.op_type == "Conv" and len(node.input) >= 2 and node.input[1] in initializer_shapes:
            weight_shape = initializer_shapes[node.input[1]]
            output_shape = shapes.get(node.output[0])
            if len(weight_shape) == 4 and output_shape and len(output_shape) == 4:
                _, out_channels, out_h, out_w = concrete_shape(output_shape, batch_size=batch_size)
                if all(value is not None for value in (out_channels, out_h, out_w)):
                    per_output = weight_shape[1] * weight_shape[2] * weight_shape[3]
                    conv_macs += int(out_channels) * int(out_h) * int(out_w) * int(per_output)
                    counted_conv_nodes += 1
        elif node.op_type in {"Gemm", "MatMul"} and len(node.input) >= 2:
            weight_shape = initializer_shapes.get(node.input[1])
            if weight_shape and len(weight_shape) == 2:
                gemm_macs += int(weight_shape[0]) * int(weight_shape[1])

    peak_bytes = 0
    peak_name: str | None = None
    for name, shape in shapes.items():
        concrete = concrete_shape(shape, batch_size=batch_size)
        if any(value is None for value in concrete):
            continue
        elements = int(np.prod([int(value) for value in concrete], dtype=np.int64))
        size = elements * 4
        if size > peak_bytes:
            peak_bytes = size
            peak_name = name
    return {
        "operator_histogram": dict(sorted(histogram.items())),
        "conv_macs_per_sample_estimate": int(conv_macs),
        "gemm_matmul_macs_per_sample_estimate": int(gemm_macs),
        "known_macs_per_sample_estimate": int(conv_macs + gemm_macs),
        "counted_conv_nodes": counted_conv_nodes,
        "peak_intermediate_bytes_batch": batch_size,
        "peak_intermediate_bytes_estimate": peak_bytes,
        "peak_intermediate_value": peak_name,
        "note": (
            "MAC estimate covers ordinary Conv/Gemm/MatMul only; GridSample, TopK, "
            "trigonometric, gather and pooling work is intentionally reported through the operator histogram."
        ),
    }


def collect_onnx_shapes(model: Any) -> dict[str, tuple[int | str | None, ...]]:
    result: dict[str, tuple[int | str | None, ...]] = {}
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int | str | None] = []
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value") and dimension.dim_value > 0:
                dims.append(int(dimension.dim_value))
            elif dimension.HasField("dim_param") and dimension.dim_param:
                dims.append(str(dimension.dim_param))
            else:
                dims.append(None)
        result[value.name] = tuple(dims)
    return result


def concrete_shape(
    shape: Sequence[int | str | None],
    *,
    batch_size: int,
) -> tuple[int | None, ...]:
    result: list[int | None] = []
    for index, value in enumerate(shape):
        if isinstance(value, int):
            result.append(value)
        elif isinstance(value, str) and (value == "batch" or index == 0):
            result.append(batch_size)
        else:
            result.append(None)
    return tuple(result)


def shared_architecture_benchmark(
    architecture: str,
    *,
    onnx_path: Path,
    graph: dict[str, Any],
    output_root: Path,
    batch_size: int,
    image_size: int,
    warmup: int,
    runs: int,
    seed: int,
) -> dict[str, Any]:
    shared_root = output_root / "_architecture_benchmarks"
    shared_root.mkdir(parents=True, exist_ok=True)
    shared_path = shared_root / f"{architecture}.json"
    prior = read_json_if_exists(shared_path)
    signature = {
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "operator_histogram": graph.get("operator_histogram"),
        "known_macs_per_sample_estimate": graph.get("known_macs_per_sample_estimate"),
        "benchmark_batch_size": batch_size,
        "warmup": warmup,
        "runs": runs,
    }
    if (
        prior is not None
        and prior.get("status") == "completed"
        and prior.get("graph_signature") == signature
    ):
        reused = dict(prior["benchmark"])
        reused["reused_architecture_benchmark"] = True
        reused["shared_result"] = str(shared_path)
        return reused

    benchmark = benchmark_onnx_cpu(
        onnx_path,
        batch_size=batch_size,
        image_size=image_size,
        warmup=warmup,
        runs=runs,
        seed=seed,
    )
    atomic_write_json(
        shared_path,
        {
            "status": "completed",
            "architecture": architecture,
            "source_onnx": str(onnx_path),
            "graph_signature": signature,
            "benchmark": benchmark,
        },
    )
    benchmark = dict(benchmark)
    benchmark["reused_architecture_benchmark"] = False
    benchmark["shared_result"] = str(shared_path)
    return benchmark


def benchmark_onnx_cpu(
    onnx_path: Path,
    *,
    batch_size: int,
    image_size: int,
    warmup: int,
    runs: int,
    seed: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((batch_size, 1, image_size, image_size), dtype=np.float32)
    for _ in range(warmup):
        session.run([output_name], {input_name: inputs})
    samples_ms: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run([output_name], {input_name: inputs})
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
    median = float(statistics.median(samples_ms))
    mean = float(statistics.fmean(samples_ms))
    return {
        "provider": "CPUExecutionProvider",
        "batch_size": batch_size,
        "warmup_runs": warmup,
        "measurement_runs": runs,
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "execution_mode": "sequential",
        "mean_ms_per_batch": mean,
        "median_ms_per_batch": median,
        "p95_ms_per_batch": float(ordered[p95_index]),
        "median_ms_per_image": median / batch_size,
        "onnxruntime_version": ort.__version__,
        "cpu": platform.processor(),
    }


def angle_key(angle: float) -> str:
    value = float(angle)
    return f"{int(value)}deg" if value.is_integer() else f"{value:g}deg"


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import onnx

        info["onnx"] = onnx.__version__
    except ImportError:
        info["onnx"] = None
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
    except ImportError:
        info["onnxruntime"] = None
    return info


def write_aggregate_summary(
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
    accuracy_rows: list[dict[str, Any]] = []
    deployment_rows: list[dict[str, Any]] = []
    deployment_failure_count = 0
    for condition in conditions:
        result = results.get(condition.name)
        if not result or result.get("status") != "completed":
            continue
        accuracy = result.get("accuracy", {}).get("splits", {})
        manual = accuracy.get("manual_val", {}).get("summary", {})
        jp = accuracy.get("jp_val", {}).get("summary", {})
        accuracy_rows.append(
            {
                "condition": condition.name,
                "architecture": condition.architecture,
                "augmentation": condition.augmentation,
                "manual_mean_accuracy": manual.get("mean_accuracy"),
                "manual_worst_accuracy": manual.get("worst_accuracy"),
                "manual_worst_angle_deg": manual.get("worst_angle_deg"),
                "jp_mean_accuracy": jp.get("mean_accuracy"),
                "jp_worst_accuracy": jp.get("worst_accuracy"),
                "jp_worst_angle_deg": jp.get("worst_angle_deg"),
            }
        )
        deployment = result.get("deployment", {})
        if deployment.get("status") == "failed":
            deployment_failure_count += 1
        if deployment.get("status") == "completed":
            graph = deployment.get("graph", {})
            benchmark = deployment.get("benchmark", {})
            deployment_rows.append(
                {
                    "condition": condition.name,
                    "architecture": condition.architecture,
                    "onnx_bytes": deployment.get("onnx_bytes"),
                    "known_macs_per_sample_estimate": graph.get(
                        "known_macs_per_sample_estimate"
                    ),
                    "peak_intermediate_bytes_estimate": graph.get(
                        "peak_intermediate_bytes_estimate"
                    ),
                    "operator_histogram": graph.get("operator_histogram"),
                    "ort_cpu_median_ms_batch": benchmark.get("median_ms_per_batch"),
                    "ort_cpu_p95_ms_batch": benchmark.get("p95_ms_per_batch"),
                }
            )
    status = "completed" if final and not pending else "in_progress"
    if final and status_counts.get("failed", 0):
        status = "completed_with_failures"
    elif final and deployment_failure_count:
        status = "completed_with_deployment_failures"
    summary = {
        "status": status,
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "status_counts": dict(status_counts),
        "deployment_failure_count": deployment_failure_count,
        "pending": pending,
        "accuracy_rows": accuracy_rows,
        "deployment_rows": deployment_rows,
        "results": {name: result.get("status") for name, result in results.items()},
    }
    atomic_write_json(output_root / "summary.json", summary)
    return summary


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def read_last_json_line(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def append_json_line(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        output.write("\n")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
