from __future__ import annotations

"""Run PRODUCT-INV-RECOGNITION-011 mobile classifier comparison.

This runner deliberately reuses the frozen INV-007/008 data, augmentation, dense-angle
accuracy, ONNX graph, and CPU benchmark helpers without changing their implementation.
Model construction/training recovery is INV-011-specific so historical experiments stay
reproducible.
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
from torch import nn

try:
    from mobile_classifier_experiment_models import (
        build_mobile_classifier,
        describe_mobile_classifier,
    )
    from run_rotation_classifier_experiment import (
        CHECKPOINT_ANGLES,
        DENSE_ANGLES,
        analyze_onnx_graph,
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
        load_cache,
        read_json_if_exists,
        read_last_json_line,
        save_checkpoint,
        seed_everything,
        train_one_epoch,
    )
except ModuleNotFoundError:  # package-style import used by tests
    from tools.recognition.mobile_classifier_experiment_models import (
        build_mobile_classifier,
        describe_mobile_classifier,
    )
    from tools.recognition.run_rotation_classifier_experiment import (
        CHECKPOINT_ANGLES,
        DENSE_ANGLES,
        analyze_onnx_graph,
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
        load_cache,
        read_json_if_exists,
        read_last_json_line,
        save_checkpoint,
        seed_everything,
        train_one_epoch,
    )


EXPERIMENT_IMPLEMENTATION_VERSION = "inv011-mobile-v1"
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


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    width_mult: float


CONDITIONS: tuple[Condition, ...] = (
    Condition("shufflenet-v2-0.5x", "shufflenet-v2", 0.5),
    Condition("shufflenet-v2-1.0x", "shufflenet-v2", 1.0),
    Condition("mobilenet-v3-small-0.5x", "mobilenet-v3-small", 0.5),
    Condition("mobilenet-v3-small-1.0x", "mobilenet-v3-small", 1.0),
)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run INV-011 mobile-oriented gray64 tile-classifier comparison."
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
        default=repository_root / ".local" / "recognition" / "mobile_classifier_experiment",
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
        help="Optional subset. Default: all four mobile candidates.",
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
        raise ValueError("INV-011 requires ONNX opset >= 16")
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
            "Missing INV-011 dependencies: "
            + ", ".join(missing)
            + ". Install them into the existing classifier CUDA environment."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_runtime_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for INV-011 training/evaluation")

    repository_root = args.repository_root.resolve()
    database = args.database.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configure_cuda(tf32=not bool(args.no_tf32))
    seed_everything(int(args.seed))
    cache = load_cache(database, cache_device=str(args.cache_device))
    assert_v3_contract(cache)
    conditions = selected_conditions(args)

    reference = build_plain_reference(args, image_size=cache.image_size)
    atomic_write_json(output_root / "plain_e150_reference.json", reference)
    atomic_write_json(
        output_root / "manifest.json",
        {
            "status": "in_progress",
            "investigation": "PRODUCT-INV-RECOGNITION-011",
            "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
            "repository_root": str(repository_root),
            "database": str(database),
            "output_root": str(output_root),
            "conditions": [asdict(condition) for condition in conditions],
            "plain_reference": reference,
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
        if (
            prior_result_is_reusable(condition, prior)
            and not bool(args.overwrite_completed)
        ):
            print(f"[resume] skip completed {condition.name}", flush=True)
            results[condition.name] = prior
            write_summary(output_root, conditions, results, reference=reference)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(f"\n===== {condition.name} =====", flush=True)
        try:
            result = run_condition(
                condition,
                run_dir=run_dir,
                cache=cache,
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
        atomic_write_json(result_path, result)
        results[condition.name] = result
        write_summary(output_root, conditions, results, reference=reference)
        if result["status"] != "completed" and bool(args.fail_fast):
            raise RuntimeError(f"Condition failed: {condition.name}: {result.get('error')}")

    summary = write_summary(
        output_root,
        conditions,
        results,
        reference=reference,
        final=True,
    )
    manifest = read_json_if_exists(output_root / "manifest.json") or {}
    manifest["status"] = summary["status"]
    atomic_write_json(output_root / "manifest.json", manifest)
    print("\n===== INV-011 experiment finished =====", flush=True)
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

    deployment = deploy_and_benchmark(
        condition,
        model=model,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        cache=cache,
        args=args,
    )
    return {
        "condition": asdict(condition),
        "training": training,
        "accuracy": dense_accuracy,
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
    model = build_mobile_classifier(
        condition.name,
        class_count=len(cache.class_labels),
    ).to(device)
    description = describe_mobile_classifier(model, condition.name)
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
            "width_mult": description.width_mult,
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
                        manual_validation[_angle_key(angle)]["accuracy"]
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
    best_model = load_mobile_checkpoint_model(
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
    normalization = config.get("normalization")
    if normalization != {"mean": cache.mean, "std": cache.std}:
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
    model = load_mobile_checkpoint_model(
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


def load_mobile_checkpoint_model(
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
        raise ValueError(f"Invalid mobile classifier checkpoint: {checkpoint_path}")
    model = build_mobile_classifier(model_name, class_count=class_count)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(torch.device(device))
    model.eval()
    return model


def deploy_and_benchmark(
    condition: Condition,
    *,
    model: nn.Module,
    checkpoint_path: Path,
    run_dir: Path,
    cache: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    deployment_dir = run_dir / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = deployment_dir / f"{condition.name}.onnx"

    cpu_model = load_mobile_checkpoint_model(
        checkpoint_path,
        model_name=condition.name,
        class_count=len(cache.class_labels),
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
        batch_sizes=(1, 16),
        seed=int(args.seed),
    )
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


def smoke_onnx_dynamic_batch(
    onnx_path: Path,
    *,
    image_size: int,
    batch_sizes: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}
    for batch_size in batch_sizes:
        inputs = rng.standard_normal(
            (int(batch_size), 1, image_size, image_size), dtype=np.float32
        )
        output = np.asarray(
            session.run([output_meta.name], {input_meta.name: inputs})[0]
        )
        expected_shape = (int(batch_size), 35)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"Dynamic batch smoke returned {tuple(output.shape)} for N={batch_size}; "
                f"expected {expected_shape}"
            )
        results[str(batch_size)] = {
            "input_shape": list(inputs.shape),
            "output_shape": list(output.shape),
        }
    return {
        "input_name": input_meta.name,
        "input_shape": list(input_meta.shape),
        "output_name": output_meta.name,
        "output_shape": list(output_meta.shape),
        "batches": results,
    }


def analyze_mobile_onnx_graph(onnx_path: Path, *, batch_size: int) -> dict[str, Any]:
    import onnx

    base = analyze_onnx_graph(onnx_path, batch_size=batch_size)
    model = onnx.load(str(onnx_path))
    initializer_shapes = {
        initializer.name: tuple(int(value) for value in initializer.dims)
        for initializer in model.graph.initializer
    }
    depthwise_conv_count = 0
    pointwise_conv_count = 0
    grouped_conv_count = 0
    ordinary_conv_count = 0
    conv_details: list[dict[str, Any]] = []
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        weight_shape = initializer_shapes.get(node.input[1])
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        groups = int(attributes.get("group", 1))
        kernel = tuple(int(value) for value in (weight_shape[2:] if weight_shape and len(weight_shape) == 4 else ()))
        out_channels = int(weight_shape[0]) if weight_shape and len(weight_shape) == 4 else None
        in_channels_per_group = int(weight_shape[1]) if weight_shape and len(weight_shape) == 4 else None
        is_depthwise = bool(
            groups > 1
            and out_channels is not None
            and in_channels_per_group == 1
            and groups == out_channels
        )
        is_pointwise = kernel == (1, 1) and groups == 1
        if is_depthwise:
            depthwise_conv_count += 1
        elif groups > 1:
            grouped_conv_count += 1
        else:
            ordinary_conv_count += 1
        if is_pointwise:
            pointwise_conv_count += 1
        conv_details.append(
            {
                "name": node.name,
                "groups": groups,
                "kernel": list(kernel),
                "out_channels": out_channels,
                "in_channels_per_group": in_channels_per_group,
                "depthwise": is_depthwise,
                "pointwise": is_pointwise,
            }
        )
    return {
        **base,
        "depthwise_conv_count": depthwise_conv_count,
        "pointwise_conv_count": pointwise_conv_count,
        "other_grouped_conv_count": grouped_conv_count,
        "ordinary_conv_count": ordinary_conv_count,
        "conv_details": conv_details,
    }


def build_plain_reference(args: argparse.Namespace, *, image_size: int) -> dict[str, Any]:
    path = args.plain_reference_onnx.resolve()
    payload: dict[str, Any] = {
        "name": "plain-random360-e150",
        "accuracy": dict(PLAIN_E150_ACCURACY_REFERENCE),
        "onnx": str(path),
        "onnx_exists": path.is_file(),
    }
    if not path.is_file():
        payload["deployment"] = {
            "status": "unavailable",
            "reason": "reference ONNX does not exist on this machine",
        }
        return payload
    try:
        graph = analyze_mobile_onnx_graph(
            path,
            batch_size=int(args.benchmark_batch_size),
        )
        smoke = smoke_onnx_dynamic_batch(
            path,
            image_size=image_size,
            batch_sizes=(1, 16),
            seed=int(args.seed),
        )
        benchmark = benchmark_onnx_cpu(
            path,
            batch_size=int(args.benchmark_batch_size),
            image_size=image_size,
            warmup=int(args.benchmark_warmup),
            runs=int(args.benchmark_runs),
            seed=int(args.seed),
        )
        payload["deployment"] = {
            "status": "completed",
            "onnx_bytes": path.stat().st_size,
            "dynamic_batch_smoke": smoke,
            "graph": graph,
            "benchmark": benchmark,
        }
    except Exception as error:
        payload["deployment"] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    return payload


def write_summary(
    output_root: Path,
    conditions: Sequence[Condition],
    results: dict[str, Any],
    *,
    reference: dict[str, Any],
    final: bool = False,
) -> dict[str, Any]:
    status_counts = Counter(
        str(result.get("status", "unknown")) for result in results.values()
    )
    pending = [condition.name for condition in conditions if condition.name not in results]
    accuracy_rows: list[dict[str, Any]] = [
        {
            "condition": "plain-random360-e150-reference",
            "family": "plain",
            "width_mult": None,
            **PLAIN_E150_ACCURACY_REFERENCE,
        }
    ]
    deployment_rows: list[dict[str, Any]] = []
    reference_deployment = reference.get("deployment", {})
    if reference_deployment.get("status") == "completed":
        graph = reference_deployment.get("graph", {})
        benchmark = reference_deployment.get("benchmark", {})
        deployment_rows.append(
            {
                "condition": "plain-random360-e150-reference",
                "family": "plain",
                "width_mult": None,
                "onnx_bytes": reference_deployment.get("onnx_bytes"),
                "known_macs_per_sample_estimate": graph.get(
                    "known_macs_per_sample_estimate"
                ),
                "depthwise_conv_count": graph.get("depthwise_conv_count"),
                "pointwise_conv_count": graph.get("pointwise_conv_count"),
                "ort_cpu_median_ms_batch": benchmark.get("median_ms_per_batch"),
                "ort_cpu_p95_ms_batch": benchmark.get("p95_ms_per_batch"),
            }
        )

    for condition in conditions:
        result = results.get(condition.name)
        if not result or result.get("status") != "completed":
            continue
        splits = result.get("accuracy", {}).get("splits", {})
        manual = splits.get("manual_val", {}).get("summary", {})
        jp = splits.get("jp_val", {}).get("summary", {})
        accuracy_rows.append(
            {
                "condition": condition.name,
                "family": condition.family,
                "width_mult": condition.width_mult,
                "manual_mean_accuracy": manual.get("mean_accuracy"),
                "manual_worst_accuracy": manual.get("worst_accuracy"),
                "manual_worst_angle_deg": manual.get("worst_angle_deg"),
                "jp_mean_accuracy": jp.get("mean_accuracy"),
                "jp_worst_accuracy": jp.get("worst_accuracy"),
                "jp_worst_angle_deg": jp.get("worst_angle_deg"),
            }
        )
        deployment = result.get("deployment", {})
        if deployment.get("status") == "completed":
            graph = deployment.get("graph", {})
            benchmark = deployment.get("benchmark", {})
            deployment_rows.append(
                {
                    "condition": condition.name,
                    "family": condition.family,
                    "width_mult": condition.width_mult,
                    "onnx_bytes": deployment.get("onnx_bytes"),
                    "known_macs_per_sample_estimate": graph.get(
                        "known_macs_per_sample_estimate"
                    ),
                    "depthwise_conv_count": graph.get("depthwise_conv_count"),
                    "pointwise_conv_count": graph.get("pointwise_conv_count"),
                    "peak_intermediate_bytes_estimate": graph.get(
                        "peak_intermediate_bytes_estimate"
                    ),
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
        "accuracy_rows": accuracy_rows,
        "deployment_rows": deployment_rows,
        "results": {name: result.get("status") for name, result in results.items()},
        "plain_reference": reference,
    }
    atomic_write_json(output_root / "summary.json", summary)
    return summary


def _angle_key(angle: float) -> str:
    value = float(angle)
    return f"{int(value)}deg" if value.is_integer() else f"{value:g}deg"


if __name__ == "__main__":
    main()
