from __future__ import annotations

import hashlib
import inspect
import json
import math
import statistics
import time
from pathlib import Path

import torch

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate


_IMAGE_SHAPE = (1, 64, 64)
_OPSET_VERSION = 16


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[rank - 1])


def _export_onnx(model: torch.nn.Module, path: Path, *, batch_size: int) -> None:
    model = model.to("cpu").eval()
    example = torch.zeros((batch_size, *_IMAGE_SHAPE), dtype=torch.float32)
    kwargs: dict[str, object] = {
        "export_params": True,
        "opset_version": _OPSET_VERSION,
        "do_constant_folding": True,
        "input_names": ["images"],
        "output_names": ["logits"],
        "dynamic_axes": {"images": {0: "batch"}, "logits": {0: "batch"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(model, example, str(path), **kwargs)


def _benchmark_onnx(
    model_path: Path,
    *,
    batch_size: int,
    warmup_runs: int,
    measure_runs: int,
    intra_op_threads: int,
    inter_op_threads: int,
) -> tuple[list[float], dict[str, object]]:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnx and onnxruntime are required for CPU ONNX latency evaluation") from error

    onnx.checker.check_model(onnx.load(str(model_path)))

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"CPU-only ONNX Runtime provider contract not satisfied: {session.get_providers()}")
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise RuntimeError("latency benchmark expects exactly one ONNX input and one output")

    input_name = session.get_inputs()[0].name
    input_tensor = torch.zeros((batch_size, *_IMAGE_SHAPE), dtype=torch.float32).numpy()

    for _ in range(warmup_runs):
        session.run(None, {input_name: input_tensor})

    samples_ms: list[float] = []
    for _ in range(measure_runs):
        started = time.perf_counter_ns()
        session.run(None, {input_name: input_tensor})
        ended = time.perf_counter_ns()
        samples_ms.append((ended - started) / 1_000_000.0)

    environment = {
        "onnxruntime_version": ort.__version__,
        "available_providers": list(ort.get_available_providers()),
        "active_providers": list(session.get_providers()),
        "intra_op_threads": intra_op_threads,
        "inter_op_threads": inter_op_threads,
        "execution_mode": "ORT_SEQUENTIAL",
        "graph_optimization_level": "ORT_ENABLE_ALL",
    }
    return samples_ms, environment


def evaluate(context):
    parameters = context.parameters
    batch_size = int(parameters["batch_size"])
    warmup_runs = int(parameters["warmup_runs"])
    measure_runs = int(parameters["measure_runs"])
    intra_op_threads = int(parameters["intra_op_threads"])
    inter_op_threads = int(parameters["inter_op_threads"])
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if warmup_runs < 1:
        raise ValueError("warmup_runs must be at least 1")
    if measure_runs < 10:
        raise ValueError("measure_runs must be at least 10")
    if intra_op_threads < 1 or inter_op_threads < 1:
        raise ValueError("ONNX Runtime thread counts must be at least 1")

    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = work_dir / "model.onnx"
    report_path = work_dir / "onnx-cpu-latency.json"

    _export_onnx(context.model.module, onnx_path, batch_size=batch_size)
    samples_ms, environment = _benchmark_onnx(
        onnx_path,
        batch_size=batch_size,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
    )

    p50_ms = float(statistics.median(samples_ms))
    p95_ms = _nearest_rank_percentile(samples_ms, 0.95)
    mean_ms = float(statistics.fmean(samples_ms))
    metrics = {
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "latency_mean_ms": mean_ms,
    }

    onnx_bytes = onnx_path.read_bytes()
    report = {
        "schema": "mjtensu.recognition/tile-onnx-cpu-latency/v1",
        "measurement_scope": "onnxruntime session.run only; excludes ONNX export, session creation, and preprocessing",
        "input": {
            "shape": [batch_size, *_IMAGE_SHAPE],
            "dtype": "float32",
            "data": "zeros",
        },
        "export": {
            "opset_version": _OPSET_VERSION,
            "dynamic_batch": True,
            "onnx_bytes": len(onnx_bytes),
            "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        },
        "benchmark": {
            "batch_size": batch_size,
            "warmup_runs": warmup_runs,
            "measure_runs": measure_runs,
            "p50_definition": "median of measured session.run durations",
            "p95_definition": "nearest-rank 95th percentile of measured session.run durations",
            "metrics_ms": metrics,
            "minimum_ms": float(min(samples_ms)),
            "maximum_ms": float(max(samples_ms)),
            "samples_ms": samples_ms,
        },
        "environment": environment,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return EvaluationCandidate(
        metrics=metrics,
        artifacts={
            "onnx_model": onnx_path,
            "latency_report": report_path,
        },
    )
