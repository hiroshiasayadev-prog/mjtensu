from __future__ import annotations

import hashlib
import inspect
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate


_IMAGE_SHAPE = (1, 64, 64)
_OPSET_VERSION = 16
_PARITY_SAMPLES = 4
_PARITY_ATOL = 1.0e-4
_PARITY_RTOL = 1.0e-4


class _CyclicGroupMaxPool(nn.Module):
    def __init__(self, group_size: int) -> None:
        super().__init__()
        self.group_size = int(group_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim != 4:
            raise ValueError(f"Expected NCHW tensor, got {tuple(input_tensor.shape)}")
        batch, channels, height, width = input_tensor.shape
        if channels % self.group_size != 0:
            raise ValueError(
                f"Channel count {channels} is not divisible by cyclic group size {self.group_size}"
            )
        fields = channels // self.group_size
        grouped = input_tensor.reshape(
            batch, fields, self.group_size, height, width
        )
        return grouped.amax(dim=2)


class _ExportedCyclicClassifier(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        spatial_pool: nn.Module,
        classifier: nn.Module,
        group_size: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.group_pool = _CyclicGroupMaxPool(group_size)
        self.spatial_pool = spatial_pool
        self.classifier = classifier

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        invariant = self.group_pool(features)
        pooled = self.spatial_pool(invariant)
        return self.classifier(pooled)


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[rank - 1])


def _prepare_exportable_model(
    model: nn.Module,
) -> tuple[nn.Module, str]:
    source = model.to("cpu").eval()
    backbone = getattr(source, "equivariant_backbone", None)
    group_pool = getattr(source, "group_pool", None)
    spatial_pool = getattr(source, "spatial_pool", None)
    classifier = getattr(source, "classifier", None)
    field_counts = getattr(source, "field_counts", None)

    cyclic_parts = (backbone, group_pool, spatial_pool, classifier, field_counts)
    if all(part is not None for part in cyclic_parts):
        export = getattr(backbone, "export", None)
        in_type = getattr(group_pool, "in_type", None)
        out_type = getattr(group_pool, "out_type", None)
        in_size = getattr(in_type, "size", None)
        out_size = getattr(out_type, "size", None)
        if not callable(export) or not isinstance(field_counts, tuple):
            raise TypeError("Cyclic classifier export contract is malformed")
        if in_size is None or out_size is None:
            raise TypeError("Cyclic GroupPooling type sizes are unavailable")
        tensor_channels = int(in_size)
        invariant_channels = int(out_size)
        if invariant_channels != int(field_counts[-1]):
            raise ValueError(
                "Cyclic GroupPooling output does not match final field count"
            )
        if invariant_channels < 1 or tensor_channels % invariant_channels != 0:
            raise ValueError(
                "Cyclic GroupPooling input width is not an integer multiple of output width"
            )
        group_size = tensor_channels // invariant_channels
        if group_size < 2:
            raise ValueError("Cyclic group size must be at least 2")
        return (
            _ExportedCyclicClassifier(
                backbone=export(),
                spatial_pool=spatial_pool,
                classifier=classifier,
                group_size=group_size,
            ).eval(),
            "cyclic-equivariant-tensor-export",
        )

    return source, "direct-torch-export"


def _parity_input() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    return torch.rand(
        (_PARITY_SAMPLES, *_IMAGE_SHAPE),
        generator=generator,
        dtype=torch.float32,
    )


def _compare_torch_models(
    source: nn.Module,
    exported: nn.Module,
) -> dict[str, object]:
    inputs = _parity_input()
    with torch.no_grad():
        expected = source(inputs)
        actual = exported(inputs)
    delta = (expected - actual).abs()
    allclose = bool(
        torch.allclose(
            expected,
            actual,
            atol=_PARITY_ATOL,
            rtol=_PARITY_RTOL,
        )
    )
    prediction_mismatches = int(
        (expected.argmax(dim=1) != actual.argmax(dim=1)).sum().item()
    )
    if not allclose or prediction_mismatches:
        raise RuntimeError("Tensor-only export does not match source model")
    return {
        "samples": _PARITY_SAMPLES,
        "atol": _PARITY_ATOL,
        "rtol": _PARITY_RTOL,
        "allclose": allclose,
        "prediction_mismatches": prediction_mismatches,
        "max_abs_error": float(delta.max().item()),
    }


def _export_onnx(
    model: nn.Module,
    path: Path,
    *,
    batch_size: int,
) -> tuple[nn.Module, dict[str, object]]:
    source = model.to("cpu").eval()
    export_model, mode = _prepare_exportable_model(source)
    torch_parity = _compare_torch_models(source, export_model)

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
    torch.onnx.export(export_model, example, str(path), **kwargs)
    return source, {"mode": mode, "torch_parity": torch_parity}


def _onnx_parity(
    source: nn.Module,
    session,
    input_name: str,
) -> dict[str, object]:
    inputs = _parity_input()
    with torch.no_grad():
        expected = source(inputs)
    actual_array = session.run(None, {input_name: inputs.numpy()})[0]
    actual = torch.from_numpy(actual_array)
    delta = (expected - actual).abs()
    allclose = bool(
        torch.allclose(
            expected,
            actual,
            atol=_PARITY_ATOL,
            rtol=_PARITY_RTOL,
        )
    )
    prediction_mismatches = int(
        (expected.argmax(dim=1) != actual.argmax(dim=1)).sum().item()
    )
    if not allclose or prediction_mismatches:
        raise RuntimeError("ONNX Runtime output does not match source model")
    return {
        "samples": _PARITY_SAMPLES,
        "atol": _PARITY_ATOL,
        "rtol": _PARITY_RTOL,
        "allclose": allclose,
        "prediction_mismatches": prediction_mismatches,
        "max_abs_error": float(delta.max().item()),
    }


def _benchmark_onnx(
    model_path: Path,
    *,
    source_model: nn.Module,
    batch_size: int,
    warmup_runs: int,
    measure_runs: int,
    intra_op_threads: int,
    inter_op_threads: int,
) -> tuple[list[float], dict[str, object], dict[str, object]]:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnx and onnxruntime are required for CPU ONNX latency evaluation"
        ) from error

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
        raise RuntimeError(
            f"CPU-only ONNX Runtime provider contract not satisfied: {session.get_providers()}"
        )
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise RuntimeError("latency benchmark expects exactly one ONNX input and one output")

    input_name = session.get_inputs()[0].name
    parity = _onnx_parity(source_model, session, input_name)
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
    return samples_ms, environment, parity


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

    source_model, export_info = _export_onnx(
        context.model.module,
        onnx_path,
        batch_size=batch_size,
    )
    samples_ms, environment, onnx_parity = _benchmark_onnx(
        onnx_path,
        source_model=source_model,
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
        "schema": "mjtensu.recognition/tile-onnx-cpu-latency/v2",
        "measurement_scope": (
            "onnxruntime session.run only; excludes ONNX export, parity checks, "
            "session creation, and preprocessing"
        ),
        "input": {
            "shape": [batch_size, *_IMAGE_SHAPE],
            "dtype": "float32",
            "data": "zeros",
        },
        "export": {
            "mode": export_info["mode"],
            "opset_version": _OPSET_VERSION,
            "dynamic_batch": True,
            "onnx_bytes": len(onnx_bytes),
            "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
            "torch_parity": export_info["torch_parity"],
            "onnx_parity": onnx_parity,
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

