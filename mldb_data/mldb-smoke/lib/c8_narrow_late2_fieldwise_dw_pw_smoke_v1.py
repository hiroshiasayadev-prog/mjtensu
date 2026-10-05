from __future__ import annotations

import base64
import hashlib
import inspect
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate


GROUP_SIZE = 8
FIELD_COUNTS = (4, 8, 16, 32)
CLASS_COUNT = 35
IMAGE_SHAPE = (1, 64, 64)
OPSET_VERSION = 16
PARITY_SAMPLES = 4
PARITY_ATOL = 1.0e-4
PARITY_RTOL = 1.0e-4
PROTOTYPE_SEED = 42
EXPECTED_CONV_KERNEL_GROUP_SIGNATURE = (
    ((5, 5), 1),
    ((3, 3), 1),
    ((3, 3), 8),
    ((1, 1), 1),
    ((3, 3), 16),
    ((1, 1), 1),
)

ORT_WEB_VERSION = "1.27.0"
ORT_WEB_DIST_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/"
ORT_WEB_MODULE_URL = ORT_WEB_DIST_URL + "ort.wasm.min.mjs"
DEFAULT_RUNNER_URL = "http://192.168.11.22:8877"
RUNNER_RESULT_TIMEOUT_SEC = 300
RUNNER_HTTP_TIMEOUT_SEC = 20
RUNNER_POLL_INTERVAL_SEC = 0.25


class C8NarrowLateFieldwiseSeparablePrototype(nn.Module):
    """Untrained C8-narrow prototype with field-wise separable stage 3 and 4."""

    def __init__(self) -> None:
        super().__init__()
        try:
            from escnn import gspaces
            from escnn import nn as enn
        except ImportError as error:
            raise RuntimeError(
                "C8 grouped-R2Conv smoke requires escnn==1.0.11-compatible semantics."
            ) from error

        self.group_size = GROUP_SIZE
        self.field_counts = FIELD_COUNTS
        self.tensor_channels = tuple(GROUP_SIZE * count for count in FIELD_COUNTS)
        self.prototype_seed = PROTOTYPE_SEED
        self._enn = enn
        self.gspace = gspaces.rot2dOnR2(GROUP_SIZE)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])

        layers: list[nn.Module] = []
        in_type = self.input_type
        for block_index, field_count in enumerate(FIELD_COUNTS):
            out_type = enn.FieldType(
                self.gspace,
                [self.gspace.regular_repr] * field_count,
            )
            kernel_size = 5 if block_index == 0 else 3
            padding = kernel_size // 2

            if block_index >= 2:
                depthwise_type = in_type
                layers.extend(
                    [
                        enn.R2Conv(
                            in_type,
                            depthwise_type,
                            kernel_size=3,
                            padding=1,
                            groups=len(in_type),
                            bias=False,
                        ),
                        enn.InnerBatchNorm(depthwise_type),
                        enn.ReLU(depthwise_type, inplace=True),
                        enn.R2Conv(
                            depthwise_type,
                            out_type,
                            kernel_size=1,
                            padding=0,
                            bias=False,
                        ),
                        enn.InnerBatchNorm(out_type),
                        enn.ReLU(out_type, inplace=True),
                    ]
                )
            else:
                layers.extend(
                    [
                        enn.R2Conv(
                            in_type,
                            out_type,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                        enn.InnerBatchNorm(out_type),
                        enn.ReLU(out_type, inplace=True),
                    ]
                )

            if block_index < len(FIELD_COUNTS) - 1:
                layers.append(
                    enn.PointwiseMaxPool(
                        out_type,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )
            in_type = out_type

        self.equivariant_backbone = enn.SequentialModule(*layers)
        self.group_pool = enn.GroupPooling(in_type)
        invariant_channels = self.group_pool.out_type.size
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = max(128, invariant_channels * 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(invariant_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, CLASS_COUNT),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(
                f"Expected grayscale NCHW tensor [N,1,H,W], got {tuple(images.shape)}"
            )
        geometric = self._enn.GeometricTensor(images, self.input_type)
        features = self.equivariant_backbone(geometric)
        invariant = self.group_pool(features).tensor
        pooled = self.spatial_pool(invariant)
        return self.classifier(pooled)


def build_prototype() -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(PROTOTYPE_SEED)
        return C8NarrowLateFieldwiseSeparablePrototype()


class _CyclicGroupMaxPool(nn.Module):
    def __init__(self, group_size: int) -> None:
        super().__init__()
        self.group_size = int(group_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = input_tensor.shape
        fields = channels // self.group_size
        return input_tensor.reshape(
            batch, fields, self.group_size, height, width
        ).amax(dim=2)


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
        return self.classifier(self.spatial_pool(invariant))


def _prepare_exportable_model(model: nn.Module) -> tuple[nn.Module, dict[str, object]]:
    source = model.to("cpu").eval()
    backbone = source.equivariant_backbone
    exported_backbone = backbone.export()
    tensor_channels = int(source.group_pool.in_type.size)
    invariant_channels = int(source.group_pool.out_type.size)
    group_size = tensor_channels // invariant_channels

    torch_convs = []
    for name, module in exported_backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            torch_convs.append(
                {
                    "name": name,
                    "in_channels": int(module.in_channels),
                    "out_channels": int(module.out_channels),
                    "kernel_size": [int(v) for v in module.kernel_size],
                    "groups": int(module.groups),
                }
            )

    return (
        _ExportedCyclicClassifier(
            backbone=exported_backbone,
            spatial_pool=source.spatial_pool,
            classifier=source.classifier,
            group_size=group_size,
        ).eval(),
        {
            "mode": "cyclic-equivariant-tensor-export",
            "torch_exported_convs": torch_convs,
        },
    )


def _parity_input() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    return torch.rand(
        (PARITY_SAMPLES, *IMAGE_SHAPE),
        generator=generator,
        dtype=torch.float32,
    )


def _compare_torch_models(source: nn.Module, exported: nn.Module) -> dict[str, object]:
    inputs = _parity_input()
    with torch.no_grad():
        expected = source(inputs)
        actual = exported(inputs)
    delta = (expected - actual).abs()
    allclose = bool(
        torch.allclose(expected, actual, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    )
    prediction_mismatches = int(
        (expected.argmax(dim=1) != actual.argmax(dim=1)).sum().item()
    )
    if not allclose or prediction_mismatches:
        raise RuntimeError("Tensor-only export does not match source prototype")
    return {
        "samples": PARITY_SAMPLES,
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "allclose": allclose,
        "prediction_mismatches": prediction_mismatches,
        "max_abs_error": float(delta.max().item()),
    }


def _onnx_conv_inventory(path: Path) -> list[dict[str, object]]:
    import onnx

    graph = onnx.load(str(path)).graph
    initializers = {item.name: list(item.dims) for item in graph.initializer}
    inventory: list[dict[str, object]] = []
    for index, node in enumerate(graph.node):
        if node.op_type != "Conv":
            continue
        attrs: dict[str, object] = {}
        for attr in node.attribute:
            if attr.name == "group":
                attrs["group"] = int(attr.i)
            elif attr.name in {"kernel_shape", "pads", "strides", "dilations"}:
                attrs[attr.name] = [int(value) for value in attr.ints]
        weight_name = node.input[1] if len(node.input) > 1 else ""
        inventory.append(
            {
                "index": index,
                "name": node.name or f"Conv_{index}",
                "group": int(attrs.get("group", 1)),
                "kernel_shape": attrs.get("kernel_shape"),
                "pads": attrs.get("pads"),
                "strides": attrs.get("strides"),
                "dilations": attrs.get("dilations"),
                "weight_shape": initializers.get(weight_name),
            }
        )
    return inventory


def _conv_kernel_group_signature(
    convs: list[dict[str, object]],
    *,
    kernel_key: str,
    group_key: str,
) -> list[dict[str, object]]:
    return [
        {
            "kernel": [int(value) for value in item[kernel_key]],
            "group": int(item[group_key]),
        }
        for item in convs
    ]


def _assert_exact_conv_signature(
    torch_convs: list[dict[str, object]],
    onnx_convs: list[dict[str, object]],
) -> dict[str, object]:
    expected = [
        {"kernel": list(kernel), "group": group}
        for kernel, group in EXPECTED_CONV_KERNEL_GROUP_SIGNATURE
    ]
    torch_signature = _conv_kernel_group_signature(
        torch_convs,
        kernel_key="kernel_size",
        group_key="groups",
    )
    onnx_signature = _conv_kernel_group_signature(
        onnx_convs,
        kernel_key="kernel_shape",
        group_key="group",
    )
    if torch_signature != expected:
        raise RuntimeError(
            f"Unexpected exported Torch Conv kernel/group signature: "
            f"{torch_signature} != {expected}"
        )
    if onnx_signature != expected:
        raise RuntimeError(
            f"Unexpected ONNX Conv kernel/group signature: "
            f"{onnx_signature} != {expected}"
        )
    return {
        "expected": expected,
        "torch_export": torch_signature,
        "onnx": onnx_signature,
    }


def export_onnx(
    model: nn.Module,
    path: Path,
    *,
    batch_size: int,
) -> tuple[nn.Module, dict[str, object]]:
    import onnx

    source = model.to("cpu").eval()
    export_model, export_info = _prepare_exportable_model(source)
    export_info["torch_parity"] = _compare_torch_models(source, export_model)

    example = torch.zeros((batch_size, *IMAGE_SHAPE), dtype=torch.float32)
    kwargs: dict[str, object] = {
        "export_params": True,
        "opset_version": OPSET_VERSION,
        "do_constant_folding": True,
        "input_names": ["images"],
        "output_names": ["logits"],
        "dynamic_axes": {"images": {0: "batch"}, "logits": {0: "batch"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(export_model, example, str(path), **kwargs)

    model_proto = onnx.load(str(path))
    onnx.checker.check_model(model_proto)
    onnx_convs = _onnx_conv_inventory(path)
    export_info["onnx_convs"] = onnx_convs
    export_info["conv_kernel_group_signature"] = _assert_exact_conv_signature(
        export_info["torch_exported_convs"],
        onnx_convs,
    )
    return source, export_info


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[rank - 1])


def _summary_metrics(samples_ms: list[float]) -> dict[str, float]:
    return {
        "latency_p50_ms": float(statistics.median(samples_ms)),
        "latency_p95_ms": _nearest_rank_percentile(samples_ms, 0.95),
        "latency_mean_ms": float(statistics.fmean(samples_ms)),
    }


def _onnx_parity(source: nn.Module, session, input_name: str) -> dict[str, object]:
    inputs = _parity_input()
    with torch.no_grad():
        expected = source(inputs)
    actual = torch.from_numpy(session.run(None, {input_name: inputs.numpy()})[0])
    delta = (expected - actual).abs()
    allclose = bool(
        torch.allclose(expected, actual, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    )
    prediction_mismatches = int(
        (expected.argmax(dim=1) != actual.argmax(dim=1)).sum().item()
    )
    if not allclose or prediction_mismatches:
        raise RuntimeError("ONNX Runtime output does not match source prototype")
    return {
        "samples": PARITY_SAMPLES,
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "allclose": allclose,
        "prediction_mismatches": prediction_mismatches,
        "max_abs_error": float(delta.max().item()),
    }


def _prototype_report() -> dict[str, object]:
    return {
        "training": "none",
        "weights": "deterministic random initialization for runtime feasibility only",
        "seed": PROTOTYPE_SEED,
        "group": "C8 regular representations",
        "fields": list(FIELD_COUNTS),
        "tensor_channels": [GROUP_SIZE * value for value in FIELD_COUNTS],
        "replacement": {
            "blocks": ["stage3", "stage4"],
            "original": [
                "8 regular fields -> 16 regular fields, equivariant 3x3",
                "16 regular fields -> 32 regular fields, equivariant 3x3",
            ],
            "separable": [
                {
                    "depthwise": {
                        "input_fields": 8,
                        "output_fields": 8,
                        "kernel": [3, 3],
                        "groups": 8,
                        "tensor_channels_per_group": 8,
                    },
                    "pointwise": {
                        "input_fields": 8,
                        "output_fields": 16,
                        "kernel": [1, 1],
                        "groups": 1,
                    },
                },
                {
                    "depthwise": {
                        "input_fields": 16,
                        "output_fields": 16,
                        "kernel": [3, 3],
                        "groups": 16,
                        "tensor_channels_per_group": 8,
                    },
                    "pointwise": {
                        "input_fields": 16,
                        "output_fields": 32,
                        "kernel": [1, 1],
                        "groups": 1,
                    },
                },
            ],
            "normalization_activation": "InnerBatchNorm + ReLU after each depthwise and pointwise convolution",
        },
    }


def evaluate_cpu(context):
    parameters = context.parameters
    batch_size = int(parameters["batch_size"])
    warmup_runs = int(parameters["warmup_runs"])
    measure_runs = int(parameters["measure_runs"])
    intra_op_threads = int(parameters["intra_op_threads"])
    inter_op_threads = int(parameters["inter_op_threads"])

    if batch_size < 1 or warmup_runs < 1 or measure_runs < 10:
        raise ValueError("invalid CPU smoke benchmark counts")
    if intra_op_threads < 1 or inter_op_threads < 1:
        raise ValueError("ONNX Runtime thread counts must be at least 1")

    import onnxruntime as ort

    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = work_dir / "model.onnx"
    report_path = work_dir / "c8-narrow-fieldwise-dw-pw-onnx-cpu-smoke.json"

    source, export_info = export_onnx(
        build_prototype(),
        onnx_path,
        batch_size=batch_size,
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Unexpected active ORT providers: {session.get_providers()}")
    input_name = session.get_inputs()[0].name
    onnx_parity = _onnx_parity(source, session, input_name)

    input_tensor = torch.zeros((batch_size, *IMAGE_SHAPE), dtype=torch.float32).numpy()
    for _ in range(warmup_runs):
        session.run(None, {input_name: input_tensor})

    samples_ms: list[float] = []
    for _ in range(measure_runs):
        started = time.perf_counter_ns()
        session.run(None, {input_name: input_tensor})
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

    metrics = _summary_metrics(samples_ms)
    onnx_bytes = onnx_path.read_bytes()
    report = {
        "schema": "mjtensu.smoke/c8-narrow-fieldwise-dw-pw-onnx-cpu/v1",
        "purpose": "untrained infrastructure/export/runtime feasibility smoke; not an accuracy result",
        "model_anchor": "Study model is schema/lineage anchor only; context.model.module is intentionally ignored",
        "prototype": _prototype_report(),
        "export": {
            **export_info,
            "opset_version": OPSET_VERSION,
            "dynamic_batch": True,
            "onnx_bytes": len(onnx_bytes),
            "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
            "onnx_parity": onnx_parity,
        },
        "measurement_scope": (
            "onnxruntime session.run only; excludes construction, export, parity, "
            "session creation, and preprocessing"
        ),
        "benchmark": {
            "batch_size": batch_size,
            "warmup_runs": warmup_runs,
            "measure_runs": measure_runs,
            "metrics_ms": metrics,
            "minimum_ms": float(min(samples_ms)),
            "maximum_ms": float(max(samples_ms)),
            "samples_ms": samples_ms,
        },
        "environment": {
            "onnxruntime_version": ort.__version__,
            "active_providers": list(session.get_providers()),
            "intra_op_threads": intra_op_threads,
            "inter_op_threads": inter_op_threads,
            "execution_mode": "ORT_SEQUENTIAL",
            "graph_optimization_level": "ORT_ENABLE_ALL",
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return EvaluationCandidate(
        metrics=metrics,
        artifacts={"onnx_model": onnx_path, "smoke_report": report_path},
    )


def _tensor_base64(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return base64.b64encode(array.tobytes()).decode("ascii")


BENCHMARK_TEMPLATE = r"""<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<title>MLDB C8 grouped R2Conv iPhone smoke</title>
<pre id="status">starting</pre>
<script type="module">
const ORT_WEB_MODULE_URL = __ORT_MODULE__;
const MODEL_B64 = __MODEL_B64__;
const PARITY_INPUT_B64 = __PARITY_B64__;
const BATCH_SIZE = __BATCH_SIZE__;
const WARMUP_RUNS = __WARMUP_RUNS__;
const RUNS_PER_BLOCK = __RUNS_PER_BLOCK__;
const MEASURE_BLOCKS = __MEASURE_BLOCKS__;
const IMAGE_SIZE = 64;
const CLASS_COUNT = 35;
const PARITY_SAMPLES = 4;
const ORT_WEB_VERSION = "1.27.0";
const ORT_WEB_DIST_URL = __ORT_DIST__;
const statusNode = document.getElementById("status");

function bytesFromBase64(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function float32FromBase64(value) {
  const bytes = bytesFromBase64(value);
  return new Float32Array(bytes.buffer);
}

async function submitResult(payload) {
  const query = new URLSearchParams(location.search);
  const jobId = query.get("runner_job_id");
  const token = query.get("runner_token");
  if (!jobId || !token) throw new Error("browser runner job identity is missing");
  const response = await fetch(
    "/v1/jobs/" + encodeURIComponent(jobId) + "/result?token=" + encodeURIComponent(token),
    {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)}
  );
  if (!response.ok) throw new Error("result callback failed: " + response.status);
}

async function main() {
  let session;
  try {
    const ort = await import(ORT_WEB_MODULE_URL);
    ort.env.logLevel = "warning";
    ort.env.wasm.proxy = false;
    ort.env.wasm.simd = true;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = ORT_WEB_DIST_URL;

    const modelBytes = bytesFromBase64(MODEL_B64);
    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      executionMode: "sequential",
    });
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const parityTensor = new ort.Tensor(
      "float32",
      float32FromBase64(PARITY_INPUT_B64),
      [PARITY_SAMPLES, 1, IMAGE_SIZE, IMAGE_SIZE],
    );
    const parityOutput = await session.run({[inputName]: parityTensor});
    const parityLogits = parityOutput[outputName];

    const benchmarkTensor = new ort.Tensor(
      "float32",
      new Float32Array(BATCH_SIZE * IMAGE_SIZE * IMAGE_SIZE),
      [BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE],
    );
    const feeds = {[inputName]: benchmarkTensor};

    for (let index = 0; index < WARMUP_RUNS; index += 1) await session.run(feeds);

    const samplesMs = [];
    const blockTotalsMs = [];
    for (let block = 0; block < MEASURE_BLOCKS; block += 1) {
      const started = performance.now();
      let output;
      for (let index = 0; index < RUNS_PER_BLOCK; index += 1) output = await session.run(feeds);
      const elapsed = performance.now() - started;
      const logits = output[outputName];
      if (!logits || logits.dims[0] !== BATCH_SIZE || logits.dims[1] !== CLASS_COUNT) {
        throw new Error("measurement output shape mismatch");
      }
      blockTotalsMs.push(elapsed);
      samplesMs.push(elapsed / RUNS_PER_BLOCK);
    }

    const versions = ort.env && ort.env.versions ? ort.env.versions : {};
    await submitResult({
      ok: true,
      parity: {
        output_name: outputName,
        dims: Array.from(parityLogits.dims),
        logits: Array.from(parityLogits.data, Number),
      },
      benchmark: {
        batch_size: BATCH_SIZE,
        warmup_runs: WARMUP_RUNS,
        runs_per_block: RUNS_PER_BLOCK,
        measure_blocks: MEASURE_BLOCKS,
        total_measure_runs: RUNS_PER_BLOCK * MEASURE_BLOCKS,
        block_totals_ms: blockTotalsMs,
        samples_ms: samplesMs,
      },
      environment: {
        provider: "wasm-simd",
        num_threads: 1,
        wasm_proxy: false,
        graph_optimization_level: "all",
        execution_mode: "sequential",
        onnxruntime_web_version_expected: ORT_WEB_VERSION,
        onnxruntime_web_version_reported: versions.web || null,
        hardware_concurrency: navigator.hardwareConcurrency || 1,
        cross_origin_isolated: globalThis.crossOriginIsolated === true,
        secure_context: globalThis.isSecureContext === true,
        user_agent: navigator.userAgent,
      },
    });
  } catch (error) {
    try {
      await submitResult({ok: false, error: error instanceof Error ? error.message : String(error)});
    } catch (_) {}
  } finally {
    if (session) await session.release();
  }
}
void main();
</script>
"""


def _benchmark_document(
    *,
    model_bytes: bytes,
    parity_input: torch.Tensor,
    batch_size: int,
    warmup_runs: int,
    runs_per_block: int,
    measure_blocks: int,
) -> str:
    replacements = {
        "__ORT_MODULE__": json.dumps(ORT_WEB_MODULE_URL),
        "__MODEL_B64__": json.dumps(base64.b64encode(model_bytes).decode("ascii")),
        "__PARITY_B64__": json.dumps(_tensor_base64(parity_input)),
        "__BATCH_SIZE__": str(batch_size),
        "__WARMUP_RUNS__": str(warmup_runs),
        "__RUNS_PER_BLOCK__": str(runs_per_block),
        "__MEASURE_BLOCKS__": str(measure_blocks),
        "__ORT_DIST__": json.dumps(ORT_WEB_DIST_URL),
    }
    document = BENCHMARK_TEMPLATE
    for key, value in replacements.items():
        document = document.replace(key, value)
    return document


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, object] | None = None,
    timeout: float = RUNNER_HTTP_TIMEOUT_SEC,
) -> dict[str, Any]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"iPhone browser runner HTTP {error.code} for {url}: {detail}"
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"iPhone browser runner request failed for {url}: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError("iPhone browser runner returned non-object JSON")
    return data


def _run_browser_job(document: str) -> tuple[dict[str, Any], dict[str, Any]]:
    runner_url = os.environ.get("MLDB_IPHONE_BROWSER_RUNNER_URL", DEFAULT_RUNNER_URL)
    runner_url = runner_url.strip().rstrip("/")
    health = _request_json("GET", runner_url + "/healthz")
    if health.get("ok") is not True:
        raise RuntimeError(f"iPhone browser runner is not ready: {health}")

    created = _request_json(
        "POST",
        runner_url + "/v1/jobs",
        payload={"document": document, "result_timeout_sec": RUNNER_RESULT_TIMEOUT_SEC},
    )
    job_id = created.get("id")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("iPhone browser runner did not return a job id")

    deadline = time.monotonic() + RUNNER_RESULT_TIMEOUT_SEC + 30
    while time.monotonic() < deadline:
        state = _request_json("GET", runner_url + "/v1/jobs/" + job_id)
        status = state.get("state")
        if status == "completed":
            result = state.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("completed browser job has no result object")
            return result, health
        if status == "failed":
            raise RuntimeError("iPhone browser benchmark failed: " + str(state.get("error")))
        if status not in {"queued", "launching", "launched"}:
            raise RuntimeError(f"unexpected iPhone browser job state: {status!r}")
        time.sleep(RUNNER_POLL_INTERVAL_SEC)
    raise RuntimeError("iPhone browser benchmark timed out while polling runner")


def _browser_parity(source: nn.Module, browser_result: dict[str, Any]) -> dict[str, object]:
    parity = browser_result.get("parity")
    if not isinstance(parity, dict):
        raise RuntimeError("browser result is missing parity data")
    if parity.get("dims") != [PARITY_SAMPLES, CLASS_COUNT]:
        raise RuntimeError(f"unexpected browser parity dimensions: {parity.get('dims')!r}")
    logits = parity.get("logits")
    if not isinstance(logits, list) or len(logits) != PARITY_SAMPLES * CLASS_COUNT:
        raise RuntimeError("browser parity logits have unexpected length")
    actual = torch.tensor([float(value) for value in logits], dtype=torch.float32).reshape(
        PARITY_SAMPLES, CLASS_COUNT
    )
    inputs = _parity_input()
    with torch.no_grad():
        expected = source(inputs)
    delta = (expected - actual).abs()
    allclose = bool(
        torch.allclose(expected, actual, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    )
    prediction_mismatches = int(
        (expected.argmax(dim=1) != actual.argmax(dim=1)).sum().item()
    )
    if not allclose or prediction_mismatches:
        raise RuntimeError("ORT Web output does not match source prototype")
    return {
        "samples": PARITY_SAMPLES,
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "allclose": allclose,
        "prediction_mismatches": prediction_mismatches,
        "max_abs_error": float(delta.max().item()),
    }


def evaluate_iphone(context):
    parameters = context.parameters
    batch_size = int(parameters["batch_size"])
    warmup_runs = int(parameters["warmup_runs"])
    runs_per_block = int(parameters["runs_per_block"])
    measure_blocks = int(parameters["measure_blocks"])
    if batch_size < 1 or warmup_runs < 1 or runs_per_block < 10 or measure_blocks < 10:
        raise ValueError("invalid iPhone smoke benchmark counts")

    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = work_dir / "model.onnx"
    report_path = work_dir / "c8-narrow-fieldwise-dw-pw-iphone-smoke.json"

    source, export_info = export_onnx(
        build_prototype(),
        onnx_path,
        batch_size=batch_size,
    )
    onnx_bytes = onnx_path.read_bytes()
    document = _benchmark_document(
        model_bytes=onnx_bytes,
        parity_input=_parity_input(),
        batch_size=batch_size,
        warmup_runs=warmup_runs,
        runs_per_block=runs_per_block,
        measure_blocks=measure_blocks,
    )
    browser_result, runner_health = _run_browser_job(document)
    if browser_result.get("ok") is not True:
        raise RuntimeError("ORT Web benchmark page reported failure: " + str(browser_result.get("error")))

    benchmark = browser_result.get("benchmark")
    if not isinstance(benchmark, dict):
        raise RuntimeError("browser result is missing benchmark data")
    expected_benchmark = {
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "runs_per_block": runs_per_block,
        "measure_blocks": measure_blocks,
        "total_measure_runs": runs_per_block * measure_blocks,
    }
    for key, expected in expected_benchmark.items():
        if benchmark.get(key) != expected:
            raise RuntimeError(
                f"browser benchmark {key} mismatch: {benchmark.get(key)!r} != {expected!r}"
            )
    samples_raw = benchmark.get("samples_ms")
    totals_raw = benchmark.get("block_totals_ms")
    if not isinstance(samples_raw, list) or len(samples_raw) != measure_blocks:
        raise RuntimeError("browser benchmark returned unexpected sample count")
    if not isinstance(totals_raw, list) or len(totals_raw) != measure_blocks:
        raise RuntimeError("browser benchmark returned unexpected block-total count")
    samples_ms = [float(value) for value in samples_raw]
    block_totals_ms = [float(value) for value in totals_raw]
    if not all(math.isfinite(value) and value >= 0.0 for value in samples_ms):
        raise RuntimeError("browser benchmark samples must be finite non-negative values")

    browser_parity = _browser_parity(source, browser_result)
    environment = browser_result.get("environment")
    if not isinstance(environment, dict):
        raise RuntimeError("browser result is missing environment data")
    if environment.get("provider") != "wasm-simd":
        raise RuntimeError(f"unexpected browser provider: {environment.get('provider')!r}")
    if environment.get("num_threads") != 1 or environment.get("wasm_proxy") is not False:
        raise RuntimeError("iPhone smoke requires ORT Web wasm-simd, numThreads=1, proxy=false")

    metrics = _summary_metrics(samples_ms)
    device_id = runner_health.get("udid")
    device_id_sha256 = (
        hashlib.sha256(device_id.encode("utf-8")).hexdigest()
        if isinstance(device_id, str) and device_id
        else None
    )
    report = {
        "schema": "mjtensu.smoke/c8-narrow-fieldwise-dw-pw-ort-web-iphone/v1",
        "purpose": "untrained infrastructure/export/device feasibility smoke; not an accuracy result",
        "model_anchor": "Study model is schema/lineage anchor only; context.model.module is intentionally ignored",
        "prototype": _prototype_report(),
        "export": {
            **export_info,
            "opset_version": OPSET_VERSION,
            "dynamic_batch": True,
            "onnx_bytes": len(onnx_bytes),
            "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
            "browser_parity": browser_parity,
        },
        "measurement_scope": (
            "browser performance.now elapsed time across blocks of sequential awaited ORT Web "
            "session.run calls divided by runs_per_block; excludes loading, session creation, "
            "input creation, parity validation, warmup, and result transport"
        ),
        "benchmark": {
            "batch_size": batch_size,
            "warmup_runs": warmup_runs,
            "runs_per_block": runs_per_block,
            "measure_blocks": measure_blocks,
            "total_measure_runs": runs_per_block * measure_blocks,
            "metrics_ms": metrics,
            "minimum_ms": float(min(samples_ms)),
            "maximum_ms": float(max(samples_ms)),
            "samples_ms": samples_ms,
            "block_totals_ms": block_totals_ms,
        },
        "runtime": {
            "onnxruntime_web_version": ORT_WEB_VERSION,
            "module_url": ORT_WEB_MODULE_URL,
            "wasm_path_prefix": ORT_WEB_DIST_URL,
        },
        "environment": {
            **environment,
            "runner_device_id_sha256": device_id_sha256,
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return EvaluationCandidate(
        metrics=metrics,
        artifacts={"onnx_model": onnx_path, "smoke_report": report_path},
    )
