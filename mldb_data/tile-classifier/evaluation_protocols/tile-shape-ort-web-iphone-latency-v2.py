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


_IMAGE_SHAPE = (1, 64, 64)
_CLASS_COUNT = 35
_OPSET_VERSION = 16
_C8_GROUP_SIZE = 8
_PARITY_SAMPLES = 4
_PARITY_ATOL = 1.0e-4
_PARITY_RTOL = 1.0e-4
_ORT_WEB_VERSION = "1.27.0"
_ORT_WEB_DIST_URL = (
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/"
)
_ORT_WEB_MODULE_URL = _ORT_WEB_DIST_URL + "ort.wasm.min.mjs"
_DEFAULT_RUNNER_URL = "http://192.168.11.22:8877"
_RUNNER_RESULT_TIMEOUT_SEC = 300
_RUNNER_HTTP_TIMEOUT_SEC = 20
_RUNNER_POLL_INTERVAL_SEC = 0.25


class _C8GroupMaxPool(nn.Module):
    def __init__(self, group_size: int = _C8_GROUP_SIZE) -> None:
        super().__init__()
        self.group_size = int(group_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim != 4:
            raise ValueError(f"Expected NCHW tensor, got {tuple(input_tensor.shape)}")
        batch, channels, height, width = input_tensor.shape
        fields = channels // self.group_size
        grouped = input_tensor.reshape(
            batch, fields, self.group_size, height, width
        )
        return grouped.amax(dim=2)


class _ExportedC8Classifier(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        spatial_pool: nn.Module,
        classifier: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.group_pool = _C8GroupMaxPool()
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

    c8_parts = (backbone, group_pool, spatial_pool, classifier, field_counts)
    if all(part is not None for part in c8_parts):
        export = getattr(backbone, "export", None)
        out_type = getattr(group_pool, "out_type", None)
        out_size = getattr(out_type, "size", None)
        if not callable(export) or not isinstance(field_counts, tuple):
            raise TypeError("C8 classifier export contract is malformed")
        if int(out_size) != int(field_counts[-1]):
            raise ValueError("C8 GroupPooling output does not match final field count")
        return (
            _ExportedC8Classifier(
                backbone=export(),
                spatial_pool=spatial_pool,
                classifier=classifier,
            ).eval(),
            "c8-equivariant-tensor-export",
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

    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("onnx is required for iPhone ORT Web evaluation") from error
    onnx.checker.check_model(onnx.load(str(path)))
    return source, {"mode": mode, "torch_parity": torch_parity}


def _tensor_base64(tensor: torch.Tensor) -> str:
    values = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return base64.b64encode(values.tobytes()).decode("ascii")


_BENCHMARK_TEMPLATE = r"""<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<title>MLDB iPhone ORT Web latency</title>
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
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function float32FromBase64(value) {
  const bytes = bytesFromBase64(value);
  if (bytes.byteLength % 4 !== 0) {
    throw new Error("float32 payload byte length is not divisible by 4");
  }
  return new Float32Array(bytes.buffer);
}

async function submitResult(payload) {
  const query = new URLSearchParams(location.search);
  const jobId = query.get("runner_job_id");
  const token = query.get("runner_token");
  if (!jobId || !token) {
    throw new Error("browser runner job identity is missing");
  }
  const url = "/v1/jobs/" + encodeURIComponent(jobId) +
    "/result?token=" + encodeURIComponent(token);
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error("result callback failed: " + response.status);
  }
}

async function main() {
  let session;
  try {
    statusNode.textContent = "loading ORT Web";
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
    if (session.inputNames.length !== 1 || session.outputNames.length !== 1) {
      throw new Error(
        "expected one input/output, got " +
        session.inputNames.length + "/" + session.outputNames.length
      );
    }
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    statusNode.textContent = "checking browser parity";
    const parityTensor = new ort.Tensor(
      "float32",
      float32FromBase64(PARITY_INPUT_B64),
      [PARITY_SAMPLES, 1, IMAGE_SIZE, IMAGE_SIZE],
    );
    const parityOutput = await session.run({[inputName]: parityTensor});
    const parityLogits = parityOutput[outputName];
    if (!parityLogits) {
      throw new Error("browser parity output is missing");
    }
    if (
      parityLogits.dims[0] !== PARITY_SAMPLES ||
      parityLogits.dims[1] !== CLASS_COUNT
    ) {
      throw new Error("unexpected parity output shape " + parityLogits.dims.join("x"));
    }

    const benchmarkTensor = new ort.Tensor(
      "float32",
      new Float32Array(BATCH_SIZE * IMAGE_SIZE * IMAGE_SIZE),
      [BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE],
    );
    const feeds = {[inputName]: benchmarkTensor};

    statusNode.textContent = "warming up";
    for (let index = 0; index < WARMUP_RUNS; index += 1) {
      const output = await session.run(feeds);
      const logits = output[outputName];
      if (!logits || logits.dims[0] !== BATCH_SIZE || logits.dims[1] !== CLASS_COUNT) {
        throw new Error("warmup output shape mismatch");
      }
    }

    statusNode.textContent = "measuring";
    const samplesMs = [];
    const blockTotalsMs = [];
    for (let block = 0; block < MEASURE_BLOCKS; block += 1) {
      const started = performance.now();
      let output;
      for (let index = 0; index < RUNS_PER_BLOCK; index += 1) {
        output = await session.run(feeds);
      }
      const elapsed = performance.now() - started;
      const logits = output[outputName];
      if (!logits || logits.dims[0] !== BATCH_SIZE || logits.dims[1] !== CLASS_COUNT) {
        throw new Error("measurement output shape mismatch");
      }
      blockTotalsMs.push(elapsed);
      samplesMs.push(elapsed / RUNS_PER_BLOCK);
    }

    const versions = ort.env && ort.env.versions ? ort.env.versions : {};
    statusNode.textContent = "completed";
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
    statusNode.textContent = "failed: " +
      (error instanceof Error ? error.message : String(error));
    try {
      await submitResult({
        ok: false,
        error: error instanceof Error ? error.message : String(error),
        environment: {
          user_agent: navigator.userAgent,
          hardware_concurrency: navigator.hardwareConcurrency || 1,
        },
      });
    } catch (_) {
    }
  } finally {
    if (session) {
      await session.release();
    }
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
        "__ORT_MODULE__": json.dumps(_ORT_WEB_MODULE_URL),
        "__MODEL_B64__": json.dumps(base64.b64encode(model_bytes).decode("ascii")),
        "__PARITY_B64__": json.dumps(_tensor_base64(parity_input)),
        "__BATCH_SIZE__": str(batch_size),
        "__WARMUP_RUNS__": str(warmup_runs),
        "__RUNS_PER_BLOCK__": str(runs_per_block),
        "__MEASURE_BLOCKS__": str(measure_blocks),
        "__ORT_DIST__": json.dumps(_ORT_WEB_DIST_URL),
    }
    document = _BENCHMARK_TEMPLATE
    for key, value in replacements.items():
        document = document.replace(key, value)
    return document


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, object] | None = None,
    timeout: float = _RUNNER_HTTP_TIMEOUT_SEC,
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
        raise RuntimeError(
            f"iPhone browser runner request failed for {url}: {error}"
        ) from error
    if not isinstance(data, dict):
        raise RuntimeError("iPhone browser runner returned non-object JSON")
    return data


def _browser_runner_url() -> str:
    value = os.environ.get("MLDB_IPHONE_BROWSER_RUNNER_URL", _DEFAULT_RUNNER_URL)
    value = value.strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        raise ValueError("MLDB_IPHONE_BROWSER_RUNNER_URL must be an http(s) URL")
    return value


def _run_browser_job(document: str) -> tuple[dict[str, Any], dict[str, Any]]:
    runner_url = _browser_runner_url()
    health = _request_json("GET", runner_url + "/healthz")
    if health.get("ok") is not True:
        raise RuntimeError(f"iPhone browser runner is not ready: {health}")

    created = _request_json(
        "POST",
        runner_url + "/v1/jobs",
        payload={
            "document": document,
            "result_timeout_sec": _RUNNER_RESULT_TIMEOUT_SEC,
        },
    )
    job_id = created.get("id")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("iPhone browser runner did not return a job id")

    deadline = time.monotonic() + _RUNNER_RESULT_TIMEOUT_SEC + 30
    while time.monotonic() < deadline:
        state = _request_json("GET", runner_url + "/v1/jobs/" + job_id)
        status = state.get("state")
        if status == "completed":
            result = state.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("completed browser job has no result object")
            return result, health
        if status == "failed":
            raise RuntimeError(
                "iPhone browser benchmark failed: " +
                str(state.get("error", "unknown error"))
            )
        if status not in {"queued", "launching", "launched"}:
            raise RuntimeError(
                f"unexpected iPhone browser job state: {status!r}"
            )
        time.sleep(_RUNNER_POLL_INTERVAL_SEC)
    raise RuntimeError("iPhone browser benchmark timed out while polling runner")


def _validate_browser_parity(
    source_model: nn.Module,
    browser_result: dict[str, Any],
) -> dict[str, object]:
    parity = browser_result.get("parity")
    if not isinstance(parity, dict):
        raise RuntimeError("browser result is missing parity data")
    dims = parity.get("dims")
    logits = parity.get("logits")
    if dims != [_PARITY_SAMPLES, _CLASS_COUNT]:
        raise RuntimeError(f"unexpected browser parity dimensions: {dims!r}")
    if (
        not isinstance(logits, list)
        or len(logits) != _PARITY_SAMPLES * _CLASS_COUNT
    ):
        raise RuntimeError("browser parity logits have unexpected length")
    try:
        actual = torch.tensor(
            [float(value) for value in logits],
            dtype=torch.float32,
        ).reshape(_PARITY_SAMPLES, _CLASS_COUNT)

    except (TypeError, ValueError) as error:
        raise RuntimeError("browser parity logits are not numeric") from error

    inputs = _parity_input()
    with torch.no_grad():
        expected = source_model(inputs)
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
        raise RuntimeError("ORT Web output does not match source model")
    return {
        "samples": _PARITY_SAMPLES,
        "atol": _PARITY_ATOL,
        "rtol": _PARITY_RTOL,
        "allclose": allclose,
        "prediction_mismatches": prediction_mismatches,
        "max_abs_error": float(delta.max().item()),
    }


def _browser_samples(
    browser_result: dict[str, Any],
    *,
    batch_size: int,
    warmup_runs: int,
    runs_per_block: int,
    measure_blocks: int,
) -> tuple[list[float], list[float]]:
    if browser_result.get("ok") is not True:
        raise RuntimeError(
            "ORT Web benchmark page reported failure: " +
            str(browser_result.get("error"))
        )
    benchmark = browser_result.get("benchmark")
    if not isinstance(benchmark, dict):
        raise RuntimeError("browser result is missing benchmark data")
    expected = {
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "runs_per_block": runs_per_block,
        "measure_blocks": measure_blocks,
        "total_measure_runs": runs_per_block * measure_blocks,
    }
    for key, value in expected.items():
        if benchmark.get(key) != value:
            raise RuntimeError(
                f"browser benchmark {key} mismatch: "
                f"{benchmark.get(key)!r} != {value!r}"
            )
    raw_samples = benchmark.get("samples_ms")
    if not isinstance(raw_samples, list) or len(raw_samples) != measure_blocks:
        raise RuntimeError("browser benchmark returned unexpected sample count")
    samples: list[float] = []
    for value in raw_samples:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("browser benchmark sample is not numeric")
        sample = float(value)
        if not math.isfinite(sample) or sample < 0.0:
            raise RuntimeError(
                "browser benchmark sample is not a finite non-negative value"
            )
        samples.append(sample)
    if not any(sample > 0.0 for sample in samples):
        raise RuntimeError(
            "browser benchmark timer produced only zero-duration samples"
        )
    raw_totals = benchmark.get("block_totals_ms")
    if not isinstance(raw_totals, list) or len(raw_totals) != measure_blocks:
        raise RuntimeError("browser benchmark returned unexpected block-total count")
    totals: list[float] = []
    for value in raw_totals:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("browser benchmark block total is not numeric")
        total = float(value)
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("browser benchmark block total is not a finite positive value")
        totals.append(total)
    return samples, totals


def evaluate(context):
    parameters = context.parameters
    batch_size = int(parameters["batch_size"])
    warmup_runs = int(parameters["warmup_runs"])
    runs_per_block = int(parameters["runs_per_block"])
    measure_blocks = int(parameters["measure_blocks"])
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if warmup_runs < 1:
        raise ValueError("warmup_runs must be at least 1")
    if runs_per_block < 10:
        raise ValueError("runs_per_block must be at least 10")
    if measure_blocks < 10:
        raise ValueError("measure_blocks must be at least 10")

    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = work_dir / "model.onnx"
    report_path = work_dir / "ort-web-iphone-latency.json"

    source_model, export_info = _export_onnx(
        context.model.module,
        onnx_path,
        batch_size=batch_size,
    )
    parity_input = _parity_input()
    onnx_bytes = onnx_path.read_bytes()
    document = _benchmark_document(
        model_bytes=onnx_bytes,
        parity_input=parity_input,
        batch_size=batch_size,
        warmup_runs=warmup_runs,
        runs_per_block=runs_per_block,
        measure_blocks=measure_blocks,
    )
    browser_result, runner_health = _run_browser_job(document)
    samples_ms, block_totals_ms = _browser_samples(
        browser_result,
        batch_size=batch_size,
        warmup_runs=warmup_runs,
        runs_per_block=runs_per_block,
        measure_blocks=measure_blocks,
    )
    browser_parity = _validate_browser_parity(
        source_model,
        browser_result,
    )

    p50_ms = float(statistics.median(samples_ms))
    p95_ms = _nearest_rank_percentile(samples_ms, 0.95)
    mean_ms = float(statistics.fmean(samples_ms))
    metrics = {
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "latency_mean_ms": mean_ms,
    }

    environment = browser_result.get("environment")
    if not isinstance(environment, dict):
        raise RuntimeError("browser result is missing environment data")
    if environment.get("provider") != "wasm-simd":
        raise RuntimeError(
            f"unexpected browser provider: {environment.get('provider')!r}"
        )
    if environment.get("num_threads") != 1:
        raise RuntimeError(
            "iPhone latency protocol requires ORT Web numThreads=1"
        )
    if environment.get("wasm_proxy") is not False:
        raise RuntimeError(
            "iPhone latency protocol requires ORT Web wasm proxy disabled"
        )

    device_id = runner_health.get("udid")
    device_id_sha256 = (
        hashlib.sha256(device_id.encode("utf-8")).hexdigest()
        if isinstance(device_id, str) and device_id
        else None
    )
    report = {
        "schema": "mjtensu.recognition/tile-ort-web-iphone-latency/v2",
        "measurement_scope": (
            "browser performance.now elapsed time across blocks of sequential awaited ORT Web "
            "session.run calls, divided by runs_per_block to obtain per-run block averages; "
            "excludes page/runtime/model loading, session creation, input tensor creation, "
            "parity validation, warmup, and result transport"
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
            "browser_parity": browser_parity,
        },
        "runtime": {
            "onnxruntime_web_version": _ORT_WEB_VERSION,
            "module_url": _ORT_WEB_MODULE_URL,
            "wasm_path_prefix": _ORT_WEB_DIST_URL,
        },
        "benchmark": {
            "batch_size": batch_size,
            "warmup_runs": warmup_runs,
            "runs_per_block": runs_per_block,
            "measure_blocks": measure_blocks,
            "total_measure_runs": runs_per_block * measure_blocks,
            "p50_definition": (
                "median of per-run block-average awaited session.run durations"
            ),
            "p95_definition": (
                "nearest-rank 95th percentile of per-run block-average awaited "
                "session.run durations"
            ),
            "metrics_ms": metrics,
            "minimum_ms": float(min(samples_ms)),
            "maximum_ms": float(max(samples_ms)),
            "samples_ms": samples_ms,
            "block_totals_ms": block_totals_ms,
        },
        "environment": {
            **environment,
            "runner_device_id_sha256": device_id_sha256,
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    return EvaluationCandidate(
        metrics=metrics,
        artifacts={
            "onnx_model": onnx_path,
            "latency_report": report_path,
        },
    )
