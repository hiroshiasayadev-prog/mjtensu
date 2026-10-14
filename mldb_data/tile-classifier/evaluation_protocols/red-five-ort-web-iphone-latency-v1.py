from __future__ import annotations
import base64
import hashlib
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

_PARITY_SAMPLES = 4
_CLASS_COUNT = 2
_ORT_WEB_VERSION = "1.27.0"
_ORT_WEB_DIST_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/"
_ORT_WEB_MODULE_URL = _ORT_WEB_DIST_URL + "ort.wasm.min.mjs"
_DEFAULT_RUNNER_URL = "http://192.168.11.22:8877"
_RUNNER_RESULT_TIMEOUT_SEC = 300
_RUNNER_HTTP_TIMEOUT_SEC = 20
_RUNNER_POLL_INTERVAL_SEC = 0.25


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[rank - 1])

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
const IMAGE_CHANNELS = 3;
const CLASS_COUNT = 2;
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
      [PARITY_SAMPLES, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
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
      new Float32Array(BATCH_SIZE * IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE),
      [BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
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


def _benchmark_document(*, model_bytes: bytes, batch_size: int,
                        warmup_runs: int, runs_per_block: int, measure_blocks: int) -> str:
    # A fixed, zero-valued shape check; upstream CPU export has already checked
    # numerical parity against the PyTorch source model.
    parity_bytes = bytes(_PARITY_SAMPLES * 3 * 64 * 64 * 4)
    replacements = {
        "__ORT_MODULE__": json.dumps(_ORT_WEB_MODULE_URL),
        "__MODEL_B64__": json.dumps(base64.b64encode(model_bytes).decode("ascii")),
        "__PARITY_B64__": json.dumps(base64.b64encode(parity_bytes).decode("ascii")),
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


def evaluate(context):
    if context.model is not None or not context.inputs or "onnx_model" not in context.inputs:
        raise ValueError("iPhone Eval requires the declared ONNX stage input")
    if type(context.inputs["onnx_model"].data) is not bytes:
        raise ValueError("ONNX-only iPhone Eval requires verified ONNX bytes, not PyTorch")
    if not context.inputs["onnx_model"].data or not context.inputs["onnx_model"].source_evaluation_result:
        raise ValueError("ONNX source EvaluationResult is required")
    p = context.parameters
    batch_size = int(p["batch_size"])
    warmup = int(p["warmup_runs"])
    per_block = int(p["runs_per_block"])
    blocks = int(p["measure_blocks"])
    if not (1 <= batch_size <= 64 and warmup >= 1 and per_block >= 10 and blocks >= 10):
        raise ValueError("invalid latency benchmark parameters")
    document = _benchmark_document(
        model_bytes=context.inputs["onnx_model"].data,
        batch_size=batch_size, warmup_runs=warmup,
        runs_per_block=per_block, measure_blocks=blocks,
    )
    browser, runner = _run_browser_job(document)
    samples, totals = _browser_samples(
        browser, batch_size=batch_size,
        warmup_runs=warmup, runs_per_block=per_block, measure_blocks=blocks,
    )
    parity = browser.get("parity")
    if not isinstance(parity, dict) or parity.get("dims") != [_PARITY_SAMPLES, _CLASS_COUNT]:
        raise ValueError("iPhone ONNX output shape validation failed")
    logits = parity.get("logits")
    if (type(logits) is not list or len(logits) != _PARITY_SAMPLES * _CLASS_COUNT
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in logits)):
        raise ValueError("iPhone ONNX output is not finite or has wrong size")
    env = browser.get("environment")
    if (not isinstance(env, dict) or env.get("provider") != "wasm-simd"
            or env.get("num_threads") != 1 or env.get("wasm_proxy") is not False):
        raise ValueError("iPhone ONNX benchmark runtime does not meet the protocol")
    metrics = {
        "latency_p50_ms": float(statistics.median(samples)),
        "latency_p95_ms": _nearest_rank_percentile(samples, 0.95),
        "latency_mean_ms": float(statistics.fmean(samples)),
    }
    report = {
        "schema": "mjtensu.recognition/red-five-ort-web-iphone-onnx-input/v1",
        "source_evaluation_result": context.inputs["onnx_model"].source_evaluation_result,
        "onnx_sha256": hashlib.sha256(context.inputs["onnx_model"].data).hexdigest(),
        "onnx_bytes": len(context.inputs["onnx_model"].data),
        "benchmark": {
            "batch_size": batch_size,
            "warmup_runs": warmup,
            "runs_per_block": per_block,
            "measure_blocks": blocks,
            "samples_ms": samples,
            "block_totals_ms": totals,
        },
        "environment": {
            **env,
            "runner_device_id_sha256": hashlib.sha256(str(runner.get("udid")).encode()).hexdigest(),
        },
        "metrics_ms": metrics,
        "verification": "browser finite/shape check; source ONNX export parity accepted separately",
    }
    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "iphone-onnx-latency.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return EvaluationCandidate(metrics=metrics, artifacts={"latency_report": report_path})
