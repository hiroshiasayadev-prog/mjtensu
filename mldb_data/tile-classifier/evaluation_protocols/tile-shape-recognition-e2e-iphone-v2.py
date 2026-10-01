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
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

_IMAGE_SHAPE = (1, 64, 64)
_CLASS_COUNT = 35
_OPSET_VERSION = 16
_C8_GROUP_SIZE = 8
_ORT_WEB_VERSION = "1.27.0"
_DEFAULT_RUNNER_URL = "http://192.168.11.22:8877"
_DEFAULT_ASSET_BASE_URL = "https://nas.internal.thebugrat.dev/mldb-assets"
_RUNNER_RESULT_TIMEOUT_SEC = 300
_RUNNER_HTTP_TIMEOUT_SEC = 20
_RUNNER_POLL_INTERVAL_SEC = 0.25
_EXPECTED_TAKES = ("t1", "t2", "t3", "t4", "t5")
_BROWSER_BUNDLE_SHA256 = "e230fea7603a9d547fbab2835fea6535a0f21227f2595b0a7645be090296f2f3"
_BROWSER_BUNDLE_ZLIB_BASE64 = "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"

# Runtime preprocessing is a property of the training corpus used by the
# canonical classifier Model. Keep this mapping explicit and fail closed for
# unknown corpora rather than silently applying production constants.
_BASE_RUNTIME_BY_TRAINING_CORPUS: dict[str, dict[str, object]] = {
    "tile-classifier/gray35-jp500-seed42-v3-jp189-v1": {
        "runtime_spec": "gray64-tile-35-v1",
        "normalization": {
            "mean": [0.6815832403977466],
            "std": [0.2725553681973976],
        },
    },
}


class _C8GroupMaxPool(nn.Module):
    def __init__(self, group_size: int = _C8_GROUP_SIZE) -> None:
        super().__init__()
        self.group_size = int(group_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = input_tensor.shape
        fields = channels // self.group_size
        return input_tensor.reshape(
            batch, fields, self.group_size, height, width
        ).amax(dim=2)


class _ExportedC8Classifier(nn.Module):
    def __init__(self, *, backbone: nn.Module, spatial_pool: nn.Module, classifier: nn.Module) -> None:
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


def _prepare_exportable_model(model: nn.Module) -> tuple[nn.Module, str]:
    source = model.to("cpu").eval()
    backbone = getattr(source, "equivariant_backbone", None)
    group_pool = getattr(source, "group_pool", None)
    spatial_pool = getattr(source, "spatial_pool", None)
    classifier = getattr(source, "classifier", None)
    field_counts = getattr(source, "field_counts", None)
    if all(part is not None for part in (backbone, group_pool, spatial_pool, classifier, field_counts)):
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


def _export_onnx(model: nn.Module, path: Path) -> dict[str, object]:
    source = model.to("cpu").eval()
    export_model, mode = _prepare_exportable_model(source)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    parity_input = torch.rand((4, *_IMAGE_SHAPE), generator=generator, dtype=torch.float32)
    with torch.no_grad():
        expected = source(parity_input)
        actual = export_model(parity_input)
    if expected.shape != (4, _CLASS_COUNT) or actual.shape != expected.shape:
        raise RuntimeError("classifier export parity shape mismatch")
    if not torch.allclose(expected, actual, atol=1.0e-4, rtol=1.0e-4):
        raise RuntimeError("classifier tensor-only export does not match source model")

    example = torch.zeros((1, *_IMAGE_SHAPE), dtype=torch.float32)
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
        raise RuntimeError("onnx is required for iPhone recognition E2E evaluation") from error
    onnx.checker.check_model(onnx.load(str(path)))
    payload = path.read_bytes()
    return {
        "mode": mode,
        "opset_version": _OPSET_VERSION,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "torch_parity_max_abs_error": float((expected - actual).abs().max().item()),
    }


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
        raise RuntimeError(f"iPhone browser runner HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"iPhone browser runner request failed: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError("iPhone browser runner returned non-object JSON")
    return data


def _browser_runner_url() -> str:
    value = os.environ.get("MLDB_IPHONE_BROWSER_RUNNER_URL", _DEFAULT_RUNNER_URL).strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        raise ValueError("MLDB_IPHONE_BROWSER_RUNNER_URL must be an http(s) URL")
    return value


def _asset_base_url() -> str:
    value = os.environ.get("MLDB_BROWSER_ASSET_BASE_URL", _DEFAULT_ASSET_BASE_URL).strip().rstrip("/")
    if not value.startswith("https://"):
        raise ValueError("MLDB_BROWSER_ASSET_BASE_URL must be an https URL")
    return value


def _run_browser_job(document: str) -> tuple[dict[str, Any], dict[str, Any]]:
    runner_url = _browser_runner_url()
    health = _request_json("GET", runner_url + "/healthz")
    if health.get("ok") is not True:
        raise RuntimeError(f"iPhone browser runner is not ready: {health}")
    created = _request_json(
        "POST",
        runner_url + "/v1/jobs",
        payload={"document": document, "result_timeout_sec": _RUNNER_RESULT_TIMEOUT_SEC},
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
            raise RuntimeError(f"iPhone browser E2E job failed: {state.get('error', 'unknown error')}")
        if status not in {"queued", "launching", "launched"}:
            raise RuntimeError(f"unexpected iPhone browser job state: {status!r}")
        time.sleep(_RUNNER_POLL_INTERVAL_SEC)
    raise RuntimeError("iPhone browser E2E job timed out")


def _s3_uri_to_asset_url(uri: str, sha256: str | None = None) -> str:
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"unsupported logical object URI: {uri!r}")
    key = urllib.parse.quote(parsed.path.lstrip("/"), safe="/")
    url = f"{_asset_base_url()}/{urllib.parse.quote(parsed.netloc, safe='')}/{key}"
    if sha256:
        url += "?sha256=" + urllib.parse.quote(sha256, safe="")
    return url


def _runtime_model_config(loaded: object, *, expected_role: str) -> dict[str, object]:
    definition = getattr(loaded, "definition", None)
    artifact_bytes = getattr(loaded, "artifact", None)
    if not isinstance(definition, dict) or not isinstance(artifact_bytes, bytes):
        raise ValueError(
            f"{expected_role} must resolve to an immutable Runtime Model ONNX artifact for this classifier-primary protocol"
        )
    if definition.get("role") != expected_role or definition.get("format") != "onnx":
        raise ValueError(f"auxiliary model role mismatch for {expected_role}")
    artifact = definition.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Runtime Model artifact metadata is missing")
    sha256 = artifact.get("sha256")
    uri = artifact.get("uri")
    if not isinstance(sha256, str) or not isinstance(uri, str):
        raise ValueError("Runtime Model artifact identity is malformed")
    if hashlib.sha256(artifact_bytes).hexdigest() != sha256:
        raise RuntimeError("verified Runtime Model bytes do not match catalog sha256")
    runtime_spec = definition.get("runtime_spec")
    if not isinstance(runtime_spec, str):
        raise ValueError("Runtime Model runtime_spec is missing")
    result: dict[str, object] = {
        "id": definition.get("id"),
        "url": _s3_uri_to_asset_url(uri, sha256),
        "sha256": sha256,
        "runtimeSpec": runtime_spec,
    }
    if expected_role == "red-five-classifier":
        provenance = definition.get("provenance")
        normalization = provenance.get("normalization") if isinstance(provenance, dict) else None
        if not isinstance(normalization, dict):
            raise ValueError("red-five Runtime Model must declare provenance.normalization")
        mean = normalization.get("mean")
        std = normalization.get("std")
        if not isinstance(mean, list) or not isinstance(std, list):
            raise ValueError("red-five normalization is malformed")
        result["normalization"] = {"mean": mean, "std": std}
    return result


def _classifier_runtime(context) -> dict[str, object]:
    training_result = context.model.training_result
    training_corpus = training_result.get("corpus")
    if not isinstance(training_corpus, str):
        raise ValueError("classifier TrainingResult corpus is missing")
    runtime = _BASE_RUNTIME_BY_TRAINING_CORPUS.get(training_corpus)
    if runtime is None:
        raise ValueError(
            "classifier training corpus has no recognized browser preprocessing contract: "
            + training_corpus
        )
    return {"training_corpus": training_corpus, **runtime}


def _load_take_configs(context) -> list[dict[str, object]]:
    corpus = context.corpus.definition
    storage = corpus.get("storage")
    if not isinstance(storage, dict) or not isinstance(storage.get("root_uri"), str):
        raise ValueError("recognition E2E Corpus storage.root_uri is missing")
    root_uri = str(storage["root_uri"]).rstrip("/")
    gt_path = context.corpus.root / "ground-truth.json"
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    takes_gt = ground_truth.get("takes") if isinstance(ground_truth, dict) else None
    if not isinstance(takes_gt, dict):
        raise ValueError("recognition E2E ground-truth.json has no takes mapping")

    result: list[dict[str, object]] = []
    for take_id in _EXPECTED_TAKES:
        capture_path = context.corpus.root / take_id / "capture.json"
        video_path = context.corpus.root / take_id / "video.mp4"
        if not capture_path.is_file() or not video_path.is_file():
            raise ValueError(f"recognition E2E Corpus is missing {take_id} assets")
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        gt = takes_gt.get(take_id)
        if not isinstance(gt, dict):
            raise ValueError(f"ground truth is missing {take_id}")
        result.append(
            {
                "id": take_id,
                "videoUrl": _s3_uri_to_asset_url(f"{root_uri}/{take_id}/video.mp4"),
                "capture": {
                    "logicalCapture": capture["logicalCapture"],
                    "recognitionRegions": capture["recognitionRegions"],
                },
                "groundTruth": {
                    "completed_hand": gt["completed_hand"],
                    "dora_indicators": gt["dora_indicators"],
                    "melds": gt["melds"],
                },
            }
        )
    return result


def _browser_bundle() -> str:
    try:
        payload = zlib.decompress(base64.b64decode(_BROWSER_BUNDLE_ZLIB_BASE64, validate=True))
    except Exception as error:
        raise RuntimeError("recognition E2E embedded browser bundle is malformed") from error
    if hashlib.sha256(payload).hexdigest() != _BROWSER_BUNDLE_SHA256:
        raise RuntimeError("recognition E2E browser bundle sha256 mismatch")
    return payload.decode("utf-8").replace("</script", "<\\/script")


def _document(config: dict[str, object]) -> str:
    config_json = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
    return (
        "<!doctype html><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<meta name=\"robots\" content=\"noindex\"><title>MLDB recognition E2E</title>"
        "<pre id=\"status\">running recognition E2E</pre>"
        "<script>globalThis.__MLDB_RECOGNITION_E2E_CONFIG__="
        + config_json.replace("</script", "<\\/script")
        + ";</script><script type=\"module\">"
        + _browser_bundle()
        + "</script>"
    )


def _numeric(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"browser result {label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"browser result {label} is not finite")
    return result


def _aggregate_browser_result(result: dict[str, Any]) -> tuple[dict[str, float], dict[str, object]]:
    if result.get("ok") is not True:
        raise RuntimeError("browser E2E page reported failure: " + str(result.get("error")))
    raw_takes = result.get("takes")
    if not isinstance(raw_takes, list) or len(raw_takes) != len(_EXPECTED_TAKES):
        raise RuntimeError("browser E2E result has unexpected take count")

    total_evaluations = 0
    total_ticks = 0
    total_skips = 0
    total_eligible = 0
    exact_frames = 0
    hand_exact = 0
    dora_exact = 0
    meld_exact = 0
    confirmed = 0
    confirmed_exact = 0
    durations = 0.0
    total_latency_samples: list[float] = []

    for index, take in enumerate(raw_takes):
        if not isinstance(take, dict) or take.get("id") != _EXPECTED_TAKES[index]:
            raise RuntimeError("browser E2E take identity mismatch")
        evaluations = int(_numeric(take.get("evaluations"), "evaluations"))
        ticks = int(_numeric(take.get("ticks"), "ticks"))
        skips = int(_numeric(take.get("skipped_in_flight"), "skipped_in_flight"))
        total_evaluations += evaluations
        total_ticks += ticks
        total_skips += skips
        total_eligible += int(_numeric(take.get("eligible_frames"), "eligible_frames"))
        exact_frames += int(_numeric(take.get("exact_frames"), "exact_frames"))
        hand_exact += int(_numeric(take.get("completed_hand_exact_frames"), "completed_hand_exact_frames"))
        dora_exact += int(_numeric(take.get("dora_exact_frames"), "dora_exact_frames"))
        meld_exact += int(_numeric(take.get("meld_exact_frames"), "meld_exact_frames"))
        durations += _numeric(take.get("source_video_duration_sec"), "source_video_duration_sec")
        if take.get("first_confirmed_video_time_sec") is not None:
            confirmed += 1
        if take.get("confirmed_exact") is True:
            confirmed_exact += 1
        timing = take.get("timing")
        samples = timing.get("samples") if isinstance(timing, dict) else None
        if not isinstance(samples, list):
            raise RuntimeError("browser E2E timing.samples is missing")
        for sample in samples:
            if not isinstance(sample, dict):
                raise RuntimeError("browser E2E timing sample is malformed")
            total_latency_samples.append(_numeric(sample.get("totalMs"), "timing.totalMs"))

    if total_evaluations <= 0 or not total_latency_samples:
        raise RuntimeError("browser E2E produced no evaluations")
    ordered = sorted(total_latency_samples)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    metrics = {
        "frame_semantic_exact_rate": exact_frames / total_evaluations,
        "completed_hand_exact_rate": hand_exact / total_evaluations,
        "dora_exact_rate": dora_exact / total_evaluations,
        "meld_exact_rate": meld_exact / total_evaluations,
        "eligible_frame_rate": total_eligible / total_evaluations,
        "take_confirmed_rate": confirmed / len(_EXPECTED_TAKES),
        "take_confirmed_exact_rate": confirmed_exact / len(_EXPECTED_TAKES),
        "latency_mean_ms": float(statistics.fmean(total_latency_samples)),
        "latency_p50_ms": float(statistics.median(total_latency_samples)),
        "latency_p95_ms": float(ordered[p95_index]),
        "effective_eval_hz": total_evaluations / durations if durations > 0 else 0.0,
        "cadence_skip_rate": total_skips / total_ticks if total_ticks > 0 else 0.0,
    }
    summary = {
        "evaluations": total_evaluations,
        "ticks": total_ticks,
        "skipped_in_flight": total_skips,
        "latency_sample_count": len(total_latency_samples),
        "source_duration_sec": durations,
    }
    return metrics, summary


def evaluate(context):
    detector = _runtime_model_config(context.models["detector"], expected_role="detector")
    red_five = _runtime_model_config(
        context.models["red-five-classifier"], expected_role="red-five-classifier"
    )
    classifier_runtime = _classifier_runtime(context)

    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = work_dir / "base-classifier.onnx"
    report_path = work_dir / "recognition-e2e-iphone-report.json"
    export_info = _export_onnx(context.model.module, onnx_path)
    onnx_bytes = onnx_path.read_bytes()

    config = {
        "baseClassifierBase64": base64.b64encode(onnx_bytes).decode("ascii"),
        "baseClassifierSha256": export_info["sha256"],
        "baseClassifierRuntimeSpec": classifier_runtime["runtime_spec"],
        "baseNormalization": classifier_runtime["normalization"],
        "detector": {
            "url": detector["url"],
            "sha256": detector["sha256"],
            "runtimeSpec": detector["runtimeSpec"],
        },
        "redFive": {
            "url": red_five["url"],
            "sha256": red_five["sha256"],
            "runtimeSpec": red_five["runtimeSpec"],
            "normalization": red_five["normalization"],
        },
        "takes": _load_take_configs(context),
    }
    document = _document(config)
    browser_result, runner_health = _run_browser_job(document)
    metrics, aggregate = _aggregate_browser_result(browser_result)

    for key, value in metrics.items():
        context.telemetry.report_scalar(
            group="recognition-e2e",
            series=key,
            value=float(value),
            step=0,
        )

    udid = runner_health.get("udid")
    report = {
        "schema": "mjtensu.recognition/e2e-iphone/v1",
        "models": {
            "base_classifier": {
                "id": context.model.definition.get("id"),
                "training_result": context.model.training_result.get("id"),
                "training_corpus": classifier_runtime["training_corpus"],
                "runtime_spec": classifier_runtime["runtime_spec"],
                "normalization": classifier_runtime["normalization"],
                "export": export_info,
            },
            "detector": {
                "id": detector["id"],
                "runtime_spec": detector["runtimeSpec"],
                "sha256": detector["sha256"],
            },
            "red_five_classifier": {
                "id": red_five["id"],
                "runtime_spec": red_five["runtimeSpec"],
                "sha256": red_five["sha256"],
            },
        },
        "corpus": context.corpus.definition.get("id"),
        "cadence_ms": 100,
        "measurement_scope": (
            "Five fixed 30-second iPhone videos are replayed at normal speed. Recognition requests "
            "are scheduled at the production 100 ms cadence and skipped while one evaluation is in flight. "
            "Each evaluation uses the production detector preprocessing/postprocessing, crop extraction, "
            "base classifier, red-five specialist, semantic grouping, and three-consecutive stabilizer."
        ),
        "metrics": metrics,
        "aggregate": aggregate,
        "browser": browser_result,
        "runner": {
            "device_id_sha256": (
                hashlib.sha256(udid.encode("utf-8")).hexdigest()
                if isinstance(udid, str) and udid
                else None
            )
        },
        "transport": {
            "browser_runner": _browser_runner_url(),
            "asset_base_url": _asset_base_url(),
            "onnxruntime_web_version": _ORT_WEB_VERSION,
            "document_bytes": len(document.encode("utf-8")),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return EvaluationCandidate(
        metrics=metrics,
        artifacts={"onnx_model": onnx_path, "e2e_report": report_path},
    )
