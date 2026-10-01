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
_BROWSER_BUNDLE_SHA256 = "5a87e6850092125f65a8d495485385d349e8d2c2f6635bd25446b18c324e79de"
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
