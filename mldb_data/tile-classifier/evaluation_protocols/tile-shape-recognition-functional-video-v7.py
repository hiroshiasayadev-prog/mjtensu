from __future__ import annotations

import base64
import csv
import hashlib
import inspect
import json
import math
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
_HONOR_GT_TO_TILE_KIND = {
    "east": "1z",
    "south": "2z",
    "west": "3z",
    "north": "4z",
    "white": "5z",
    "green": "6z",
    "red": "7z",
}
_BROWSER_BUNDLE_SHA256 = "cc201bff66a3115766bd954d5ab37d9de5ab26672e1aad7733e2f09ba435eb0f"
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


def _detector_model_config(loaded: object, *, work_dir: Path) -> dict[str, object]:
    definition = getattr(loaded, "definition", None)
    artifact_bytes = getattr(loaded, "artifact", None)
    if isinstance(definition, dict) and isinstance(artifact_bytes, bytes):
        return _runtime_model_config(loaded, expected_role="detector")

    module = getattr(loaded, "module", None)
    architecture = getattr(loaded, "architecture", None)
    if not isinstance(definition, dict) or not isinstance(module, nn.Module) or not isinstance(architecture, dict):
        raise ValueError("detector must resolve to a canonical MLDB Model or detector Runtime Model")
    architecture_id = architecture.get("id")
    interface = architecture.get("interface")
    output = interface.get("output") if isinstance(interface, dict) else None
    output_shape = output.get("shape") if isinstance(output, dict) else None
    if (
        architecture_id != "nanodet/nanodet-plus-m320-ghostpan-baseline-v1"
        or output_shape != ["N", 2125, 33]
    ):
        raise ValueError(
            "canonical detector Model is not compatible with nanodet-plus-m-320-v1 browser runtime"
        )

    source = module.to("cpu").eval()
    example = torch.zeros((1, 3, 320, 320), dtype=torch.float32)
    with torch.inference_mode():
        expected = source(example)
    if not isinstance(expected, torch.Tensor) or tuple(expected.shape) != (1, 2125, 33):
        raise RuntimeError(
            f"canonical detector forward shape mismatch: {getattr(expected, 'shape', None)!r}"
        )

    detector_path = work_dir / "detector.onnx"
    kwargs: dict[str, object] = {
        "export_params": True,
        "opset_version": _OPSET_VERSION,
        "do_constant_folding": True,
        "input_names": ["images"],
        "output_names": ["output"],
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(source, example, str(detector_path), **kwargs)
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("onnx is required for canonical NanoDet browser export") from error
    onnx.checker.check_model(onnx.load(str(detector_path)))

    payload = detector_path.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    return {
        "id": definition.get("id"),
        "url": "inline://mldb-aux-detector",
        "sha256": sha256,
        "runtimeSpec": "nanodet-plus-m-320-v1",
        "inlineBase64": base64.b64encode(payload).decode("ascii"),
        "export": {
            "mode": "canonical-model-torch-onnx",
            "opset_version": _OPSET_VERSION,
            "bytes": len(payload),
            "sha256": sha256,
            "output_shape": [1, 2125, 33],
        },
    }


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


def _canonical_gt_tile(tile: object) -> dict[str, object]:
    if not isinstance(tile, dict):
        raise ValueError("recognition functional GT tile must be an object")
    kind = tile.get("kind")
    red = tile.get("red")
    if not isinstance(kind, str) or not isinstance(red, bool):
        raise ValueError("recognition functional GT tile kind/red is invalid")
    return {"kind": _HONOR_GT_TO_TILE_KIND.get(kind, kind), "red": red}


def _canonical_ground_truth(gt: dict[str, object]) -> dict[str, object]:
    completed_hand = gt.get("completed_hand")
    dora_indicators = gt.get("dora_indicators")
    melds = gt.get("melds")
    if not isinstance(completed_hand, list) or not isinstance(dora_indicators, list) or not isinstance(melds, list):
        raise ValueError("recognition functional GT arrays are invalid")
    canonical_melds: list[dict[str, object]] = []
    for meld in melds:
        if not isinstance(meld, dict) or not isinstance(meld.get("kind"), str) or not isinstance(meld.get("tiles"), list):
            raise ValueError("recognition functional GT meld is invalid")
        canonical_melds.append({
            "kind": meld["kind"],
            "tiles": [_canonical_gt_tile(tile) for tile in meld["tiles"]],
        })
    return {
        "completed_hand": [_canonical_gt_tile(tile) for tile in completed_hand],
        "dora_indicators": [_canonical_gt_tile(tile) for tile in dora_indicators],
        "melds": canonical_melds,
    }


def _load_take_configs(context) -> list[dict[str, object]]:
    corpus = context.corpus.definition
    storage = corpus.get("storage")
    if not isinstance(storage, dict) or not isinstance(storage.get("root_uri"), str):
        raise ValueError("recognition functional Corpus storage.root_uri is missing")
    root_uri = str(storage["root_uri"]).rstrip("/")
    ground_truth = json.loads(
        (context.corpus.root / "ground-truth.json").read_text(encoding="utf-8")
    )
    takes_gt = ground_truth.get("takes") if isinstance(ground_truth, dict) else None
    if not isinstance(takes_gt, dict):
        raise ValueError("recognition functional ground-truth.json has no takes mapping")

    result: list[dict[str, object]] = []
    for take_id in _EXPECTED_TAKES:
        capture_path = context.corpus.root / take_id / "capture.json"
        video_path = context.corpus.root / take_id / "video.mp4"
        if not capture_path.is_file() or not video_path.is_file():
            raise ValueError(f"recognition functional Corpus is missing {take_id} assets")
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        frame_rate = (capture.get("trackSettings") or {}).get("frameRate")
        gt = takes_gt.get(take_id)
        if (
            not isinstance(frame_rate, (int, float))
            or isinstance(frame_rate, bool)
            or frame_rate <= 0
            or not isinstance(gt, dict)
        ):
            raise ValueError(f"recognition functional metadata is invalid for {take_id}")
        result.append(
            {
                "id": take_id,
                "videoUrl": _s3_uri_to_asset_url(f"{root_uri}/{take_id}/video.mp4"),
                "sourceFps": float(frame_rate),
                "capture": {
                    "logicalCapture": capture["logicalCapture"],
                    "recognitionRegions": capture["recognitionRegions"],
                },
                "groundTruth": _canonical_ground_truth(gt),
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


def _run_local_browser(
    config: dict[str, object],
    *,
    timeout_sec: float = 900.0,
) -> dict[str, Any]:
    state: dict[str, object] = {}
    completed = threading.Event()
    document_box: dict[str, bytes] = {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *args: object) -> None:
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send(HTTPStatus.OK, document_box["document"], "text/html; charset=utf-8")
                return
            if self.path == "/healthz":
                self._send(HTTPStatus.OK, b"ok\n", "text/plain")
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/result":
                self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send(HTTPStatus.BAD_REQUEST, b"bad length\n", "text/plain")
                return
            if length <= 0 or length > 256 * 1024 * 1024:
                self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, b"bad body\n", "text/plain")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                self._send(HTTPStatus.BAD_REQUEST, b"bad json\n", "text/plain")
                return
            if not isinstance(payload, dict):
                self._send(HTTPStatus.BAD_REQUEST, b"bad result\n", "text/plain")
                return
            state["result"] = payload
            completed.set()
            self._send(HTTPStatus.OK, b"ok\n", "text/plain")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = int(server.server_address[1])
    browser_config = dict(config)
    browser_config["resultUrl"] = f"http://127.0.0.1:{port}/result"
    document_box["document"] = _document(browser_config).encode("utf-8")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    chrome = os.environ.get("MLDB_FUNCTIONAL_CHROME", "google-chrome")
    command = [
        chrome,
        "--headless=new",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--autoplay-policy=no-user-gesture-required",
        "--disable-background-networking",
        "--disable-default-apps",
        "--disable-extensions",
        "--disable-sync",
        f"http://127.0.0.1:{port}/",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if not completed.wait(timeout_sec):
            output = ""
            if process.poll() is not None and process.stdout is not None:
                output = process.stdout.read()[-8000:]
            raise RuntimeError(
                "functional browser result timed out"
                + (f": {output}" if output else "")
            )
        result = state.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("functional browser returned no result object")
        return result
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        server.shutdown()
        server.server_close()


def _aggregate_functional_result(
    result: dict[str, Any],
) -> tuple[dict[str, float], dict[str, object]]:
    if result.get("ok") is not True or result.get("mode") != "functional":
        raise RuntimeError("functional browser page reported failure: " + str(result.get("error")))
    raw_takes = result.get("takes")
    if not isinstance(raw_takes, list) or len(raw_takes) != len(_EXPECTED_TAKES):
        raise RuntimeError("functional browser result has unexpected take count")

    evaluations = eligible = exact = hand = dora = meld = 0
    gt_streak3_takes = product_confirmed_takes = product_exact_takes = 0
    for index, take in enumerate(raw_takes):
        if not isinstance(take, dict) or take.get("id") != _EXPECTED_TAKES[index]:
            raise RuntimeError("functional take identity mismatch")
        count = int(_numeric(take.get("evaluations"), "evaluations"))
        evaluations += count
        eligible += int(_numeric(take.get("eligible_frames"), "eligible_frames"))
        exact += int(_numeric(take.get("exact_frames"), "exact_frames"))
        hand += int(_numeric(take.get("completed_hand_exact_frames"), "completed_hand_exact_frames"))
        dora += int(_numeric(take.get("dora_exact_frames"), "dora_exact_frames"))
        meld += int(_numeric(take.get("meld_exact_frames"), "meld_exact_frames"))
        if take.get("first_gt_streak3") is not None:
            gt_streak3_takes += 1
        confirmation = take.get("first_product_confirm")
        if isinstance(confirmation, dict):
            product_confirmed_takes += 1
            if confirmation.get("exact") is True:
                product_exact_takes += 1

    if evaluations <= 0:
        raise RuntimeError("functional browser produced no evaluations")
    take_count = len(_EXPECTED_TAKES)
    metrics = {
        "frame_semantic_exact_rate": exact / evaluations,
        "completed_hand_exact_rate": hand / evaluations,
        "dora_exact_rate": dora / evaluations,
        "meld_exact_rate": meld / evaluations,
        "eligible_frame_rate": eligible / evaluations,
        "take_gt_streak3_rate": gt_streak3_takes / take_count,
        "take_product_confirmed_rate": product_confirmed_takes / take_count,
        "take_product_confirmed_exact_rate": product_exact_takes / take_count,
    }
    return metrics, {
        "evaluations": evaluations,
        "take_count": take_count,
        "gt_streak3_takes": gt_streak3_takes,
        "product_confirmed_takes": product_confirmed_takes,
        "product_confirmed_exact_takes": product_exact_takes,
    }



_DIAGNOSTIC_REGIONS = ("completed-hand", "dora-indicators", "melds")
_REGION_GT_KEYS = {
    "completed-hand": "completed_hand",
    "dora-indicators": "dora_indicators",
    "melds": "melds",
}


def _tile_token(tile: object) -> str:
    if not isinstance(tile, dict):
        raise RuntimeError("detector semantic diagnostic tile is malformed")
    kind = tile.get("kind")
    red = tile.get("red")
    if not isinstance(kind, str) or not kind or type(red) is not bool:
        raise RuntimeError("detector semantic diagnostic tile identity is malformed")
    return kind + ("R" if red else "")


def _gt_region_tokens(ground_truth: dict[str, object], region: str) -> list[str]:
    if region == "melds":
        melds = ground_truth.get("melds")
        if not isinstance(melds, list):
            raise RuntimeError("detector semantic diagnostic meld GT is malformed")
        tokens: list[str] = []
        for meld in melds:
            if not isinstance(meld, dict) or not isinstance(meld.get("tiles"), list):
                raise RuntimeError("detector semantic diagnostic meld GT group is malformed")
            tokens.extend(_tile_token(tile) for tile in meld["tiles"])
        return tokens
    key = _REGION_GT_KEYS[region]
    tiles = ground_truth.get(key)
    if not isinstance(tiles, list):
        raise RuntimeError(f"detector semantic diagnostic GT is malformed for {region}")
    return [_tile_token(tile) for tile in tiles]


def _observation_tile_token(observation: object) -> str | None:
    if not isinstance(observation, dict):
        return None
    classification = observation.get("classification")
    if not isinstance(classification, dict) or classification.get("kind") != "tile":
        return None
    return _tile_token(classification.get("tile"))


def _observation_geometry(observation: dict[str, object]) -> tuple[float, float, float]:
    obb = observation.get("obb")
    if isinstance(obb, dict):
        try:
            return float(obb["cx"]), float(obb["cy"]), float(obb["height"])
        except (KeyError, TypeError, ValueError):
            pass
    bbox = observation.get("bbox")
    if not isinstance(bbox, dict):
        raise RuntimeError("detector semantic diagnostic observation has no bbox")
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        width = float(bbox["width"])
        height = float(bbox["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("detector semantic diagnostic bbox is malformed") from error
    return x + width / 2.0, y + height / 2.0, height


def _fallback_meld_order(observations: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(observations) <= 1:
        return list(observations)
    heights = sorted(_observation_geometry(observation)[2] for observation in observations)
    median_height = heights[len(heights) // 2]
    row_tolerance = max(1.0e-9, median_height * 0.55)
    pending = sorted(
        observations,
        key=lambda observation: (
            _observation_geometry(observation)[1],
            _observation_geometry(observation)[0],
            str(observation.get("id", "")),
        ),
    )
    rows: list[list[dict[str, object]]] = []
    row_means: list[float] = []
    for observation in pending:
        _x, cy, _height = _observation_geometry(observation)
        if not rows or abs(cy - row_means[-1]) > row_tolerance:
            rows.append([observation])
            row_means.append(cy)
            continue
        rows[-1].append(observation)
        row_means[-1] = sum(_observation_geometry(item)[1] for item in rows[-1]) / len(rows[-1])
    ordered: list[dict[str, object]] = []
    for row in rows:
        ordered.extend(sorted(
            row,
            key=lambda observation: (
                _observation_geometry(observation)[0],
                _observation_geometry(observation)[1],
                str(observation.get("id", "")),
            ),
        ))
    return ordered


def _predicted_region_tokens(
    row: dict[str, object],
    region: str,
) -> tuple[list[str], bool, int]:
    snapshot = row.get("snapshot")
    if not isinstance(snapshot, dict):
        raise RuntimeError("detector semantic diagnostic row has no snapshot")
    draft = snapshot.get("draft")
    observations_raw = snapshot.get("observations")
    if not isinstance(draft, dict) or not isinstance(observations_raw, list):
        raise RuntimeError("detector semantic diagnostic snapshot is malformed")

    invalid_count = 0
    observations: list[dict[str, object]] = []
    for raw in observations_raw:
        if not isinstance(raw, dict) or raw.get("region") != region:
            continue
        classification = raw.get("classification")
        if isinstance(classification, dict) and classification.get("kind") == "tile":
            observations.append(raw)
        else:
            invalid_count += 1

    if region == "completed-hand":
        tiles = draft.get("completedHand")
        if not isinstance(tiles, list):
            raise RuntimeError("completed-hand production draft is malformed")
        return [_tile_token(tile) for tile in tiles], False, invalid_count
    if region == "dora-indicators":
        tiles = draft.get("doraIndicators")
        if not isinstance(tiles, list):
            raise RuntimeError("dora production draft is malformed")
        return [_tile_token(tile) for tile in tiles], False, invalid_count

    by_id = {
        str(observation.get("id")): observation
        for observation in observations
        if isinstance(observation.get("id"), str)
    }
    groups = snapshot.get("meldGroups")
    ordered: list[dict[str, object]] = []
    used: set[str] = set()
    if isinstance(groups, list) and groups:
        for group in groups:
            if not isinstance(group, dict):
                continue
            member_ids = group.get("memberObservationIds")
            if not isinstance(member_ids, list):
                continue
            for member_id in member_ids:
                if not isinstance(member_id, str) or member_id in used:
                    continue
                observation = by_id.get(member_id)
                if observation is not None:
                    ordered.append(observation)
                    used.add(member_id)
        remaining = [
            observation
            for observation in observations
            if str(observation.get("id")) not in used
        ]
        ordered.extend(_fallback_meld_order(remaining))
        used_fallback = bool(remaining)
    else:
        ordered = _fallback_meld_order(observations)
        used_fallback = bool(observations)

    return [
        token
        for token in (_observation_tile_token(observation) for observation in ordered)
        if token is not None
    ], used_fallback, invalid_count


def _align_tile_sequences(
    ground_truth: list[str],
    predicted: list[str],
) -> list[dict[str, str | None]]:
    rows = len(ground_truth) + 1
    cols = len(predicted) + 1
    cost = [[0] * cols for _ in range(rows)]
    operation = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0] = i
        operation[i][0] = "deletion"
    for j in range(1, cols):
        cost[0][j] = j
        operation[0][j] = "insertion"

    priority = {"match": 0, "substitution": 1, "deletion": 2, "insertion": 3}
    for i in range(1, rows):
        for j in range(1, cols):
            same = ground_truth[i - 1] == predicted[j - 1]
            candidates = [
                (cost[i - 1][j - 1] + (0 if same else 1), "match" if same else "substitution"),
                (cost[i - 1][j] + 1, "deletion"),
                (cost[i][j - 1] + 1, "insertion"),
            ]
            best_cost, best_op = min(candidates, key=lambda item: (item[0], priority[item[1]]))
            cost[i][j] = best_cost
            operation[i][j] = best_op

    aligned: list[dict[str, str | None]] = []
    i = len(ground_truth)
    j = len(predicted)
    while i > 0 or j > 0:
        current = operation[i][j]
        if current in {"match", "substitution"}:
            aligned.append({"kind": current, "gt": ground_truth[i - 1], "predicted": predicted[j - 1]})
            i -= 1
            j -= 1
        elif current == "deletion":
            aligned.append({"kind": "deletion", "gt": ground_truth[i - 1], "predicted": None})
            i -= 1
        elif current == "insertion":
            aligned.append({"kind": "insertion", "gt": None, "predicted": predicted[j - 1]})
            j -= 1
        else:
            raise RuntimeError("detector semantic alignment backtrace is malformed")
    aligned.reverse()
    return aligned


def _empty_detector_semantic_bucket() -> dict[str, object]:
    return {
        "frames": 0,
        "gt_tiles": 0,
        "predicted_tiles": 0,
        "correct": 0,
        "substitutions": 0,
        "insertions": 0,
        "deletions": 0,
        "exact_sequence_frames": 0,
        "fallback_order_frames": 0,
        "invalid_detections": 0,
        "substitution_pairs": {},
        "inserted_predictions": {},
        "deleted_gt": {},
    }


def _increment_count(mapping: dict[str, object], key: str) -> None:
    mapping[key] = int(mapping.get(key, 0)) + 1


def _accumulate_detector_semantic_bucket(
    bucket: dict[str, object],
    *,
    ground_truth: list[str],
    predicted: list[str],
    alignment: list[dict[str, str | None]],
    used_fallback: bool,
    invalid_count: int,
) -> None:
    bucket["frames"] = int(bucket["frames"]) + 1
    bucket["gt_tiles"] = int(bucket["gt_tiles"]) + len(ground_truth)
    bucket["predicted_tiles"] = int(bucket["predicted_tiles"]) + len(predicted)
    bucket["exact_sequence_frames"] = int(bucket["exact_sequence_frames"]) + int(ground_truth == predicted)
    bucket["fallback_order_frames"] = int(bucket["fallback_order_frames"]) + int(used_fallback)
    bucket["invalid_detections"] = int(bucket["invalid_detections"]) + invalid_count

    substitution_pairs = bucket["substitution_pairs"]
    inserted_predictions = bucket["inserted_predictions"]
    deleted_gt = bucket["deleted_gt"]
    if not isinstance(substitution_pairs, dict) or not isinstance(inserted_predictions, dict) or not isinstance(deleted_gt, dict):
        raise RuntimeError("detector semantic diagnostic accumulator is malformed")

    for item in alignment:
        kind = item["kind"]
        if kind == "match":
            bucket["correct"] = int(bucket["correct"]) + 1
        elif kind == "substitution":
            bucket["substitutions"] = int(bucket["substitutions"]) + 1
            _increment_count(substitution_pairs, f"{item['gt']}->{item['predicted']}")
        elif kind == "insertion":
            bucket["insertions"] = int(bucket["insertions"]) + 1
            predicted_token = item["predicted"]
            if isinstance(predicted_token, str):
                _increment_count(inserted_predictions, predicted_token)
        elif kind == "deletion":
            bucket["deletions"] = int(bucket["deletions"]) + 1
            gt_token = item["gt"]
            if isinstance(gt_token, str):
                _increment_count(deleted_gt, gt_token)


def _finalize_detector_semantic_bucket(bucket: dict[str, object]) -> dict[str, object]:
    correct = int(bucket["correct"])
    substitutions = int(bucket["substitutions"])
    insertions = int(bucket["insertions"])
    deletions = int(bucket["deletions"])
    precision_denominator = correct + substitutions + insertions
    recall_denominator = correct + substitutions + deletions
    precision = correct / precision_denominator if precision_denominator else 1.0
    recall = correct / recall_denominator if recall_denominator else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    frames = int(bucket["frames"])
    result = dict(bucket)
    result.update({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_sequence_rate": int(bucket["exact_sequence_frames"]) / frames if frames else 0.0,
    })
    for key in ("substitution_pairs", "inserted_predictions", "deleted_gt"):
        mapping = result[key]
        if isinstance(mapping, dict):
            result[key] = dict(sorted(mapping.items(), key=lambda item: (-int(item[1]), item[0])))
    return result


def _write_functional_take_summary_csv(path: Path, takes: list[dict[str, object]]) -> None:
    fields = [
        "take_id",
        "evaluations",
        "eligible_frames",
        "eligible_rate",
        "exact_frame_rate",
        "completed_hand_exact_rate",
        "dora_exact_rate",
        "meld_exact_rate",
        "first_gt_streak3_eval_index",
        "first_gt_streak3_time_sec",
        "first_product_confirm_eval_index",
        "first_product_confirm_time_sec",
        "first_product_confirm_exact",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for take in takes:
            evaluations = int(take.get("evaluations", 0) or 0)
            eligible_frames = int(take.get("eligible_frames", 0) or 0)
            gt_streak = take.get("first_gt_streak3")
            confirm = take.get("first_product_confirm")
            gt_streak_map = gt_streak if isinstance(gt_streak, dict) else {}
            confirm_map = confirm if isinstance(confirm, dict) else {}
            writer.writerow({
                "take_id": str(take.get("id", "")),
                "evaluations": evaluations,
                "eligible_frames": eligible_frames,
                "eligible_rate": eligible_frames / evaluations if evaluations else 0.0,
                "exact_frame_rate": float(take.get("exact_frame_rate", 0.0) or 0.0),
                "completed_hand_exact_rate": float(take.get("completed_hand_exact_rate", 0.0) or 0.0),
                "dora_exact_rate": float(take.get("dora_exact_rate", 0.0) or 0.0),
                "meld_exact_rate": float(take.get("meld_exact_rate", 0.0) or 0.0),
                "first_gt_streak3_eval_index": gt_streak_map.get("eval_index", ""),
                "first_gt_streak3_time_sec": gt_streak_map.get("video_time_sec", ""),
                "first_product_confirm_eval_index": confirm_map.get("eval_index", ""),
                "first_product_confirm_time_sec": confirm_map.get("video_time_sec", ""),
                "first_product_confirm_exact": confirm_map.get("exact", ""),
            })


def _write_detector_semantic_take_summary_csv(path: Path, takes: dict[str, object]) -> None:
    fields = [
        "take_id",
        "region",
        "frames",
        "gt_tiles",
        "predicted_tiles",
        "correct",
        "substitutions",
        "insertions",
        "deletions",
        "invalid_detections",
        "fallback_order_frames",
        "precision",
        "recall",
        "f1",
        "exact_sequence_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for take_id in sorted(takes):
            per_region = takes[take_id]
            if not isinstance(per_region, dict):
                continue
            for region in _DIAGNOSTIC_REGIONS:
                summary = per_region.get(region)
                if not isinstance(summary, dict):
                    continue
                writer.writerow({
                    "take_id": take_id,
                    "region": region,
                    "frames": int(summary.get("frames", 0) or 0),
                    "gt_tiles": int(summary.get("gt_tiles", 0) or 0),
                    "predicted_tiles": int(summary.get("predicted_tiles", 0) or 0),
                    "correct": int(summary.get("correct", 0) or 0),
                    "substitutions": int(summary.get("substitutions", 0) or 0),
                    "insertions": int(summary.get("insertions", 0) or 0),
                    "deletions": int(summary.get("deletions", 0) or 0),
                    "invalid_detections": int(summary.get("invalid_detections", 0) or 0),
                    "fallback_order_frames": int(summary.get("fallback_order_frames", 0) or 0),
                    "precision": float(summary.get("precision", 0.0) or 0.0),
                    "recall": float(summary.get("recall", 0.0) or 0.0),
                    "f1": float(summary.get("f1", 0.0) or 0.0),
                    "exact_sequence_rate": float(summary.get("exact_sequence_rate", 0.0) or 0.0),
                })


def _detector_semantic_diagnostics(
    browser_result: dict[str, Any],
    take_configs: list[dict[str, object]],
) -> tuple[dict[str, float], dict[str, object]]:
    raw_takes = browser_result.get("takes")
    if not isinstance(raw_takes, list):
        raise RuntimeError("functional result has no takes for detector semantic diagnostics")

    gt_by_take: dict[str, dict[str, object]] = {}
    for take in take_configs:
        take_id = take.get("id")
        ground_truth = take.get("groundTruth")
        if isinstance(take_id, str) and isinstance(ground_truth, dict):
            gt_by_take[take_id] = ground_truth

    aggregate = {region: _empty_detector_semantic_bucket() for region in _DIAGNOSTIC_REGIONS}
    per_take: dict[str, object] = {}
    frame_alignments: dict[tuple[str, int], dict[str, object]] = {}
    for raw_take in raw_takes:
        if not isinstance(raw_take, dict):
            raise RuntimeError("functional detector semantic take is malformed")
        take_id = raw_take.get("id")
        rows = raw_take.get("rows")
        if not isinstance(take_id, str) or not isinstance(rows, list):
            raise RuntimeError("functional detector semantic take trace is malformed")
        ground_truth = gt_by_take.get(take_id)
        if ground_truth is None:
            raise RuntimeError(f"missing detector semantic GT for {take_id}")
        take_buckets = {region: _empty_detector_semantic_bucket() for region in _DIAGNOSTIC_REGIONS}

        for raw_row in rows:
            if not isinstance(raw_row, dict):
                raise RuntimeError("functional detector semantic trace row is malformed")
            eval_index = int(_numeric(raw_row.get("eval_index"), "eval_index"))
            aligned_regions: dict[str, object] = {}
            for region in _DIAGNOSTIC_REGIONS:
                gt_tokens = _gt_region_tokens(ground_truth, region)
                predicted_tokens, used_fallback, invalid_count = _predicted_region_tokens(raw_row, region)
                alignment = _align_tile_sequences(gt_tokens, predicted_tokens)
                for bucket in (aggregate[region], take_buckets[region]):
                    _accumulate_detector_semantic_bucket(
                        bucket,
                        ground_truth=gt_tokens,
                        predicted=predicted_tokens,
                        alignment=alignment,
                        used_fallback=used_fallback,
                        invalid_count=invalid_count,
                    )
                aligned_regions[region] = {
                    "gt": gt_tokens,
                    "predicted": predicted_tokens,
                    "alignment": alignment,
                    "used_fallback_order": used_fallback,
                    "invalid_detections": invalid_count,
                }
            frame_alignments[(take_id, eval_index)] = aligned_regions

        per_take[take_id] = {
            region: _finalize_detector_semantic_bucket(bucket)
            for region, bucket in take_buckets.items()
        }

    finalized = {
        region: _finalize_detector_semantic_bucket(bucket)
        for region, bucket in aggregate.items()
    }
    all_bucket = _empty_detector_semantic_bucket()
    scalar_keys = (
        "frames", "gt_tiles", "predicted_tiles", "correct", "substitutions",
        "insertions", "deletions", "exact_sequence_frames", "fallback_order_frames",
        "invalid_detections",
    )
    for region_bucket in aggregate.values():
        for key in scalar_keys:
            all_bucket[key] = int(all_bucket[key]) + int(region_bucket[key])
        for key in ("substitution_pairs", "inserted_predictions", "deleted_gt"):
            target = all_bucket[key]
            source = region_bucket[key]
            if not isinstance(target, dict) or not isinstance(source, dict):
                raise RuntimeError("detector semantic aggregate mapping is malformed")
            for label, count in source.items():
                target[str(label)] = int(target.get(str(label), 0)) + int(count)
    finalized_all = _finalize_detector_semantic_bucket(all_bucket)

    metrics = {
        "completed_hand_detection_semantic_precision": float(finalized["completed-hand"]["precision"]),
        "completed_hand_detection_semantic_recall": float(finalized["completed-hand"]["recall"]),
        "completed_hand_detection_semantic_f1": float(finalized["completed-hand"]["f1"]),
        "dora_detection_semantic_precision": float(finalized["dora-indicators"]["precision"]),
        "dora_detection_semantic_recall": float(finalized["dora-indicators"]["recall"]),
        "dora_detection_semantic_f1": float(finalized["dora-indicators"]["f1"]),
        "meld_detection_semantic_precision": float(finalized["melds"]["precision"]),
        "meld_detection_semantic_recall": float(finalized["melds"]["recall"]),
        "meld_detection_semantic_f1": float(finalized["melds"]["f1"]),
        "all_regions_detection_semantic_f1": float(finalized_all["f1"]),
    }
    report: dict[str, object] = {
        "schema": "mjtensu.recognition/detector-semantic-diagnostics/v1",
        "contract": {
            "purpose": (
                "Detector diagnosis with the fixed production classifier stack treated as a trusted semantic probe. "
                "Tile identity mismatches are substitutions; extra classified tiles are insertions; missing GT tiles are deletions."
            ),
            "precision": "correct / (correct + substitutions + insertions)",
            "recall": "correct / (correct + substitutions + deletions)",
            "ordering": {
                "completed-hand": "production snapshot completedHand order",
                "dora-indicators": "production snapshot doraIndicators order",
                "melds": (
                    "production meld-group/member order when available; otherwise a deterministic "
                    "row-cluster then x-order fallback over raw tile-classified meld observations"
                ),
            },
            "classifier_assumption": (
                "Base classifier and red-five classifier are fixed evaluation dependencies rather than comparison "
                "factors; their semantic output is intentionally used to score useful detections."
            ),
            "invalid_detection_semantics": (
                "Detector boxes rejected by the classifier are counted separately as invalid_detections and are not "
                "treated as predicted tiles in semantic precision/recall."
            ),
        },
        "metrics": metrics,
        "aggregate": {**finalized, "all-regions": finalized_all},
        "takes": per_take,
    }
    return metrics, {"report": report, "frame_alignments": frame_alignments}


def _open_raw_video(path: Path) -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python-headless is required for functional overlay rendering") from error

    source = cv2.VideoCapture(str(path))
    if not source.isOpened():
        raise RuntimeError(f"could not open functional video: {path}")
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None:
        source.set(orientation_auto, 0)
    return source


def _presentation_frame(frame: Any, source: Any, capture: dict[str, Any]) -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python-headless is required for functional overlay rendering") from error

    orientation_meta = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    rotation = 0
    if orientation_meta is not None:
        value = float(source.get(orientation_meta))
        if math.isfinite(value):
            rotation = int(round(value)) % 360

    if rotation == 90:
        presented = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        presented = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        presented = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 0:
        presented = frame
    else:
        raise RuntimeError(f"unsupported MP4 display rotation: {rotation}")

    track = capture.get("trackSettings")
    if isinstance(track, dict):
        expected_width = track.get("width")
        expected_height = track.get("height")
        if (
            isinstance(expected_width, (int, float))
            and not isinstance(expected_width, bool)
            and isinstance(expected_height, (int, float))
            and not isinstance(expected_height, bool)
            and expected_width > 0
            and expected_height > 0
        ):
            actual_height, actual_width = presented.shape[:2]
            expected_aspect = float(expected_width) / float(expected_height)
            actual_aspect = float(actual_width) / float(actual_height)
            if not math.isclose(actual_aspect, expected_aspect, rel_tol=0.01, abs_tol=0.01):
                raise RuntimeError(
                    "decoded video presentation orientation does not match capture trackSettings: "
                    f"decoded={actual_width}x{actual_height} "
                    f"expected_aspect={float(expected_width):g}x{float(expected_height):g} "
                    f"mp4_rotation={rotation}"
                )
    return presented


def _canonical_frame(frame: Any, capture: dict[str, Any]) -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python-headless is required for functional overlay rendering") from error

    logical = capture["logicalCapture"]
    aspect = logical["aspectRatio"]
    rotation = int(logical["rotation"])
    source_h, source_w = frame.shape[:2]
    target_aspect = 16.0 / 9.0 if aspect == "16:9" else 9.0 / 16.0
    source_aspect = source_w / source_h
    if source_aspect > target_aspect:
        crop_w = int(round(source_h * target_aspect))
        x = max(0, (source_w - crop_w) // 2)
        cropped = frame[:, x : x + crop_w]
    else:
        crop_h = int(round(source_w / target_aspect))
        y = max(0, (source_h - crop_h) // 2)
        cropped = frame[y : y + crop_h, :]

    source_units = (16, 9) if aspect == "16:9" else (9, 16)
    output_units = source_units if rotation == 0 else (source_units[1], source_units[0])
    units = max(
        1,
        int(
            min(
                cropped.shape[1] / source_units[0],
                cropped.shape[0] / source_units[1],
                1280 / max(output_units),
            )
        ),
    )
    target_w = units * output_units[0]
    target_h = units * output_units[1]
    if rotation == 0:
        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    pre_rotated = cv2.resize(
        cropped,
        (target_h, target_w),
        interpolation=cv2.INTER_LINEAR,
    )
    if rotation == 90:
        return cv2.rotate(pre_rotated, cv2.ROTATE_90_CLOCKWISE)
    if rotation == -90:
        return cv2.rotate(pre_rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported capture rotation: {rotation}")


def _recognition_region_rects(
    capture: dict[str, Any],
    width: int,
    height: int,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    regions = capture.get("recognitionRegions")
    if not isinstance(regions, dict):
        raise RuntimeError("capture metadata has no recognitionRegions")

    result: list[tuple[str, tuple[int, int, int, int]]] = []
    for name in ("dora-indicators", "completed-hand", "melds"):
        region = regions.get(name)
        if not isinstance(region, dict):
            raise RuntimeError(f"capture metadata is missing recognition region: {name}")
        try:
            x = float(region["x"])
            y = float(region["y"])
            region_width = float(region["width"])
            region_height = float(region["height"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"malformed recognition region: {name}") from error
        x0 = max(0, min(width - 1, int(round(x * width))))
        y0 = max(0, min(height - 1, int(round(y * height))))
        x1 = max(x0 + 1, min(width - 1, int(round((x + region_width) * width))))
        y1 = max(y0 + 1, min(height - 1, int(round((y + region_height) * height))))
        result.append((name, (x0, y0, x1, y1)))
    return result


def _draw_recognition_regions(frame: Any, capture: dict[str, Any]) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python-headless is required for overlay rendering") from error

    colors = {
        "dora-indicators": (255, 140, 0),
        "completed-hand": (255, 0, 255),
        "melds": (0, 200, 0),
    }
    height, width = frame.shape[:2]
    for name, (x0, y0, x1, y1) in _recognition_region_rects(capture, width, height):
        color = colors[name]
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        label_y = y0 + 20 if y0 < 24 else y0 - 6
        label_origin = (x0 + 4, max(18, label_y))
        cv2.putText(
            frame,
            name,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            name,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )


def _classification_label(value: object) -> str:
    if not isinstance(value, dict):
        return "?"
    if value.get("kind") != "tile":
        return "invalid"
    tile = value.get("tile")
    if not isinstance(tile, dict):
        return "?"
    kind = str(tile.get("kind", "?"))
    return kind + ("R" if tile.get("red") is True else "")


def _draw_trace_overlay(
    frame: Any,
    row: dict[str, Any],
    take_id: str,
    capture: dict[str, Any],
) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("opencv-python-headless and numpy are required for overlay rendering") from error

    _draw_recognition_regions(frame, capture)

    detections = row.get("detections")
    if isinstance(detections, list):
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            confidence = float(detection.get("confidence", 0.0))
            label = _classification_label(detection.get("classification"))
            region = str(detection.get("region", "?"))
            box = detection.get("sourceOrientedBox")
            if isinstance(box, dict):
                cx = float(box.get("cx", 0.0))
                cy = float(box.get("cy", 0.0))
                width = float(box.get("width", 0.0))
                height = float(box.get("height", 0.0))
                angle = math.radians(float(box.get("angleDeg", 0.0)))
                corners = []
                for ux, uy in ((-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)):
                    x = cx + ux * math.cos(angle) - uy * math.sin(angle)
                    y = cy + ux * math.sin(angle) + uy * math.cos(angle)
                    corners.append((int(round(x)), int(round(y))))
                points = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], True, (0, 255, 255), 2, cv2.LINE_AA)
                tx, ty = corners[0]
            else:
                rect = detection.get("sourceBox")
                if not isinstance(rect, dict):
                    continue
                x = int(round(float(rect.get("x", 0.0))))
                y = int(round(float(rect.get("y", 0.0))))
                width = int(round(float(rect.get("width", 0.0))))
                height = int(round(float(rect.get("height", 0.0))))
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)
                tx, ty = x, y
            cv2.putText(
                frame,
                f"{label} {confidence:.2f} {region}",
                (max(0, tx), max(18, ty - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    exact = row.get("gt_exact") is True
    streak = int(row.get("gt_exact_consecutive", 0))
    stabilization = row.get("stabilization")
    product_state = stabilization.get("kind", "?") if isinstance(stabilization, dict) else "?"
    hud = (
        f"{take_id}  src#{row.get('source_frame', '?')}  eval#{row.get('eval_index', '?')}  "
        f"GT={'YES' if exact else 'NO'} streak={min(streak, 3)}/3 product={product_state}"
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        frame,
        hud,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 0) if exact else (0, 180, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _render_overlay_video(
    context: Any,
    browser_result: dict[str, Any],
    destination: Path,
) -> dict[str, object]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python-headless is required for functional overlay rendering") from error

    takes = browser_result.get("takes")
    if not isinstance(takes, list):
        raise RuntimeError("functional result has no takes for overlay")

    first_capture = json.loads(
        (context.corpus.root / _EXPECTED_TAKES[0] / "capture.json").read_text(encoding="utf-8")
    )
    first_video = _open_raw_video(context.corpus.root / _EXPECTED_TAKES[0] / "video.mp4")
    ok, first_frame = first_video.read()
    if not ok or first_frame is None:
        first_video.release()
        raise RuntimeError("could not decode first functional video")
    first_presented = _presentation_frame(first_frame, first_video, first_capture)
    first_video.release()
    canonical = _canonical_frame(first_presented, first_capture)
    height, width = canonical.shape[:2]
    output_fps = 30.0

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(output_fps),
        "-i", "-", "-an",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(destination),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    if encoder.stdin is None:
        raise RuntimeError("ffmpeg overlay encoder stdin unavailable")

    total_frames = 0
    try:
        for take_id, take_result in zip(_EXPECTED_TAKES, takes, strict=True):
            if not isinstance(take_result, dict):
                raise RuntimeError("functional take result is malformed")
            rows = take_result.get("rows")
            if not isinstance(rows, list) or not rows:
                raise RuntimeError(f"functional take {take_id} has no trace rows")
            capture = json.loads(
                (context.corpus.root / take_id / "capture.json").read_text(encoding="utf-8")
            )
            source = _open_raw_video(context.corpus.root / take_id / "video.mp4")
            row_index = 0
            frame_index = 0
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                while (
                    row_index + 1 < len(rows)
                    and isinstance(rows[row_index + 1], dict)
                    and int(rows[row_index + 1].get("source_frame", 0)) <= frame_index
                ):
                    row_index += 1
                row = rows[row_index]
                if not isinstance(row, dict):
                    raise RuntimeError("functional trace row is malformed")
                presented = _presentation_frame(frame, source, capture)
                visual = _canonical_frame(presented, capture)
                visual = _draw_trace_overlay(visual, row, take_id, capture)
                encoder.stdin.write(visual.tobytes())
                total_frames += 1
                frame_index += 1
            source.release()
    finally:
        encoder.stdin.close()
        return_code = encoder.wait(timeout=120)
    if return_code != 0 or not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg overlay encoding failed with code {return_code}")
    return {
        "frames": total_frames,
        "fps": output_fps,
        "width": width,
        "height": height,
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def evaluate(context):
    detector_score_threshold = float(context.parameters["detector_score_threshold"])
    if not math.isfinite(detector_score_threshold) or not 0.0 <= detector_score_threshold <= 1.0:
        raise ValueError("detector_score_threshold must be finite and within [0, 1]")

    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    detector = _detector_model_config(context.models["detector"], work_dir=work_dir)
    red_five = _runtime_model_config(
        context.models["red-five-classifier"], expected_role="red-five-classifier"
    )
    classifier_runtime = _classifier_runtime(context)
    onnx_path = work_dir / "base-classifier.onnx"
    report_path = work_dir / "recognition-functional-report.json"
    trace_path = work_dir / "prediction-trace.jsonl"
    overlay_path = work_dir / "recognition-overlay.mp4"
    export_info = _export_onnx(context.model.module, onnx_path)
    onnx_bytes = onnx_path.read_bytes()

    take_configs = _load_take_configs(context)
    config = {
        "mode": "functional",
        "sampleIntervalMs": 100,
        "detectorScoreThreshold": detector_score_threshold,
        "baseClassifierBase64": base64.b64encode(onnx_bytes).decode("ascii"),
        "baseClassifierSha256": export_info["sha256"],
        "baseClassifierRuntimeSpec": classifier_runtime["runtime_spec"],
        "baseNormalization": classifier_runtime["normalization"],
        "detector": {
            "url": detector["url"],
            "sha256": detector["sha256"],
            "runtimeSpec": detector["runtimeSpec"],
            **(
                {"inlineBase64": detector["inlineBase64"]}
                if isinstance(detector.get("inlineBase64"), str)
                else {}
            ),
        },
        "redFive": {
            "url": red_five["url"],
            "sha256": red_five["sha256"],
            "runtimeSpec": red_five["runtimeSpec"],
            "normalization": red_five["normalization"],
        },
        "takes": take_configs,
    }
    browser_result = _run_local_browser(config)
    metrics, aggregate = _aggregate_functional_result(browser_result)
    detector_metrics, detector_diagnostics = _detector_semantic_diagnostics(
        browser_result,
        take_configs,
    )
    metrics.update(detector_metrics)

    diagnostic_path = work_dir / "detector-semantic-diagnostics.json"
    diagnostic_report = detector_diagnostics["report"]
    diagnostic_path.write_text(
        json.dumps(diagnostic_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    frame_alignments = detector_diagnostics["frame_alignments"]
    if not isinstance(frame_alignments, dict):
        raise RuntimeError("detector semantic frame alignment index is malformed")

    rows_written = 0
    with trace_path.open("w", encoding="utf-8") as handle:
        takes = browser_result.get("takes")
        if not isinstance(takes, list):
            raise RuntimeError("functional result has no take traces")
        for take in takes:
            if not isinstance(take, dict) or not isinstance(take.get("rows"), list):
                raise RuntimeError("functional take trace is malformed")
            for row in take["rows"]:
                if not isinstance(row, dict):
                    raise RuntimeError("functional take trace row is malformed")
                take_id = row.get("take_id")
                eval_index = int(_numeric(row.get("eval_index"), "eval_index"))
                alignment = frame_alignments.get((take_id, eval_index))
                enriched = dict(row)
                if isinstance(alignment, dict):
                    enriched["detector_semantic"] = alignment
                handle.write(
                    json.dumps(enriched, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                rows_written += 1

    overlay = _render_overlay_video(context, browser_result, overlay_path)
    report = {
        "schema": "mjtensu.recognition/functional-video/v2",
        "models": {
            "base_classifier": {
                "id": context.model.definition.get("id"),
                "training_result": context.model.training_result.get("id"),
                "training_corpus": classifier_runtime["training_corpus"],
                "runtime_spec": classifier_runtime["runtime_spec"],
                "export": export_info,
            },
            "detector": {
                "id": detector["id"],
                "runtime_spec": detector["runtimeSpec"],
                "sha256": detector["sha256"],
                "score_threshold": detector_score_threshold,
                "export": detector.get("export"),
            },
            "red_five_classifier": {
                "id": red_five["id"],
                "runtime_spec": red_five["runtimeSpec"],
                "sha256": red_five["sha256"],
            },
        },
        "corpus": context.corpus.definition.get("id"),
        "sample_interval_ms": 100,
        "evaluation_scope": (
            "Deterministic 10 Hz source-video functional evaluation using the production TypeScript "
            "recognition pipeline, production semantic grouping, and production three-consecutive "
            "RecognitionSemanticStabilizer, with an evaluation-only NanoDet score-threshold override. "
            "This stage is intentionally separate from physical-iPhone latency."
        ),
        "metrics": metrics,
        "aggregate": aggregate,
        "detector_semantic_diagnostics": diagnostic_report,
        "takes": [
            {key: value for key, value in take.items() if key != "rows"}
            for take in browser_result["takes"]
            if isinstance(take, dict)
        ],
        "trace_rows": rows_written,
        "overlay": overlay,
        "browser_environment": browser_result.get("environment"),
        "model_diagnostics": browser_result.get("model_diagnostics"),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    functional_take_summary_path = work_dir / "functional-take-summary.csv"
    report_takes = report.get("takes")
    if not isinstance(report_takes, list) or not all(isinstance(item, dict) for item in report_takes):
        raise RuntimeError("functional report takes are malformed")
    _write_functional_take_summary_csv(functional_take_summary_path, report_takes)

    detector_semantic_take_summary_path = work_dir / "detector-semantic-take-summary.csv"
    diagnostic_takes = diagnostic_report.get("takes")
    if not isinstance(diagnostic_takes, dict):
        raise RuntimeError("detector semantic per-take diagnostics are malformed")
    _write_detector_semantic_take_summary_csv(detector_semantic_take_summary_path, diagnostic_takes)

    return EvaluationCandidate(
        metrics=metrics,
        artifacts={
            "onnx_model": onnx_path,
            "functional_report": report_path,
            "functional_take_summary": functional_take_summary_path,
            "prediction_trace": trace_path,
            "overlay_video": overlay_path,
            "detector_semantic_diagnostics": diagnostic_path,
            "detector_semantic_take_summary": detector_semantic_take_summary_path,
        },
    )
