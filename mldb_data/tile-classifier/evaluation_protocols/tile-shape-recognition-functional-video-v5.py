from __future__ import annotations

import base64
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
_BROWSER_BUNDLE_SHA256 = "141921e0e4a863831ab5fbda029d5db99f662c52d500aebdd35de22aa1a95952"
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


def _canonical_nanodet_config(loaded: object, destination: Path) -> dict[str, object]:
    definition = getattr(loaded, "definition", None)
    training_result = getattr(loaded, "training_result", None)
    architecture = getattr(loaded, "architecture", None)
    module = getattr(loaded, "module", None)
    if (
        not isinstance(definition, dict)
        or not isinstance(training_result, dict)
        or not isinstance(architecture, dict)
        or not isinstance(module, nn.Module)
    ):
        raise ValueError("detector_model must resolve to a canonical MLDB Model")
    expected_architecture = "nanodet/nanodet-plus-m320-ghostpan-baseline-v1"
    if architecture.get("id") != expected_architecture:
        raise ValueError(
            "functional-video-v5 requires detector architecture " + expected_architecture
        )

    source = module.eval().cpu()
    example = torch.zeros((1, 3, 320, 320), dtype=torch.float32)
    torch.onnx.export(
        source,
        example,
        str(destination),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["data"],
        output_names=["output"],
    )

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "numpy, onnx, and onnxruntime are required for canonical NanoDet export validation"
        ) from error

    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise RuntimeError("canonical NanoDet ONNX must have exactly one input and one output")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    parity_input = torch.rand((1, 3, 320, 320), generator=generator, dtype=torch.float32)
    with torch.no_grad():
        features = source.backbone(parity_input)
        features = source.fpn(features)
        expected = source.head._forward_onnx(features)
    if tuple(expected.shape) != (1, 2125, 33):
        raise RuntimeError(
            "canonical NanoDet export-path output shape mismatch: "
            + str(tuple(expected.shape))
        )

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()
    output_meta = session.get_outputs()
    if len(input_meta) != 1 or len(output_meta) != 1:
        raise RuntimeError("canonical NanoDet ONNX Runtime contract is malformed")
    actual = session.run(
        None,
        {input_meta[0].name: parity_input.numpy()},
    )[0]
    if tuple(actual.shape) != (1, 2125, 33):
        raise RuntimeError(
            "canonical NanoDet ONNX output shape mismatch: " + str(tuple(actual.shape))
        )
    max_abs_error = float(np.max(np.abs(actual - expected.numpy())))
    if not math.isfinite(max_abs_error) or max_abs_error > 5.0e-4:
        raise RuntimeError(
            "canonical NanoDet ONNX parity failed: max_abs_error="
            + repr(max_abs_error)
        )

    payload = destination.read_bytes()
    return {
        "id": definition.get("id"),
        "training_result": training_result.get("id"),
        "architecture": architecture.get("id"),
        "runtimeSpec": "nanodet-plus-m-320-v1",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "opset_version": 11,
        "torch_onnx_path_parity_max_abs_error": max_abs_error,
    }


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
    red_five = _runtime_model_config(
        context.models["red-five-classifier"], expected_role="red-five-classifier"
    )
    classifier_runtime = _classifier_runtime(context)

    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = work_dir / "base-classifier.onnx"
    detector_onnx_path = work_dir / "detector.onnx"
    report_path = work_dir / "recognition-functional-report.json"
    trace_path = work_dir / "prediction-trace.jsonl"
    overlay_path = work_dir / "recognition-overlay.mp4"
    export_info = _export_onnx(context.model.module, onnx_path)
    detector = _canonical_nanodet_config(
        context.models["detector"], detector_onnx_path
    )
    onnx_bytes = onnx_path.read_bytes()
    detector_onnx_bytes = detector_onnx_path.read_bytes()

    config = {
        "mode": "functional",
        "sampleIntervalMs": 100,
        "detectorScoreThreshold": detector_score_threshold,
        "baseClassifierBase64": base64.b64encode(onnx_bytes).decode("ascii"),
        "baseClassifierSha256": export_info["sha256"],
        "baseClassifierRuntimeSpec": classifier_runtime["runtime_spec"],
        "baseNormalization": classifier_runtime["normalization"],
        "detector": {
            "base64": base64.b64encode(detector_onnx_bytes).decode("ascii"),
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
    browser_result = _run_local_browser(config)
    metrics, aggregate = _aggregate_functional_result(browser_result)

    rows_written = 0
    with trace_path.open("w", encoding="utf-8") as handle:
        takes = browser_result.get("takes")
        if not isinstance(takes, list):
            raise RuntimeError("functional result has no take traces")
        for take in takes:
            if not isinstance(take, dict) or not isinstance(take.get("rows"), list):
                raise RuntimeError("functional take trace is malformed")
            for row in take["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows_written += 1

    overlay = _render_overlay_video(context, browser_result, overlay_path)
    report = {
        "schema": "mjtensu.recognition/functional-video/v1",
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
                "training_result": detector["training_result"],
                "architecture": detector["architecture"],
                "runtime_spec": detector["runtimeSpec"],
                "sha256": detector["sha256"],
                "score_threshold": detector_score_threshold,
                "export": {
                    "bytes": detector["bytes"],
                    "opset_version": detector["opset_version"],
                    "torch_onnx_path_parity_max_abs_error": detector[
                        "torch_onnx_path_parity_max_abs_error"
                    ],
                },
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
            "RecognitionSemanticStabilizer, with an explicit evaluation-only NanoDet score threshold. "
            "This stage is intentionally separate from physical-iPhone latency."
        ),
        "metrics": metrics,
        "aggregate": aggregate,
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
    return EvaluationCandidate(
        metrics=metrics,
        artifacts={
            "onnx_model": onnx_path,
            "detector_onnx": detector_onnx_path,
            "functional_report": report_path,
            "prediction_trace": trace_path,
            "overlay_video": overlay_path,
        },
    )
