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
_BROWSER_BUNDLE_SHA256 = "cc9b79b29cb54939fc2061c3bbac5ba9ef21989dc8d2175a6f80f81ea2245598"
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
    detector = _runtime_model_config(context.models["detector"], expected_role="detector")
    red_five = _runtime_model_config(
        context.models["red-five-classifier"], expected_role="red-five-classifier"
    )
    classifier_runtime = _classifier_runtime(context)

    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = work_dir / "base-classifier.onnx"
    report_path = work_dir / "recognition-functional-report.json"
    trace_path = work_dir / "prediction-trace.jsonl"
    overlay_path = work_dir / "recognition-overlay.mp4"
    export_info = _export_onnx(context.model.module, onnx_path)
    onnx_bytes = onnx_path.read_bytes()

    config = {
        "mode": "functional",
        "sampleIntervalMs": 100,
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
        "sample_interval_ms": 100,
        "evaluation_scope": (
            "Deterministic 10 Hz source-video functional evaluation using the production TypeScript "
            "recognition pipeline, production semantic grouping, and production three-consecutive "
            "RecognitionSemanticStabilizer. This stage is intentionally separate from physical-iPhone latency."
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
            "functional_report": report_path,
            "prediction_trace": trace_path,
            "overlay_video": overlay_path,
        },
    )
