from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
PROTOCOL_ID = "tile-classifier/tile-shape-recognition-e2e-iphone-v1"
IMPLEMENTATION = ROOT / "tile-classifier" / "evaluation_protocols" / "tile-shape-recognition-e2e-iphone-v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("tile_shape_recognition_e2e_iphone_v1", IMPLEMENTATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_take(take_id: str, *, exact: int = 8, evaluations: int = 10, confirmed=True):
    timing = {
        "totalMs": 50.0,
        "candidateCount": 14,
        "redFiveCandidateCount": 1,
        "detectorPreprocessingMs": 2.0,
        "detectorInferenceMs": 10.0,
        "detectorPostprocessingMs": 3.0,
        "cropExtractionMs": 5.0,
        "baseClassifierPreprocessingMs": 4.0,
        "baseClassifierInferenceMs": 20.0,
        "redFiveClassifierPreprocessingMs": 2.0,
        "redFiveClassifierInferenceMs": 4.0,
    }
    return {
        "id": take_id,
        "source_video_duration_sec": 30.0,
        "ticks": 300,
        "skipped_in_flight": 200,
        "evaluations": evaluations,
        "eligible_frames": evaluations - 1,
        "exact_frames": exact,
        "completed_hand_exact_frames": 9,
        "dora_exact_frames": 8,
        "meld_exact_frames": 7,
        "first_confirmed_video_time_sec": 0.8 if confirmed else None,
        "confirmed_exact": True if confirmed else None,
        "timing": {"samples": [timing] * evaluations},
    }


def _runtime_loaded(role: str, runtime_spec: str, payload: bytes, *, normalization=None):
    sha = hashlib.sha256(payload).hexdigest()
    provenance = {} if normalization is None else {"normalization": normalization}
    return SimpleNamespace(
        definition={
            "id": f"recognition-runtime/{role}-v1",
            "role": role,
            "format": "onnx",
            "runtime_spec": runtime_spec,
            "artifact": {
                "uri": f"s3://mldb-runtime-models/{role}.onnx",
                "bytes": len(payload),
                "sha256": sha,
            },
            "provenance": provenance,
        },
        artifact=payload,
    )


def _make_corpus(tmp_path: Path):
    root = tmp_path / "corpus"
    root.mkdir()
    gt = {"takes": {}}
    regions = {
        "dora-indicators": {"x": 0.04, "y": 0.22, "width": 0.62, "height": 0.2593464052},
        "completed-hand": {"x": 0.04, "y": 0.5, "width": 0.62, "height": 0.2593464052},
        "melds": {"x": 0.72, "y": 0.2866666667, "width": 0.24, "height": 0.4266666667},
    }
    for take_id in ("t1", "t2", "t3", "t4", "t5"):
        take = root / take_id
        take.mkdir()
        (take / "video.mp4").write_bytes(b"video")
        (take / "capture.json").write_text(json.dumps({
            "logicalCapture": {"aspectRatio": "9:16", "rotation": -90},
            "recognitionRegions": regions,
        }), encoding="utf-8")
        gt["takes"][take_id] = {
            "completed_hand": [{"kind": "1m", "red": False}],
            "dora_indicators": [{"kind": "east", "red": False}],
            "melds": [],
        }
    (root / "ground-truth.json").write_text(json.dumps(gt), encoding="utf-8")
    return SimpleNamespace(
        definition={
            "id": "tile-classifier/recognition-e2e-night-iphone13-v1",
            "storage": {"root_uri": "s3://mldb-corpora/tile-classifier/recognition-e2e-night-iphone13-v1"},
        },
        root=root,
    )


class Telemetry:
    def __init__(self):
        self.events = []

    def report_scalar(self, **kwargs):
        self.events.append(kwargs)


def test_evaluation_callable_contract() -> None:
    assert callable(_load_evaluation_callable(ROOT, PROTOCOL_ID))


def test_browser_bundle_is_self_contained_except_pinned_ort_cdn() -> None:
    module = _load_module()
    bundle = module._browser_bundle()
    assert "onnxruntime-web@1.27.0" in bundle
    assert "100" in bundle
    assert "Recognition" in bundle or "recognition" in bundle
    assert "node_modules" not in bundle


def test_classifier_runtime_is_derived_from_training_corpus() -> None:
    module = _load_module()
    context = SimpleNamespace(model=SimpleNamespace(training_result={
        "corpus": "tile-classifier/gray35-jp500-seed42-v3-jp189-v1"
    }))
    runtime = module._classifier_runtime(context)
    assert runtime["runtime_spec"] == "gray64-tile-35-v1"
    assert runtime["normalization"]["mean"] == [0.6815832403977466]
    assert runtime["normalization"]["std"][0] == pytest.approx(0.2725553681973976)


def test_runtime_model_config_uses_verified_catalog_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_asset_base_url", lambda: "https://assets.test/mldb-assets")
    payload = b"detector"
    loaded = _runtime_loaded("detector", "nanodet-plus-m-320-v1", payload)
    config = module._runtime_model_config(loaded, expected_role="detector")
    assert config["runtimeSpec"] == "nanodet-plus-m-320-v1"
    assert config["sha256"] == hashlib.sha256(payload).hexdigest()
    assert str(config["url"]).startswith("https://assets.test/mldb-assets/mldb-runtime-models/")


def test_aggregate_browser_result() -> None:
    module = _load_module()
    result = {"ok": True, "takes": [_fake_take(take_id) for take_id in module._EXPECTED_TAKES]}
    metrics, summary = module._aggregate_browser_result(result)
    assert metrics["frame_semantic_exact_rate"] == pytest.approx(0.8)
    assert metrics["completed_hand_exact_rate"] == pytest.approx(0.9)
    assert metrics["dora_exact_rate"] == pytest.approx(0.8)
    assert metrics["meld_exact_rate"] == pytest.approx(0.7)
    assert metrics["take_confirmed_rate"] == 1.0
    assert metrics["take_confirmed_exact_rate"] == 1.0
    assert metrics["latency_p50_ms"] == 50.0
    assert metrics["latency_p95_ms"] == 50.0
    assert metrics["effective_eval_hz"] == pytest.approx(50 / 150)
    assert metrics["cadence_skip_rate"] == pytest.approx(1000 / 1500)
    assert summary["evaluations"] == 50


def test_evaluate_builds_five_take_job_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_asset_base_url", lambda: "https://assets.test/mldb-assets")
    monkeypatch.setattr(module, "_browser_runner_url", lambda: "http://runner.test")

    def fake_export(_model, path: Path):
        path.write_bytes(b"base-onnx")
        return {
            "mode": "test-export",
            "opset_version": 16,
            "bytes": 9,
            "sha256": hashlib.sha256(b"base-onnx").hexdigest(),
            "torch_parity_max_abs_error": 0.0,
        }

    captured = {}

    def fake_browser(document: str):
        captured["document"] = document
        return (
            {"ok": True, "takes": [_fake_take(take_id) for take_id in module._EXPECTED_TAKES], "environment": {}},
            {"ok": True, "udid": "device-secret"},
        )

    monkeypatch.setattr(module, "_export_onnx", fake_export)
    monkeypatch.setattr(module, "_run_browser_job", fake_browser)

    detector = _runtime_loaded("detector", "nanodet-plus-m-320-v1", b"detector")
    red = _runtime_loaded(
        "red-five-classifier",
        "c8-red-five-v1",
        b"red",
        normalization={
            "mean": [0.66, 0.69, 0.65],
            "std": [0.30, 0.25, 0.27],
        },
    )
    telemetry = Telemetry()
    context = SimpleNamespace(
        models={"detector": detector, "red-five-classifier": red},
        model=SimpleNamespace(
            module=object(),
            definition={"id": "tile-classifier/base-model"},
            training_result={
                "id": "tile-classifier/base-train",
                "corpus": "tile-classifier/gray35-jp500-seed42-v3-jp189-v1",
            },
        ),
        corpus=_make_corpus(tmp_path),
        work_dir=tmp_path / "work",
        telemetry=telemetry,
    )
    candidate = module.evaluate(context)
    assert len(candidate.metrics) == 12
    assert len(telemetry.events) == 12
    assert candidate.artifacts["onnx_model"].read_bytes() == b"base-onnx"
    report = json.loads(candidate.artifacts["e2e_report"].read_text(encoding="utf-8"))
    assert report["models"]["detector"]["runtime_spec"] == "nanodet-plus-m-320-v1"
    assert report["runner"]["device_id_sha256"] == hashlib.sha256(b"device-secret").hexdigest()
    assert "device-secret" not in json.dumps(report)
    assert captured["document"].count("video.mp4") == 5
    assert "baseClassifierBase64" in captured["document"]
