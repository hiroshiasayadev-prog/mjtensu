from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
IMPLEMENTATION = ROOT / "tile-classifier" / "evaluation_protocols" / "tile-shape-recognition-iphone-latency-v2.py"


def _module():
    spec = importlib.util.spec_from_file_location("recognition_iphone_latency_v2", IMPLEMENTATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _corpus(tmp_path: Path):
    root = tmp_path / "corpus"
    root.mkdir()
    regions = {
        "dora-indicators": {"x": 0.04, "y": 0.22, "width": 0.62, "height": 0.25},
        "completed-hand": {"x": 0.04, "y": 0.5, "width": 0.62, "height": 0.25},
        "melds": {"x": 0.72, "y": 0.28, "width": 0.24, "height": 0.42},
    }
    for take_id in ("t1", "t2", "t3", "t4", "t5"):
        take = root / take_id
        take.mkdir()
        (take / "video.mp4").write_bytes(b"video")
        (take / "capture.json").write_text(json.dumps({
            "trackSettings": {"frameRate": 30},
            "logicalCapture": {"aspectRatio": "9:16", "rotation": -90},
            "recognitionRegions": regions,
        }))
    return SimpleNamespace(
        definition={"storage": {"root_uri": "s3://mldb-corpora/tile-classifier/e2e"}},
        root=root,
    )


def test_take_config_is_latency_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "_asset_base_url", lambda: "https://assets.test")
    takes = module._load_take_configs(SimpleNamespace(corpus=_corpus(tmp_path)))
    assert len(takes) == 5
    assert all(take["sourceFps"] == 30.0 for take in takes)
    assert all("groundTruth" not in take for take in takes)


def test_latency_aggregate_reports_all_pipeline_components() -> None:
    module = _module()
    timing = []
    for total_ms, detector_ms, base_ms, red_ms, candidates, red_candidates in (
        (100.0, 30.0, 40.0, 10.0, 14, 1),
        (120.0, 35.0, 50.0, 15.0, 18, 2),
        (150.0, 40.0, 60.0, 25.0, 20, 4),
    ):
        timing.append({
            "totalMs": total_ms,
            "candidateCount": candidates,
            "redFiveCandidateCount": red_candidates,
            "detectorPreprocessingMs": 5.0,
            "detectorInferenceMs": detector_ms,
            "detectorPostprocessingMs": 1.0,
            "cropExtractionMs": 2.0,
            "baseClassifierPreprocessingMs": 3.0,
            "baseClassifierInferenceMs": base_ms,
            "redFiveClassifierPreprocessingMs": 1.0,
            "redFiveClassifierInferenceMs": red_ms,
        })
    takes = []
    for take_id in module._EXPECTED_TAKES:
        takes.append({
            "id": take_id,
            "source_video_duration_sec": 30.0,
            "ticks": 300,
            "skipped_in_flight": 20,
            "evaluations": 200,
            "target_evaluations": 300,
            "eval_gap_ms_p50": 120.0,
            "eval_gap_ms_p95": 220.0,
            "eval_gap_ms_max": 300.0,
            "timing": {"samples": timing},
        })
    metrics, summary = module._aggregate_browser_result({
        "ok": True,
        "mode": "iphone-latency",
        "takes": takes,
    })
    assert metrics["cadence_fulfillment_rate"] == pytest.approx(2 / 3)
    assert metrics["cadence_skip_rate"] == pytest.approx(1 / 15)
    assert metrics["effective_eval_hz"] == pytest.approx(1000 / 150)
    assert metrics["latency_p50_ms"] == 120.0
    assert metrics["latency_p95_ms"] == 150.0
    assert metrics["latency_max_ms"] == 150.0
    assert metrics["detector_inference_mean_ms"] == pytest.approx(35.0)
    assert metrics["base_classifier_inference_p95_ms"] == 60.0
    assert metrics["red_five_classifier_inference_max_ms"] == 25.0
    assert metrics["candidate_count_mean"] == pytest.approx(52 / 3)
    assert metrics["red_five_candidate_count_p95"] == 4.0
    assert metrics["pipeline_unattributed_overhead_mean_ms"] == pytest.approx(29 / 3)
    assert metrics["eval_gap_p95_ms"] == 220.0
    assert metrics["eval_gap_max_ms"] == 300.0
    assert summary["target_evaluations"] == 1500
    assert summary["ticks"] == 1500
    assert summary["skipped_in_flight"] == 100


def test_embedded_bundle_contains_split_modes() -> None:
    module = _module()
    bundle = module._browser_bundle()
    assert "iphone-latency" in bundle
    assert "functional" in bundle
    assert "loadedmetadata" in bundle
