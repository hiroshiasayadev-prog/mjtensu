from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
IMPLEMENTATION = ROOT / "tile-classifier" / "evaluation_protocols" / "tile-shape-recognition-iphone-latency-v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("recognition_iphone_latency_v1", IMPLEMENTATION)
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


def test_latency_aggregate_reports_fulfillment_and_gaps() -> None:
    module = _module()
    timing = [{"totalMs": value} for value in (100.0, 120.0, 150.0)]
    takes = []
    for take_id in module._EXPECTED_TAKES:
        takes.append({
            "id": take_id,
            "source_video_duration_sec": 30.0,
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
    assert metrics["effective_eval_hz"] == pytest.approx(1000 / 150)
    assert metrics["latency_p50_ms"] == 120.0
    assert metrics["latency_p95_ms"] == 150.0
    assert metrics["eval_gap_p95_ms"] == 220.0
    assert metrics["eval_gap_max_ms"] == 300.0
    assert summary["target_evaluations"] == 1500


def test_embedded_bundle_contains_split_modes() -> None:
    module = _module()
    bundle = module._browser_bundle()
    assert "iphone-latency" in bundle
    assert "functional" in bundle
    assert "loadedmetadata" in bundle
