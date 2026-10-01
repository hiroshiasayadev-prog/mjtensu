from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
IMPLEMENTATION = ROOT / "tile-classifier" / "evaluation_protocols" / "tile-shape-recognition-functional-video-v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("recognition_functional_video_v1", IMPLEMENTATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _corpus(tmp_path: Path):
    root = tmp_path / "corpus"
    root.mkdir()
    gt = {"takes": {}}
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
        gt["takes"][take_id] = {
            "completed_hand": [{"kind": "1m", "red": False}],
            "dora_indicators": [{"kind": "1p", "red": False}],
            "melds": [],
        }
    (root / "ground-truth.json").write_text(json.dumps(gt))
    return SimpleNamespace(
        definition={"storage": {"root_uri": "s3://mldb-corpora/tile-classifier/e2e"}},
        root=root,
    )


def test_take_config_includes_gt_and_source_fps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "_asset_base_url", lambda: "https://assets.test")
    takes = module._load_take_configs(SimpleNamespace(corpus=_corpus(tmp_path)))
    assert len(takes) == 5
    assert all(take["sourceFps"] == 30.0 for take in takes)
    assert takes[0]["groundTruth"]["completed_hand"] == [{"kind": "1m", "red": False}]


def test_functional_aggregate_separates_gt_streak_and_product_confirmation() -> None:
    module = _module()
    takes = []
    for index, take_id in enumerate(module._EXPECTED_TAKES):
        takes.append({
            "id": take_id,
            "evaluations": 100,
            "eligible_frames": 80,
            "exact_frames": 60,
            "completed_hand_exact_frames": 70,
            "dora_exact_frames": 75,
            "meld_exact_frames": 90,
            "first_gt_streak3": {"source_frame": 12} if index < 4 else None,
            "first_product_confirm": {"exact": index < 3} if index < 4 else None,
        })
    metrics, summary = module._aggregate_functional_result({
        "ok": True,
        "mode": "functional",
        "takes": takes,
    })
    assert metrics["frame_semantic_exact_rate"] == pytest.approx(0.6)
    assert metrics["take_gt_streak3_rate"] == pytest.approx(0.8)
    assert metrics["take_product_confirmed_rate"] == pytest.approx(0.8)
    assert metrics["take_product_confirmed_exact_rate"] == pytest.approx(0.6)
    assert summary["evaluations"] == 500


def test_embedded_bundle_contains_trace_and_functional_mode() -> None:
    module = _module()
    bundle = module._browser_bundle()
    assert "functional" in bundle
    assert "gt_exact_consecutive" in bundle
    assert "source_frame" in bundle
