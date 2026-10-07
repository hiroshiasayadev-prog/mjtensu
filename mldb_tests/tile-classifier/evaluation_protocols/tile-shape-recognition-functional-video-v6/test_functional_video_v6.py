from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
IMPLEMENTATION = ROOT / "tile-classifier" / "evaluation_protocols" / "tile-shape-recognition-functional-video-v6.py"


def _module():
    spec = importlib.util.spec_from_file_location("recognition_functional_video_v6", IMPLEMENTATION)
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
            "trackSettings": {"frameRate": 30, "width": 720, "height": 1280},
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


def test_overlay_encoder_pads_odd_dimensions_for_h264() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    assert 'pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black' in source



def test_mp4_presentation_rotation_precedes_logical_capture_rotation() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    module = _module()

    raw = np.zeros((720, 1280, 3), dtype=np.uint8)
    raw[:, :640] = (10, 20, 30)
    raw[:, 640:] = (40, 50, 60)
    source = SimpleNamespace(
        get=lambda prop: 90.0 if prop == cv2.CAP_PROP_ORIENTATION_META else 0.0,
    )
    capture = {
        "trackSettings": {"width": 720, "height": 1280},
        "logicalCapture": {"aspectRatio": "9:16", "rotation": -90},
        "recognitionRegions": {},
    }

    presented = module._presentation_frame(raw, source, capture)
    assert presented.shape == (1280, 720, 3)
    canonical = module._canonical_frame(presented, capture)
    assert canonical.shape == (720, 1280, 3)
    assert np.array_equal(canonical, raw)


def test_recognition_regions_use_capture_metadata() -> None:
    module = _module()
    capture = {
        "recognitionRegions": {
            "dora-indicators": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
            "completed-hand": {"x": 0.2, "y": 0.5, "width": 0.5, "height": 0.2},
            "melds": {"x": 0.75, "y": 0.1, "width": 0.2, "height": 0.6},
        }
    }
    rects = dict(module._recognition_region_rects(capture, 1000, 500))
    assert rects["dora-indicators"] == (100, 100, 400, 300)
    assert rects["completed-hand"] == (200, 250, 700, 350)
    assert rects["melds"] == (750, 50, 950, 350)



def test_video_decoder_disables_implicit_orientation(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()

    class FakeCapture:
        def __init__(self) -> None:
            self.set_calls: list[tuple[int, int]] = []

        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: int) -> bool:
            self.set_calls.append((prop, value))
            return True

    capture = FakeCapture()
    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda _path: capture,
        CAP_PROP_ORIENTATION_AUTO=123,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    opened = module._open_raw_video(Path("capture.mp4"))
    assert opened is capture
    assert capture.set_calls == [(123, 0)]


def test_trace_overlay_draws_all_three_capture_regions() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    module = _module()
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    capture = {
        "recognitionRegions": {
            "dora-indicators": {"x": 0.1, "y": 0.2, "width": 0.25, "height": 0.15},
            "completed-hand": {"x": 0.1, "y": 0.5, "width": 0.5, "height": 0.15},
            "melds": {"x": 0.7, "y": 0.3, "width": 0.2, "height": 0.4},
        }
    }
    row = {"detections": [], "gt_exact": False, "gt_exact_consecutive": 0, "stabilization": {"kind": "collecting"}}
    rendered = module._draw_trace_overlay(frame, row, "t1", capture)
    for _name, (x0, y0, _x1, _y1) in module._recognition_region_rects(capture, 800, 400):
        assert rendered[y0, x0].any()


def test_ground_truth_honor_labels_are_canonicalized_to_product_tile_kinds() -> None:
    module = _module()
    actual = module._canonical_ground_truth({
        "completed_hand": [
            {"kind": "east", "red": False},
            {"kind": "south", "red": False},
            {"kind": "5m", "red": True},
        ],
        "dora_indicators": [
            {"kind": "white", "red": False},
            {"kind": "green", "red": False},
            {"kind": "red", "red": False},
        ],
        "melds": [{
            "kind": "pon",
            "tiles": [{"kind": "west", "red": False}] * 3,
        }],
    })
    assert [tile["kind"] for tile in actual["completed_hand"]] == ["1z", "2z", "5m"]
    assert [tile["kind"] for tile in actual["dora_indicators"]] == ["5z", "6z", "7z"]
    assert [tile["kind"] for tile in actual["melds"][0]["tiles"]] == ["3z", "3z", "3z"]
    assert actual["completed_hand"][2]["red"] is True


def test_embedded_bundle_contains_detector_threshold_override() -> None:
    module = _module()
    bundle = module._browser_bundle()
    assert "detectorScoreThreshold" in bundle
    assert "detectorScoreThreshold is only valid for NanoDet runtime" in bundle
    assert "inlineBase64" in bundle


def test_canonical_detector_model_exports_inline_onnx(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    module = _module()

    class FakeDetector(torch.nn.Module):
        def forward(self, images):
            return torch.zeros(
                (images.shape[0], 2125, 33),
                dtype=images.dtype,
                device=images.device,
            )

    loaded = SimpleNamespace(
        definition={"id": "nanodet/fake-model"},
        module=FakeDetector(),
        architecture={
            "id": "nanodet/nanodet-plus-m320-ghostpan-baseline-v1",
            "interface": {
                "output": {
                    "shape": ["N", 2125, 33],
                }
            },
        },
    )
    actual = module._detector_model_config(loaded, work_dir=tmp_path)
    assert actual["runtimeSpec"] == "nanodet-plus-m-320-v1"
    assert actual["url"] == "inline://mldb-aux-detector"
    payload = __import__("base64").b64decode(actual["inlineBase64"])
    assert __import__("hashlib").sha256(payload).hexdigest() == actual["sha256"]
    assert actual["export"]["output_shape"] == [1, 2125, 33]


def _diag_tile(kind: str, red: bool = False) -> dict[str, object]:
    return {"kind": kind, "red": red}


def _diag_observation(
    obs_id: str,
    kind: str,
    *,
    region: str,
    x: float,
    y: float,
    red: bool = False,
) -> dict[str, object]:
    return {
        "id": obs_id,
        "region": region,
        "bbox": {"x": x, "y": y, "width": 0.05, "height": 0.1},
        "classification": {
            "kind": "tile",
            "tile": _diag_tile(kind, red),
        },
    }


def test_ordered_alignment_penalizes_wrong_duplicate() -> None:
    module = _module()
    alignment = module._align_tile_sequences(["4p"], ["2p", "2p"])
    bucket = module._empty_detector_semantic_bucket()
    module._accumulate_detector_semantic_bucket(
        bucket,
        ground_truth=["4p"],
        predicted=["2p", "2p"],
        alignment=alignment,
        used_fallback=False,
        invalid_count=0,
    )
    result = module._finalize_detector_semantic_bucket(bucket)
    assert result["correct"] == 0
    assert result["substitutions"] == 1
    assert result["insertions"] == 1
    assert result["deletions"] == 0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["substitution_pairs"] == {"4p->2p": 1}
    assert result["inserted_predictions"] == {"2p": 1}


def test_meld_diagnostic_uses_actual_members_not_inferred_kan() -> None:
    module = _module()
    row = {
        "snapshot": {
            "observations": [
                _diag_observation("a", "1z", region="melds", x=0.75, y=0.5),
                _diag_observation("b", "1z", region="melds", x=0.80, y=0.5),
            ],
            "meldGroups": [{
                "memberObservationIds": ["a", "b"],
                "interpretation": {
                    "kind": "concealed-kan",
                    "tiles": [_diag_tile("1z")] * 4,
                },
            }],
            "draft": {
                "completedHand": [],
                "doraIndicators": [],
                "meldGroups": [],
            },
        },
    }
    predicted, used_fallback, invalid = module._predicted_region_tokens(
        row, "melds",
    )
    assert predicted == ["1z", "1z"]
    assert used_fallback is False
    assert invalid == 0


def test_meld_fallback_orders_rows_then_left_to_right() -> None:
    module = _module()
    row = {
        "snapshot": {
            "observations": [
                _diag_observation(
                    "bottom-right", "9s", region="melds", x=0.86, y=0.62,
                ),
                _diag_observation(
                    "top-right", "5z", region="melds", x=0.82, y=0.35,
                ),
                _diag_observation(
                    "bottom-left", "7s", region="melds", x=0.75, y=0.64,
                ),
                _diag_observation(
                    "top-left", "5z", region="melds", x=0.74, y=0.36,
                ),
            ],
            "meldGroups": [],
            "draft": {
                "completedHand": [],
                "doraIndicators": [],
                "meldGroups": [],
            },
        },
    }
    predicted, used_fallback, _invalid = module._predicted_region_tokens(
        row, "melds",
    )
    assert predicted == ["5z", "5z", "7s", "9s"]
    assert used_fallback is True


def test_detector_semantic_diagnostics_exposes_region_confusions() -> None:
    module = _module()
    take_configs = []
    takes = []
    for index, take_id in enumerate(module._EXPECTED_TAKES):
        gt = {
            "completed_hand": [_diag_tile("1m"), _diag_tile("2m")],
            "dora_indicators": [_diag_tile("3p")],
            "melds": [{
                "kind": "chi",
                "tiles": [
                    _diag_tile("4p"),
                    _diag_tile("5p"),
                    _diag_tile("6p"),
                ],
            }],
        }
        take_configs.append({"id": take_id, "groundTruth": gt})
        meld_kinds = (
            ["2p", "2p", "6p"]
            if index == 0
            else ["4p", "5p", "6p"]
        )
        observations = [
            _diag_observation(
                f"m{member}",
                kind,
                region="melds",
                x=0.75 + member * 0.05,
                y=0.5,
            )
            for member, kind in enumerate(meld_kinds)
        ]
        takes.append({
            "id": take_id,
            "rows": [{
                "eval_index": 0,
                "snapshot": {
                    "observations": observations,
                    "meldGroups": [{
                        "memberObservationIds": ["m0", "m1", "m2"],
                        "interpretation": {
                            "kind": "chi",
                            "tiles": [
                                _diag_tile("4p"),
                                _diag_tile("5p"),
                                _diag_tile("6p"),
                            ],
                        },
                    }],
                    "draft": {
                        "completedHand": [
                            _diag_tile("1m"),
                            _diag_tile("2m"),
                        ],
                        "doraIndicators": [_diag_tile("3p")],
                        "meldGroups": [],
                    },
                },
            }],
        })
    metrics, diagnostic = module._detector_semantic_diagnostics(
        {"takes": takes},
        take_configs,
    )
    assert metrics["completed_hand_detection_semantic_f1"] == 1.0
    assert metrics["dora_detection_semantic_f1"] == 1.0
    assert metrics["meld_detection_semantic_f1"] < 1.0
    meld = diagnostic["report"]["aggregate"]["melds"]
    assert meld["substitution_pairs"]["4p->2p"] == 1
    assert meld["substitution_pairs"]["5p->2p"] == 1
