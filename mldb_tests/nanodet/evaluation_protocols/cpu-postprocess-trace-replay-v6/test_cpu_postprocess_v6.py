from __future__ import annotations

import importlib.util
from pathlib import Path

from mldb_data.nanodet.lib import nanodet_postprocess_trace_replay_v3 as replay


def _d(x, score, ident):
    return {
        "id": ident,
        "detectionIndex": int(ident),
        "confidence": score,
        "sourceBox": {"x": x, "y": 0., "width": 10., "height": 10.},
        "classification": {"kind": "tile", "tile": {"kind": "4p", "red": False}},
    }


def test_conditions_and_selectivity():
    assert len(replay.CASES) == 6
    strong, duplicate, neighbor = _d(0., .91, "1"), _d(2., .42, "2"), _d(12., .88, "3")
    ds = [strong, duplicate, neighbor]
    assert replay.iou(strong["sourceBox"], duplicate["sourceBox"]) > .5
    assert len(replay.suppress(ds, "coverage", .8)) == 2
    assert len(replay.suppress(ds, "coverage", .7)) == 2
    assert len(replay.suppress(ds, "nms", .5)) == 2
    assert len(replay.suppress(ds, "soft", .5)) == 2


def test_exact_sequence_alignment_and_f1():
    same = replay.align(["1m", "2m"], ["1m", "2m"])
    assert same["match"] == 2
    assert replay.f1_from_counts(same) == (1., 1., 1.)
    extra = replay.align(["1m"], ["1m", "1m"])
    assert extra["insertion"] == 1
    assert replay.f1_from_counts(extra)[2] < 1.


def test_suppression_keeps_different_neighbors():
    ds = [_d(0., .90, "1"), _d(11., .86, "2"), _d(22., .81, "3")]
    for _, mode, threshold in replay.CASES:
        assert len(replay.suppress(ds, mode, threshold)) == 3


def test_formal_evaluate_produces_required_video_artifact(monkeypatch, tmp_path):
    """MLDB EvaluationContext.definition is a dict, not an object with .id."""
    import json
    import importlib.util
    from types import SimpleNamespace

    entry = (Path(__file__).resolve().parents[4] / "mldb_data" / "nanodet"
             / "evaluation_protocols" / "cpu-postprocess-trace-replay-v6.py")
    spec = importlib.util.spec_from_file_location("cpu_replay_v5_test", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "prediction_trace").write_text("test-artifact", encoding="utf-8")
    def fake_run(trace, out):
        assert trace == corpus / "prediction_trace"
        out.mkdir(parents=True, exist_ok=True)
        values = []
        for region in ("completed-hand", "melds", "dora-indicators"):
            values.append({"condition": "baseline", "region": region, "exact_frame_rate": 0.75,
                           "semantic_f1_proxy": 0.91, "remaining_iou50_overlap_frames": 3,
                           "removed_boxes": 0, "exact_lost": 0})
        (out / "comparison.json").write_text(json.dumps({"frames": 1497, "conditions": values}))
        (out / "comparison.csv").write_text("test,one\n")
        (out / "exact_frame_changes.jsonl").write_text('{"test":1}\n')
    monkeypatch.setattr(module.replay, "run", fake_run)
    def fake_render(*, corpus_root, trace_path, condition, destination):
        assert condition == "baseline"
        assert trace_path == corpus / "prediction_trace"
        assert corpus_root == corpus
        destination.write_bytes(b"fake-test-video-bytes")
        return {"frames":1497,"fps":10,"width":1280,"height":720,"sha256":"0"*64,"bytes":len(b"fake-test-video-bytes"),"condition":condition}
    monkeypatch.setattr(module.overlay, "render_condition_video", fake_render)

    class Telemetry:
        def report_scalar(self, **kwargs):
            assert kwargs["group"] == "postprocess"
    context = SimpleNamespace(
        corpus=SimpleNamespace(root=corpus, definition={"id": "nanodet/nanodet-jp25-functional-trace-v1"}),
        model=SimpleNamespace(definition={"id": "nanodet/run-09b6de7dfb7e4add819a4eef0047bf81-trial-0002-model"}),
        work_dir=tmp_path / "work",
        parameters={"postprocess_condition": "baseline"},
        telemetry=Telemetry(),
    )
    result = module.evaluate(context)
    assert result.metrics["hand_exact_rate_proxy"] == 0.75
    assert result.metrics["hand_overlap_frames"] == 3
    assert set(result.artifacts) == {"selected_result", "comparison_csv", "exact_frame_changes", "overlay_video"}
    assert all(path.is_file() for path in result.artifacts.values())
    report = json.loads(result.artifacts["selected_result"].read_text())
    assert report["source_corpus"] == context.corpus.definition["id"]
    assert report["source_model"] == context.model.definition["id"]



def test_canonical_video_overlay_source_filters_and_region_labels():
    from mldb_data.nanodet.lib import nanodet_postprocess_overlay_v1 as overlay
    row = {"detections":[
        _d(0.,0.96,"1") | {"region":"completed-hand"},
        _d(1.,0.41,"2") | {"region":"completed-hand"},
        _d(20.,0.80,"3") | {"region":"melds"},
    ]}
    assert len(overlay._selected_detections(row,"baseline")) == 3
    filtered=overlay._selected_detections(row,"gaussian_soft_nms_sigma_0.50")
    assert len(filtered)==2
    assert set(x["region"] for x in filtered)=={"completed-hand","melds"}
