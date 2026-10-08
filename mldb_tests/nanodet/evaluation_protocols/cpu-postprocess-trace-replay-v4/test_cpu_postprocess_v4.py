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
