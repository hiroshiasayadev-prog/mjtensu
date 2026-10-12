from __future__ import annotations

from mldb_data.nanodet.lib import nanodet_postprocess_soft_union_v1 as replay


def _d(x, width, score, ident):
    return {
        "id": ident,
        "detectionIndex": int(ident),
        "confidence": score,
        "sourceBox": {"x": float(x), "y": 0.0, "width": float(width), "height": 10.0},
        "classification": {"kind": "tile", "tile": {"kind": "6z", "red": False}},
    }


def test_union_area_does_not_double_count_shared_coverage():
    candidate = _d(0, 10, 0.60, "3")
    left = _d(0, 6, 0.90, "1")
    right = _d(4, 6, 0.80, "2")
    assert replay.union_coverage(candidate, [left, right]) == 1.0


def test_union_suppresses_only_when_every_intersector_has_higher_confidence():
    candidate = _d(0, 10, 0.60, "3")
    left = _d(0, 6, 0.90, "1")
    right = _d(4, 6, 0.80, "2")
    kept, decisions = replay.suppress_union_confidence([left, right, candidate])
    assert [d["id"] for d in kept] == ["1", "2"]
    assert decisions[0]["id"] == "3"
    assert decisions[0]["union_coverage"] == 1.0

    lower_neighbor = _d(4, 6, 0.50, "4")
    kept, decisions = replay.suppress_union_confidence([left, lower_neighbor, candidate])
    assert "3" in {d["id"] for d in kept}
    assert all(decision["id"] != "3" for decision in decisions)


def test_combined_condition_applies_soft_nms_before_union():
    strong = _d(0, 10, 0.95, "1")
    soft_duplicate = _d(1, 10, 0.40, "2")
    independent = _d(20, 10, 0.80, "3")
    kept, stats = replay.apply([strong, soft_duplicate, independent])
    assert [d["id"] for d in kept] == ["1", "3"]
    assert stats["soft_nms_removed"] == 1
    assert stats["union_removed"] == 0
