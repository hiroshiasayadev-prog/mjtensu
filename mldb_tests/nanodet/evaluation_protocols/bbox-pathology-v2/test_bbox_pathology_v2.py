from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable


ROOT = Path(__file__).resolve().parents[4]
MLDB_DATA = ROOT / "mldb_data"
IMPLEMENTATION = MLDB_DATA / "nanodet" / "evaluation_protocols" / "bbox-pathology-v2.py"


def _module():
    name = "nanodet_bbox_pathology_v2_test_target"
    spec = importlib.util.spec_from_file_location(name, IMPLEMENTATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluation_callable_contract() -> None:
    evaluate = _load_evaluation_callable(MLDB_DATA, "nanodet/bbox-pathology-v2")
    assert callable(evaluate)


def test_pathology_categories_are_distinct() -> None:
    module = _module()
    box = module.Box
    detection = module.Detection
    ground_truths = [
        box(0, 0, 10, 10),
        box(12, 0, 22, 10),
        box(30, 0, 40, 10),
    ]
    predictions = [
        detection(box(0, 0, 10, 10), 0.90),
        detection(box(1, 0, 9, 10), 0.80),
        detection(box(5, 0, 17, 10), 0.70),
        detection(box(60, 60, 70, 70), 0.95),
    ]

    result, detail = module._analyze(ground_truths, predictions, 0.25)

    assert result.duplicate_gt_count == 1
    assert result.duplicate_extra_prediction_count == 2
    assert result.multi_gt_prediction_count == 1
    assert result.spurious_prediction_count == 1
    assert result.missed_gt_count == 1
    assert result.tangled_component_count == 1
    assert result.maximum_gt_multiplicity == 3
    assert result.maximum_pred_gt_degree == 2
    assert detail["predictions"][2]["gt_indices"] == [0, 1]


def test_product_bridge_suppression_matches_expected_shape() -> None:
    module = _module()
    box = module.Box
    detection = module.Detection
    first = detection(box(10, 10, 20, 20), 0.80)
    second = detection(box(22, 10, 32, 20), 0.70)
    bridge = detection(box(9, 9, 33, 21), 0.95)

    kept = module._suppress_product_duplicates([first, second, bridge], 0.8)

    assert bridge not in kept
    assert first in kept
    assert second in kept
    assert len(kept) == 2


def test_raw_nanodet_logits_are_sigmoid_and_dfl_decoded() -> None:
    module = _module()
    # One stride-320 point, one class, reg_max=1. Class logit 0 -> score 0.5.
    output = torch.zeros((1, 1, 9), dtype=torch.float32)
    regression = output[0, 0, 1:].reshape(4, 2)
    regression[:, 0] = -20.0
    regression[:, 1] = 20.0

    decoded, raw_candidate_counts = module._decode_batch(
        output,
        strides=(320,),
        reg_max=1,
        score_threshold=0.35,
        nms_iou_threshold=0.6,
        max_detections=200,
    )

    assert raw_candidate_counts == [1]
    assert len(decoded) == 1
    assert len(decoded[0]) == 1
    candidate = decoded[0][0]
    assert abs(candidate.score - 0.5) < 1.0e-6
    assert candidate.box.x1 == 0.0
    assert candidate.box.y1 == 0.0
    assert candidate.box.x2 > 319.9
    assert candidate.box.y2 > 319.9


def test_explosion_summary_preserves_tail_severity() -> None:
    module = _module()
    summary = module._explosion_summary(
        ground_truth_counts=[10, 10, 10, 10, 10],
        prediction_counts=[10, 11, 12, 40, 200],
        raw_candidate_counts=[12, 20, 30, 500, 5000],
        max_detections=200,
    )

    assert summary["raw_candidate_count_p95"] == 5000.0
    assert summary["raw_candidate_count_max"] == 5000
    assert summary["bbox_excess_p95"] == 190.0
    assert summary["bbox_excess_max"] == 190
    assert summary["bbox_count_ratio_p95"] == 20.0
    assert summary["bbox_count_ratio_max"] == 20.0
    assert summary["max_detections_hit_count"] == 1
    assert summary["max_detections_hit_rate"] == 0.2
