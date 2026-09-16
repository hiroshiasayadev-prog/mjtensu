"""Draft custom Evaluation Protocol example derived from PRODUCT-INV-RECOGNITION-010.

The real evaluation is product-facing: GT coverage = intersection(pred, gt)/area(gt), crop purity =
intersection(pred, gt)/area(pred), normalized center error, predicted/GT scale ratio, semantic-region
breakdown, and worst-case GT/prediction/actual-crop contact sheets.
"""
from __future__ import annotations


def gt_coverage(intersection_area: float, gt_area: float) -> float:
    return 0.0 if gt_area <= 0.0 else intersection_area / gt_area


def crop_purity(intersection_area: float, prediction_area: float) -> float:
    return 0.0 if prediction_area <= 0.0 else intersection_area / prediction_area


def normalized_center_error(pred_center: float, gt_center: float, gt_extent: float) -> float:
    return 0.0 if gt_extent <= 0.0 else abs(pred_center - gt_center) / gt_extent


def scale_ratio(pred_extent: float, gt_extent: float) -> float:
    return 0.0 if gt_extent <= 0.0 else pred_extent / gt_extent


def evaluate(context):
    raise NotImplementedError(
        "Draft v2 example: port INV-010 decode/matching and artifact publication once the v2 execution interface exists."
    )
