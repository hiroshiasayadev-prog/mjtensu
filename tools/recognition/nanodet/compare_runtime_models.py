from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .evaluate_composite_onnx import (
        Box,
        Detection,
        decode_output,
        intersection_over_union,
        load_coco,
        preprocess_image,
        resolve_image_path,
    )
except ImportError:  # direct script execution
    from evaluate_composite_onnx import (
        Box,
        Detection,
        decode_output,
        intersection_over_union,
        load_coco,
        preprocess_image,
        resolve_image_path,
    )


CONFIDENCE_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.60
MAX_DETECTIONS = 200
DUPLICATE_OVERLAP_THRESHOLD = 0.80
MATCH_IOU_THRESHOLD = 0.50

REGIONS: dict[str, tuple[float, float, float, float]] = {
    "completed-hand": (7.0, 0.0, 306.0, 72.0),
    "dora-indicators": (7.0, 74.0, 306.0, 72.0),
    "melds": (74.0, 148.0, 172.0, 172.0),
}


@dataclass(frozen=True)
class RuntimeDetection:
    box: Box
    score: float
    region: str


@dataclass(frozen=True)
class DatasetResult:
    image_count: int
    ground_truth_count: int
    unassigned_ground_truth_count: int
    retained_prediction_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    clean_image_count: int
    merged_bridge_rejection_count: int
    pairwise_duplicate_rejection_count: int
    by_region: dict[str, dict[str, int | float]]

    @property
    def precision(self) -> float:
        denominator = self.true_positive_count + self.false_positive_count
        return 0.0 if denominator == 0 else self.true_positive_count / denominator

    @property
    def recall(self) -> float:
        denominator = self.true_positive_count + self.false_negative_count
        return 0.0 if denominator == 0 else self.true_positive_count / denominator

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 0.0 if denominator == 0.0 else 2.0 * self.precision * self.recall / denominator

    def to_json(self) -> dict[str, Any]:
        return {
            "image_count": self.image_count,
            "ground_truth_count": self.ground_truth_count,
            "unassigned_ground_truth_count": self.unassigned_ground_truth_count,
            "retained_prediction_count": self.retained_prediction_count,
            "true_positive_count": self.true_positive_count,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "clean_image_count": self.clean_image_count,
            "merged_bridge_rejection_count": self.merged_bridge_rejection_count,
            "pairwise_duplicate_rejection_count": self.pairwise_duplicate_rejection_count,
            "by_region": self.by_region,
        }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[3]
    composite_augmented_baseline_model = (
        repository_root
        / ".local"
        / "recognition"
        / "nanodet_runs"
        / "E1_plus_m_320_composite_augmented_amp40_seed42"
        / "model_best"
        / "nanodet-plus-m-320-composite-augmented.onnx"
    )
    fine_tune_model = (
        repository_root
        / ".local"
        / "recognition"
        / "nanodet_runs"
        / "E1_plus_m_320_real_capture_ft10_l10_seed42"
        / "model_best"
        / "nanodet-plus-m-320-real-capture-ft10-l10.onnx"
    )
    datasets = {
        "real_val": (
            repository_root
            / ".local"
            / "recognition"
            / "nanodet_capture_finetune_dataset"
            / "annotations"
            / "instances_real_val.json"
        ),
        "composite_val": (
            repository_root
            / ".local"
            / "recognition"
            / "nanodet_composite_augmented_dataset"
            / "annotations"
            / "instances_composite_val.json"
        ),
    }

    for path in (composite_augmented_baseline_model, fine_tune_model, *datasets.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required") from error

    models = {
        "composite_augmented_baseline": composite_augmented_baseline_model,
        "real_capture_fine_tune": fine_tune_model,
    }
    sessions: dict[str, Any] = {}
    for name, model_path in models.items():
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise RuntimeError(f"Expected one input/output for {name}: {model_path}")
        sessions[name] = session

    report: dict[str, Any] = {
        "configuration": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "maximum_detections": MAX_DETECTIONS,
            "duplicate_overlap_threshold": DUPLICATE_OVERLAP_THRESHOLD,
            "match_iou_threshold": MATCH_IOU_THRESHOLD,
            "duplicate_policy": "merged-bridge rejection then confidence-ordered greedy pairwise suppression",
        },
        "models": {name: str(path) for name, path in models.items()},
        "datasets": {},
    }

    for dataset_name, annotation_path in datasets.items():
        print(f"=== {dataset_name} ===")
        coco = load_coco(annotation_path)
        image_records = coco["images"]
        ground_truths = build_ground_truths(coco["annotations"])
        dataset_report: dict[str, Any] = {}

        for model_name, session in sessions.items():
            result = evaluate_dataset(
                session,
                image_records=image_records,
                ground_truths_by_image=ground_truths,
                image_root=repository_root,
            )
            dataset_report[model_name] = result.to_json()
            print_result(model_name, result)
        report["datasets"][dataset_name] = dataset_report
        print()

    output_path = (
        repository_root
        / ".local"
        / "recognition"
        / "runtime_detector_model_comparison.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {output_path}")
    return 0


def evaluate_dataset(
    session: Any,
    *,
    image_records: Sequence[dict[str, Any]],
    ground_truths_by_image: dict[int, list[RuntimeDetection]],
    image_root: Path,
) -> DatasetResult:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    totals = empty_counts()
    region_totals = {region: empty_counts() for region in REGIONS}
    clean_images = 0
    unassigned_ground_truths = 0
    bridge_rejections = 0
    pairwise_rejections = 0

    for ordinal, image_record in enumerate(image_records, start=1):
        image_id = int(image_record["id"])
        image_path = resolve_image_path(image_root, image_record)
        tensor, source_image = preprocess_image(image_path)
        try:
            raw_output = session.run([output_name], {input_name: tensor})[0]
        finally:
            source_image.close()
        output = np.ascontiguousarray(raw_output, dtype=np.float32)
        after_nms = decode_output(
            output,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
            max_detections=MAX_DETECTIONS,
        )
        assigned = [assign_region(detection) for detection in after_nms]
        assigned = [detection for detection in assigned if detection is not None]
        retained, image_bridge_rejections, image_pairwise_rejections = suppress_duplicates(assigned)
        bridge_rejections += image_bridge_rejections
        pairwise_rejections += image_pairwise_rejections

        ground_truths = ground_truths_by_image.get(image_id, [])
        unassigned_ground_truths += sum(gt.region == "outside" for gt in ground_truths)
        ground_truths = [gt for gt in ground_truths if gt.region != "outside"]

        image_tp = image_fp = image_fn = 0
        for region in REGIONS:
            predictions = [item for item in retained if item.region == region]
            truths = [item for item in ground_truths if item.region == region]
            tp, fp, fn = match_predictions(predictions, truths)
            image_tp += tp
            image_fp += fp
            image_fn += fn
            region_totals[region]["tp"] += tp
            region_totals[region]["fp"] += fp
            region_totals[region]["fn"] += fn
            region_totals[region]["predictions"] += len(predictions)
            region_totals[region]["ground_truths"] += len(truths)

        totals["tp"] += image_tp
        totals["fp"] += image_fp
        totals["fn"] += image_fn
        totals["predictions"] += len(retained)
        totals["ground_truths"] += len(ground_truths)
        if image_fp == 0 and image_fn == 0:
            clean_images += 1

        if ordinal % 25 == 0 or ordinal == len(image_records):
            print(f"  evaluated {ordinal}/{len(image_records)} images")

    by_region = {region: summarize_counts(counts) for region, counts in region_totals.items()}
    return DatasetResult(
        image_count=len(image_records),
        ground_truth_count=totals["ground_truths"],
        unassigned_ground_truth_count=unassigned_ground_truths,
        retained_prediction_count=totals["predictions"],
        true_positive_count=totals["tp"],
        false_positive_count=totals["fp"],
        false_negative_count=totals["fn"],
        clean_image_count=clean_images,
        merged_bridge_rejection_count=bridge_rejections,
        pairwise_duplicate_rejection_count=pairwise_rejections,
        by_region=by_region,
    )


def build_ground_truths(
    annotations: Sequence[dict[str, Any]],
) -> dict[int, list[RuntimeDetection]]:
    result: dict[int, list[RuntimeDetection]] = {}
    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        box = Box.from_coco(annotation["bbox"])
        region = region_for_box(box) or "outside"
        result.setdefault(int(annotation["image_id"]), []).append(
            RuntimeDetection(box=box, score=1.0, region=region)
        )
    return result


def assign_region(detection: Detection) -> RuntimeDetection | None:
    region = region_for_box(detection.box)
    if region is None:
        return None
    return RuntimeDetection(box=detection.box, score=detection.score, region=region)


def region_for_box(box: Box) -> str | None:
    center_x = (box.x1 + box.x2) / 2.0
    center_y = (box.y1 + box.y2) / 2.0
    for region, (x, y, width, height) in REGIONS.items():
        if x <= center_x < x + width and y <= center_y < y + height:
            return region
    return None


def suppress_duplicates(
    detections: Sequence[RuntimeDetection],
) -> tuple[list[RuntimeDetection], int, int]:
    by_region: dict[str, list[RuntimeDetection]] = {}
    for detection in detections:
        by_region.setdefault(detection.region, []).append(detection)

    retained: list[RuntimeDetection] = []
    bridge_rejections = 0
    pairwise_rejections = 0
    for group in by_region.values():
        candidates: list[RuntimeDetection] = []
        for candidate in group:
            if is_merged_bridge_candidate(candidate, group):
                bridge_rejections += 1
            else:
                candidates.append(candidate)
        candidates.sort(key=lambda item: item.score, reverse=True)

        kept: list[RuntimeDetection] = []
        for candidate in candidates:
            if any(
                overlap_over_smaller(candidate.box, winner.box) >= DUPLICATE_OVERLAP_THRESHOLD
                for winner in kept
            ):
                pairwise_rejections += 1
                continue
            kept.append(candidate)
        retained.extend(kept)
    return retained, bridge_rejections, pairwise_rejections


def is_merged_bridge_candidate(
    candidate: RuntimeDetection,
    group: Sequence[RuntimeDetection],
) -> bool:
    candidate_area = box_area(candidate.box)
    if candidate_area <= 0.0:
        return False
    smaller = [
        other
        for other in group
        if other is not candidate
        and box_area(other.box) < candidate_area
        and overlap_over_smaller(candidate.box, other.box) >= DUPLICATE_OVERLAP_THRESHOLD
    ]
    for left_index, left in enumerate(smaller):
        for right in smaller[left_index + 1 :]:
            if overlap_over_smaller(left.box, right.box) < DUPLICATE_OVERLAP_THRESHOLD:
                return True
    return False


def match_predictions(
    predictions: Sequence[RuntimeDetection],
    truths: Sequence[RuntimeDetection],
) -> tuple[int, int, int]:
    matched = [False] * len(truths)
    true_positives = 0
    false_positives = 0
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        best_index: int | None = None
        best_iou = 0.0
        for index, truth in enumerate(truths):
            if matched[index]:
                continue
            value = intersection_over_union(prediction.box, truth.box)
            if value > best_iou:
                best_iou = value
                best_index = index
        if best_index is not None and best_iou >= MATCH_IOU_THRESHOLD:
            matched[best_index] = True
            true_positives += 1
        else:
            false_positives += 1
    false_negatives = len(truths) - true_positives
    return true_positives, false_positives, false_negatives


def overlap_over_smaller(left: Box, right: Box) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    smaller = min(box_area(left), box_area(right))
    return 0.0 if smaller <= 0.0 else intersection / smaller


def box_area(box: Box) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "predictions": 0, "ground_truths": 0}


def summarize_counts(counts: dict[str, int]) -> dict[str, int | float]:
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    precision = 0.0 if precision_denominator == 0 else counts["tp"] / precision_denominator
    recall = 0.0 if recall_denominator == 0 else counts["tp"] / recall_denominator
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def print_result(name: str, result: DatasetResult) -> None:
    print(
        f"{name}: TP={result.true_positive_count} FP={result.false_positive_count} "
        f"FN={result.false_negative_count} precision={result.precision:.4f} "
        f"recall={result.recall:.4f} F1={result.f1:.4f} clean={result.clean_image_count}/{result.image_count} "
        f"bridge_removed={result.merged_bridge_rejection_count} "
        f"pairwise_removed={result.pairwise_duplicate_rejection_count}"
    )
    melds = result.by_region["melds"]
    print(
        f"  melds: TP={melds['tp']} FP={melds['fp']} FN={melds['fn']} "
        f"precision={melds['precision']:.4f} recall={melds['recall']:.4f} F1={melds['f1']:.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
