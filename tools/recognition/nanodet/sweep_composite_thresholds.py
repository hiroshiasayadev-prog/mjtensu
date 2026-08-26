from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from evaluate_composite_onnx import (
    Box,
    Detection,
    build_ground_truth_index,
    calculate_per_image_matches,
    load_coco,
    summarize_operating_point,
)


DEFAULT_THRESHOLDS = tuple(round(index * 0.05, 2) for index in range(1, 20))


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    prediction_count: int
    predictions_per_image: float
    true_positive_count: int
    false_positive_count: int
    false_positives_per_image: float
    false_negative_count: int
    precision: float
    recall: float
    f1: float
    images_with_no_errors: int
    perfect_image_rate: float
    images_with_false_positives: int
    images_with_false_negatives: int


@dataclass(frozen=True)
class ParsedArguments:
    repository_root: Path
    annotation_path: Path
    predictions_path: Path
    output_directory: Path
    thresholds: tuple[float, ...]
    iou_threshold: float
    recall_target: float


def parse_args() -> ParsedArguments:
    repository_root_default = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Sweep confidence thresholds over an existing NanoDet COCO prediction file. "
            "The ONNX model is not run again."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root_default)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--thresholds",
        default=",".join(f"{value:.2f}" for value in DEFAULT_THRESHOLDS),
        help=(
            "Comma-separated confidence thresholds. "
            "Default: 0.05 through 0.95 in 0.05 increments."
        ),
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU required to count a prediction as a true positive (default: 0.50).",
    )
    parser.add_argument(
        "--recall-target",
        type=float,
        default=0.98,
        help=(
            "Recall floor used to select the highest viable threshold "
            "from the sweep (default: 0.98)."
        ),
    )
    namespace = parser.parse_args()

    repository_root = namespace.repository_root.resolve()
    baseline_directory = (
        repository_root
        / ".local"
        / "recognition"
        / "composite_capture_baseline_eval"
    )
    dataset_root = (
        repository_root
        / ".local"
        / "recognition"
        / "composite_capture_test_dataset"
    )
    annotation_path = (
        namespace.annotations or dataset_root / "annotations" / "instances.json"
    ).resolve()
    predictions_path = (
        namespace.predictions or baseline_directory / "predictions.json"
    ).resolve()
    output_directory = (
        namespace.output_directory or baseline_directory
    ).resolve()

    thresholds = parse_thresholds(namespace.thresholds)
    arguments = ParsedArguments(
        repository_root=repository_root,
        annotation_path=annotation_path,
        predictions_path=predictions_path,
        output_directory=output_directory,
        thresholds=thresholds,
        iou_threshold=float(namespace.iou_threshold),
        recall_target=float(namespace.recall_target),
    )
    validate_arguments(arguments)
    return arguments


def parse_thresholds(value: str) -> tuple[float, ...]:
    parsed: list[float] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            threshold = float(part)
        except ValueError as error:
            raise ValueError(f"Invalid confidence threshold: {part!r}") from error
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Confidence threshold must be between zero and one: {threshold}"
            )
        parsed.append(threshold)
    if not parsed:
        raise ValueError("At least one confidence threshold is required")
    return tuple(sorted(set(parsed)))


def validate_arguments(arguments: ParsedArguments) -> None:
    if not arguments.annotation_path.is_file():
        raise FileNotFoundError(
            f"Composite COCO annotations do not exist: {arguments.annotation_path}"
        )
    if not arguments.predictions_path.is_file():
        raise FileNotFoundError(
            "Baseline predictions do not exist. Run evaluate_composite_onnx.py first: "
            f"{arguments.predictions_path}"
        )
    if not 0.0 <= arguments.iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be between zero and one")
    if not 0.0 <= arguments.recall_target <= 1.0:
        raise ValueError("Recall target must be between zero and one")


def main() -> int:
    arguments = parse_args()
    coco = load_coco(arguments.annotation_path)
    images = coco["images"]
    annotations = coco["annotations"]
    image_records_by_id = {int(image["id"]): image for image in images}
    ground_truths_by_image = build_ground_truth_index(annotations)
    detections_by_image = load_predictions(
        arguments.predictions_path,
        valid_image_ids=set(image_records_by_id),
    )

    results = [
        evaluate_threshold(
            threshold,
            image_records_by_id=image_records_by_id,
            ground_truths_by_image=ground_truths_by_image,
            detections_by_image=detections_by_image,
            iou_threshold=arguments.iou_threshold,
        )
        for threshold in arguments.thresholds
    ]

    best_f1 = max(
        results,
        key=lambda result: (
            result.f1,
            result.recall,
            result.precision,
            -result.predictions_per_image,
            result.threshold,
        ),
    )
    recall_eligible = [
        result for result in results if result.recall >= arguments.recall_target
    ]
    highest_threshold_meeting_recall_target = (
        max(recall_eligible, key=lambda result: result.threshold)
        if recall_eligible
        else None
    )

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = arguments.output_directory / "threshold_sweep.csv"
    json_path = arguments.output_directory / "threshold_sweep.json"
    write_csv(csv_path, results)

    report = {
        "annotations": str(arguments.annotation_path),
        "predictions": str(arguments.predictions_path),
        "image_count": len(images),
        "ground_truth_count": len(annotations),
        "candidate_prediction_count": sum(
            len(detections) for detections in detections_by_image.values()
        ),
        "iou_threshold": arguments.iou_threshold,
        "recall_target": arguments.recall_target,
        "thresholds": list(arguments.thresholds),
        "results": [asdict(result) for result in results],
        "best_f1": asdict(best_f1),
        "highest_threshold_meeting_recall_target": (
            asdict(highest_threshold_meeting_recall_target)
            if highest_threshold_meeting_recall_target is not None
            else None
        ),
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }
    write_json(json_path, report)

    print_table(results)
    console_summary = {
        "status": "completed",
        "images": len(images),
        "ground_truths": len(annotations),
        "iou_threshold": arguments.iou_threshold,
        "best_f1": asdict(best_f1),
        "recall_target": arguments.recall_target,
        "highest_threshold_meeting_recall_target": (
            asdict(highest_threshold_meeting_recall_target)
            if highest_threshold_meeting_recall_target is not None
            else None
        ),
        "csv": str(csv_path),
        "json": str(json_path),
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))
    return 0


def load_predictions(
    path: Path,
    *,
    valid_image_ids: set[int],
) -> dict[int, list[Detection]]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, list):
        raise ValueError(f"COCO predictions root must be a list: {path}")

    detections_by_image: defaultdict[int, list[Detection]] = defaultdict(list)
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"Prediction {index} must be an object")
        image_id = int(record["image_id"])
        if image_id not in valid_image_ids:
            raise ValueError(
                f"Prediction {index} refers to unknown image_id {image_id}"
            )
        score = float(record["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Prediction {index} has invalid score {score}")
        detections_by_image[image_id].append(
            Detection(box=Box.from_coco(record["bbox"]), score=score)
        )

    for detections in detections_by_image.values():
        detections.sort(key=lambda detection: detection.score, reverse=True)
    return dict(detections_by_image)


def evaluate_threshold(
    threshold: float,
    *,
    image_records_by_id: dict[int, dict[str, Any]],
    ground_truths_by_image: dict[int, list[Box]],
    detections_by_image: dict[int, list[Detection]],
    iou_threshold: float,
) -> ThresholdResult:
    selected_by_image = {
        image_id: [
            detection
            for detection in detections_by_image.get(image_id, [])
            if detection.score >= threshold
        ]
        for image_id in image_records_by_id
    }
    per_image_results = calculate_per_image_matches(
        image_records_by_id,
        ground_truths_by_image,
        selected_by_image,
        iou_threshold=iou_threshold,
    )
    summary = summarize_operating_point(per_image_results)

    true_positives = int(summary["true_positives"])
    false_positives = int(summary["false_positives"])
    false_negatives = int(summary["false_negatives"])
    precision = float(summary["precision"])
    recall = float(summary["recall"])
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    prediction_count = true_positives + false_positives
    image_count = len(image_records_by_id)
    images_with_no_errors = int(summary["images_with_no_errors"])

    return ThresholdResult(
        threshold=threshold,
        prediction_count=prediction_count,
        predictions_per_image=(prediction_count / image_count if image_count else 0.0),
        true_positive_count=true_positives,
        false_positive_count=false_positives,
        false_positives_per_image=(
            false_positives / image_count if image_count else 0.0
        ),
        false_negative_count=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        images_with_no_errors=images_with_no_errors,
        perfect_image_rate=(
            images_with_no_errors / image_count if image_count else 0.0
        ),
        images_with_false_positives=int(summary["images_with_false_positives"]),
        images_with_false_negatives=int(summary["images_with_false_negatives"]),
    )


def write_csv(path: Path, results: Iterable[ThresholdResult]) -> None:
    records = [asdict(result) for result in results]
    if not records:
        raise ValueError("No threshold results to write")
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")


def print_table(results: Iterable[ThresholdResult]) -> None:
    print(
        "\n"
        " threshold  pred/img   TP    FP   FN  precision  recall     F1  clean\n"
        " ---------  --------  ----  ----  ---  ---------  ------  -----  -----"
    )
    for result in results:
        print(
            f" {result.threshold:9.2f}"
            f"  {result.predictions_per_image:8.2f}"
            f"  {result.true_positive_count:4d}"
            f"  {result.false_positive_count:4d}"
            f"  {result.false_negative_count:3d}"
            f"  {result.precision:9.4f}"
            f"  {result.recall:6.4f}"
            f"  {result.f1:5.4f}"
            f"  {result.images_with_no_errors:5d}"
        )
    print()


if __name__ == "__main__":
    raise SystemExit(main())
