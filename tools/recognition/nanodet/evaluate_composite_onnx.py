from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


INPUT_SIZE = 320
NUM_CLASSES = 1
REG_MAX = 7
OUTPUT_POINTS = 2125
OUTPUT_CHANNELS = NUM_CLASSES + 4 * (REG_MAX + 1)
STRIDES = (8, 16, 32, 64)
BGR_MEAN = np.asarray([103.53, 116.28, 123.675], dtype=np.float32)
BGR_STD = np.asarray([57.375, 57.12, 58.395], dtype=np.float32)
NMS_IOU_THRESHOLD = 0.6
MAX_DETECTIONS = 200


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_coco(cls, bbox: Sequence[Any]) -> "Box":
        if len(bbox) != 4:
            raise ValueError(f"COCO bbox must have four values: {bbox!r}")
        x, y, width, height = (float(value) for value in bbox)
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"COCO bbox must have positive size: {bbox!r}")
        return cls(x1=x, y1=y, x2=x + width, y2=y + height)

    def to_coco(self) -> list[float]:
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float


@dataclass(frozen=True)
class ImageMatchResult:
    image_id: int
    file_name: str
    ground_truth_count: int
    prediction_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    mean_matched_iou: float
    minimum_matched_iou: float

    @property
    def issue_count(self) -> int:
        return self.false_positive_count + self.false_negative_count


@dataclass(frozen=True)
class ParsedArguments:
    repository_root: Path
    model_path: Path
    annotation_path: Path
    image_root: Path
    output_directory: Path
    candidate_threshold: float
    operating_threshold: float
    nms_iou_threshold: float
    max_detections: int
    overlay_limit: int | None


def parse_args() -> ParsedArguments:
    repository_root_default = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current NanoDet ONNX model on the manually composed "
            "320x320 capture-layout COCO dataset."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root_default)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.001,
        help="Low score floor used when producing COCO predictions (default: 0.001).",
    )
    parser.add_argument(
        "--operating-threshold",
        type=float,
        default=0.05,
        help="PWA-like score threshold used for per-image matching and overlays (default: 0.05).",
    )
    parser.add_argument("--nms-iou-threshold", type=float, default=NMS_IOU_THRESHOLD)
    parser.add_argument("--max-detections", type=int, default=MAX_DETECTIONS)
    parser.add_argument(
        "--overlay-limit",
        type=int,
        help="Only write this many worst overlays. By default all images are written.",
    )
    namespace = parser.parse_args()

    repository_root = namespace.repository_root.resolve()
    dataset_root = (
        repository_root
        / ".local"
        / "recognition"
        / "composite_capture_test_dataset"
    )
    model_path = (
        namespace.model
        or repository_root
        / "tools"
        / "recognition"
        / "pwa_detector_probe"
        / "public"
        / "models"
        / "nanodet-plus-m-320.onnx"
    ).resolve()
    annotation_path = (
        namespace.annotations or dataset_root / "annotations" / "instances.json"
    ).resolve()
    image_root = (namespace.image_root or dataset_root).resolve()
    output_directory = (
        namespace.output_directory
        or repository_root
        / ".local"
        / "recognition"
        / "composite_capture_baseline_eval"
    ).resolve()

    arguments = ParsedArguments(
        repository_root=repository_root,
        model_path=model_path,
        annotation_path=annotation_path,
        image_root=image_root,
        output_directory=output_directory,
        candidate_threshold=float(namespace.candidate_threshold),
        operating_threshold=float(namespace.operating_threshold),
        nms_iou_threshold=float(namespace.nms_iou_threshold),
        max_detections=int(namespace.max_detections),
        overlay_limit=namespace.overlay_limit,
    )
    validate_arguments(arguments)
    return arguments


def validate_arguments(arguments: ParsedArguments) -> None:
    if not arguments.model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {arguments.model_path}")
    if not arguments.annotation_path.is_file():
        raise FileNotFoundError(
            f"Composite COCO annotations do not exist: {arguments.annotation_path}"
        )
    if not arguments.image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {arguments.image_root}")
    for label, value in (
        ("candidate threshold", arguments.candidate_threshold),
        ("operating threshold", arguments.operating_threshold),
        ("NMS IoU threshold", arguments.nms_iou_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be between zero and one: {value}")
    if arguments.candidate_threshold > arguments.operating_threshold:
        raise ValueError("candidate threshold must not exceed operating threshold")
    if arguments.max_detections <= 0:
        raise ValueError("max detections must be positive")
    if arguments.overlay_limit is not None and arguments.overlay_limit <= 0:
        raise ValueError("overlay limit must be positive")


def main() -> int:
    arguments = parse_args()
    coco = load_coco(arguments.annotation_path)
    images = coco["images"]
    annotations = coco["annotations"]
    ground_truths_by_image = build_ground_truth_index(annotations)

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required. Install it with: py -m pip install onnxruntime"
        ) from error

    session = ort.InferenceSession(
        str(arguments.model_path),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise AssertionError(
            f"Expected one model input and output, found {len(inputs)} and {len(outputs)}"
        )
    input_name = inputs[0].name
    output_name = outputs[0].name

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    overlay_directory = arguments.output_directory / "overlays"
    overlay_directory.mkdir(parents=True, exist_ok=True)

    detections_by_image: dict[int, list[Detection]] = {}
    inference_times_ms: list[float] = []
    preprocessing_times_ms: list[float] = []

    for ordinal, image_record in enumerate(images, start=1):
        image_id = int(image_record["id"])
        image_path = resolve_image_path(arguments.image_root, image_record)

        preprocess_started = time.perf_counter()
        tensor, source_image = preprocess_image(image_path)
        preprocessing_times_ms.append((time.perf_counter() - preprocess_started) * 1000.0)

        inference_started = time.perf_counter()
        raw_output = session.run([output_name], {input_name: tensor})[0]
        inference_times_ms.append((time.perf_counter() - inference_started) * 1000.0)

        output = np.ascontiguousarray(raw_output, dtype=np.float32)
        detections_by_image[image_id] = decode_output(
            output,
            confidence_threshold=arguments.candidate_threshold,
            nms_iou_threshold=arguments.nms_iou_threshold,
            max_detections=arguments.max_detections,
        )
        source_image.close()
        print(
            f"[{ordinal:03d}/{len(images):03d}] {image_record['file_name']} "
            f"detections={len(detections_by_image[image_id])}"
        )

    prediction_records = build_coco_predictions(detections_by_image)
    predictions_path = arguments.output_directory / "predictions.json"
    write_json(predictions_path, prediction_records)

    official_metrics, official_error = run_official_coco_evaluation(
        arguments.annotation_path,
        predictions_path,
        arguments.max_detections,
    )
    approximate_metrics = calculate_coco_style_metrics(
        ground_truths_by_image,
        detections_by_image,
        max_detections=arguments.max_detections,
    )

    image_records_by_id = {int(image["id"]): image for image in images}
    operating_detections_by_image = {
        image_id: [
            detection
            for detection in detections
            if detection.score >= arguments.operating_threshold
        ]
        for image_id, detections in detections_by_image.items()
    }
    per_image_results = calculate_per_image_matches(
        image_records_by_id,
        ground_truths_by_image,
        operating_detections_by_image,
        iou_threshold=0.5,
    )
    ordered_results = sorted(
        per_image_results,
        key=lambda result: (
            result.issue_count,
            result.false_negative_count,
            result.false_positive_count,
            -result.mean_matched_iou,
        ),
        reverse=True,
    )

    overlay_results = (
        ordered_results
        if arguments.overlay_limit is None
        else ordered_results[: arguments.overlay_limit]
    )
    for result in overlay_results:
        image_record = image_records_by_id[result.image_id]
        image_path = resolve_image_path(arguments.image_root, image_record)
        output_path = overlay_directory / Path(result.file_name).name
        write_overlay(
            image_path,
            output_path,
            ground_truths_by_image.get(result.image_id, []),
            operating_detections_by_image.get(result.image_id, []),
        )

    operating_summary = summarize_operating_point(per_image_results)
    failures_path = arguments.output_directory / "per_image_results.json"
    write_json(failures_path, [asdict(result) for result in ordered_results])

    report = {
        "model": str(arguments.model_path),
        "annotations": str(arguments.annotation_path),
        "image_root": str(arguments.image_root),
        "image_count": len(images),
        "ground_truth_count": len(annotations),
        "prediction_count": len(prediction_records),
        "runtime": {
            "onnxruntime_version": ort.__version__,
            "providers": session.get_providers(),
            "input_name": input_name,
            "output_name": output_name,
        },
        "preprocess": {
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "resize": "direct 320x320 resize; composite inputs are already 320x320",
            "channel_order": "BGR planar NCHW",
            "mean": BGR_MEAN.tolist(),
            "std": BGR_STD.tolist(),
        },
        "postprocess": {
            "candidate_threshold": arguments.candidate_threshold,
            "operating_threshold": arguments.operating_threshold,
            "nms_iou_threshold": arguments.nms_iou_threshold,
            "max_detections": arguments.max_detections,
        },
        "official_coco_metrics": official_metrics,
        "official_coco_error": official_error,
        "coco_style_metrics_fallback": approximate_metrics,
        "operating_point_iou_0_50": operating_summary,
        "timing_ms": {
            "preprocess_median": median(preprocessing_times_ms),
            "preprocess_p95": percentile(preprocessing_times_ms, 95.0),
            "inference_median": median(inference_times_ms),
            "inference_p95": percentile(inference_times_ms, 95.0),
        },
        "artifacts": {
            "predictions": str(predictions_path),
            "per_image_results": str(failures_path),
            "overlays": str(overlay_directory),
        },
        "worst_images": [asdict(result) for result in ordered_results[:20]],
    }
    report_path = arguments.output_directory / "report.json"
    write_json(report_path, report)

    selected_metrics = official_metrics or approximate_metrics
    console_summary = {
        "status": "completed",
        "images": len(images),
        "ground_truths": len(annotations),
        "predictions": len(prediction_records),
        "AP": selected_metrics.get("AP"),
        "AP50": selected_metrics.get("AP50"),
        "AP75": selected_metrics.get("AP75"),
        "operating_threshold": arguments.operating_threshold,
        "precision_at_iou_0_50": operating_summary["precision"],
        "recall_at_iou_0_50": operating_summary["recall"],
        "images_with_no_errors": operating_summary["images_with_no_errors"],
        "report": str(report_path),
        "overlays": str(overlay_directory),
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))
    return 0


def load_coco(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"COCO root must be an object: {path}")
    result: dict[str, list[dict[str, Any]]] = {}
    for field in ("images", "annotations", "categories"):
        value = payload.get(field)
        if not isinstance(value, list):
            raise ValueError(f"COCO field must be a list: {path}: {field}")
        result[field] = value
    return result


def resolve_image_path(image_root: Path, image_record: dict[str, Any]) -> Path:
    file_name = str(image_record["file_name"]).replace("\\", "/")
    relative = Path(*file_name.split("/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image file_name: {file_name}")
    path = image_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def preprocess_image(path: Path) -> tuple[np.ndarray, Image.Image]:
    source = Image.open(path).convert("RGB")
    if source.size != (INPUT_SIZE, INPUT_SIZE):
        resized = source.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    else:
        resized = source
    rgb = np.asarray(resized, dtype=np.float32)
    bgr = rgb[..., ::-1]
    normalized = (bgr - BGR_MEAN) / BGR_STD
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])
    if resized is not source:
        resized.close()
    return tensor, source


def build_ground_truth_index(
    annotations: Iterable[dict[str, Any]],
) -> dict[int, list[Box]]:
    result: defaultdict[int, list[Box]] = defaultdict(list)
    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        result[int(annotation["image_id"])].append(Box.from_coco(annotation["bbox"]))
    return dict(result)


def decode_output(
    output: np.ndarray,
    *,
    confidence_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> list[Detection]:
    if output.shape != (1, OUTPUT_POINTS, OUTPUT_CHANNELS):
        raise AssertionError(
            f"Unexpected ONNX output shape {output.shape}; expected "
            f"(1, {OUTPUT_POINTS}, {OUTPUT_CHANNELS})"
        )
    values = output[0]
    priors = build_center_priors()
    candidates: list[Detection] = []
    for point_index, row in enumerate(values):
        score = float(row[0])
        if score <= confidence_threshold:
            continue
        prior_x, prior_y, stride = priors[point_index]
        distances = distribution_expectation(row[NUM_CLASSES:]) * stride
        left, top, right, bottom = (float(value) for value in distances)
        box = Box(
            x1=clamp(prior_x - left, 0.0, float(INPUT_SIZE)),
            y1=clamp(prior_y - top, 0.0, float(INPUT_SIZE)),
            x2=clamp(prior_x + right, 0.0, float(INPUT_SIZE)),
            y2=clamp(prior_y + bottom, 0.0, float(INPUT_SIZE)),
        )
        if box.x2 > box.x1 and box.y2 > box.y1:
            candidates.append(Detection(box=box, score=score))
    return non_maximum_suppression(candidates, nms_iou_threshold, max_detections)


def build_center_priors() -> list[tuple[float, float, float]]:
    priors: list[tuple[float, float, float]] = []
    for stride in STRIDES:
        feature_size = math.ceil(INPUT_SIZE / stride)
        for row in range(feature_size):
            for column in range(feature_size):
                priors.append(
                    (float(column * stride), float(row * stride), float(stride))
                )
    if len(priors) != OUTPUT_POINTS:
        raise AssertionError(f"Generated {len(priors)} priors, expected {OUTPUT_POINTS}")
    return priors


def distribution_expectation(regression_logits: np.ndarray) -> np.ndarray:
    reshaped = regression_logits.reshape(4, REG_MAX + 1).astype(np.float64)
    shifted = reshaped - reshaped.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    bins = np.arange(REG_MAX + 1, dtype=np.float64)
    return probabilities @ bins


def non_maximum_suppression(
    detections: Sequence[Detection],
    iou_threshold: float,
    max_detections: int,
) -> list[Detection]:
    retained: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.score, reverse=True):
        if any(intersection_over_union(candidate.box, item.box) > iou_threshold for item in retained):
            continue
        retained.append(candidate)
        if len(retained) >= max_detections:
            break
    return retained


def build_coco_predictions(
    detections_by_image: dict[int, list[Detection]],
) -> list[dict[str, Any]]:
    return [
        {
            "image_id": image_id,
            "category_id": 1,
            "bbox": detection.box.to_coco(),
            "score": detection.score,
        }
        for image_id, detections in detections_by_image.items()
        for detection in detections
    ]


def run_official_coco_evaluation(
    annotation_path: Path,
    predictions_path: Path,
    max_detections: int,
) -> tuple[dict[str, float] | None, str | None]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None, "pycocotools is not installed; fallback metrics were calculated"

    try:
        ground_truth = COCO(str(annotation_path))
        detections = ground_truth.loadRes(str(predictions_path))
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.maxDets = [1, 10, max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = evaluator.stats
        return (
            {
                "AP": float(stats[0]),
                "AP50": float(stats[1]),
                "AP75": float(stats[2]),
                "AP_small": float(stats[3]),
                "AP_medium": float(stats[4]),
                "AP_large": float(stats[5]),
                "AR_max_1": float(stats[6]),
                "AR_max_10": float(stats[7]),
                f"AR_max_{max_detections}": float(stats[8]),
                "AR_small": float(stats[9]),
                "AR_medium": float(stats[10]),
                "AR_large": float(stats[11]),
            },
            None,
        )
    except Exception as error:
        return None, f"official COCO evaluation failed: {type(error).__name__}: {error}"


def calculate_coco_style_metrics(
    ground_truths_by_image: dict[int, list[Box]],
    detections_by_image: dict[int, list[Detection]],
    *,
    max_detections: int,
) -> dict[str, float]:
    thresholds = [0.50 + 0.05 * index for index in range(10)]
    average_precisions = [
        calculate_average_precision(
            ground_truths_by_image,
            detections_by_image,
            iou_threshold=threshold,
            max_detections=max_detections,
        )
        for threshold in thresholds
    ]
    recalls = [
        calculate_maximum_recall(
            ground_truths_by_image,
            detections_by_image,
            iou_threshold=threshold,
            max_detections=max_detections,
        )
        for threshold in thresholds
    ]
    return {
        "AP": float(statistics.fmean(average_precisions)),
        "AP50": float(average_precisions[0]),
        "AP75": float(average_precisions[5]),
        f"AR_max_{max_detections}": float(statistics.fmean(recalls)),
    }


def calculate_average_precision(
    ground_truths_by_image: dict[int, list[Box]],
    detections_by_image: dict[int, list[Detection]],
    *,
    iou_threshold: float,
    max_detections: int,
) -> float:
    total_ground_truths = sum(len(boxes) for boxes in ground_truths_by_image.values())
    if total_ground_truths == 0:
        return 0.0

    scored_matches: list[tuple[float, bool]] = []
    image_ids = set(ground_truths_by_image) | set(detections_by_image)
    for image_id in image_ids:
        ground_truths = ground_truths_by_image.get(image_id, [])
        matched = [False] * len(ground_truths)
        detections = sorted(
            detections_by_image.get(image_id, []),
            key=lambda item: item.score,
            reverse=True,
        )[:max_detections]
        for detection in detections:
            best_index, best_iou = best_unmatched_ground_truth(
                detection.box,
                ground_truths,
                matched,
            )
            true_positive = best_index is not None and best_iou >= iou_threshold
            if true_positive:
                matched[best_index] = True
            scored_matches.append((detection.score, true_positive))

    scored_matches.sort(key=lambda item: item[0], reverse=True)
    cumulative_true_positives = 0
    cumulative_false_positives = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for _score, true_positive in scored_matches:
        if true_positive:
            cumulative_true_positives += 1
        else:
            cumulative_false_positives += 1
        recalls.append(cumulative_true_positives / total_ground_truths)
        precisions.append(
            cumulative_true_positives
            / (cumulative_true_positives + cumulative_false_positives)
        )

    sampled_precisions = []
    for recall_target in np.linspace(0.0, 1.0, 101):
        candidates = [
            precision
            for recall, precision in zip(recalls, precisions)
            if recall >= recall_target
        ]
        sampled_precisions.append(max(candidates, default=0.0))
    return float(statistics.fmean(sampled_precisions))


def calculate_maximum_recall(
    ground_truths_by_image: dict[int, list[Box]],
    detections_by_image: dict[int, list[Detection]],
    *,
    iou_threshold: float,
    max_detections: int,
) -> float:
    total_ground_truths = sum(len(boxes) for boxes in ground_truths_by_image.values())
    if total_ground_truths == 0:
        return 0.0
    matched_count = 0
    for image_id, ground_truths in ground_truths_by_image.items():
        matched = [False] * len(ground_truths)
        detections = sorted(
            detections_by_image.get(image_id, []),
            key=lambda item: item.score,
            reverse=True,
        )[:max_detections]
        for detection in detections:
            best_index, best_iou = best_unmatched_ground_truth(
                detection.box,
                ground_truths,
                matched,
            )
            if best_index is not None and best_iou >= iou_threshold:
                matched[best_index] = True
                matched_count += 1
    return matched_count / total_ground_truths


def calculate_per_image_matches(
    image_records_by_id: dict[int, dict[str, Any]],
    ground_truths_by_image: dict[int, list[Box]],
    detections_by_image: dict[int, list[Detection]],
    *,
    iou_threshold: float,
) -> list[ImageMatchResult]:
    results: list[ImageMatchResult] = []
    for image_id, image_record in image_records_by_id.items():
        ground_truths = ground_truths_by_image.get(image_id, [])
        matched = [False] * len(ground_truths)
        matched_ious: list[float] = []
        false_positive_count = 0
        detections = sorted(
            detections_by_image.get(image_id, []),
            key=lambda item: item.score,
            reverse=True,
        )
        for detection in detections:
            best_index, best_iou = best_unmatched_ground_truth(
                detection.box,
                ground_truths,
                matched,
            )
            if best_index is not None and best_iou >= iou_threshold:
                matched[best_index] = True
                matched_ious.append(best_iou)
            else:
                false_positive_count += 1
        true_positive_count = len(matched_ious)
        false_negative_count = len(ground_truths) - true_positive_count
        results.append(
            ImageMatchResult(
                image_id=image_id,
                file_name=str(image_record["file_name"]),
                ground_truth_count=len(ground_truths),
                prediction_count=len(detections),
                true_positive_count=true_positive_count,
                false_positive_count=false_positive_count,
                false_negative_count=false_negative_count,
                mean_matched_iou=(
                    float(statistics.fmean(matched_ious)) if matched_ious else 0.0
                ),
                minimum_matched_iou=min(matched_ious, default=0.0),
            )
        )
    return results


def best_unmatched_ground_truth(
    prediction: Box,
    ground_truths: Sequence[Box],
    matched: Sequence[bool],
) -> tuple[int | None, float]:
    best_index: int | None = None
    best_iou = 0.0
    for index, ground_truth in enumerate(ground_truths):
        if matched[index]:
            continue
        iou = intersection_over_union(prediction, ground_truth)
        if iou > best_iou:
            best_index = index
            best_iou = iou
    return best_index, best_iou


def summarize_operating_point(
    results: Sequence[ImageMatchResult],
) -> dict[str, float | int]:
    true_positives = sum(result.true_positive_count for result in results)
    false_positives = sum(result.false_positive_count for result in results)
    false_negatives = sum(result.false_negative_count for result in results)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "images_with_no_errors": sum(result.issue_count == 0 for result in results),
        "images_with_false_positives": sum(
            result.false_positive_count > 0 for result in results
        ),
        "images_with_false_negatives": sum(
            result.false_negative_count > 0 for result in results
        ),
    }


def write_overlay(
    image_path: Path,
    output_path: Path,
    ground_truths: Sequence[Box],
    detections: Sequence[Detection],
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for ground_truth in ground_truths:
        draw.rectangle(
            (ground_truth.x1, ground_truth.y1, ground_truth.x2, ground_truth.y2),
            outline=(0, 255, 0),
            width=2,
        )
    for detection in detections:
        draw.rectangle(
            (detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2),
            outline=(255, 0, 0),
            width=2,
        )
        draw.text(
            (detection.box.x1 + 2, detection.box.y1 + 2),
            f"{detection.score:.2f}",
            fill=(255, 255, 0),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    image.close()


def intersection_over_union(left: Box, right: Box) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection_area = intersection_width * intersection_height
    if intersection_area <= 0.0:
        return 0.0
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union_area = left_area + right_area - intersection_area
    return 0.0 if union_area <= 0.0 else intersection_area / union_area


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def percentile(values: Sequence[float], percentile_value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value)) if values else 0.0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
