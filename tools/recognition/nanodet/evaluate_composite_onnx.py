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
class PathologyImageResult:
    image_id: int
    file_name: str
    ground_truth_count: int
    prediction_count: int
    duplicate_gt_count: int
    duplicate_extra_prediction_count: int
    multi_gt_prediction_count: int
    spurious_prediction_count: int
    missed_gt_count: int
    tangled_component_count: int
    maximum_gt_multiplicity: int
    maximum_pred_gt_degree: int
    count_delta: int
    duplicate_score: float
    multi_gt_score: float
    spurious_score: float

    @property
    def affected(self) -> bool:
        return bool(
            self.duplicate_gt_count
            or self.multi_gt_prediction_count
            or self.spurious_prediction_count
            or self.missed_gt_count
        )


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
    pathology_overlap_threshold: float
    duplicate_overlap_threshold: float
    pathology_worst_count: int
    native_composite_only: bool
    deduplicate_file_name: bool


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
        default=0.35,
        help="Product NanoDet score threshold used for pathology matching and overlays (default: 0.35).",
    )
    parser.add_argument("--nms-iou-threshold", type=float, default=NMS_IOU_THRESHOLD)
    parser.add_argument("--max-detections", type=int, default=MAX_DETECTIONS)
    parser.add_argument(
        "--overlay-limit",
        type=int,
        help="Only write this many legacy IoU-matching overlays. By default all images are written.",
    )
    parser.add_argument(
        "--pathology-overlap-threshold",
        type=float,
        default=0.25,
        help="Minimum intersection/min(area) used to connect a prediction and GT (default: 0.25).",
    )
    parser.add_argument(
        "--duplicate-overlap-threshold",
        type=float,
        default=0.8,
        help="Product-style duplicate suppression overlap threshold (default: 0.8).",
    )
    parser.add_argument(
        "--pathology-worst-count",
        type=int,
        default=16,
        help="Maximum images in each pathology contact sheet (default: 16).",
    )
    parser.add_argument(
        "--native-composite-only",
        action="store_true",
        help="Evaluate only COCO image records already stored as native 320x320 detector composites.",
    )
    parser.add_argument(
        "--deduplicate-file-name",
        action="store_true",
        help="Keep only the first COCO image record for each file_name and its annotations.",
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
        pathology_overlap_threshold=float(namespace.pathology_overlap_threshold),
        duplicate_overlap_threshold=float(namespace.duplicate_overlap_threshold),
        pathology_worst_count=int(namespace.pathology_worst_count),
        native_composite_only=bool(namespace.native_composite_only),
        deduplicate_file_name=bool(namespace.deduplicate_file_name),
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
        ("pathology overlap threshold", arguments.pathology_overlap_threshold),
        ("duplicate overlap threshold", arguments.duplicate_overlap_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be between zero and one: {value}")
    if arguments.candidate_threshold > arguments.operating_threshold:
        raise ValueError("candidate threshold must not exceed operating threshold")
    if arguments.max_detections <= 0:
        raise ValueError("max detections must be positive")
    if arguments.overlay_limit is not None and arguments.overlay_limit <= 0:
        raise ValueError("overlay limit must be positive")
    if arguments.pathology_worst_count <= 0:
        raise ValueError("pathology worst count must be positive")


def main() -> int:
    arguments = parse_args()
    coco = load_coco(arguments.annotation_path)
    images = coco["images"]
    annotations = coco["annotations"]
    if arguments.native_composite_only:
        images = [
            image
            for image in images
            if int(image.get("width", 0)) == INPUT_SIZE
            and int(image.get("height", 0)) == INPUT_SIZE
        ]
        selected_image_ids = {int(image["id"]) for image in images}
        annotations = [
            annotation
            for annotation in annotations
            if int(annotation["image_id"]) in selected_image_ids
        ]
        if not images:
            raise ValueError("native-composite-only selected no 320x320 images")
    if arguments.deduplicate_file_name:
        unique_images: list[dict[str, Any]] = []
        seen_file_names: set[str] = set()
        for image in images:
            file_name = str(image["file_name"])
            if file_name in seen_file_names:
                continue
            seen_file_names.add(file_name)
            unique_images.append(image)
        images = unique_images
        selected_image_ids = {int(image["id"]) for image in images}
        annotations = [
            annotation
            for annotation in annotations
            if int(annotation["image_id"]) in selected_image_ids
        ]
        if not images:
            raise ValueError("deduplicate-file-name selected no images")
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

    if arguments.native_composite_only or arguments.deduplicate_file_name:
        official_metrics = None
        official_error = (
            "official COCO evaluation skipped because dataset filtering "
            "changes the source annotation set"
        )
    else:
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
    product_detections_by_image = {
        image_id: suppress_product_duplicates(
            detections,
            arguments.duplicate_overlap_threshold,
        )
        for image_id, detections in operating_detections_by_image.items()
    }
    pathology_nms_results, pathology_nms_details = analyze_pathologies(
        image_records_by_id,
        ground_truths_by_image,
        operating_detections_by_image,
        overlap_threshold=arguments.pathology_overlap_threshold,
    )
    pathology_product_results, pathology_product_details = analyze_pathologies(
        image_records_by_id,
        ground_truths_by_image,
        product_detections_by_image,
        overlap_threshold=arguments.pathology_overlap_threshold,
    )
    pathology_nms_summary = summarize_pathologies(pathology_nms_results)
    pathology_product_summary = summarize_pathologies(pathology_product_results)

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

    pathology_directory = arguments.output_directory / "pathologies"
    pathology_directory.mkdir(parents=True, exist_ok=True)
    pathology_nms_path = pathology_directory / "after_nms.json"
    pathology_product_path = pathology_directory / "after_product_postprocess.json"
    pathology_summary_path = pathology_directory / "summary.json"
    write_json(pathology_nms_path, pathology_nms_details)
    write_json(pathology_product_path, pathology_product_details)
    write_json(
        pathology_summary_path,
        {
            "association_metric": "intersection / min(area(prediction), area(ground_truth))",
            "association_threshold": arguments.pathology_overlap_threshold,
            "operating_threshold": arguments.operating_threshold,
            "nms_iou_threshold": arguments.nms_iou_threshold,
            "product_duplicate_overlap_threshold": arguments.duplicate_overlap_threshold,
            "definitions": {
                "duplicate_gt": "GT connected to two or more predictions.",
                "multi_gt_prediction": "Prediction connected to two or more GT boxes.",
                "spurious_prediction": "Prediction connected to no GT box.",
                "missed_gt": "GT connected to no prediction under the pathology overlap threshold.",
                "tangled_component": "Connected component containing at least two GTs and two predictions.",
                "positive_count_delta_image": "Image where prediction count exceeds GT count.",
            },
            "overlay_legend": {
                "ground_truth": "green",
                "normal_prediction": "red",
                "duplicate_member_prediction": "cyan",
                "multi_gt_prediction": "orange",
                "spurious_prediction": "magenta",
            },
            "after_nms": pathology_nms_summary,
            "after_product_postprocess": pathology_product_summary,
        },
    )

    pathology_contact_sheets: dict[str, str] = {}
    for stage_name, stage_results, stage_detections in (
        ("after_nms", pathology_nms_results, operating_detections_by_image),
        ("after_product", pathology_product_results, product_detections_by_image),
    ):
        for category in ("duplicate", "multi_gt", "spurious", "miss", "overall"):
            selected = select_pathology_worst(
                stage_results,
                category,
                arguments.pathology_worst_count,
            )
            contact_sheet_path = (
                pathology_directory / f"{stage_name}-{category}-worst.png"
            )
            write_pathology_contact_sheet(
                contact_sheet_path,
                selected,
                image_records_by_id,
                arguments.image_root,
                ground_truths_by_image,
                stage_detections,
                overlap_threshold=arguments.pathology_overlap_threshold,
            )
            pathology_contact_sheets[f"{stage_name}_{category}"] = str(
                contact_sheet_path
            )

    report = {
        "model": str(arguments.model_path),
        "annotations": str(arguments.annotation_path),
        "image_root": str(arguments.image_root),
        "native_composite_only": arguments.native_composite_only,
        "deduplicate_file_name": arguments.deduplicate_file_name,
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
            "product_duplicate_overlap_threshold": arguments.duplicate_overlap_threshold,
        },
        "pathology": {
            "association_metric": "intersection / min(area(prediction), area(ground_truth))",
            "association_threshold": arguments.pathology_overlap_threshold,
            "after_nms": pathology_nms_summary,
            "after_product_postprocess": pathology_product_summary,
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
            "pathology_summary": str(pathology_summary_path),
            "pathology_after_nms": str(pathology_nms_path),
            "pathology_after_product_postprocess": str(pathology_product_path),
            "pathology_contact_sheets": pathology_contact_sheets,
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
        "duplicate_gt_rate_after_nms": pathology_nms_summary["duplicate_gt_rate"],
        "multi_gt_prediction_rate_after_nms": pathology_nms_summary["multi_gt_prediction_rate"],
        "spurious_prediction_rate_after_nms": pathology_nms_summary["spurious_prediction_rate"],
        "affected_image_rate_after_nms": pathology_nms_summary["affected_image_rate"],
        "duplicate_gt_rate_after_product": pathology_product_summary["duplicate_gt_rate"],
        "multi_gt_prediction_rate_after_product": pathology_product_summary["multi_gt_prediction_rate"],
        "spurious_prediction_rate_after_product": pathology_product_summary["spurious_prediction_rate"],
        "affected_image_rate_after_product": pathology_product_summary["affected_image_rate"],
        "report": str(report_path),
        "overlays": str(overlay_directory),
        "pathologies": str(pathology_directory),
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



FIXED_COMPOSITE_REGIONS: tuple[tuple[str, Box], ...] = (
    ("completed_hand", Box(7.0, 0.0, 313.0, 72.0)),
    ("dora_indicators", Box(7.0, 74.0, 313.0, 146.0)),
    ("melds", Box(74.0, 148.0, 246.0, 320.0)),
)


def box_area(box: Box) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def intersection_area(left: Box, right: Box) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return width * height


def overlap_over_smaller_box(left: Box, right: Box) -> float:
    smaller = min(box_area(left), box_area(right))
    return 0.0 if smaller <= 0.0 else intersection_area(left, right) / smaller


def pair_overlap_details(prediction: Box, ground_truth: Box) -> dict[str, float]:
    intersection = intersection_area(prediction, ground_truth)
    pred_area = box_area(prediction)
    gt_area = box_area(ground_truth)
    union = pred_area + gt_area - intersection
    return {
        "iou": 0.0 if union <= 0.0 else intersection / union,
        "gt_coverage": 0.0 if gt_area <= 0.0 else intersection / gt_area,
        "pred_coverage": 0.0 if pred_area <= 0.0 else intersection / pred_area,
        "overlap_over_smaller": (
            0.0 if min(pred_area, gt_area) <= 0.0 else intersection / min(pred_area, gt_area)
        ),
    }


def clip_box(left: Box, right: Box) -> Box | None:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1, y1, x2, y2)


def assign_composite_region(box: Box) -> tuple[str, Box] | None:
    center_x = 0.5 * (box.x1 + box.x2)
    center_y = 0.5 * (box.y1 + box.y2)
    for name, region in FIXED_COMPOSITE_REGIONS:
        if region.x1 <= center_x < region.x2 and region.y1 <= center_y < region.y2:
            clipped = clip_box(box, region)
            if clipped is not None:
                return name, clipped
    return None


def suppress_product_duplicates(
    detections: Sequence[Detection],
    overlap_threshold: float,
) -> list[Detection]:
    by_region: defaultdict[str, list[tuple[Detection, Box]]] = defaultdict(list)
    for detection in detections:
        assigned = assign_composite_region(detection.box)
        if assigned is not None:
            region, clipped = assigned
            by_region[region].append((detection, clipped))

    winners: list[Detection] = []
    for group in by_region.values():
        candidates: list[tuple[Detection, Box]] = []
        for candidate_index, candidate in enumerate(group):
            _candidate_detection, candidate_box = candidate
            candidate_area = box_area(candidate_box)
            smaller = [
                other
                for other_index, other in enumerate(group)
                if other_index != candidate_index
                and box_area(other[1]) < candidate_area
                and overlap_over_smaller_box(candidate_box, other[1]) >= overlap_threshold
            ]
            is_bridge = any(
                overlap_over_smaller_box(smaller[left][1], smaller[right][1])
                < overlap_threshold
                for left in range(len(smaller))
                for right in range(left + 1, len(smaller))
            )
            if not is_bridge:
                candidates.append(candidate)

        candidates.sort(key=lambda item: item[0].score, reverse=True)
        kept: list[tuple[Detection, Box]] = []
        for candidate in candidates:
            if any(
                overlap_over_smaller_box(candidate[1], accepted[1]) >= overlap_threshold
                for accepted in kept
            ):
                continue
            kept.append(candidate)
        winners.extend(detection for detection, _clipped in kept)
    return winners


def _bipartite_tangled_component_count(
    gt_neighbors: Sequence[Sequence[int]],
    pred_neighbors: Sequence[Sequence[int]],
) -> int:
    seen_gt: set[int] = set()
    seen_pred: set[int] = set()
    tangled = 0
    for start_gt, neighbors in enumerate(gt_neighbors):
        if start_gt in seen_gt or not neighbors:
            continue
        pending: list[tuple[str, int]] = [("gt", start_gt)]
        component_gt: set[int] = set()
        component_pred: set[int] = set()
        while pending:
            kind, index = pending.pop()
            if kind == "gt":
                if index in seen_gt:
                    continue
                seen_gt.add(index)
                component_gt.add(index)
                pending.extend(("pred", pred) for pred in gt_neighbors[index])
            else:
                if index in seen_pred:
                    continue
                seen_pred.add(index)
                component_pred.add(index)
                pending.extend(("gt", gt) for gt in pred_neighbors[index])
        if len(component_gt) >= 2 and len(component_pred) >= 2:
            tangled += 1
    return tangled


def analyze_pathologies(
    image_records_by_id: dict[int, dict[str, Any]],
    ground_truths_by_image: dict[int, list[Box]],
    detections_by_image: dict[int, list[Detection]],
    *,
    overlap_threshold: float,
) -> tuple[list[PathologyImageResult], list[dict[str, Any]]]:
    results: list[PathologyImageResult] = []
    details: list[dict[str, Any]] = []
    for image_id, image_record in image_records_by_id.items():
        ground_truths = ground_truths_by_image.get(image_id, [])
        detections = detections_by_image.get(image_id, [])
        pair_details = [
            [pair_overlap_details(detection.box, ground_truth) for ground_truth in ground_truths]
            for detection in detections
        ]
        pred_neighbors = [
            [
                gt_index
                for gt_index, values in enumerate(row)
                if values["overlap_over_smaller"] >= overlap_threshold
            ]
            for row in pair_details
        ]
        gt_neighbors = [
            [
                pred_index
                for pred_index, row in enumerate(pair_details)
                if row[gt_index]["overlap_over_smaller"] >= overlap_threshold
            ]
            for gt_index in range(len(ground_truths))
        ]

        duplicate_gt_count = sum(len(neighbors) >= 2 for neighbors in gt_neighbors)
        duplicate_extra = sum(max(0, len(neighbors) - 1) for neighbors in gt_neighbors)
        multi_gt_count = sum(len(neighbors) >= 2 for neighbors in pred_neighbors)
        spurious_count = sum(len(neighbors) == 0 for neighbors in pred_neighbors)
        missed_count = sum(len(neighbors) == 0 for neighbors in gt_neighbors)
        duplicate_score = sum(
            sorted(
                (
                    pair_details[pred_index][gt_index]["overlap_over_smaller"]
                    for pred_index in neighbors
                ),
                reverse=True,
            )[1]
            for gt_index, neighbors in enumerate(gt_neighbors)
            if len(neighbors) >= 2
        )
        multi_gt_score = sum(
            sorted(
                (pair_details[pred_index][gt]["overlap_over_smaller"] for gt in neighbors),
                reverse=True,
            )[1]
            for pred_index, neighbors in enumerate(pred_neighbors)
            if len(neighbors) >= 2
        )
        spurious_score = sum(
            detection.score
            for detection, neighbors in zip(detections, pred_neighbors, strict=True)
            if not neighbors
        )

        result = PathologyImageResult(
            image_id=image_id,
            file_name=str(image_record["file_name"]),
            ground_truth_count=len(ground_truths),
            prediction_count=len(detections),
            duplicate_gt_count=duplicate_gt_count,
            duplicate_extra_prediction_count=duplicate_extra,
            multi_gt_prediction_count=multi_gt_count,
            spurious_prediction_count=spurious_count,
            missed_gt_count=missed_count,
            tangled_component_count=_bipartite_tangled_component_count(
                gt_neighbors, pred_neighbors
            ),
            maximum_gt_multiplicity=max((len(row) for row in gt_neighbors), default=0),
            maximum_pred_gt_degree=max((len(row) for row in pred_neighbors), default=0),
            count_delta=len(detections) - len(ground_truths),
            duplicate_score=float(duplicate_score),
            multi_gt_score=float(multi_gt_score),
            spurious_score=float(spurious_score),
        )
        results.append(result)
        details.append(
            {
                **asdict(result),
                "affected": result.affected,
                "ground_truths": [
                    {
                        "gt_index": gt_index,
                        "box": ground_truth.to_coco(),
                        "prediction_indices": gt_neighbors[gt_index],
                    }
                    for gt_index, ground_truth in enumerate(ground_truths)
                ],
                "predictions": [
                    {
                        "prediction_index": pred_index,
                        "box": detection.box.to_coco(),
                        "score": detection.score,
                        "gt_indices": pred_neighbors[pred_index],
                        "overlaps": [
                            {"gt_index": gt_index, **values}
                            for gt_index, values in enumerate(pair_details[pred_index])
                            if values["overlap_over_smaller"] > 0.0
                        ],
                    }
                    for pred_index, detection in enumerate(detections)
                ],
            }
        )
    return results, details


def summarize_pathologies(
    results: Sequence[PathologyImageResult],
) -> dict[str, float | int]:
    image_count = len(results)
    gt_count = sum(result.ground_truth_count for result in results)
    prediction_count = sum(result.prediction_count for result in results)
    duplicate_gt_count = sum(result.duplicate_gt_count for result in results)
    duplicate_extra = sum(result.duplicate_extra_prediction_count for result in results)
    multi_gt = sum(result.multi_gt_prediction_count for result in results)
    spurious = sum(result.spurious_prediction_count for result in results)
    missed = sum(result.missed_gt_count for result in results)
    return {
        "image_count": image_count,
        "ground_truth_count": gt_count,
        "prediction_count": prediction_count,
        "duplicate_gt_count": duplicate_gt_count,
        "duplicate_gt_rate": duplicate_gt_count / gt_count if gt_count else 0.0,
        "duplicate_extra_prediction_count": duplicate_extra,
        "duplicate_extra_per_gt": duplicate_extra / gt_count if gt_count else 0.0,
        "multi_gt_prediction_count": multi_gt,
        "multi_gt_prediction_rate": multi_gt / prediction_count if prediction_count else 0.0,
        "spurious_prediction_count": spurious,
        "spurious_prediction_rate": spurious / prediction_count if prediction_count else 0.0,
        "missed_gt_count": missed,
        "missed_gt_rate": missed / gt_count if gt_count else 0.0,
        "affected_image_count": sum(result.affected for result in results),
        "affected_image_rate": (
            sum(result.affected for result in results) / image_count if image_count else 0.0
        ),
        "positive_count_delta_image_count": sum(result.count_delta > 0 for result in results),
        "positive_count_delta_image_rate": (
            sum(result.count_delta > 0 for result in results) / image_count
            if image_count
            else 0.0
        ),
        "maximum_gt_multiplicity": max(
            (result.maximum_gt_multiplicity for result in results), default=0
        ),
        "maximum_pred_gt_degree": max(
            (result.maximum_pred_gt_degree for result in results), default=0
        ),
        "tangled_component_count": sum(result.tangled_component_count for result in results),
    }


def pathology_sort_key(result: PathologyImageResult, category: str) -> tuple[float, ...]:
    if category == "duplicate":
        return (
            float(result.duplicate_extra_prediction_count),
            result.duplicate_score,
            float(result.maximum_gt_multiplicity),
        )
    if category == "multi_gt":
        return (
            float(result.multi_gt_prediction_count),
            result.multi_gt_score,
            float(result.maximum_pred_gt_degree),
        )
    if category == "spurious":
        return (float(result.spurious_prediction_count), result.spurious_score)
    if category == "miss":
        return (float(result.missed_gt_count),)
    return (
        float(
            result.duplicate_extra_prediction_count
            + result.multi_gt_prediction_count
            + result.spurious_prediction_count
            + result.missed_gt_count
        ),
        float(abs(result.count_delta)),
        result.duplicate_score + result.multi_gt_score + result.spurious_score,
    )


def select_pathology_worst(
    results: Sequence[PathologyImageResult],
    category: str,
    count: int,
) -> list[PathologyImageResult]:
    if category == "duplicate":
        candidates = [row for row in results if row.duplicate_gt_count]
    elif category == "multi_gt":
        candidates = [row for row in results if row.multi_gt_prediction_count]
    elif category == "spurious":
        candidates = [row for row in results if row.spurious_prediction_count]
    elif category == "miss":
        candidates = [row for row in results if row.missed_gt_count]
    else:
        candidates = [row for row in results if row.affected]
    return sorted(candidates, key=lambda row: pathology_sort_key(row, category), reverse=True)[
        :count
    ]


def write_pathology_overlay(
    image_path: Path,
    output_path: Path,
    ground_truths: Sequence[Box],
    detections: Sequence[Detection],
    *,
    overlap_threshold: float,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    pair_details = [
        [pair_overlap_details(detection.box, ground_truth) for ground_truth in ground_truths]
        for detection in detections
    ]
    pred_neighbors = [
        [
            gt_index
            for gt_index, values in enumerate(row)
            if values["overlap_over_smaller"] >= overlap_threshold
        ]
        for row in pair_details
    ]
    gt_neighbors = [
        [
            pred_index
            for pred_index, row in enumerate(pair_details)
            if row[gt_index]["overlap_over_smaller"] >= overlap_threshold
        ]
        for gt_index in range(len(ground_truths))
    ]
    for gt_index, ground_truth in enumerate(ground_truths):
        degree = len(gt_neighbors[gt_index])
        draw.rectangle(
            (ground_truth.x1, ground_truth.y1, ground_truth.x2, ground_truth.y2),
            outline=(0, 255, 0),
            width=2,
        )
        if degree != 1:
            draw.text(
                (ground_truth.x1 + 1, ground_truth.y1 + 1),
                f"G{gt_index} p={degree}",
                fill=(0, 255, 0),
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
    for pred_index, detection in enumerate(detections):
        degree = len(pred_neighbors[pred_index])
        duplicate_member = (
            degree == 1
            and len(gt_neighbors[pred_neighbors[pred_index][0]]) >= 2
        )
        if degree == 0:
            outline = (255, 0, 255)
        elif degree >= 2:
            outline = (255, 165, 0)
        elif duplicate_member:
            outline = (0, 255, 255)
        else:
            outline = (255, 0, 0)
        draw.rectangle(
            (detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2),
            outline=outline,
            width=2,
        )
        if degree != 1 or duplicate_member:
            draw.text(
                (detection.box.x1 + 1, detection.box.y2 - 11),
                f"P{pred_index} {detection.score:.2f} g={degree}",
                fill=outline,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    image.close()


def write_pathology_contact_sheet(
    output_path: Path,
    rows: Sequence[PathologyImageResult],
    image_records_by_id: dict[int, dict[str, Any]],
    image_root: Path,
    ground_truths_by_image: dict[int, list[Box]],
    detections_by_image: dict[int, list[Detection]],
    *,
    overlap_threshold: float,
) -> None:
    if not rows:
        image = Image.new("RGB", (640, 80), "white")
        ImageDraw.Draw(image).text((10, 10), "No matching pathology samples", fill="black")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        image.close()
        return
    columns = min(4, len(rows))
    cell_width, cell_height = 340, 385
    rows_count = math.ceil(len(rows) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows_count * cell_height), "white")
    for ordinal, result in enumerate(rows):
        temp_path = output_path.parent / f".tmp-{output_path.stem}-{ordinal}.png"
        image_record = image_records_by_id[result.image_id]
        write_pathology_overlay(
            resolve_image_path(image_root, image_record),
            temp_path,
            ground_truths_by_image.get(result.image_id, []),
            detections_by_image.get(result.image_id, []),
            overlap_threshold=overlap_threshold,
        )
        with Image.open(temp_path) as overlay:
            tile = overlay.convert("RGB").resize((320, 320), Image.Resampling.BILINEAR)
        temp_path.unlink(missing_ok=True)
        x = (ordinal % columns) * cell_width + 10
        y = (ordinal // columns) * cell_height + 54
        sheet.paste(tile, (x, y))
        draw = ImageDraw.Draw(sheet)
        file_name = Path(result.file_name).name
        if len(file_name) > 48:
            file_name = f"{file_name[:24]}...{file_name[-21:]}"
        draw.text((x, y - 50), file_name, fill="black")
        draw.text(
            (x, y - 36),
            (
                f"dup={result.duplicate_extra_prediction_count} "
                f"multi={result.multi_gt_prediction_count} "
                f"stray={result.spurious_prediction_count} miss={result.missed_gt_count}"
            ),
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    sheet.close()


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
