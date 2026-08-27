from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


INPUT_SIZE = 320
OUTPUT_POINTS = 2125
OUTPUT_CHANNELS = 33
REG_MAX = 7
STRIDES = (8, 16, 32, 64)
CONFIDENCE_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.6
MAX_DETECTIONS = 200
DUPLICATE_OVERLAP_THRESHOLD = 0.8
BGR_MEAN = np.asarray([103.53, 116.28, 123.675], dtype=np.float32)
BGR_STD = np.asarray([57.375, 57.12, 58.395], dtype=np.float32)

REGIONS = {
    "completed-hand": (7.0, 0.0, 306.0, 72.0),
    "dora-indicators": (7.0, 74.0, 306.0, 72.0),
    "melds": (74.0, 148.0, 172.0, 172.0),
}


@dataclass(frozen=True)
class Detection:
    point: int
    confidence: float
    x: float
    y: float
    width: float
    height: float
    region: str | None = None

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    def to_json(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "confidence": self.confidence,
            "region": self.region,
            "box": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an iPhone recognition debug capture with desktop ONNX inference."
    )
    parser.add_argument(
        "debug_directory",
        type=Path,
        help="Extracted mjtensu-recognition-debug-* directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    debug_dir = args.debug_directory.resolve()
    repo_root = Path(__file__).resolve().parents[1]

    summary_path = debug_dir / "summary.json"
    composite_path = debug_dir / "composite.png"
    input_path = debug_dir / "detector-input.f32"
    output_path = debug_dir / "detector-output.f32"
    for path in (summary_path, composite_path, input_path, output_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    capture = summary["capture"]
    browser_input = np.fromfile(input_path, dtype="<f4").reshape(1, 3, 320, 320)
    browser_output = np.fromfile(output_path, dtype="<f4").reshape(1, 2125, 33)
    recomputed_input = preprocess_composite(composite_path)

    input_comparison = compare_arrays(browser_input, recomputed_input)
    browser_detections = postprocess(browser_output)
    browser_postprocess_stages = inspect_postprocess_stages(browser_output)

    model_set_path = (
        repo_root
        / "product"
        / "frontend"
        / "src"
        / "recognition"
        / "model-runtime"
        / "production-model-set.json"
    )
    model_set = json.loads(model_set_path.read_text(encoding="utf-8"))
    current_model_set_version = str(model_set["modelSetVersion"])
    current_detector_artifact = str(model_set["models"]["detector"]["url"]).split("?", 1)[0]
    current_prod_model = repo_root / "vendor" / "recognition-models" / current_detector_artifact
    captured_baseline_model = (
        repo_root
        / ".local"
        / "recognition"
        / "nanodet_runs"
        / "E1_plus_m_320_composite_augmented_amp40_seed42"
        / "model_best"
        / "nanodet-plus-m-320-composite-augmented.onnx"
    )
    fine_model = (
        repo_root
        / ".local"
        / "recognition"
        / "nanodet_runs"
        / "E1_plus_m_320_real_capture_ft10_l10_seed42"
        / "model_best"
        / "nanodet-plus-m-320-real-capture-ft10-l10.onnx"
    )

    captured_baseline_result = run_model(captured_baseline_model, browser_input)
    captured_baseline_comparison = compare_arrays(
        browser_output, captured_baseline_result["raw"]
    )
    captured_baseline_detections = postprocess(captured_baseline_result["raw"])

    current_prod_result = run_model(current_prod_model, browser_input)
    current_prod_detections = postprocess(current_prod_result["raw"])
    current_prod_regression = captured_meld_localization_regression(current_prod_detections)

    fine_result: dict[str, Any] | None = None
    fine_detections: list[Detection] | None = None
    if fine_model.is_file():
        fine_result = run_model(fine_model, browser_input)
        fine_detections = postprocess(fine_result["raw"])

    captured_summary_detections = capture.get("detections", [])
    captured_melds = [d for d in captured_summary_detections if d.get("region") == "melds"]

    report: dict[str, Any] = {
        "debug_directory": str(debug_dir),
        "model_set_version": capture.get("modelSetVersion"),
        "detector_provider_on_iphone": next(
            (
                model.get("selectedProvider")
                for model in summary.get("runtimeDiagnostics", {}).get("models", [])
                if model.get("role") == "detector"
            ),
            None,
        ),
        "input_png_vs_captured_tensor": input_comparison,
        "browser_raw_vs_captured_baseline_raw": captured_baseline_comparison,
        "current_model_set_version": current_model_set_version,
        "models": {
            "captured_baseline": str(captured_baseline_model),
            "current_production": str(current_prod_model),
            "fine_tune": str(fine_model) if fine_model.is_file() else None,
        },
        "current_production_captured_meld_regression": current_prod_regression,
        "detections": {
            "captured_summary": captured_summary_detections,
            "browser_raw_redecoded": [d.to_json() for d in browser_detections],
            "browser_postprocess_stages": browser_postprocess_stages,
            "desktop_captured_baseline": [d.to_json() for d in captured_baseline_detections],
            "desktop_current_production": [d.to_json() for d in current_prod_detections],
            "desktop_fine_tune": (
                [d.to_json() for d in fine_detections]
                if fine_detections is not None
                else None
            ),
        },
    }
    comparison_path = debug_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== capture identity ===")
    print(f"model set: {report['model_set_version']}")
    print(f"iPhone detector provider: {report['detector_provider_on_iphone']}")
    print()

    print("=== composite PNG -> detector input verification ===")
    print_comparison(input_comparison)
    print()

    print("=== captured iPhone WASM raw output vs desktop CPU captured-baseline ONNX ===")
    print_comparison(captured_baseline_comparison)
    print()

    print("=== captured final meld detections ===")
    print_summary_melds(captured_melds)
    print()

    print("=== browser raw re-decoded at production thresholds ===")
    print_detection_summary(browser_detections)
    print()

    print("=== browser meld postprocess stages ===")
    print_postprocess_stages(browser_postprocess_stages)
    print()

    print("=== desktop captured-baseline ONNX on exact captured tensor ===")
    print_detection_summary(captured_baseline_detections)
    print()

    print(
        f"=== current production ONNX ({current_model_set_version}) on exact captured tensor ==="
    )
    print_detection_summary(current_prod_detections)
    print(
        "captured meld localization regression: "
        f"{'PASS' if current_prod_regression['pass'] else 'FAIL'} "
        f"melds={current_prod_regression['retained_meld_count']} "
        f"oversized={current_prod_regression['oversized_meld_points']}"
    )
    print()

    print("=== desktop real-capture fine-tune on exact captured tensor ===")
    if fine_detections is None:
        print(f"fine-tune model not found: {fine_model}")
    else:
        print_detection_summary(fine_detections)
    print()
    print(f"wrote: {comparison_path}")
    return 0


def preprocess_composite(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
    if rgb.shape != (320, 320, 3):
        raise AssertionError(f"unexpected composite shape: {rgb.shape}")
    bgr = rgb[..., ::-1]
    normalized = (bgr - BGR_MEAN) / BGR_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])


def run_model(model_path: Path, tensor: np.ndarray) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required in this venv: python -m pip install onnxruntime"
        ) from error

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise AssertionError(
            f"expected one ONNX input/output, got {len(inputs)}/{len(outputs)}"
        )
    raw = session.run([outputs[0].name], {inputs[0].name: tensor})[0]
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    if raw.shape != (1, OUTPUT_POINTS, OUTPUT_CHANNELS):
        raise AssertionError(f"unexpected ONNX output shape: {raw.shape}")
    return {"raw": raw, "providers": session.get_providers()}


def compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"same_shape": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "same_shape": True,
        "shape": list(left.shape),
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "fraction_abs_le_1e-6": float(np.mean(delta <= 1e-6)),
        "fraction_abs_le_1e-5": float(np.mean(delta <= 1e-5)),
        "fraction_abs_le_1e-4": float(np.mean(delta <= 1e-4)),
        "fraction_abs_le_1e-3": float(np.mean(delta <= 1e-3)),
    }


def captured_meld_localization_regression(
    detections: Sequence[Detection],
) -> dict[str, Any]:
    melds = [detection for detection in detections if detection.region == "melds"]
    oversized = [
        detection
        for detection in melds
        if detection.width > 60.0 or detection.height > 60.0
    ]
    return {
        "minimum_expected_meld_candidates": 6,
        "maximum_expected_box_extent": 60.0,
        "retained_meld_count": len(melds),
        "oversized_meld_points": [detection.point for detection in oversized],
        "pass": len(melds) >= 6 and len(oversized) == 0,
    }


def print_comparison(comparison: dict[str, Any]) -> None:
    for key, value in comparison.items():
        print(f"{key}: {value}")


def postprocess(output: np.ndarray) -> list[Detection]:
    raw = decode(output)
    nms = non_maximum_suppression(raw, NMS_IOU_THRESHOLD, MAX_DETECTIONS)
    assigned = [assign_region(d) for d in nms]
    assigned = [d for d in assigned if d.region is not None]
    return suppress_duplicates(assigned, DUPLICATE_OVERLAP_THRESHOLD)


def inspect_postprocess_stages(output: np.ndarray) -> dict[str, Any]:
    decoded = decode(output)
    after_nms = non_maximum_suppression(decoded, NMS_IOU_THRESHOLD, MAX_DETECTIONS)
    assigned = [assign_region(d) for d in after_nms]
    assigned = [d for d in assigned if d.region is not None]
    melds = [d for d in assigned if d.region == "melds"]
    merged_bridges = [
        detection
        for detection in melds
        if is_merged_bridge_candidate(detection, melds, DUPLICATE_OVERLAP_THRESHOLD)
    ]
    bridge_points = {detection.point for detection in merged_bridges}
    after_bridge_rejection = [detection for detection in melds if detection.point not in bridge_points]
    final = suppress_duplicates(assigned, DUPLICATE_OVERLAP_THRESHOLD)
    final_melds = [d for d in final if d.region == "melds"]

    return {
        "decoded_count": len(decoded),
        "after_nms_count": len(after_nms),
        "melds_after_nms": [d.to_json() for d in melds],
        "merged_bridge_rejections": [d.to_json() for d in merged_bridges],
        "melds_after_merged_bridge_rejection": [
            d.to_json() for d in after_bridge_rejection
        ],
        "melds_after_duplicate_suppression": [d.to_json() for d in final_melds],
    }


def decode(output: np.ndarray) -> list[Detection]:
    if output.shape != (1, OUTPUT_POINTS, OUTPUT_CHANNELS):
        raise AssertionError(f"unexpected detector shape: {output.shape}")
    priors = build_center_priors()
    detections: list[Detection] = []
    for point, row in enumerate(output[0]):
        confidence = float(row[0])
        if not math.isfinite(confidence) or confidence <= CONFIDENCE_THRESHOLD:
            continue
        prior_x, prior_y, stride = priors[point]
        distances = distribution_expectation(row[1:]) * stride
        left, top, right, bottom = (float(v) for v in distances)
        x1 = clamp(prior_x - left, 0.0, 320.0)
        y1 = clamp(prior_y - top, 0.0, 320.0)
        x2 = clamp(prior_x + right, 0.0, 320.0)
        y2 = clamp(prior_y + bottom, 0.0, 320.0)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                point=point,
                confidence=confidence,
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1,
            )
        )
    return detections


def build_center_priors() -> list[tuple[float, float, float]]:
    priors: list[tuple[float, float, float]] = []
    for stride in STRIDES:
        feature_size = math.ceil(INPUT_SIZE / stride)
        for row in range(feature_size):
            for column in range(feature_size):
                priors.append((float(column * stride), float(row * stride), float(stride)))
    if len(priors) != OUTPUT_POINTS:
        raise AssertionError(f"generated {len(priors)} priors")
    return priors


def distribution_expectation(values: np.ndarray) -> np.ndarray:
    logits = values.reshape(4, REG_MAX + 1).astype(np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ np.arange(REG_MAX + 1, dtype=np.float64)


def non_maximum_suppression(
    detections: Sequence[Detection], threshold: float, maximum: int
) -> list[Detection]:
    retained: list[Detection] = []
    for candidate in sorted(detections, key=lambda d: (-d.confidence, d.point)):
        if any(iou(candidate, accepted) > threshold for accepted in retained):
            continue
        retained.append(candidate)
        if len(retained) == maximum:
            break
    return retained


def assign_region(detection: Detection) -> Detection:
    center_x = detection.x + detection.width / 2.0
    center_y = detection.y + detection.height / 2.0
    region: str | None = None
    for name, (x, y, width, height) in REGIONS.items():
        if x <= center_x < x + width and y <= center_y < y + height:
            region = name
            break
    return Detection(
        point=detection.point,
        confidence=detection.confidence,
        x=detection.x,
        y=detection.y,
        width=detection.width,
        height=detection.height,
        region=region,
    )


def suppress_duplicates(detections: Sequence[Detection], threshold: float) -> list[Detection]:
    by_region: dict[str, list[Detection]] = {}
    for detection in detections:
        assert detection.region is not None
        by_region.setdefault(detection.region, []).append(detection)

    winners: list[Detection] = []
    for group in by_region.values():
        candidates = [
            detection
            for detection in group
            if not is_merged_bridge_candidate(detection, group, threshold)
        ]
        candidates.sort(key=lambda d: (-d.confidence, d.point))
        kept: list[Detection] = []
        for candidate in candidates:
            if any(overlap_over_smaller(candidate, winner) >= threshold for winner in kept):
                continue
            kept.append(candidate)
        winners.extend(kept)

    return sorted(winners, key=lambda d: d.point)


def is_merged_bridge_candidate(
    candidate: Detection,
    group: Sequence[Detection],
    threshold: float,
) -> bool:
    candidate_area = candidate.width * candidate.height
    if candidate_area <= 0.0:
        return False
    covered_smaller = [
        other
        for other in group
        if other.point != candidate.point
        and other.width * other.height < candidate_area
        and overlap_over_smaller(candidate, other) >= threshold
    ]
    for left_index, left in enumerate(covered_smaller):
        for right in covered_smaller[left_index + 1 :]:
            if overlap_over_smaller(left, right) < threshold:
                return True
    return False


def iou(left: Detection, right: Detection) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x, right.x))
    height = max(0.0, min(left.y2, right.y2) - max(left.y, right.y))
    intersection = width * height
    if intersection <= 0.0:
        return 0.0
    union = left.width * left.height + right.width * right.height - intersection
    return 0.0 if union <= 0.0 else intersection / union


def overlap_over_smaller(left: Detection, right: Detection) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x, right.x))
    height = max(0.0, min(left.y2, right.y2) - max(left.y, right.y))
    intersection = width * height
    smaller = min(left.width * left.height, right.width * right.height)
    return 0.0 if smaller <= 0.0 else intersection / smaller


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def print_summary_melds(detections: Sequence[dict[str, Any]]) -> None:
    if not detections:
        print("none")
        return
    for detection in detections:
        box = detection["compositeBox"]
        print(
            f"point={detection['detectionIndex']} conf={detection['confidence']:.6f} "
            f"box=({box['x']:.2f},{box['y']:.2f},{box['width']:.2f},{box['height']:.2f}) "
            f"classification={detection.get('classification')}"
        )


def print_postprocess_stages(stages: dict[str, Any]) -> None:
    print(f"decoded candidates above {CONFIDENCE_THRESHOLD}: {stages['decoded_count']}")
    print(f"after IoU NMS: {stages['after_nms_count']}")
    melds = stages["melds_after_nms"]
    print(f"melds after IoU NMS / before duplicate suppression: {len(melds)}")
    for detection in melds:
        box = detection["box"]
        print(
            f"  point={detection['point']} conf={detection['confidence']:.6f} "
            f"box=({box['x']:.2f},{box['y']:.2f},{box['width']:.2f},{box['height']:.2f})"
        )
    merged = stages["merged_bridge_rejections"]
    print(
        "merged bridge rejections: "
        f"{len(merged)} {[detection['point'] for detection in merged]}"
    )
    after_bridge = stages["melds_after_merged_bridge_rejection"]
    print(
        "melds after merged bridge rejection: "
        f"{len(after_bridge)} {[detection['point'] for detection in after_bridge]}"
    )
    final = stages["melds_after_duplicate_suppression"]
    print(
        "melds after pairwise duplicate suppression: "
        f"{len(final)} {[detection['point'] for detection in final]}"
    )


def print_detection_summary(detections: Sequence[Detection]) -> None:
    counts: dict[str, int] = {name: 0 for name in REGIONS}
    for detection in detections:
        if detection.region is not None:
            counts[detection.region] += 1
    print("counts:", counts)
    melds = [d for d in detections if d.region == "melds"]
    if not melds:
        print("melds: none")
        return
    for d in melds:
        print(
            f"meld point={d.point} conf={d.confidence:.6f} "
            f"box=({d.x:.2f},{d.y:.2f},{d.width:.2f},{d.height:.2f})"
        )


if __name__ == "__main__":
    raise SystemExit(main())
