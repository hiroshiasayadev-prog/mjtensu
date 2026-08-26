from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


PINNED_NANODET_COMMIT = "d3fb34fa91d6020f273d6d063bf324dcd97bac12"
INPUT_SIZE = 320
NUM_CLASSES = 1
REG_MAX = 7
OUTPUT_POINTS = 2125
OUTPUT_CHANNELS = NUM_CLASSES + 4 * (REG_MAX + 1)
STRIDES = (8, 16, 32, 64)
BGR_MEAN = np.asarray([103.53, 116.28, 123.675], dtype=np.float32)
BGR_STD = np.asarray([57.375, 57.12, 58.395], dtype=np.float32)


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass(frozen=True)
class FileInventoryEntry:
    path: str
    size_bytes: int
    sha256: str


class ConsoleLogger:
    def log(self, message: object) -> None:
        print(message)


@dataclass(frozen=True)
class ParsedArguments:
    repository_root: Path
    nanodet_root: Path
    run_directory: Path
    config_path: Path
    model_path: Path | None
    image_path: Path | None
    output_path: Path
    pwa_model_path: Path
    report_path: Path
    confidence_threshold: float
    nms_iou_threshold: float
    max_detections: int
    raw_atol: float
    raw_rtol: float
    bbox_atol: float
    skip_export: bool


def parse_args() -> ParsedArguments:
    repository_root_default = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the E1 epoch-40 model_best artifact, export it through NanoDet "
            "v1.0.0 tools/export_onnx.py, then verify raw output and decoded bbox "
            "parity between PyTorch and ONNX Runtime Python."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root_default)
    parser.add_argument("--nanodet-root", type=Path)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Explicit checkpoint path. When omitted, model_best is inventoried and resolved.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        help=(
            "Parity input image. When omitted, select a validation image whose annotation "
            "count is closest to fourteen."
        ),
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--pwa-model-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=200)
    parser.add_argument("--raw-atol", type=float, default=1.0e-4)
    parser.add_argument("--raw-rtol", type=float, default=1.0e-4)
    parser.add_argument("--bbox-atol", type=float, default=1.0e-3)
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Validate an already-exported output-path without invoking export_onnx.py.",
    )
    namespace = parser.parse_args()

    repository_root = namespace.repository_root.resolve()
    nanodet_root = (namespace.nanodet_root or repository_root / "nanodet").resolve()
    run_directory = (
        namespace.run_directory
        or repository_root
        / ".local"
        / "recognition"
        / "nanodet_runs"
        / "E1_plus_m_320_amp30_seed42"
    ).resolve()
    config_path = (
        namespace.config_path
        or repository_root
        / "tools"
        / "recognition"
        / "nanodet"
        / "configs"
        / "e1_nanodet_plus_m_320_stage40_amp_resume.yml"
    ).resolve()
    output_directory = repository_root / ".local" / "recognition" / "browser_probe"
    output_path = (
        namespace.output_path or output_directory / "nanodet-plus-m-320.onnx"
    ).resolve()
    pwa_model_path = (
        namespace.pwa_model_path
        or repository_root
        / "tools"
        / "recognition"
        / "pwa_detector_probe"
        / "public"
        / "models"
        / "nanodet-plus-m-320.onnx"
    ).resolve()
    report_path = (
        namespace.report_path or output_directory / "e1-320-onnx-parity.json"
    ).resolve()

    return ParsedArguments(
        repository_root=repository_root,
        nanodet_root=nanodet_root,
        run_directory=run_directory,
        config_path=config_path,
        model_path=namespace.model_path.resolve() if namespace.model_path else None,
        image_path=namespace.image_path.resolve() if namespace.image_path else None,
        output_path=output_path,
        pwa_model_path=pwa_model_path,
        report_path=report_path,
        confidence_threshold=namespace.confidence_threshold,
        nms_iou_threshold=namespace.nms_iou_threshold,
        max_detections=namespace.max_detections,
        raw_atol=namespace.raw_atol,
        raw_rtol=namespace.raw_rtol,
        bbox_atol=namespace.bbox_atol,
        skip_export=namespace.skip_export,
    )


def main() -> int:
    arguments = parse_args()
    validate_arguments(arguments)

    nanodet_evidence = inspect_nanodet_source(arguments.nanodet_root)
    model_best_path = arguments.run_directory / "model_best"
    model_inventory = inventory_path(model_best_path)
    if not model_inventory:
        raise FileNotFoundError(f"No model_best artifact found under {model_best_path}")
    print(json.dumps({"model_best_inventory": [asdict(item) for item in model_inventory]}, indent=2))

    checkpoint_path = arguments.model_path or resolve_model_best_checkpoint(model_best_path)
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resolved checkpoint does not exist: {checkpoint_path}")

    parity_image_path, image_selection = resolve_parity_image(arguments)
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.pwa_model_path.parent.mkdir(parents=True, exist_ok=True)

    if not arguments.skip_export:
        export_onnx(arguments, checkpoint_path)
    elif not arguments.output_path.is_file():
        raise FileNotFoundError(
            f"--skip-export was set but ONNX output does not exist: {arguments.output_path}"
        )

    onnx_contract = validate_onnx_contract(arguments.output_path)
    input_tensor, preprocessing = preprocess_image(parity_image_path)
    pytorch_output = run_pytorch(
        arguments.nanodet_root,
        arguments.config_path,
        checkpoint_path,
        input_tensor,
    )
    onnx_output, ort_metadata = run_onnx_runtime(arguments.output_path, input_tensor)

    raw_comparison = compare_raw_outputs(
        pytorch_output,
        onnx_output,
        atol=arguments.raw_atol,
        rtol=arguments.raw_rtol,
    )

    pytorch_detections = decode_output(
        pytorch_output,
        confidence_threshold=arguments.confidence_threshold,
        nms_iou_threshold=arguments.nms_iou_threshold,
        max_detections=arguments.max_detections,
    )
    onnx_detections = decode_output(
        onnx_output,
        confidence_threshold=arguments.confidence_threshold,
        nms_iou_threshold=arguments.nms_iou_threshold,
        max_detections=arguments.max_detections,
    )
    bbox_comparison = compare_detection_sets(
        pytorch_detections,
        onnx_detections,
        coordinate_and_score_atol=arguments.bbox_atol,
    )

    if not raw_comparison["allclose"]:
        raise AssertionError(
            "PyTorch export-contract output and ONNX Runtime output are not within "
            f"atol={arguments.raw_atol}, rtol={arguments.raw_rtol}: {raw_comparison}"
        )
    if not bbox_comparison["equivalent"]:
        raise AssertionError(f"Decoded bbox outputs are not equivalent: {bbox_comparison}")

    shutil.copy2(arguments.output_path, arguments.pwa_model_path)
    model_metadata_path = arguments.pwa_model_path.with_suffix(".metadata.json")

    report: dict[str, Any] = {
        "status": "passed",
        "nanodet": nanodet_evidence,
        "run_directory": path_for_report(arguments.run_directory, arguments.repository_root),
        "model_best_inventory": [asdict(item) for item in model_inventory],
        "selected_checkpoint": file_entry(checkpoint_path, arguments.repository_root),
        "config_path": path_for_report(arguments.config_path, arguments.repository_root),
        "parity_image": {
            **image_selection,
            "path": path_for_report(parity_image_path, arguments.repository_root),
            "sha256": sha256_file(parity_image_path),
        },
        "preprocessing": preprocessing,
        "onnx": {
            "path": path_for_report(arguments.output_path, arguments.repository_root),
            "size_bytes": arguments.output_path.stat().st_size,
            "sha256": sha256_file(arguments.output_path),
            "contract": onnx_contract,
            "onnx_runtime": ort_metadata,
        },
        "postprocess": {
            "class_count": NUM_CLASSES,
            "reg_max": REG_MAX,
            "strides": list(STRIDES),
            "center_prior": "(column * stride, row * stride), no half-stride offset",
            "class_activation": "sigmoid is part of the exported ONNX graph",
            "distance_decode": "softmax over bins 0..7, expected value multiplied by stride",
            "confidence_threshold": arguments.confidence_threshold,
            "nms_iou_threshold": arguments.nms_iou_threshold,
            "max_detections": arguments.max_detections,
        },
        "raw_output_comparison": raw_comparison,
        "bbox_comparison": bbox_comparison,
        "detections": {
            "pytorch_count": len(pytorch_detections),
            "onnx_runtime_count": len(onnx_detections),
            "pytorch": [asdict(detection) for detection in pytorch_detections],
            "onnx_runtime": [asdict(detection) for detection in onnx_detections],
        },
        "pwa_model_path": path_for_report(arguments.pwa_model_path, arguments.repository_root),
    }
    atomic_write_json(arguments.report_path, report)

    metadata = {
        "source_report": path_for_report(arguments.report_path, arguments.repository_root),
        "sha256": report["onnx"]["sha256"],
        "size_bytes": report["onnx"]["size_bytes"],
        "input_shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
        "output_shape": [1, OUTPUT_POINTS, OUTPUT_CHANNELS],
        "preprocess": preprocessing,
        "postprocess": report["postprocess"],
    }
    atomic_write_json(model_metadata_path, metadata)

    print(
        json.dumps(
            {
                "status": "passed",
                "checkpoint": str(checkpoint_path),
                "onnx": str(arguments.output_path),
                "pwa_model": str(arguments.pwa_model_path),
                "report": str(arguments.report_path),
                "raw_max_abs_error": raw_comparison["max_abs_error"],
                "detections": len(onnx_detections),
            },
            indent=2,
        )
    )
    return 0


def validate_arguments(arguments: ParsedArguments) -> None:
    required_paths = {
        "repository root": arguments.repository_root,
        "NanoDet root": arguments.nanodet_root,
        "run directory": arguments.run_directory,
        "config": arguments.config_path,
    }
    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not 0.0 <= arguments.confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be between zero and one")
    if not 0.0 <= arguments.nms_iou_threshold <= 1.0:
        raise ValueError("NMS IoU threshold must be between zero and one")
    if arguments.max_detections <= 0:
        raise ValueError("max detections must be positive")


def inspect_nanodet_source(nanodet_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=nanodet_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if commit != PINNED_NANODET_COMMIT:
        raise RuntimeError(
            f"NanoDet commit is {commit}; expected pinned v1.0.0 commit {PINNED_NANODET_COMMIT}"
        )

    export_script = nanodet_root / "tools" / "export_onnx.py"
    head_source = nanodet_root / "nanodet" / "model" / "head" / "nanodet_plus_head.py"
    integral_source = nanodet_root / "nanodet" / "model" / "module" / "conv.py"
    for source_path in (export_script, head_source):
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing required NanoDet source: {source_path}")

    head_text = head_source.read_text(encoding="utf-8")
    required_head_markers = (
        "_forward_onnx",
        "sigmoid",
        "distribution_project",
        "center_priors",
        "multiclass_nms",
    )
    missing_markers = [marker for marker in required_head_markers if marker not in head_text]
    if missing_markers:
        raise RuntimeError(
            f"NanoDet head source does not contain expected postprocess markers: {missing_markers}"
        )

    source_files = [export_script, head_source]
    if integral_source.is_file():
        source_files.append(integral_source)
    return {
        "commit": commit,
        "expected_commit": PINNED_NANODET_COMMIT,
        "source_files": [
            {
                "path": str(path.relative_to(nanodet_root)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
    }


def inventory_path(path: Path) -> list[FileInventoryEntry]:
    if path.is_file():
        paths = [path]
        root = path.parent
    elif path.is_dir():
        paths = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        root = path
    else:
        return []
    return [
        FileInventoryEntry(
            path=str(candidate.relative_to(root)).replace("\\", "/"),
            size_bytes=candidate.stat().st_size,
            sha256=sha256_file(candidate),
        )
        for candidate in paths
    ]


def resolve_model_best_checkpoint(model_best_path: Path) -> Path:
    if model_best_path.is_file():
        return model_best_path

    exact_candidates = [
        model_best_path / "model_best.ckpt",
        model_best_path / "model_best.pth",
        model_best_path / "model_best.pt",
    ]
    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate

    candidates = sorted(
        candidate
        for candidate in model_best_path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".ckpt", ".pth", ".pt"}
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-like file found under {model_best_path}")
    raise RuntimeError(
        "Multiple checkpoint-like files exist under model_best; pass --model-path explicitly: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_parity_image(arguments: ParsedArguments) -> tuple[Path, dict[str, Any]]:
    if arguments.image_path is not None:
        if not arguments.image_path.is_file():
            raise FileNotFoundError(f"Parity image does not exist: {arguments.image_path}")
        return arguments.image_path, {"selection": "explicit --image-path"}

    annotation_path = (
        arguments.repository_root
        / ".local"
        / "recognition"
        / "nanodet_single_class_dataset"
        / "annotations"
        / "instances_val.json"
    )
    if not annotation_path.is_file():
        raise FileNotFoundError(
            "No --image-path was supplied and the generated validation annotation file is missing: "
            f"{annotation_path}"
        )

    with annotation_path.open("r", encoding="utf-8") as source:
        coco = json.load(source)
    images = coco.get("images")
    annotations = coco.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"Invalid COCO payload: {annotation_path}")

    counts = Counter(int(annotation["image_id"]) for annotation in annotations)
    ranked_images = sorted(
        images,
        key=lambda image: (
            abs(counts[int(image["id"])] - 14),
            int(image["id"]),
        ),
    )
    if not ranked_images:
        raise ValueError(f"No images in validation annotations: {annotation_path}")
    selected = ranked_images[0]
    image_id = int(selected["id"])
    file_name = str(selected["file_name"])
    image_path = arguments.repository_root / "data" / Path(file_name)
    if not image_path.is_file():
        raise FileNotFoundError(f"Selected validation image does not exist: {image_path}")
    return image_path.resolve(), {
        "selection": "validation image with annotation count nearest to fourteen",
        "generated_image_id": image_id,
        "annotation_count": counts[image_id],
        "annotation_path": path_for_report(annotation_path, arguments.repository_root),
    }


def export_onnx(arguments: ParsedArguments, checkpoint_path: Path) -> None:
    export_script = arguments.nanodet_root / "tools" / "export_onnx.py"
    if arguments.output_path.exists():
        arguments.output_path.unlink()
    command = [
        sys.executable,
        str(export_script),
        "--cfg_path",
        str(arguments.config_path),
        "--model_path",
        str(checkpoint_path),
        "--out_path",
        str(arguments.output_path),
        "--input_shape",
        f"{INPUT_SIZE},{INPUT_SIZE}",
    ]
    print(json.dumps({"official_export_command": command}, indent=2))
    subprocess.run(command, cwd=arguments.nanodet_root, check=True)
    if not arguments.output_path.is_file():
        raise FileNotFoundError(
            f"NanoDet export command returned successfully but did not create {arguments.output_path}"
        )


def validate_onnx_contract(path: Path) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("onnx is required for graph validation") from error

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1:
        raise AssertionError(f"Expected one ONNX input, found {len(model.graph.input)}")
    if len(model.graph.output) != 1:
        raise AssertionError(f"Expected one ONNX output, found {len(model.graph.output)}")

    input_value = model.graph.input[0]
    output_value = model.graph.output[0]
    input_shape = tensor_shape(input_value)
    output_shape = tensor_shape(output_value)
    expected_input_shape = [1, 3, INPUT_SIZE, INPUT_SIZE]
    if input_shape != expected_input_shape:
        raise AssertionError(f"ONNX input shape is {input_shape}; expected {expected_input_shape}")
    if all(dimension is not None for dimension in output_shape):
        expected_output_shape = [1, OUTPUT_POINTS, OUTPUT_CHANNELS]
        if output_shape != expected_output_shape:
            raise AssertionError(
                f"ONNX output shape is {output_shape}; expected {expected_output_shape}"
            )

    return {
        "opset_imports": [
            {"domain": opset.domain or "ai.onnx", "version": opset.version}
            for opset in model.opset_import
        ],
        "input_name": input_value.name,
        "input_shape": input_shape,
        "output_name": output_value.name,
        "output_shape": output_shape,
    }


def tensor_shape(value_info: Any) -> list[int | None]:
    result: list[int | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        else:
            result.append(None)
    return result


def preprocess_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode parity image: {path}")
    source_height, source_width = image.shape[:2]
    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) - BGR_MEAN) / BGR_STD
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])
    return tensor, {
        "source_size": [source_width, source_height],
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "resize": "direct stretch to 320 x 320 using OpenCV bilinear interpolation",
        "channel_order": "BGR planar NCHW",
        "mean": BGR_MEAN.tolist(),
        "std": BGR_STD.tolist(),
        "dtype": "float32",
    }


def run_pytorch(
    nanodet_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    input_tensor: np.ndarray,
) -> np.ndarray:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for parity validation") from error

    sys.path.insert(0, str(nanodet_root))
    try:
        from nanodet.model.arch import build_model
        from nanodet.util import cfg, load_config, load_model_weight
    finally:
        try:
            sys.path.remove(str(nanodet_root))
        except ValueError:
            pass

    load_config(cfg, str(config_path))
    model = build_model(cfg.model)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    load_model_weight(model, checkpoint, ConsoleLogger())
    model.eval()

    with torch.inference_mode():
        raw_prediction = model(torch.from_numpy(input_tensor))
        if isinstance(raw_prediction, (tuple, list)):
            tensor_predictions = [
                candidate for candidate in raw_prediction if isinstance(candidate, torch.Tensor)
            ]
            if len(tensor_predictions) != 1:
                raise TypeError(
                    "Expected one tensor prediction from the eval model, found "
                    f"{len(tensor_predictions)} tensor values in {type(raw_prediction).__name__}"
                )
            raw_prediction = tensor_predictions[0]
        if not isinstance(raw_prediction, torch.Tensor):
            raise TypeError(
                f"NanoDet eval forward returned {type(raw_prediction).__name__}, expected Tensor"
            )
        if tuple(raw_prediction.shape) != (1, OUTPUT_POINTS, OUTPUT_CHANNELS):
            raise AssertionError(
                f"PyTorch output shape is {tuple(raw_prediction.shape)}, expected "
                f"(1, {OUTPUT_POINTS}, {OUTPUT_CHANNELS})"
            )
        export_contract = torch.cat(
            [raw_prediction[..., :NUM_CLASSES].sigmoid(), raw_prediction[..., NUM_CLASSES:]],
            dim=-1,
        )
    return np.ascontiguousarray(export_contract.cpu().numpy(), dtype=np.float32)


def run_onnx_runtime(path: Path, input_tensor: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required for parity validation") from error

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise AssertionError(
            f"Expected one input and one output, found {len(inputs)} inputs and {len(outputs)} outputs"
        )
    output = session.run([outputs[0].name], {inputs[0].name: input_tensor})[0]
    output = np.ascontiguousarray(output, dtype=np.float32)
    if output.shape != (1, OUTPUT_POINTS, OUTPUT_CHANNELS):
        raise AssertionError(
            f"ONNX Runtime output shape is {output.shape}, expected "
            f"(1, {OUTPUT_POINTS}, {OUTPUT_CHANNELS})"
        )
    return output, {
        "version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "selected_providers": session.get_providers(),
    }


def compare_raw_outputs(
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if pytorch_output.shape != onnx_output.shape:
        return {
            "allclose": False,
            "pytorch_shape": list(pytorch_output.shape),
            "onnx_runtime_shape": list(onnx_output.shape),
            "reason": "shape mismatch",
        }
    absolute_error = np.abs(pytorch_output - onnx_output)
    maximum_index = np.unravel_index(int(np.argmax(absolute_error)), absolute_error.shape)
    return {
        "allclose": bool(np.allclose(pytorch_output, onnx_output, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
        "shape": list(pytorch_output.shape),
        "max_abs_error": float(absolute_error[maximum_index]),
        "max_abs_error_index": [int(index) for index in maximum_index],
        "mean_abs_error": float(absolute_error.mean()),
        "pytorch_value_at_max_error": float(pytorch_output[maximum_index]),
        "onnx_value_at_max_error": float(onnx_output[maximum_index]),
    }


def decode_output(
    output: np.ndarray,
    *,
    confidence_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> list[Detection]:
    values = output[0]
    priors = build_center_priors()
    if values.shape != (len(priors), OUTPUT_CHANNELS):
        raise AssertionError(
            f"Decoder received output shape {values.shape}, expected {(len(priors), OUTPUT_CHANNELS)}"
        )

    candidates: list[Detection] = []
    for point_index, row in enumerate(values):
        score = float(row[0])
        if score <= confidence_threshold:
            continue
        prior_x, prior_y, stride = priors[point_index]
        distances = distribution_expectation(row[NUM_CLASSES:]).reshape(4) * stride
        left, top, right, bottom = (float(value) for value in distances)
        x1 = clamp(prior_x - left, 0.0, float(INPUT_SIZE))
        y1 = clamp(prior_y - top, 0.0, float(INPUT_SIZE))
        x2 = clamp(prior_x + right, 0.0, float(INPUT_SIZE))
        y2 = clamp(prior_y + bottom, 0.0, float(INPUT_SIZE))
        if x2 > x1 and y2 > y1:
            candidates.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score))

    return non_maximum_suppression(candidates, nms_iou_threshold, max_detections)


def build_center_priors() -> list[tuple[float, float, float]]:
    priors: list[tuple[float, float, float]] = []
    for stride in STRIDES:
        feature_width = math.ceil(INPUT_SIZE / stride)
        feature_height = math.ceil(INPUT_SIZE / stride)
        for row in range(feature_height):
            for column in range(feature_width):
                priors.append((float(column * stride), float(row * stride), float(stride)))
    if len(priors) != OUTPUT_POINTS:
        raise AssertionError(f"Generated {len(priors)} center priors, expected {OUTPUT_POINTS}")
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
    sorted_detections = sorted(detections, key=lambda detection: detection.score, reverse=True)
    retained: list[Detection] = []
    for candidate in sorted_detections:
        if any(intersection_over_union(candidate, accepted) > iou_threshold for accepted in retained):
            continue
        retained.append(candidate)
        if len(retained) >= max_detections:
            break
    return retained


def compare_detection_sets(
    pytorch_detections: Sequence[Detection],
    onnx_detections: Sequence[Detection],
    *,
    coordinate_and_score_atol: float,
) -> dict[str, Any]:
    if len(pytorch_detections) != len(onnx_detections):
        return {
            "equivalent": False,
            "pytorch_count": len(pytorch_detections),
            "onnx_runtime_count": len(onnx_detections),
            "reason": "detection count mismatch",
        }

    unmatched_onnx = set(range(len(onnx_detections)))
    match_records: list[dict[str, Any]] = []
    maximum_error = 0.0
    minimum_match_iou = 1.0
    for pytorch_index, pytorch_detection in enumerate(pytorch_detections):
        if not unmatched_onnx:
            break
        matched_index = max(
            unmatched_onnx,
            key=lambda index: intersection_over_union(
                pytorch_detection,
                onnx_detections[index],
            ),
        )
        onnx_detection = onnx_detections[matched_index]
        match_iou = intersection_over_union(pytorch_detection, onnx_detection)
        errors = [
            abs(pytorch_detection.x1 - onnx_detection.x1),
            abs(pytorch_detection.y1 - onnx_detection.y1),
            abs(pytorch_detection.x2 - onnx_detection.x2),
            abs(pytorch_detection.y2 - onnx_detection.y2),
            abs(pytorch_detection.score - onnx_detection.score),
        ]
        record_maximum_error = max(errors)
        maximum_error = max(maximum_error, record_maximum_error)
        minimum_match_iou = min(minimum_match_iou, match_iou)
        match_records.append(
            {
                "pytorch_index": pytorch_index,
                "onnx_runtime_index": matched_index,
                "iou": match_iou,
                "maximum_coordinate_or_score_error": record_maximum_error,
            }
        )
        unmatched_onnx.remove(matched_index)

    equivalent = maximum_error <= coordinate_and_score_atol and not unmatched_onnx
    return {
        "equivalent": equivalent,
        "atol": coordinate_and_score_atol,
        "count": len(pytorch_detections),
        "maximum_coordinate_or_score_error": maximum_error,
        "minimum_matched_iou": minimum_match_iou if match_records else 1.0,
        "unmatched_onnx_indices": sorted(unmatched_onnx),
        "matches": match_records,
    }


def intersection_over_union(left: Detection, right: Detection) -> float:
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


def file_entry(path: Path, repository_root: Path) -> dict[str, Any]:
    return {
        "path": path_for_report(path, repository_root),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_for_report(path: Path, repository_root: Path) -> str:
    try:
        report_path = path.resolve().relative_to(repository_root.resolve())
    except ValueError:
        report_path = path.resolve()
    return str(report_path).replace("\\", "/")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
