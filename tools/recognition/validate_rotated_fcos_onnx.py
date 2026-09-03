from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__:
    from .rotated_fcos_nano import RotatedFCOSNano, decode_batch, rotated_iou
    from .train_rotated_fcos_nano import RotatedCocoDataset, collate_batch
else:
    repository_root_for_import = Path(__file__).resolve().parents[2]
    if str(repository_root_for_import) not in sys.path:
        sys.path.insert(0, str(repository_root_for_import))
    from tools.recognition.rotated_fcos_nano import (  # type: ignore[no-redef]
        RotatedFCOSNano,
        decode_batch,
        rotated_iou,
    )
    from tools.recognition.train_rotated_fcos_nano import (  # type: ignore[no-redef]
        RotatedCocoDataset,
        collate_batch,
    )


EXPECTED_OUTPUT_NAMES = ("stride8", "stride16", "stride32")
EXPECTED_OUTPUT_SHAPES = ((1, 8, 40, 40), (1, 8, 20, 20), (1, 8, 10, 10))


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    run_directory = (
        repository_root
        / ".local"
        / "recognition"
        / "rotated_fcos_runs"
        / "rfcos_nano_s05_f64_obb120_gn_seed42"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Validate raw-output and decoded-detection parity between a RotatedFCOSNano "
            "PyTorch checkpoint and its exported ONNX model on real validation composites."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--checkpoint", type=Path, default=run_directory / "model_best.pt")
    parser.add_argument("--onnx", type=Path, default=run_directory / "model_best.onnx")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=(
            repository_root
            / ".local"
            / "recognition"
            / "rotated_detector_corpus"
            / "annotations"
            / "val.json"
        ),
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=64)
    parser.add_argument("--raw-atol", type=float, default=1.0e-4)
    parser.add_argument("--raw-rtol", type=float, default=1.0e-4)
    parser.add_argument("--detection-atol", type=float, default=1.0e-3)
    parser.add_argument(
        "--report",
        type=Path,
        default=run_directory / "onnx_parity.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    onnx_path = args.onnx.resolve()
    annotations_path = args.annotations.resolve()
    report_path = args.report.resolve()

    for path in (checkpoint_path, onnx_path, annotations_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if int(args.samples) < 1:
        raise ValueError("--samples must be positive")

    require_onnx_dependencies()
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_contract = validate_onnx_contract(session)

    payload = load_trusted_checkpoint(checkpoint_path)
    config = payload["model_config"]
    model = RotatedFCOSNano(
        backbone=str(config["backbone"]),
        fpn_channels=int(config["fpn_channels"]),
        head_convs=int(config["head_convs"]),
        head_normalization=str(config.get("head_normalization", "batch")),
        pretrained_backbone=False,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    dataset = RotatedCocoDataset(annotations_path, repository_root=repository_root)
    sample_count = min(int(args.samples), len(dataset))
    sample_indices = evenly_spaced_indices(len(dataset), sample_count)

    raw_level_summaries = [new_error_accumulator() for _ in EXPECTED_OUTPUT_NAMES]
    decoded_sample_rows: list[dict[str, Any]] = []
    decoded_equivalent = True

    for sample_index in sample_indices:
        image, _boxes, metadata = dataset[sample_index]
        batch = image.unsqueeze(0).contiguous()
        with torch.inference_mode():
            pytorch_outputs = tuple(output.detach().cpu() for output in model(batch))

        ort_outputs = session.run(
            list(EXPECTED_OUTPUT_NAMES),
            {session.get_inputs()[0].name: batch.numpy()},
        )
        onnx_outputs = tuple(torch.from_numpy(np.asarray(output, dtype=np.float32)) for output in ort_outputs)

        for level_index, (expected, observed) in enumerate(
            zip(pytorch_outputs, onnx_outputs, strict=True)
        ):
            update_error_accumulator(
                raw_level_summaries[level_index],
                expected.numpy(),
                observed.numpy(),
                atol=float(args.raw_atol),
                rtol=float(args.raw_rtol),
            )

        pytorch_detections = decode_batch(
            pytorch_outputs,
            score_threshold=float(args.score_threshold),
            nms_iou_threshold=float(args.nms_iou_threshold),
            max_detections=int(args.max_detections),
        )[0]
        onnx_detections = decode_batch(
            onnx_outputs,
            score_threshold=float(args.score_threshold),
            nms_iou_threshold=float(args.nms_iou_threshold),
            max_detections=int(args.max_detections),
        )[0]
        decoded = compare_detection_sets(
            pytorch_detections,
            onnx_detections,
            atol=float(args.detection_atol),
        )
        decoded_equivalent = decoded_equivalent and bool(decoded["equivalent"])
        decoded_sample_rows.append(
            {
                "dataset_index": sample_index,
                "image_id": metadata.get("image_id"),
                "file_name": metadata.get("file_name"),
                **decoded,
            }
        )

    finalized_raw = [finalize_error_accumulator(item) for item in raw_level_summaries]
    raw_allclose = all(bool(item["allclose"]) for item in finalized_raw)
    status = "passed" if raw_allclose and decoded_equivalent else "failed"

    report = {
        "status": status,
        "checkpoint": {
            "path": relative_or_absolute(checkpoint_path, repository_root),
            "epoch": int(payload.get("epoch", -1)),
            "sha256": sha256_file(checkpoint_path),
        },
        "onnx": {
            "path": relative_or_absolute(onnx_path, repository_root),
            "sha256": sha256_file(onnx_path),
            "bytes": onnx_path.stat().st_size,
            "contract": onnx_contract,
            "onnxruntime": {
                "version": ort.__version__,
                "providers": session.get_providers(),
            },
        },
        "model_config": config,
        "samples": {
            "count": sample_count,
            "indices": sample_indices,
            "annotations": relative_or_absolute(annotations_path, repository_root),
        },
        "raw_output_parity": {
            "atol": float(args.raw_atol),
            "rtol": float(args.raw_rtol),
            "allclose": raw_allclose,
            "levels": {
                name: result
                for name, result in zip(EXPECTED_OUTPUT_NAMES, finalized_raw, strict=True)
            },
        },
        "decoded_parity": {
            "score_threshold": float(args.score_threshold),
            "nms_iou_threshold": float(args.nms_iou_threshold),
            "max_detections": int(args.max_detections),
            "atol": float(args.detection_atol),
            "equivalent": decoded_equivalent,
            "samples": decoded_sample_rows,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": status,
                "onnx_sha256": report["onnx"]["sha256"],
                "onnx_bytes": report["onnx"]["bytes"],
                "raw_allclose": raw_allclose,
                "raw_max_abs_error": max(
                    float(item["max_abs_error"]) for item in finalized_raw
                ),
                "decoded_equivalent": decoded_equivalent,
                "sample_count": sample_count,
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if status != "passed":
        return 1
    return 0


def require_onnx_dependencies() -> None:
    missing: list[str] = []
    for name in ("onnx", "onnxruntime"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing ONNX parity dependencies: "
            + ", ".join(missing)
            + ". Install them into the current virtual environment."
        )


def validate_onnx_contract(session: Any) -> dict[str, Any]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise AssertionError(f"Expected one ONNX input, found {len(inputs)}")
    if tuple(inputs[0].shape) != (1, 3, 320, 320):
        raise AssertionError(f"Unexpected ONNX input shape: {inputs[0].shape}")
    output_names = tuple(output.name for output in outputs)
    if output_names != EXPECTED_OUTPUT_NAMES:
        raise AssertionError(
            f"Unexpected ONNX outputs: {output_names}; expected {EXPECTED_OUTPUT_NAMES}"
        )
    output_shapes = tuple(tuple(int(value) for value in output.shape) for output in outputs)
    if output_shapes != EXPECTED_OUTPUT_SHAPES:
        raise AssertionError(
            f"Unexpected ONNX output shapes: {output_shapes}; expected {EXPECTED_OUTPUT_SHAPES}"
        )
    return {
        "input_name": inputs[0].name,
        "input_shape": list(inputs[0].shape),
        "output_names": list(output_names),
        "output_shapes": [list(shape) for shape in output_shapes],
    }


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def new_error_accumulator() -> dict[str, Any]:
    return {
        "allclose": True,
        "count": 0,
        "sum_abs_error": 0.0,
        "max_abs_error": 0.0,
    }


def update_error_accumulator(
    accumulator: dict[str, Any],
    expected: np.ndarray,
    observed: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> None:
    expected = np.asarray(expected, dtype=np.float32)
    observed = np.asarray(observed, dtype=np.float32)
    if expected.shape != observed.shape:
        accumulator["allclose"] = False
        accumulator["shape_mismatch"] = {
            "expected": list(expected.shape),
            "observed": list(observed.shape),
        }
        return
    difference = np.abs(expected - observed)
    accumulator["allclose"] = bool(accumulator["allclose"]) and bool(
        np.allclose(expected, observed, atol=atol, rtol=rtol)
    )
    accumulator["count"] = int(accumulator["count"]) + int(difference.size)
    accumulator["sum_abs_error"] = float(accumulator["sum_abs_error"]) + float(
        difference.sum(dtype=np.float64)
    )
    accumulator["max_abs_error"] = max(
        float(accumulator["max_abs_error"]),
        float(difference.max()) if difference.size else 0.0,
    )


def finalize_error_accumulator(accumulator: dict[str, Any]) -> dict[str, Any]:
    count = int(accumulator["count"])
    return {
        "allclose": bool(accumulator["allclose"]),
        "max_abs_error": float(accumulator["max_abs_error"]),
        "mean_abs_error": (
            float(accumulator["sum_abs_error"]) / count if count else 0.0
        ),
        "value_count": count,
        **(
            {"shape_mismatch": accumulator["shape_mismatch"]}
            if "shape_mismatch" in accumulator
            else {}
        ),
    }


def compare_detection_sets(
    expected: Sequence[Any], observed: Sequence[Any], *, atol: float
) -> dict[str, Any]:
    if len(expected) != len(observed):
        return {
            "equivalent": False,
            "pytorch_count": len(expected),
            "onnx_count": len(observed),
            "reason": "detection count mismatch",
        }
    unmatched = set(range(len(observed)))
    max_parameter_error = 0.0
    minimum_iou = 1.0
    for expected_detection in expected:
        if not unmatched:
            return {
                "equivalent": False,
                "pytorch_count": len(expected),
                "onnx_count": len(observed),
                "reason": "ran out of ONNX detections",
            }
        matched_index = max(
            unmatched,
            key=lambda index: rotated_iou(expected_detection.obb, observed[index].obb),
        )
        observed_detection = observed[matched_index]
        iou = rotated_iou(expected_detection.obb, observed_detection.obb)
        parameter_error = max(
            abs(float(expected_detection.score) - float(observed_detection.score)),
            abs(float(expected_detection.cx) - float(observed_detection.cx)),
            abs(float(expected_detection.cy) - float(observed_detection.cy)),
            abs(float(expected_detection.width) - float(observed_detection.width)),
            abs(float(expected_detection.height) - float(observed_detection.height)),
            angle_difference_deg(
                float(expected_detection.angle_deg), float(observed_detection.angle_deg)
            ),
        )
        max_parameter_error = max(max_parameter_error, parameter_error)
        minimum_iou = min(minimum_iou, iou)
        unmatched.remove(matched_index)
    return {
        "equivalent": max_parameter_error <= atol and not unmatched,
        "pytorch_count": len(expected),
        "onnx_count": len(observed),
        "max_parameter_error": max_parameter_error,
        "minimum_matched_rotated_iou": minimum_iou if expected else 1.0,
    }


def angle_difference_deg(first: float, second: float) -> float:
    delta = abs(((first - second + 90.0) % 180.0) - 90.0)
    return min(delta, 180.0 - delta)


def load_trusted_checkpoint(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    payload = torch.load(path, **kwargs)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint root must be a dict")
    return payload


def relative_or_absolute(path: Path, repository_root: Path) -> str:
    try:
        return str(path.relative_to(repository_root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
