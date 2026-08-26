from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

try:
    from red_five_classifier import INPUT_CHANNELS, build_model as build_red_five_model
    from tile_shape_classifier import build_model as build_tile_shape_model
except ModuleNotFoundError:  # package import path used by tests/tools
    from tools.recognition.red_five_classifier import (
        INPUT_CHANNELS,
        build_model as build_red_five_model,
    )
    from tools.recognition.tile_shape_classifier import build_model as build_tile_shape_model


DEFAULT_OPSET = 16
DEFAULT_PARITY_SAMPLES = 16
C8_GROUP_SIZE = 8


class C8GroupMaxPool(nn.Module):
    """Pure-PyTorch equivalent of escnn GroupPooling for C8 regular fields.

    The final equivariant backbone contains consecutive regular C8 fields, each
    represented by eight channels. GroupPooling takes the maximum over those
    eight orientation channels for every field. Keeping this operation here
    removes the escnn MaxPoolChannels dependency from the deployment graph.
    """

    def __init__(self, group_size: int = C8_GROUP_SIZE) -> None:
        super().__init__()
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.group_size = int(group_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim != 4:
            raise ValueError(f"Expected NCHW input, got {tuple(input_tensor.shape)}")
        batch, channels, height, width = input_tensor.shape
        if channels % self.group_size != 0:
            raise ValueError(
                f"Channel count {channels} is not divisible by C8 group size {self.group_size}"
            )
        fields = channels // self.group_size
        grouped = input_tensor.reshape(
            batch,
            fields,
            self.group_size,
            height,
            width,
        )
        return grouped.amax(dim=2)


class ExportedC8Classifier(nn.Module):
    """Tensor-only deployment form of the trained escnn classifier."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        spatial_pool: nn.Module,
        classifier: nn.Module,
        group_size: int = C8_GROUP_SIZE,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.group_pool = C8GroupMaxPool(group_size)
        self.spatial_pool = spatial_pool
        self.classifier = classifier

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        invariant = self.group_pool(features)
        pooled = self.spatial_pool(invariant)
        return self.classifier(pooled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained escnn C8 tile classifier to a tensor-only PyTorch graph, "
            "then ONNX, and verify escnn -> exported PyTorch -> ONNX Runtime parity. "
            "Both the base tile-shape classifier and red-five specialist are supported."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("auto", "tile-shape", "red-five"),
        default="auto",
        help="Normally inferred from checkpoint config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to <checkpoint stem>.onnx beside the checkpoint.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Defaults to <output>.metadata.json.",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--parity-samples", type=int, default=DEFAULT_PARITY_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument(
        "--fixed-batch",
        action="store_true",
        help="Export batch size 1 instead of a dynamic batch dimension.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing ONNX/metadata outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.opset < 11:
        raise ValueError("--opset must be at least 11")
    if args.parity_samples < 1:
        raise ValueError("--parity-samples must be positive")
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("parity tolerances must be non-negative")

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_path = (
        args.output.resolve()
        if args.output is not None
        else checkpoint_path.with_suffix(".onnx")
    )
    metadata_path = (
        args.metadata_output.resolve()
        if args.metadata_output is not None
        else output_path.with_suffix(output_path.suffix + ".metadata.json")
    )
    for path in (output_path, metadata_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}; use --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint root must be a dict")
    config = checkpoint.get("config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing dict config")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict")

    kind = resolve_kind(str(args.kind), config)
    source_model, model_info = build_source_model(kind, config, state_dict)
    source_model.load_state_dict(state_dict, strict=True)
    source_model.eval()

    exported_model = build_exported_model(source_model)
    exported_model.eval()

    image_size = int(config.get("image_size", 64))
    input_channels = int(model_info["input_channels"])
    example = torch.zeros((1, input_channels, image_size, image_size), dtype=torch.float32)

    parity_inputs = build_parity_inputs(
        kind=kind,
        config=config,
        count=int(args.parity_samples),
        seed=int(args.seed),
    )
    torch_parity = compare_torch_models(
        source_model,
        exported_model,
        parity_inputs,
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    print_parity("escnn -> exported PyTorch", torch_parity)
    if not torch_parity["allclose"] or torch_parity["prediction_mismatches"] != 0:
        raise RuntimeError("Exported PyTorch model does not match the escnn source model")

    require_onnx_dependencies()
    torch.onnx.export(
        exported_model,
        example,
        str(output_path),
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes=(
            None
            if bool(args.fixed_batch)
            else {
                "images": {0: "batch"},
                "logits": {0: "batch"},
            }
        ),
    )

    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    onnx_parity = compare_onnx_runtime(
        source_model,
        output_path,
        parity_inputs,
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    print_parity("escnn -> ONNX Runtime", onnx_parity)
    if not onnx_parity["allclose"] or onnx_parity["prediction_mismatches"] != 0:
        raise RuntimeError("ONNX Runtime output does not match the escnn source model")

    metadata_payload = build_metadata(
        kind=kind,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        config=config,
        model_info=model_info,
        output_path=output_path,
        opset=int(args.opset),
        fixed_batch=bool(args.fixed_batch),
        torch_parity=torch_parity,
        onnx_parity=onnx_parity,
    )
    metadata_path.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"ONNX:     {output_path}")
    print(f"metadata: {metadata_path}")
    print(f"sha256:   {metadata_payload['onnx']['sha256']}")


def resolve_kind(requested: str, config: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    model_config = config.get("model", {})
    model_name = str(model_config.get("name", "")) if isinstance(model_config, dict) else ""
    if model_name == "c8_red_five" or "input_mode" in config:
        return "red-five"
    if model_name == "c8" or "class_labels" in config:
        return "tile-shape"
    raise ValueError("Could not infer classifier kind from checkpoint config; pass --kind")


def build_source_model(
    kind: str,
    config: dict[str, Any],
    state_dict: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        model_config = {}
    fields = tuple(int(value) for value in model_config.get("c8_fields", (8, 16, 32, 64)))

    if kind == "tile-shape":
        if str(model_config.get("name", "c8")) != "c8":
            raise ValueError("Only C8 tile-shape checkpoints are supported")
        labels = config.get("class_labels")
        if isinstance(labels, list) and labels:
            class_labels = [str(value) for value in labels]
            class_count = len(class_labels)
        else:
            class_count = infer_output_classes(state_dict)
            class_labels = [str(index) for index in range(class_count)]
        model = build_tile_shape_model(
            "c8",
            class_count=class_count,
            c8_fields=fields,
        )
        return model, {
            "input_channels": 1,
            "output_classes": class_count,
            "class_labels": class_labels,
            "input_mode": "grayscale",
            "c8_fields": list(fields),
        }

    if kind == "red-five":
        input_mode = str(config.get("input_mode", model_config.get("input_mode", "rgb")))
        if input_mode not in INPUT_CHANNELS:
            raise ValueError(f"Unsupported red-five input mode in checkpoint: {input_mode!r}")
        model = build_red_five_model(input_mode, c8_fields=fields)
        return model, {
            "input_channels": int(INPUT_CHANNELS[input_mode]),
            "output_classes": 2,
            "class_labels": ["normal", "red"],
            "input_mode": input_mode,
            "c8_fields": list(fields),
        }

    raise ValueError(f"Unsupported kind: {kind}")


def infer_output_classes(state_dict: dict[str, Any]) -> int:
    candidate = state_dict.get("classifier.3.weight")
    if isinstance(candidate, torch.Tensor) and candidate.ndim == 2:
        return int(candidate.shape[0])
    raise ValueError("Could not infer tile-shape class count from checkpoint")


def build_exported_model(source_model: nn.Module) -> ExportedC8Classifier:
    backbone = getattr(source_model, "equivariant_backbone", None)
    spatial_pool = getattr(source_model, "spatial_pool", None)
    classifier = getattr(source_model, "classifier", None)
    group_pool = getattr(source_model, "group_pool", None)
    if backbone is None or spatial_pool is None or classifier is None or group_pool is None:
        raise TypeError("Checkpoint model does not have the expected C8 classifier structure")

    source_model.eval()
    # escnn SequentialModule.export() replaces R2Conv/InnerBatchNorm/ReLU/
    # PointwiseMaxPool with ordinary Tensor-only PyTorch modules.
    exported_backbone = backbone.export()
    invariant_channels = int(group_pool.out_type.size)
    final_field_count = int(getattr(source_model, "field_counts")[-1])
    if invariant_channels != final_field_count:
        raise ValueError(
            "Unexpected GroupPooling contract: invariant channel count does not match "
            f"final field count ({invariant_channels} != {final_field_count})"
        )

    return ExportedC8Classifier(
        backbone=exported_backbone,
        spatial_pool=spatial_pool,
        classifier=classifier,
        group_size=C8_GROUP_SIZE,
    )


def build_parity_inputs(
    *,
    kind: str,
    config: dict[str, Any],
    count: int,
    seed: int,
) -> torch.Tensor:
    image_size = int(config.get("image_size", 64))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if kind == "tile-shape":
        gray_u8 = torch.randint(
            0,
            256,
            (count, 1, image_size, image_size),
            dtype=torch.uint8,
            generator=generator,
        )
        images = gray_u8.float().mul_(1.0 / 255.0)
        normalization = require_normalization(config, channels=1)
        return normalize(images, *normalization)

    input_mode = str(config.get("input_mode", "rgb"))
    rgb_u8 = torch.randint(
        0,
        256,
        (count, image_size, image_size, 3),
        dtype=torch.uint8,
        generator=generator,
    )
    images = rgb_to_red_five_input(rgb_u8, input_mode=input_mode)
    normalization = require_normalization(config, channels=images.shape[1])
    return normalize(images, *normalization)


def require_normalization(
    config: dict[str, Any],
    *,
    channels: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    normalization = config.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Checkpoint config is missing normalization")
    mean_value = normalization.get("mean")
    std_value = normalization.get("std")
    mean = to_float_tuple(mean_value)
    std = to_float_tuple(std_value)
    if len(mean) == 1 and channels == 1:
        pass
    elif len(mean) != channels:
        raise ValueError(f"Normalization mean has {len(mean)} channels; expected {channels}")
    if len(std) == 1 and channels == 1:
        pass
    elif len(std) != channels:
        raise ValueError(f"Normalization std has {len(std)} channels; expected {channels}")
    if any(value <= 0.0 for value in std):
        raise ValueError("Normalization std must be positive")
    return mean, std


def to_float_tuple(value: Any) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(float(item) for item in value)
    raise ValueError(f"Expected scalar or sequence normalization value, got {value!r}")


def normalize(
    images: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    mean_tensor = images.new_tensor(mean).view(1, -1, 1, 1)
    std_tensor = images.new_tensor(std).view(1, -1, 1, 1)
    return images.sub(mean_tensor).div(std_tensor)


def rgb_to_red_five_input(images_u8: torch.Tensor, *, input_mode: str) -> torch.Tensor:
    if images_u8.ndim != 4 or images_u8.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB uint8 tensor, got {tuple(images_u8.shape)}")
    rgb = images_u8.permute(0, 3, 1, 2).float().mul_(1.0 / 255.0)
    if input_mode == "rgb":
        return rgb
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (0.5 * r - 0.418688 * g - 0.081312 * b + 0.5).clamp(0.0, 1.0)
    if input_mode == "cr":
        return cr
    if input_mode == "ycr":
        return torch.cat((y, cr), dim=1)
    raise ValueError(f"Unsupported input mode: {input_mode}")


def compare_torch_models(
    source_model: nn.Module,
    exported_model: nn.Module,
    inputs: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    with torch.inference_mode():
        source = source_model(inputs).detach().cpu()
        exported = exported_model(inputs).detach().cpu()
    return compare_arrays(
        source.numpy(),
        exported.numpy(),
        atol=atol,
        rtol=rtol,
    )


def compare_onnx_runtime(
    source_model: nn.Module,
    output_path: Path,
    inputs: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Expected exactly one ONNX input and one ONNX output")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    with torch.inference_mode():
        source = source_model(inputs).detach().cpu().numpy()
    observed = session.run([output_name], {input_name: inputs.numpy()})[0]
    return compare_arrays(source, observed, atol=atol, rtol=rtol)


def compare_arrays(
    expected: np.ndarray,
    observed: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    expected = np.asarray(expected, dtype=np.float32)
    observed = np.asarray(observed, dtype=np.float32)
    if expected.shape != observed.shape:
        return {
            "allclose": False,
            "expected_shape": list(expected.shape),
            "observed_shape": list(observed.shape),
            "max_abs_error": None,
            "mean_abs_error": None,
            "prediction_mismatches": None,
        }
    difference = np.abs(expected - observed)
    prediction_mismatches = int(
        np.count_nonzero(expected.argmax(axis=1) != observed.argmax(axis=1))
    )
    return {
        "allclose": bool(np.allclose(expected, observed, atol=atol, rtol=rtol)),
        "expected_shape": list(expected.shape),
        "observed_shape": list(observed.shape),
        "max_abs_error": float(difference.max()) if difference.size else 0.0,
        "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
        "prediction_mismatches": prediction_mismatches,
        "sample_count": int(expected.shape[0]),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def print_parity(label: str, result: dict[str, Any]) -> None:
    print(
        f"[{label}] allclose={result['allclose']} "
        f"max_abs={result['max_abs_error']} "
        f"mean_abs={result['mean_abs_error']} "
        f"prediction_mismatches={result['prediction_mismatches']}"
    )


def require_onnx_dependencies() -> None:
    missing: list[str] = []
    for package_name in ("onnx", "onnxruntime"):
        try:
            importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            missing.append(package_name)
    if missing:
        raise RuntimeError(
            "Missing ONNX export dependencies: "
            + ", ".join(missing)
            + ". Install them into the existing classifier environment without replacing "
            "the CUDA-specific torch build, e.g. `uv pip install onnx onnxruntime`."
        )


def build_metadata(
    *,
    kind: str,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    model_info: dict[str, Any],
    output_path: Path,
    opset: int,
    fixed_batch: bool,
    torch_parity: dict[str, Any],
    onnx_parity: dict[str, Any],
) -> dict[str, Any]:
    normalization = config.get("normalization", {})
    return {
        "schema_version": 1,
        "kind": kind,
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "sha256": sha256_file(checkpoint_path),
        },
        "input": {
            "name": "images",
            "dtype": "float32",
            "shape": [
                1 if fixed_batch else "batch",
                int(model_info["input_channels"]),
                int(config.get("image_size", 64)),
                int(config.get("image_size", 64)),
            ],
            "input_mode": model_info["input_mode"],
            "normalization": {
                "mean": normalization.get("mean"),
                "std": normalization.get("std"),
            },
            "preprocessing": preprocessing_description(kind, config),
        },
        "output": {
            "name": "logits",
            "dtype": "float32",
            "shape": [1 if fixed_batch else "batch", int(model_info["output_classes"])],
            "class_labels": model_info["class_labels"],
        },
        "model": {
            "c8_fields": model_info["c8_fields"],
            "group_size": C8_GROUP_SIZE,
        },
        "onnx": {
            "path": str(output_path),
            "opset": opset,
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
        },
        "runtime_versions": {
            "torch": torch.__version__,
            "escnn": package_version("escnn"),
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime"),
        },
        "parity": {
            "escnn_to_exported_torch": torch_parity,
            "escnn_to_onnxruntime_cpu": onnx_parity,
        },
    }


def preprocessing_description(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    if kind == "tile-shape":
        return {
            "source": "64x64 grayscale uint8 letterboxed crop from the classifier dataset/runtime crop policy",
            "steps": [
                "convert uint8 grayscale to float32 in [0,1]",
                "normalize per checkpoint mean/std",
            ],
        }
    input_mode = str(config.get("input_mode", "rgb"))
    if input_mode == "rgb":
        conversion = "RGB channels scaled to [0,1]"
    elif input_mode == "cr":
        conversion = "Cr=0.5R-0.418688G-0.081312B+0.5, clipped to [0,1]"
    else:
        conversion = (
            "Y=0.299R+0.587G+0.114B and "
            "Cr=0.5R-0.418688G-0.081312B+0.5, Cr clipped to [0,1]"
        )
    return {
        "source": "64x64 RGB uint8 crop",
        "steps": [conversion, "normalize each model-input channel by checkpoint mean/std"],
    }


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
