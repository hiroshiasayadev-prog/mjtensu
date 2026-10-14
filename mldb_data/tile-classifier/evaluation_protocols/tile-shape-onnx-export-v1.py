"""Export one canonical tile classifier Model to ONNX with CPU parity (no latency)."""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import torch
from torch import nn
from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

_IMAGE_SHAPE = (1, 64, 64)
_OPSET_VERSION = 16
_PARITY_SAMPLES = 4
_PARITY_ATOL = 1e-4
_PARITY_RTOL = 1e-4


class _CyclicGroupMaxPool(nn.Module):
    def __init__(self, group_size: int) -> None:
        super().__init__()
        self.group_size = int(group_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim != 4:
            raise ValueError(f"Expected NCHW tensor, got {tuple(input_tensor.shape)}")
        batch, channels, height, width = input_tensor.shape
        if channels % self.group_size != 0:
            raise ValueError(
                f"Channel count {channels} is not divisible by cyclic group size {self.group_size}"
            )
        fields = channels // self.group_size
        grouped = input_tensor.reshape(
            batch, fields, self.group_size, height, width
        )
        return grouped.amax(dim=2)

class _ExportedCyclicClassifier(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        spatial_pool: nn.Module,
        classifier: nn.Module,
        group_size: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.group_pool = _CyclicGroupMaxPool(group_size)
        self.spatial_pool = spatial_pool
        self.classifier = classifier

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        invariant = self.group_pool(features)
        pooled = self.spatial_pool(invariant)
        return self.classifier(pooled)

def _prepare_exportable_model(
    model: nn.Module,
) -> tuple[nn.Module, str]:
    source = model.to("cpu").eval()
    backbone = getattr(source, "equivariant_backbone", None)
    group_pool = getattr(source, "group_pool", None)
    spatial_pool = getattr(source, "spatial_pool", None)
    classifier = getattr(source, "classifier", None)
    field_counts = getattr(source, "field_counts", None)

    cyclic_parts = (backbone, group_pool, spatial_pool, classifier, field_counts)
    if all(part is not None for part in cyclic_parts):
        export = getattr(backbone, "export", None)
        in_type = getattr(group_pool, "in_type", None)
        out_type = getattr(group_pool, "out_type", None)
        in_size = getattr(in_type, "size", None)
        out_size = getattr(out_type, "size", None)
        if not callable(export) or not isinstance(field_counts, tuple):
            raise TypeError("Cyclic classifier export contract is malformed")
        if in_size is None or out_size is None:
            raise TypeError("Cyclic GroupPooling type sizes are unavailable")
        tensor_channels = int(in_size)
        invariant_channels = int(out_size)
        if invariant_channels != int(field_counts[-1]):
            raise ValueError(
                "Cyclic GroupPooling output does not match final field count"
            )
        if invariant_channels < 1 or tensor_channels % invariant_channels != 0:
            raise ValueError(
                "Cyclic GroupPooling input width is not an integer multiple of output width"
            )
        group_size = tensor_channels // invariant_channels
        if group_size < 2:
            raise ValueError("Cyclic group size must be at least 2")
        return (
            _ExportedCyclicClassifier(
                backbone=export(),
                spatial_pool=spatial_pool,
                classifier=classifier,
                group_size=group_size,
            ).eval(),
            "cyclic-equivariant-tensor-export",
        )

    return source, "direct-torch-export"

def _parity_input() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    return torch.rand(
        (_PARITY_SAMPLES, *_IMAGE_SHAPE),
        generator=generator,
        dtype=torch.float32,
    )

def _compare_torch_models(
    source: nn.Module,
    exported: nn.Module,
) -> dict[str, object]:
    inputs = _parity_input()
    with torch.no_grad():
        expected = source(inputs)
        actual = exported(inputs)
    delta = (expected - actual).abs()
    allclose = bool(
        torch.allclose(
            expected,
            actual,
            atol=_PARITY_ATOL,
            rtol=_PARITY_RTOL,
        )
    )
    prediction_mismatches = int(
        (expected.argmax(dim=1) != actual.argmax(dim=1)).sum().item()
    )
    if not allclose or prediction_mismatches:
        raise RuntimeError("Tensor-only export does not match source model")
    return {
        "samples": _PARITY_SAMPLES,
        "atol": _PARITY_ATOL,
        "rtol": _PARITY_RTOL,
        "allclose": allclose,
        "prediction_mismatches": prediction_mismatches,
        "max_abs_error": float(delta.max().item()),
    }

def _export_onnx(
    model: nn.Module,
    path: Path,
    *,
    batch_size: int,
) -> tuple[nn.Module, dict[str, object]]:
    source = model.to("cpu").eval()
    export_model, mode = _prepare_exportable_model(source)
    torch_parity = _compare_torch_models(source, export_model)

    example = torch.zeros((batch_size, *_IMAGE_SHAPE), dtype=torch.float32)
    kwargs: dict[str, object] = {
        "export_params": True,
        "opset_version": _OPSET_VERSION,
        "do_constant_folding": True,
        "input_names": ["images"],
        "output_names": ["logits"],
        "dynamic_axes": {"images": {0: "batch"}, "logits": {0: "batch"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(export_model, example, str(path), **kwargs)
    return source, {"mode": mode, "torch_parity": torch_parity}

def _onnx_parity(
    source: nn.Module,
    session,
    input_name: str,
) -> dict[str, object]:
    inputs = _parity_input()
    with torch.no_grad():
        expected = source(inputs)
    actual_array = session.run(None, {input_name: inputs.numpy()})[0]
    actual = torch.from_numpy(actual_array)
    delta = (expected - actual).abs()
    allclose = bool(
        torch.allclose(
            expected,
            actual,
            atol=_PARITY_ATOL,
            rtol=_PARITY_RTOL,
        )
    )
    prediction_mismatches = int(
        (expected.argmax(dim=1) != actual.argmax(dim=1)).sum().item()
    )
    if not allclose or prediction_mismatches:
        raise RuntimeError("ONNX Runtime output does not match source model")
    return {
        "samples": _PARITY_SAMPLES,
        "atol": _PARITY_ATOL,
        "rtol": _PARITY_RTOL,
        "allclose": allclose,
        "prediction_mismatches": prediction_mismatches,
        "max_abs_error": float(delta.max().item()),
    }

def evaluate(context):
    if context.model is None:
        raise ValueError("ONNX Export requires canonical PyTorch Model")
    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "model.onnx"
    source_model, export_info = _export_onnx(
        context.model.module, path, batch_size=1
    )
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("CPU-only ONNX parity provider required")
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("ONNX model must have one input and one output")
    parity = _onnx_parity(source_model, session, session.get_inputs()[0].name)
    data = path.read_bytes()
    report = {
        "schema": "mjtensu.recognition/tile-onnx-export/v1",
        "model": context.model.definition["id"],
        "onnx_sha256": hashlib.sha256(data).hexdigest(),
        "onnx_bytes": len(data),
        "opset": _OPSET_VERSION,
        "torch_export": export_info,
        "onnx_cpu_parity": parity,
    }
    report_path = work_dir / "onnx-export-report.json"
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    return EvaluationCandidate(
        metrics={"onnx_parity_max_abs_error": parity["max_abs_error"]},
        artifacts={"onnx_model": path, "export_report": report_path},
    )
