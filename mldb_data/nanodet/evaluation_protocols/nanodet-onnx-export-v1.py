"""Export accepted NanoDet Model into a verified, immutable ONNX Evaluation artifact."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate


def evaluate(context):
    if context.model is None:
        raise ValueError("ONNX exporter requires a PyTorch source Model")
    model = context.model.module.to("cpu").eval()
    sample = torch.zeros((1, 3, 320, 320), dtype=torch.float32)
    with torch.no_grad():
        expected = model(sample)
    if expected.ndim != 3 or expected.shape[0] != 1 or expected.shape[-1] != 33:
        raise ValueError("NanoDet model output must be [1, N, 33]")
    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_path = work_dir / "detector.onnx"
    options = {
        "opset_version": 16,
        "input_names": ["images"],
        "output_names": ["dense_detections"],
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    torch.onnx.export(model, sample, str(model_path), **options)

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(model_path)))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("exported ONNX model has an invalid input/output contract")
    actual = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
    reference = expected.detach().cpu().numpy()
    if actual.shape != reference.shape:
        raise ValueError("exported ONNX output shape differs from source model")
    diff = np.abs(reference - actual)
    max_abs = float(np.max(diff))
    if not np.allclose(reference, actual, atol=1e-4, rtol=1e-4):
        raise ValueError(f"exported NanoDet ONNX parity mismatch: {max_abs}")
    data = model_path.read_bytes()
    report = {
        "schema": "mjtensu.nanodet/onnx-export-report/v1",
        "model_id": context.model.definition["id"],
        "onnx_sha256": hashlib.sha256(data).hexdigest(),
        "onnx_bytes": len(data),
        "opset_version": 16,
        "input_shape": [1, 3, 320, 320],
        "output_shape": list(actual.shape),
        "onnx_parity_max_abs_error": max_abs,
    }
    report_path = work_dir / "onnx-export-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return EvaluationCandidate(
        metrics={"onnx_parity_max_abs_error": max_abs},
        artifacts={"onnx_model": model_path, "export_report": report_path},
    )
