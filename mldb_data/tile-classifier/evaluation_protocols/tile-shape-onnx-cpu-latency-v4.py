"""Measure an already exported, accepted ONNX artifact using CPUExecutionProvider."""
from __future__ import annotations
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

_1 = 1
_64 = 64
_64 = 64
_DYNAMIC_BATCH = True

def evaluate(context):
    if context.model is not None or not context.inputs or "onnx_model" not in context.inputs:
        raise ValueError("ONNX-only CPU Eval requires a producer ONNX artifact")
    src = context.inputs["onnx_model"]
    parameters = context.parameters
    batch = int(parameters["batch_size"])
    warmup = int(parameters["warmup_runs"])
    runs = int(parameters["measure_runs"])
    intra = int(parameters["intra_op_threads"])
    inter = int(parameters["inter_op_threads"])
    if (batch < 1 or (not _DYNAMIC_BATCH and batch != 1)
            or warmup < 1 or runs < 10 or intra < 1 or inter < 1):
        raise ValueError("invalid ONNX CPU benchmark parameters")

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra
    options.inter_op_num_threads = inter
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        src.data, sess_options=options, providers=["CPUExecutionProvider"]
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("CPUExecutionProvider is required")
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("ONNX must have one input and output")
    name = session.get_inputs()[0].name
    x = np.zeros((batch, _1, _64, _64), dtype=np.float32)
    check = session.run(None, {name: x})[0]
    if (check.ndim != (2 if _DYNAMIC_BATCH else 3)
            or check.shape[0] != batch
            or check.shape[-1] != (35 if _DYNAMIC_BATCH else 33)):
        raise ValueError("ONNX model output dimensions do not match the task")
    for _ in range(warmup):
        session.run(None, {name: x})
    values = []
    for _ in range(runs):
        started = time.perf_counter_ns()
        session.run(None, {name: x})
        values.append((time.perf_counter_ns() - started)/1_000_000)
    values.sort()
    metrics = {
        "latency_p50_ms": float(statistics.median(values)),
        "latency_p95_ms": float(values[max(0,math.ceil(.95 * len(values))-1)]),
        "latency_mean_ms": float(statistics.fmean(values)),
    }
    report = {
        "schema": "mjtensu.recognition/tile-onnx-cpu-artifact-latency/v1",
        "source_evaluation_result": src.source_evaluation_result,
        "source_artifact": src.artifact,
        "onnx_sha256": hashlib.sha256(src.data).hexdigest(),
        "onnx_bytes": len(src.data),
        "input_shape": list(x.shape),
        "output_shape": list(check.shape),
        "metrics_ms": metrics,
        "samples_ms": values,
        "warmup_runs": warmup,
        "environment": {
            "onnxruntime_version": ort.__version__,
            "active_providers": session.get_providers(),
            "intra_op_threads": intra, "inter_op_threads": inter,
            "execution_mode": "ORT_SEQUENTIAL",
            "graph_optimization_level": "ORT_ENABLE_ALL",
        },
        "measurement_scope": "ORT session.run only, excludes source Export, session creation and input allocation",
    }
    output = Path(context.work_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"onnx-cpu-latency.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    return EvaluationCandidate(metrics=metrics, artifacts={"latency_report":path})
