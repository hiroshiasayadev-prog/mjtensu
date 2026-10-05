from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable


ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
PROTOCOL_ID = "mldb-smoke/c8-narrow-late2-fieldwise-dw-pw-onnx-cpu-smoke-v1"


def _context(tmp_path: Path):
    return SimpleNamespace(
        parameters={
            "batch_size": 1,
            "warmup_runs": 1,
            "measure_runs": 10,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
        },
        model=SimpleNamespace(module=None),
        work_dir=tmp_path,
    )


def test_evaluation_callable_contract() -> None:
    assert callable(_load_evaluation_callable(ROOT, PROTOCOL_ID))


def test_grouped_r2conv_export_and_cpu_runtime(tmp_path: Path) -> None:
    pytest.importorskip("escnn")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    evaluate = _load_evaluation_callable(ROOT, PROTOCOL_ID)
    candidate = evaluate(_context(tmp_path))
    report = json.loads(candidate.artifacts["smoke_report"].read_text(encoding="utf-8"))

    assert report["prototype"]["training"] == "none"
    assert report["prototype"]["fields"] == [4, 8, 16, 32]
    assert report["export"]["torch_parity"]["allclose"] is True
    assert report["export"]["onnx_parity"]["allclose"] is True

    expected_signature = [
        {"kernel": [5, 5], "group": 1},
        {"kernel": [3, 3], "group": 1},
        {"kernel": [3, 3], "group": 8},
        {"kernel": [1, 1], "group": 1},
        {"kernel": [3, 3], "group": 16},
        {"kernel": [1, 1], "group": 1},
    ]
    assert report["export"]["conv_kernel_group_signature"] == {
        "expected": expected_signature,
        "torch_export": expected_signature,
        "onnx": expected_signature,
    }
    grouped = {
        node["group"]: node
        for node in report["export"]["onnx_convs"]
        if node["group"] in {8, 16} and node["kernel_shape"] == [3, 3]
    }
    assert grouped[8]["weight_shape"] == [64, 8, 3, 3]
    assert grouped[16]["weight_shape"] == [128, 8, 3, 3]
    assert all(float(value) > 0.0 for value in candidate.metrics.values())
