from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from mldb_v2.src.training.canonical_weights import _serialize_canonical_state_dict
from mldb_v2.src.training.model import (
    Model,
    _load_model_runtime_from_weight_bytes,
    _resolve_model_lineage,
    _validate_model,
)
from mldb_v2.src.training.training_result import (
    TrainingResult,
    TrainingResultPayload,
    _validate_training_result,
)


MODEL_ID = "demo/run-abc-trial-0001-model"
TRAINING_RESULT_ID = "demo/run-abc-trial-0001-train"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_root(tmp_path: Path, *, architecture_task: str = "demo/task-v1") -> Path:
    root = tmp_path / "mldb_data"
    _write_json(
        root / "demo" / "namespace.yaml",
        {
            "schema": "mjtensu.mldb-v2/namespace/v1",
            "id": "demo",
            "name": "Demo",
            "description": "",
        },
    )
    _write_json(
        root / "demo" / "tasks" / "task-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/task/v1",
            "id": "demo/task-v1",
            "status": "draft",
            "name": "Task",
            "problem_type": "regression",
            "description": "",
            "input": {},
            "target": {"type": "continuous"},
            "semantics": {},
            "scope": {},
        },
    )
    architecture_dir = root / "demo" / "architectures"
    _write_json(
        architecture_dir / "linear-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/architecture/v1",
            "id": "demo/linear-v1",
            "status": "draft",
            "task": architecture_task,
            "name": "Linear",
            "family": "toy",
            "description": "",
            "implementation": {"framework": "pytorch", "entrypoint": "build"},
            "interface": {"input": {"kind": "tensor"}, "output": {"kind": "tensor"}},
            "structure": {"summary": "linear"},
        },
    )
    architecture_dir.mkdir(parents=True, exist_ok=True)
    (architecture_dir / "linear-v1.py").write_text(
        "import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n",
        encoding="utf-8",
    )
    return root


def _weight_ref(data: bytes) -> dict[str, object]:
    return {
        "uri": "s3://mldb-artifacts/demo/model.pt",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "format": "pytorch-state-dict/v1",
    }


def _training_result(weight_data: bytes, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "mjtensu.mldb-v2/training-result/v1",
        "id": TRAINING_RESULT_ID,
        "study_result": "demo/run-abc",
        "plan": "demo/plan-v1",
        "trial": "trial-0001",
        "task": "demo/task-v1",
        "architecture": "demo/linear-v1",
        "corpus": "demo/corpus-v1",
        "train_protocol": "demo/train-v1",
        "parameters": {"epochs": 1},
        "seed": 42,
        "source_commit": "a" * 40,
        "attempts": [{
            "backend": "fake", "execution_id": "attempt-1", "status": "completed",
            "started_at": "2026-09-13T00:00:00Z", "ended_at": "2026-09-13T00:00:01Z",
            "diagnostic": None,
        }],
        "status": "completed", "diagnostic": None,
        "result": {"weights": _weight_ref(weight_data), "model": MODEL_ID},
    }
    value.update(overrides)
    return value


def _install_lineage(
    root: Path,
    weight_data: bytes,
    *,
    model: dict[str, object] | None = None,
    training_result: dict[str, object] | None = None,
) -> None:
    _write_json(
        root / "demo" / "models" / "run-abc-trial-0001-model.yaml",
        model or {
            "schema": "mjtensu.mldb-v2/model/v1",
            "id": MODEL_ID,
            "training_result": TRAINING_RESULT_ID,
        },
    )
    _write_json(
        root / "demo" / "training_results" / "run-abc-trial-0001-train.yaml",
        training_result or _training_result(weight_data),
    )


def test_frozen_public_shapes_are_exact() -> None:
    assert Model.__required_keys__ == frozenset({"schema", "id", "training_result"})
    assert TrainingResultPayload.__required_keys__ == frozenset({"weights", "model"})
    assert TrainingResult.__required_keys__ == frozenset({
        "schema", "id", "study_result", "plan", "trial", "task", "architecture",
        "corpus", "train_protocol", "parameters", "seed", "source_commit", "attempts",
        "status", "diagnostic", "result",
    })


def test_valid_completed_model_lineage_and_runtime_load(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    source = torch.nn.Linear(3, 2)
    weight_data = _serialize_canonical_state_dict(source.state_dict())
    _install_lineage(root, weight_data)

    lineage = _resolve_model_lineage(root, MODEL_ID)
    assert lineage.model["id"] == MODEL_ID
    assert lineage.training_result["id"] == TRAINING_RESULT_ID
    assert lineage.architecture["id"] == "demo/linear-v1"
    assert lineage.task["id"] == "demo/task-v1"
    loaded = _load_model_runtime_from_weight_bytes(root, MODEL_ID, weight_data)
    for key, tensor in source.state_dict().items():
        assert torch.equal(loaded.module.state_dict()[key], tensor)


def test_model_exact_shape_and_canonical_path_identity() -> None:
    valid = {"schema": "mjtensu.mldb-v2/model/v1", "id": MODEL_ID, "training_result": TRAINING_RESULT_ID}
    assert _validate_model(valid, expected_id=MODEL_ID)["id"] == MODEL_ID
    with pytest.raises(ValueError, match="fields"):
        _validate_model({**valid, "extra": 1}, expected_id=MODEL_ID)
    with pytest.raises(ValueError, match="path identity"):
        _validate_model(valid, expected_id="demo/other-model")


def test_non_completed_training_result_is_rejected_by_lineage(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    weight_data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    failed = _training_result(
        weight_data,
        status="failed",
        diagnostic={"code": "train_failed", "message": "failed"},
        result=None,
    )
    _install_lineage(root, weight_data, training_result=failed)
    with pytest.raises(ValueError, match="completed TrainingResult"):
        _resolve_model_lineage(root, MODEL_ID)


def test_model_result_model_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    weight_data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    result = _training_result(weight_data)
    result["result"] = {"weights": _weight_ref(weight_data), "model": "demo/other-model"}
    _install_lineage(root, weight_data, training_result=result)
    with pytest.raises(ValueError, match="does not match Model id"):
        _resolve_model_lineage(root, MODEL_ID)


def test_architecture_task_lineage_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _make_root(tmp_path, architecture_task="demo/other-task-v1")
    weight_data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    _install_lineage(root, weight_data)
    with pytest.raises(ValueError, match="Architecture task"):
        _resolve_model_lineage(root, MODEL_ID)


def test_invalid_weight_format_ref_and_seed_bool_are_rejected() -> None:
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    bad_format = _training_result(data)
    bad_format["result"] = {
        "weights": {**_weight_ref(data), "format": "checkpoint/v1"},
        "model": MODEL_ID,
    }
    with pytest.raises(ValueError, match="format"):
        _validate_training_result(bad_format, expected_id=TRAINING_RESULT_ID)

    bad_ref = _training_result(data)
    bad_ref["result"] = {
        "weights": {"uri": "file:///tmp/model.pt", "bytes": len(data), "sha256": "0" * 64,
                    "format": "pytorch-state-dict/v1"},
        "model": MODEL_ID,
    }
    with pytest.raises(ValueError):
        _validate_training_result(bad_ref, expected_id=TRAINING_RESULT_ID)

    with pytest.raises(ValueError, match="seed"):
        _validate_training_result(_training_result(data, seed=True), expected_id=TRAINING_RESULT_ID)


def test_lineage_never_guesses_sibling_identity(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    weight_data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    result = _training_result(weight_data, task="demo/missing-task-v1")
    _install_lineage(root, weight_data, training_result=result)
    _write_json(
        root / "demo" / "tasks" / "similar-task-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/task/v1", "id": "demo/similar-task-v1",
            "status": "draft", "name": "Similar", "problem_type": "regression",
            "description": "", "input": {}, "target": {"type": "continuous"},
            "semantics": {}, "scope": {},
        },
    )
    with pytest.raises(FileNotFoundError, match="missing-task-v1"):
        _resolve_model_lineage(root, MODEL_ID)


def test_supplied_weight_bytes_must_match_canonical_artifact_ref(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    expected = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    _install_lineage(root, expected)
    different = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    with pytest.raises(ValueError, match="sha256|byte length"):
        _load_model_runtime_from_weight_bytes(root, MODEL_ID, different)
