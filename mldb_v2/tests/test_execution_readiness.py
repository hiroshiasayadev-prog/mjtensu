from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from mldb_v2.src.common.ids import _canonical_json_bytes
from mldb_v2.src.results.study_result import _validate_study_result
from mldb_v2.src.study._plan_build import _plan_id, _validate_study_plan
from mldb_v2.src.study.execution_readiness import (
    ExecutionReadinessResolver,
    _build_stage_input,
)


SOURCE_COMMIT = "a" * 40
EXECUTION_KEY = "1234567812344abc8def1234567890ab"
STUDY_RESULT_ID = f"demo/run-{EXECUTION_KEY}"
RUNTIME_TRAINING_RESULT_ID = f"{STUDY_RESULT_ID}-trial-0001-train"
RUNTIME_MODEL_ID = f"{STUDY_RESULT_ID}-trial-0001-model"
EXISTING_TRAINING_RESULT_ID = "demo/existing-training-result-v1"
EXISTING_MODEL_ID = "demo/existing-model-v1"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _pin(
    kind: str,
    entity_id: str,
    *,
    executable: bool = False,
    corpus: bool = False,
) -> dict[str, object]:
    return {
        "kind": kind,
        "id": entity_id,
        "yaml_sha256": "1" * 64,
        "companion_sha256": "2" * 64 if executable else None,
        "sources": [],
        "manifest_sha256": "3" * 64 if corpus else None,
        "manifest_entries": 1 if corpus else None,
    }


def _plan() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "mjtensu.mldb-v2/study-plan/v1",
        "study": "demo/study-v1",
        "source_commit": SOURCE_COMMIT,
        "pins": [
            _pin("namespace", "demo"),
            _pin("task", "demo/task-v1"),
            _pin("corpus", "demo/eval-corpus-v1", corpus=True),
            _pin("corpus", "demo/train-corpus-v1", corpus=True),
            _pin("architecture", "demo/linear-v1", executable=True),
            _pin("train_protocol", "demo/train-v1", executable=True),
            _pin("evaluation_protocol", "demo/eval-v1", executable=True),
            _pin("study", "demo/study-v1"),
            _pin("model", EXISTING_MODEL_ID),
            _pin("training_result", EXISTING_TRAINING_RESULT_ID),
        ],
        "trials": [
            {
                "trial": "trial-0001",
                "source": {
                    "kind": "training",
                    "task": "demo/task-v1",
                    "corpus": "demo/train-corpus-v1",
                    "architecture": "demo/linear-v1",
                    "train_protocol": "demo/train-v1",
                    "parameters": {"epochs": 2, "learning_rate": 0.25},
                    "seed": 42,
                },
                "evaluations": [
                    {
                        "coordinate": "eval-0001",
                        "stage": "holdout-a",
                        "task": "demo/task-v1",
                        "corpus": "demo/eval-corpus-v1",
                        "evaluation_protocol": "demo/eval-v1",
                        "parameters": {"threshold": 0.5},
                    },
                    {
                        "coordinate": "eval-0002",
                        "stage": "holdout-b",
                        "task": "demo/task-v1",
                        "corpus": "demo/eval-corpus-v1",
                        "evaluation_protocol": "demo/eval-v1",
                        "parameters": {"threshold": 0.75},
                    },
                ],
            },
            {
                "trial": "trial-0002",
                "source": {"kind": "existing_model", "model": EXISTING_MODEL_ID},
                "evaluations": [
                    {
                        "coordinate": "eval-0001",
                        "stage": "holdout-a",
                        "task": "demo/task-v1",
                        "corpus": "demo/eval-corpus-v1",
                        "evaluation_protocol": "demo/eval-v1",
                        "parameters": {"threshold": 0.5},
                    },
                    {
                        "coordinate": "eval-0002",
                        "stage": "holdout-b",
                        "task": "demo/task-v1",
                        "corpus": "demo/eval-corpus-v1",
                        "evaluation_protocol": "demo/eval-v1",
                        "parameters": {"threshold": 0.75},
                    },
                ],
            },
        ],
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return dict(
        _validate_study_plan(
            {
                **payload,
                "id": _plan_id("demo/study-v1", digest),
                "content_sha256": digest,
            }
        )
    )


def _pending_evaluation(coordinate: str, stage: str) -> dict[str, object]:
    return {
        "coordinate": coordinate,
        "stage": stage,
        "disposition": "pending",
        "result": None,
        "reason": None,
    }


def _study_result(plan: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/study-result/v1",
        "id": STUDY_RESULT_ID,
        "execution_key": EXECUTION_KEY,
        "plan": plan["id"],
        "study": plan["study"],
        "source_commit": SOURCE_COMMIT,
        "backend": "fake",
        "created_at": "2026-09-13T00:00:00Z",
        "status": "submitted",
        "diagnostic": None,
        "trials": [
            {
                "trial": "trial-0001",
                "training": {
                    "disposition": "pending",
                    "result": None,
                    "reason": None,
                },
                "evaluations": [
                    _pending_evaluation("eval-0001", "holdout-a"),
                    _pending_evaluation("eval-0002", "holdout-b"),
                ],
            },
            {
                "trial": "trial-0002",
                "training": None,
                "evaluations": [
                    _pending_evaluation("eval-0001", "holdout-a"),
                    _pending_evaluation("eval-0002", "holdout-b"),
                ],
            },
        ],
    }


def _attempt() -> dict[str, object]:
    return {
        "backend": "fake",
        "execution_id": "train-1",
        "status": "completed",
        "started_at": "2026-09-13T00:00:00Z",
        "ended_at": "2026-09-13T00:00:01Z",
        "diagnostic": None,
    }


def _weights(uri: str = "s3://bucket/model.pt") -> dict[str, object]:
    return {
        "uri": uri,
        "bytes": 11,
        "sha256": "4" * 64,
        "format": "pytorch-state-dict/v1",
    }


def _training_result(
    *,
    training_result_id: str,
    model_id: str,
    plan_id: str,
    study_result_id: str,
    trial: str,
    source_commit: str,
    seed: int,
) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/training-result/v1",
        "id": training_result_id,
        "study_result": study_result_id,
        "plan": plan_id,
        "trial": trial,
        "task": "demo/task-v1",
        "architecture": "demo/linear-v1",
        "corpus": "demo/train-corpus-v1",
        "train_protocol": "demo/train-v1",
        "parameters": {"epochs": 2, "learning_rate": 0.25} if seed == 42 else {},
        "seed": seed,
        "source_commit": source_commit,
        "attempts": [_attempt()],
        "status": "completed",
        "diagnostic": None,
        "result": {"weights": _weights(), "model": model_id},
    }


def _install_catalog(root: Path) -> None:
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
    _write_json(
        root / "demo" / "architectures" / "linear-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/architecture/v1",
            "id": "demo/linear-v1",
            "status": "draft",
            "task": "demo/task-v1",
            "name": "Linear",
            "family": "toy",
            "description": "",
            "implementation": {"framework": "pytorch", "entrypoint": "build"},
            "interface": {"input": {"kind": "tensor"}, "output": {"kind": "tensor"}},
            "structure": {"summary": "linear"},
        },
    )


def _write_lineage(root: Path, training_result: dict[str, object], model_id: str) -> None:
    training_result_id = str(training_result["id"])
    _write_json(
        root / "demo" / "training_results" / f"{training_result_id.split('/', 1)[1]}.yaml",
        training_result,
    )
    _write_json(
        root / "demo" / "models" / f"{model_id.split('/', 1)[1]}.yaml",
        {
            "schema": "mjtensu.mldb-v2/model/v1",
            "id": model_id,
            "training_result": training_result_id,
        },
    )


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    root = tmp_path / "mldb_data"
    _install_catalog(root)
    plan = _plan()
    result = _study_result(plan)
    existing = _training_result(
        training_result_id=EXISTING_TRAINING_RESULT_ID,
        model_id=EXISTING_MODEL_ID,
        plan_id="demo/legacy-plan-v1",
        study_result_id="demo/legacy-run-v1",
        trial="trial-0001",
        source_commit="b" * 40,
        seed=7,
    )
    _write_lineage(root, existing, EXISTING_MODEL_ID)
    return root, plan, result


def _complete_runtime_training(
    root: Path,
    plan: dict[str, object],
    result: dict[str, object],
) -> None:
    result["trials"][0]["training"] = {
        "disposition": "completed",
        "result": RUNTIME_TRAINING_RESULT_ID,
        "reason": None,
    }
    training = _training_result(
        training_result_id=RUNTIME_TRAINING_RESULT_ID,
        model_id=RUNTIME_MODEL_ID,
        plan_id=str(plan["id"]),
        study_result_id=STUDY_RESULT_ID,
        trial="trial-0001",
        source_commit=SOURCE_COMMIT,
        seed=42,
    )
    _write_lineage(root, training, RUNTIME_MODEL_ID)


def _resolver(root: Path) -> ExecutionReadinessResolver:
    return ExecutionReadinessResolver(mldb_data_root=root)


def test_initial_training_ready_and_training_evaluations_blocked_while_existing_model_is_ready(
    tmp_path: Path,
) -> None:
    root, plan, result = _fixture(tmp_path)
    readiness = _resolver(root).derive(plan=plan, result=result)
    assert readiness == {
        "ready": [
            {"kind": "training", "trial": "trial-0001"},
            {"kind": "evaluation", "trial": "trial-0002", "coordinate": "eval-0001"},
            {"kind": "evaluation", "trial": "trial-0002", "coordinate": "eval-0002"},
        ],
        "skipped": [],
    }


def test_completed_training_with_deterministic_model_releases_all_pending_evaluations(
    tmp_path: Path,
) -> None:
    root, plan, result = _fixture(tmp_path)
    _complete_runtime_training(root, plan, result)
    readiness = _resolver(root).derive(plan=plan, result=result)
    assert readiness["ready"] == [
        {"kind": "evaluation", "trial": "trial-0001", "coordinate": "eval-0001"},
        {"kind": "evaluation", "trial": "trial-0001", "coordinate": "eval-0002"},
        {"kind": "evaluation", "trial": "trial-0002", "coordinate": "eval-0001"},
        {"kind": "evaluation", "trial": "trial-0002", "coordinate": "eval-0002"},
    ]
    assert readiness["skipped"] == []


@pytest.mark.parametrize(
    ("disposition", "reason"),
    [("failed", "upstream_failed"), ("cancelled", "upstream_cancelled")],
)
def test_terminal_training_derives_only_upstream_skips(
    tmp_path: Path,
    disposition: str,
    reason: str,
) -> None:
    root, plan, result = _fixture(tmp_path)
    result["trials"][0]["training"] = {
        "disposition": disposition,
        "result": RUNTIME_TRAINING_RESULT_ID,
        "reason": None,
    }
    readiness = _resolver(root).derive(plan=plan, result=result)
    assert readiness["skipped"] == [
        {
            "stage": {"kind": "evaluation", "trial": "trial-0001", "coordinate": "eval-0001"},
            "reason": reason,
        },
        {
            "stage": {"kind": "evaluation", "trial": "trial-0001", "coordinate": "eval-0002"},
            "reason": reason,
        },
    ]
    assert all(item["trial"] != "trial-0001" for item in readiness["ready"])


@pytest.mark.parametrize("disposition", ["completed", "failed", "skipped"])
def test_terminal_evaluation_sibling_is_not_readied_and_does_not_block_other_sibling(
    tmp_path: Path,
    disposition: str,
) -> None:
    root, plan, result = _fixture(tmp_path)
    _complete_runtime_training(root, plan, result)
    slot = result["trials"][0]["evaluations"][0]
    slot["disposition"] = disposition
    if disposition == "skipped":
        slot["result"] = None
        slot["reason"] = "global_failure"
    else:
        slot["result"] = f"{STUDY_RESULT_ID}-trial-0001-eval-0001"
        slot["reason"] = None
    readiness = _resolver(root).derive(plan=plan, result=result)
    assert {"kind": "evaluation", "trial": "trial-0001", "coordinate": "eval-0001"} not in readiness["ready"]
    assert {"kind": "evaluation", "trial": "trial-0001", "coordinate": "eval-0002"} in readiness["ready"]


def test_cancelling_admits_nothing_and_never_derives_study_cancelled(tmp_path: Path) -> None:
    root, plan, result = _fixture(tmp_path)
    result["status"] = "cancelling"
    readiness = _resolver(root).derive(plan=plan, result=result)
    assert readiness == {"ready": [], "skipped": []}

    result["trials"][0]["training"] = {
        "disposition": "failed",
        "result": RUNTIME_TRAINING_RESULT_ID,
        "reason": None,
    }
    readiness = _resolver(root).derive(plan=plan, result=result)
    assert readiness["ready"] == []
    assert {item["reason"] for item in readiness["skipped"]} == {"upstream_failed"}
    assert "study_cancelled" not in repr(readiness)


def test_invalid_existing_model_or_weight_lineage_is_rejected(tmp_path: Path) -> None:
    root, plan, result = _fixture(tmp_path)
    model_path = root / "demo" / "models" / "existing-model-v1.yaml"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["training_result"] = "demo/missing-training-result-v1"
    _write_json(model_path, model)
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        _resolver(root).derive(plan=plan, result=result)

    root, plan, result = _fixture(tmp_path / "weights")
    training_path = root / "demo" / "training_results" / "existing-training-result-v1.yaml"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["result"]["weights"]["format"] = "checkpoint/v1"
    _write_json(training_path, training)
    with pytest.raises(ValueError, match="weights|format"):
        _resolver(root).derive(plan=plan, result=result)


def test_training_stage_input_is_exact_plan_materialization(tmp_path: Path) -> None:
    root, plan, result = _fixture(tmp_path)
    stage_input = _build_stage_input(
        plan=plan,
        result=result,
        stage={"kind": "training", "trial": "trial-0001"},
        mldb_data_root=root,
    )
    assert stage_input == {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": STUDY_RESULT_ID,
        "plan": plan["id"],
        "plan_sha256": plan["content_sha256"],
        "trial": "trial-0001",
        "kind": "training",
        "coordinate": None,
        "source_commit": SOURCE_COMMIT,
        "pins": plan["pins"],
        "stage": {
            "task": "demo/task-v1",
            "corpus": "demo/train-corpus-v1",
            "architecture": "demo/linear-v1",
            "train_protocol": "demo/train-v1",
            "parameters": {"epochs": 2, "learning_rate": 0.25},
            "seed": 42,
        },
        "runtime_model": None,
    }
    assert stage_input["pins"] is not plan["pins"]


def test_evaluation_stage_inputs_use_same_runtime_model_shape_for_both_model_sources(
    tmp_path: Path,
) -> None:
    root, plan, result = _fixture(tmp_path)
    _complete_runtime_training(root, plan, result)
    runtime_input = _build_stage_input(
        plan=plan,
        result=result,
        stage={"kind": "evaluation", "trial": "trial-0001", "coordinate": "eval-0001"},
        mldb_data_root=root,
    )
    existing_input = _build_stage_input(
        plan=plan,
        result=result,
        stage={"kind": "evaluation", "trial": "trial-0002", "coordinate": "eval-0001"},
        mldb_data_root=root,
    )
    expected_runtime_keys = {"model", "training_result", "task", "architecture", "weights"}
    assert set(runtime_input["runtime_model"]) == expected_runtime_keys
    assert set(existing_input["runtime_model"]) == expected_runtime_keys
    assert runtime_input["runtime_model"]["model"] == RUNTIME_MODEL_ID
    assert runtime_input["runtime_model"]["training_result"] == RUNTIME_TRAINING_RESULT_ID
    assert existing_input["runtime_model"]["model"] == EXISTING_MODEL_ID
    assert existing_input["runtime_model"]["training_result"] == EXISTING_TRAINING_RESULT_ID
    assert runtime_input["stage"] == {
        "name": "holdout-a",
        "task": "demo/task-v1",
        "corpus": "demo/eval-corpus-v1",
        "evaluation_protocol": "demo/eval-v1",
        "parameters": {"threshold": 0.5},
    }
    assert runtime_input["pins"] == plan["pins"]


def test_stage_input_cannot_be_built_for_blocked_or_terminal_stage(tmp_path: Path) -> None:
    root, plan, result = _fixture(tmp_path)
    with pytest.raises(ValueError, match="currently ready"):
        _build_stage_input(
            plan=plan,
            result=result,
            stage={"kind": "evaluation", "trial": "trial-0001", "coordinate": "eval-0001"},
            mldb_data_root=root,
        )


def test_study_result_exact_shape_identity_topology_and_slot_combinations(tmp_path: Path) -> None:
    _root, plan, result = _fixture(tmp_path)
    validated = _validate_study_result(result)
    assert set(validated) == {
        "schema", "id", "execution_key", "plan", "study", "source_commit",
        "backend", "created_at", "status", "diagnostic", "trials",
    }

    bad = copy.deepcopy(result)
    bad["id"] = "demo/run-ffffffffffffffffffffffffffffffff"
    with pytest.raises(ValueError, match="execution_key"):
        _validate_study_result(bad)

    bad = copy.deepcopy(result)
    bad["trials"][1]["trial"] = "trial-0003"
    with pytest.raises(ValueError, match="trial order"):
        _validate_study_result(bad)

    bad = copy.deepcopy(result)
    bad["trials"][0]["evaluations"][0]["coordinate"] = "eval-0002"
    with pytest.raises(ValueError, match="evaluation order"):
        _validate_study_result(bad)

    bad = copy.deepcopy(result)
    bad["trials"][0]["training"] = {
        "disposition": "pending",
        "result": RUNTIME_TRAINING_RESULT_ID,
        "reason": None,
    }
    with pytest.raises(ValueError, match="pending"):
        _validate_study_result(bad)

    bad = copy.deepcopy(result)
    bad["trials"][0]["training"] = {
        "disposition": "skipped",
        "result": None,
        "reason": "unknown_reason",
    }
    with pytest.raises(ValueError, match="stable reason"):
        _validate_study_result(bad)


def test_plan_cross_topology_mismatch_is_rejected_by_readiness(tmp_path: Path) -> None:
    root, plan, result = _fixture(tmp_path)
    result["trials"][1]["training"] = {
        "disposition": "pending",
        "result": None,
        "reason": None,
    }
    with pytest.raises(ValueError, match="training topology"):
        _resolver(root).derive(plan=plan, result=result)


def test_readiness_and_builder_do_not_write_or_import_backend_policy(tmp_path: Path) -> None:
    root, plan, result = _fixture(tmp_path)
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    _resolver(root).derive(plan=plan, result=result)
    _build_stage_input(
        plan=plan,
        result=result,
        stage={"kind": "training", "trial": "trial-0001"},
        mldb_data_root=root,
    )
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    assert after == before

    source_path = Path(__file__).resolve().parents[1] / "src" / "study" / "execution_readiness.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert all(not module.startswith("mldb_v2.skeleton") for module in imports)
    assert all("backend_port" not in module for module in imports)
    assert "clearml" not in source.casefold()
    assert "CanonicalRepositoryWriter" not in source
    assert "create_immutable" not in source
    assert "replace_nonterminal_study_result" not in source
    assert "admit(" not in source
    assert "collect(" not in source
    assert "cancel_study(" not in source
