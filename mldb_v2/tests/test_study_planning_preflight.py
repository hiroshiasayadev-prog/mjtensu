from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mldb_v2.src.common.ids import DefinitionKind
from mldb_v2.src.study._planning_preflight import (
    _ExistingModelPlanningInput,
    _PlanningPreflightError,
    _StudyPlanningPreflight,
    _TrainingPlanningInput,
)
from mldb_v2.src.study._study_validation import _load_study_definition
from mldb_v2.src.verification._definition_lifecycle import _RepositoryDefinitionVerifier


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _namespace(root: Path, namespace: str) -> None:
    _write(
        root / namespace / "namespace.yaml",
        {
            "schema": "mjtensu.mldb-v2/namespace/v1",
            "id": namespace,
            "name": namespace,
            "description": "",
        },
    )


def _task(entity_id: str, *, status: str = "sealed") -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/task/v1",
        "id": entity_id,
        "status": status,
        "name": "Task",
        "problem_type": "classification",
        "description": "",
        "input": {},
        "target": {"type": "categorical", "labels": ["a"]},
        "semantics": {},
        "scope": {},
    }


def _corpus(entity_id: str, *, task: str) -> dict[str, object]:
    local = entity_id.split("/", 1)[1]
    return {
        "schema": "mjtensu.mldb-v2/corpus/v1",
        "id": entity_id,
        "status": "sealed",
        "task": task,
        "description": "",
        "storage": {"root_uri": f"s3://bucket/{local}"},
        "manifest": {
            "file": f"{local}.manifest.jsonl",
            "sha256": _sha(b""),
            "entries": 0,
        },
        "representation": {"kind": "bytes"},
        "splits": {"train": 0},
    }


def _architecture(entity_id: str, *, task: str) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/architecture/v1",
        "id": entity_id,
        "status": "sealed",
        "task": task,
        "name": entity_id,
        "family": "demo",
        "description": "",
        "implementation": {
            "framework": "pytorch",
            "entrypoint": "build",
            "sha256": "0" * 64,
        },
        "interface": {"input": {"kind": "x"}, "output": {"kind": "y"}},
        "structure": {"summary": "demo"},
    }


def _train_protocol(entity_id: str, *, task: str) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/train-protocol/v1",
        "id": entity_id,
        "status": "sealed",
        "task": task,
        "name": entity_id,
        "description": "",
        "implementation": {"entrypoint": "train", "sha256": "0" * 64},
        "parameters": {
            "zeta": {"default": 9, "type": "integer"},
            "alpha": {"default": 1, "type": "integer"},
            "batch_size": {"default": 32, "type": "integer", "minimum": 1},
        },
    }


def _evaluation_protocol(entity_id: str, *, task: str) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/evaluation-protocol/v1",
        "id": entity_id,
        "status": "sealed",
        "task": task,
        "name": entity_id,
        "description": "",
        "implementation": {"entrypoint": "evaluate", "sha256": "0" * 64},
        "parameters": {
            "zeta_eval": {"default": 0.9, "type": "number"},
            "alpha_eval": {"default": 0.1, "type": "number"},
            "limit": {"default": 10, "type": "integer"},
        },
        "metrics": {"score": {"type": "number", "required": True}},
        "artifacts": {},
    }


def _study(
    *,
    entity_id: str,
    training_corpus: str,
    train_protocol: str,
    architectures: list[str],
    evaluation_pairs: list[tuple[str, str, str]],
    status: str = "sealed",
) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/study/v1",
        "id": entity_id,
        "status": status,
        "name": "Study",
        "description": "",
        "model": {
            "train": {
                "corpus": training_corpus,
                "protocol": train_protocol,
                "architectures": architectures,
                "parameters": {
                    "zeta": {"values": [9, 7]},
                    "alpha": {"values": [2, 1]},
                },
                "seeds": [7, 3],
            }
        },
        "evaluations": [
            {
                "stage": stage,
                "corpus": corpus,
                "protocol": protocol,
                "parameters": {
                    "zeta_eval": {"values": [0.8, 0.7]},
                    "alpha_eval": {"values": [0.2, 0.1]},
                },
            }
            for stage, corpus, protocol in evaluation_pairs
        ],
    }


def _install(root: Path, kind: str, entity_id: str, document: object) -> None:
    namespace, local = entity_id.split("/", 1)
    domain = {
        "task": "tasks",
        "corpus": "corpora",
        "architecture": "architectures",
        "train_protocol": "train_protocols",
        "evaluation_protocol": "evaluation_protocols",
        "study": "studies",
    }[kind]
    path = root / namespace / domain / f"{local}.yaml"
    _write(path, document)
    if kind in {"architecture", "train_protocol", "evaluation_protocol"}:
        path.with_suffix(".py").write_text("# companion\n", encoding="utf-8")
    if kind == "corpus":
        (root / namespace / domain / f"{local}.manifest.jsonl").write_bytes(b"")


class _Verifier:
    def __init__(self, result: object | None = None, error: Exception | None = None) -> None:
        self.result = result if result is not None else {"valid": True, "diagnostics": []}
        self.error = error
        self.calls: list[dict[str, object]] = []

    def verify(self, *, request):
        self.calls.append(dict(request))
        if self.error is not None:
            raise self.error
        return self.result


class _NeverVerifier:
    def verify(self, *, request):
        raise AssertionError("draft Study must fail before W002 verification")



def _training_repo(tmp_path: Path, *, cross_namespace: bool = False) -> tuple[Path, str]:
    root = tmp_path / "mldb_data"
    if cross_namespace:
        ids = {
            "study": "study-ns/study-v1",
            "task": "task-ns/task-v1",
            "train_corpus": "train-data/train-v1",
            "train_protocol": "train-proto/protocol-v1",
            "arch_a": "arch-ns/arch-a-v1",
            "arch_b": "arch-ns/arch-b-v1",
            "eval_corpus_a": "eval-data/corpus-a-v1",
            "eval_corpus_b": "eval-data/corpus-b-v1",
            "eval_protocol_a": "eval-proto/protocol-a-v1",
            "eval_protocol_b": "eval-proto/protocol-b-v1",
        }
    else:
        ids = {
            "study": "demo/study-v1",
            "task": "demo/task-v1",
            "train_corpus": "demo/train-corpus-v1",
            "train_protocol": "demo/train-protocol-v1",
            "arch_a": "demo/arch-a-v1",
            "arch_b": "demo/arch-b-v1",
            "eval_corpus_a": "demo/eval-corpus-a-v1",
            "eval_corpus_b": "demo/eval-corpus-b-v1",
            "eval_protocol_a": "demo/eval-protocol-a-v1",
            "eval_protocol_b": "demo/eval-protocol-b-v1",
        }

    for entity_id in ids.values():
        _namespace(root, entity_id.split("/", 1)[0])
    _install(root, "task", ids["task"], _task(ids["task"]))
    _install(
        root,
        "corpus",
        ids["train_corpus"],
        _corpus(ids["train_corpus"], task=ids["task"]),
    )
    _install(
        root,
        "architecture",
        ids["arch_a"],
        _architecture(ids["arch_a"], task=ids["task"]),
    )
    _install(
        root,
        "architecture",
        ids["arch_b"],
        _architecture(ids["arch_b"], task=ids["task"]),
    )
    _install(
        root,
        "train_protocol",
        ids["train_protocol"],
        _train_protocol(ids["train_protocol"], task=ids["task"]),
    )
    _install(
        root,
        "corpus",
        ids["eval_corpus_a"],
        _corpus(ids["eval_corpus_a"], task=ids["task"]),
    )
    _install(
        root,
        "corpus",
        ids["eval_corpus_b"],
        _corpus(ids["eval_corpus_b"], task=ids["task"]),
    )
    _install(
        root,
        "evaluation_protocol",
        ids["eval_protocol_a"],
        _evaluation_protocol(ids["eval_protocol_a"], task=ids["task"]),
    )
    _install(
        root,
        "evaluation_protocol",
        ids["eval_protocol_b"],
        _evaluation_protocol(ids["eval_protocol_b"], task=ids["task"]),
    )
    study = _study(
        entity_id=ids["study"],
        training_corpus=ids["train_corpus"],
        train_protocol=ids["train_protocol"],
        architectures=[ids["arch_b"], ids["arch_a"]],
        evaluation_pairs=[
            ("stage-z", ids["eval_corpus_b"], ids["eval_protocol_b"]),
            ("stage-a", ids["eval_corpus_a"], ids["eval_protocol_a"]),
        ],
    )
    _install(root, "study", ids["study"], study)
    return root, ids["study"]


def _install_existing_lineage(
    root: Path,
    *,
    model_id: str,
    result_id: str,
    task_id: str,
    architecture_id: str,
) -> None:
    namespace, model_local = model_id.split("/", 1)
    _namespace(root, namespace)
    _write(
        root / namespace / "models" / f"{model_local}.yaml",
        {
            "schema": "mjtensu.mldb-v2/model/v1",
            "id": model_id,
            "training_result": result_id,
        },
    )
    result_namespace, result_local = result_id.split("/", 1)
    _namespace(root, result_namespace)
    _write(
        root / result_namespace / "training_results" / f"{result_local}.yaml",
        {
            "schema": "mjtensu.mldb-v2/training-result/v1",
            "id": result_id,
            "task": task_id,
            "architecture": architecture_id,
            "status": "completed",
        },
    )


class _Integrity:
    def verify_executable_definition(self, *, request):
        return {"valid": True, "diagnostics": []}

    def verify_corpus_builder(self, *, request):
        return {"valid": True, "diagnostics": []}


class _Asset:
    def verify(self, *, request, runner):
        return {
            "valid": True,
            "run": {"collected": 1, "passed": 1, "failed": 0, "errors": 0},
            "diagnostics": [],
        }


class _Runner:
    pass


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_missing_and_malformed_study_are_bounded(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "demo")
    preflight = _StudyPlanningPreflight(mldb_data_root=root, verifier=_Verifier())
    with pytest.raises(_PlanningPreflightError, match="study_unavailable"):
        preflight.prepare("demo/missing-v1")

    path = root / "demo/studies/bad-v1.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-valid", encoding="utf-8")
    with pytest.raises(_PlanningPreflightError, match="study_unavailable"):
        preflight.prepare("demo/bad-v1")


def test_draft_study_rejects_before_verifier(tmp_path: Path) -> None:
    root, study_id = _training_repo(tmp_path)
    path = root / "demo/studies/study-v1.yaml"
    study = json.loads(path.read_text(encoding="utf-8"))
    study["status"] = "draft"
    _write(path, study)
    preflight = _StudyPlanningPreflight(mldb_data_root=root, verifier=_NeverVerifier())
    with pytest.raises(_PlanningPreflightError, match="study_not_sealed"):
        preflight.prepare(study_id)


def test_verifier_invalid_and_exception_are_bounded(tmp_path: Path) -> None:
    root, study_id = _training_repo(tmp_path)
    invalid = _Verifier({"valid": False, "diagnostics": [{"code": "x", "message": "bad"}]})
    with pytest.raises(_PlanningPreflightError, match="study_verification_failed"):
        _StudyPlanningPreflight(mldb_data_root=root, verifier=invalid).prepare(study_id)
    assert invalid.calls == [{"kind": DefinitionKind.STUDY, "id": study_id}]

    noisy_success = _Verifier({"valid": True, "diagnostics": [{"code": "x", "message": "bad"}]})
    with pytest.raises(_PlanningPreflightError, match="study_verification_failed"):
        _StudyPlanningPreflight(mldb_data_root=root, verifier=noisy_success).prepare(study_id)

    raising = _Verifier(error=RuntimeError("backend detail must stay bounded"))
    with pytest.raises(_PlanningPreflightError, match="study_verification_failed"):
        _StudyPlanningPreflight(mldb_data_root=root, verifier=raising).prepare(study_id)


def test_training_preflight_retains_exact_inputs_and_authored_order(tmp_path: Path) -> None:
    root, study_id = _training_repo(tmp_path)
    result = _StudyPlanningPreflight(mldb_data_root=root, verifier=_Verifier()).prepare(study_id)
    assert result.study["id"] == study_id
    assert result.task["id"] == "demo/task-v1"
    assert isinstance(result.model, _TrainingPlanningInput)
    assert result.model.corpus["id"] == "demo/train-corpus-v1"
    assert result.model.protocol["id"] == "demo/train-protocol-v1"
    assert [item["id"] for item in result.model.architectures] == [
        "demo/arch-b-v1",
        "demo/arch-a-v1",
    ]
    assert result.model.seeds == (7, 3)
    assert list(result.model.parameter_axes) == ["zeta", "alpha"]
    assert result.model.parameter_axes["zeta"]["values"] == [9, 7]
    assert list(result.model.parameter_declarations) == ["zeta", "alpha", "batch_size"]
    assert result.model.parameter_declarations["batch_size"]["default"] == 32
    assert "batch_size" not in result.model.parameter_axes

    assert [item.stage["stage"] for item in result.evaluations] == ["stage-z", "stage-a"]
    first = result.evaluations[0]
    assert first.corpus["id"] == "demo/eval-corpus-b-v1"
    assert first.protocol["id"] == "demo/eval-protocol-b-v1"
    assert list(first.parameter_axes) == ["zeta_eval", "alpha_eval"]
    assert first.parameter_axes["zeta_eval"]["values"] == [0.8, 0.7]
    assert first.parameter_declarations["limit"]["default"] == 10
    assert "limit" not in first.parameter_axes


def test_cross_namespace_references_resolve_exactly(tmp_path: Path) -> None:
    root, study_id = _training_repo(tmp_path, cross_namespace=True)
    result = _StudyPlanningPreflight(mldb_data_root=root, verifier=_Verifier()).prepare(study_id)
    assert result.task["id"] == "task-ns/task-v1"
    assert isinstance(result.model, _TrainingPlanningInput)
    assert result.model.corpus["id"] == "train-data/train-v1"
    assert result.model.protocol["id"] == "train-proto/protocol-v1"
    assert [item["id"] for item in result.model.architectures] == [
        "arch-ns/arch-b-v1",
        "arch-ns/arch-a-v1",
    ]
    assert [item.corpus["id"] for item in result.evaluations] == [
        "eval-data/corpus-b-v1",
        "eval-data/corpus-a-v1",
    ]
    assert [item.protocol["id"] for item in result.evaluations] == [
        "eval-proto/protocol-b-v1",
        "eval-proto/protocol-a-v1",
    ]


def test_existing_model_preflight_preserves_model_order_and_lineage(tmp_path: Path) -> None:
    root, study_id = _training_repo(tmp_path)
    _install_existing_lineage(
        root,
        model_id="demo/model-b",
        result_id="demo/result-b",
        task_id="demo/task-v1",
        architecture_id="demo/arch-b-v1",
    )
    _install_existing_lineage(
        root,
        model_id="demo/model-a",
        result_id="demo/result-a",
        task_id="demo/task-v1",
        architecture_id="demo/arch-a-v1",
    )
    study_path = root / "demo/studies/study-v1.yaml"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["model"] = {"existing": ["demo/model-b", "demo/model-a"]}
    _write(study_path, study)

    result = _StudyPlanningPreflight(mldb_data_root=root, verifier=_Verifier()).prepare(study_id)
    assert result.task["id"] == "demo/task-v1"
    assert isinstance(result.model, _ExistingModelPlanningInput)
    assert [entry.model["id"] for entry in result.model.models] == [
        "demo/model-b",
        "demo/model-a",
    ]
    assert [entry.training_result["id"] for entry in result.model.models] == [
        "demo/result-b",
        "demo/result-a",
    ]
    assert [entry.architecture_id for entry in result.model.models] == [
        "demo/arch-b-v1",
        "demo/arch-a-v1",
    ]


def test_post_verification_missing_exact_reference_is_bounded(tmp_path: Path) -> None:
    root, study_id = _training_repo(tmp_path)
    (root / "demo/corpora/train-corpus-v1.yaml").unlink()
    with pytest.raises(_PlanningPreflightError, match="planning_input_resolution_failed"):
        _StudyPlanningPreflight(mldb_data_root=root, verifier=_Verifier()).prepare(study_id)


def test_preflight_is_read_only_and_does_not_execute_companions(tmp_path: Path) -> None:
    root, study_id = _training_repo(tmp_path)
    for path in root.rglob("*.py"):
        path.write_text("raise AssertionError('companion executed')\n", encoding="utf-8")
    before = _tree_bytes(root)
    result = _StudyPlanningPreflight(mldb_data_root=root, verifier=_Verifier()).prepare(study_id)
    assert result.study["id"] == study_id
    assert _tree_bytes(root) == before


def test_private_boundary_contains_no_plan_or_runtime_materialization(tmp_path: Path) -> None:
    root, study_id = _training_repo(tmp_path)
    result = _StudyPlanningPreflight(mldb_data_root=root, verifier=_Verifier()).prepare(study_id)
    assert set(result.__dict__) == {"study", "task", "model", "evaluations"}
    assert isinstance(result.model, _TrainingPlanningInput)
    assert set(result.model.__dict__) == {
        "corpus",
        "protocol",
        "architectures",
        "parameter_axes",
        "seeds",
    }
    source = (
        Path(__file__).resolve().parents[1]
        / "src/study/_planning_preflight.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "torch",
        "PlanPin",
        "from mldb_v2.src.study.plan",
        "source_commit",
        "trial-",
        "eval-",
        "git ",
        "mldb.src",
        "mldb_v2.skeleton",
        "backend",
        "ClearML",
    ):
        assert forbidden not in source


def test_actual_repository_definition_verifier_composition_connects(tmp_path: Path) -> None:
    root, study_id = _training_repo(tmp_path)
    tests_root = tmp_path / "mldb_tests"
    tests_root.mkdir()
    verifier = _RepositoryDefinitionVerifier(
        repository_root=tmp_path,
        mldb_data_root=root,
        mldb_tests_root=tests_root,
        integrity_verifier=_Integrity(),
        asset_test_verifier=_Asset(),
        pytest_runner=_Runner(),
        architecture_interface_loader=lambda root_path, entity_id: object(),
        train_interface_loader=lambda root_path, entity_id: object(),
        evaluation_interface_loader=lambda root_path, entity_id: object(),
    )
    result = _StudyPlanningPreflight(
        mldb_data_root=root,
        verifier=verifier,
    ).prepare(study_id)
    assert result.study["id"] == study_id
    assert result.task["id"] == "demo/task-v1"


def test_current_examples_load_locally_but_preflight_rejects_draft() -> None:
    repo = Path(__file__).resolve().parents[2]
    root = repo / "mldb_data"
    for study_id in (
        "tile-classifier/rotation-robustness-example-v1",
        "rotated-fcos/spatial-screen-example-v1",
    ):
        study = _load_study_definition(root, study_id)
        assert study["status"] == "draft"
        with pytest.raises(_PlanningPreflightError, match="study_not_sealed"):
            _StudyPlanningPreflight(
                mldb_data_root=root,
                verifier=_NeverVerifier(),
            ).prepare(study_id)
