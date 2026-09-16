from __future__ import annotations

import json
from pathlib import Path

import pytest

from mldb_v2.src.api._authoring import AuthoringPlanningService
from mldb_v2.src.api._errors import _ApplicationBoundaryError, _translate_application_error
from mldb_v2.src.common.ids import DefinitionKind
from mldb_v2.tests.test_study_source_pinning import _training_repo


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _namespace(root: Path, namespace: str) -> None:
    _write(root / namespace / "namespace.yaml", {
        "schema": "mjtensu.mldb-v2/namespace/v1", "id": namespace,
        "name": namespace, "description": "",
    })


def _definition(root: Path, kind: DefinitionKind, entity_id: str) -> None:
    domain, schema = {
        DefinitionKind.TASK: ("tasks", "mjtensu.mldb-v2/task/v1"),
        DefinitionKind.CORPUS: ("corpora", "mjtensu.mldb-v2/corpus/v1"),
        DefinitionKind.ARCHITECTURE: ("architectures", "mjtensu.mldb-v2/architecture/v1"),
        DefinitionKind.TRAIN_PROTOCOL: ("train_protocols", "mjtensu.mldb-v2/train-protocol/v1"),
        DefinitionKind.EVALUATION_PROTOCOL: ("evaluation_protocols", "mjtensu.mldb-v2/evaluation-protocol/v1"),
        DefinitionKind.STUDY: ("studies", "mjtensu.mldb-v2/study/v1"),
    }[kind]
    namespace, local = entity_id.split("/", 1)
    _write(root / namespace / domain / f"{local}.yaml", {"schema": schema, "id": entity_id})


class _Check:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set(); self.calls: list[tuple[DefinitionKind, str]] = []
    def validate(self, *, request):
        self.calls.append((request["kind"], request["id"]))
        if request["id"] in self.failures: raise RuntimeError("secret validate failure")
        return {"valid": True, "diagnostics": []}
    def verify(self, *, request):
        self.calls.append((request["kind"], request["id"]))
        if request["id"] in self.failures: raise RuntimeError("secret verify failure")
        return {"valid": True, "diagnostics": []}


class _Sealer:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set(); self.calls: list[tuple[DefinitionKind, str]] = []
    def seal(self, *, request):
        self.calls.append((request["kind"], request["id"]))
        if request["id"] in self.failures:
            raise ValueError("definition lifecycle conflict: injected failure")
        return {"definition": {"id": request["id"], "status": "sealed"}}


def _service(tmp_path: Path, *, validator=None, verifier=None, sealer=None):
    repo = tmp_path / "repo"; root = repo / "mldb_data"; tests = repo / "mldb_tests"
    tests.mkdir(parents=True, exist_ok=True)
    return AuthoringPlanningService(
        repository_root=repo,
        mldb_data_root=root,
        mldb_tests_root=tests,
        validator=validator or _Check(),
        verifier=verifier or _Check(),
        sealer=sealer or _Sealer(),
    ), root


def test_errors_runtime_is_exact_frozen_mirror() -> None:
    repo = Path(__file__).resolve().parents[2]
    runtime = (repo / "mldb_v2/src/api/errors.py").read_text(encoding="utf-8")
    frozen = (repo / "mldb_v2/skeleton/api/errors.py").read_text(encoding="utf-8")
    assert runtime.replace("\r\n", "\n") == frozen.replace("\r\n", "\n")


def test_validate_empty_scope_repository_issues_order_and_continue(tmp_path: Path) -> None:
    validator = _Check({"alpha/z-v1"})
    service, root = _service(tmp_path, validator=validator)
    for namespace in ("beta", "alpha"):
        _namespace(root, namespace)
    _definition(root, DefinitionKind.TASK, "alpha/z-v1")
    _definition(root, DefinitionKind.TASK, "alpha/a-v1")
    _definition(root, DefinitionKind.STUDY, "alpha/study-v1")
    _definition(root, DefinitionKind.TASK, "beta/c-v1")
    (root / "alpha" / "unknown").mkdir()

    report = service.validate_scope()
    assert [(item["kind"], item["id"]) for item in report["items"]] == [
        (DefinitionKind.TASK, "alpha/a-v1"),
        (DefinitionKind.TASK, "alpha/z-v1"),
        (DefinitionKind.STUDY, "alpha/study-v1"),
        (DefinitionKind.TASK, "beta/c-v1"),
    ]
    assert report["items"][1]["valid"] is False
    assert report["items"][2]["valid"] is True
    assert [issue["code"] for issue in report["repository_issues"]] == ["repository_unknown_domain"]


def test_verify_empty_and_scope_narrowing(tmp_path: Path) -> None:
    verifier = _Check()
    service, root = _service(tmp_path, verifier=verifier)
    for namespace in ("alpha", "beta"):
        _namespace(root, namespace)
    for entity_id in ("alpha/a-v1", "alpha/b-v1", "beta/c-v1"):
        _definition(root, DefinitionKind.TASK, entity_id)
    _definition(root, DefinitionKind.STUDY, "alpha/study-v1")

    assert [x["id"] for x in service.verify_scope(scope={"namespace": "alpha"})["items"]] == [
        "alpha/a-v1", "alpha/b-v1", "alpha/study-v1"
    ]
    assert [x["id"] for x in service.verify_scope(scope={"kind": DefinitionKind.TASK})["items"]] == [
        "alpha/a-v1", "alpha/b-v1", "beta/c-v1"
    ]
    exact = service.verify_scope(scope={"kind": DefinitionKind.TASK, "id": "alpha/b-v1"})
    assert [x["id"] for x in exact["items"]] == ["alpha/b-v1"]


def test_invalid_exact_scope_and_missing_exact_are_stable_application_errors(tmp_path: Path) -> None:
    service, root = _service(tmp_path)
    _namespace(root, "alpha")
    _definition(root, DefinitionKind.TASK, "alpha/a-v1")
    with pytest.raises(_ApplicationBoundaryError) as exc:
        service.validate_scope(scope={"id": "alpha/a-v1"})
    assert exc.value.error == {"code": "invalid_request", "message": "exact definition id requires explicit kind"}
    with pytest.raises(_ApplicationBoundaryError) as exc:
        service.validate_scope(scope={"kind": DefinitionKind.TASK, "id": "alpha/missing-v1"})
    assert exc.value.error["code"] == "not_found"
    assert set(exc.value.error) == {"code", "message"}


def test_seal_single_multi_guard_and_bulk_partial_success(tmp_path: Path) -> None:
    sealer = _Sealer({"alpha/b-v1"})
    service, root = _service(tmp_path, sealer=sealer)
    _namespace(root, "alpha")
    for entity_id in ("alpha/a-v1", "alpha/b-v1"):
        _definition(root, DefinitionKind.TASK, entity_id)

    single = service.seal_scope(scope={"kind": DefinitionKind.TASK, "id": "alpha/a-v1"})
    assert single == [{"kind": DefinitionKind.TASK, "id": "alpha/a-v1", "sealed": True, "diagnostics": []}]
    with pytest.raises(_ApplicationBoundaryError) as exc:
        service.seal_scope(scope={"kind": DefinitionKind.TASK})
    assert exc.value.error["code"] == "invalid_request"

    sealer.calls.clear()
    bulk = service.seal_scope(scope={"kind": DefinitionKind.TASK}, bulk=True)
    assert [item["id"] for item in bulk] == ["alpha/a-v1", "alpha/b-v1"]
    assert [item["sealed"] for item in bulk] == [True, False]
    assert sealer.calls == [(DefinitionKind.TASK, "alpha/a-v1"), (DefinitionKind.TASK, "alpha/b-v1")]
    assert bulk[1]["diagnostics"][0]["code"] == "lifecycle_conflict"


def test_seal_omitted_target_rejected(tmp_path: Path) -> None:
    service, _root = _service(tmp_path)
    with pytest.raises(_ApplicationBoundaryError) as exc:
        service.seal_scope(scope={}, bulk=True)
    assert exc.value.error["code"] == "invalid_request"


def _planning_service(repo: Path) -> AuthoringPlanningService:
    return AuthoringPlanningService(
        repository_root=repo,
        mldb_data_root=repo / "mldb_data",
        mldb_tests_root=repo / "mldb_v2" / "tests",
        validator=_Check(),
        verifier=_Check(),
        sealer=_Sealer(),
    )


def test_plan_sealed_study_and_replay_are_idempotent(tmp_path: Path) -> None:
    repo, study_id, _commit, _ = _training_repo(tmp_path)
    service = _planning_service(repo)
    first = service.plan_study(study=study_id)
    second = service.plan_study(study=study_id)
    assert first == second
    assert first["study"] == study_id
    namespace, local = first["id"].split("/", 1)
    paths = list((repo / "mldb_data" / namespace / "study_plans").glob("*.yaml"))
    assert [path.name for path in paths] == [f"{local}.yaml"]


def test_plan_unsealed_study_maps_not_sealed(tmp_path: Path) -> None:
    repo, study_id, _commit, _ = _training_repo(tmp_path)
    namespace, local = study_id.split("/", 1)
    path = repo / "mldb_data" / namespace / "studies" / f"{local}.yaml"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["status"] = "draft"
    _write(path, document)
    with pytest.raises(_ApplicationBoundaryError) as exc:
        _planning_service(repo).plan_study(study=study_id)
    assert exc.value.error["code"] == "not_sealed"


def test_plan_dirty_selected_source_maps_source_not_pinned(tmp_path: Path) -> None:
    repo, study_id, _commit, _sources = _training_repo(tmp_path)
    companion = repo / "mldb_data" / "arch-ns" / "architectures" / "arch-a-v1.py"
    companion.write_bytes(companion.read_bytes() + b"# dirty\n")
    with pytest.raises(_ApplicationBoundaryError) as exc:
        _planning_service(repo).plan_study(study=study_id)
    assert exc.value.error == {
        "code": "source_not_pinned",
        "message": "required committed source inputs are not pinned",
    }


def test_error_translation_is_bounded_and_backend_free() -> None:
    error = _translate_application_error(ValueError("secret credential=abc traceback"))
    assert error == {
        "code": "validation_failed",
        "message": "canonical validation rejected the operation",
    }
    source = (Path(__file__).resolve().parents[1] / "src" / "api" / "_authoring.py").read_text(encoding="utf-8")
    assert "mldb_v2.src.backend" not in source
    assert "study_driver" not in source
