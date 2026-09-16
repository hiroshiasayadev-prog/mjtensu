from __future__ import annotations

import copy
import hashlib
import json
import uuid
from pathlib import Path

import pytest

import mldb_v2.src.api._execution as execution
from mldb_v2.src.api._errors import _ApplicationBoundaryError
from mldb_v2.src.api.application import Application
from mldb_v2.src.backend._registry import BackendRegistry
from mldb_v2.src.common.ids import EntityKind, _canonical_json_bytes
from mldb_v2.src.repository.canonical_writes import CanonicalRepositoryWriter
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study._plan_build import _plan_id, _validate_study_plan
import mldb_v2.tests.test_execution_readiness as ready_fx

SOURCE_KEY = ready_fx.EXECUTION_KEY
FRESH_UUID_A = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
FRESH_UUID_B = uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class _NoopValidator:
    def validate(self, *, kind, entity_id, document) -> None:
        return None


def _install_plan(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root, plan, _result = ready_fx._fixture(tmp_path)
    local_id = str(plan["id"]).split("/", 1)[1]
    _write_json(root / "demo" / "study_plans" / f"{local_id}.yaml", plan)
    return root, plan


def _alternate_plan(root: Path, plan: dict[str, object]) -> dict[str, object]:
    payload = {
        "schema": plan["schema"],
        "study": plan["study"],
        "source_commit": plan["source_commit"],
        "pins": copy.deepcopy(plan["pins"]),
        "trials": copy.deepcopy(plan["trials"]),
    }
    payload["trials"][0]["evaluations"][0]["parameters"]["threshold"] = 0.125
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    alternate = dict(_validate_study_plan({**payload, "id": _plan_id(str(plan["study"]), digest), "content_sha256": digest}))
    local_id = str(alternate["id"]).split("/", 1)[1]
    _write_json(root / "demo" / "study_plans" / f"{local_id}.yaml", alternate)
    return alternate


def _result_path(root: Path, result_id: str) -> Path:
    local_id = result_id.split("/", 1)[1]
    return root / "demo" / "study_results" / f"{local_id}.yaml"


def _complete_result(repository_root: Path, result_id: str) -> None:
    resolver = CanonicalRepositoryResolver(repository_root / "mldb_data")
    current = dict(resolver.resolve(kind=EntityKind.STUDY_RESULT, entity_id=result_id))
    replacement = copy.deepcopy(current)
    for trial in replacement["trials"]:
        if trial["training"] is not None and trial["training"]["disposition"] == "pending":
            trial["training"] = {
                "disposition": "completed",
                "result": f"{result_id}-{trial['trial']}-train",
                "reason": None,
            }
        for slot in trial["evaluations"]:
            if slot["disposition"] == "pending":
                slot["disposition"] = "completed"
                slot["result"] = f"{result_id}-{trial['trial']}-{slot['coordinate']}"
                slot["reason"] = None
    replacement["status"] = "completed"
    replacement["diagnostic"] = None
    CanonicalRepositoryWriter(
        repository_root,
        record_validator=_NoopValidator(),
    ).replace_nonterminal_study_result(
        entity_id=result_id,
        replacement=replacement,
    )


class _FakeAdvance:
    def __init__(self, repository_root: Path, *, terminal_on: int = 1) -> None:
        self.repository_root = repository_root
        self.terminal_on = terminal_on
        self.calls: list[str] = []

    def __call__(self, *, study_result):
        self.calls.append(str(study_result))
        terminal = len(self.calls) >= self.terminal_on
        if terminal:
            _complete_result(self.repository_root, str(study_result))
        return {
            "study_result": study_result,
            "status": "completed" if terminal else "submitted",
            "changed": terminal,
            "admitted": [],
            "finalized_results": [],
            "finalized_models": [],
            "active": [],
            "terminal": terminal,
        }


class _FakeCancel:
    def __init__(self, response: dict[str, object] | None = None) -> None:
        self.calls: list[str] = []
        self.response = response or {
            "study_result": "demo/run-placeholder",
            "outcome": "accepted",
            "status": "cancelling",
        }

    def __call__(self, *, study_result):
        self.calls.append(str(study_result))
        return self.response


def _never_plan(*, study):
    raise AssertionError("planning must not be called")


def _never_advance(*, study_result):
    raise AssertionError("advancement must not be called")


def _never_cancel(*, study_result):
    raise AssertionError("cancellation must not be called")


def _shell(
    tmp_path: Path,
    *,
    plan_study=_never_plan,
    advance_one_pass=_never_advance,
    request_cancel=_never_cancel,
    wait=lambda: None,
    execution_key_factory=lambda: FRESH_UUID_A,
    created_at_factory=lambda: "2026-09-13T12:00:00Z",
):
    return execution._ExecutionCompositionShell(
        repository_root=tmp_path,
        plan_study=plan_study,
        advance_one_pass=advance_one_pass,
        request_cancel=request_cancel,
        wait=wait,
        execution_key_factory=execution_key_factory,
        created_at_factory=created_at_factory,
    )


def test_start_persists_without_progression_and_mirrors_plan_topology(tmp_path: Path) -> None:
    root, plan = _install_plan(tmp_path)
    clock_calls: list[str] = []

    def created_at():
        clock_calls.append("called")
        return "2026-09-13T12:34:56Z"
    shell = _shell(tmp_path, created_at_factory=created_at)
    result = shell.start_study(plan=plan["id"], backend="fake", execution_key=SOURCE_KEY)

    assert _result_path(root, result["id"]).is_file()
    assert clock_calls == ["called"]
    assert result["id"] == f"demo/run-{SOURCE_KEY}"
    assert result["execution_key"] == SOURCE_KEY
    assert result["status"] == "submitted"
    assert result["diagnostic"] is None
    assert result["backend"] == "fake"
    assert result["created_at"] == "2026-09-13T12:34:56Z"
    assert result["trials"][0]["training"] == {
        "disposition": "pending", "result": None, "reason": None
    }
    assert result["trials"][1]["training"] is None
    assert [slot["coordinate"] for slot in result["trials"][0]["evaluations"]] == [
        item["coordinate"] for item in plan["trials"][0]["evaluations"]
    ]
    assert [slot["stage"] for slot in result["trials"][1]["evaluations"]] == [
        item["stage"] for item in plan["trials"][1]["evaluations"]
    ]
    assert all(
        slot["disposition"] == "pending"
        for trial in result["trials"] for slot in trial["evaluations"]
    )


def test_start_same_identity_is_idempotent_and_does_not_reallocate_created_at(tmp_path: Path) -> None:
    _root, plan = _install_plan(tmp_path)
    calls: list[int] = []

    def created_at():
        calls.append(1)
        return "2026-09-13T12:34:56Z"

    shell = _shell(tmp_path, created_at_factory=created_at)
    first = shell.start_study(plan=plan["id"], backend="fake", execution_key=SOURCE_KEY)
    second = shell.start_study(plan=plan["id"], backend="fake", execution_key=SOURCE_KEY)

    assert second == first
    assert calls == [1]


def test_start_same_key_conflicts_for_backend_or_plan_change(tmp_path: Path) -> None:
    root, plan = _install_plan(tmp_path)
    alternate = _alternate_plan(root, plan)
    shell = _shell(tmp_path)
    shell.start_study(plan=plan["id"], backend="fake", execution_key=SOURCE_KEY)

    with pytest.raises(ValueError, match="lifecycle conflict"):
        shell.start_study(plan=plan["id"], backend="other", execution_key=SOURCE_KEY)
    with pytest.raises(ValueError, match="lifecycle conflict"):
        shell.start_study(plan=alternate["id"], backend="fake", execution_key=SOURCE_KEY)


def test_advance_delegates_exactly_once_and_passes_response_through(tmp_path: Path) -> None:
    _root, plan = _install_plan(tmp_path)
    result = _shell(tmp_path).start_study(
        plan=plan["id"], backend="fake", execution_key=SOURCE_KEY
    )
    calls: list[str] = []
    response = {
        "study_result": result["id"],
        "status": "submitted",
        "changed": False,
        "admitted": [],
        "finalized_results": [],
        "finalized_models": [],
        "active": [],
        "terminal": False,
    }

    def advance(*, study_result):
        calls.append(str(study_result))
        return response

    shell = _shell(tmp_path, advance_one_pass=advance)
    actual = shell.advance_study(study_result=result["id"])
    assert actual is response
    assert calls == [result["id"]]


def test_run_plans_once_starts_fresh_uuid4_and_loops_only_until_terminal(tmp_path: Path) -> None:
    _root, plan = _install_plan(tmp_path)
    plan_calls: list[str] = []
    waits: list[int] = []
    driver = _FakeAdvance(tmp_path, terminal_on=2)
    def plan_study(*, study):
        plan_calls.append(str(study))
        return plan

    shell = _shell(
        tmp_path,
        plan_study=plan_study,
        advance_one_pass=driver,
        wait=lambda: waits.append(1),
        execution_key_factory=lambda: FRESH_UUID_A,
    )
    result = shell.run_study(study=plan["study"], backend="fake")

    expected_id = f"demo/run-{FRESH_UUID_A.hex}"
    assert plan_calls == [plan["study"]]
    assert driver.calls == [expected_id, expected_id]
    assert waits == [1]
    assert result["id"] == expected_id
    assert result["execution_key"] == FRESH_UUID_A.hex
    assert result["status"] == "completed"


def test_resume_reuses_existing_execution_identity_and_rejects_terminal_input(tmp_path: Path) -> None:
    _root, plan = _install_plan(tmp_path)
    starter = _shell(tmp_path)
    source = starter.start_study(plan=plan["id"], backend="fake", execution_key=SOURCE_KEY)
    driver = _FakeAdvance(tmp_path, terminal_on=1)
    shell = _shell(
        tmp_path,
        advance_one_pass=driver,
        execution_key_factory=lambda: (_ for _ in ()).throw(
            AssertionError("resume must not allocate an execution key")
        ),
    )
    resumed = shell.resume_study(study_result=source["id"])

    assert resumed["id"] == source["id"]
    assert resumed["execution_key"] == SOURCE_KEY
    assert driver.calls == [source["id"]]
    with pytest.raises(ValueError, match="cannot resume terminal"):
        shell.resume_study(study_result=source["id"])


def test_rerun_uses_exact_source_plan_without_replanning_and_inherits_backend(tmp_path: Path) -> None:
    _root, plan = _install_plan(tmp_path)
    source = _shell(tmp_path).start_study(
        plan=plan["id"], backend="source-backend", execution_key=SOURCE_KEY
    )
    driver = _FakeAdvance(tmp_path)
    shell = _shell(
        tmp_path,
        plan_study=_never_plan,
        advance_one_pass=driver,
        execution_key_factory=lambda: FRESH_UUID_A,
    )

    rerun = shell.rerun_study(source=source["id"])
    assert rerun["plan"] == source["plan"]
    assert rerun["backend"] == "source-backend"
    assert rerun["execution_key"] == FRESH_UUID_A.hex


def test_rerun_backend_override_and_fresh_uuid_each_execution(tmp_path: Path) -> None:
    _root, plan = _install_plan(tmp_path)
    source = _shell(tmp_path).start_study(
        plan=plan["id"], backend="source-backend", execution_key=SOURCE_KEY
    )
    keys = iter([FRESH_UUID_A, FRESH_UUID_B])
    driver = _FakeAdvance(tmp_path)
    shell = _shell(
        tmp_path,
        advance_one_pass=driver,
        execution_key_factory=lambda: next(keys),
    )

    first = shell.rerun_study(source=source["id"], backend="override")
    second = shell.rerun_study(source=source["id"], backend="override")

    assert first["backend"] == "override"
    assert second["backend"] == "override"
    assert first["execution_key"] == FRESH_UUID_A.hex
    assert second["execution_key"] == FRESH_UUID_B.hex
    assert first["execution_key"] != second["execution_key"]


def test_cancel_delegates_once_and_returns_seam_response(tmp_path: Path) -> None:
    _root, plan = _install_plan(tmp_path)
    result = _shell(tmp_path).start_study(
        plan=plan["id"], backend="fake", execution_key=SOURCE_KEY
    )
    response = {
        "study_result": result["id"],
        "outcome": "accepted",
        "status": "cancelling",
    }
    cancel = _FakeCancel(response)
    shell = _shell(tmp_path, request_cancel=cancel)

    actual = shell.cancel_study(study_result=result["id"])
    assert actual == response
    assert cancel.calls == [result["id"]]


def test_run_persists_start_before_first_advance_call(tmp_path: Path) -> None:
    root, plan = _install_plan(tmp_path)
    expected_id = f"demo/run-{FRESH_UUID_A.hex}"
    seen: list[str] = []

    def advance(*, study_result):
        assert study_result == expected_id
        assert _result_path(root, expected_id).is_file()
        seen.append(str(study_result))
        _complete_result(tmp_path, expected_id)
        return {
            "study_result": study_result,
            "status": "completed",
            "changed": True,
            "admitted": [],
            "finalized_results": [],
            "finalized_models": [],
            "active": [],
            "terminal": True,
        }

    shell = _shell(
        tmp_path,
        plan_study=lambda *, study: plan,
        advance_one_pass=advance,
        execution_key_factory=lambda: FRESH_UUID_A,
    )
    result = shell.run_study(study=plan["study"], backend="fake")
    assert seen == [expected_id]
    assert result["id"] == expected_id



class _NullTransport:
    def read_bytes(self, uri: str) -> bytes:
        raise AssertionError("error-boundary test must not read object bytes")

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        raise AssertionError("error-boundary test must not publish object bytes")


def _application(tmp_path: Path) -> Application:
    return Application(
        repository_root=tmp_path,
        backend_registry=BackendRegistry(),
        object_bytes=_ObjectByteAccess(_NullTransport()),
    )


def _assert_invalid_request(call) -> None:
    with pytest.raises(_ApplicationBoundaryError) as caught:
        call()
    assert caught.value.error["code"] == "invalid_request"
    assert set(caught.value.error) == {"code", "message"}


def test_public_execution_request_values_are_bounded_invalid_request(tmp_path: Path) -> None:
    _root, plan = _install_plan(tmp_path)
    starter = _shell(tmp_path)
    source = starter.start_study(
        plan=plan["id"], backend="fake", execution_key=SOURCE_KEY
    )
    shell = _shell(tmp_path)

    cases = (
        lambda: shell.start_study(plan="bad", backend="fake", execution_key=SOURCE_KEY),
        lambda: shell.start_study(plan=plan["id"], backend="fake", execution_key="not-uuid4"),
        lambda: shell.start_study(plan=plan["id"], backend="", execution_key=SOURCE_KEY),
        lambda: shell.advance_study(study_result="bad"),
        lambda: shell.run_study(study="bad", backend="fake"),
        lambda: shell.run_study(study=plan["study"], backend=""),
        lambda: shell.resume_study(study_result="bad"),
        lambda: shell.rerun_study(source="bad"),
        lambda: shell.rerun_study(source=source["id"], backend=""),
        lambda: shell.cancel_study(study_result="bad"),
    )
    for call in cases:
        _assert_invalid_request(call)


def test_public_application_preserves_request_domain_and_lifecycle_categories(tmp_path: Path) -> None:
    root, plan = _install_plan(tmp_path)
    app = _application(tmp_path)

    for call in (
        lambda: app.start_study(plan="bad", backend="fake", execution_key=SOURCE_KEY),
        lambda: app.start_study(plan=plan["id"], backend="fake", execution_key="not-uuid4"),
        lambda: app.start_study(plan=plan["id"], backend="", execution_key=SOURCE_KEY),
        lambda: app.advance_study(study_result="bad"),
        lambda: app.run_study(study="bad", backend="fake"),
        lambda: app.run_study(study=plan["study"], backend=""),
        lambda: app.resume_study(study_result="bad"),
        lambda: app.rerun_study(source="bad"),
        lambda: app.cancel_study(study_result="bad"),
    ):
        _assert_invalid_request(call)

    source = app.start_study(
        plan=plan["id"], backend="fake", execution_key=SOURCE_KEY
    )
    _assert_invalid_request(lambda: app.rerun_study(source=source["id"], backend=""))

    missing_plan = str(plan["id"])[:-1] + ("0" if not str(plan["id"]).endswith("0") else "1")
    with pytest.raises(_ApplicationBoundaryError) as caught:
        app.start_study(plan=missing_plan, backend="fake", execution_key=FRESH_UUID_A.hex)
    assert caught.value.error["code"] == "not_found"

    with pytest.raises(_ApplicationBoundaryError) as caught:
        app.start_study(plan=plan["id"], backend="other", execution_key=SOURCE_KEY)
    assert caught.value.error["code"] == "lifecycle_conflict"

    plan_path = root / "demo" / "study_plans" / f"{str(plan['id']).split('/', 1)[1]}.yaml"
    corrupted = json.loads(plan_path.read_text(encoding="utf-8"))
    corrupted["schema"] = "mjtensu.mldb-v2/study-plan/corrupt"
    _write_json(plan_path, corrupted)
    with pytest.raises(_ApplicationBoundaryError) as caught:
        app.start_study(plan=plan["id"], backend="fake", execution_key=FRESH_UUID_B.hex)
    assert caught.value.error["code"] == "validation_failed"


def test_internal_execution_key_failure_is_not_request_invalid(tmp_path: Path) -> None:
    _root, plan = _install_plan(tmp_path)
    shell = _shell(
        tmp_path,
        plan_study=lambda *, study: plan,
        execution_key_factory=lambda: uuid.UUID("aaaaaaaa-aaaa-1aaa-8aaa-aaaaaaaaaaaa"),
    )
    with pytest.raises(ValueError, match="UUID4"):
        shell.run_study(study=plan["study"], backend="fake")

def test_execution_shell_contains_no_w006_progression_algorithm_dependencies() -> None:
    source = Path(execution.__file__).read_text(encoding="utf-8")
    folded = source.casefold()
    assert "execution_readiness" not in source
    assert "result_acceptance" not in source
    assert "backend_port" not in source
    assert "study_driver" not in source
    assert "clearml" not in folded
    assert "queue" not in folded
    assert "retry" not in folded

    tree = __import__("ast").parse(source)
    advance = next(
        node for node in __import__("ast").walk(tree)
        if isinstance(node, __import__("ast").FunctionDef) and node.name == "advance_study"
    )
    calls = [node for node in __import__("ast").walk(advance) if isinstance(node, __import__("ast").Call)]
    assert any(
        isinstance(call.func, __import__("ast").Attribute)
        and call.func.attr == "_advance_one_pass"
        for call in calls
    )
