from __future__ import annotations

import ast
import inspect
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mldb_v2.src.cli.adapter import CliApplicationAdapter
from mldb_v2.src.cli.types import CliCommandName, CliDefinitionKind, CliResource
from mldb_v2.src.common.ids import DefinitionKind, EntityKind


REPO = Path(__file__).resolve().parents[2]
_ACTIVE = ("submitted", "cancelling")


class FakeApplication:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.responses: dict[str, object] = {}

    def _call(self, name: str, **kwargs: object) -> object:
        self.calls.append((name, kwargs))
        return self.responses.setdefault(name, object())

    def list_entities(self, **kwargs: object) -> object:
        return self._call("list_entities", **kwargs)

    def get_entity(self, **kwargs: object) -> object:
        return self._call("get_entity", **kwargs)

    def list_study_results(self, **kwargs: object) -> object:
        return self._call("list_study_results", **kwargs)

    def observe_study(self, **kwargs: object) -> object:
        return self._call("observe_study", **kwargs)

    def validate_scope(self, **kwargs: object) -> object:
        return self._call("validate_scope", **kwargs)

    def verify_scope(self, **kwargs: object) -> object:
        return self._call("verify_scope", **kwargs)

    def seal_scope(self, **kwargs: object) -> object:
        return self._call("seal_scope", **kwargs)

    def plan_study(self, **kwargs: object) -> object:
        return self._call("plan_study", **kwargs)

    def run_study(self, **kwargs: object) -> object:
        return self._call("run_study", **kwargs)

    def resume_study(self, **kwargs: object) -> object:
        return self._call("resume_study", **kwargs)

    def rerun_study(self, **kwargs: object) -> object:
        return self._call("rerun_study", **kwargs)

    def cancel_study(self, **kwargs: object) -> object:
        return self._call("cancel_study", **kwargs)

    def advance_study(self, **kwargs: object) -> object:
        return self._call("advance_study", **kwargs)

    def read_backend_logs(self, **kwargs: object) -> object:
        return self._call("read_backend_logs", **kwargs)

    def diagnose(self, **kwargs: object) -> object:
        return self._call("diagnose", **kwargs)


def _dispatch(request: dict[str, object]) -> tuple[FakeApplication, object]:
    application = FakeApplication()
    result = CliApplicationAdapter().dispatch(application=application, request=request)  # type: ignore[arg-type]
    return application, result


def _assert_identity(application: FakeApplication, name: str, result: object) -> None:
    assert result is application.responses[name]


def test_ps_and_idless_status_are_active_discovery_with_exact_filters() -> None:
    application, result = _dispatch({"command": CliCommandName.PS, "all": False})
    _assert_identity(application, "list_study_results", result)
    assert application.calls == [("list_study_results", {
        "namespace": None, "study": None, "statuses": _ACTIVE,
        "created_at_from": None, "created_at_to": None, "limit": None,
    })]

    application, result = _dispatch({"command": CliCommandName.STATUS})
    _assert_identity(application, "list_study_results", result)
    assert application.calls[0][1]["statuses"] == _ACTIVE


def test_ps_all_and_execution_selectors_map_exactly() -> None:
    request = {
        "command": CliCommandName.PS,
        "all": True,
        "selectors": {
            "namespace": "demo",
            "status": "completed",
            "study": "demo/study-v1",
            "since": "2026-09-01T00:00:00Z",
            "limit": 7,
        },
    }
    application, result = _dispatch(request)
    _assert_identity(application, "list_study_results", result)
    assert application.calls == [("list_study_results", {
        "namespace": "demo",
        "study": "demo/study-v1",
        "statuses": ("completed",),
        "created_at_from": datetime(2026, 9, 1, tzinfo=timezone.utc),
        "created_at_to": None,
        "limit": 7,
    })]


def test_ps_without_all_rejects_terminal_status_instead_of_ignoring_it() -> None:
    application = FakeApplication()
    with pytest.raises(ValueError, match="terminal --status"):
        CliApplicationAdapter().dispatch(
            application=application,
            request={"command": CliCommandName.PS, "all": False, "selectors": {"status": "failed"}},  # type: ignore[arg-type]
        )
    assert application.calls == []


@pytest.mark.parametrize(
    ("resource", "kind"),
    [
        (CliResource.NAMESPACES, EntityKind.NAMESPACE),
        (CliResource.TASKS, EntityKind.TASK),
        (CliResource.CORPORA, EntityKind.CORPUS),
        (CliResource.ARCHITECTURES, EntityKind.ARCHITECTURE),
        (CliResource.TRAIN_PROTOCOLS, EntityKind.TRAIN_PROTOCOL),
        (CliResource.EVALUATION_PROTOCOLS, EntityKind.EVALUATION_PROTOCOL),
        (CliResource.STUDIES, EntityKind.STUDY),
        (CliResource.PLANS, EntityKind.STUDY_PLAN),
        (CliResource.TRAINING_RESULTS, EntityKind.TRAINING_RESULT),
        (CliResource.MODELS, EntityKind.MODEL),
        (CliResource.EVALUATION_RESULTS, EntityKind.EVALUATION_RESULT),
        (CliResource.RUNS, EntityKind.STUDY_RESULT),
    ],
)
def test_exact_get_resource_mapping(resource: CliResource, kind: EntityKind) -> None:
    application, result = _dispatch({
        "command": CliCommandName.GET,
        "resource": resource,
        "typed_id": "demo/item-v1" if resource is not CliResource.NAMESPACES else "demo",
    })
    _assert_identity(application, "get_entity", result)
    assert application.calls[0][0] == "get_entity"
    assert application.calls[0][1]["kind"] is kind


def test_idless_get_definitions_and_runs_are_discovery_operations() -> None:
    application, result = _dispatch({
        "command": CliCommandName.GET,
        "resource": CliResource.DEFINITIONS,
        "selectors": {"namespace": "demo", "status": "sealed"},
    })
    _assert_identity(application, "list_entities", result)
    assert application.calls == [("list_entities", {
        "resource": "definitions", "namespace": "demo", "status": "sealed",
    })]

    application, result = _dispatch({
        "command": CliCommandName.GET,
        "resource": CliResource.RUNS,
        "selectors": {"namespace": "demo", "status": "failed", "limit": 3},
    })
    _assert_identity(application, "list_study_results", result)
    assert application.calls[0][1]["statuses"] == ("failed",)
    assert application.calls[0][1]["limit"] == 3


def test_describe_uses_exact_public_get_entity_only() -> None:
    application, result = _dispatch({
        "command": CliCommandName.DESCRIBE,
        "resource": CliResource.MODELS,
        "typed_id": "demo/model-v1",
    })
    _assert_identity(application, "get_entity", result)
    assert application.calls == [("get_entity", {
        "kind": EntityKind.MODEL, "entity_id": "demo/model-v1",
    })]


def test_validate_verify_and_seal_map_definition_scope_exactly() -> None:
    application, result = _dispatch({
        "command": CliCommandName.VALIDATE,
        "kind": CliDefinitionKind.TASK,
        "typed_id": "demo/task-v1",
        "namespace": "demo",
        "fail_fast": False,
    })
    _assert_identity(application, "validate_scope", result)
    assert application.calls == [("validate_scope", {"scope": {
        "kind": DefinitionKind.TASK, "id": "demo/task-v1", "namespace": "demo",
    }})]

    application, result = _dispatch({"command": CliCommandName.VERIFY, "fail_fast": False})
    _assert_identity(application, "verify_scope", result)
    assert application.calls == [("verify_scope", {"scope": None})]

    application, result = _dispatch({
        "command": CliCommandName.SEAL,
        "kind": CliDefinitionKind.STUDY,
        "typed_id": "demo/study-v1",
    })
    _assert_identity(application, "seal_scope", result)
    assert application.calls == [("seal_scope", {
        "scope": {"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}, "bulk": False,
    })]



def test_fail_fast_true_reports_public_seam_mismatch_without_delegation() -> None:
    application = FakeApplication()
    with pytest.raises(ValueError, match="no ApplicationInterface delegation seam"):
        CliApplicationAdapter().dispatch(
            application=application,
            request={
                "command": CliCommandName.VALIDATE,
                "fail_fast": True,
            },  # type: ignore[arg-type]
        )
    assert application.calls == []

def test_bulk_seal_and_plan_delegate_once() -> None:
    application, result = _dispatch({
        "command": CliCommandName.SEAL,
        "kind": CliDefinitionKind.ARCHITECTURE,
        "namespace": "demo",
        "all": True,
    })
    _assert_identity(application, "seal_scope", result)
    assert application.calls == [("seal_scope", {
        "scope": {"kind": DefinitionKind.ARCHITECTURE, "namespace": "demo"},
        "bulk": True,
    })]

    application, result = _dispatch({
        "command": CliCommandName.PLAN,
        "study": "demo/study-v1",
    })
    _assert_identity(application, "plan_study", result)
    assert application.calls == [("plan_study", {"study": "demo/study-v1"})]


def test_execution_commands_delegate_to_application_owned_methods() -> None:
    cases = [
        ({"command": CliCommandName.RUN, "study": "demo/study-v1", "backend": "clearml"},
         "run_study", {"study": "demo/study-v1", "backend": "clearml"}),
        ({"command": CliCommandName.RESUME, "study_result": "demo/run-v1"},
         "resume_study", {"study_result": "demo/run-v1"}),
        ({"command": CliCommandName.RERUN, "study_result": "demo/run-v1"},
         "rerun_study", {"source": "demo/run-v1", "backend": None}),
        ({"command": CliCommandName.CANCEL, "study_result": "demo/run-v1"},
         "cancel_study", {"study_result": "demo/run-v1"}),
    ]
    for request, method, kwargs in cases:
        application, result = _dispatch(request)
        _assert_identity(application, method, result)
        assert application.calls == [(method, kwargs)]


def test_advance_is_exactly_one_application_call() -> None:
    application, result = _dispatch({
        "command": CliCommandName.ADVANCE,
        "study_result": "demo/run-v1",
    })
    _assert_identity(application, "advance_study", result)
    assert application.calls == [("advance_study", {"study_result": "demo/run-v1"})]


def test_watch_is_strictly_read_only_for_exact_and_discovery_forms() -> None:
    application, result = _dispatch({
        "command": CliCommandName.WATCH,
        "study_result": "demo/run-v1",
    })
    _assert_identity(application, "observe_study", result)
    assert application.calls == [("observe_study", {"study_result": "demo/run-v1"})]

    application, result = _dispatch({
        "command": CliCommandName.WATCH,
        "selectors": {"namespace": "demo", "limit": 2},
    })
    _assert_identity(application, "list_study_results", result)
    assert application.calls[0][0] == "list_study_results"
    assert application.calls[0][1]["statuses"] == _ACTIVE


def test_logs_builds_exact_backend_log_request_and_preserves_response_identity() -> None:
    application, result = _dispatch({
        "command": CliCommandName.LOGS,
        "study_result": "demo/run-v1",
        "trial": "trial-0002",
        "stage": "eval-0003",
        "failed_only": True,
        "follow": True,
    })
    _assert_identity(application, "read_backend_logs", result)
    assert application.calls == [("read_backend_logs", {"request": {
        "study_result": "demo/run-v1",
        "trial": "trial-0002",
        "coordinate": "eval-0003",
        "failed_only": True,
        "follow": True,
    }})]


def test_doctor_delegates_only_to_application_diagnose() -> None:
    application, result = _dispatch({"command": CliCommandName.DOCTOR})
    _assert_identity(application, "diagnose", result)
    assert application.calls == [("diagnose", {})]


def test_rerun_preserves_explicit_backend_override() -> None:
    application, result = _dispatch({
        "command": CliCommandName.RERUN,
        "study_result": "demo/run-v1",
        "backend": "local",
    })
    _assert_identity(application, "rerun_study", result)
    assert application.calls == [("rerun_study", {
        "source": "demo/run-v1", "backend": "local",
    })]


def test_run_without_backend_reports_public_seam_mismatch_without_calling_application() -> None:
    application = FakeApplication()
    with pytest.raises(ValueError, match="no default-backend seam"):
        CliApplicationAdapter().dispatch(
            application=application,
            request={"command": CliCommandName.RUN, "study": "demo/study-v1"},  # type: ignore[arg-type]
        )
    assert application.calls == []


@pytest.mark.parametrize("command", [CliCommandName.GET, CliCommandName.WATCH])
def test_exact_target_with_selectors_rejects_instead_of_silently_ignoring(command: CliCommandName) -> None:
    application = FakeApplication()
    request: dict[str, object]
    if command is CliCommandName.GET:
        request = {
            "command": command,
            "resource": CliResource.TASKS,
            "typed_id": "demo/task-v1",
            "selectors": {"namespace": "demo"},
        }
    else:
        request = {
            "command": command,
            "study_result": "demo/run-v1",
            "selectors": {"namespace": "demo"},
        }
    with pytest.raises(ValueError, match="selectors are not valid"):
        CliApplicationAdapter().dispatch(
            application=application,
            request=request,  # type: ignore[arg-type]
        )
    assert application.calls == []


def test_exact_aggregate_definitions_target_rejects() -> None:
    application = FakeApplication()
    with pytest.raises(ValueError, match="aggregate definitions"):
        CliApplicationAdapter().dispatch(
            application=application,
            request={
                "command": CliCommandName.GET,
                "resource": CliResource.DEFINITIONS,
                "typed_id": "demo/task-v1",
            },  # type: ignore[arg-type]
        )
    assert application.calls == []


def _dispatch_shape(path: Path) -> tuple[object, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CliApplicationAdapter"
    )
    method = next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "dispatch"
    )
    return (
        tuple(arg.arg for arg in method.args.posonlyargs),
        tuple(arg.arg for arg in method.args.args),
        tuple(arg.arg for arg in method.args.kwonlyargs),
        tuple(default is None for default in method.args.kw_defaults),
        ast.unparse(method.returns) if method.returns is not None else None,
    )


def test_adapter_dispatch_public_shape_matches_frozen_skeleton() -> None:
    runtime = REPO / "mldb_v2/src/cli/adapter.py"
    frozen = REPO / "mldb_v2/skeleton/cli/adapter.py"
    assert _dispatch_shape(runtime) == _dispatch_shape(frozen)

    signature = inspect.signature(CliApplicationAdapter.dispatch)
    assert tuple(signature.parameters) == ("self", "application", "request")
    assert signature.parameters["application"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["request"].kind is inspect.Parameter.KEYWORD_ONLY


def test_adapter_has_no_forbidden_runtime_dependencies_or_private_api_helpers() -> None:
    path = REPO / "mldb_v2/src/cli/adapter.py"
    source = path.read_text(encoding="utf-8")
    assert "mldb_v2.skeleton" not in source

    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    forbidden = (
        "mldb_v2.src.repository",
        "mldb_v2.src.backend",
        "mldb_v2.src.api._",
        "clearml",
        "yaml",
        "pathlib",
        "os",
    )
    assert not any(name.startswith(prefix) for name in imported for prefix in forbidden)


def test_watch_never_delegates_to_progression_methods() -> None:
    application, _ = _dispatch({
        "command": CliCommandName.WATCH,
        "study_result": "demo/run-v1",
    })
    called = {name for name, _ in application.calls}
    assert called.isdisjoint({"advance_study", "run_study", "resume_study"})
