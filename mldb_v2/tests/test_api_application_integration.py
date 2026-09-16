from __future__ import annotations

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest

import mldb_v2.src.api as public_api
import mldb_v2.src.api.application as appmod
import mldb_v2.src.study.study_driver as study_driver
from mldb_v2.src.api._errors import _ApplicationBoundaryError
from mldb_v2.src.api.application import Application
from mldb_v2.src.api.errors import ApplicationErrorCode
from mldb_v2.src.backend._registry import BackendRegistry
from mldb_v2.src.common.ids import EntityKind
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
import mldb_v2.tests.test_api_execution as execution_fx


class NullTransport:
    def read_bytes(self, uri: str) -> bytes:
        raise AssertionError("this integration path must not read object bytes")

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        raise AssertionError("application must not publish object bytes directly")


class FakeBackend:
    def __init__(self) -> None:
        self.admit_calls: list[dict[str, object]] = []
        self.observe_calls: list[dict[str, object]] = []
        self.collect_calls: list[dict[str, object]] = []
        self.cancel_calls: list[str] = []
        self.observations: dict[tuple[str, str, str | None], dict[str, object]] = {}

    @staticmethod
    def _token(stage_key):
        return stage_key["trial"], stage_key["kind"], stage_key["coordinate"]

    def observe(self, *, stage_key):
        self.observe_calls.append(copy.deepcopy(stage_key))
        return copy.deepcopy(self.observations.get(self._token(stage_key)))

    def collect(self, *, stage_key):
        self.collect_calls.append(copy.deepcopy(stage_key))
        return None

    def admit(self, *, stage_input):
        self.admit_calls.append(copy.deepcopy(stage_input))
        key = {
            "study_result": stage_input["study_result"],
            "plan": stage_input["plan"],
            "trial": stage_input["trial"],
            "kind": stage_input["kind"],
            "coordinate": stage_input["coordinate"],
            "source_commit": stage_input["source_commit"],
        }
        observation = {
            "state": "active",
            "stage_key": key,
            "backend": "fake",
            "execution_ids": [f"exec-{len(self.admit_calls)}"],
        }
        self.observations[self._token(key)] = copy.deepcopy(observation)
        return observation

    def cancel_study(self, *, study_result):
        self.cancel_calls.append(str(study_result))


class DirectLogBackend(FakeBackend):
    def read_backend_logs(self, *, request):
        return (
            {"execution_id": "exec-log", "text": "hello"},
            {"execution_id": None, "text": "world"},
        )


def _registry(backend: FakeBackend) -> BackendRegistry:
    registry = BackendRegistry()
    registry.register("fake", lambda _config: backend)
    return registry


def _application(tmp_path: Path, backend: FakeBackend) -> Application:
    return Application(
        repository_root=tmp_path,
        backend_registry=_registry(backend),
        object_bytes=_ObjectByteAccess(NullTransport()),
    )


def _result_path(root: Path, result_id: str) -> Path:
    local_id = result_id.split("/", 1)[1]
    return root / "demo" / "study_results" / f"{local_id}.yaml"


def test_application_wires_owned_object_bytes_into_authoring_verifier(tmp_path: Path) -> None:
    (tmp_path / "mldb_data").mkdir()
    backend = FakeBackend()
    object_bytes = _ObjectByteAccess(NullTransport())
    application = Application(
        repository_root=tmp_path,
        backend_registry=_registry(backend),
        object_bytes=object_bytes,
    )

    assert application._object_bytes is object_bytes
    assert application._authoring._verifier._object_access is object_bytes


def test_runtime_application_interface_is_exact_frozen_runtime_mirror() -> None:
    root = Path(__file__).parents[1]
    frozen = (root / "skeleton/api/application_interface.py").read_text(encoding="utf-8")
    runtime = (root / "src/api/application_interface.py").read_text(encoding="utf-8")
    assert runtime == frozen.replace("mldb_v2.skeleton.", "mldb_v2.src.")
    assert "mldb_v2.skeleton" not in runtime


def test_application_delegates_authoring_and_planning_to_t007_01() -> None:
    calls: list[tuple[str, object]] = []
    app = Application.__new__(Application)
    app._authoring = SimpleNamespace(
        validate_scope=lambda *, scope=None: calls.append(("validate", scope)) or {"items": [], "repository_issues": []},
        plan_study=lambda *, study: calls.append(("plan", study)) or {"id": "demo/plan"},
    )
    assert app.validate_scope(scope=None) == {"items": [], "repository_issues": []}
    assert app.plan_study(study="demo/study-v1")["id"] == "demo/plan"
    assert calls == [("validate", None), ("plan", "demo/study-v1")]


def test_application_delegates_query_to_t007_02() -> None:
    calls: list[tuple[object, object]] = []
    app = Application.__new__(Application)
    app._query = SimpleNamespace(
        get_entity=lambda *, kind, entity_id: calls.append((kind, entity_id)) or {"id": entity_id}
    )
    result = app.get_entity(kind=EntityKind.TASK, entity_id="demo/task-v1")
    assert result == {"id": "demo/task-v1"}
    assert calls == [(EntityKind.TASK, "demo/task-v1")]


def test_application_delegates_execution_to_t007_03() -> None:
    calls: list[tuple[str, object]] = []
    app = Application.__new__(Application)
    app._execution = SimpleNamespace(
        run_study=lambda *, study, backend: calls.append((str(study), backend)) or {"id": "demo/run-x"}
    )
    result = app.run_study(study="demo/study-v1", backend="fake")
    assert result["id"] == "demo/run-x"
    assert calls == [("demo/study-v1", "fake")]


def test_start_has_zero_backend_calls_then_actual_w006_advance_admits(tmp_path: Path) -> None:
    root, plan = execution_fx._install_plan(tmp_path)
    backend = FakeBackend()
    app = _application(tmp_path, backend)
    result = app.start_study(
        plan=plan["id"],
        backend="fake",
        execution_key=execution_fx.SOURCE_KEY,
    )
    assert _result_path(root, result["id"]).is_file()
    assert backend.admit_calls == []
    assert backend.observe_calls == []
    assert backend.collect_calls == []
    assert backend.cancel_calls == []
    assert appmod._advance_study is study_driver.advance_study

    response = app.advance_study(study_result=result["id"])
    assert response["study_result"] == result["id"]
    assert response["status"] == "submitted"
    assert response["terminal"] is False
    assert response["admitted"]
    assert len(backend.admit_calls) == len(response["admitted"])


def test_run_and_resume_share_application_w006_adapter(tmp_path: Path) -> None:
    _root, plan = execution_fx._install_plan(tmp_path)
    backend = FakeBackend()
    app = _application(tmp_path, backend)
    bound = app._execution._advance_one_pass
    assert bound.__self__ is app
    assert bound.__func__ is Application._advance_one_pass

    calls: list[str] = []
    def terminal_pass(*, study_result):
        calls.append(str(study_result))
        execution_fx._complete_result(tmp_path, str(study_result))
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

    app._execution._plan_study = lambda *, study: plan
    app._execution._advance_one_pass = terminal_pass
    run_result = app.run_study(study=plan["study"], backend="fake")

    source = app.start_study(
        plan=plan["id"],
        backend="fake",
        execution_key=execution_fx.SOURCE_KEY,
    )
    resumed = app.resume_study(study_result=source["id"])
    assert run_result["status"] == "completed"
    assert resumed["id"] == source["id"]
    assert calls == [run_result["id"], source["id"]]


def test_rerun_preserves_exact_source_plan_without_replanning(tmp_path: Path) -> None:
    _root, plan = execution_fx._install_plan(tmp_path)
    app = _application(tmp_path, FakeBackend())
    source = app.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    app._execution._plan_study = lambda *, study: (_ for _ in ()).throw(
        AssertionError("rerun must not replan the current Study")
    )

    def terminal_pass(*, study_result):
        execution_fx._complete_result(tmp_path, str(study_result))
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

    app._execution._advance_one_pass = terminal_pass
    rerun = app.rerun_study(source=source["id"])
    assert rerun["plan"] == source["plan"]
    assert rerun["source_commit"] == source["source_commit"]
    assert rerun["execution_key"] != source["execution_key"]


def test_cancel_request_mutates_only_status_and_calls_backend_outside_progression(tmp_path: Path) -> None:
    _root, plan = execution_fx._install_plan(tmp_path)
    backend = FakeBackend()
    app = _application(tmp_path, backend)
    result = app.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    before = copy.deepcopy(result)

    first = app.cancel_study(study_result=result["id"])
    current = app.get_study_result(study_result=result["id"])["study_result"]
    assert first == {
        "study_result": result["id"],
        "outcome": "accepted",
        "status": "cancelling",
    }
    assert current["status"] == "cancelling"
    assert current["trials"] == before["trials"]
    assert {key: current[key] for key in current if key not in {"status", "trials"}} == {
        key: before[key] for key in before if key not in {"status", "trials"}
    }
    assert backend.cancel_calls == [result["id"]]

    second = app.cancel_study(study_result=result["id"])
    assert second["outcome"] == "already_cancelling"
    assert second["status"] == "cancelling"
    assert backend.cancel_calls == [result["id"], result["id"]]

    adapter_source = inspect.getsource(appmod._CancellationRequestAdapter)
    assert "study_cancelled" not in adapter_source
    assert "_advance_study" not in adapter_source


def test_cancel_backend_call_occurs_outside_mutation_lock(
    tmp_path: Path, monkeypatch
) -> None:
    _root, plan = execution_fx._install_plan(tmp_path)
    held = {"value": False}
    real_coordinator = appmod.StudyResultMutationCoordinator

    class TrackingCoordinator(real_coordinator):
        def acquire(self, *, study_result_id):
            inner = super().acquire(study_result_id=study_result_id)

            class Guard:
                def __enter__(self_nonlocal):
                    inner.__enter__()
                    held["value"] = True

                def __exit__(self_nonlocal, exc_type, exc, tb):
                    held["value"] = False
                    return inner.__exit__(exc_type, exc, tb)

            return Guard()

    class AssertUnlockedBackend(FakeBackend):
        def cancel_study(self, *, study_result):
            assert held["value"] is False
            super().cancel_study(study_result=study_result)

    monkeypatch.setattr(appmod, "StudyResultMutationCoordinator", TrackingCoordinator)
    backend = AssertUnlockedBackend()
    app = _application(tmp_path, backend)
    result = app.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )

    response = app.cancel_study(study_result=result["id"])
    assert response["outcome"] == "accepted"
    assert backend.cancel_calls == [result["id"]]


def test_cancel_terminal_is_noop_and_does_not_call_backend(tmp_path: Path) -> None:
    _root, plan = execution_fx._install_plan(tmp_path)
    backend = FakeBackend()
    app = _application(tmp_path, backend)
    result = app.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.FRESH_UUID_A.hex
    )
    execution_fx._complete_result(tmp_path, result["id"])

    response = app.cancel_study(study_result=result["id"])
    assert response["outcome"] == "already_terminal"
    assert response["status"] == "completed"
    assert backend.cancel_calls == []


def test_query_observation_does_not_mutate_canonical_state(tmp_path: Path) -> None:
    root, plan = execution_fx._install_plan(tmp_path)
    backend = FakeBackend()
    app = _application(tmp_path, backend)
    result = app.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    path = _result_path(root, result["id"])
    before = path.read_bytes()

    observation = app.observe_study(study_result=result["id"])
    assert observation["study_result"]["id"] == result["id"]
    assert path.read_bytes() == before
    assert backend.admit_calls == []
    assert backend.collect_calls == []
    assert backend.cancel_calls == []


def test_backend_logs_supported_through_optional_capability(tmp_path: Path) -> None:
    _root, plan = execution_fx._install_plan(tmp_path)
    backend = DirectLogBackend()
    app = _application(tmp_path, backend)
    result = app.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    chunks = list(
        app.read_backend_logs(
            request={
                "study_result": result["id"],
                "trial": None,
                "coordinate": None,
                "failed_only": False,
                "follow": False,
            }
        )
    )
    assert chunks == [
        {"execution_id": "exec-log", "text": "hello"},
        {"execution_id": None, "text": "world"},
    ]


def test_backend_logs_unsupported_maps_to_public_error(tmp_path: Path) -> None:
    _root, plan = execution_fx._install_plan(tmp_path)
    app = _application(tmp_path, FakeBackend())
    result = app.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    with pytest.raises(_ApplicationBoundaryError) as caught:
        list(
            app.read_backend_logs(
                request={
                    "study_result": result["id"],
                    "trial": None,
                    "coordinate": None,
                    "failed_only": False,
                    "follow": False,
                }
            )
        )
    assert caught.value.error["code"] == "unsupported_capability"


def test_public_error_codes_match_frozen_categories_exactly() -> None:
    assert set(get_args(ApplicationErrorCode)) == {
        "not_found",
        "invalid_request",
        "validation_failed",
        "not_sealed",
        "source_not_pinned",
        "lifecycle_conflict",
        "backend_unavailable",
        "unsupported_capability",
        "result_rejected",
        "internal_failure",
    }


def test_unknown_backend_observation_maps_to_backend_unavailable(tmp_path: Path) -> None:
    _root, plan = execution_fx._install_plan(tmp_path)
    app = Application(
        repository_root=tmp_path,
        backend_registry=BackendRegistry(),
        object_bytes=_ObjectByteAccess(NullTransport()),
    )
    result = app.start_study(
        plan=plan["id"], backend="missing", execution_key=execution_fx.SOURCE_KEY
    )
    with pytest.raises(_ApplicationBoundaryError) as caught:
        app.observe_study(study_result=result["id"])
    assert caught.value.error["code"] == "backend_unavailable"


def test_generic_application_has_no_clearml_or_cli_branching() -> None:
    source = Path(appmod.__file__).read_text(encoding="utf-8")
    folded = source.casefold()
    assert "clearml" not in folded
    assert "mldb_v2.src.cli" not in source
    assert "mldb_v2.skeleton" not in source
    assert public_api.Application is Application
    assert set(public_api.__all__) == {
        "Application",
        "ApplicationComposition",
        "ApplicationError",
        "ApplicationErrorCode",
        "ApplicationInterface",
        "QueryInterface",
        "compose_application",
    }
