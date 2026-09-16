from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

from mldb_v2.src.cli.main import main


REPO = Path(__file__).resolve().parents[2]


class PublicApplicationFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.error = {"code": code, "message": message}
        super().__init__(message)


class FakeApplication:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.responses: dict[str, object] = {}
        self.failures: dict[str, BaseException] = {}

    def _call(self, name: str, **kwargs: object) -> object:
        self.calls.append((name, kwargs))
        failure = self.failures.get(name)
        if failure is not None:
            raise failure
        return self.responses.get(name, {"operation": name})
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


def _run(argv: list[str], application: FakeApplication) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    status = main(
        argv,
        application_factory=lambda: application,  # type: ignore[return-value]
        stdout=stdout,
        stderr=stderr,
    )
    return status, stdout.getvalue(), stderr.getvalue()


def test_discovery_argv_dispatch_render_and_exit() -> None:
    application = FakeApplication()
    application.responses["list_entities"] = [
        {"id": "demo/task-v1", "status": "sealed"}
    ]
    status, stdout, stderr = _run(
        ["get", "tasks", "--namespace", "demo", "--json"],
        application,
    )
    assert status == 0
    assert stderr == ""
    assert stdout.lstrip().startswith("[")
    assert '"id": "demo/task-v1"' in stdout
    assert [name for name, _ in application.calls] == ["list_entities"]


def test_authoring_validate_report_controls_exit() -> None:
    application = FakeApplication()
    application.responses["validate_scope"] = {
        "items": [
            {"kind": "task", "id": "demo/task-v1", "valid": True, "diagnostics": []}
        ],
        "repository_issues": [],
    }
    status, stdout, stderr = _run(
        ["validate", "task", "demo/task-v1", "--json"],
        application,
    )
    assert status == 0
    assert stderr == ""
    assert '"valid": true' in stdout
    assert [name for name, _ in application.calls] == ["validate_scope"]

def test_validate_fail_fast_is_accepted_as_full_evaluation() -> None:
    application = FakeApplication()
    application.responses["validate_scope"] = {
        "items": [
            {"kind": "task", "id": "demo/task-v1", "valid": True, "diagnostics": []},
            {"kind": "study", "id": "demo/study-v1", "valid": True, "diagnostics": []},
        ],
        "repository_issues": [],
    }
    status, _, stderr = _run(["validate", "--fail-fast", "--json"], application)
    assert status == 0
    assert stderr == ""
    assert application.calls == [("validate_scope", {"scope": None})]


def test_validate_failure_report_is_stdout_with_nonzero_exit() -> None:
    application = FakeApplication()
    application.responses["verify_scope"] = {
        "items": [
            {"kind": "study", "id": "demo/study-v1", "valid": False, "diagnostics": []}
        ],
        "repository_issues": [],
    }
    status, stdout, stderr = _run(["verify", "--json"], application)
    assert status == 1
    assert stderr == ""
    assert '"valid": false' in stdout


def test_execution_run_dispatches_once_with_explicit_backend() -> None:
    application = FakeApplication()
    status, stdout, stderr = _run(
        ["run", "demo/study-v1", "--backend", "clearml"],
        application,
    )
    assert status == 0
    assert stderr == ""
    assert "operation" in stdout
    assert application.calls == [
        ("run_study", {"study": "demo/study-v1", "backend": "clearml"})
    ]


def test_run_without_backend_does_not_invent_default() -> None:
    application = FakeApplication()
    status, stdout, stderr = _run(["run", "demo/study-v1"], application)
    assert status == 1
    assert stdout == ""
    assert stderr.startswith("invalid_request:")
    assert "no default backend is configured" in stderr
    assert application.calls == []


def test_advance_dispatches_exactly_once() -> None:
    application = FakeApplication()
    status, _, stderr = _run(["advance", "demo/run-v1"], application)
    assert status == 0
    assert stderr == ""
    assert application.calls == [
        ("advance_study", {"study_result": "demo/run-v1"})
    ]


def test_watch_exact_is_read_only() -> None:
    application = FakeApplication()
    status, _, stderr = _run(["watch", "demo/run-v1"], application)
    assert status == 0
    assert stderr == ""
    assert application.calls == [
        ("observe_study", {"study_result": "demo/run-v1"})
    ]
    called = {name for name, _ in application.calls}
    assert called.isdisjoint({"advance_study", "run_study", "resume_study", "cancel_study"})


def test_keyboard_interrupt_is_local_and_never_cancels() -> None:
    application = FakeApplication()
    application.failures["run_study"] = KeyboardInterrupt()
    status, stdout, stderr = _run(
        ["run", "demo/study-v1", "--backend", "clearml"],
        application,
    )
    assert status == 130
    assert stdout == ""
    assert stderr == "interrupted\n"
    assert application.calls == [
        ("run_study", {"study": "demo/study-v1", "backend": "clearml"})
    ]
    assert all(name != "cancel_study" for name, _ in application.calls)


def test_application_error_uses_stderr_and_nonzero_exit() -> None:
    application = FakeApplication()
    application.failures["observe_study"] = PublicApplicationFailure(
        "not_found", "requested run was not found"
    )
    status, stdout, stderr = _run(["status", "demo/run-v1"], application)
    assert status == 1
    assert stdout == ""
    assert stderr == "not_found: requested run was not found\n"


def test_repository_local_mldb_entrypoint_help_smoke() -> None:
    completed = subprocess.run(
        ["cmd", "/c", "mldb", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "usage: mldb" in completed.stdout
    assert completed.stderr == ""


def test_main_has_no_repository_backend_or_clearml_runtime_imports() -> None:
    source = (REPO / "mldb_v2/src/cli/main.py").read_text(encoding="utf-8")
    forbidden = (
        "mldb_v2.src.repository",
        "mldb_v2.src.backend",
        "clearml",
        "yaml",
    )
    assert not any(token in source for token in forbidden)
