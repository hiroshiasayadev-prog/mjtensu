from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import pytest

import mldb_v2.src.api.application as appmod
import mldb_v2.src.study.study_driver as study_driver
from mldb_v2.src.api._errors import _ApplicationBoundaryError, _application_error
from mldb_v2.src.api.application import Application
from mldb_v2.src.api.errors import ApplicationErrorCode
from mldb_v2.src.common.ids import EntityKind
import mldb_v2.tests.test_api_application_integration as app_fx
import mldb_v2.tests.test_api_execution as execution_fx


_ERROR_CODES = {
    "not_found", "invalid_request", "validation_failed", "not_sealed",
    "source_not_pinned", "lifecycle_conflict", "backend_unavailable",
    "unsupported_capability", "result_rejected", "internal_failure",
}


def _assert_public_code(call, code: str) -> None:
    with pytest.raises(_ApplicationBoundaryError) as caught:
        call()
    assert caught.value.error["code"] == code
    assert set(caught.value.error) == {"code", "message"}


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_frozen_public_shapes_and_error_categories_are_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("application_interface.py", "query_interface.py"):
        frozen = (root / "skeleton" / "api" / name).read_text(encoding="utf-8")
        runtime = (root / "src" / "api" / name).read_text(encoding="utf-8")
        assert runtime == frozen.replace("mldb_v2.skeleton.", "mldb_v2.src.")
    assert (root / "src/api/errors.py").read_text(encoding="utf-8") == (
        root / "skeleton/api/errors.py"
    ).read_text(encoding="utf-8")
    assert set(get_args(ApplicationErrorCode)) == _ERROR_CODES

    api_paths = tuple((root / "src/api").glob("*.py"))
    api_source = "\n".join(path.read_text(encoding="utf-8") for path in api_paths)
    assert "mldb_v2.skeleton" not in api_source
    assert "mldb_v2.src.cli" not in api_source

    generic_api_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in api_paths
        if path.name != "composition.py"
    )
    assert "clearml" not in generic_api_source.casefold()


def test_public_boundary_preserves_all_frozen_bounded_error_categories() -> None:
    app = Application.__new__(Application)
    for code in sorted(_ERROR_CODES):
        def fail(code=code):
            raise _ApplicationBoundaryError(_application_error(code, "bounded"))

        _assert_public_code(lambda: app._public(fail), code)


def test_execution_request_domain_and_lifecycle_categories_are_separate(tmp_path: Path) -> None:
    root, plan = execution_fx._install_plan(tmp_path)
    app = execution_fx._application(tmp_path)

    invalid_calls = (
        lambda: app.start_study(plan="bad", backend="fake", execution_key=execution_fx.SOURCE_KEY),
        lambda: app.start_study(plan=plan["id"], backend="fake", execution_key="not-uuid4"),
        lambda: app.start_study(plan=plan["id"], backend="", execution_key=execution_fx.SOURCE_KEY),
        lambda: app.advance_study(study_result="bad"),
        lambda: app.run_study(study="bad", backend="fake"),
        lambda: app.run_study(study=plan["study"], backend=""),
        lambda: app.resume_study(study_result="bad"),
        lambda: app.rerun_study(source="bad"),
        lambda: app.cancel_study(study_result="bad"),
    )
    for call in invalid_calls:
        _assert_public_code(call, "invalid_request")

    source = app.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    _assert_public_code(lambda: app.rerun_study(source=source["id"], backend=""), "invalid_request")

    missing_plan = str(plan["id"])[:-1] + (
        "0" if not str(plan["id"]).endswith("0") else "1"
    )
    _assert_public_code(
        lambda: app.start_study(
            plan=missing_plan,
            backend="fake",
            execution_key=execution_fx.FRESH_UUID_A.hex,
        ),
        "not_found",
    )
    _assert_public_code(
        lambda: app.start_study(
            plan=plan["id"], backend="other", execution_key=execution_fx.SOURCE_KEY
        ),
        "lifecycle_conflict",
    )

    plan_path = root / "demo/study_plans" / f"{str(plan['id']).split('/', 1)[1]}.yaml"
    corrupted = json.loads(plan_path.read_text(encoding="utf-8"))
    corrupted["schema"] = "mjtensu.mldb-v2/study-plan/corrupt"
    execution_fx._write_json(plan_path, corrupted)
    _assert_public_code(
        lambda: app.start_study(
            plan=plan["id"], backend="fake", execution_key=execution_fx.FRESH_UUID_B.hex
        ),
        "validation_failed",
    )


def test_query_surface_is_read_only_and_backend_neutral(tmp_path: Path) -> None:
    root, plan = execution_fx._install_plan(tmp_path)
    backend = app_fx.DirectLogBackend()
    app = app_fx._application(tmp_path, backend)
    result = app.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    before = _snapshot(root)

    assert app.list_entities(resource=EntityKind.STUDY_RESULT)["items"]
    assert app.get_entity(
        kind=EntityKind.STUDY_RESULT, entity_id=result["id"]
    )["id"] == result["id"]
    assert app.list_study_results()["items"]
    assert app.get_study_result(study_result=result["id"])["study_result"]["id"] == result["id"]
    assert app.observe_study(study_result=result["id"])["study_result"]["id"] == result["id"]
    assert list(app.read_backend_logs(request={
        "study_result": result["id"], "trial": None, "coordinate": None,
        "failed_only": False, "follow": False,
    }))
    assert app.diagnose()

    assert _snapshot(root) == before
    assert backend.admit_calls == []
    assert backend.collect_calls == []
    assert backend.cancel_calls == []


def test_start_then_cancel_then_w006_progression_owns_terminal_closure(tmp_path: Path) -> None:
    _root, plan = execution_fx._install_plan(tmp_path)
    backend = app_fx.FakeBackend()
    app = app_fx._application(tmp_path, backend)

    started = app.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    assert backend.admit_calls == []
    assert appmod._advance_study is study_driver.advance_study

    cancelled = app.cancel_study(study_result=started["id"])
    assert cancelled["outcome"] == "accepted"
    assert cancelled["status"] == "cancelling"
    assert backend.admit_calls == []

    response = app.advance_study(study_result=started["id"])
    current = app.get_study_result(study_result=started["id"])["study_result"]
    assert response["terminal"] is True
    assert response["status"] == "cancelled"
    assert current["status"] == "cancelled"
    assert backend.admit_calls == []


def test_unsupported_logs_is_exact_public_capability_failure(tmp_path: Path) -> None:
    _root, plan = execution_fx._install_plan(tmp_path)
    app = app_fx._application(tmp_path, app_fx.FakeBackend())
    result = app.start_study(
        plan=plan["id"], backend="fake", execution_key=execution_fx.SOURCE_KEY
    )
    _assert_public_code(lambda: list(app.read_backend_logs(request={
        "study_result": result["id"], "trial": None, "coordinate": None,
        "failed_only": False, "follow": False,
    })), "unsupported_capability")
