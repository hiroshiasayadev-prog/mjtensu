import builtins
import json
import sys
from types import SimpleNamespace

import pytest

from mldb_v2.src.cli.exit_status import (
    EXIT_FAILURE,
    EXIT_SUCCESS,
    exit_status_for_application_error,
    exit_status_for_definition_report,
)
from mldb_v2.src.cli.rendering import (
    YamlSerializerUnavailable,
    render_application_error,
    render_success,
    render_value,
)


def _report(*, valid: bool = True, repository_issues: list[dict[str, str]] | None = None):
    return {
        "items": [
            {
                "kind": "task",
                "id": "task:tiles",
                "valid": valid,
                "diagnostics": [] if valid else [{"code": "bad", "message": "invalid"}],
            }
        ],
        "repository_issues": repository_issues or [],
    }


def test_json_list_remains_array_and_is_deterministic() -> None:
    value = [{"id": "task:b", "valid": True}, {"id": "task:a", "valid": False}]

    first = render_value(value, "json")
    second = render_value(value, "json")

    assert first == second
    assert json.loads(first) == value
    assert isinstance(json.loads(first), list)


def test_json_single_value_remains_object_and_preserves_fields() -> None:
    value = {"study_result": "sr:1", "status": "completed", "terminal": True}

    rendered = render_value(value, "json")

    assert json.loads(rendered) == value
    assert isinstance(json.loads(rendered), dict)


def test_report_structured_output_preserves_application_shape() -> None:
    report = _report(valid=False)

    rendered = render_value(report, "json")

    assert json.loads(rendered) == report
    assert set(json.loads(rendered)) == {"items", "repository_issues"}


def test_success_and_application_error_keep_stdout_stderr_separate() -> None:
    success = render_success({"id": "task:tiles"}, "json")
    error = render_application_error({"code": "not_found", "message": "missing target"})

    assert success.stdout
    assert success.stderr == ""
    assert error.stdout == ""
    assert error.stderr == "not_found: missing target\n"


def test_structured_json_has_no_human_decoration() -> None:
    rendered = render_success({"id": "task:tiles", "status": "ok"}, "json").stdout

    assert "\x1b[" not in rendered
    assert "spinner" not in rendered.lower()
    assert "progress" not in rendered.lower()
    assert "mldb" not in rendered.lower()
    assert json.loads(rendered) == {"id": "task:tiles", "status": "ok"}


def test_table_and_wide_are_deterministic_human_presentations() -> None:
    value = [{"status": "ok", "id": "task:tiles"}]

    assert render_value(value, "table") == render_value(value, "table")
    assert render_value(value, "wide") == render_value(value, "wide")
    assert "\x1b[" not in render_value(value, "table")


def test_yaml_uses_deterministic_safe_dump_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def safe_dump(value: object, **kwargs: object) -> str:
        calls.append(kwargs)
        return json.dumps(value, ensure_ascii=False, sort_keys=bool(kwargs["sort_keys"])) + "\n"

    monkeypatch.setitem(sys.modules, "yaml", SimpleNamespace(safe_dump=safe_dump))
    value = {"z": 1, "a": [{"id": "task:tiles", "valid": True}]}

    first = render_value(value, "yaml")
    second = render_value(value, "yaml")

    assert first == second
    assert json.loads(first) == value
    assert calls == [
        {"allow_unicode": True, "default_flow_style": False, "sort_keys": True},
        {"allow_unicode": True, "default_flow_style": False, "sort_keys": True},
    ]


def test_yaml_real_serializer_preserves_list_and_object_semantics() -> None:
    import yaml

    values = [
        [{"id": "task:b", "valid": True}, {"id": "task:a", "valid": False}],
        {"study_result": "sr:1", "status": "completed", "terminal": True},
    ]

    for value in values:
        first = render_value(value, "yaml")
        second = render_value(value, "yaml")
        assert first == second
        assert yaml.safe_load(first) == value
        assert "\x1b[" not in first


def test_yaml_without_serializer_reports_focused_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "yaml", raising=False)
    original_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(YamlSerializerUnavailable, match="YAML output requires"):
        render_value({"id": "task:tiles"}, "yaml")


def test_validate_verify_exit_nonzero_when_any_target_fails() -> None:
    assert exit_status_for_definition_report(_report(valid=True)) == EXIT_SUCCESS
    assert exit_status_for_definition_report(_report(valid=False)) == EXIT_FAILURE


def test_validate_verify_exit_nonzero_for_repository_issue() -> None:
    report = _report(
        valid=True,
        repository_issues=[{"code": "repository_invalid", "message": "broken root"}],
    )

    assert exit_status_for_definition_report(report) == EXIT_FAILURE


def test_application_error_exit_is_nonzero() -> None:
    error = {"code": "unsupported_capability", "message": "logs unavailable"}

    assert exit_status_for_application_error(error) == EXIT_FAILURE
