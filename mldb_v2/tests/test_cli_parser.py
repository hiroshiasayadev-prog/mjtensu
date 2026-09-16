from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mldb_v2.src.cli.parser import parse_cli_request
from mldb_v2.src.cli.types import (
    CliCommandName,
    CliDefinitionKind,
    CliResource,
    OutputFormat,
)


REPO = Path(__file__).resolve().parents[2]


def test_public_command_resource_kind_and_output_spellings_are_exact() -> None:
    assert [item.value for item in CliCommandName] == [
        "ps", "get", "describe", "status", "validate", "verify", "seal", "plan",
        "run", "resume", "rerun", "cancel", "advance", "watch", "logs", "doctor",
    ]
    assert [item.value for item in CliResource] == [
        "namespaces", "definitions", "tasks", "corpora", "architectures",
        "train-protocols", "evaluation-protocols", "studies", "plans", "runs",
        "training-results", "models", "evaluation-results",
    ]
    assert [item.value for item in CliDefinitionKind] == [
        "task", "corpus", "architecture", "train-protocol", "evaluation-protocol", "study",
    ]
    assert [item.value for item in OutputFormat] == ["table", "wide", "json", "yaml"]


def test_ps_normalizes_shared_selectors_all_and_json_alias() -> None:
    assert parse_cli_request([
        "ps", "--all", "--namespace", "demo", "--status", "submitted",
        "--study", "demo/study-v1", "--since", "1h", "--limit", "5", "--json",
    ]) == {
        "command": CliCommandName.PS,
        "all": True,
        "selectors": {
            "namespace": "demo", "status": "submitted", "study": "demo/study-v1",
            "since": "1h", "limit": 5,
        },
        "presentation": {"format": OutputFormat.JSON},
    }


def test_get_and_describe_normalize_exact_targets_and_list_selectors() -> None:
    assert parse_cli_request(["get", "tasks", "demo/task-v1", "-o", "yaml"]) == {
        "command": CliCommandName.GET,
        "resource": CliResource.TASKS,
        "typed_id": "demo/task-v1",
        "presentation": {"format": OutputFormat.YAML},
    }
    assert parse_cli_request(["get", "definitions", "--namespace", "demo", "--status", "sealed"]) == {
        "command": CliCommandName.GET,
        "resource": CliResource.DEFINITIONS,
        "selectors": {"namespace": "demo", "status": "sealed"},
    }
    assert parse_cli_request(["describe", "namespaces", "demo"]) == {
        "command": CliCommandName.DESCRIBE,
        "resource": CliResource.NAMESPACES,
        "typed_id": "demo",
    }


def test_get_runs_accepts_execution_selectors() -> None:
    assert parse_cli_request([
        "get", "runs", "--namespace", "demo", "--status", "failed",
        "--study", "demo/study-v1", "--since", "2026-09-01T00:00:00Z", "--limit", "7",
    ]) == {
        "command": CliCommandName.GET,
        "resource": CliResource.RUNS,
        "selectors": {
            "namespace": "demo", "status": "failed", "study": "demo/study-v1",
            "since": "2026-09-01T00:00:00Z", "limit": 7,
        },
    }


@pytest.mark.parametrize("argv", [
    ["get", "definitions", "demo/task-v1"],
    ["describe", "definitions", "demo/task-v1"],
    ["get", "models", "--status", "sealed"],
    ["get", "tasks", "demo/task-v1", "--namespace", "demo"],
    ["watch", "demo/run-abc", "--status", "submitted"],
])
def test_unsupported_or_ambiguous_selector_target_combinations_reject(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_cli_request(argv)


@pytest.mark.parametrize("argv", [
    ["ps", "--limit", "0"],
    ["ps", "--limit", "-1"],
    ["ps", "--limit", "nope"],
    ["ps", "--status", "draft"],
    ["get", "tasks", "--status", "submitted"],
])
def test_invalid_limit_and_status_values_reject(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_cli_request(argv)


def test_validate_verify_normalize_scope_fail_fast_and_output() -> None:
    assert parse_cli_request(["validate"]) == {
        "command": CliCommandName.VALIDATE,
        "fail_fast": False,
    }
    assert parse_cli_request([
        "verify", "study", "demo/study-v1", "--namespace", "demo", "--fail-fast", "-o", "wide",
    ]) == {
        "command": CliCommandName.VERIFY,
        "kind": CliDefinitionKind.STUDY,
        "typed_id": "demo/study-v1",
        "namespace": "demo",
        "fail_fast": True,
        "presentation": {"format": OutputFormat.WIDE},
    }


def test_validate_rejects_namespace_mismatch_and_missing_explicit_kind() -> None:
    with pytest.raises(ValueError):
        parse_cli_request(["validate", "task", "demo/task-v1", "--namespace", "other"])
    with pytest.raises(ValueError):
        parse_cli_request(["validate", "demo/task-v1"])


def test_seal_distinguishes_exact_from_explicit_bulk() -> None:
    assert parse_cli_request(["seal", "task", "demo/task-v1"]) == {
        "command": CliCommandName.SEAL,
        "kind": CliDefinitionKind.TASK,
        "typed_id": "demo/task-v1",
    }
    assert parse_cli_request(["seal", "--namespace", "demo", "--all"]) == {
        "command": CliCommandName.SEAL,
        "namespace": "demo",
        "all": True,
    }
    assert parse_cli_request(["seal", "study", "--namespace", "demo", "--all"]) == {
        "command": CliCommandName.SEAL,
        "kind": CliDefinitionKind.STUDY,
        "namespace": "demo",
        "all": True,
    }


@pytest.mark.parametrize("argv", [
    ["seal", "task", "--namespace", "demo"],
    ["seal", "--all"],
    ["seal", "task", "demo/task-v1", "--all"],
    ["seal", "task", "demo/task-v1", "--namespace", "demo"],
])
def test_seal_rejects_implicit_or_mixed_bulk_forms(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_cli_request(argv)


def test_execution_commands_normalize_typed_ids_and_backend() -> None:
    assert parse_cli_request(["plan", "demo/study-v1"]) == {
        "command": CliCommandName.PLAN, "study": "demo/study-v1",
    }
    assert parse_cli_request(["run", "demo/study-v1", "--backend", "clearml"]) == {
        "command": CliCommandName.RUN, "study": "demo/study-v1", "backend": "clearml",
    }
    assert parse_cli_request(["resume", "demo/run-v1"]) == {
        "command": CliCommandName.RESUME, "study_result": "demo/run-v1",
    }
    assert parse_cli_request(["rerun", "demo/run-v1", "--backend", "local"]) == {
        "command": CliCommandName.RERUN, "study_result": "demo/run-v1", "backend": "local",
    }
    assert parse_cli_request(["cancel", "demo/run-v1"]) == {
        "command": CliCommandName.CANCEL, "study_result": "demo/run-v1",
    }
    assert parse_cli_request(["advance", "demo/run-v1"]) == {
        "command": CliCommandName.ADVANCE, "study_result": "demo/run-v1",
    }


def test_status_watch_logs_and_doctor_normalize_monitoring_options() -> None:
    assert parse_cli_request(["status", "demo/run-v1", "--json"]) == {
        "command": CliCommandName.STATUS,
        "study_result": "demo/run-v1",
        "presentation": {"format": OutputFormat.JSON},
    }
    assert parse_cli_request(["watch", "--namespace", "demo", "--limit", "3"]) == {
        "command": CliCommandName.WATCH,
        "selectors": {"namespace": "demo", "limit": 3},
    }
    assert parse_cli_request([
        "logs", "demo/run-v1", "--trial", "trial-0001", "--stage", "eval-0002",
        "--failed", "-f", "-o", "table",
    ]) == {
        "command": CliCommandName.LOGS,
        "study_result": "demo/run-v1",
        "trial": "trial-0001",
        "stage": "eval-0002",
        "failed_only": True,
        "follow": True,
        "presentation": {"format": OutputFormat.TABLE},
    }
    assert parse_cli_request(["doctor", "-o", "json"]) == {
        "command": CliCommandName.DOCTOR,
        "presentation": {"format": OutputFormat.JSON},
    }


def test_json_alias_conflicts_only_with_non_json_output() -> None:
    assert parse_cli_request(["doctor", "--json", "-o", "json"])["presentation"] == {
        "format": OutputFormat.JSON,
    }
    with pytest.raises(ValueError):
        parse_cli_request(["doctor", "--json", "-o", "yaml"])


@pytest.mark.parametrize("argv", [
    [],
    ["unknown"],
    ["get", "unknown-resource"],
    ["validate", "unknown-kind"],
    ["plan"],
    ["plan", "not-a-typed-reference"],
    ["status", "not-a-typed-reference"],
    ["logs", "demo/run-v1", "--trial", "trial-0000"],
    ["logs", "demo/run-v1", "--stage", "eval-0000"],
    ["get", "training-results", "--limit", "2"],
])
def test_invalid_or_ambiguous_argv_rejects(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_cli_request(argv)


def test_public_types_and_requests_are_exact_frozen_mirrors() -> None:
    for relative in ("cli/types.py", "cli/requests.py"):
        runtime = (REPO / "mldb_v2/src" / relative).read_text(encoding="utf-8")
        skeleton = (REPO / "mldb_v2/skeleton" / relative).read_text(encoding="utf-8")
        expected = skeleton.replace("mldb_v2.skeleton.", "mldb_v2.src.")
        assert runtime.replace("\r\n", "\n") == expected.replace("\r\n", "\n")


def test_cli_runtime_has_no_skeleton_or_infrastructure_imports() -> None:
    cli_root = REPO / "mldb_v2/src/cli"
    for path in (cli_root / "types.py", cli_root / "requests.py", cli_root / "parser.py"):
        source = path.read_text(encoding="utf-8")
        assert "mldb_v2.skeleton" not in source

    tree = ast.parse((cli_root / "parser.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    forbidden = (
        "mldb_v2.src.repository",
        "mldb_v2.src.backend",
        "clearml",
        "yaml",
    )
    assert not any(name.startswith(prefix) for name in imported for prefix in forbidden)
