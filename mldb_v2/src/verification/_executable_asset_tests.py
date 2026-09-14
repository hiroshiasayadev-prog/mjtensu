"""Private repository-backed executable asset pytest verification."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from mldb_v2.src.catalog._executable_definition_loading import _validate_definition_id
from mldb_v2.src.common.diagnostic import Diagnostic, _validate_diagnostic
from mldb_v2.src.verification.executable_asset_tests import (
    ExecutableAssetTestRequest,
    ExecutableAssetTestResult,
    PytestRunner,
    PytestRunResult,
)
from mldb_v2.src.verification.executable_integrity import ExecutableIntegrityVerifier


_ASSET_DOMAIN = {
    "architecture": "architectures",
    "train_protocol": "train_protocols",
    "evaluation_protocol": "evaluation_protocols",
}


def _diagnostic(code: str, message: str) -> Diagnostic:
    return {"code": code, "message": message}


def _validate_request(
    request: ExecutableAssetTestRequest,
) -> tuple[str, str] | None:
    if type(request) is not dict or set(request) != {"kind", "id"}:
        return None
    kind = request["kind"]
    entity_id = request["id"]
    if type(kind) is not str or kind not in _ASSET_DOMAIN or type(entity_id) is not str:
        return None
    try:
        entity_id = _validate_definition_id(entity_id)
    except ValueError:
        return None
    return kind, entity_id


def _validate_integrity_result(value: object) -> tuple[bool, list[Diagnostic]] | None:
    if type(value) is not dict or set(value) != {"valid", "diagnostics"}:
        return None
    valid = value["valid"]
    raw_diagnostics = value["diagnostics"]
    if type(valid) is not bool or type(raw_diagnostics) is not list:
        return None
    diagnostics: list[Diagnostic] = []
    try:
        for item in raw_diagnostics:
            diagnostic = _validate_diagnostic(item)
            if diagnostic is None:
                return None
            diagnostics.append(diagnostic)
    except ValueError:
        return None
    if valid and diagnostics:
        return None
    return valid, diagnostics


def _validate_run_result(value: object) -> PytestRunResult | None:
    required = {"collected", "passed", "failed", "errors"}
    if type(value) is not dict or set(value) != required:
        return None
    counts: dict[str, int] = {}
    for key in ("collected", "passed", "failed", "errors"):
        count = value[key]
        if type(count) is not int or count < 0:
            return None
        counts[key] = count
    if counts["passed"] + counts["failed"] + counts["errors"] > counts["collected"]:
        return None
    return {
        "collected": counts["collected"],
        "passed": counts["passed"],
        "failed": counts["failed"],
        "errors": counts["errors"],
    }


class _RepositoryExecutableAssetTestVerifier:
    """Read-only asset-test gate over one canonical mldb_tests root."""

    def __init__(
        self,
        *,
        mldb_tests_root: str | Path,
        integrity_verifier: ExecutableIntegrityVerifier,
    ) -> None:
        self._mldb_tests_root = Path(mldb_tests_root).absolute()
        self._integrity_verifier = integrity_verifier

    def _derived_directory(self, *, kind: str, entity_id: str) -> Path:
        namespace, local_id = entity_id.split("/", 1)
        return self._mldb_tests_root / namespace / _ASSET_DOMAIN[kind] / local_id

    def verify(
        self,
        *,
        request: ExecutableAssetTestRequest,
        runner: PytestRunner,
    ) -> ExecutableAssetTestResult:
        validated = _validate_request(request)
        if validated is None:
            return {
                "valid": False,
                "run": None,
                "diagnostics": [
                    _diagnostic(
                        "asset_test_request_invalid",
                        "executable asset-test request kind or id is invalid",
                    )
                ],
            }
        kind, entity_id = validated

        try:
            integrity_raw = self._integrity_verifier.verify_executable_definition(
                request={"kind": kind, "id": entity_id}
            )
        except Exception:
            return {
                "valid": False,
                "run": None,
                "diagnostics": [
                    _diagnostic(
                        "executable_integrity_failed",
                        "executable integrity prerequisite could not be established",
                    )
                ],
            }
        integrity = _validate_integrity_result(integrity_raw)
        if integrity is None:
            return {
                "valid": False,
                "run": None,
                "diagnostics": [
                    _diagnostic(
                        "executable_integrity_failed",
                        "executable integrity prerequisite returned an invalid result",
                    )
                ],
            }
        integrity_valid, integrity_diagnostics = integrity
        if not integrity_valid:
            return {
                "valid": False,
                "run": None,
                "diagnostics": [
                    _diagnostic(
                        "executable_integrity_failed",
                        "executable integrity prerequisite failed",
                    ),
                    *integrity_diagnostics,
                ],
            }

        test_directory = self._derived_directory(kind=kind, entity_id=entity_id)
        if not test_directory.exists():
            return {
                "valid": False,
                "run": None,
                "diagnostics": [
                    _diagnostic(
                        "asset_test_directory_missing",
                        "canonical executable asset-test directory is missing",
                    )
                ],
            }
        if not test_directory.is_dir():
            return {
                "valid": False,
                "run": None,
                "diagnostics": [
                    _diagnostic(
                        "asset_test_directory_invalid",
                        "canonical executable asset-test path must be a directory",
                    )
                ],
            }
        try:
            resolved_root = self._mldb_tests_root.resolve(strict=True)
            resolved_directory = test_directory.resolve(strict=True)
        except OSError:
            return {
                "valid": False,
                "run": None,
                "diagnostics": [
                    _diagnostic(
                        "asset_test_directory_invalid",
                        "canonical executable asset-test directory cannot be resolved",
                    )
                ],
            }
        if not resolved_directory.is_relative_to(resolved_root):
            return {
                "valid": False,
                "run": None,
                "diagnostics": [
                    _diagnostic(
                        "asset_test_directory_invalid",
                        "canonical executable asset-test directory escapes mldb_tests root",
                    )
                ],
            }

        try:
            run_raw = runner.run(test_directory=test_directory)
        except Exception:
            return {
                "valid": False,
                "run": None,
                "diagnostics": [
                    _diagnostic(
                        "asset_test_runner_failed",
                        "pytest runner failed while executing the canonical asset-test directory",
                    )
                ],
            }
        run = _validate_run_result(run_raw)
        if run is None:
            return {
                "valid": False,
                "run": None,
                "diagnostics": [
                    _diagnostic(
                        "asset_test_invalid_result",
                        "pytest runner returned invalid count data",
                    )
                ],
            }

        diagnostics: list[Diagnostic] = []
        if run["collected"] == 0:
            diagnostics.append(
                _diagnostic("asset_test_zero_collected", "pytest collected zero asset tests")
            )
        if run["passed"] == 0:
            diagnostics.append(
                _diagnostic("asset_test_no_passed", "no executable asset test passed")
            )
        if run["failed"] > 0:
            diagnostics.append(
                _diagnostic("asset_test_failed", "one or more executable asset tests failed")
            )
        if run["errors"] > 0:
            diagnostics.append(
                _diagnostic("asset_test_error", "one or more executable asset tests errored")
            )
        return {"valid": not diagnostics, "run": run, "diagnostics": diagnostics}


_PYTEST_WORKER = r"""
import json
import sys
from pathlib import Path

import pytest


class CountingPlugin:
    def __init__(self, result_path):
        self.result_path = Path(result_path)
        self.states = {}
        self.collection_errors = 0

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors += 1

    def pytest_runtest_logreport(self, report):
        nodeid = report.nodeid
        if report.when == "call":
            if report.passed:
                self.states[nodeid] = "passed"
            elif report.failed:
                self.states[nodeid] = "failed"
        elif report.failed:
            self.states[nodeid] = "errors"

    def pytest_sessionfinish(self, session, exitstatus):
        passed = sum(value == "passed" for value in self.states.values())
        failed = sum(value == "failed" for value in self.states.values())
        errors = sum(value == "errors" for value in self.states.values())
        errors += self.collection_errors
        collected = session.testscollected + self.collection_errors
        collected = max(collected, passed + failed + errors)
        self.result_path.write_text(
            json.dumps(
                {
                    "collected": collected,
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )


plugin = CountingPlugin(sys.argv[2])
pytest.main([sys.argv[1], "-q"], plugins=[plugin])
if not Path(sys.argv[2]).is_file():
    raise SystemExit(2)
"""


class _SubprocessPytestRunner:
    """Run exactly one directory in an isolated pytest subprocess."""

    def run(self, *, test_directory: Path) -> PytestRunResult:
        directory = Path(test_directory)
        if not directory.is_dir():
            raise ValueError("test_directory must be an existing directory")
        with tempfile.TemporaryDirectory(prefix="mldb-v2-pytest-") as temporary:
            result_path = Path(temporary) / "result.json"
            completed = subprocess.run(
                [sys.executable, "-c", _PYTEST_WORKER, str(directory), str(result_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode not in {0, 1, 5} and not result_path.is_file():
                raise RuntimeError("pytest subprocess failed before producing count data")
            try:
                raw = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RuntimeError("pytest subprocess did not produce valid count data") from error
        result = _validate_run_result(raw)
        if result is None:
            raise RuntimeError("pytest subprocess produced impossible count data")
        return result


def _default_pytest_runner() -> PytestRunner:
    return _SubprocessPytestRunner()
