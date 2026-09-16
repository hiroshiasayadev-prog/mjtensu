from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from mldb_v2.src.common.diagnostic import _validate_diagnostic
from mldb_v2.src.verification._executable_asset_tests import (
    _RepositoryExecutableAssetTestVerifier,
    _SubprocessPytestRunner,
    _default_pytest_runner,
)
from mldb_v2.src.verification.executable_asset_tests import (
    ExecutableAssetTestRequest,
    ExecutableAssetTestResult,
    ExecutableAssetTestVerifier,
    PytestRunner,
    PytestRunResult,
)


class _FakeIntegrityVerifier:
    def __init__(self, result: object | None = None, *, error: Exception | None = None) -> None:
        self.result = {"valid": True, "diagnostics": []} if result is None else result
        self.error = error
        self.calls: list[dict[str, object]] = []

    def verify_executable_definition(self, *, request):
        self.calls.append(dict(request))
        if self.error is not None:
            raise self.error
        return self.result

    def verify_corpus_builder(self, *, request):
        raise AssertionError("Corpus builder verification is out of scope")


class _FakeRunner:
    def __init__(self, result: object | None = None, *, error: Exception | None = None) -> None:
        self.result = (
            {"collected": 1, "passed": 1, "failed": 0, "errors": 0}
            if result is None
            else result
        )
        self.error = error
        self.calls: list[Path] = []

    def run(self, *, test_directory: Path):
        self.calls.append(test_directory)
        if self.error is not None:
            raise self.error
        return self.result


def _verifier(tmp_path: Path, integrity: _FakeIntegrityVerifier | None = None):
    root = tmp_path / "mldb_tests"
    root.mkdir()
    return (
        _RepositoryExecutableAssetTestVerifier(
            mldb_tests_root=root,
            integrity_verifier=integrity or _FakeIntegrityVerifier(),
        ),
        root,
    )


def _request(kind: str, entity_id: str) -> ExecutableAssetTestRequest:
    return {"kind": kind, "id": entity_id}  # type: ignore[typeddict-item]


def _codes(result: ExecutableAssetTestResult) -> list[str]:
    for diagnostic in result["diagnostics"]:
        assert _validate_diagnostic(diagnostic) == diagnostic
    return [diagnostic["code"] for diagnostic in result["diagnostics"]]


@pytest.mark.parametrize(
    ("kind", "entity_id", "domain", "local_id"),
    [
        ("architecture", "demo/model-v1", "architectures", "model-v1"),
        ("train_protocol", "demo/train-v2", "train_protocols", "train-v2"),
        ("evaluation_protocol", "demo/eval-v3", "evaluation_protocols", "eval-v3"),
    ],
)
def test_exact_canonical_directory_is_derived_from_kind_and_id(
    tmp_path: Path, kind: str, entity_id: str, domain: str, local_id: str
) -> None:
    verifier, root = _verifier(tmp_path)
    directory = root / "demo" / domain / local_id
    directory.mkdir(parents=True)
    runner = _FakeRunner()

    result = verifier.verify(request=_request(kind, entity_id), runner=runner)

    assert result == {
        "valid": True,
        "run": {"collected": 1, "passed": 1, "failed": 0, "errors": 0},
        "diagnostics": [],
    }
    assert runner.calls == [directory.absolute()]


@pytest.mark.parametrize(
    ("raw_request", "code"),
    [
        ({"kind": "corpus", "id": "demo/model-v1"}, "asset_test_request_invalid"),
        ({"kind": "architecture", "id": "demo/model"}, "asset_test_request_invalid"),
        ({"kind": "architecture", "id": "../model-v1"}, "asset_test_request_invalid"),
        ({"kind": "architecture", "id": "demo/../model-v1"}, "asset_test_request_invalid"),
        ({"kind": "architecture", "id": "C:/model-v1"}, "asset_test_request_invalid"),
    ],
)
def test_malformed_kind_or_id_rejected_before_integrity_and_runner(
    tmp_path: Path, raw_request: dict[str, object], code: str
) -> None:
    integrity = _FakeIntegrityVerifier()
    verifier, _root = _verifier(tmp_path, integrity)
    runner = _FakeRunner()

    result = verifier.verify(request=raw_request, runner=runner)  # type: ignore[arg-type]

    assert _codes(result) == [code]
    assert integrity.calls == []
    assert runner.calls == []


def test_integrity_invalid_short_circuits_runner_and_preserves_cause(tmp_path: Path) -> None:
    integrity = _FakeIntegrityVerifier(
        {"valid": False, "diagnostics": [{"code": "companion_hash_mismatch", "message": "mismatch"}]}
    )
    verifier, root = _verifier(tmp_path, integrity)
    (root / "demo" / "architectures" / "model-v1").mkdir(parents=True)
    runner = _FakeRunner()

    result = verifier.verify(request=_request("architecture", "demo/model-v1"), runner=runner)

    assert _codes(result) == ["executable_integrity_failed", "companion_hash_mismatch"]
    assert runner.calls == []


def test_integrity_exception_is_clean_invalid_and_runner_not_called(tmp_path: Path) -> None:
    verifier, root = _verifier(
        tmp_path, _FakeIntegrityVerifier(error=RuntimeError("secret stack details"))
    )
    (root / "demo" / "architectures" / "model-v1").mkdir(parents=True)
    runner = _FakeRunner()

    result = verifier.verify(request=_request("architecture", "demo/model-v1"), runner=runner)

    assert _codes(result) == ["executable_integrity_failed"]
    assert "secret" not in result["diagnostics"][0]["message"]
    assert runner.calls == []


def test_missing_directory_is_rejected(tmp_path: Path) -> None:
    verifier, _root = _verifier(tmp_path)
    runner = _FakeRunner()
    result = verifier.verify(request=_request("architecture", "demo/model-v1"), runner=runner)
    assert _codes(result) == ["asset_test_directory_missing"]
    assert runner.calls == []


def test_file_instead_of_directory_is_rejected(tmp_path: Path) -> None:
    verifier, root = _verifier(tmp_path)
    path = root / "demo" / "architectures" / "model-v1"
    path.parent.mkdir(parents=True)
    path.write_text("not a directory", encoding="utf-8")
    runner = _FakeRunner()
    result = verifier.verify(request=_request("architecture", "demo/model-v1"), runner=runner)
    assert _codes(result) == ["asset_test_directory_invalid"]
    assert runner.calls == []


def test_symlinked_directory_escaping_root_is_rejected_when_supported(tmp_path: Path) -> None:
    verifier, root = _verifier(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "demo" / "architectures" / "model-v1"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    runner = _FakeRunner()
    result = verifier.verify(request=_request("architecture", "demo/model-v1"), runner=runner)
    assert _codes(result) == ["asset_test_directory_invalid"]
    assert runner.calls == []


@pytest.mark.parametrize(
    ("run", "codes"),
    [
        ({"collected": 1, "passed": 1, "failed": 0, "errors": 0}, []),
        ({"collected": 0, "passed": 0, "failed": 0, "errors": 0},
         ["asset_test_zero_collected", "asset_test_no_passed"]),
        ({"collected": 3, "passed": 0, "failed": 0, "errors": 0}, ["asset_test_no_passed"]),
        ({"collected": 2, "passed": 1, "failed": 1, "errors": 0}, ["asset_test_failed"]),
        ({"collected": 2, "passed": 1, "failed": 0, "errors": 1}, ["asset_test_error"]),
        ({"collected": 3, "passed": 1, "failed": 1, "errors": 1},
         ["asset_test_failed", "asset_test_error"]),
    ],
)
def test_gate_semantics_for_valid_zero_skipped_failure_and_error_counts(
    tmp_path: Path, run: dict[str, int], codes: list[str]
) -> None:
    verifier, root = _verifier(tmp_path)
    (root / "demo" / "architectures" / "model-v1").mkdir(parents=True)
    runner = _FakeRunner(run)
    result = verifier.verify(request=_request("architecture", "demo/model-v1"), runner=runner)
    assert result["run"] == run
    assert result["valid"] is (not codes)
    assert _codes(result) == codes


@pytest.mark.parametrize(
    "run",
    [
        {"collected": -1, "passed": 0, "failed": 0, "errors": 0},
        {"collected": 1, "passed": True, "failed": 0, "errors": 0},
        {"collected": 1, "passed": 1, "failed": 1, "errors": 0},
        {"collected": 1, "passed": 1, "failed": 0},
        {"collected": 1, "passed": 1, "failed": 0, "errors": 0, "extra": 0},
    ],
)
def test_malformed_runner_counts_are_rejected(tmp_path: Path, run: object) -> None:
    verifier, root = _verifier(tmp_path)
    (root / "demo" / "architectures" / "model-v1").mkdir(parents=True)
    result = verifier.verify(
        request=_request("architecture", "demo/model-v1"),
        runner=_FakeRunner(run),
    )
    assert result["run"] is None
    assert _codes(result) == ["asset_test_invalid_result"]


def test_runner_exception_is_clean_invalid(tmp_path: Path) -> None:
    verifier, root = _verifier(tmp_path)
    (root / "demo" / "architectures" / "model-v1").mkdir(parents=True)
    result = verifier.verify(
        request=_request("architecture", "demo/model-v1"),
        runner=_FakeRunner(error=RuntimeError("boom")),
    )
    assert result["run"] is None
    assert _codes(result) == ["asset_test_runner_failed"]


def _write_test(directory: Path, source: str) -> None:
    directory.mkdir(parents=True)
    (directory / "test_asset.py").write_text(source, encoding="utf-8")


def test_real_runner_passing_directory(tmp_path: Path) -> None:
    directory = tmp_path / "passing"
    _write_test(directory, "def test_ok():\n    assert True\n")
    assert _SubprocessPytestRunner().run(test_directory=directory) == {
        "collected": 1, "passed": 1, "failed": 0, "errors": 0
    }


def test_real_runner_failing_directory(tmp_path: Path) -> None:
    directory = tmp_path / "failing"
    _write_test(directory, "def test_bad():\n    assert False\n")
    assert _SubprocessPytestRunner().run(test_directory=directory) == {
        "collected": 1, "passed": 0, "failed": 1, "errors": 0
    }


def test_real_runner_collection_error(tmp_path: Path) -> None:
    directory = tmp_path / "collection-error"
    _write_test(directory, "raise RuntimeError('collection boom')\n")
    assert _SubprocessPytestRunner().run(test_directory=directory) == {
        "collected": 1, "passed": 0, "failed": 0, "errors": 1
    }


def test_real_runner_zero_directory(tmp_path: Path) -> None:
    directory = tmp_path / "zero"
    directory.mkdir()
    assert _SubprocessPytestRunner().run(test_directory=directory) == {
        "collected": 0, "passed": 0, "failed": 0, "errors": 0
    }


def test_real_runner_skipped_only_and_gate_rejects_it(tmp_path: Path) -> None:
    verifier, root = _verifier(tmp_path)
    directory = root / "demo" / "architectures" / "model-v1"
    _write_test(directory, "import pytest\n\ndef test_skip():\n    pytest.skip('not applicable')\n")
    result = verifier.verify(
        request=_request("architecture", "demo/model-v1"),
        runner=_SubprocessPytestRunner(),
    )
    assert result["run"] == {"collected": 1, "passed": 0, "failed": 0, "errors": 0}
    assert _codes(result) == ["asset_test_no_passed"]


def test_default_runner_is_private_subprocess_implementation() -> None:
    assert isinstance(_default_pytest_runner(), _SubprocessPytestRunner)


def test_public_executable_asset_test_shape_is_exact_skeleton_mirror() -> None:
    repo = Path(__file__).resolve().parents[2]
    runtime = (repo / "mldb_v2" / "src" / "verification" / "executable_asset_tests.py").read_text(
        encoding="utf-8"
    )
    skeleton = (
        repo / "mldb_v2" / "skeleton" / "verification" / "executable_asset_tests.py"
    ).read_text(encoding="utf-8")
    normalized_skeleton = skeleton.replace("mldb_v2.skeleton.", "mldb_v2.src.")
    assert runtime.replace("\r\n", "\n") == normalized_skeleton.replace("\r\n", "\n")
    assert ExecutableAssetTestRequest.__required_keys__ == frozenset({"kind", "id"})
    assert PytestRunResult.__required_keys__ == frozenset(
        {"collected", "passed", "failed", "errors"}
    )
    assert ExecutableAssetTestResult.__required_keys__ == frozenset(
        {"valid", "run", "diagnostics"}
    )
    assert list(inspect.signature(PytestRunner.run).parameters) == ["self", "test_directory"]
    assert list(inspect.signature(ExecutableAssetTestVerifier.verify).parameters) == [
        "self", "request", "runner"
    ]


def test_asset_gate_has_no_forbidden_runtime_dependencies_or_direct_companion_import() -> None:
    repo = Path(__file__).resolve().parents[2]
    sources = "\n".join(
        (
            (repo / "mldb_v2" / "src" / "verification" / "executable_asset_tests.py").read_text(
                encoding="utf-8"
            ),
            (repo / "mldb_v2" / "src" / "verification" / "_executable_asset_tests.py").read_text(
                encoding="utf-8"
            ),
        )
    )
    forbidden = (
        "mldb_v2.skeleton",
        "mldb.src",
        "clearml",
        "boto",
        "minio",
        "tools.",
        "importlib.spec_from_file_location",
    )
    for token in forbidden:
        assert token not in sources.lower()
