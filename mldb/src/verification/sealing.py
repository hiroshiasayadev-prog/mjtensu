"""Public Python signatures for executable-asset sealing verification.

This module fixes only the asset-specific pytest gate used while sealing Architecture,
Train Protocol, and Evaluation Protocol definitions. It intentionally does not derive
repository paths, resolve or load executable assets, import pytest, implement subprocess
execution, calculate implementation hashes, mutate lifecycle state, persist metadata,
or participate in ordinary Training/Evaluation/Model/Study execution.

Canonical test-directory derivation belongs to the repository-layout boundary. The
verification operations therefore receive an already-derived ``test_dir`` and never
encode ``mldb_tests/<kind>/<id>/`` path grammar themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..common.ids import ArchitectureId, EvaluationProtocolId, TrainProtocolId
from ..repository.ports import FilesystemPort


@dataclass(frozen=True, slots=True)
class PytestRunResult:
    """Small runner-produced summary of one pytest invocation.

    ``invocation_completed`` reports whether the runner successfully obtained a pytest
    session outcome. It is ``False`` for unexpected tooling failures such as inability
    to start the configured Python/pytest process; it does not mean that collected tests
    passed.

    ``collected_count`` is the number of asset tests collected for this invocation.
    ``passed_count``, ``failed_count``, and ``error_count`` summarize the relevant test
    outcomes without exposing pytest result objects, exit-code conventions, stdout,
    stderr, subprocess handles, command strings, or environment-management details.

    ``diagnostics`` contains only concise caller-facing detail suitable for a sealing
    failure report. Full pytest output remains adapter/tooling-local.
    """

    invocation_completed: bool
    collected_count: int
    passed_count: int
    failed_count: int
    error_count: int
    diagnostics: tuple[str, ...] = ()


class PytestRunner(Protocol):
    """Consumer-owned port for executing pytest against one derived test directory.

    A concrete sealing/tooling adapter may implement this by invoking
    ``python -m pytest <test_dir>``. The command, subprocess API, Python executable,
    environment, CI integration, and pytest dependency remain implementation details.

    The runner receives only the test directory. It does not receive or inject an
    Architecture build callable, Train entrypoint, Evaluation entrypoint, runtime
    handle, or implementation module. Asset tests themselves resolve their target
    through the normal runtime resolution/executable-loading boundary.
    """

    def run(self, test_dir: Path) -> PytestRunResult:
        """Run the asset test session rooted at ``test_dir`` and summarize it."""
        ...


@dataclass(frozen=True, slots=True)
class ExecutableAssetVerificationResult:
    """Sealing-gate outcome for one executable asset's mandatory pytest directory.

    ``test_dir`` is the canonical directory already derived by repository-layout
    tooling for the exact typed asset ID supplied to the verification operation.
    Verification does not re-derive or reinterpret that path.

    ``test_dir_exists`` is explicit because a missing mandatory directory is a normal
    verification failure and must never be confused with zero collected tests.
    ``pytest`` is ``None`` when no pytest session was run, including the missing-directory
    case; otherwise it contains the small invocation summary returned by
    :class:`PytestRunner`.

    ``successful`` is true exactly when the directory exists, a pytest result is
    present, invocation completed, at least one test was collected, at least one test
    passed, and both failed/error counts are zero. Statuses not represented by
    :class:`PytestRunResult` (for example skipped or xfailed tests) do not independently
    make the gate fail. Consequently an all-skipped session fails because
    ``passed_count == 0``, while a session such as one passed plus skipped tests may
    succeed when the represented failure/error conditions are also clear.

    Equivalently, the gate is:
    ``test_dir_exists and pytest is not None and pytest.invocation_completed and
    pytest.collected_count > 0 and pytest.passed_count > 0 and
    pytest.failed_count == 0 and pytest.error_count == 0``.

    This value is intentionally distinct from ``common.errors.ValidationReport``:
    pytest process/test-session outcome is not a collection of static metadata
    validation issues.
    """

    test_dir: Path
    test_dir_exists: bool
    pytest: PytestRunResult | None

    @property
    def successful(self) -> bool:
        """Whether this result satisfies the executable-asset pytest sealing gate."""
        return (
            self.test_dir_exists
            and self.pytest is not None
            and self.pytest.invocation_completed
            and self.pytest.collected_count > 0
            and self.pytest.passed_count > 0
            and self.pytest.failed_count == 0
            and self.pytest.error_count == 0
        )


def _verify_test_dir(
    test_dir: Path,
    filesystem: FilesystemPort,
    runner: PytestRunner,
) -> ExecutableAssetVerificationResult:
    if not filesystem.directory_exists(test_dir):
        return ExecutableAssetVerificationResult(
            test_dir=test_dir, test_dir_exists=False, pytest=None
        )
    return ExecutableAssetVerificationResult(
        test_dir=test_dir, test_dir_exists=True, pytest=runner.run(test_dir)
    )


def verify_architecture_tests(
    architecture_id: ArchitectureId,
    test_dir: Path,
    filesystem: FilesystemPort,
    runner: PytestRunner,
) -> ExecutableAssetVerificationResult:
    """Verify the mandatory asset-specific tests for one Architecture.

    ``architecture_id`` preserves the frozen typed Architecture identity at this
    boundary. ``test_dir`` must already be the canonical Architecture test directory
    derived for that exact ID by repository-layout tooling; this operation neither
    derives path grammar nor scans alternative locations.

    ``filesystem`` is used only to determine whether ``test_dir`` exists as a directory;
    verification performs no direct physical filesystem access. A missing directory
    returns an unsuccessful result without invoking ``runner``. Otherwise the operation
    runs exactly that directory through ``runner`` and returns the sealing-gate result.
    The operation does not resolve/load the Architecture, import its sibling
    implementation, calculate SHA-256, or mutate sealing state.
    """

    return _verify_test_dir(test_dir, filesystem, runner)


def verify_train_protocol_tests(
    protocol_id: TrainProtocolId,
    test_dir: Path,
    filesystem: FilesystemPort,
    runner: PytestRunner,
) -> ExecutableAssetVerificationResult:
    """Verify the mandatory asset-specific tests for one Train Protocol.

    ``protocol_id`` preserves the frozen typed Train Protocol identity at this boundary.
    ``test_dir`` must already be the canonical Train Protocol test directory derived for
    that exact ID by repository-layout tooling; this operation neither derives path
    grammar nor scans alternative locations.

    ``filesystem`` is used only to determine whether ``test_dir`` exists as a directory;
    verification performs no direct physical filesystem access. A missing directory
    returns an unsuccessful result without invoking ``runner``. Otherwise the operation
    runs exactly that directory through ``runner`` and returns the sealing-gate result.
    The operation does not resolve/load the Train Protocol, import its sibling
    implementation, calculate SHA-256, or mutate sealing state.
    """

    return _verify_test_dir(test_dir, filesystem, runner)


def verify_evaluation_protocol_tests(
    protocol_id: EvaluationProtocolId,
    test_dir: Path,
    filesystem: FilesystemPort,
    runner: PytestRunner,
) -> ExecutableAssetVerificationResult:
    """Verify the mandatory asset-specific tests for one Evaluation Protocol.

    ``protocol_id`` preserves the frozen typed Evaluation Protocol identity at this
    boundary. ``test_dir`` must already be the canonical Evaluation Protocol test
    directory derived for that exact ID by repository-layout tooling; this operation
    neither derives path grammar nor scans alternative locations.

    ``filesystem`` is used only to determine whether ``test_dir`` exists as a directory;
    verification performs no direct physical filesystem access. A missing directory
    returns an unsuccessful result without invoking ``runner``. Otherwise the operation
    runs exactly that directory through ``runner`` and returns the sealing-gate result.
    The operation does not resolve/load the Evaluation Protocol, import its sibling
    implementation, calculate SHA-256, or mutate sealing state.
    """

    return _verify_test_dir(test_dir, filesystem, runner)
