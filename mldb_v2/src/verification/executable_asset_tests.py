"""Public pytest gate boundary for executable reusable definitions."""

from pathlib import Path
from typing import Literal, Protocol, TypeAlias, TypedDict

from mldb_v2.src.common.diagnostic import Diagnostic
from mldb_v2.src.common.ids import ArchitectureId, EvaluationProtocolId, TrainProtocolId

ExecutableAssetKind: TypeAlias = Literal[
    "architecture",
    "train_protocol",
    "evaluation_protocol",
]
ExecutableAssetId: TypeAlias = ArchitectureId | TrainProtocolId | EvaluationProtocolId


class ExecutableAssetTestRequest(TypedDict):
    kind: ExecutableAssetKind
    id: ExecutableAssetId


class PytestRunResult(TypedDict):
    collected: int
    passed: int
    failed: int
    errors: int


class PytestRunner(Protocol):
    def run(self, *, test_directory: Path) -> PytestRunResult: ...


class ExecutableAssetTestResult(TypedDict):
    valid: bool
    run: PytestRunResult | None
    diagnostics: list[Diagnostic]


class ExecutableAssetTestVerifier(Protocol):
    """Verify the canonical derived test directory through an injected pytest runner."""

    def verify(
        self,
        *,
        request: ExecutableAssetTestRequest,
        runner: PytestRunner,
    ) -> ExecutableAssetTestResult: ...
