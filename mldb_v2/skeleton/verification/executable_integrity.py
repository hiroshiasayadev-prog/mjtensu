"""Public verification boundary for reusable executable-source integrity."""

from typing import Literal, Protocol, TypeAlias, TypedDict

from mldb_v2.skeleton.common.diagnostic import Diagnostic
from mldb_v2.skeleton.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    TrainProtocolId,
)


class ExecutableSource(TypedDict):
    path: str
    sha256: str


ExecutableDefinitionKind: TypeAlias = Literal[
    "architecture",
    "train_protocol",
    "evaluation_protocol",
]
ExecutableDefinitionId: TypeAlias = ArchitectureId | TrainProtocolId | EvaluationProtocolId


class ExecutableDefinitionIntegrityRequest(TypedDict):
    kind: ExecutableDefinitionKind
    id: ExecutableDefinitionId


class CorpusBuilderIntegrityRequest(TypedDict):
    corpus: CorpusId


class ExecutableIntegrityResult(TypedDict):
    valid: bool
    diagnostics: list[Diagnostic]


class ExecutableIntegrityVerifier(Protocol):
    """Verify recorded companion/source hashes without defining hashing or filesystem mechanics."""

    def verify_executable_definition(
        self,
        *,
        request: ExecutableDefinitionIntegrityRequest,
    ) -> ExecutableIntegrityResult: ...

    def verify_corpus_builder(
        self,
        *,
        request: CorpusBuilderIntegrityRequest,
    ) -> ExecutableIntegrityResult: ...
