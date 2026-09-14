"""Entity-level validate / verify / seal contracts for reusable definitions."""

from typing import Literal, Protocol, TypeAlias, TypedDict

from mldb_v2.src.catalog.architecture import Architecture
from mldb_v2.src.catalog.corpus import Corpus
from mldb_v2.src.catalog.task import Task
from mldb_v2.src.common.diagnostic import Diagnostic
from mldb_v2.src.common.ids import (
    ArchitectureId,
    CorpusId,
    DefinitionKind,
    EvaluationProtocolId,
    StudyId,
    TaskId,
    TrainProtocolId,
)
from mldb_v2.src.evaluation.evaluation_protocol import EvaluationProtocol
from mldb_v2.src.study.study import Study
from mldb_v2.src.training.train_protocol import TrainProtocol

DefinitionLifecycleStatus: TypeAlias = Literal["draft", "sealed"]
SealableDefinitionId: TypeAlias = (
    TaskId | CorpusId | ArchitectureId | TrainProtocolId | EvaluationProtocolId | StudyId
)
SealableDefinition: TypeAlias = (
    Task | Corpus | Architecture | TrainProtocol | EvaluationProtocol | Study
)


class DefinitionValidationRequest(TypedDict):
    kind: DefinitionKind
    id: SealableDefinitionId


class DefinitionValidationResult(TypedDict):
    valid: bool
    diagnostics: list[Diagnostic]


class DefinitionVerificationRequest(TypedDict):
    kind: DefinitionKind
    id: SealableDefinitionId


class DefinitionVerificationResult(TypedDict):
    valid: bool
    diagnostics: list[Diagnostic]


class DefinitionSealingRequest(TypedDict):
    kind: DefinitionKind
    id: SealableDefinitionId


class DefinitionSealingResult(TypedDict):
    definition: SealableDefinition


class DefinitionValidator(Protocol):
    def validate(
        self,
        *,
        request: DefinitionValidationRequest,
    ) -> DefinitionValidationResult: ...


class DefinitionVerifier(Protocol):
    def verify(
        self,
        *,
        request: DefinitionVerificationRequest,
    ) -> DefinitionVerificationResult: ...


class DefinitionSealer(Protocol):
    def seal(
        self,
        *,
        request: DefinitionSealingRequest,
    ) -> DefinitionSealingResult: ...
