"""Read-only discovery and observation boundary for MLDB v2."""

from datetime import datetime
from typing import Collection, Iterable, Literal, Protocol, TypeAlias, TypedDict

from mldb_v2.skeleton.backend.candidate_outcome import BackendObservation
from mldb_v2.skeleton.common.diagnostic import Diagnostic
from mldb_v2.skeleton.common.ids import (
    EntityKind,
    EvaluationCoordinateId,
    NamespaceId,
    StudyId,
    StudyResultId,
    TrialId,
)
from mldb_v2.skeleton.repository.listing import CanonicalListing
from mldb_v2.skeleton.repository.resolution import CanonicalDocument, CanonicalEntityId
from mldb_v2.skeleton.results.study_result import StudyResult, StudyResultStatus
from mldb_v2.skeleton.verification.definition_lifecycle import DefinitionLifecycleStatus

EntityResource: TypeAlias = EntityKind | Literal["definitions"]


class ProgressCounter(TypedDict):
    planned: int
    pending: int
    completed: int
    failed: int
    cancelled: int
    skipped: int


class StudyProgress(TypedDict):
    training: ProgressCounter
    evaluations: ProgressCounter
    total: ProgressCounter


class StudyResultView(TypedDict):
    study_result: StudyResult
    progress: StudyProgress


class StudyObservation(TypedDict):
    study_result: StudyResult
    progress: StudyProgress
    backend_observations: list[BackendObservation]


class BackendLogRequest(TypedDict):
    study_result: StudyResultId
    trial: TrialId | None
    coordinate: EvaluationCoordinateId | None
    failed_only: bool
    follow: bool


class BackendLogChunk(TypedDict):
    execution_id: str | None
    text: str


DiagnosisStatus: TypeAlias = Literal["ok", "warning", "error", "unsupported"]


class DiagnosisCheck(TypedDict):
    name: str
    status: DiagnosisStatus
    diagnostic: Diagnostic | None


class QueryInterface(Protocol):
    """Read-only public discovery/query/observation operations."""

    def list_entities(
        self,
        *,
        resource: EntityResource,
        namespace: NamespaceId | None = None,
        status: DefinitionLifecycleStatus | None = None,
    ) -> CanonicalListing: ...

    def get_entity(
        self,
        *,
        kind: EntityKind,
        entity_id: CanonicalEntityId,
    ) -> CanonicalDocument: ...

    def list_study_results(
        self,
        *,
        namespace: NamespaceId | None = None,
        study: StudyId | None = None,
        statuses: Collection[StudyResultStatus] | None = None,
        created_at_from: datetime | None = None,
        created_at_to: datetime | None = None,
        limit: int | None = None,
    ) -> CanonicalListing: ...

    def get_study_result(
        self,
        *,
        study_result: StudyResultId,
    ) -> StudyResultView: ...

    def observe_study(
        self,
        *,
        study_result: StudyResultId,
    ) -> StudyObservation: ...

    def read_backend_logs(
        self,
        *,
        request: BackendLogRequest,
    ) -> Iterable[BackendLogChunk]: ...

    def diagnose(self) -> list[DiagnosisCheck]: ...
