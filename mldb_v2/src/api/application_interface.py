"""Transport-independent MLDB v2 application mutation/check boundary."""

from typing import Literal, Protocol, TypeAlias, TypedDict

from mldb_v2.src.api.query_interface import QueryInterface
from mldb_v2.src.backend.candidate_outcome import StageKey
from mldb_v2.src.common.diagnostic import Diagnostic
from mldb_v2.src.common.ids import (
    DefinitionKind,
    EvaluationResultId,
    ExecutionKey,
    ModelId,
    NamespaceId,
    StudyId,
    StudyPlanId,
    StudyResultId,
    TrainingResultId,
)
from mldb_v2.src.results.study_result import StudyResult, StudyResultStatus
from mldb_v2.src.study.plan import StudyPlan
from mldb_v2.src.verification.definition_lifecycle import SealableDefinitionId


class DefinitionScope(TypedDict, total=False):
    kind: DefinitionKind
    namespace: NamespaceId
    id: SealableDefinitionId


class DefinitionReportItem(TypedDict):
    kind: DefinitionKind
    id: SealableDefinitionId
    valid: bool
    diagnostics: list[Diagnostic]


class DefinitionReport(TypedDict):
    items: list[DefinitionReportItem]
    repository_issues: list[Diagnostic]


class SealResultItem(TypedDict):
    kind: DefinitionKind
    id: SealableDefinitionId
    sealed: bool
    diagnostics: list[Diagnostic]


FinalizedResultId: TypeAlias = TrainingResultId | EvaluationResultId


class AdvanceStudyResponse(TypedDict):
    study_result: StudyResultId
    status: StudyResultStatus
    changed: bool
    admitted: list[StageKey]
    finalized_results: list[FinalizedResultId]
    finalized_models: list[ModelId]
    active: list[StageKey]
    terminal: bool


CancelStudyOutcome: TypeAlias = Literal[
    "accepted",
    "already_cancelling",
    "already_terminal",
]


class CancelStudyResponse(TypedDict):
    study_result: StudyResultId
    outcome: CancelStudyOutcome
    status: StudyResultStatus


class ApplicationInterface(QueryInterface, Protocol):
    """One public application boundary; adapters add transport only."""

    def validate_scope(
        self,
        *,
        scope: DefinitionScope | None = None,
    ) -> DefinitionReport: ...

    def verify_scope(
        self,
        *,
        scope: DefinitionScope | None = None,
    ) -> DefinitionReport: ...

    def seal_scope(
        self,
        *,
        scope: DefinitionScope,
        bulk: bool = False,
    ) -> list[SealResultItem]: ...

    def plan_study(self, *, study: StudyId) -> StudyPlan: ...

    def start_study(
        self,
        *,
        plan: StudyPlanId,
        backend: str,
        execution_key: ExecutionKey,
    ) -> StudyResult: ...

    def advance_study(
        self,
        *,
        study_result: StudyResultId,
    ) -> AdvanceStudyResponse: ...

    def run_study(
        self,
        *,
        study: StudyId,
        backend: str,
    ) -> StudyResult: ...

    def resume_study(
        self,
        *,
        study_result: StudyResultId,
    ) -> StudyResult: ...

    def rerun_study(
        self,
        *,
        source: StudyResultId,
        backend: str | None = None,
    ) -> StudyResult: ...

    def cancel_study(
        self,
        *,
        study_result: StudyResultId,
    ) -> CancelStudyResponse: ...
