"""Backend-neutral observation and terminal candidate shapes."""

from typing import Literal, Mapping, TypeAlias, TypedDict

from mldb_v2.src.common.diagnostic import Diagnostic
from mldb_v2.src.common.ids import (
    EvaluationCoordinateId,
    StudyPlanId,
    StudyResultId,
    TrialId,
)
from mldb_v2.src.evaluation.evaluation_result import EvaluationArtifactRef
from mldb_v2.src.results.attempt_summary import AttemptSummary
from mldb_v2.src.training.canonical_weights import CanonicalWeightsArtifactRef


class TrainingStageKey(TypedDict):
    study_result: StudyResultId
    plan: StudyPlanId
    trial: TrialId
    kind: Literal["training"]
    coordinate: None
    source_commit: str


class EvaluationStageKey(TypedDict):
    study_result: StudyResultId
    plan: StudyPlanId
    trial: TrialId
    kind: Literal["evaluation"]
    coordinate: EvaluationCoordinateId
    source_commit: str


StageKey: TypeAlias = TrainingStageKey | EvaluationStageKey


class ActiveBackendObservation(TypedDict):
    state: Literal["active"]
    stage_key: StageKey
    backend: str
    execution_ids: list[str]


class TrainingCandidateResult(TypedDict):
    weights: CanonicalWeightsArtifactRef


class EvaluationCandidateResult(TypedDict):
    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, EvaluationArtifactRef]


class CompletedTrainingCandidate(TypedDict):
    state: Literal["terminal"]
    stage_key: TrainingStageKey
    attempts: list[AttemptSummary]
    status: Literal["completed"]
    diagnostic: None
    result: TrainingCandidateResult


class CompletedEvaluationCandidate(TypedDict):
    state: Literal["terminal"]
    stage_key: EvaluationStageKey
    attempts: list[AttemptSummary]
    status: Literal["completed"]
    diagnostic: None
    result: EvaluationCandidateResult


class FailedTrainingCandidate(TypedDict):
    state: Literal["terminal"]
    stage_key: TrainingStageKey
    attempts: list[AttemptSummary]
    status: Literal["failed", "cancelled"]
    diagnostic: Diagnostic
    result: None


class FailedEvaluationCandidate(TypedDict):
    state: Literal["terminal"]
    stage_key: EvaluationStageKey
    attempts: list[AttemptSummary]
    status: Literal["failed", "cancelled"]
    diagnostic: Diagnostic
    result: None


TerminalCandidate: TypeAlias = (
    CompletedTrainingCandidate
    | CompletedEvaluationCandidate
    | FailedTrainingCandidate
    | FailedEvaluationCandidate
)
BackendObservation: TypeAlias = ActiveBackendObservation | TerminalCandidate
