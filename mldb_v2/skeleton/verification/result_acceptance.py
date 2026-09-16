"""Canonical result-acceptance boundary for terminal backend candidates."""

from typing import Protocol, TypeAlias, TypedDict

from mldb_v2.skeleton.backend.candidate_outcome import (
    CompletedEvaluationCandidate,
    CompletedTrainingCandidate,
    FailedEvaluationCandidate,
    FailedTrainingCandidate,
)
from mldb_v2.skeleton.backend.stage_input import EvaluationStageInput, TrainingStageInput
from mldb_v2.skeleton.evaluation.evaluation_result import EvaluationResult
from mldb_v2.skeleton.results.study_result import StudyResult
from mldb_v2.skeleton.study.plan import StudyPlan
from mldb_v2.skeleton.training.model import Model
from mldb_v2.skeleton.training.training_result import TrainingResult

TrainingTerminalCandidate: TypeAlias = CompletedTrainingCandidate | FailedTrainingCandidate
EvaluationTerminalCandidate: TypeAlias = CompletedEvaluationCandidate | FailedEvaluationCandidate


class TrainingAcceptanceRequest(TypedDict):
    study_result: StudyResult
    plan: StudyPlan
    stage_input: TrainingStageInput
    candidate: TrainingTerminalCandidate


class EvaluationAcceptanceRequest(TypedDict):
    study_result: StudyResult
    plan: StudyPlan
    stage_input: EvaluationStageInput
    candidate: EvaluationTerminalCandidate


class TrainingAcceptanceResult(TypedDict):
    training_result: TrainingResult
    model: Model | None


class EvaluationAcceptanceResult(TypedDict):
    evaluation_result: EvaluationResult


class ResultAcceptor(Protocol):
    """Accept terminal candidates into canonical formal results without repairing success payloads."""

    def accept_training(
        self,
        *,
        request: TrainingAcceptanceRequest,
    ) -> TrainingAcceptanceResult: ...

    def accept_evaluation(
        self,
        *,
        request: EvaluationAcceptanceRequest,
    ) -> EvaluationAcceptanceResult: ...
