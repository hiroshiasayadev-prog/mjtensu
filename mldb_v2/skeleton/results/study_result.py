"""MLDB v2 canonical Study Result public shape."""

from typing import Literal, TypeAlias, TypedDict

from mldb_v2.skeleton.common.diagnostic import Diagnostic
from mldb_v2.skeleton.common.ids import (
    EvaluationCoordinateId,
    EvaluationResultId,
    ExecutionKey,
    StudyId,
    StudyPlanId,
    StudyResultId,
    TrainingResultId,
    TrialId,
)

StudyResultStatus: TypeAlias = Literal[
    "submitted",
    "cancelling",
    "completed",
    "completed_with_failures",
    "failed",
    "cancelled",
]

StageDisposition: TypeAlias = Literal[
    "pending",
    "completed",
    "failed",
    "cancelled",
    "skipped",
]

SkipReason: TypeAlias = Literal[
    "upstream_failed",
    "upstream_cancelled",
    "study_cancelled",
    "global_failure",
]


class TrainingSlot(TypedDict):
    disposition: StageDisposition
    result: TrainingResultId | None
    reason: SkipReason | None


class EvaluationSlot(TypedDict):
    coordinate: EvaluationCoordinateId
    stage: str
    disposition: StageDisposition
    result: EvaluationResultId | None
    reason: SkipReason | None


class StudyResultTrial(TypedDict):
    trial: TrialId
    training: TrainingSlot | None
    evaluations: list[EvaluationSlot]


class StudyResult(TypedDict):
    schema: Literal["mjtensu.mldb-v2/study-result/v1"]
    id: StudyResultId
    execution_key: ExecutionKey
    plan: StudyPlanId
    study: StudyId
    source_commit: str
    backend: str
    created_at: str
    status: StudyResultStatus
    diagnostic: Diagnostic | None
    trials: list[StudyResultTrial]
