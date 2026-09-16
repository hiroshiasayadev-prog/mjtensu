"""MLDB v2 terminal Training Result public shape."""

from typing import Literal, TypedDict

from mldb_v2.skeleton.common.diagnostic import Diagnostic
from mldb_v2.skeleton.common.ids import (
    ArchitectureId,
    CorpusId,
    ModelId,
    StudyPlanId,
    StudyResultId,
    TaskId,
    TrainProtocolId,
    TrainingResultId,
    TrialId,
)
from mldb_v2.skeleton.common.parameters import ResolvedPublicParameters
from mldb_v2.skeleton.results.attempt_summary import AttemptSummary
from mldb_v2.skeleton.training.canonical_weights import CanonicalWeightsArtifactRef


class TrainingResultPayload(TypedDict):
    weights: CanonicalWeightsArtifactRef
    model: ModelId


class TrainingResult(TypedDict):
    schema: Literal["mjtensu.mldb-v2/training-result/v1"]
    id: TrainingResultId
    study_result: StudyResultId
    plan: StudyPlanId
    trial: TrialId
    task: TaskId
    architecture: ArchitectureId
    corpus: CorpusId
    train_protocol: TrainProtocolId
    parameters: ResolvedPublicParameters
    seed: int
    source_commit: str
    attempts: list[AttemptSummary]
    status: Literal["completed", "failed", "cancelled"]
    diagnostic: Diagnostic | None
    result: TrainingResultPayload | None
