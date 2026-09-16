"""Backend-neutral immutable input for one ready MLDB v2 stage."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

from mldb_v2.skeleton.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationCoordinateId,
    EvaluationProtocolId,
    ModelId,
    StudyPlanId,
    StudyResultId,
    TaskId,
    TrainProtocolId,
    TrainingResultId,
    TrialId,
)
from mldb_v2.skeleton.common.parameters import ResolvedPublicParameters
from mldb_v2.skeleton.study.plan import PlanPin
from mldb_v2.skeleton.training.canonical_weights import CanonicalWeightsArtifactRef


class TrainingStage(TypedDict):
    task: TaskId
    corpus: CorpusId
    architecture: ArchitectureId
    train_protocol: TrainProtocolId
    parameters: ResolvedPublicParameters
    seed: int


class EvaluationStage(TypedDict):
    name: str
    task: TaskId
    corpus: CorpusId
    evaluation_protocol: EvaluationProtocolId
    parameters: ResolvedPublicParameters


class RuntimeModel(TypedDict):
    model: ModelId
    training_result: TrainingResultId
    task: TaskId
    architecture: ArchitectureId
    weights: CanonicalWeightsArtifactRef


class TrainingStageInput(TypedDict):
    schema: Literal["mjtensu.mldb-v2/stage-input/v1"]
    study_result: StudyResultId
    plan: StudyPlanId
    plan_sha256: str
    trial: TrialId
    kind: Literal["training"]
    coordinate: None
    source_commit: str
    pins: list[PlanPin]
    stage: TrainingStage
    runtime_model: None


class EvaluationStageInput(TypedDict):
    schema: Literal["mjtensu.mldb-v2/stage-input/v1"]
    study_result: StudyResultId
    plan: StudyPlanId
    plan_sha256: str
    trial: TrialId
    kind: Literal["evaluation"]
    coordinate: EvaluationCoordinateId
    source_commit: str
    pins: list[PlanPin]
    stage: EvaluationStage
    runtime_model: RuntimeModel


StageInput: TypeAlias = TrainingStageInput | EvaluationStageInput
