"""MLDB v2 immutable Study Plan public shapes."""

from typing import Literal, TypeAlias, TypedDict

from mldb_v2.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationCoordinateId,
    EvaluationProtocolId,
    ModelId,
    NamespaceId,
    StudyId,
    StudyPlanId,
    TaskId,
    TrainProtocolId,
    TrainingResultId,
    TrialId,
)
from mldb_v2.src.common.parameters import ResolvedPublicParameters
from mldb_v2.src.verification.executable_integrity import ExecutableSource

PlanPinKind: TypeAlias = Literal[
    "namespace",
    "task",
    "corpus",
    "architecture",
    "train_protocol",
    "evaluation_protocol",
    "study",
    "model",
    "training_result",
]

PlanPinId: TypeAlias = (
    NamespaceId
    | TaskId
    | CorpusId
    | ArchitectureId
    | TrainProtocolId
    | EvaluationProtocolId
    | StudyId
    | ModelId
    | TrainingResultId
)


class PlanPin(TypedDict):
    kind: PlanPinKind
    id: PlanPinId
    yaml_sha256: str
    companion_sha256: str | None
    sources: list[ExecutableSource]
    manifest_sha256: str | None
    manifest_entries: int | None


class TrainingTrialSource(TypedDict):
    kind: Literal["training"]
    task: TaskId
    corpus: CorpusId
    architecture: ArchitectureId
    train_protocol: TrainProtocolId
    parameters: ResolvedPublicParameters
    seed: int


class ExistingModelTrialSource(TypedDict):
    kind: Literal["existing_model"]
    model: ModelId


PlanTrialSource: TypeAlias = TrainingTrialSource | ExistingModelTrialSource


class EvaluationCoordinate(TypedDict):
    coordinate: EvaluationCoordinateId
    stage: str
    task: TaskId
    corpus: CorpusId
    evaluation_protocol: EvaluationProtocolId
    parameters: ResolvedPublicParameters


class PlanTrial(TypedDict):
    trial: TrialId
    source: PlanTrialSource
    evaluations: list[EvaluationCoordinate]


class StudyPlan(TypedDict):
    schema: Literal["mjtensu.mldb-v2/study-plan/v1"]
    id: StudyPlanId
    content_sha256: str
    study: StudyId
    source_commit: str
    pins: list[PlanPin]
    trials: list[PlanTrial]
