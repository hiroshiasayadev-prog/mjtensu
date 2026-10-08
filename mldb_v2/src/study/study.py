from typing import Literal, Mapping, NotRequired, TypeAlias, TypedDict

from mldb_v2.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    ModelId,
    StudyId,
    TrainProtocolId,
)
from mldb_v2.src.common.parameters import PublicParameterValue


class ParameterAxis(TypedDict):
    values: list[PublicParameterValue]


TrainingParameterGrid: TypeAlias = Mapping[str, ParameterAxis]
EvaluationParameterGrid: TypeAlias = Mapping[str, ParameterAxis]


class TrainingModelSource(TypedDict):
    corpus: CorpusId
    protocol: TrainProtocolId
    architectures: list[ArchitectureId]
    parameters: TrainingParameterGrid
    seeds: list[int]


class TrainingCase(TypedDict):
    architecture: ArchitectureId
    parameters: Mapping[str, PublicParameterValue]
    seed: int


class ExplicitTrainingModelSource(TypedDict):
    corpus: CorpusId
    protocol: TrainProtocolId
    cases: list[TrainingCase]


TrainingSource: TypeAlias = TrainingModelSource | ExplicitTrainingModelSource
ExistingModelSource: TypeAlias = list[ModelId]


class TrainingStudyModelSource(TypedDict):
    train: TrainingSource


class ExistingStudyModelSource(TypedDict):
    existing: ExistingModelSource


StudyModelSource: TypeAlias = TrainingStudyModelSource | ExistingStudyModelSource


class EvaluationStage(TypedDict):
    stage: str
    corpus: CorpusId
    protocol: EvaluationProtocolId
    parameters: EvaluationParameterGrid
    inputs: NotRequired[Mapping[str, Mapping[str, str]]]


class Study(TypedDict):
    schema: Literal["mjtensu.mldb-v2/study/v1"]
    id: StudyId
    status: Literal["draft", "sealed"]
    name: str
    description: str
    model: StudyModelSource
    evaluations: list[EvaluationStage]
