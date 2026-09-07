"""Shared MLDB identity signatures."""

from enum import Enum, auto
from typing import Literal, NewType, TypeAlias


TaskId = NewType("TaskId", str)
CorpusId = NewType("CorpusId", str)
ArchitectureId = NewType("ArchitectureId", str)
TrainProtocolId = NewType("TrainProtocolId", str)
ModelId = NewType("ModelId", str)
EvaluationProtocolId = NewType("EvaluationProtocolId", str)
StudyId = NewType("StudyId", str)

TrainingRunId = NewType("TrainingRunId", str)
EvaluationRunId = NewType("EvaluationRunId", str)
StudyRunId = NewType("StudyRunId", str)


class EntityKind(Enum):
    """MLDB entity kind used for typed resolution and entity selection."""

    TASK = auto()
    CORPUS = auto()
    ARCHITECTURE = auto()
    TRAIN_PROTOCOL = auto()
    TRAINING_RUN = auto()
    MODEL = auto()
    EVALUATION_PROTOCOL = auto()
    EVALUATION_RUN = auto()
    STUDY = auto()
    STUDY_RUN = auto()


ExecutableAssetKind: TypeAlias = Literal[
    EntityKind.ARCHITECTURE,
    EntityKind.TRAIN_PROTOCOL,
    EntityKind.EVALUATION_PROTOCOL,
]
