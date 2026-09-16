"""MLDB v2 public identity signatures.

This skeleton freezes Python-level identity distinctions only. Concrete grammar and repository
placement remain owned by the corresponding Specifications.
"""

from enum import Enum
from typing import NewType, TypeAlias

NamespaceId = NewType("NamespaceId", str)
TaskId = NewType("TaskId", str)
CorpusId = NewType("CorpusId", str)
ArchitectureId = NewType("ArchitectureId", str)
TrainProtocolId = NewType("TrainProtocolId", str)
EvaluationProtocolId = NewType("EvaluationProtocolId", str)
StudyId = NewType("StudyId", str)
StudyPlanId = NewType("StudyPlanId", str)
StudyResultId = NewType("StudyResultId", str)
TrainingResultId = NewType("TrainingResultId", str)
ModelId = NewType("ModelId", str)
EvaluationResultId = NewType("EvaluationResultId", str)
TrialId = NewType("TrialId", str)
EvaluationCoordinateId = NewType("EvaluationCoordinateId", str)
ExecutionKey = NewType("ExecutionKey", str)

TypedEntityId: TypeAlias = TaskId | CorpusId | ArchitectureId | TrainProtocolId | EvaluationProtocolId | StudyId | StudyPlanId | StudyResultId | TrainingResultId | ModelId | EvaluationResultId


class EntityKind(str, Enum):
    NAMESPACE = "namespace"
    TASK = "task"
    CORPUS = "corpus"
    ARCHITECTURE = "architecture"
    TRAIN_PROTOCOL = "train_protocol"
    EVALUATION_PROTOCOL = "evaluation_protocol"
    STUDY = "study"
    STUDY_PLAN = "study_plan"
    TRAINING_RESULT = "training_result"
    MODEL = "model"
    EVALUATION_RESULT = "evaluation_result"
    STUDY_RESULT = "study_result"


class DefinitionKind(str, Enum):
    TASK = "task"
    CORPUS = "corpus"
    ARCHITECTURE = "architecture"
    TRAIN_PROTOCOL = "train_protocol"
    EVALUATION_PROTOCOL = "evaluation_protocol"
    STUDY = "study"
