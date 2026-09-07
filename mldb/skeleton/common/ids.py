"""Shared MLDB identity signatures.

This skeleton fixes Python-level distinctions between MLDB entity identities and the
small cross-feature entity-kind vocabulary. Concrete identifier grammar, allocation,
validation, lifecycle state, repository lookup, and persistence/wire encoding belong
to the entity or feature contracts that own those concerns.
"""

from enum import Enum, auto
from typing import Literal, NewType, TypeAlias


# Reusable definition and immutable-object identities.
TaskId = NewType("TaskId", str)
CorpusId = NewType("CorpusId", str)
ArchitectureId = NewType("ArchitectureId", str)
TrainProtocolId = NewType("TrainProtocolId", str)
ModelId = NewType("ModelId", str)
EvaluationProtocolId = NewType("EvaluationProtocolId", str)
StudyId = NewType("StudyId", str)

# Concrete execution-event identities. These remain distinct from definition IDs
# even where both are represented as strings in persisted records.
TrainingRunId = NewType("TrainingRunId", str)
EvaluationRunId = NewType("EvaluationRunId", str)
StudyRunId = NewType("StudyRunId", str)


class EntityKind(Enum):
    """MLDB entity kind used for typed resolution and entity selection.

    Member identity is part of the public Python signature. ``value`` is intentionally
    opaque here because current Design Records do not define a persistence or transport
    encoding for an entity-kind discriminator.
    """

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


# The complete v1 subset whose definitions have executable sibling Python assets.
ExecutableAssetKind: TypeAlias = Literal[
    EntityKind.ARCHITECTURE,
    EntityKind.TRAIN_PROTOCOL,
    EntityKind.EVALUATION_PROTOCOL,
]
