from __future__ import annotations

import unittest

from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EntityKind,
    EvaluationProtocolId,
    EvaluationRunId,
    ModelId,
    StudyId,
    StudyRunId,
    TaskId,
    TrainingRunId,
    TrainProtocolId,
)


class CommonIdsTests(unittest.TestCase):
    def test_newtypes_remain_distinct_public_identity_constructors(self) -> None:
        constructors = (
            TaskId,
            CorpusId,
            ArchitectureId,
            TrainProtocolId,
            ModelId,
            EvaluationProtocolId,
            StudyId,
            TrainingRunId,
            EvaluationRunId,
            StudyRunId,
        )

        self.assertEqual(len(constructors), len(set(constructors)))
        for constructor in constructors:
            self.assertEqual("id-v1", constructor("id-v1"))

    def test_entity_kind_members_preserve_frozen_order(self) -> None:
        self.assertEqual(
            [
                "TASK",
                "CORPUS",
                "ARCHITECTURE",
                "TRAIN_PROTOCOL",
                "TRAINING_RUN",
                "MODEL",
                "EVALUATION_PROTOCOL",
                "EVALUATION_RUN",
                "STUDY",
                "STUDY_RUN",
            ],
            [member.name for member in EntityKind],
        )


if __name__ == "__main__":
    unittest.main()
