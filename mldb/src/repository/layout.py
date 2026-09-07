"""Canonical MLDB repository path derivation.

This module implements the frozen repository layout contract without performing
filesystem access or entity resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..common.ids import (
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


_ENTITY_DIRECTORY_NAMES = {
    EntityKind.TASK: "tasks",
    EntityKind.CORPUS: "corpora",
    EntityKind.ARCHITECTURE: "architectures",
    EntityKind.TRAIN_PROTOCOL: "train_protocols",
    EntityKind.TRAINING_RUN: "training_runs",
    EntityKind.MODEL: "models",
    EntityKind.EVALUATION_PROTOCOL: "evaluation_protocols",
    EntityKind.EVALUATION_RUN: "evaluation_runs",
    EntityKind.STUDY: "studies",
    EntityKind.STUDY_RUN: "study_runs",
}


@dataclass(frozen=True, slots=True)
class TrainingRunPaths:
    """Canonical filesystem locations owned by one Training Run.

    ``directory`` is ``mldb_data/training_runs/<id>/`` beneath the repository root.
    ``metadata_path`` is its authoritative ``run.yaml``. ``work_dir`` is execution-
    owned working space, ``artifacts_dir`` is MLDB-owned formal artifact storage, and
    ``weights_path`` is the v1 canonical learned artifact
    ``artifacts/weights.pt``.

    This value contains locations only and carries no Training Run lifecycle or
    persistence semantics.
    """

    directory: Path
    metadata_path: Path
    work_dir: Path
    artifacts_dir: Path
    weights_path: Path


@dataclass(frozen=True, slots=True)
class EvaluationRunPaths:
    """Canonical filesystem locations owned by one Evaluation Run.

    ``directory`` is ``mldb_data/evaluation_runs/<id>/`` beneath the repository root.
    ``metadata_path`` is its authoritative ``run.yaml``. ``work_dir`` is execution-
    owned working space and ``artifacts_dir`` is MLDB-owned formal artifact storage.

    Individual formal artifact relative names are defined by Evaluation result
    acceptance and are not invented by this layout value.
    """

    directory: Path
    metadata_path: Path
    work_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True, slots=True)
class StudyRunPaths:
    """Canonical filesystem locations owned by one Study Run.

    ``directory`` is ``mldb_data/study_runs/<id>/`` beneath the repository root.
    ``metadata_path`` is its authoritative ``run.yaml`` and ``plan_path`` is the v1
    immutable execution plan ``plan.jsonl`` stored beside that record.

    Study Run v1 has no canonical ``work/`` or ``artifacts/`` directory.
    """

    directory: Path
    metadata_path: Path
    plan_path: Path


@dataclass(frozen=True, slots=True)
class RepositoryLayout:
    """Small immutable root value for canonical MLDB repository placement.

    ``root`` is the repository root containing sibling ``mldb_data/`` and
    ``mldb_tests/`` roots. Construction performs no normalization, existence check,
    filesystem access, or repository validation.

    Every derivation operation accepts the existing entity-specific ``NewType`` ID
    rather than flattening identities through ``str`` or a generic ``EntityKind`` API.
    """

    root: Path

    @property
    def data_root(self) -> Path:
        """Return ``<root>/mldb_data``."""

        return self.root / "mldb_data"

    @property
    def tests_root(self) -> Path:
        """Return ``<root>/mldb_tests``."""

        return self.root / "mldb_tests"

    def entity_directory(self, kind: EntityKind) -> Path:
        """Return the canonical ``mldb_data`` containing directory for ``kind``.

        This operation exists for kind-oriented entity query/listing. It does not
        accept an entity ID and therefore does not erase any entity-specific ID type.
        Callers resolving one known entity use the typed derivation operations below
        rather than scanning this or any other kind directory.
        """

        return self.data_root / _ENTITY_DIRECTORY_NAMES[kind]

    def task_metadata_path(self, task_id: TaskId) -> Path:
        """Return ``mldb_data/tasks/<task-id>.yaml``."""

        return self.entity_directory(EntityKind.TASK) / f"{task_id}.yaml"

    def corpus_metadata_path(self, corpus_id: CorpusId) -> Path:
        """Return ``mldb_data/corpora/<corpus-id>.yaml``."""

        return self.entity_directory(EntityKind.CORPUS) / f"{corpus_id}.yaml"

    def corpus_artifact_path(self, corpus_id: CorpusId) -> Path:
        """Return the canonical Corpus SQLite sibling ``<corpus-id>.sqlite``."""

        return self.entity_directory(EntityKind.CORPUS) / f"{corpus_id}.sqlite"

    def corpus_builder_path(self, corpus_id: CorpusId) -> Path:
        """Return the canonical Corpus Python builder sibling ``<corpus-id>.py``."""

        return self.entity_directory(EntityKind.CORPUS) / f"{corpus_id}.py"

    def architecture_metadata_path(self, architecture_id: ArchitectureId) -> Path:
        """Return ``mldb_data/architectures/<architecture-id>.yaml``."""

        return self.entity_directory(EntityKind.ARCHITECTURE) / f"{architecture_id}.yaml"

    def architecture_implementation_path(
        self,
        architecture_id: ArchitectureId,
    ) -> Path:
        """Return the canonical Architecture Python sibling ``<id>.py``."""

        return self.entity_directory(EntityKind.ARCHITECTURE) / f"{architecture_id}.py"

    def train_protocol_metadata_path(
        self,
        train_protocol_id: TrainProtocolId,
    ) -> Path:
        """Return ``mldb_data/train_protocols/<train-protocol-id>.yaml``."""

        return self.entity_directory(EntityKind.TRAIN_PROTOCOL) / f"{train_protocol_id}.yaml"

    def train_protocol_implementation_path(
        self,
        train_protocol_id: TrainProtocolId,
    ) -> Path:
        """Return the canonical Train Protocol Python sibling ``<id>.py``."""

        return self.entity_directory(EntityKind.TRAIN_PROTOCOL) / f"{train_protocol_id}.py"

    def training_run_paths(self, training_run_id: TrainingRunId) -> TrainingRunPaths:
        """Return all current canonical paths owned by one Training Run."""

        directory = self.entity_directory(EntityKind.TRAINING_RUN) / str(training_run_id)
        artifacts_dir = directory / "artifacts"
        return TrainingRunPaths(
            directory=directory,
            metadata_path=directory / "run.yaml",
            work_dir=directory / "work",
            artifacts_dir=artifacts_dir,
            weights_path=artifacts_dir / "weights.pt",
        )

    def model_metadata_path(self, model_id: ModelId) -> Path:
        """Return ``mldb_data/models/<model-id>.yaml``."""

        return self.entity_directory(EntityKind.MODEL) / f"{model_id}.yaml"

    def evaluation_protocol_metadata_path(
        self,
        evaluation_protocol_id: EvaluationProtocolId,
    ) -> Path:
        """Return ``mldb_data/evaluation_protocols/<evaluation-protocol-id>.yaml``."""

        return self.entity_directory(EntityKind.EVALUATION_PROTOCOL) / f"{evaluation_protocol_id}.yaml"

    def evaluation_protocol_implementation_path(
        self,
        evaluation_protocol_id: EvaluationProtocolId,
    ) -> Path:
        """Return the canonical Evaluation Protocol Python sibling ``<id>.py``."""

        return self.entity_directory(EntityKind.EVALUATION_PROTOCOL) / f"{evaluation_protocol_id}.py"

    def evaluation_run_paths(
        self,
        evaluation_run_id: EvaluationRunId,
    ) -> EvaluationRunPaths:
        """Return all current canonical paths owned by one Evaluation Run."""

        directory = self.entity_directory(EntityKind.EVALUATION_RUN) / str(evaluation_run_id)
        return EvaluationRunPaths(
            directory=directory,
            metadata_path=directory / "run.yaml",
            work_dir=directory / "work",
            artifacts_dir=directory / "artifacts",
        )

    def study_metadata_path(self, study_id: StudyId) -> Path:
        """Return ``mldb_data/studies/<study-id>.yaml``."""

        return self.entity_directory(EntityKind.STUDY) / f"{study_id}.yaml"

    def study_run_paths(self, study_run_id: StudyRunId) -> StudyRunPaths:
        """Return all current canonical paths owned by one Study Run."""

        directory = self.entity_directory(EntityKind.STUDY_RUN) / str(study_run_id)
        return StudyRunPaths(
            directory=directory,
            metadata_path=directory / "run.yaml",
            plan_path=directory / "plan.jsonl",
        )

    def architecture_test_dir(self, architecture_id: ArchitectureId) -> Path:
        """Return ``mldb_tests/architectures/<architecture-id>/``."""

        return self.tests_root / _ENTITY_DIRECTORY_NAMES[EntityKind.ARCHITECTURE] / str(architecture_id)

    def train_protocol_test_dir(self, train_protocol_id: TrainProtocolId) -> Path:
        """Return ``mldb_tests/train_protocols/<train-protocol-id>/``."""

        return self.tests_root / _ENTITY_DIRECTORY_NAMES[EntityKind.TRAIN_PROTOCOL] / str(train_protocol_id)

    def evaluation_protocol_test_dir(
        self,
        evaluation_protocol_id: EvaluationProtocolId,
    ) -> Path:
        """Return ``mldb_tests/evaluation_protocols/<evaluation-protocol-id>/``."""

        return self.tests_root / _ENTITY_DIRECTORY_NAMES[EntityKind.EVALUATION_PROTOCOL] / str(evaluation_protocol_id)
