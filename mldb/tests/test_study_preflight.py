from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mldb.src.catalog.architecture import (
    Architecture,
    ArchitectureImplementation,
    ArchitectureInterface,
    ArchitectureStatus,
    ArchitectureStructure,
)
from mldb.src.catalog.corpus import Corpus, CorpusArtifact, CorpusDataSpec
from mldb.src.catalog.task import CategoricalTarget, Task, TaskInput, TaskScope
from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    ModelId,
    StudyId,
    TaskId,
    TrainingRunId,
    TrainProtocolId,
)
from mldb.src.common.parameters import PublicParameterDeclaration
from mldb.src.evaluation.protocol import (
    EvaluationOutputs,
    EvaluationProtocol,
    EvaluationProtocolImplementation,
    EvaluationProtocolStatus,
)
from mldb.src.model.identity import Model
from mldb.src.model.loading import ModelHandle
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.catalog_handles import (
    ArchitectureHandle,
    CorpusHandle,
    TaskHandle,
)
from mldb.src.runtime.definition_handles import (
    EvaluationProtocolHandle,
    StudyHandle,
    TrainProtocolHandle,
)
from mldb.src.study import preflight as subject
from mldb.src.study.definition import (
    Study,
    StudyEvaluationStage,
    StudyExistingModelSource,
    StudyStatus,
    StudyTrainingModelSource,
    StudyTrainingParameterAxis,
)
from mldb.src.training.protocol import (
    TrainProtocol,
    TrainProtocolImplementation,
    TrainProtocolStatus,
)
from mldb.src.training.run import (
    TrainingRun,
    TrainingRunExecution,
    TrainingRunResult,
    TrainingRunStatus,
)
from mldb.src.training.weights import CanonicalWeightsArtifact


class _RecordingFilesystem:
    def __init__(self, files: dict[Path, bytes]) -> None:
        self.files = files
        self.mutations = 0

    def file_exists(self, path: Path) -> bool:
        return path in self.files

    def directory_exists(self, path: Path) -> bool:
        return False

    def read_text(self, path: Path, *, encoding: str) -> str:
        raise AssertionError("preflight must use resolver-produced metadata")

    def read_bytes(self, path: Path) -> bytes:
        return self.files[path]
    def list_directory(self, path: Path) -> tuple[Path, ...]:
        return ()

    def ensure_directory(self, path: Path) -> None:
        self.mutations += 1
        raise AssertionError("preflight must not mutate the repository")

    def replace_text(self, path: Path, text: str, *, encoding: str) -> None:
        self.mutations += 1
        raise AssertionError("preflight must not mutate the repository")

    def replace_bytes(self, path: Path, data: bytes) -> None:
        self.mutations += 1
        raise AssertionError("preflight must not mutate the repository")


class StudyPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = RepositoryLayout(Path("repository"))
        self.task_id = TaskId("tile-task-v1")
        self.task = TaskHandle(
            metadata=Task(
                schema="mjtensu.mldb/task/v1",
                id=self.task_id,
                name="tiles",
                problem_type="classification",
                description="tiles",
                input=TaskInput(semantic_unit="tile"),
                target=CategoricalTarget(
                    type="categorical",
                    labels=("a", "b"),
                ),
                semantics={},
                scope=TaskScope(includes=("tiles",), excludes=()),
            ),
            metadata_path=Path("task.yaml"),
        )

        self.training_corpus = self._corpus("train-v1")
        self.evaluation_corpus = self._corpus("eval-v1")
        self.architecture_bytes = b"raise RuntimeError('must not import')\n"
        self.train_protocol_bytes = b"raise RuntimeError('must not import')\n"
        self.evaluation_protocol_bytes = b"raise RuntimeError('must not import')\n"
        self.weights_bytes = b"canonical learned weights"

        self.architecture_path = Path("plain-v1.py")
        self.train_protocol_path = Path("train-v1.py")
        self.evaluation_protocol_path = Path("evaluation-v1.py")
        self.weights_path = Path("weights.pt")
        self.filesystem = _RecordingFilesystem(
            {
                self.architecture_path: self.architecture_bytes,
                self.train_protocol_path: self.train_protocol_bytes,
                self.evaluation_protocol_path: self.evaluation_protocol_bytes,
                self.weights_path: self.weights_bytes,
            }
        )
        self.architecture = ArchitectureHandle(
            metadata=Architecture(
                schema="mjtensu.mldb/architecture/v1",
                id=ArchitectureId("plain-v1"),
                status=ArchitectureStatus.SEALED,
                task=self.task_id,
                name="plain",
                family="plain",
                description="plain",
                implementation=ArchitectureImplementation(
                    framework="pytorch",
                    entrypoint="build",
                    sha256=hashlib.sha256(
                        self.architecture_bytes
                    ).hexdigest(),
                ),
                interface=ArchitectureInterface(input={}, output={}),
                structure=ArchitectureStructure(summary="plain"),
            ),
            metadata_path=Path("plain-v1.yaml"),
            implementation_path=self.architecture_path,
        )
        self.train_protocol = TrainProtocolHandle(
            metadata=TrainProtocol(
                schema="mjtensu.mldb/train-protocol/v1",
                id=TrainProtocolId("train-v1"),
                status=TrainProtocolStatus.SEALED,
                task=self.task_id,
                name="train",
                description="train",
                implementation=TrainProtocolImplementation(
                    entrypoint="train",
                    sha256=hashlib.sha256(
                        self.train_protocol_bytes
                    ).hexdigest(),
                ),
                parameters={
                    "epochs": PublicParameterDeclaration(default=10),
                    "optimizer": PublicParameterDeclaration(default="adam"),
                },
            ),
            metadata_path=Path("train-v1.yaml"),
            implementation_path=self.train_protocol_path,
        )
        self.evaluation_protocol = EvaluationProtocolHandle(
            metadata=EvaluationProtocol(
                schema="mjtensu.mldb/evaluation-protocol/v1",
                id=EvaluationProtocolId("evaluation-v1"),
                status=EvaluationProtocolStatus.SEALED,
                task=self.task_id,
                name="evaluation",
                description="evaluation",
                implementation=EvaluationProtocolImplementation(
                    entrypoint="evaluate",
                    sha256=hashlib.sha256(
                        self.evaluation_protocol_bytes
                    ).hexdigest(),
                ),
                parameters={
                    "batch_size": PublicParameterDeclaration(default=32),
                },
                outputs=EvaluationOutputs(metrics={}, artifacts={}),
            ),
            metadata_path=Path("evaluation-v1.yaml"),
            implementation_path=self.evaluation_protocol_path,
        )

        weights = CanonicalWeightsArtifact(
            format="pytorch-state-dict",
            path="artifacts/weights.pt",
            sha256=hashlib.sha256(self.weights_bytes).hexdigest(),
            bytes=len(self.weights_bytes),
        )
        run = TrainingRun(
            schema="mjtensu.mldb/training-run/v1",
            id=TrainingRunId("tr-20260908-001"),
            status=TrainingRunStatus.COMPLETED,
            corpus=self.training_corpus.metadata.id,
            architecture=self.architecture.metadata.id,
            train_protocol=self.train_protocol.metadata.id,
            parameters={"epochs": 10},
            execution=TrainingRunExecution(
                seed=42,
                started_at="start",
                finished_at="finish",
            ),
            result=TrainingRunResult(weights=weights),
        )
        self.model = ModelHandle(
            metadata=Model(
                schema="mjtensu.mldb/model/v1",
                id=ModelId("mdl-20260908-001"),
                training_run=run.id,
            ),
            metadata_path=Path("model.yaml"),
            training_run=run,
            training_run_metadata_path=Path("run.yaml"),
            weights_path=self.weights_path,
            architecture=self.architecture,
        )
        self.training_study = self._study_handle(
            StudyId("training-study-v1"),
            StudyTrainingModelSource(
                corpus=self.training_corpus.metadata.id,
                protocol=self.train_protocol.metadata.id,
                architectures=(self.architecture.metadata.id,),
                parameters={
                    "epochs": StudyTrainingParameterAxis(values=(10, 20)),
                },
                seeds=(42,),
            ),
        )
        self.existing_study = self._study_handle(
            StudyId("existing-study-v1"),
            StudyExistingModelSource(models=(self.model.metadata.id,)),
        )

    def _corpus(self, suffix: str) -> CorpusHandle:
        corpus_id = CorpusId(suffix)
        return CorpusHandle(
            metadata=Corpus(
                schema="mjtensu.mldb/corpus/v1",
                id=corpus_id,
                task=self.task_id,
                artifact=CorpusArtifact(
                    format="sqlite",
                    sha256="0" * 64,
                ),
                data=CorpusDataSpec(schema="image-v1", table="samples"),
                representation={},
                builder_parameters={},
                splits={},
            ),
            metadata_path=Path(f"{suffix}.yaml"),
            artifact_path=Path(f"{suffix}.sqlite"),
            builder_path=Path(f"{suffix}.py"),
        )

    def _study_handle(
        self,
        study_id: StudyId,
        model: StudyTrainingModelSource | StudyExistingModelSource,
        *,
        status: StudyStatus = StudyStatus.SEALED,
        evaluations: tuple[StudyEvaluationStage, ...] | None = None,
    ) -> StudyHandle:
        if evaluations is None:
            evaluations = (
                StudyEvaluationStage(
                    stage="primary",
                    corpus=self.evaluation_corpus.metadata.id,
                    protocol=self.evaluation_protocol.metadata.id,
                    parameters={"batch_size": 64},
                ),
                StudyEvaluationStage(
                    stage="secondary",
                    corpus=self.evaluation_corpus.metadata.id,
                    protocol=self.evaluation_protocol.metadata.id,
                    parameters={},
                ),
            )
        metadata = Study(
            schema="mjtensu.mldb/study/v1",
            id=study_id,
            status=status,
            name="study",
            description="study",
            model=model,
            evaluations=evaluations,
        )
        return StudyHandle(
            metadata=metadata,
            metadata_path=self.layout.study_metadata_path(study_id),
        )

    @contextmanager
    def _resolved_dependencies(self):
        corpora = {
            self.training_corpus.metadata.id: self.training_corpus,
            self.evaluation_corpus.metadata.id: self.evaluation_corpus,
        }
        with (
            patch.object(
                subject,
                "resolve_corpus",
                side_effect=lambda asset_id, *_: corpora[asset_id],
            ) as resolve_corpus,
            patch.object(
                subject,
                "resolve_architecture",
                side_effect=lambda *_: self.architecture,
            ) as resolve_architecture,
            patch.object(
                subject,
                "resolve_train_protocol",
                side_effect=lambda *_: self.train_protocol,
            ) as resolve_train_protocol,
            patch.object(
                subject,
                "resolve_evaluation_protocol",
                side_effect=lambda *_: self.evaluation_protocol,
            ) as resolve_evaluation_protocol,
            patch.object(
                subject,
                "resolve_model",
                side_effect=lambda *_: self.model,
            ) as resolve_model,
            patch.object(
                subject,
                "resolve_task",
                return_value=self.task,
            ) as resolve_task,
        ):
            yield SimpleNamespace(
                corpus=resolve_corpus,
                architecture=resolve_architecture,
                train_protocol=resolve_train_protocol,
                evaluation_protocol=resolve_evaluation_protocol,
                model=resolve_model,
                task=resolve_task,
            )

    def _preflight(self, study: StudyHandle):
        return subject.preflight_study_execution(
            study,
            self.layout,
            self.filesystem,
        )

    def test_sealed_training_source_success_and_protocol_snapshot(self) -> None:
        protocol = self.evaluation_protocol
        with self._resolved_dependencies() as calls:
            prepared = self._preflight(self.training_study)

        self.assertIs(self.training_study, prepared.study)
        self.assertIs(self.task, prepared.task)
        self.assertIs(self.train_protocol, prepared.train_protocol)
        self.assertEqual(
            {self.evaluation_protocol.metadata.id},
            set(prepared.evaluation_protocols),
        )
        self.assertIs(
            protocol,
            prepared.evaluation_protocols[protocol.metadata.id],
        )
        with self.assertRaises(TypeError):
            prepared.evaluation_protocols[protocol.metadata.id] = protocol
        calls.evaluation_protocol.assert_called_once()
        calls.task.assert_called_once_with(
            self.task_id,
            self.layout,
            self.filesystem,
        )
        self.assertEqual(0, self.filesystem.mutations)

    def test_sealed_existing_model_success_does_not_resolve_train_protocol(self) -> None:
        with self._resolved_dependencies() as calls:
            prepared = self._preflight(self.existing_study)

        self.assertIsNone(prepared.train_protocol)
        self.assertIs(self.task, prepared.task)
        calls.model.assert_called_once_with(
            self.model.metadata.id,
            self.layout,
            self.filesystem,
        )
        calls.train_protocol.assert_not_called()
        calls.architecture.assert_not_called()
        self.assertEqual(0, self.filesystem.mutations)

    def test_rejects_raw_or_draft_study_before_dependency_resolution(self) -> None:
        with self.assertRaises(TypeError):
            subject.preflight_study_execution(
                self.training_study.metadata,  # type: ignore[arg-type]
                self.layout,
                self.filesystem,
            )
        draft = replace(
            self.training_study,
            metadata=replace(
                self.training_study.metadata,
                status=StudyStatus.DRAFT,
            ),
        )
        with self._resolved_dependencies() as calls:
            with self.assertRaisesRegex(ValueError, "sealed"):
                self._preflight(draft)
        calls.corpus.assert_not_called()
        self.assertEqual(0, self.filesystem.mutations)

    def _assert_executable_states_rejected(
        self,
        attribute: str,
        draft_status: object,
    ) -> None:
        original = getattr(self, attribute)
        cases = (
            ("draft", draft_status, original.metadata.implementation.sha256),
            ("missing hash", original.metadata.status, None),
            ("hash mismatch", original.metadata.status, "0" * 64),
        )
        try:
            for label, status, sha256 in cases:
                with self.subTest(asset=attribute, condition=label):
                    changed = replace(
                        original,
                        metadata=replace(
                            original.metadata,
                            status=status,
                            implementation=replace(
                                original.metadata.implementation,
                                sha256=sha256,
                            ),
                        ),
                    )
                    setattr(self, attribute, changed)
                    with self._resolved_dependencies():
                        with self.assertRaises(ValueError):
                            self._preflight(self.training_study)
        finally:
            setattr(self, attribute, original)

    def test_rejects_draft_missing_hash_and_mismatch_for_each_executable(self) -> None:
        self._assert_executable_states_rejected(
            "train_protocol",
            TrainProtocolStatus.DRAFT,
        )
        self._assert_executable_states_rejected(
            "architecture",
            ArchitectureStatus.DRAFT,
        )
        self._assert_executable_states_rejected(
            "evaluation_protocol",
            EvaluationProtocolStatus.DRAFT,
        )

    def test_corpus_integrity_failure_propagates_without_side_effects(self) -> None:
        failure = RuntimeError("corpus integrity failure")
        with (
            patch.object(subject, "resolve_corpus", side_effect=failure),
            patch(
                "mldb.src.study.expansion.materialize_study_plan"
            ) as materialize,
        ):
            with self.assertRaises(RuntimeError) as caught:
                self._preflight(self.training_study)

        self.assertIs(failure, caught.exception)
        materialize.assert_not_called()
        self.assertEqual(0, self.filesystem.mutations)

    def test_rejects_unknown_training_and_evaluation_parameter_keys(self) -> None:
        training_source = replace(
            self.training_study.metadata.model,
            parameters={
                "unknown": StudyTrainingParameterAxis(values=(1,)),
            },
        )
        unknown_training = replace(
            self.training_study,
            metadata=replace(
                self.training_study.metadata,
                model=training_source,
            ),
        )
        with self._resolved_dependencies():
            with self.assertRaisesRegex(ValueError, "not a published"):
                self._preflight(unknown_training)

        stage = replace(
            self.training_study.metadata.evaluations[0],
            parameters={"unknown": 1},
        )
        unknown_evaluation = replace(
            self.training_study,
            metadata=replace(
                self.training_study.metadata,
                evaluations=(stage,),
            ),
        )
        with self._resolved_dependencies():
            with self.assertRaisesRegex(ValueError, "not a published"):
                self._preflight(unknown_evaluation)

    def test_rejects_cross_asset_task_mismatch(self) -> None:
        original = self.evaluation_corpus
        self.evaluation_corpus = replace(
            original,
            metadata=replace(
                original.metadata,
                task=TaskId("other-task-v1"),
            ),
        )
        try:
            with self._resolved_dependencies():
                with self.assertRaisesRegex(ValueError, "Task ID"):
                    self._preflight(self.training_study)
        finally:
            self.evaluation_corpus = original

    def test_rejects_existing_model_weights_hash_or_byte_count_mismatch(self) -> None:
        original = self.model
        assert original.training_run.result is not None
        artifact = original.training_run.result.weights
        cases = (
            ("hash", replace(artifact, sha256="0" * 64)),
            ("bytes", replace(artifact, bytes=len(self.weights_bytes) + 1)),
        )
        try:
            for label, changed_artifact in cases:
                with self.subTest(condition=label):
                    changed_run = replace(
                        original.training_run,
                        result=TrainingRunResult(
                            weights=changed_artifact,
                        ),
                    )
                    self.model = replace(
                        original,
                        training_run=changed_run,
                    )
                    with self._resolved_dependencies():
                        with self.assertRaises(ValueError):
                            self._preflight(self.existing_study)
        finally:
            self.model = original

    def test_existing_model_architecture_requires_static_integrity(self) -> None:
        original = self.model
        changed_architecture = replace(
            original.architecture,
            metadata=replace(
                original.architecture.metadata,
                implementation=replace(
                    original.architecture.metadata.implementation,
                    sha256="0" * 64,
                ),
            ),
        )
        self.model = replace(original, architecture=changed_architecture)
        try:
            with self._resolved_dependencies() as calls:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    self._preflight(self.existing_study)
            calls.train_protocol.assert_not_called()
        finally:
            self.model = original

    def test_failure_never_allocates_materializes_or_writes(self) -> None:
        mismatch = replace(
            self.train_protocol,
            metadata=replace(
                self.train_protocol.metadata,
                task=TaskId("other-task-v1"),
            ),
        )
        original = self.train_protocol
        self.train_protocol = mismatch
        try:
            with (
                self._resolved_dependencies(),
                patch(
                    "mldb.src.study.expansion.materialize_study_plan"
                ) as materialize,
            ):
                with self.assertRaises(ValueError):
                    self._preflight(self.training_study)
            materialize.assert_not_called()
            self.assertEqual(0, self.filesystem.mutations)
        finally:
            self.train_protocol = original


if __name__ == "__main__":
    unittest.main()
