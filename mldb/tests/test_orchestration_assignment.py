from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import inspect
from pathlib import Path
import unittest

import mldb.src.orchestration.assignment as assignment_module
from mldb.src.catalog.architecture import (
    Architecture,
    ArchitectureImplementation,
    ArchitectureInterface,
    ArchitectureStatus,
    ArchitectureStructure,
)
from mldb.src.catalog.corpus import Corpus, CorpusArtifact, CorpusDataSpec
from mldb.src.catalog.task import CategoricalTarget, Task, TaskInput, TaskScope
from mldb.src.evaluation.preflight import EvaluationPreflight
from mldb.src.evaluation.protocol import (
    EvaluationOutputs,
    EvaluationProtocol,
    EvaluationProtocolImplementation,
    EvaluationProtocolStatus,
)
from mldb.src.evaluation.run import EvaluationRun, EvaluationRunExecution, EvaluationRunStatus
from mldb.src.model.identity import Model
from mldb.src.model.loading import ModelHandle
from mldb.src.orchestration.assignment import (
    project_evaluation_assignment,
    project_training_assignment,
)
from mldb.src.orchestration.queue import QueueAttempt
from mldb.src.orchestration.worker_api import ImmutableAssetDescriptor
from mldb.src.runtime.catalog_handles import ArchitectureHandle, CorpusHandle, TaskHandle
from mldb.src.runtime.definition_handles import EvaluationProtocolHandle, TrainProtocolHandle
from mldb.src.training.preflight import TrainingPreflight
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


class AssignmentProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training_preflight, self.training_run, self.training_attempt = _training_fixture()
        self.evaluation_preflight, self.evaluation_run, self.evaluation_attempt = _evaluation_fixture()

    def test_training_success_projects_exact_authority_metadata_and_descriptors(self) -> None:
        assets = _RecordingAssets()
        assignment = project_training_assignment(
            self.training_preflight, self.training_run, self.training_attempt, assets
        )

        self.assertEqual(
            (
                self.training_attempt.attempt_id,
                self.training_attempt.run_id,
                self.training_attempt.lease_token,
                self.training_attempt.lease_until,
                "training",
            ),
            (
                assignment.attempt_id,
                assignment.run_id,
                assignment.lease_token,
                assignment.lease_until,
                assignment.kind,
            ),
        )
        self.assertIs(assignment.task, self.training_preflight.task.metadata)
        self.assertIs(assignment.corpus, self.training_preflight.corpus.metadata)
        self.assertIs(assignment.architecture, self.training_preflight.architecture.metadata)
        self.assertIs(assignment.protocol, self.training_preflight.protocol.metadata)
        self.assertEqual(self.training_preflight.seed, assignment.seed)
        self.assertIs(assignment.parameters, self.training_preflight.parameters)
        self.assertEqual(_expected_training_calls(self.training_preflight), assets.calls)
        self.assertEqual(
            tuple(call[1] for call in assets.calls),
            (
                assignment.corpus_artifact.key,
                assignment.architecture_implementation.key,
                assignment.train_protocol_implementation.key,
            ),
        )

    def test_evaluation_success_projects_exact_authority_lineage_and_descriptors(self) -> None:
        assets = _RecordingAssets()
        assignment = project_evaluation_assignment(
            self.evaluation_preflight, self.evaluation_run, self.evaluation_attempt, assets
        )

        self.assertEqual(
            (
                self.evaluation_attempt.attempt_id,
                self.evaluation_attempt.run_id,
                self.evaluation_attempt.lease_token,
                self.evaluation_attempt.lease_until,
                "evaluation",
            ),
            (
                assignment.attempt_id,
                assignment.run_id,
                assignment.lease_token,
                assignment.lease_until,
                assignment.kind,
            ),
        )
        self.assertIs(assignment.task, self.evaluation_preflight.task.metadata)
        self.assertIs(assignment.corpus, self.evaluation_preflight.corpus.metadata)
        self.assertIs(assignment.model, self.evaluation_preflight.model.metadata)
        self.assertIs(assignment.training_run, self.evaluation_preflight.model.training_run)
        self.assertIs(
            assignment.model_architecture,
            self.evaluation_preflight.model.architecture.metadata,
        )
        self.assertIs(assignment.protocol, self.evaluation_preflight.protocol.metadata)
        self.assertIs(assignment.parameters, self.evaluation_preflight.parameters)
        self.assertEqual(_expected_evaluation_calls(self.evaluation_preflight), assets.calls)

    def test_training_rejects_every_execution_identity_mismatch_before_describing(self) -> None:
        cases = (
            (self.training_run, replace(self.training_attempt, run_id="tr-20260908-999")),
            (replace(self.training_run, corpus="corpus-other-v1"), self.training_attempt),
            (replace(self.training_run, architecture="architecture-other-v1"), self.training_attempt),
            (replace(self.training_run, train_protocol="train-other-v1"), self.training_attempt),
            (
                replace(
                    self.training_run,
                    execution=replace(self.training_run.execution, seed=99),
                ),
                self.training_attempt,
            ),
            (replace(self.training_run, parameters={"epochs": 99}), self.training_attempt),
        )
        for run, attempt in cases:
            with self.subTest(run=run, attempt=attempt):
                assets = _RecordingAssets()
                with self.assertRaises(ValueError):
                    project_training_assignment(self.training_preflight, run, attempt, assets)
                self.assertEqual([], assets.calls)

    def test_evaluation_rejects_every_execution_identity_mismatch_before_describing(self) -> None:
        cases = (
            (self.evaluation_run, replace(self.evaluation_attempt, run_id="ev-20260908-999")),
            (replace(self.evaluation_run, model="mdl-20260908-888"), self.evaluation_attempt),
            (replace(self.evaluation_run, corpus="corpus-other-v1"), self.evaluation_attempt),
            (
                replace(self.evaluation_run, evaluation_protocol="eval-other-v1"),
                self.evaluation_attempt,
            ),
            (replace(self.evaluation_run, parameters={"batch": 99}), self.evaluation_attempt),
        )
        for run, attempt in cases:
            with self.subTest(run=run, attempt=attempt):
                assets = _RecordingAssets()
                with self.assertRaises(ValueError):
                    project_evaluation_assignment(self.evaluation_preflight, run, attempt, assets)
                self.assertEqual([], assets.calls)

    def test_terminal_runs_and_closed_attempts_are_rejected_before_describing(self) -> None:
        cases = (
            (
                project_training_assignment,
                self.training_preflight,
                replace(self.training_run, status=TrainingRunStatus.FAILED),
                self.training_attempt,
            ),
            (
                project_training_assignment,
                self.training_preflight,
                self.training_run,
                replace(self.training_attempt, finished_at="2026-09-08T08:01:00.000000Z"),
            ),
            (
                project_evaluation_assignment,
                self.evaluation_preflight,
                replace(self.evaluation_run, status=EvaluationRunStatus.FAILED),
                self.evaluation_attempt,
            ),
            (
                project_evaluation_assignment,
                self.evaluation_preflight,
                self.evaluation_run,
                replace(self.evaluation_attempt, finished_at="2026-09-08T08:01:00.000000Z"),
            ),
        )
        for projector, preflight, run, attempt in cases:
            with self.subTest(projector=projector.__name__):
                assets = _RecordingAssets()
                with self.assertRaises(ValueError):
                    projector(preflight, run, attempt, assets)
                self.assertEqual([], assets.calls)

    def test_evaluation_rejects_each_model_lineage_mismatch_before_describing(self) -> None:
        model = self.evaluation_preflight.model
        lineage_cases = (
            replace(model.training_run, id="tr-20260908-008"),
            replace(model.training_run, status=TrainingRunStatus.RUNNING),
            replace(model.training_run, result=None),
            replace(model.training_run, architecture="architecture-other-v1"),
        )
        for training_run in lineage_cases:
            with self.subTest(training_run=training_run):
                preflight = replace(self.evaluation_preflight, model=replace(model, training_run=training_run))
                assets = _RecordingAssets()
                with self.assertRaises(ValueError):
                    project_evaluation_assignment(preflight, self.evaluation_run, self.evaluation_attempt, assets)
                self.assertEqual([], assets.calls)

    def test_invalid_descriptor_response_is_rejected_and_later_descriptors_are_not_requested(self) -> None:
        for invalid_field in ("type", "key", "sha256", "bytes"):
            with self.subTest(invalid_field=invalid_field):
                assets = _RecordingAssets(invalid_call=2, invalid_field=invalid_field)
                with self.assertRaises(ValueError):
                    project_training_assignment(
                        self.training_preflight,
                        self.training_run,
                        self.training_attempt,
                        assets,
                    )
                self.assertEqual(2, len(assets.calls))

    def test_replay_reconstructs_equal_training_and_evaluation_assignments(self) -> None:
        training_assets = _RecordingAssets()
        first_training = project_training_assignment(
            self.training_preflight, self.training_run, self.training_attempt, training_assets
        )
        second_training = project_training_assignment(
            deepcopy(self.training_preflight),
            deepcopy(self.training_run),
            deepcopy(self.training_attempt),
            training_assets,
        )
        self.assertEqual(first_training, second_training)

        evaluation_assets = _RecordingAssets()
        first_evaluation = project_evaluation_assignment(
            self.evaluation_preflight, self.evaluation_run, self.evaluation_attempt, evaluation_assets
        )
        second_evaluation = project_evaluation_assignment(
            deepcopy(self.evaluation_preflight),
            deepcopy(self.evaluation_run),
            deepcopy(self.evaluation_attempt),
            evaluation_assets,
        )
        self.assertEqual(first_evaluation, second_evaluation)

    def test_projection_does_not_mutate_preflight_run_or_attempt(self) -> None:
        training_snapshot = deepcopy(
            (self.training_preflight, self.training_run, self.training_attempt)
        )
        evaluation_snapshot = deepcopy(
            (self.evaluation_preflight, self.evaluation_run, self.evaluation_attempt)
        )

        project_training_assignment(
            self.training_preflight,
            self.training_run,
            self.training_attempt,
            _RecordingAssets(),
        )
        project_evaluation_assignment(
            self.evaluation_preflight,
            self.evaluation_run,
            self.evaluation_attempt,
            _RecordingAssets(),
        )

        self.assertEqual(
            training_snapshot,
            (self.training_preflight, self.training_run, self.training_attempt),
        )
        self.assertEqual(
            evaluation_snapshot,
            (self.evaluation_preflight, self.evaluation_run, self.evaluation_attempt),
        )

    def test_assignment_module_has_no_hash_resolution_preflight_or_mutation_dependencies(self) -> None:
        path = Path(assignment_module.__file__)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imports = {
            ("." * node.level) + (node.module or "")
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        }
        imports.update(
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        self.assertNotIn("hashlib", imports)
        self.assertFalse(any("repository" in module for module in imports))
        self.assertFalse(any("resolution" in module for module in imports))
        self.assertFalse(any("queue_ports" in module for module in imports))
        self.assertNotIn("preflight_training(", source)
        self.assertNotIn("preflight_evaluation(", source)
        self.assertNotIn("read_bytes(", source)

    def test_public_projector_signatures_are_exact(self) -> None:
        self.assertEqual(
            ("preflight", "run", "attempt", "assets"),
            tuple(inspect.signature(project_training_assignment).parameters),
        )
        self.assertEqual(
            ("preflight", "run", "attempt", "assets"),
            tuple(inspect.signature(project_evaluation_assignment).parameters),
        )


class _RecordingAssets:
    def __init__(self, invalid_call: int | None = None, invalid_field: str | None = None) -> None:
        self.invalid_call = invalid_call
        self.invalid_field = invalid_field
        self.calls: list[tuple[Path, str, str, int | None]] = []

    def describe(
        self,
        source_path: Path,
        *,
        key: str,
        sha256: str,
        bytes: int | None,
    ) -> ImmutableAssetDescriptor:
        self.calls.append((source_path, key, sha256, bytes))
        if self.invalid_call == len(self.calls):
            if self.invalid_field == "type":
                return object()  # type: ignore[return-value]
            if self.invalid_field == "key":
                key = "wrong-key"
            elif self.invalid_field == "sha256":
                sha256 = "0" * 64
            elif self.invalid_field == "bytes":
                bytes = 1 if bytes is None else bytes + 1
        retrieval_id = f"stable:{key}:{source_path.as_posix()}:{sha256}:{bytes}"
        return ImmutableAssetDescriptor(key, retrieval_id, sha256, bytes)

    def retrieve(self, asset: ImmutableAssetDescriptor) -> bytes:
        raise AssertionError("retrieve() is outside assignment projection")


def _task() -> Task:
    return Task(
        schema="mjtensu.mldb/task/v1",
        id="task-tiles-v1",
        name="tiles",
        problem_type="classification",
        description="tile classification",
        input=TaskInput("tile"),
        target=CategoricalTarget("categorical", ("a", "b")),
        semantics={},
        scope=TaskScope((), ()),
    )


def _corpus(task: Task) -> Corpus:
    return Corpus(
        schema="mjtensu.mldb/corpus/v1",
        id="corpus-main-v1",
        task=task.id,
        artifact=CorpusArtifact("sqlite", "c" * 64, 123),
        data=CorpusDataSpec("records-v1", "samples"),
        representation={},
        builder_parameters={},
        splits={"train": 10},
    )


def _architecture(task: Task) -> Architecture:
    return Architecture(
        schema="mjtensu.mldb/architecture/v1",
        id="architecture-main-v1",
        status=ArchitectureStatus.SEALED,
        task=task.id,
        name="main",
        family="plain",
        description="test architecture",
        implementation=ArchitectureImplementation("pytorch", "build", "a" * 64),
        interface=ArchitectureInterface({}, {"kind": "logits"}),
        structure=ArchitectureStructure("test"),
    )


def _train_protocol(task: Task) -> TrainProtocol:
    return TrainProtocol(
        schema="mjtensu.mldb/train-protocol/v1",
        id="train-main-v1",
        status=TrainProtocolStatus.SEALED,
        task=task.id,
        name="train",
        description="test train protocol",
        implementation=TrainProtocolImplementation("train", "b" * 64),
        parameters={},
    )


def _evaluation_protocol(task: Task) -> EvaluationProtocol:
    return EvaluationProtocol(
        schema="mjtensu.mldb/evaluation-protocol/v1",
        id="evaluation-main-v1",
        status=EvaluationProtocolStatus.SEALED,
        task=task.id,
        name="evaluate",
        description="test evaluation protocol",
        implementation=EvaluationProtocolImplementation("evaluate", "e" * 64),
        parameters={},
        outputs=EvaluationOutputs({}, {}),
    )


def _training_fixture() -> tuple[TrainingPreflight, TrainingRun, QueueAttempt]:
    task = _task()
    corpus = _corpus(task)
    architecture = _architecture(task)
    protocol = _train_protocol(task)
    parameters = {"epochs": 3, "nested": {"schedule": [1, 2]}}
    preflight = TrainingPreflight(
        task=TaskHandle(task, Path("task.yaml")),
        corpus=CorpusHandle(corpus, Path("corpus.yaml"), Path("corpus.sqlite"), Path("builder.py")),
        architecture=ArchitectureHandle(architecture, Path("architecture.yaml"), Path("architecture.py")),
        protocol=TrainProtocolHandle(protocol, Path("train.yaml"), Path("train.py")),
        seed=7,
        parameters=parameters,
    )
    run = TrainingRun(
        schema="mjtensu.mldb/training-run/v1",
        id="tr-20260908-001",
        status=TrainingRunStatus.RUNNING,
        corpus=corpus.id,
        architecture=architecture.id,
        train_protocol=protocol.id,
        parameters=deepcopy(parameters),
        execution=TrainingRunExecution(seed=7, started_at="2026-09-08T08:00:00Z"),
    )
    attempt = QueueAttempt(
        attempt_id=41,
        job_id=5,
        attempt_no=1,
        run_id=run.id,
        worker_id="worker-1",
        acquire_token="acquire-1",
        lease_token="lease-1",
        lease_until="2026-09-08T09:00:00.000000Z",
        started_at="2026-09-08T08:00:00.000000Z",
    )
    return preflight, run, attempt


def _evaluation_fixture() -> tuple[EvaluationPreflight, EvaluationRun, QueueAttempt]:
    task = _task()
    corpus = _corpus(task)
    architecture = _architecture(task)
    protocol = _evaluation_protocol(task)
    completed_training = TrainingRun(
        schema="mjtensu.mldb/training-run/v1",
        id="tr-20260908-009",
        status=TrainingRunStatus.COMPLETED,
        corpus=corpus.id,
        architecture=architecture.id,
        train_protocol="train-main-v1",
        parameters={"epochs": 3},
        execution=TrainingRunExecution(
            seed=7,
            started_at="2026-09-08T07:00:00Z",
            finished_at="2026-09-08T07:30:00Z",
        ),
        result=TrainingRunResult(
            CanonicalWeightsArtifact(
                "pytorch-state-dict",
                "artifacts/weights.pt",
                "d" * 64,
                321,
            )
        ),
    )
    model = Model("mjtensu.mldb/model/v1", "mdl-20260908-009", completed_training.id)
    model_handle = ModelHandle(
        metadata=model,
        metadata_path=Path("model.yaml"),
        training_run=completed_training,
        training_run_metadata_path=Path("run.yaml"),
        weights_path=Path("weights.pt"),
        architecture=ArchitectureHandle(
            architecture,
            Path("architecture.yaml"),
            Path("architecture.py"),
        ),
    )
    parameters = {"batch": 8, "nested": {"thresholds": [0.1, 0.2]}}
    preflight = EvaluationPreflight(
        task=TaskHandle(task, Path("task.yaml")),
        corpus=CorpusHandle(corpus, Path("corpus.yaml"), Path("corpus.sqlite"), Path("builder.py")),
        model=model_handle,
        protocol=EvaluationProtocolHandle(protocol, Path("evaluation.yaml"), Path("evaluation.py")),
        parameters=parameters,
    )
    run = EvaluationRun(
        schema="mjtensu.mldb/evaluation-run/v1",
        id="ev-20260908-002",
        status=EvaluationRunStatus.RUNNING,
        model=model.id,
        corpus=corpus.id,
        evaluation_protocol=protocol.id,
        parameters=deepcopy(parameters),
        execution=EvaluationRunExecution(started_at="2026-09-08T08:00:00Z"),
    )
    attempt = QueueAttempt(
        attempt_id=42,
        job_id=6,
        attempt_no=1,
        run_id=run.id,
        worker_id="worker-2",
        acquire_token="acquire-2",
        lease_token="lease-2",
        lease_until="2026-09-08T09:00:00.000000Z",
        started_at="2026-09-08T08:00:00.000000Z",
    )
    return preflight, run, attempt


def _expected_training_calls(
    preflight: TrainingPreflight,
) -> list[tuple[Path, str, str, int | None]]:
    return [
        (
            preflight.corpus.artifact_path,
            "corpus_artifact",
            preflight.corpus.metadata.artifact.sha256,
            preflight.corpus.metadata.artifact.bytes,
        ),
        (
            preflight.architecture.implementation_path,
            "architecture_implementation",
            preflight.architecture.metadata.implementation.sha256,
            None,
        ),
        (
            preflight.protocol.implementation_path,
            "train_protocol_implementation",
            preflight.protocol.metadata.implementation.sha256,
            None,
        ),
    ]


def _expected_evaluation_calls(
    preflight: EvaluationPreflight,
) -> list[tuple[Path, str, str, int | None]]:
    assert preflight.model.training_run.result is not None
    return [
        (
            preflight.corpus.artifact_path,
            "corpus_artifact",
            preflight.corpus.metadata.artifact.sha256,
            preflight.corpus.metadata.artifact.bytes,
        ),
        (
            preflight.model.architecture.implementation_path,
            "model_architecture_implementation",
            preflight.model.architecture.metadata.implementation.sha256,
            None,
        ),
        (
            preflight.model.weights_path,
            "model_weights",
            preflight.model.training_run.result.weights.sha256,
            preflight.model.training_run.result.weights.bytes,
        ),
        (
            preflight.protocol.implementation_path,
            "evaluation_protocol_implementation",
            preflight.protocol.metadata.implementation.sha256,
            None,
        ),
    ]


if __name__ == "__main__":
    unittest.main()
