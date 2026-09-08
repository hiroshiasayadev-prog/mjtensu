from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock, call, patch

from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    ModelId,
    StudyId,
    StudyRunId,
    TaskId,
    TrainProtocolId,
)
from mldb.src.common.parameters import PublicParameterDeclaration
from mldb.src.evaluation.protocol import (
    EvaluationOutputs,
    EvaluationProtocol,
    EvaluationProtocolImplementation,
    EvaluationProtocolStatus,
)
from mldb.src.orchestration import study_launch as subject
from mldb.src.orchestration.jobs import EvaluationJob, TrainingJob
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.run_persistence import read_study_run
from mldb.src.study.definition import (
    Study,
    StudyEvaluationStage,
    StudyExistingModelSource,
    StudyStatus,
    StudyTrainingModelSource,
    StudyTrainingParameterAxis,
)
from mldb.src.study.preflight import PreparedStudyExecution
from mldb.src.study.run import (
    StudyRun,
    StudyRunExecution,
    StudyRunPlan,
    StudyRunStatus,
)
from mldb.src.training.protocol import (
    TrainProtocol,
    TrainProtocolImplementation,
    TrainProtocolStatus,
)


class _RecordingQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[StudyRunId, frozenset[object], str]] = []

    def admit_study_jobs(self, study_run_id, jobs, *, admitted_at):
        self.calls.append((study_run_id, jobs, admitted_at))
        return ()


class StudyLaunchIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.layout = RepositoryLayout(Path(self.temp.name))
        self.filesystem = LocalFilesystem()
        self.day = date(2026, 9, 8)
        self.started_at = datetime(2026, 9, 8, 7, 0, tzinfo=timezone.utc)
        self.failed_at = datetime(2026, 9, 8, 7, 1, tzinfo=timezone.utc)
        self.admitted_at = "2026-09-08T07:00:01.000000Z"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_training_source_launch_persists_finalized_running_plan_and_jobs(self) -> None:
        prepared = _training_prepared()
        queue = _RecordingQueue()

        run = subject.launch_study_execution(
            prepared,
            self.day,
            self.started_at,
            self.layout,
            self.filesystem,
            queue,
            admitted_at=self.admitted_at,
            failed_at=self.failed_at,
        )

        self.assertEqual(StudyRunStatus.RUNNING, run.status)
        self.assertIsNotNone(run.plan)
        self.assertEqual(run, read_study_run(run.id, self.layout, self.filesystem))
        self.assertEqual(1, len(queue.calls))
        admitted_id, jobs, admitted_at = queue.calls[0]
        self.assertEqual(run.id, admitted_id)
        self.assertEqual(self.admitted_at, admitted_at)
        self.assertEqual(4, sum(isinstance(job, TrainingJob) for job in jobs))
        self.assertEqual(4, sum(isinstance(job, EvaluationJob) for job in jobs))

    def test_existing_model_launch_derives_only_dependency_free_evaluations(self) -> None:
        prepared = _existing_prepared()
        queue = _RecordingQueue()

        run = subject.launch_study_execution(
            prepared,
            self.day,
            self.started_at,
            self.layout,
            self.filesystem,
            queue,
            admitted_at=self.admitted_at,
            failed_at=self.failed_at,
        )

        self.assertEqual(StudyRunStatus.RUNNING, run.status)
        self.assertIsNotNone(run.plan)
        _, jobs, _ = queue.calls[0]
        self.assertEqual(2, len(jobs))
        self.assertTrue(all(isinstance(job, EvaluationJob) for job in jobs))
        self.assertTrue(all(job.training_dependency is None for job in jobs))


class StudyLaunchCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prepared = _training_prepared()
        self.layout = Mock()
        self.filesystem = Mock()
        self.queue = Mock()
        self.day = date(2026, 9, 8)
        self.started_at = object()
        self.failed_at = object()
        self.admitted_at = "2026-09-08T07:00:01.000000Z"
        self.allocated_id = StudyRunId("sr-20260908-321")
        self.allocated = _run(self.prepared, self.allocated_id, plan=None)
        self.plan_metadata = _plan_metadata()
        self.finalized = _run(
            self.prepared, self.allocated_id, plan=self.plan_metadata
        )
        self.plan = (object(),)
        self.jobs = frozenset({"job"})

    def _launch(self):
        return subject.launch_study_execution(
            self.prepared,
            self.day,
            self.started_at,
            self.layout,
            self.filesystem,
            self.queue,
            admitted_at=self.admitted_at,
            failed_at=self.failed_at,
        )

    def test_success_ordering_and_exact_arguments(self) -> None:
        events: list[str] = []
        self.queue.admit_study_jobs.side_effect = lambda *args, **kwargs: events.append("queue")

        with (
            patch.object(subject, "allocate_study_run", side_effect=lambda *a, **k: (events.append("allocate"), self.allocated)[1]) as allocate,
            patch.object(subject, "materialize_study_plan", side_effect=lambda *a, **k: (events.append("materialize"), self.plan)[1]) as materialize,
            patch.object(subject, "finalize_study_plan", side_effect=lambda *a, **k: (events.append("finalize"), self.finalized)[1]) as finalize,
            patch.object(subject, "derive_study_jobs", side_effect=lambda *a, **k: (events.append("derive"), self.jobs)[1]) as derive,
        ):
            result = self._launch()

        self.assertIs(self.finalized, result)
        self.assertEqual(["allocate", "materialize", "finalize", "derive", "queue"], events)
        allocate.assert_called_once_with(
            self.prepared, self.day, self.started_at, self.layout, self.filesystem
        )
        materialize.assert_called_once_with(
            self.prepared.study.metadata,
            self.prepared.train_protocol.metadata,
            {
                key: handle.metadata
                for key, handle in self.prepared.evaluation_protocols.items()
            },
        )
        finalize.assert_called_once_with(
            self.allocated_id, self.plan, self.layout, self.filesystem
        )
        derive.assert_called_once_with(self.allocated_id, self.plan)
        self.queue.admit_study_jobs.assert_called_once_with(
            self.allocated_id, self.jobs, admitted_at=self.admitted_at
        )

    def test_allocation_failure_propagates_unchanged_without_cleanup(self) -> None:
        failure = RuntimeError("allocation failed")
        with (
            patch.object(subject, "allocate_study_run", side_effect=failure),
            patch.object(subject, "materialize_study_plan") as materialize,
            patch.object(subject, "read_study_run") as read,
            patch.object(subject, "persist_study_run_transition") as persist,
        ):
            with self.assertRaises(RuntimeError) as caught:
                self._launch()

        self.assertIs(failure, caught.exception)
        materialize.assert_not_called()
        read.assert_not_called()
        persist.assert_not_called()
        self.queue.assert_not_called()

    def test_each_after_allocation_failure_marks_exact_canonical_run_failed(self) -> None:
        for stage in ("materialize", "finalize", "derive", "admission"):
            with self.subTest(stage=stage):
                self.queue = Mock()
                setup_failure = RuntimeError(f"{stage} failed")
                current = (
                    self.allocated
                    if stage in {"materialize", "finalize"}
                    else self.finalized
                )
                self.queue.admit_study_jobs.side_effect = (
                    setup_failure if stage == "admission" else None
                )
                with (
                    patch.object(subject, "allocate_study_run", return_value=self.allocated),
                    patch.object(
                        subject,
                        "materialize_study_plan",
                        side_effect=setup_failure if stage == "materialize" else None,
                        return_value=None if stage == "materialize" else self.plan,
                    ),
                    patch.object(
                        subject,
                        "finalize_study_plan",
                        side_effect=setup_failure if stage == "finalize" else None,
                        return_value=None if stage == "finalize" else self.finalized,
                    ),
                    patch.object(
                        subject,
                        "derive_study_jobs",
                        side_effect=setup_failure if stage == "derive" else None,
                        return_value=None if stage == "derive" else self.jobs,
                    ),
                    patch.object(subject, "read_study_run", return_value=current) as read,
                    patch.object(subject, "persist_study_run_transition") as persist,
                ):
                    with self.assertRaises(subject.StudyExecutionSetupError) as caught:
                        self._launch()

                self.assertEqual(self.allocated_id, caught.exception.study_run_id)
                self.assertIs(setup_failure, caught.exception.__cause__)
                read.assert_called_once_with(
                    self.allocated_id, self.layout, self.filesystem
                )
                persist.assert_called_once()
                failed = persist.call_args.args[0]
                self.assertEqual(self.allocated_id, failed.id)
                self.assertEqual(StudyRunStatus.FAILED, failed.status)
                self.assertIs(self.failed_at, failed.execution.finished_at)
                self.assertEqual(current.execution.started_at, failed.execution.started_at)
                self.assertEqual(current.plan, failed.plan)
                self.assertEqual(current.summary, failed.summary)
                self.assertEqual(
                    (self.layout, self.filesystem), persist.call_args.args[1:]
                )

    def test_queue_admission_failure_does_not_attempt_rollback(self) -> None:
        failure = RuntimeError("queue admission failed")
        self.queue = Mock()
        self.queue.admit_study_jobs.side_effect = failure
        with (
            patch.object(subject, "allocate_study_run", return_value=self.allocated),
            patch.object(subject, "materialize_study_plan", return_value=self.plan),
            patch.object(subject, "finalize_study_plan", return_value=self.finalized),
            patch.object(subject, "derive_study_jobs", return_value=self.jobs),
            patch.object(subject, "read_study_run", return_value=self.finalized),
            patch.object(subject, "persist_study_run_transition"),
        ):
            with self.assertRaises(subject.StudyExecutionSetupError):
                self._launch()

        self.assertEqual(
            [call.admit_study_jobs(self.allocated_id, self.jobs, admitted_at=self.admitted_at)],
            self.queue.mock_calls,
        )

    def test_cleanup_read_failure_keeps_allocated_identity_and_chaining(self) -> None:
        setup_failure = RuntimeError("materialization failed")
        cleanup_failure = RuntimeError("cleanup read failed")
        with (
            patch.object(subject, "allocate_study_run", return_value=self.allocated),
            patch.object(subject, "materialize_study_plan", side_effect=setup_failure),
            patch.object(subject, "read_study_run", side_effect=cleanup_failure),
            patch.object(subject, "persist_study_run_transition") as persist,
        ):
            with self.assertRaises(subject.StudyExecutionSetupError) as caught:
                self._launch()

        self.assertEqual(self.allocated_id, caught.exception.study_run_id)
        self.assertIs(cleanup_failure, caught.exception.__cause__)
        self.assertIs(setup_failure, cleanup_failure.__context__)
        persist.assert_not_called()

    def test_cleanup_persist_failure_keeps_allocated_identity_and_chaining(self) -> None:
        setup_failure = RuntimeError("materialization failed")
        cleanup_failure = RuntimeError("cleanup persist failed")
        with (
            patch.object(subject, "allocate_study_run", return_value=self.allocated),
            patch.object(subject, "materialize_study_plan", side_effect=setup_failure),
            patch.object(subject, "read_study_run", return_value=self.allocated),
            patch.object(
                subject, "persist_study_run_transition", side_effect=cleanup_failure
            ),
        ):
            with self.assertRaises(subject.StudyExecutionSetupError) as caught:
                self._launch()

        self.assertEqual(self.allocated_id, caught.exception.study_run_id)
        self.assertIs(cleanup_failure, caught.exception.__cause__)
        self.assertIs(setup_failure, cleanup_failure.__context__)


def _training_prepared() -> PreparedStudyExecution:
    task_id = TaskId("task-v1")
    train_protocol = TrainProtocol(
        schema="mjtensu.mldb/train-protocol/v1",
        id=TrainProtocolId("train-v1"),
        status=TrainProtocolStatus.SEALED,
        task=task_id,
        name="train",
        description="train",
        implementation=TrainProtocolImplementation(
            entrypoint="train", sha256="a" * 64
        ),
        parameters={
            "epochs": PublicParameterDeclaration(default=10),
            "learning_rate": PublicParameterDeclaration(default=0.001),
        },
    )
    evaluation_protocol = _evaluation_protocol(task_id)
    study = Study(
        schema="mjtensu.mldb/study/v1",
        id=StudyId("launch-training-v1"),
        status=StudyStatus.SEALED,
        name="launch training",
        description="launch training",
        model=StudyTrainingModelSource(
            corpus=CorpusId("train-corpus-v1"),
            protocol=train_protocol.id,
            architectures=(ArchitectureId("arch-v1"),),
            parameters={
                "learning_rate": StudyTrainingParameterAxis(values=(0.001, 0.0003)),
            },
            seeds=(42, 43),
        ),
        evaluations=(
            StudyEvaluationStage(
                stage="holdout",
                corpus=CorpusId("eval-corpus-v1"),
                protocol=evaluation_protocol.id,
                parameters={"batch_size": 64},
            ),
        ),
    )
    evaluation_handle = SimpleNamespace(metadata=evaluation_protocol)
    return PreparedStudyExecution(
        study=SimpleNamespace(metadata=study),
        task=SimpleNamespace(metadata=SimpleNamespace(id=task_id)),
        train_protocol=SimpleNamespace(metadata=train_protocol),
        evaluation_protocols=MappingProxyType(
            {evaluation_protocol.id: evaluation_handle}
        ),
    )


def _existing_prepared() -> PreparedStudyExecution:
    task_id = TaskId("task-v1")
    evaluation_protocol = _evaluation_protocol(task_id)
    study = Study(
        schema="mjtensu.mldb/study/v1",
        id=StudyId("launch-existing-v1"),
        status=StudyStatus.SEALED,
        name="launch existing",
        description="launch existing",
        model=StudyExistingModelSource(
            models=(
                ModelId("mdl-20260908-001"),
                ModelId("mdl-20260908-002"),
            )
        ),
        evaluations=(
            StudyEvaluationStage(
                stage="holdout",
                corpus=CorpusId("eval-corpus-v1"),
                protocol=evaluation_protocol.id,
                parameters={},
            ),
        ),
    )
    evaluation_handle = SimpleNamespace(metadata=evaluation_protocol)
    return PreparedStudyExecution(
        study=SimpleNamespace(metadata=study),
        task=SimpleNamespace(metadata=SimpleNamespace(id=task_id)),
        train_protocol=None,
        evaluation_protocols=MappingProxyType(
            {evaluation_protocol.id: evaluation_handle}
        ),
    )


def _evaluation_protocol(task_id: TaskId) -> EvaluationProtocol:
    return EvaluationProtocol(
        schema="mjtensu.mldb/evaluation-protocol/v1",
        id=EvaluationProtocolId("eval-v1"),
        status=EvaluationProtocolStatus.SEALED,
        task=task_id,
        name="evaluation",
        description="evaluation",
        implementation=EvaluationProtocolImplementation(
            entrypoint="evaluate", sha256="b" * 64
        ),
        parameters={
            "batch_size": PublicParameterDeclaration(default=32),
        },
        outputs=EvaluationOutputs(metrics={}, artifacts={}),
    )


def _plan_metadata() -> StudyRunPlan:
    return StudyRunPlan(
        path="plan.jsonl",
        sha256="c" * 64,
        bytes=123,
        trials=1,
        evaluation_jobs=1,
    )


def _run(
    prepared: PreparedStudyExecution,
    run_id: StudyRunId,
    *,
    plan: StudyRunPlan | None,
) -> StudyRun:
    return StudyRun(
        schema="mjtensu.mldb/study-run/v1",
        id=run_id,
        status=StudyRunStatus.RUNNING,
        study=prepared.study.metadata.id,
        execution=StudyRunExecution(started_at="started"),
        plan=plan,
        summary=None,
    )


if __name__ == "__main__":
    unittest.main()
