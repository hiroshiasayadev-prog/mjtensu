from __future__ import annotations

from dataclasses import fields, replace
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mldb.src.common.errors import LifecycleConflictError
from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    EvaluationRunId,
    ModelId,
    StudyId,
    StudyRunId,
    TrainingRunId,
    TrainProtocolId,
)
from mldb.src.evaluation.run import (
    EvaluationRun,
    EvaluationRunExecution,
    EvaluationRunStatus,
    EvaluationRunStudyLineage,
)
from mldb.src.orchestration.jobs import (
    EvaluationJob,
    TrainingJob,
    derive_study_jobs,
)
from mldb.src.orchestration.queue import QueueAttempt, QueueJob, QueueJobStatus
from mldb.src.study import progress as subject
from mldb.src.study.plan import StudyPlanEvaluation, StudyPlanRow, StudyPlanTraining
from mldb.src.study.run import (
    StudyRun,
    StudyRunExecution,
    StudyRunPlan,
    StudyRunStatus,
)
from mldb.src.training.run import (
    TrainingRun,
    TrainingRunExecution,
    TrainingRunStatus,
    TrainingRunStudyLineage,
)


STUDY_RUN_ID = StudyRunId("sr-20260908-601")
AS_OF = "2026-09-08T12:00:00.000000Z"
STAMP = "2026-09-08T11:00:00.000000Z"


class _Queue:
    def __init__(self, jobs=(), opens=None, authorized=None):
        self.jobs = tuple(jobs)
        self.opens = dict(opens or {})
        self.authorized = dict(authorized or {})
        self.calls: list[tuple[object, ...]] = []

    def jobs_for_study_run(self, study_run_id):
        self.calls.append(("jobs", study_run_id))
        return self.jobs

    def open_attempt_for_job(self, job_id):
        self.calls.append(("open", job_id))
        return self.opens.get(job_id)

    def authorized_open_attempt(self, attempt_id, lease_token, *, as_of):
        self.calls.append(("authorized", attempt_id, lease_token, as_of))
        return self.authorized.get((attempt_id, lease_token, as_of))

    def __getattr__(self, name):
        if name in {
            "admit_study_jobs",
            "select_ready_job",
            "defer_ready_job",
            "fail_ready_job",
            "activate_attempt",
            "heartbeat_attempt",
            "expired_open_attempts",
            "close_attempt",
            "cancel_job",
            "repair_job_state",
        }:
            raise AssertionError(f"mutation/recovery operation used: {name}")
        raise AttributeError(name)


def _training_plan():
    return (
        StudyPlanRow(
            trial="trial-0001",
            training=StudyPlanTraining(
                architecture=ArchitectureId("arch-a"),
                corpus=CorpusId("corpus-train"),
                protocol=TrainProtocolId("train-proto"),
                seed=42,
                parameters={"epochs": 3, "nested": {"flag": True, "xs": [1, 2]}},
            ),
            evaluations=(
                StudyPlanEvaluation(
                    stage="quality",
                    corpus=CorpusId("corpus-eval"),
                    protocol=EvaluationProtocolId("eval-proto"),
                    parameters={"batch": 8},
                ),
            ),
        ),
    )


def _existing_plan():
    return (
        StudyPlanRow(
            trial="trial-0001",
            model=ModelId("mdl-20260908-501"),
            evaluations=(
                StudyPlanEvaluation(
                    stage="quality",
                    corpus=CorpusId("corpus-eval"),
                    protocol=EvaluationProtocolId("eval-proto"),
                    parameters={"batch": 8},
                ),
            ),
        ),
    )


def _study(*, status=StudyRunStatus.RUNNING, planned=True):
    return StudyRun(
        schema="mjtensu.mldb/study-run/v1",
        id=STUDY_RUN_ID,
        status=status,
        study=StudyId("study-a"),
        execution=StudyRunExecution(
            started_at=object(),
            finished_at=object() if status is not StudyRunStatus.RUNNING else None,
        ),
        plan=(
            StudyRunPlan(
                path="plan.jsonl",
                sha256="a" * 64,
                bytes=1,
                trials=1,
                evaluation_jobs=1,
            )
            if planned
            else None
        ),
    )


def _training_run(
    status,
    sequence=1,
    *,
    corpus=CorpusId("corpus-train"),
    parameters=None,
    trial="trial-0001",
):
    return TrainingRun(
        schema="mjtensu.mldb/training-run/v1",
        id=TrainingRunId(f"tr-20260908-{sequence:03d}"),
        status=status,
        corpus=corpus,
        architecture=ArchitectureId("arch-a"),
        train_protocol=TrainProtocolId("train-proto"),
        parameters=(
            {"epochs": 3, "nested": {"flag": True, "xs": [1, 2]}}
            if parameters is None
            else parameters
        ),
        execution=TrainingRunExecution(seed=42, started_at=object()),
        study=TrainingRunStudyLineage(run=STUDY_RUN_ID, trial=trial),
    )


def _evaluation_run(
    status,
    sequence=1,
    *,
    model=ModelId("mdl-20260908-501"),
    parameters=None,
    trial="trial-0001",
    stage="quality",
):
    return EvaluationRun(
        schema="mjtensu.mldb/evaluation-run/v1",
        id=EvaluationRunId(f"ev-20260908-{sequence:03d}"),
        status=status,
        model=model,
        corpus=CorpusId("corpus-eval"),
        evaluation_protocol=EvaluationProtocolId("eval-proto"),
        parameters={"batch": 8} if parameters is None else parameters,
        execution=EvaluationRunExecution(started_at=object()),
        study=EvaluationRunStudyLineage(
            run=STUDY_RUN_ID,
            trial=trial,
            stage=stage,
        ),
    )


def _jobs(plan, training_status=None, evaluation_status=None):
    logical = derive_study_jobs(STUDY_RUN_ID, plan)
    training = next((job for job in logical if isinstance(job, TrainingJob)), None)
    evaluation = next(job for job in logical if isinstance(job, EvaluationJob))
    rows = []
    next_id = 1
    training_row = None
    if training is not None:
        training_row = QueueJob(
            job_id=next_id,
            logical=training,
            status=training_status,
            dependency_job_id=None,
            retry_not_before=STAMP if training_status is QueueJobStatus.RETRY_WAIT else None,
            created_at=STAMP,
            updated_at=STAMP,
        )
        rows.append(training_row)
        next_id += 1
    evaluation_row = QueueJob(
        job_id=next_id,
        logical=evaluation,
        status=evaluation_status,
        dependency_job_id=training_row.job_id if training_row is not None else None,
        retry_not_before=STAMP if evaluation_status is QueueJobStatus.RETRY_WAIT else None,
        created_at=STAMP,
        updated_at=STAMP,
    )
    rows.append(evaluation_row)
    return tuple(rows)


def _attempt(job_id, run_id):
    return QueueAttempt(
        attempt_id=91,
        job_id=job_id,
        attempt_no=1,
        run_id=run_id,
        worker_id="worker-a",
        acquire_token="acquire-a",
        lease_token="lease-a",
        lease_until="2026-09-08T13:00:00.000000Z",
        started_at=STAMP,
    )


def _observe(
    plan,
    *,
    study=None,
    training_runs=(),
    evaluation_runs=(),
    queue=None,
    model_exists=False,
):
    study = study or _study()
    queue = queue or _Queue()
    layout = Mock()
    layout.model_metadata_path.side_effect = (
        lambda model_id: Path("models") / model_id / "model.yaml"
    )
    filesystem = Mock()
    filesystem.file_exists.return_value = model_exists

    completed_by_model = {
        subject.model_id_for_training_run(run.id): run
        for run in training_runs
        if run.status is TrainingRunStatus.COMPLETED
    }

    def resolve(model_id, observed_layout, observed_filesystem):
        run = completed_by_model[model_id]
        return SimpleNamespace(
            metadata=SimpleNamespace(id=model_id, training_run=run.id),
            training_run=run,
        )

    with (
        patch.object(subject, "read_study_run", return_value=study),
        patch.object(subject, "read_study_plan", return_value=plan),
        patch.object(subject, "list_training_runs", return_value=tuple(training_runs)),
        patch.object(subject, "list_evaluation_runs", return_value=tuple(evaluation_runs)),
        patch.object(subject, "resolve_model", side_effect=resolve),
    ):
        result = subject.get_study_run_progress(
            STUDY_RUN_ID,
            AS_OF,
            layout,
            filesystem,
            queue,
        )
    return result, layout, filesystem, queue


class StudyProgressCanonicalTests(unittest.TestCase):
    def test_no_plan_returns_exact_run_without_reading_plan_children_or_queue(self):
        run = _study(planned=False)
        layout = Mock()
        filesystem = Mock()
        queue = _Queue()
        with (
            patch.object(subject, "read_study_run", return_value=run),
            patch.object(subject, "read_study_plan") as read_plan,
            patch.object(subject, "list_training_runs") as list_training,
            patch.object(subject, "list_evaluation_runs") as list_evaluation,
        ):
            result = subject.get_study_run_progress(
                STUDY_RUN_ID, AS_OF, layout, filesystem, queue
            )

        self.assertIs(result.study_run, run)
        self.assertIsNone(result.training)
        self.assertIsNone(result.evaluation)
        self.assertEqual((), result.training_incomplete)
        self.assertEqual((), result.evaluation_incomplete)
        read_plan.assert_not_called()
        list_training.assert_not_called()
        list_evaluation.assert_not_called()
        self.assertEqual([], queue.calls)

    def test_existing_model_plan_has_no_training_section(self):
        plan = _existing_plan()
        queue = _Queue(_jobs(plan, evaluation_status=QueueJobStatus.READY))

        result, _, _, _ = _observe(plan, queue=queue)

        self.assertIsNone(result.training)
        self.assertEqual(
            subject.StudyRunProgressCounts(1, 0, 0, 1, 0, 0),
            result.evaluation,
        )

    def test_canonical_satisfaction_outranks_missing_queue(self):
        plan = _training_plan()
        training = _training_run(TrainingRunStatus.COMPLETED)
        evaluation = _evaluation_run(
            EvaluationRunStatus.COMPLETED,
            model=ModelId("mdl-20260908-001"),
        )
        queue = _Queue()

        result, _, _, observed_queue = _observe(
            plan,
            training_runs=(training,),
            evaluation_runs=(evaluation,),
            queue=queue,
            model_exists=True,
        )

        self.assertEqual(
            subject.StudyRunProgressCounts(1, 1, 0, 0, 0, 0),
            result.training,
        )
        self.assertEqual(
            subject.StudyRunProgressCounts(1, 1, 0, 0, 0, 0),
            result.evaluation,
        )
        self.assertEqual([], observed_queue.calls)

    def test_unsuccessful_evaluation_history_remains_retryable_and_diagnostic(self):
        plan = _existing_plan()
        for status in (
            EvaluationRunStatus.COMPLETED_PARTIAL,
            EvaluationRunStatus.FAILED,
            EvaluationRunStatus.CANCELLED,
        ):
            with self.subTest(status=status):
                run = _evaluation_run(status)
                queue = _Queue(
                    _jobs(plan, evaluation_status=QueueJobStatus.RETRY_WAIT)
                )
                result, _, _, _ = _observe(
                    plan,
                    evaluation_runs=(run,),
                    queue=queue,
                )
                self.assertEqual(1, result.evaluation.waiting)
                self.assertEqual(0, result.evaluation.satisfied)
                self.assertEqual(run.id, result.evaluation_incomplete[0].latest_run_id)
                self.assertIs(status, result.evaluation_incomplete[0].latest_status)

    def test_latest_diagnostics_use_typed_listing_order(self):
        plan = _training_plan()
        older = _training_run(TrainingRunStatus.FAILED, 1)
        latest = _training_run(TrainingRunStatus.CANCELLED, 2)
        queue = _Queue(
            _jobs(
                plan,
                training_status=QueueJobStatus.RETRY_WAIT,
                evaluation_status=QueueJobStatus.BLOCKED,
            )
        )

        result, _, _, _ = _observe(
            plan,
            training_runs=(older, latest),
            queue=queue,
        )

        self.assertEqual(latest.id, result.training_incomplete[0].latest_run_id)
        self.assertIs(
            TrainingRunStatus.CANCELLED,
            result.training_incomplete[0].latest_status,
        )
        self.assertIsNone(result.evaluation_incomplete[0].latest_run_id)


class StudyProgressQueueTests(unittest.TestCase):
    def test_valid_active_requires_matching_running_child_and_current_lease(self):
        plan = _existing_plan()
        running = _evaluation_run(EvaluationRunStatus.RUNNING)
        rows = _jobs(plan, evaluation_status=QueueJobStatus.ACTIVE)
        attempt = _attempt(rows[0].job_id, running.id)
        queue = _Queue(
            rows,
            opens={rows[0].job_id: attempt},
            authorized={(attempt.attempt_id, attempt.lease_token, AS_OF): attempt},
        )

        result, _, _, observed_queue = _observe(
            plan,
            evaluation_runs=(running,),
            queue=queue,
        )

        self.assertEqual(1, result.evaluation.active)
        self.assertIn(
            ("authorized", attempt.attempt_id, attempt.lease_token, AS_OF),
            observed_queue.calls,
        )

    def test_expired_or_mismatched_active_lease_degrades_section(self):
        plan = _existing_plan()
        running = _evaluation_run(EvaluationRunStatus.RUNNING)
        rows = _jobs(plan, evaluation_status=QueueJobStatus.ACTIVE)
        attempt = _attempt(rows[0].job_id, running.id)
        mismatched = replace(attempt, lease_token="other")
        for authorized in (None, mismatched):
            with self.subTest(authorized=authorized):
                mapping = (
                    {}
                    if authorized is None
                    else {(attempt.attempt_id, attempt.lease_token, AS_OF): authorized}
                )
                queue = _Queue(
                    rows,
                    opens={rows[0].job_id: attempt},
                    authorized=mapping,
                )
                result, _, _, _ = _observe(
                    plan,
                    evaluation_runs=(running,),
                    queue=queue,
                )
                self.assertIsNone(result.evaluation.active)
                self.assertIsNone(result.evaluation.waiting)
                self.assertIsNone(result.evaluation.blocked)
                self.assertIsNone(result.evaluation.unsatisfied_terminal)

    def test_waiting_blocked_and_upstream_terminal_matrix(self):
        plan = _training_plan()
        cases = (
            (
                QueueJobStatus.READY,
                subject.StudyRunProgressCounts(1, 0, 0, 1, 0, 0),
                subject.StudyRunProgressCounts(1, 0, 0, 0, 1, 0),
            ),
            (
                QueueJobStatus.FAILED,
                subject.StudyRunProgressCounts(1, 0, 0, 0, 0, 1),
                subject.StudyRunProgressCounts(1, 0, 0, 0, 0, 1),
            ),
        )
        for training_status, expected_training, expected_evaluation in cases:
            with self.subTest(training_status=training_status):
                queue = _Queue(
                    _jobs(
                        plan,
                        training_status=training_status,
                        evaluation_status=QueueJobStatus.BLOCKED,
                    )
                )
                result, _, _, _ = _observe(plan, queue=queue)
                self.assertEqual(expected_training, result.training)
                self.assertEqual(expected_evaluation, result.evaluation)

    def test_completed_training_missing_model_is_gap_without_model_creation(self):
        plan = _training_plan()
        completed = _training_run(TrainingRunStatus.COMPLETED)
        queue = _Queue(
            _jobs(
                plan,
                training_status=QueueJobStatus.FAILED,
                evaluation_status=QueueJobStatus.BLOCKED,
            )
        )

        with patch(
            "mldb.src.model.persistence.ensure_model_for_completed_training_run"
        ) as ensure:
            result, layout, filesystem, _ = _observe(
                plan,
                training_runs=(completed,),
                queue=queue,
                model_exists=False,
            )

        self.assertEqual(0, result.training.satisfied)
        self.assertEqual(1, result.training.unsatisfied_terminal)
        self.assertEqual(1, result.evaluation.blocked)
        self.assertEqual(completed.id, result.training_incomplete[0].latest_run_id)
        layout.model_metadata_path.assert_called_once_with(
            ModelId("mdl-20260908-001")
        )
        filesystem.file_exists.assert_called_once()
        ensure.assert_not_called()

    def test_stale_queue_satisfied_degrades_without_overriding_canonical_history(self):
        plan = _existing_plan()
        queue = _Queue(
            _jobs(plan, evaluation_status=QueueJobStatus.SATISFIED)
        )

        result, _, _, _ = _observe(plan, queue=queue)

        self.assertEqual(1, result.evaluation.total)
        self.assertEqual(0, result.evaluation.satisfied)
        self.assertIsNone(result.evaluation.active)

    def test_missing_duplicate_and_malformed_queue_degrade_only_evaluation(self):
        plan = _existing_plan()
        valid = _jobs(plan, evaluation_status=QueueJobStatus.READY)[0]
        cases = {
            "missing": (),
            "duplicate": (valid, replace(valid, job_id=2)),
            "malformed_dependency": (replace(valid, dependency_job_id=99),),
        }
        for name, rows in cases.items():
            with self.subTest(name=name):
                result, _, _, _ = _observe(plan, queue=_Queue(rows))
                self.assertIsNone(result.training)
                self.assertEqual(1, result.evaluation.total)
                self.assertEqual(0, result.evaluation.satisfied)
                self.assertIsNone(result.evaluation.active)
                self.assertIsNone(result.evaluation.waiting)
                self.assertIsNone(result.evaluation.blocked)
                self.assertIsNone(result.evaluation.unsatisfied_terminal)

    def test_training_and_evaluation_sections_degrade_independently(self):
        plan = _training_plan()
        rows = list(
            _jobs(
                plan,
                training_status=QueueJobStatus.READY,
                evaluation_status=QueueJobStatus.BLOCKED,
            )
        )
        rows[0] = replace(rows[0], retry_not_before="")
        result, _, _, _ = _observe(plan, queue=_Queue(rows))

        self.assertIsNone(result.training.active)
        self.assertIsNone(result.training.waiting)
        self.assertIsNone(result.evaluation.active)

        existing = _existing_plan()
        result, _, _, _ = _observe(
            existing,
            queue=_Queue(
                (
                    replace(
                        _jobs(
                            existing,
                            evaluation_status=QueueJobStatus.FAILED,
                        )[0],
                        retry_not_before="forbidden",
                    ),
                )
            ),
        )
        self.assertIsNone(result.training)
        self.assertIsNone(result.evaluation.unsatisfied_terminal)


class StudyProgressConsistencyTests(unittest.TestCase):
    def test_child_input_and_lineage_mismatches_fail(self):
        training_plan = _training_plan()
        existing_plan = _existing_plan()
        cases = (
            (
                training_plan,
                (_training_run(TrainingRunStatus.FAILED, corpus=CorpusId("wrong")),),
                (),
            ),
            (
                training_plan,
                (_training_run(TrainingRunStatus.FAILED, trial="trial-9999"),),
                (),
            ),
            (
                existing_plan,
                (),
                (_evaluation_run(EvaluationRunStatus.FAILED, model=ModelId("wrong")),),
            ),
            (
                existing_plan,
                (),
                (_evaluation_run(EvaluationRunStatus.FAILED, stage="unknown"),),
            ),
            (
                training_plan,
                (
                    _training_run(
                        TrainingRunStatus.FAILED,
                        parameters={
                            "epochs": 3,
                            "nested": {"flag": 1, "xs": [1, 2]},
                        },
                    ),
                ),
                (),
            ),
        )
        for plan, training_runs, evaluation_runs in cases:
            with self.subTest(
                training_runs=training_runs,
                evaluation_runs=evaluation_runs,
            ):
                with self.assertRaises(LifecycleConflictError):
                    _observe(
                        plan,
                        training_runs=training_runs,
                        evaluation_runs=evaluation_runs,
                    )

    def test_multiple_completed_training_runs_fail_as_ambiguous(self):
        plan = _training_plan()
        with self.assertRaises(LifecycleConflictError):
            _observe(
                plan,
                training_runs=(
                    _training_run(TrainingRunStatus.COMPLETED, 1),
                    _training_run(TrainingRunStatus.COMPLETED, 2),
                ),
            )

    def test_terminal_study_is_preserved_and_not_recomputed(self):
        plan = _training_plan()
        terminal = _study(status=StudyRunStatus.COMPLETED_WITH_FAILURES)
        queue = _Queue(
            _jobs(
                plan,
                training_status=QueueJobStatus.FAILED,
                evaluation_status=QueueJobStatus.BLOCKED,
            )
        )

        result, _, _, _ = _observe(plan, study=terminal, queue=queue)

        self.assertIs(result.study_run, terminal)
        self.assertIs(StudyRunStatus.COMPLETED_WITH_FAILURES, result.study_run.status)
        self.assertEqual(1, result.training.unsatisfied_terminal)
        self.assertEqual(1, result.evaluation.unsatisfied_terminal)

    def test_terminal_study_can_observe_still_authorized_active_attempt(self):
        plan = _existing_plan()
        terminal = _study(status=StudyRunStatus.CANCELLED)
        running = _evaluation_run(EvaluationRunStatus.RUNNING)
        rows = _jobs(plan, evaluation_status=QueueJobStatus.ACTIVE)
        attempt = _attempt(rows[0].job_id, running.id)
        queue = _Queue(
            rows,
            opens={rows[0].job_id: attempt},
            authorized={(attempt.attempt_id, attempt.lease_token, AS_OF): attempt},
        )

        result, _, _, _ = _observe(
            plan,
            study=terminal,
            evaluation_runs=(running,),
            queue=queue,
        )

        self.assertIs(result.study_run, terminal)
        self.assertEqual(1, result.evaluation.active)

    def test_terminal_study_with_retained_ready_work_degrades(self):
        plan = _existing_plan()
        terminal = _study(status=StudyRunStatus.CANCELLED)
        queue = _Queue(_jobs(plan, evaluation_status=QueueJobStatus.READY))

        result, _, _, _ = _observe(plan, study=terminal, queue=queue)

        self.assertIsNone(result.evaluation.active)
        self.assertIsNone(result.evaluation.waiting)


class StudyProgressSurfaceTests(unittest.TestCase):
    def test_public_signature_and_frozen_value_shapes(self):
        signature = inspect.signature(subject.get_study_run_progress)
        self.assertEqual(
            ("study_run_id", "as_of", "layout", "filesystem", "queue"),
            tuple(signature.parameters),
        )
        self.assertEqual(
            ("total", "satisfied", "active", "waiting", "blocked", "unsatisfied_terminal"),
            tuple(field.name for field in fields(subject.StudyRunProgressCounts)),
        )
        self.assertEqual(
            (
                "study_run",
                "training",
                "evaluation",
                "training_incomplete",
                "evaluation_incomplete",
            ),
            tuple(field.name for field in fields(subject.StudyRunProgress)),
        )

    def test_observation_uses_only_read_only_filesystem_and_queue_operations(self):
        plan = _existing_plan()
        queue = _Queue(
            _jobs(plan, evaluation_status=QueueJobStatus.RETRY_WAIT)
        )
        result, _, filesystem, observed_queue = _observe(plan, queue=queue)

        self.assertEqual(1, result.evaluation.waiting)
        for name in (
            "ensure_directory",
            "replace_text",
            "replace_bytes",
            "remove_file",
        ):
            getattr(filesystem, name).assert_not_called()
        self.assertTrue(
            all(call[0] in {"jobs", "open", "authorized"} for call in observed_queue.calls)
        )


if __name__ == "__main__":
    unittest.main()
