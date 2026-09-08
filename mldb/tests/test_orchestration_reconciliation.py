from __future__ import annotations

import dataclasses
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from mldb.src.common.errors import LifecycleConflictError
from mldb.src.common.ids import (
    ArchitectureId, CorpusId, EvaluationProtocolId, EvaluationRunId, ModelId,
    StudyId, StudyRunId, TrainProtocolId, TrainingRunId,
)
from mldb.src.evaluation.run import (
    EvaluationRun, EvaluationRunExecution, EvaluationRunStatus, EvaluationRunStudyLineage,
)
from mldb.src.model.identity import Model
from mldb.src.orchestration import reconciliation as subject
from mldb.src.orchestration._coordination import orchestration_exclusion
from mldb.src.orchestration.jobs import EvaluationJob, TrainingJob, derive_study_jobs
from mldb.src.orchestration.queue import QueueAttempt, QueueJob, QueueJobStatus
from mldb.src.orchestration.retry_policy import RetryDecision
from mldb.src.orchestration._sqlite_queue import SQLiteQueue
from mldb.src.study.plan import StudyPlanEvaluation, StudyPlanRow, StudyPlanTraining
from mldb.src.study.run import StudyRun, StudyRunExecution, StudyRunPlan, StudyRunStatus
from mldb.src.training.run import (
    TrainingRun, TrainingRunExecution, TrainingRunStatus, TrainingRunStudyLineage,
)
from mldb.tests.signature_guard import compare_module_signatures


STUDY_RUN_ID = StudyRunId("sr-20260908-001")
STAMP = "2026-09-08T09:00:00.000000Z"
QUEUE_AT = "2026-09-08T10:00:00.000000Z"
FINISHED = object()


def _training_plan():
    return (
        StudyPlanRow(
            trial="trial-0001",
            training=StudyPlanTraining(
                architecture=ArchitectureId("arch-a"),
                corpus=CorpusId("corpus-train"),
                protocol=TrainProtocolId("train-proto"),
                seed=42,
                parameters={"epochs": 3, "nested": {"x": [1, True]}},
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


def _study(status=StudyRunStatus.RUNNING, planned=True):
    return StudyRun(
        schema="mjtensu.mldb/study-run/v1",
        id=STUDY_RUN_ID,
        status=status,
        study=StudyId("study-a"),
        execution=StudyRunExecution(
            started_at="started",
            finished_at="already-finished" if status is not StudyRunStatus.RUNNING else None,
        ),
        plan=(StudyRunPlan("plan.jsonl", "a" * 64, 1, 1, 1) if planned else None),
        summary=None,
    )


def _training_run(status, sequence=1, **overrides):
    values = dict(
        schema="mjtensu.mldb/training-run/v1",
        id=TrainingRunId(f"tr-20260908-{sequence:03d}"),
        status=status,
        corpus=CorpusId("corpus-train"),
        architecture=ArchitectureId("arch-a"),
        train_protocol=TrainProtocolId("train-proto"),
        parameters={"epochs": 3, "nested": {"x": [1, True]}},
        execution=TrainingRunExecution(
            seed=42, started_at="child-start",
            finished_at="child-finished" if status is not TrainingRunStatus.RUNNING else None,
        ),
        study=TrainingRunStudyLineage(STUDY_RUN_ID, "trial-0001"),
    )
    values.update(overrides)
    return TrainingRun(**values)


def _evaluation_run(status, sequence=1, *, model=ModelId("mdl-20260908-501"), **overrides):
    values = dict(
        schema="mjtensu.mldb/evaluation-run/v1",
        id=EvaluationRunId(f"ev-20260908-{sequence:03d}"),
        status=status,
        model=model,
        corpus=CorpusId("corpus-eval"),
        evaluation_protocol=EvaluationProtocolId("eval-proto"),
        parameters={"batch": 8},
        execution=EvaluationRunExecution(
            started_at="child-start",
            finished_at="child-finished" if status is not EvaluationRunStatus.RUNNING else None,
        ),
        study=EvaluationRunStudyLineage(STUDY_RUN_ID, "trial-0001", "quality"),
    )
    values.update(overrides)
    return EvaluationRun(**values)


class _Policy:
    def __init__(self, retry_not_before=None):
        self.retry_not_before = retry_not_before
        self.calls = []

    def after_unsatisfied_attempt(self, job, attempt_history, outcome, *, at):
        self.calls.append((job, attempt_history, outcome, at))
        return RetryDecision(self.retry_not_before)


class _Queue:
    def __init__(self, rows=(), attempts=(), *, authorized=True, events=None):
        self.rows = list(rows)
        self.attempts = list(attempts)
        self.authorized = authorized
        self.events = events if events is not None else []
        self.admit_calls = []
        self.repair_calls = []
        self.close_calls = []

    def admit_study_jobs(self, study_run_id, jobs, *, admitted_at):
        self.admit_calls.append((study_run_id, jobs, admitted_at))
        existing = {row.logical: row for row in self.rows if isinstance(row, QueueJob)}
        next_id = max([row.job_id for row in self.rows if isinstance(row, QueueJob)] or [0]) + 1
        training_ids = {
            row.logical.coordinate.trial: row.job_id
            for row in self.rows
            if isinstance(row, QueueJob) and isinstance(row.logical, TrainingJob)
        }
        for logical in sorted(
            (j for j in jobs if isinstance(j, TrainingJob)), key=lambda j: j.coordinate.trial
        ):
            if logical not in existing:
                row = QueueJob(next_id, logical, QueueJobStatus.READY, None, None, admitted_at, admitted_at)
                self.rows.append(row); existing[logical] = row; training_ids[logical.coordinate.trial] = next_id; next_id += 1
        for logical in sorted(
            (j for j in jobs if isinstance(j, EvaluationJob)), key=lambda j: (j.coordinate.trial, j.coordinate.stage)
        ):
            if logical not in existing:
                dep = training_ids.get(logical.coordinate.trial) if logical.training_dependency is not None else None
                status = QueueJobStatus.BLOCKED if dep is not None else QueueJobStatus.READY
                row = QueueJob(next_id, logical, status, dep, None, admitted_at, admitted_at)
                self.rows.append(row); existing[logical] = row; next_id += 1
        return tuple(self.rows)

    def jobs_for_study_run(self, study_run_id): return tuple(self.rows)
    def job_by_id(self, job_id): return next((row for row in self.rows if isinstance(row, QueueJob) and row.job_id == job_id), None)
    def attempts_for_job(self, job_id): return tuple(a for a in self.attempts if a.job_id == job_id)
    def attempt_by_id(self, attempt_id): return next((a for a in self.attempts if a.attempt_id == attempt_id), None)
    def open_attempt_for_job(self, job_id): return next((a for a in self.attempts if a.job_id == job_id and a.finished_at is None), None)
    def authorized_open_attempt(self, attempt_id, lease_token, *, as_of):
        attempt = self.attempt_by_id(attempt_id)
        return attempt if self.authorized and attempt is not None and attempt.finished_at is None and attempt.lease_token == lease_token else None

    def _replace_row(self, next_row):
        self.rows = [next_row if isinstance(row, QueueJob) and row.job_id == next_row.job_id else row for row in self.rows]
        return next_row

    def repair_job_state(self, job_id, *, status, retry_not_before, at):
        self.events.append(("queue_repair", job_id, status))
        current = self.job_by_id(job_id)
        next_row = dataclasses.replace(current, status=status, retry_not_before=retry_not_before, updated_at=at)
        self.repair_calls.append((job_id, status, retry_not_before, at))
        return self._replace_row(next_row)

    def close_attempt(self, attempt_id, *, target_status, finished_at, retry_not_before=None, close_reason=None):
        self.events.append(("queue_close", attempt_id, target_status))
        attempt = self.attempt_by_id(attempt_id)
        closed = dataclasses.replace(attempt, finished_at=finished_at, close_reason=close_reason)
        self.attempts = [closed if a.attempt_id == attempt_id else a for a in self.attempts]
        current = self.job_by_id(attempt.job_id)
        next_row = dataclasses.replace(current, status=target_status, retry_not_before=retry_not_before, updated_at=finished_at)
        self._replace_row(next_row)
        self.close_calls.append((attempt_id, target_status, retry_not_before))
        return next_row, closed


def _rows(plan, training_status=QueueJobStatus.READY, evaluation_status=None):
    jobs = derive_study_jobs(STUDY_RUN_ID, plan)
    rows = []
    training = next((j for j in jobs if isinstance(j, TrainingJob)), None)
    evaluation = next(j for j in jobs if isinstance(j, EvaluationJob))
    if training is not None:
        rows.append(QueueJob(1, training, training_status, None, STAMP if training_status is QueueJobStatus.RETRY_WAIT else None, STAMP, STAMP))
        dep = 1
        if evaluation_status is None: evaluation_status = QueueJobStatus.BLOCKED
        eval_id = 2
    else:
        dep = None
        if evaluation_status is None: evaluation_status = QueueJobStatus.READY
        eval_id = 1
    rows.append(QueueJob(eval_id, evaluation, evaluation_status, dep, STAMP if evaluation_status is QueueJobStatus.RETRY_WAIT else None, STAMP, STAMP))
    return tuple(rows)


def _attempt(job_id, run_id):
    return QueueAttempt(
        attempt_id=91, job_id=job_id, attempt_no=1, run_id=run_id,
        worker_id="worker", acquire_token="acquire", lease_token="lease",
        lease_until="2026-09-08T11:00:00.000000Z", started_at=STAMP,
    )


def _invoke(plan, *, study=None, training_runs=(), evaluation_runs=(), queue=None, policy=None, ensure=None, fail_training=None, fail_evaluation=None, persist=None):
    study = study or _study()
    queue = queue or _Queue()
    policy = policy or _Policy()
    ensure = ensure or Mock(side_effect=lambda run, layout, fs: Model("mjtensu.mldb/model/v1", ModelId("mdl-" + str(run.id)[3:]), run.id))
    fail_training = fail_training or Mock(side_effect=lambda run, finished, layout, fs, failure=None: dataclasses.replace(run, status=TrainingRunStatus.FAILED, execution=dataclasses.replace(run.execution, finished_at=finished)))
    fail_evaluation = fail_evaluation or Mock(side_effect=lambda run, finished, layout, fs, failure=None: dataclasses.replace(run, status=EvaluationRunStatus.FAILED, execution=dataclasses.replace(run.execution, finished_at=finished)))
    persist = persist or Mock()
    with patch.multiple(
        subject,
        read_study_run=Mock(return_value=study),
        read_study_plan=Mock(return_value=plan),
        list_training_runs=Mock(return_value=tuple(training_runs)),
        list_evaluation_runs=Mock(return_value=tuple(evaluation_runs)),
        ensure_model_for_completed_training_run=ensure,
        fail_training_run=fail_training,
        fail_evaluation_run=fail_evaluation,
        persist_study_run_transition=persist,
    ):
        result = subject.reconcile_study_run(STUDY_RUN_ID, FINISHED, QUEUE_AT, object(), object(), queue, policy)
    return result, queue, policy, ensure, fail_training, fail_evaluation, persist


class ReconciliationTests(unittest.TestCase):
    def test_terminal_study_returns_exact_unchanged_without_plan_or_queue(self):
        terminal = _study(StudyRunStatus.CANCELLED, planned=True)
        queue = Mock()
        with patch.object(subject, "read_study_run", return_value=terminal), patch.object(subject, "read_study_plan") as read_plan:
            result = subject.reconcile_study_run(STUDY_RUN_ID, FINISHED, QUEUE_AT, object(), object(), queue, _Policy())
        self.assertIs(result, terminal); read_plan.assert_not_called(); queue.admit_study_jobs.assert_not_called()

    def test_running_without_plan_returns_exact_unchanged_without_plan_bytes_or_queue(self):
        running = _study(planned=False); queue = Mock()
        with patch.object(subject, "read_study_run", return_value=running), patch.object(subject, "read_study_plan") as read_plan:
            result = subject.reconcile_study_run(STUDY_RUN_ID, FINISHED, QUEUE_AT, object(), object(), queue, _Policy())
        self.assertIs(result, running); read_plan.assert_not_called(); queue.admit_study_jobs.assert_not_called()

    def test_missing_queue_rows_are_rebuilt_and_exact_admission_replay_preserves_state(self):
        plan = _existing_plan(); queue = _Queue()
        result, queue, *_ = _invoke(plan, queue=queue)
        self.assertEqual(StudyRunStatus.RUNNING, result.status)
        self.assertEqual(1, len(queue.rows)); self.assertEqual(QueueJobStatus.READY, queue.rows[0].status)
        created = queue.rows[0].created_at
        result, queue, *_ = _invoke(plan, queue=queue)
        self.assertEqual(created, queue.rows[0].created_at); self.assertEqual(2, len(queue.admit_calls))

    def test_extra_duplicate_or_malformed_queue_projection_is_inconsistent(self):
        plan = _existing_plan(); base = list(_rows(plan))
        logical = base[0].logical
        extras = [
            base + [QueueJob(99, EvaluationJob(dataclasses.replace(logical.coordinate, stage="extra"), None), QueueJobStatus.READY, None, None, STAMP, STAMP)],
            base + [dataclasses.replace(base[0], job_id=99)],
            [dataclasses.replace(base[0], retry_not_before=STAMP)],
        ]
        for rows in extras:
            with self.subTest(rows=rows):
                queue = _Queue(rows)
                with self.assertRaises(LifecycleConflictError): _invoke(plan, queue=queue)

    def test_completed_training_ensures_model_before_queue_satisfaction_and_repairs_stale_failed(self):
        plan = _training_plan(); training = _training_run(TrainingRunStatus.COMPLETED)
        events = []; queue = _Queue(_rows(plan, training_status=QueueJobStatus.FAILED), events=events)
        def ensure(run, layout, fs):
            events.append(("model_ensure", run.id)); return Model("mjtensu.mldb/model/v1", ModelId("mdl-20260908-001"), run.id)
        result, queue, _, ensure_mock, *_ = _invoke(plan, training_runs=(training,), queue=queue, ensure=Mock(side_effect=ensure))
        self.assertEqual(QueueJobStatus.SATISFIED, queue.job_by_id(1).status)
        self.assertEqual(QueueJobStatus.READY, queue.job_by_id(2).status)
        self.assertEqual("model_ensure", events[0][0]); self.assertEqual("queue_repair", events[1][0])
        self.assertEqual(StudyRunStatus.RUNNING, result.status); ensure_mock.assert_called_once()

    def test_duplicate_completed_training_is_ambiguity(self):
        plan = _training_plan()
        with self.assertRaisesRegex(LifecycleConflictError, "multiple completed"):
            _invoke(plan, training_runs=(_training_run(TrainingRunStatus.COMPLETED, 1), _training_run(TrainingRunStatus.COMPLETED, 2)), queue=_Queue(_rows(plan)))

    def test_multiple_running_or_satisfied_plus_running_child_history_is_inconsistent(self):
        plan = _training_plan()
        cases = (
            (_training_run(TrainingRunStatus.RUNNING, 1), _training_run(TrainingRunStatus.RUNNING, 2)),
            (_training_run(TrainingRunStatus.COMPLETED, 1), _training_run(TrainingRunStatus.RUNNING, 2)),
        )
        for runs in cases:
            with self.subTest(training_runs=runs), self.assertRaisesRegex(LifecycleConflictError, "ambiguous active Training"):
                _invoke(plan, training_runs=runs, queue=_Queue(_rows(plan)))

        plan = _existing_plan()
        cases = (
            (_evaluation_run(EvaluationRunStatus.RUNNING, 1), _evaluation_run(EvaluationRunStatus.RUNNING, 2)),
            (_evaluation_run(EvaluationRunStatus.COMPLETED, 1), _evaluation_run(EvaluationRunStatus.RUNNING, 2)),
        )
        for runs in cases:
            with self.subTest(evaluation_runs=runs), self.assertRaisesRegex(LifecycleConflictError, "ambiguous active Evaluation"):
                _invoke(plan, evaluation_runs=runs, queue=_Queue(_rows(plan)))

    def test_evaluation_completed_satisfies_but_partial_failed_cancelled_remain_unsatisfied(self):
        plan = _existing_plan()
        completed = _evaluation_run(EvaluationRunStatus.COMPLETED)
        result, queue, *_ = _invoke(plan, evaluation_runs=(completed,), queue=_Queue(_rows(plan, evaluation_status=QueueJobStatus.CANCELLED)))
        self.assertEqual(StudyRunStatus.COMPLETED, result.status); self.assertEqual(QueueJobStatus.SATISFIED, queue.rows[0].status)
        for status, outcome in ((EvaluationRunStatus.COMPLETED_PARTIAL, "completed_partial"), (EvaluationRunStatus.FAILED, "failed"), (EvaluationRunStatus.CANCELLED, "cancelled")):
            with self.subTest(status=status):
                policy = _Policy(None); queue = _Queue(_rows(plan))
                result, queue, policy, *_ = _invoke(plan, evaluation_runs=(_evaluation_run(status),), queue=queue, policy=policy)
                self.assertEqual(StudyRunStatus.COMPLETED_WITH_FAILURES, result.status)
                self.assertEqual(QueueJobStatus.FAILED, queue.rows[0].status)
                self.assertEqual(outcome, policy.calls[0][2])

    def test_lineage_or_input_mismatch_is_inconsistent(self):
        plan = _existing_plan()
        bad = _evaluation_run(EvaluationRunStatus.COMPLETED, corpus=CorpusId("wrong"))
        with self.assertRaisesRegex(LifecycleConflictError, "disagrees"):
            _invoke(plan, evaluation_runs=(bad,), queue=_Queue(_rows(plan)))

    def test_healthy_active_lease_survives_restart_same_child_and_attempt(self):
        plan = _existing_plan(); running = _evaluation_run(EvaluationRunStatus.RUNNING)
        row = _rows(plan, evaluation_status=QueueJobStatus.ACTIVE)[0]
        attempt = _attempt(row.job_id, running.id); queue = _Queue((row,), (attempt,), authorized=True)
        result, queue, policy, _, _, fail_eval, _ = _invoke(plan, evaluation_runs=(running,), queue=queue)
        self.assertEqual(StudyRunStatus.RUNNING, result.status); self.assertEqual(QueueJobStatus.ACTIVE, queue.rows[0].status)
        self.assertIsNone(queue.attempts[0].finished_at); fail_eval.assert_not_called(); self.assertEqual([], policy.calls)

    def test_expired_active_fails_canonical_first_then_closes_to_retry(self):
        plan = _existing_plan(); running = _evaluation_run(EvaluationRunStatus.RUNNING)
        row = _rows(plan, evaluation_status=QueueJobStatus.ACTIVE)[0]; attempt = _attempt(row.job_id, running.id)
        events = []; queue = _Queue((row,), (attempt,), authorized=False, events=events); policy = _Policy("2026-09-08T12:00:00.000000Z")
        def fail(run, finished, layout, fs, failure=None):
            self.assertIsNone(failure); events.append(("canonical_fail", run.id)); return dataclasses.replace(run, status=EvaluationRunStatus.FAILED, execution=dataclasses.replace(run.execution, finished_at=finished))
        result, queue, policy, _, _, fail_eval, _ = _invoke(plan, evaluation_runs=(running,), queue=queue, policy=policy, fail_evaluation=Mock(side_effect=fail))
        self.assertEqual(["canonical_fail", "queue_close"], [e[0] for e in events])
        self.assertEqual(QueueJobStatus.RETRY_WAIT, queue.rows[0].status); self.assertEqual(StudyRunStatus.RUNNING, result.status)
        fail_eval.assert_called_once(); self.assertEqual("failed", policy.calls[0][2])

    def test_orphan_running_without_attempt_is_failed_then_repaired_without_synthetic_attempt(self):
        plan = _existing_plan(); running = _evaluation_run(EvaluationRunStatus.RUNNING)
        queue = _Queue(_rows(plan)); policy = _Policy(None)
        result, queue, policy, _, _, fail_eval, _ = _invoke(plan, evaluation_runs=(running,), queue=queue, policy=policy)
        fail_eval.assert_called_once(); self.assertEqual([], queue.attempts)
        self.assertEqual(QueueJobStatus.FAILED, queue.rows[0].status); self.assertEqual(StudyRunStatus.COMPLETED_WITH_FAILURES, result.status)

    def test_retained_retry_and_terminal_decisions_are_preserved_without_policy_replay(self):
        plan = _existing_plan(); failed = _evaluation_run(EvaluationRunStatus.FAILED)
        for status, expected_study in ((QueueJobStatus.RETRY_WAIT, StudyRunStatus.RUNNING), (QueueJobStatus.FAILED, StudyRunStatus.COMPLETED_WITH_FAILURES), (QueueJobStatus.CANCELLED, StudyRunStatus.COMPLETED_WITH_FAILURES)):
            with self.subTest(status=status):
                policy = _Policy(None); queue = _Queue(_rows(plan, evaluation_status=status))
                result, queue, policy, *_ = _invoke(plan, evaluation_runs=(failed,), queue=queue, policy=policy)
                self.assertEqual(status, queue.rows[0].status); self.assertEqual(expected_study, result.status); self.assertEqual([], policy.calls)

    def test_retained_ready_decision_with_closed_attempt_is_preserved(self):
        plan = _existing_plan(); failed = _evaluation_run(EvaluationRunStatus.FAILED)
        row = _rows(plan, evaluation_status=QueueJobStatus.READY)[0]
        closed = dataclasses.replace(_attempt(row.job_id, failed.id), finished_at=STAMP)
        policy = _Policy(None); queue = _Queue((row,), (closed,))
        result, queue, policy, *_ = _invoke(plan, evaluation_runs=(failed,), queue=queue, policy=policy)
        self.assertEqual(QueueJobStatus.READY, queue.rows[0].status); self.assertEqual(StudyRunStatus.RUNNING, result.status); self.assertEqual([], policy.calls)

    def test_dependency_blocked_ready_and_terminal_matrix(self):
        plan = _training_plan()
        result, queue, *_ = _invoke(plan, queue=_Queue(_rows(plan)))
        self.assertEqual((QueueJobStatus.READY, QueueJobStatus.BLOCKED), tuple(r.status for r in queue.rows)); self.assertEqual(StudyRunStatus.RUNNING, result.status)

        completed = _training_run(TrainingRunStatus.COMPLETED)
        result, queue, *_ = _invoke(plan, training_runs=(completed,), queue=_Queue(_rows(plan)))
        self.assertEqual(QueueJobStatus.SATISFIED, queue.job_by_id(1).status); self.assertEqual(QueueJobStatus.READY, queue.job_by_id(2).status); self.assertEqual(StudyRunStatus.RUNNING, result.status)

        failed = _training_run(TrainingRunStatus.FAILED)
        rows = _rows(plan, training_status=QueueJobStatus.FAILED, evaluation_status=QueueJobStatus.BLOCKED)
        result, queue, policy, *_ = _invoke(plan, training_runs=(failed,), queue=_Queue(rows))
        self.assertEqual(QueueJobStatus.BLOCKED, queue.job_by_id(2).status); self.assertEqual(StudyRunStatus.COMPLETED_WITH_FAILURES, result.status); self.assertEqual([], policy.calls)

    def test_all_canonical_coordinates_satisfied_finalizes_completed_with_exact_shape(self):
        plan = _training_plan(); training = _training_run(TrainingRunStatus.COMPLETED)
        evaluation = _evaluation_run(EvaluationRunStatus.COMPLETED, model=ModelId("mdl-20260908-001"))
        persist = Mock()
        result, queue, *rest = _invoke(plan, training_runs=(training,), evaluation_runs=(evaluation,), queue=_Queue(_rows(plan)), persist=persist)
        self.assertEqual(StudyRunStatus.COMPLETED, result.status); self.assertEqual(FINISHED, result.execution.finished_at)
        self.assertEqual("started", result.execution.started_at); self.assertEqual(_study().plan, result.plan); self.assertIsNone(result.summary)
        persist.assert_called_once(); self.assertIs(persist.call_args.args[0], result)

    def test_progress_possible_stays_running_and_no_progress_finalizes_with_failures(self):
        plan = _existing_plan(); running = _study()
        result, _, *rest = _invoke(plan, study=running, queue=_Queue(_rows(plan)))
        self.assertIs(result, running)
        failed = _evaluation_run(EvaluationRunStatus.FAILED)
        result, _, *rest = _invoke(plan, evaluation_runs=(failed,), queue=_Queue(_rows(plan, evaluation_status=QueueJobStatus.FAILED)))
        self.assertEqual(StudyRunStatus.COMPLETED_WITH_FAILURES, result.status)

    def test_strict_restart_replay_does_not_refail_or_recompute_retry(self):
        plan = _existing_plan(); running = _evaluation_run(EvaluationRunStatus.RUNNING)
        row = _rows(plan, evaluation_status=QueueJobStatus.ACTIVE)[0]; attempt = _attempt(row.job_id, running.id)
        queue = _Queue((row,), (attempt,), authorized=False); policy = _Policy("2026-09-08T12:00:00.000000Z")
        current_runs = [running]
        def fail(run, finished, layout, fs, failure=None):
            failed = dataclasses.replace(run, status=EvaluationRunStatus.FAILED, execution=dataclasses.replace(run.execution, finished_at=finished)); current_runs[:] = [failed]; return failed
        fail_mock = Mock(side_effect=fail)
        def invoke_once():
            with patch.multiple(subject, read_study_run=Mock(return_value=_study()), read_study_plan=Mock(return_value=plan), list_training_runs=Mock(return_value=()), list_evaluation_runs=Mock(side_effect=lambda l, f: tuple(current_runs)), fail_evaluation_run=fail_mock, persist_study_run_transition=Mock()):
                return subject.reconcile_study_run(STUDY_RUN_ID, FINISHED, QUEUE_AT, object(), object(), queue, policy)
        first = invoke_once(); second = invoke_once()
        self.assertEqual(StudyRunStatus.RUNNING, first.status); self.assertEqual(StudyRunStatus.RUNNING, second.status)
        self.assertEqual(1, fail_mock.call_count); self.assertEqual(1, len(policy.calls)); self.assertEqual(QueueJobStatus.RETRY_WAIT, queue.rows[0].status)

    def test_maintenance_exclusion_serializes_other_orchestration_mutation(self):
        plan = _existing_plan(); entered = threading.Event(); release = threading.Event(); second_entered = threading.Event(); errors = []
        def read_study(*args):
            entered.set(); release.wait(1); return _study()
        def reconcile():
            try:
                with patch.multiple(subject, read_study_run=Mock(side_effect=read_study), read_study_plan=Mock(return_value=plan), list_training_runs=Mock(return_value=()), list_evaluation_runs=Mock(return_value=()), persist_study_run_transition=Mock()):
                    subject.reconcile_study_run(STUDY_RUN_ID, FINISHED, QUEUE_AT, object(), object(), _Queue(_rows(plan)), _Policy())
            except BaseException as exc: errors.append(exc)
        first = threading.Thread(target=reconcile); first.start(); self.assertTrue(entered.wait(1))
        second = threading.Thread(target=lambda: (orchestration_exclusion().__enter__(), second_entered.set()))
        # use a proper helper so the context is released
        def other():
            with orchestration_exclusion(): second_entered.set()
        second = threading.Thread(target=other); second.start()
        self.assertFalse(second_entered.wait(0.05)); release.set(); first.join(1); second.join(1)
        self.assertTrue(second_entered.is_set()); self.assertEqual([], errors)


    def test_real_sqlite_projection_training_satisfaction_releases_dependency(self):
        plan = _training_plan(); training = _training_run(TrainingRunStatus.COMPLETED)
        with tempfile.TemporaryDirectory() as temporary:
            queue = SQLiteQueue(Path(temporary)); policy = _Policy()
            result, queue, *_ = _invoke(plan, training_runs=(training,), queue=queue, policy=policy)
            rows = queue.jobs_for_study_run(STUDY_RUN_ID)
        training_row = next(r for r in rows if isinstance(r.logical, TrainingJob))
        evaluation_row = next(r for r in rows if isinstance(r.logical, EvaluationJob))
        self.assertEqual(QueueJobStatus.SATISFIED, training_row.status)
        self.assertEqual(QueueJobStatus.READY, evaluation_row.status)
        self.assertEqual(StudyRunStatus.RUNNING, result.status)

    def test_real_sqlite_expired_active_canonical_first_recovery(self):
        plan = _existing_plan(); running = _evaluation_run(EvaluationRunStatus.RUNNING)
        with tempfile.TemporaryDirectory() as temporary:
            queue = SQLiteQueue(Path(temporary)); jobs = derive_study_jobs(STUDY_RUN_ID, plan)
            admitted = queue.admit_study_jobs(STUDY_RUN_ID, jobs, admitted_at="2026-09-08T08:00:00.000000Z")
            row = admitted[0]
            queue.activate_attempt(row.job_id, running.id, "worker", "acquire", "lease", activated_at="2026-09-08T08:00:00.000000Z")
            policy = _Policy(None)
            result, queue, policy, _, _, fail_eval, _ = _invoke(plan, evaluation_runs=(running,), queue=queue, policy=policy)
            repaired = queue.jobs_for_study_run(STUDY_RUN_ID)[0]
            attempt = queue.attempts_for_job(repaired.job_id)[0]
        self.assertEqual(QueueJobStatus.FAILED, repaired.status)
        self.assertIsNotNone(attempt.finished_at)
        self.assertEqual(StudyRunStatus.COMPLETED_WITH_FAILURES, result.status)
        fail_eval.assert_called_once(); self.assertEqual(1, len(policy.calls))

    def test_public_signature_matches_skeleton(self):
        implementation = Path(subject.__file__)
        skeleton = implementation.parents[2] / "skeleton" / "orchestration" / "reconciliation.py"
        self.assertEqual((), compare_module_signatures(skeleton, implementation))


if __name__ == "__main__":
    unittest.main()
