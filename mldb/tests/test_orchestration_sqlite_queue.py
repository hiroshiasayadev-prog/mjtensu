from __future__ import annotations

import inspect
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
from threading import Barrier, Thread
import unittest

from mldb.src.common.ids import EvaluationRunId, StudyRunId, TrainingRunId
from mldb.src.orchestration._sqlite_queue import SQLiteQueue
from mldb.src.orchestration.jobs import (
    EvaluationJob,
    EvaluationJobCoordinate,
    TrainingJob,
    TrainingJobCoordinate,
)
from mldb.src.orchestration.queue import QueueJobStatus, queue_database_path
from mldb.src.orchestration.queue_ports import QueuePort


T0 = "2026-09-08T10:00:00.000000Z"
T1 = "2026-09-08T10:15:00.000000Z"
T2 = "2026-09-08T11:00:00.000000Z"
T3 = "2026-09-08T12:00:00.000000Z"


class SQLiteQueueTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.queue = SQLiteQueue(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @property
    def database(self) -> Path:
        return queue_database_path(self.root)

    def admit_training_trial(self, study: str = "sr-20260908-001"):
        study_run = StudyRunId(study)
        training_coordinate = TrainingJobCoordinate(study_run, "trial-0001")
        jobs = frozenset(
            {
                TrainingJob(training_coordinate),
                EvaluationJob(
                    EvaluationJobCoordinate(study_run, "trial-0001", "quality"),
                    training_coordinate,
                ),
            }
        )
        return study_run, jobs, self.queue.admit_study_jobs(study_run, jobs, admitted_at=T0)

    def admit_existing_model_trial(self, study: str = "sr-20260908-002"):
        study_run = StudyRunId(study)
        jobs = frozenset(
            {
                EvaluationJob(
                    EvaluationJobCoordinate(study_run, "trial-0001", "quality"),
                    None,
                ),
                EvaluationJob(
                    EvaluationJobCoordinate(study_run, "trial-0001", "latency"),
                    None,
                ),
            }
        )
        return study_run, jobs, self.queue.admit_study_jobs(study_run, jobs, admitted_at=T0)


class SchemaAndInitializationTests(SQLiteQueueTestCase):
    def test_exact_tables_columns_indexes_version_and_pragmas(self) -> None:
        with self.queue._connection() as connection:
            self.assertEqual(1, connection.execute("PRAGMA user_version").fetchone()[0])
            self.assertEqual("wal", connection.execute("PRAGMA journal_mode").fetchone()[0])
            self.assertEqual(2, connection.execute("PRAGMA synchronous").fetchone()[0])
            self.assertEqual(1, connection.execute("PRAGMA foreign_keys").fetchone()[0])
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            self.assertEqual({"jobs", "attempts"}, tables)
            self.assertEqual(
                [
                    "job_id", "study_run_id", "trial_id", "kind", "stage", "status",
                    "dependency_job_id", "retry_not_before", "created_at", "updated_at",
                ],
                [row[1] for row in connection.execute("PRAGMA table_info(jobs)")],
            )
            self.assertEqual(
                [
                    "attempt_id", "job_id", "attempt_no", "run_id", "worker_id",
                    "acquire_token", "lease_token", "lease_until", "started_at",
                    "finished_at", "close_reason",
                ],
                [row[1] for row in connection.execute("PRAGMA table_info(attempts)")],
            )
            indexes = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
                )
            }
            self.assertEqual(
                {
                    "uq_jobs_training_coordinate",
                    "uq_jobs_evaluation_coordinate",
                    "idx_jobs_study_status",
                    "idx_jobs_schedulable",
                    "idx_jobs_dependency",
                    "uq_attempts_one_open_per_job",
                    "uq_attempts_one_open_per_worker",
                    "idx_attempts_open_lease",
                },
                indexes,
            )
            foreign_keys = list(connection.execute("PRAGMA foreign_key_list(attempts)"))
            self.assertEqual(("jobs", "job_id", "job_id", "RESTRICT"),
                             (foreign_keys[0][2], foreign_keys[0][3], foreign_keys[0][4], foreign_keys[0][6]))

    def test_fresh_initialization_and_reopen_preserve_state(self) -> None:
        study, _, admitted = self.admit_training_trial()
        reopened = SQLiteQueue(self.root)
        self.assertEqual(admitted, reopened.jobs_for_study_run(study))

    def test_unsupported_nonzero_user_version_is_refused(self) -> None:
        other = self.root / "other"
        database = queue_database_path(other)
        database.parent.mkdir(parents=True)
        with closing(sqlite3.connect(database)) as connection:
            with connection:
                connection.execute("PRAGMA user_version = 2")
        with self.assertRaisesRegex(RuntimeError, "unsupported Queue schema version"):
            SQLiteQueue(other)

    def test_unversioned_existing_schema_is_refused(self) -> None:
        other = self.root / "other"
        database = queue_database_path(other)
        database.parent.mkdir(parents=True)
        with closing(sqlite3.connect(database)) as connection:
            with connection:
                connection.execute("CREATE TABLE alien (value TEXT)")
        with self.assertRaisesRegex(RuntimeError, "unversioned Queue schema"):
            SQLiteQueue(other)

    def test_ddl_checks_and_uniqueness_are_enforced_by_sqlite(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute("PRAGMA foreign_keys = ON")
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                    """INSERT INTO jobs
                    (study_run_id, trial_id, kind, stage, status, dependency_job_id,
                     retry_not_before, created_at, updated_at)
                    VALUES ('sr', 'trial', 'training', 'bad', 'ready', NULL, NULL, ?, ?)""",
                    (T0, T0),
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                    """INSERT INTO jobs
                    (study_run_id, trial_id, kind, stage, status, dependency_job_id,
                     retry_not_before, created_at, updated_at)
                    VALUES ('sr', 'trial', 'training', NULL, 'retry_wait', NULL, NULL, ?, ?)""",
                    (T0, T0),
                )


class TimestampAndAdmissionTests(SQLiteQueueTestCase):
    def test_malformed_timestamps_are_strictly_rejected(self) -> None:
        study = StudyRunId("sr-20260908-010")
        job = EvaluationJob(EvaluationJobCoordinate(study, "trial-0001", "eval"), None)
        malformed = (
            "2026-09-08T10:00:00Z",
            "2026-09-08T10:00:00.00000Z",
            "2026-09-08T10:00:00.000000+00:00",
            "2026-09-08 10:00:00.000000Z",
            "2026-02-30T10:00:00.000000Z",
            "2026-09-08T25:00:00.000000Z",
            "2026-09-08T10:00:00.000000z",
        )
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.queue.admit_study_jobs(study, frozenset({job}), admitted_at=value)
        self.assertEqual((), self.queue.jobs_for_study_run(study))

    def test_training_and_existing_model_admission_round_trip_losslessly(self) -> None:
        training_study, training_jobs, queued_training = self.admit_training_trial()
        existing_study, existing_jobs, queued_existing = self.admit_existing_model_trial()

        self.assertEqual(training_jobs, frozenset(job.logical for job in queued_training))
        training = next(job for job in queued_training if isinstance(job.logical, TrainingJob))
        evaluation = next(job for job in queued_training if isinstance(job.logical, EvaluationJob))
        self.assertIs(QueueJobStatus.READY, training.status)
        self.assertIs(QueueJobStatus.BLOCKED, evaluation.status)
        self.assertEqual(training.job_id, evaluation.dependency_job_id)
        self.assertEqual(existing_jobs, frozenset(job.logical for job in queued_existing))
        self.assertTrue(all(job.status is QueueJobStatus.READY for job in queued_existing))
        self.assertTrue(all(job.dependency_job_id is None for job in queued_existing))
        self.assertEqual(queued_training, self.queue.jobs_for_study_run(training_study))
        self.assertEqual(queued_existing, self.queue.jobs_for_study_run(existing_study))

    def test_agreed_replay_preserves_operational_state_and_creation_time(self) -> None:
        study, jobs, admitted = self.admit_existing_model_trial()
        first = admitted[0]
        deferred = self.queue.defer_ready_job(first.job_id, retry_not_before=T2, at=T1)

        replayed = self.queue.admit_study_jobs(study, jobs, admitted_at=T3)

        same = next(job for job in replayed if job.job_id == first.job_id)
        self.assertEqual(deferred, same)
        self.assertEqual(T0, same.created_at)

    def test_conflicting_replay_is_all_or_nothing(self) -> None:
        study, existing_jobs, before = self.admit_existing_model_trial()
        training_coordinate = TrainingJobCoordinate(study, "trial-0001")
        conflicting = frozenset(
            {
                TrainingJob(training_coordinate),
                EvaluationJob(
                    EvaluationJobCoordinate(study, "trial-0001", "quality"),
                    training_coordinate,
                ),
                next(
                    job
                    for job in existing_jobs
                    if job.coordinate.stage == "latency"
                ),
            }
        )
        with self.assertRaises(ValueError):
            self.queue.admit_study_jobs(study, conflicting, admitted_at=T1)
        self.assertEqual(before, self.queue.jobs_for_study_run(study))

    def test_cross_trial_or_missing_training_dependency_is_refused_without_writes(self) -> None:
        study = StudyRunId("sr-20260908-011")
        training = TrainingJobCoordinate(study, "trial-0001")
        invalid = EvaluationJob(
            EvaluationJobCoordinate(study, "trial-0002", "eval"), training
        )
        with self.assertRaises(ValueError):
            self.queue.admit_study_jobs(study, frozenset({invalid}), admitted_at=T0)
        self.assertEqual((), self.queue.jobs_for_study_run(study))


class SchedulingAndAttemptTests(SQLiteQueueTestCase):
    def test_capability_filter_and_due_retry_promotion(self) -> None:
        _, _, training_jobs = self.admit_training_trial()
        _, _, evaluation_jobs = self.admit_existing_model_trial()
        training = next(job for job in training_jobs if isinstance(job.logical, TrainingJob))
        evaluation = evaluation_jobs[0]
        self.queue.defer_ready_job(evaluation.job_id, retry_not_before=T2, at=T1)

        self.assertEqual(
            training.job_id,
            self.queue.select_ready_job(
                accepts_training=True, accepts_evaluation=False, as_of=T1
            ).job_id,
        )
        selected = self.queue.select_ready_job(
            accepts_training=False, accepts_evaluation=True, as_of=T2
        )
        self.assertEqual(evaluation.job_id, selected.job_id)
        self.assertIs(QueueJobStatus.READY, self.queue.job_by_id(evaluation.job_id).status)
        with self.assertRaises(ValueError):
            self.queue.select_ready_job(
                accepts_training=False, accepts_evaluation=False, as_of=T2
            )

    def test_defer_and_fail_only_ready_without_attempts(self) -> None:
        _, _, jobs = self.admit_existing_model_trial()
        deferred = self.queue.defer_ready_job(jobs[0].job_id, retry_not_before=T2, at=T1)
        failed = self.queue.fail_ready_job(jobs[1].job_id, at=T1)
        self.assertIs(QueueJobStatus.RETRY_WAIT, deferred.status)
        self.assertEqual(T2, deferred.retry_not_before)
        self.assertIs(QueueJobStatus.FAILED, failed.status)
        self.assertEqual((), self.queue.attempts_for_job(deferred.job_id))
        with self.assertRaises(ValueError):
            self.queue.fail_ready_job(deferred.job_id, at=T2)

    def test_activation_is_atomic_and_sets_exact_one_hour_lease(self) -> None:
        _, _, jobs = self.admit_training_trial()
        training = next(job for job in jobs if isinstance(job.logical, TrainingJob))
        attempt = self.queue.activate_attempt(
            training.job_id,
            TrainingRunId("tr-20260908-001"),
            "worker-1",
            "acquire-1",
            "lease-1",
            activated_at=T0,
        )
        self.assertEqual(1, attempt.attempt_no)
        self.assertEqual(T2, attempt.lease_until)
        self.assertIs(QueueJobStatus.ACTIVE, self.queue.job_by_id(training.job_id).status)
        self.assertEqual(attempt, self.queue.open_attempt_for_job(training.job_id))
        self.assertEqual(attempt, self.queue.open_attempt_for_worker("worker-1"))

    def test_activation_rejects_run_kind_and_all_identity_conflicts(self) -> None:
        _, _, training_jobs = self.admit_training_trial()
        _, _, evaluation_jobs = self.admit_existing_model_trial()
        training = next(job for job in training_jobs if isinstance(job.logical, TrainingJob))
        first_eval, second_eval = evaluation_jobs
        with self.assertRaises(ValueError):
            self.queue.activate_attempt(
                training.job_id, EvaluationRunId("ev-20260908-001"), "worker-x",
                "acquire-x", "lease-x", activated_at=T0,
            )
        first = self.queue.activate_attempt(
            first_eval.job_id, EvaluationRunId("ev-20260908-001"), "worker-1",
            "acquire-1", "lease-1", activated_at=T0,
        )
        with self.assertRaises(ValueError):
            self.queue.activate_attempt(
                first_eval.job_id, EvaluationRunId("ev-20260908-002"), "worker-2",
                "acquire-2", "lease-2", activated_at=T0,
            )
        for kwargs in (
            dict(run_id=EvaluationRunId("ev-20260908-002"), worker_id="worker-1", acquire_token="acquire-2", lease_token="lease-2"),
            dict(run_id=EvaluationRunId("ev-20260908-001"), worker_id="worker-2", acquire_token="acquire-2", lease_token="lease-2"),
            dict(run_id=EvaluationRunId("ev-20260908-002"), worker_id="worker-2", acquire_token="acquire-1", lease_token="lease-2"),
            dict(run_id=EvaluationRunId("ev-20260908-002"), worker_id="worker-2", acquire_token="acquire-2", lease_token="lease-1"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.queue.activate_attempt(second_eval.job_id, activated_at=T1, **kwargs)
        self.assertEqual((), self.queue.attempts_for_job(second_eval.job_id))
        self.assertEqual(first, self.queue.attempt_by_acquire_token("acquire-1"))

    def test_acquire_token_lookup_is_historical_after_close(self) -> None:
        _, _, jobs = self.admit_existing_model_trial()
        attempt = self.queue.activate_attempt(
            jobs[0].job_id, EvaluationRunId("ev-20260908-003"), "worker-1",
            "acquire-history", "lease-history", activated_at=T0,
        )
        self.queue.close_attempt(
            attempt.attempt_id, target_status=QueueJobStatus.FAILED, finished_at=T1
        )
        historical = self.queue.attempt_by_acquire_token("acquire-history")
        self.assertEqual(T1, historical.finished_at)
        self.assertIsNone(self.queue.open_attempt_for_job(jobs[0].job_id))

    def test_authorization_heartbeat_and_expiry_boundaries(self) -> None:
        _, _, jobs = self.admit_existing_model_trial()
        attempt = self.queue.activate_attempt(
            jobs[0].job_id, EvaluationRunId("ev-20260908-004"), "worker-1",
            "acquire-1", "lease-1", activated_at=T0,
        )
        self.assertEqual(
            attempt,
            self.queue.authorized_open_attempt(attempt.attempt_id, "lease-1", as_of=T1),
        )
        self.assertIsNone(
            self.queue.authorized_open_attempt(attempt.attempt_id, "wrong", as_of=T1)
        )
        self.assertIsNone(
            self.queue.authorized_open_attempt(attempt.attempt_id, "lease-1", as_of=T2)
        )
        self.assertEqual((attempt,), self.queue.expired_open_attempts(as_of=T2))
        with self.assertRaises(ValueError):
            self.queue.heartbeat_attempt(attempt.attempt_id, "lease-1", accepted_at=T2)

        other = self.queue.activate_attempt(
            jobs[1].job_id, EvaluationRunId("ev-20260908-005"), "worker-2",
            "acquire-2", "lease-2", activated_at=T0,
        )
        extended = self.queue.heartbeat_attempt(other.attempt_id, "lease-2", accepted_at=T1)
        self.assertEqual("2026-09-08T11:15:00.000000Z", extended.lease_until)

    def test_attempt_and_job_changes_rollback_together_on_sql_error(self) -> None:
        _, _, jobs = self.admit_existing_model_trial()
        job = jobs[0]
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute(
                    """CREATE TRIGGER reject_active BEFORE UPDATE OF status ON jobs
                       WHEN NEW.status = 'active'
                       BEGIN SELECT RAISE(ABORT, 'reject active'); END"""
                )
        with self.assertRaises(sqlite3.IntegrityError):
            self.queue.activate_attempt(
                job.job_id, EvaluationRunId("ev-20260908-006"), "worker-1",
                "acquire-1", "lease-1", activated_at=T0,
            )
        self.assertEqual((), self.queue.attempts_for_job(job.job_id))
        self.assertIs(QueueJobStatus.READY, self.queue.job_by_id(job.job_id).status)

    def test_concurrent_activation_allows_exactly_one_winner(self) -> None:
        _, _, jobs = self.admit_existing_model_trial()
        job = jobs[0]
        barrier = Barrier(3)
        successes = []
        errors = []

        def activate(sequence: int) -> None:
            contender = SQLiteQueue(self.root)
            barrier.wait()
            try:
                successes.append(
                    contender.activate_attempt(
                        job.job_id,
                        EvaluationRunId(f"ev-20260908-{sequence:03d}"),
                        f"worker-{sequence}",
                        f"acquire-{sequence}",
                        f"lease-{sequence}",
                        activated_at=T0,
                    )
                )
            except BaseException as error:
                errors.append(error)

        threads = [Thread(target=activate, args=(index,), daemon=True) for index in (20, 21)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(1, len(successes))
        self.assertEqual(1, len(errors))
        self.assertIsInstance(errors[0], ValueError)
        self.assertEqual(1, len(self.queue.attempts_for_job(job.job_id)))


class ClosureCancellationAndRepairTests(SQLiteQueueTestCase):
    def test_close_supports_retry_failure_and_success_with_monotonic_attempts(self) -> None:
        _, _, jobs = self.admit_existing_model_trial()
        retry_job, fail_job = jobs
        first = self.queue.activate_attempt(
            retry_job.job_id, EvaluationRunId("ev-20260908-030"), "worker-1",
            "acquire-1", "lease-1", activated_at=T0,
        )
        retried_job, closed = self.queue.close_attempt(
            first.attempt_id,
            target_status=QueueJobStatus.RETRY_WAIT,
            finished_at=T1,
            retry_not_before=T2,
            close_reason="partial",
        )
        self.assertIs(QueueJobStatus.RETRY_WAIT, retried_job.status)
        self.assertEqual("partial", closed.close_reason)
        self.queue.select_ready_job(
            accepts_training=False, accepts_evaluation=True, as_of=T2
        )
        second = self.queue.activate_attempt(
            retry_job.job_id, EvaluationRunId("ev-20260908-031"), "worker-1",
            "acquire-2", "lease-2", activated_at=T2,
        )
        self.assertEqual(2, second.attempt_no)
        satisfied, _ = self.queue.close_attempt(
            second.attempt_id, target_status=QueueJobStatus.SATISFIED, finished_at=T3
        )
        self.assertIs(QueueJobStatus.SATISFIED, satisfied.status)

        failed_attempt = self.queue.activate_attempt(
            fail_job.job_id, EvaluationRunId("ev-20260908-032"), "worker-2",
            "acquire-3", "lease-3", activated_at=T0,
        )
        failed, _ = self.queue.close_attempt(
            failed_attempt.attempt_id,
            target_status=QueueJobStatus.FAILED,
            finished_at=T1,
        )
        self.assertIs(QueueJobStatus.FAILED, failed.status)

    def test_training_satisfaction_releases_only_its_blocked_dependents(self) -> None:
        _, _, jobs = self.admit_training_trial()
        training = next(job for job in jobs if isinstance(job.logical, TrainingJob))
        evaluation = next(job for job in jobs if isinstance(job.logical, EvaluationJob))
        attempt = self.queue.activate_attempt(
            training.job_id, TrainingRunId("tr-20260908-040"), "worker-1",
            "acquire-1", "lease-1", activated_at=T0,
        )
        self.queue.close_attempt(
            attempt.attempt_id,
            target_status=QueueJobStatus.SATISFIED,
            finished_at=T1,
        )
        released = self.queue.job_by_id(evaluation.job_id)
        self.assertIs(QueueJobStatus.READY, released.status)
        self.assertEqual(T1, released.updated_at)

    def test_cancel_nonactive_and_active_jobs_and_idempotent_replay(self) -> None:
        _, _, jobs = self.admit_existing_model_trial()
        idle, active = jobs
        cancelled_idle = self.queue.cancel_job(idle.job_id, at=T1)
        self.assertIs(QueueJobStatus.CANCELLED, cancelled_idle.status)
        self.assertEqual(cancelled_idle, self.queue.cancel_job(idle.job_id, at=T2))

        attempt = self.queue.activate_attempt(
            active.job_id, EvaluationRunId("ev-20260908-050"), "worker-1",
            "acquire-1", "lease-1", activated_at=T0,
        )
        cancelled_active = self.queue.cancel_job(
            active.job_id, at=T1, close_reason="study_cancelled"
        )
        self.assertIs(QueueJobStatus.CANCELLED, cancelled_active.status)
        closed = self.queue.attempt_by_id(attempt.attempt_id)
        self.assertEqual(T1, closed.finished_at)
        self.assertEqual("study_cancelled", closed.close_reason)

    def test_repair_rejects_active_and_open_attempt_then_releases_dependents(self) -> None:
        _, _, jobs = self.admit_training_trial()
        training = next(job for job in jobs if isinstance(job.logical, TrainingJob))
        evaluation = next(job for job in jobs if isinstance(job.logical, EvaluationJob))
        with self.assertRaises(ValueError):
            self.queue.repair_job_state(
                training.job_id, status=QueueJobStatus.ACTIVE,
                retry_not_before=None, at=T1,
            )
        repaired = self.queue.repair_job_state(
            training.job_id, status=QueueJobStatus.SATISFIED,
            retry_not_before=None, at=T1,
        )
        self.assertIs(QueueJobStatus.SATISFIED, repaired.status)
        self.assertIs(QueueJobStatus.READY, self.queue.job_by_id(evaluation.job_id).status)

        attempt = self.queue.activate_attempt(
            evaluation.job_id, EvaluationRunId("ev-20260908-060"), "worker-1",
            "acquire-1", "lease-1", activated_at=T1,
        )
        with self.assertRaises(ValueError):
            self.queue.repair_job_state(
                evaluation.job_id, status=QueueJobStatus.FAILED,
                retry_not_before=None, at=T2,
            )
        self.assertEqual(attempt, self.queue.open_attempt_for_job(evaluation.job_id))

    def test_restart_durability_retains_attempt_and_lease_state(self) -> None:
        study, _, jobs = self.admit_existing_model_trial()
        attempt = self.queue.activate_attempt(
            jobs[0].job_id, EvaluationRunId("ev-20260908-070"), "worker-1",
            "acquire-1", "lease-1", activated_at=T0,
        )
        reopened = SQLiteQueue(self.root)
        self.assertEqual(attempt, reopened.attempt_by_id(attempt.attempt_id))
        self.assertEqual(
            self.queue.jobs_for_study_run(study), reopened.jobs_for_study_run(study)
        )

    def test_adapter_implements_every_queue_port_operation_and_is_not_exported(self) -> None:
        protocol_methods = {
            name
            for name, value in QueuePort.__dict__.items()
            if inspect.isfunction(value) and not name.startswith("__")
        }
        self.assertTrue(protocol_methods.issubset(vars(SQLiteQueue)))
        import mldb.src.orchestration as package
        self.assertNotIn("SQLiteQueue", vars(package))


if __name__ == "__main__":
    unittest.main()
