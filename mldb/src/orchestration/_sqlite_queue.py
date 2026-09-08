"""Implementation-private SQLite adapter for the v1 operational Queue."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sqlite3

from ..common.ids import EvaluationRunId, StudyRunId, TrainingRunId
from .jobs import (
    EvaluationJob,
    EvaluationJobCoordinate,
    StudyJob,
    TrainingJob,
    TrainingJobCoordinate,
)
from .queue import QueueAttempt, QueueJob, QueueJobStatus, queue_database_path


_SCHEMA_VERSION = 1
_TIMESTAMP_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z")
_TRAINING_RUN_RE = re.compile(r"\Atr-\d{8}-\d{3}\Z")
_EVALUATION_RUN_RE = re.compile(r"\Aev-\d{8}-\d{3}\Z")

_DDL = (
    """CREATE TABLE jobs (
    job_id INTEGER PRIMARY KEY,
    study_run_id TEXT NOT NULL,
    trial_id TEXT NOT NULL,
    kind TEXT NOT NULL
        CHECK (kind IN ('training', 'evaluation')),
    stage TEXT,
    status TEXT NOT NULL
        CHECK (status IN (
            'blocked',
            'ready',
            'active',
            'retry_wait',
            'satisfied',
            'failed',
            'cancelled'
        )),
    dependency_job_id INTEGER,
    retry_not_before TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY (dependency_job_id)
        REFERENCES jobs(job_id)
        ON DELETE RESTRICT,

    CHECK (
        (kind = 'training' AND stage IS NULL AND dependency_job_id IS NULL)
        OR
        (kind = 'evaluation' AND stage IS NOT NULL)
    ),

    CHECK (
        (status = 'retry_wait' AND retry_not_before IS NOT NULL)
        OR
        (status <> 'retry_wait' AND retry_not_before IS NULL)
    )
)""",
    """CREATE UNIQUE INDEX uq_jobs_training_coordinate
    ON jobs(study_run_id, trial_id)
    WHERE kind = 'training'""",
    """CREATE UNIQUE INDEX uq_jobs_evaluation_coordinate
    ON jobs(study_run_id, trial_id, stage)
    WHERE kind = 'evaluation'""",
    """CREATE INDEX idx_jobs_study_status
    ON jobs(study_run_id, status)""",
    """CREATE INDEX idx_jobs_schedulable
    ON jobs(status, retry_not_before, job_id)""",
    """CREATE INDEX idx_jobs_dependency
    ON jobs(dependency_job_id, status)
    WHERE dependency_job_id IS NOT NULL""",
    """CREATE TABLE attempts (
    attempt_id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL,
    attempt_no INTEGER NOT NULL
        CHECK (attempt_no > 0),
    run_id TEXT NOT NULL UNIQUE,
    worker_id TEXT NOT NULL,
    acquire_token TEXT NOT NULL UNIQUE,
    lease_token TEXT NOT NULL UNIQUE,
    lease_until TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    close_reason TEXT,

    FOREIGN KEY (job_id)
        REFERENCES jobs(job_id)
        ON DELETE RESTRICT,

    UNIQUE (job_id, attempt_no)
)""",
    """CREATE UNIQUE INDEX uq_attempts_one_open_per_job
    ON attempts(job_id)
    WHERE finished_at IS NULL""",
    """CREATE UNIQUE INDEX uq_attempts_one_open_per_worker
    ON attempts(worker_id)
    WHERE finished_at IS NULL""",
    """CREATE INDEX idx_attempts_open_lease
    ON attempts(lease_until)
    WHERE finished_at IS NULL""",
)

_JOB_SELECT = """SELECT
    j.job_id, j.study_run_id, j.trial_id, j.kind, j.stage, j.status,
    j.dependency_job_id, j.retry_not_before, j.created_at, j.updated_at,
    d.kind AS dependency_kind,
    d.study_run_id AS dependency_study_run_id,
    d.trial_id AS dependency_trial_id
FROM jobs AS j
LEFT JOIN jobs AS d ON d.job_id = j.dependency_job_id"""

_ATTEMPT_SELECT = """SELECT
    a.attempt_id, a.job_id, a.attempt_no, a.run_id, a.worker_id,
    a.acquire_token, a.lease_token, a.lease_until, a.started_at,
    a.finished_at, a.close_reason, j.kind AS job_kind
FROM attempts AS a
JOIN jobs AS j ON j.job_id = a.job_id"""


class SQLiteQueue:
    """Concrete QueuePort implementation backed by the repository-local v1 DB."""

    def __init__(self, repository_root: Path) -> None:
        self._database_path = queue_database_path(repository_root)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def admit_study_jobs(
        self,
        study_run_id: StudyRunId,
        jobs: frozenset[StudyJob],
        *,
        admitted_at: str,
    ) -> tuple[QueueJob, ...]:
        _require_timestamp(admitted_at, "admitted_at")
        study_text = _require_nonempty(str(study_run_id), "study_run_id")
        supplied = _normalize_admission(study_run_id, jobs)

        with self._transaction() as connection:
            existing_rows = self._job_rows_for_study(connection, study_text)
            existing = {_job_key_from_row(row): row for row in existing_rows}
            extras = set(existing) - set(supplied)
            if extras:
                raise ValueError("persisted Study jobs conflict with complete admission")

            training_ids: dict[str, int] = {}
            for key, logical in supplied.items():
                if key[0] != "training":
                    continue
                row = existing.get(key)
                if row is None:
                    cursor = connection.execute(
                        """INSERT INTO jobs (
                            study_run_id, trial_id, kind, stage, status,
                            dependency_job_id, retry_not_before, created_at, updated_at
                        ) VALUES (?, ?, 'training', NULL, 'ready', NULL, NULL, ?, ?)""",
                        (study_text, logical.coordinate.trial, admitted_at, admitted_at),
                    )
                    training_ids[logical.coordinate.trial] = _lastrowid(cursor)
                else:
                    training_ids[logical.coordinate.trial] = int(row["job_id"])

            for key, logical in supplied.items():
                if key[0] != "evaluation":
                    continue
                assert isinstance(logical, EvaluationJob)
                expected_dependency = (
                    training_ids[logical.coordinate.trial]
                    if logical.training_dependency is not None
                    else None
                )
                row = existing.get(key)
                if row is not None:
                    if row["dependency_job_id"] != expected_dependency:
                        raise ValueError("persisted Queue dependency conflicts with admission")
                    continue
                status = "blocked" if expected_dependency is not None else "ready"
                connection.execute(
                    """INSERT INTO jobs (
                        study_run_id, trial_id, kind, stage, status,
                        dependency_job_id, retry_not_before, created_at, updated_at
                    ) VALUES (?, ?, 'evaluation', ?, ?, ?, NULL, ?, ?)""",
                    (
                        study_text,
                        logical.coordinate.trial,
                        logical.coordinate.stage,
                        status,
                        expected_dependency,
                        admitted_at,
                        admitted_at,
                    ),
                )

            rows = self._job_rows_for_study(connection, study_text)
            if {_job_key_from_row(row) for row in rows} != set(supplied):
                raise RuntimeError("Queue admission did not preserve the complete job set")
            return tuple(_job_from_row(row) for row in rows)

    def jobs_for_study_run(self, study_run_id: StudyRunId) -> tuple[QueueJob, ...]:
        study_text = _require_nonempty(str(study_run_id), "study_run_id")
        with self._connection() as connection:
            return tuple(
                _job_from_row(row)
                for row in self._job_rows_for_study(connection, study_text)
            )

    def job_by_id(self, job_id: int) -> QueueJob | None:
        with self._connection() as connection:
            row = connection.execute(
                _JOB_SELECT + " WHERE j.job_id = ?", (job_id,)
            ).fetchone()
            return None if row is None else _job_from_row(row)

    def attempts_for_job(self, job_id: int) -> tuple[QueueAttempt, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                _ATTEMPT_SELECT
                + " WHERE a.job_id = ? ORDER BY a.attempt_no, a.attempt_id",
                (job_id,),
            ).fetchall()
            return tuple(_attempt_from_row(row) for row in rows)

    def attempt_by_id(self, attempt_id: int) -> QueueAttempt | None:
        with self._connection() as connection:
            row = connection.execute(
                _ATTEMPT_SELECT + " WHERE a.attempt_id = ?", (attempt_id,)
            ).fetchone()
            return None if row is None else _attempt_from_row(row)

    def open_attempt_for_job(self, job_id: int) -> QueueAttempt | None:
        with self._connection() as connection:
            row = connection.execute(
                _ATTEMPT_SELECT
                + " WHERE a.job_id = ? AND a.finished_at IS NULL",
                (job_id,),
            ).fetchone()
            return None if row is None else _attempt_from_row(row)

    def attempt_by_acquire_token(self, acquire_token: str) -> QueueAttempt | None:
        _require_nonempty(acquire_token, "acquire_token")
        with self._connection() as connection:
            row = connection.execute(
                _ATTEMPT_SELECT + " WHERE a.acquire_token = ?", (acquire_token,)
            ).fetchone()
            return None if row is None else _attempt_from_row(row)

    def open_attempt_for_worker(self, worker_id: str) -> QueueAttempt | None:
        _require_nonempty(worker_id, "worker_id")
        with self._connection() as connection:
            row = connection.execute(
                _ATTEMPT_SELECT
                + " WHERE a.worker_id = ? AND a.finished_at IS NULL",
                (worker_id,),
            ).fetchone()
            return None if row is None else _attempt_from_row(row)

    def authorized_open_attempt(
        self,
        attempt_id: int,
        lease_token: str,
        *,
        as_of: str,
    ) -> QueueAttempt | None:
        _require_nonempty(lease_token, "lease_token")
        _require_timestamp(as_of, "as_of")
        with self._connection() as connection:
            row = connection.execute(
                _ATTEMPT_SELECT
                + """ WHERE a.attempt_id = ? AND a.lease_token = ?
                      AND a.finished_at IS NULL AND a.lease_until > ?""",
                (attempt_id, lease_token, as_of),
            ).fetchone()
            return None if row is None else _attempt_from_row(row)

    def select_ready_job(
        self,
        *,
        accepts_training: bool,
        accepts_evaluation: bool,
        as_of: str,
    ) -> QueueJob | None:
        _require_timestamp(as_of, "as_of")
        if not accepts_training and not accepts_evaluation:
            raise ValueError("at least one job kind must be accepted")
        kinds = []
        if accepts_training:
            kinds.append("training")
        if accepts_evaluation:
            kinds.append("evaluation")
        placeholders = ", ".join("?" for _ in kinds)
        with self._transaction() as connection:
            connection.execute(
                """UPDATE jobs
                   SET status = 'ready', retry_not_before = NULL, updated_at = ?
                   WHERE status = 'retry_wait' AND retry_not_before <= ?""",
                (as_of, as_of),
            )
            row = connection.execute(
                _JOB_SELECT
                + f" WHERE j.status = 'ready' AND j.kind IN ({placeholders})"
                + " ORDER BY j.job_id LIMIT 1",
                tuple(kinds),
            ).fetchone()
            return None if row is None else _job_from_row(row)

    def defer_ready_job(
        self,
        job_id: int,
        *,
        retry_not_before: str,
        at: str,
    ) -> QueueJob:
        _require_timestamp(retry_not_before, "retry_not_before")
        _require_timestamp(at, "at")
        with self._transaction() as connection:
            changed = connection.execute(
                """UPDATE jobs
                   SET status = 'retry_wait', retry_not_before = ?, updated_at = ?
                   WHERE job_id = ? AND status = 'ready'""",
                (retry_not_before, at, job_id),
            ).rowcount
            if changed != 1:
                raise ValueError("only a ready job can be deferred")
            return self._required_job(connection, job_id)

    def fail_ready_job(self, job_id: int, *, at: str) -> QueueJob:
        _require_timestamp(at, "at")
        with self._transaction() as connection:
            changed = connection.execute(
                """UPDATE jobs
                   SET status = 'failed', retry_not_before = NULL, updated_at = ?
                   WHERE job_id = ? AND status = 'ready'""",
                (at, job_id),
            ).rowcount
            if changed != 1:
                raise ValueError("only a ready job can be failed without an attempt")
            return self._required_job(connection, job_id)

    def activate_attempt(
        self,
        job_id: int,
        run_id: TrainingRunId | EvaluationRunId,
        worker_id: str,
        acquire_token: str,
        lease_token: str,
        *,
        activated_at: str,
    ) -> QueueAttempt:
        activated = _require_timestamp(activated_at, "activated_at")
        worker_id = _require_nonempty(worker_id, "worker_id")
        acquire_token = _require_nonempty(acquire_token, "acquire_token")
        lease_token = _require_nonempty(lease_token, "lease_token")
        run_text = _require_nonempty(str(run_id), "run_id")
        lease_until = _format_timestamp(activated + timedelta(hours=1))

        with self._transaction() as connection:
            job = self._required_job(connection, job_id)
            if job.status is not QueueJobStatus.READY:
                raise ValueError("only a ready job can be activated")
            kind = _logical_kind(job.logical)
            _require_run_kind(run_text, kind)
            if connection.execute(
                "SELECT 1 FROM attempts WHERE job_id = ? AND finished_at IS NULL",
                (job_id,),
            ).fetchone():
                raise ValueError("job already has an open attempt")
            if connection.execute(
                "SELECT 1 FROM attempts WHERE worker_id = ? AND finished_at IS NULL",
                (worker_id,),
            ).fetchone():
                raise ValueError("worker already has an open attempt")
            for column, value in (
                ("run_id", run_text),
                ("acquire_token", acquire_token),
                ("lease_token", lease_token),
            ):
                if connection.execute(
                    f"SELECT 1 FROM attempts WHERE {column} = ?", (value,)
                ).fetchone():
                    raise ValueError(f"{column} has already been used")
            attempt_no = int(
                connection.execute(
                    "SELECT COALESCE(MAX(attempt_no), 0) + 1 FROM attempts WHERE job_id = ?",
                    (job_id,),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                """INSERT INTO attempts (
                    job_id, attempt_no, run_id, worker_id, acquire_token,
                    lease_token, lease_until, started_at, finished_at, close_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
                (
                    job_id,
                    attempt_no,
                    run_text,
                    worker_id,
                    acquire_token,
                    lease_token,
                    lease_until,
                    activated_at,
                ),
            )
            changed = connection.execute(
                """UPDATE jobs
                   SET status = 'active', retry_not_before = NULL, updated_at = ?
                   WHERE job_id = ? AND status = 'ready'""",
                (activated_at, job_id),
            ).rowcount
            if changed != 1:
                raise ValueError("job activation lost its ready state")
            return self._required_attempt(connection, _lastrowid(cursor))

    def heartbeat_attempt(
        self,
        attempt_id: int,
        lease_token: str,
        *,
        accepted_at: str,
    ) -> QueueAttempt:
        accepted = _require_timestamp(accepted_at, "accepted_at")
        lease_token = _require_nonempty(lease_token, "lease_token")
        lease_until = _format_timestamp(accepted + timedelta(hours=1))
        with self._transaction() as connection:
            attempt = self._required_attempt(connection, attempt_id)
            if attempt.finished_at is not None:
                raise ValueError("closed attempts cannot be heartbeated")
            if attempt.lease_token != lease_token:
                raise ValueError("lease token is not authorized")
            if attempt.lease_until <= accepted_at:
                raise ValueError("expired leases cannot be revived")
            if accepted_at < attempt.started_at:
                raise ValueError("heartbeat precedes attempt start")
            connection.execute(
                "UPDATE attempts SET lease_until = ? WHERE attempt_id = ?",
                (lease_until, attempt_id),
            )
            return self._required_attempt(connection, attempt_id)

    def expired_open_attempts(self, *, as_of: str) -> tuple[QueueAttempt, ...]:
        _require_timestamp(as_of, "as_of")
        with self._connection() as connection:
            rows = connection.execute(
                _ATTEMPT_SELECT
                + """ WHERE a.finished_at IS NULL AND a.lease_until <= ?
                      ORDER BY a.lease_until, a.attempt_id""",
                (as_of,),
            ).fetchall()
            return tuple(_attempt_from_row(row) for row in rows)

    def close_attempt(
        self,
        attempt_id: int,
        *,
        target_status: QueueJobStatus,
        finished_at: str,
        retry_not_before: str | None = None,
        close_reason: str | None = None,
    ) -> tuple[QueueJob, QueueAttempt]:
        _require_timestamp(finished_at, "finished_at")
        allowed = {
            QueueJobStatus.SATISFIED,
            QueueJobStatus.RETRY_WAIT,
            QueueJobStatus.FAILED,
        }
        if target_status not in allowed:
            raise ValueError("attempt target must be satisfied, retry_wait, or failed")
        if target_status is QueueJobStatus.RETRY_WAIT:
            if retry_not_before is None:
                raise ValueError("retry_wait requires retry_not_before")
            _require_timestamp(retry_not_before, "retry_not_before")
        elif retry_not_before is not None:
            raise ValueError("retry_not_before is only valid for retry_wait")

        with self._transaction() as connection:
            attempt = self._required_attempt(connection, attempt_id)
            if attempt.finished_at is not None:
                raise ValueError("attempt is already closed")
            if finished_at < attempt.started_at:
                raise ValueError("attempt finish precedes its start")
            job = self._required_job(connection, attempt.job_id)
            if job.status is not QueueJobStatus.ACTIVE:
                raise ValueError("open attempt does not own an active job")
            current = connection.execute(
                """SELECT attempt_id FROM attempts
                   WHERE job_id = ? AND finished_at IS NULL""",
                (attempt.job_id,),
            ).fetchall()
            if [int(row[0]) for row in current] != [attempt_id]:
                raise ValueError("attempt is not the current open attempt")
            connection.execute(
                """UPDATE attempts SET finished_at = ?, close_reason = ?
                   WHERE attempt_id = ? AND finished_at IS NULL""",
                (finished_at, close_reason, attempt_id),
            )
            connection.execute(
                """UPDATE jobs SET status = ?, retry_not_before = ?, updated_at = ?
                   WHERE job_id = ? AND status = 'active'""",
                (target_status.value, retry_not_before, finished_at, attempt.job_id),
            )
            if (
                target_status is QueueJobStatus.SATISFIED
                and isinstance(job.logical, TrainingJob)
            ):
                self._release_dependents(connection, attempt.job_id, finished_at)
            return (
                self._required_job(connection, attempt.job_id),
                self._required_attempt(connection, attempt_id),
            )

    def cancel_job(
        self,
        job_id: int,
        *,
        at: str,
        close_reason: str | None = None,
    ) -> QueueJob:
        _require_timestamp(at, "at")
        with self._transaction() as connection:
            job = self._required_job(connection, job_id)
            open_rows = connection.execute(
                _ATTEMPT_SELECT
                + " WHERE a.job_id = ? AND a.finished_at IS NULL",
                (job_id,),
            ).fetchall()
            if job.status is QueueJobStatus.CANCELLED:
                if open_rows:
                    raise RuntimeError("cancelled job retains an open attempt")
                return job
            if job.status in {QueueJobStatus.SATISFIED, QueueJobStatus.FAILED}:
                raise ValueError("terminal satisfied/failed jobs cannot be cancelled")
            if job.status is QueueJobStatus.ACTIVE:
                if len(open_rows) != 1:
                    raise RuntimeError("active job must have exactly one open attempt")
                attempt = _attempt_from_row(open_rows[0])
                if at < attempt.started_at:
                    raise ValueError("cancellation precedes attempt start")
                connection.execute(
                    """UPDATE attempts SET finished_at = ?, close_reason = ?
                       WHERE attempt_id = ? AND finished_at IS NULL""",
                    (at, close_reason, attempt.attempt_id),
                )
            elif open_rows:
                raise RuntimeError("non-active job retains an open attempt")
            connection.execute(
                """UPDATE jobs
                   SET status = 'cancelled', retry_not_before = NULL, updated_at = ?
                   WHERE job_id = ?""",
                (at, job_id),
            )
            return self._required_job(connection, job_id)

    def repair_job_state(
        self,
        job_id: int,
        *,
        status: QueueJobStatus,
        retry_not_before: str | None,
        at: str,
    ) -> QueueJob:
        _require_timestamp(at, "at")
        if not isinstance(status, QueueJobStatus):
            raise ValueError("repair status is not a QueueJobStatus")
        if status is QueueJobStatus.ACTIVE:
            raise ValueError("repair cannot create active execution authority")
        if status is QueueJobStatus.RETRY_WAIT:
            if retry_not_before is None:
                raise ValueError("retry_wait requires retry_not_before")
            _require_timestamp(retry_not_before, "retry_not_before")
        elif retry_not_before is not None:
            raise ValueError("retry_not_before is only valid for retry_wait")

        with self._transaction() as connection:
            job = self._required_job(connection, job_id)
            if connection.execute(
                "SELECT 1 FROM attempts WHERE job_id = ? AND finished_at IS NULL",
                (job_id,),
            ).fetchone():
                raise ValueError("repair requires all attempts to be closed")
            connection.execute(
                """UPDATE jobs SET status = ?, retry_not_before = ?, updated_at = ?
                   WHERE job_id = ?""",
                (status.value, retry_not_before, at, job_id),
            )
            if status is QueueJobStatus.SATISFIED and isinstance(job.logical, TrainingJob):
                self._release_dependents(connection, job_id, at)
            return self._required_job(connection, job_id)

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                application_tables = {
                    str(row[0])
                    for row in connection.execute(
                        """SELECT name FROM sqlite_master
                           WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"""
                    )
                }
                if version not in (0, _SCHEMA_VERSION):
                    raise RuntimeError(f"unsupported Queue schema version: {version}")
                if version == 0:
                    if application_tables:
                        raise RuntimeError("unversioned Queue schema is not supported")
                    for statement in _DDL:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                elif application_tables != {"jobs", "attempts"}:
                    raise RuntimeError("Queue v1 schema does not contain exactly jobs/attempts")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self._database_path,
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0])
            if mode.lower() != "wal":
                raise RuntimeError("Queue database could not enable WAL mode")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            if int(connection.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
                raise RuntimeError("Queue database could not enable foreign keys")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version != _SCHEMA_VERSION:
                    raise RuntimeError(f"unsupported Queue schema version: {version}")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _job_rows_for_study(
        self, connection: sqlite3.Connection, study_run_id: str
    ) -> list[sqlite3.Row]:
        return connection.execute(
            _JOB_SELECT + " WHERE j.study_run_id = ? ORDER BY j.job_id",
            (study_run_id,),
        ).fetchall()

    def _required_job(self, connection: sqlite3.Connection, job_id: int) -> QueueJob:
        row = connection.execute(
            _JOB_SELECT + " WHERE j.job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown Queue job_id: {job_id}")
        return _job_from_row(row)

    def _required_attempt(
        self, connection: sqlite3.Connection, attempt_id: int
    ) -> QueueAttempt:
        row = connection.execute(
            _ATTEMPT_SELECT + " WHERE a.attempt_id = ?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown Queue attempt_id: {attempt_id}")
        return _attempt_from_row(row)

    @staticmethod
    def _release_dependents(
        connection: sqlite3.Connection, training_job_id: int, at: str
    ) -> None:
        connection.execute(
            """UPDATE jobs SET status = 'ready', updated_at = ?
               WHERE dependency_job_id = ? AND status = 'blocked'""",
            (at, training_job_id),
        )


def _normalize_admission(
    study_run_id: StudyRunId, jobs: frozenset[StudyJob]
) -> dict[tuple[str, str, str | None], StudyJob]:
    normalized: dict[tuple[str, str, str | None], StudyJob] = {}
    training_trials: set[str] = set()
    for logical in jobs:
        if isinstance(logical, TrainingJob):
            coordinate = logical.coordinate
            _require_coordinate_study(coordinate.study_run, study_run_id)
            trial = _require_nonempty(coordinate.trial, "trial")
            key = ("training", trial, None)
            training_trials.add(trial)
        elif isinstance(logical, EvaluationJob):
            coordinate = logical.coordinate
            _require_coordinate_study(coordinate.study_run, study_run_id)
            trial = _require_nonempty(coordinate.trial, "trial")
            stage = _require_nonempty(coordinate.stage, "stage")
            key = ("evaluation", trial, stage)
        else:
            raise TypeError("admission contains an unsupported StudyJob value")
        prior = normalized.get(key)
        if prior is not None and prior != logical:
            raise ValueError("admission contains conflicting logical coordinates")
        normalized[key] = logical

    for logical in normalized.values():
        if not isinstance(logical, EvaluationJob):
            continue
        trial = logical.coordinate.trial
        dependency = logical.training_dependency
        if dependency is None:
            if trial in training_trials:
                raise ValueError("evaluation omits its same-trial training dependency")
            continue
        _require_coordinate_study(dependency.study_run, study_run_id)
        if dependency.trial != trial:
            raise ValueError("evaluation dependency must be from the same trial")
        if ("training", trial, None) not in normalized:
            raise ValueError("evaluation dependency is absent from complete admission")
    return normalized


def _require_coordinate_study(actual: StudyRunId, expected: StudyRunId) -> None:
    if str(actual) != str(expected):
        raise ValueError("StudyJob belongs to a different Study Run")


def _job_key_from_row(row: sqlite3.Row) -> tuple[str, str, str | None]:
    return str(row["kind"]), str(row["trial_id"]), row["stage"]


def _job_from_row(row: sqlite3.Row) -> QueueJob:
    study = StudyRunId(_require_nonempty(str(row["study_run_id"]), "study_run_id"))
    trial = _require_nonempty(str(row["trial_id"]), "trial_id")
    kind = str(row["kind"])
    dependency_id = row["dependency_job_id"]
    if kind == "training":
        if row["stage"] is not None or dependency_id is not None:
            raise RuntimeError("invalid persisted training job projection")
        logical: StudyJob = TrainingJob(TrainingJobCoordinate(study, trial))
    elif kind == "evaluation":
        stage = row["stage"]
        if not isinstance(stage, str) or not stage:
            raise RuntimeError("invalid persisted evaluation stage")
        dependency = None
        if dependency_id is not None:
            if (
                row["dependency_kind"] != "training"
                or row["dependency_study_run_id"] != str(study)
                or row["dependency_trial_id"] != trial
            ):
                raise RuntimeError("invalid persisted Queue dependency")
            dependency = TrainingJobCoordinate(study, trial)
        logical = EvaluationJob(EvaluationJobCoordinate(study, trial, stage), dependency)
    else:
        raise RuntimeError(f"invalid persisted Queue job kind: {kind}")
    try:
        status = QueueJobStatus(str(row["status"]))
    except ValueError as error:
        raise RuntimeError("invalid persisted Queue job status") from error
    retry = row["retry_not_before"]
    if retry is not None:
        _require_timestamp(retry, "retry_not_before")
    if (status is QueueJobStatus.RETRY_WAIT) != (retry is not None):
        raise RuntimeError("invalid persisted Queue retry timing")
    created = str(row["created_at"])
    updated = str(row["updated_at"])
    _require_timestamp(created, "created_at")
    _require_timestamp(updated, "updated_at")
    return QueueJob(
        job_id=int(row["job_id"]),
        logical=logical,
        status=status,
        dependency_job_id=None if dependency_id is None else int(dependency_id),
        retry_not_before=retry,
        created_at=created,
        updated_at=updated,
    )


def _attempt_from_row(row: sqlite3.Row) -> QueueAttempt:
    kind = str(row["job_kind"])
    run_text = _require_nonempty(str(row["run_id"]), "run_id")
    _require_run_kind(run_text, kind)
    run_id: TrainingRunId | EvaluationRunId
    if kind == "training":
        run_id = TrainingRunId(run_text)
    elif kind == "evaluation":
        run_id = EvaluationRunId(run_text)
    else:
        raise RuntimeError("attempt belongs to invalid Queue job kind")
    attempt_no = int(row["attempt_no"])
    if attempt_no <= 0:
        raise RuntimeError("invalid persisted Queue attempt number")
    lease_until = str(row["lease_until"])
    started_at = str(row["started_at"])
    _require_timestamp(lease_until, "lease_until")
    _require_timestamp(started_at, "started_at")
    finished_at = row["finished_at"]
    if finished_at is not None:
        _require_timestamp(finished_at, "finished_at")
        if finished_at < started_at:
            raise RuntimeError("persisted attempt finishes before it starts")
    return QueueAttempt(
        attempt_id=int(row["attempt_id"]),
        job_id=int(row["job_id"]),
        attempt_no=attempt_no,
        run_id=run_id,
        worker_id=_require_nonempty(str(row["worker_id"]), "worker_id"),
        acquire_token=_require_nonempty(str(row["acquire_token"]), "acquire_token"),
        lease_token=_require_nonempty(str(row["lease_token"]), "lease_token"),
        lease_until=lease_until,
        started_at=started_at,
        finished_at=finished_at,
        close_reason=row["close_reason"],
    )


def _logical_kind(logical: StudyJob) -> str:
    if isinstance(logical, TrainingJob):
        return "training"
    if isinstance(logical, EvaluationJob):
        return "evaluation"
    raise TypeError("unsupported StudyJob value")


def _require_run_kind(run_id: str, kind: str) -> None:
    pattern = _TRAINING_RUN_RE if kind == "training" else _EVALUATION_RUN_RE
    if pattern.fullmatch(run_id) is None:
        raise ValueError(f"run_id does not match {kind} Queue job kind")


def _require_nonempty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_timestamp(value: str, field: str) -> datetime:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must use UTC RFC3339 with six fractional digits")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as error:
        raise ValueError(f"{field} is not a valid UTC timestamp") from error
    return parsed.replace(tzinfo=timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _lastrowid(cursor: sqlite3.Cursor) -> int:
    if cursor.lastrowid is None:
        raise RuntimeError("SQLite did not return an inserted row identity")
    return int(cursor.lastrowid)
