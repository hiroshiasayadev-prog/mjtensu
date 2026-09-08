from __future__ import annotations

from datetime import date
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from mldb.src.common.ids import StudyId, StudyRunId
from mldb.src.orchestration import acquire as acquire_subject
from mldb.src.orchestration import outcome as outcome_subject
from mldb.src.orchestration import study_cancellation as subject
from mldb.src.orchestration.jobs import TrainingJob, TrainingJobCoordinate
from mldb.src.orchestration.queue import QueueJob, QueueJobStatus
from mldb.src.orchestration.worker_api import AcquireRejection, AcquireWorkRequest
from mldb.src.study.run import (
    StudyRun,
    StudyRunEvaluationSummary,
    StudyRunExecution,
    StudyRunPlan,
    StudyRunStatus,
    StudyRunSummary,
    StudyRunTrainingSummary,
)


STUDY_RUN_ID = StudyRunId("sr-20260908-001")
STUDY_ID = StudyId("study.test")
TRIAL_ID = "trial-0001"
FINISHED_AT = object()
QUEUE_AT = "2026-09-08T09:30:00.000000Z"
AS_OF = "2026-09-08T09:31:00.000000Z"
PLAN = StudyRunPlan(
    path="plan.jsonl",
    sha256="a" * 64,
    bytes=123,
    trials=2,
    evaluation_jobs=3,
)
SUMMARY = StudyRunSummary(
    training=StudyRunTrainingSummary(completed=1, failed=1, cancelled=0),
    evaluation=StudyRunEvaluationSummary(
        completed=1,
        completed_partial=1,
        failed=1,
        cancelled=0,
        blocked=0,
    ),
)


def _study(
    status: StudyRunStatus = StudyRunStatus.RUNNING,
    *,
    plan: StudyRunPlan | None = PLAN,
    summary: StudyRunSummary | None = SUMMARY,
) -> StudyRun:
    return StudyRun(
        schema="mjtensu.mldb/study-run/v1",
        id=STUDY_RUN_ID,
        status=status,
        study=STUDY_ID,
        execution=StudyRunExecution(
            started_at="2026-09-08T09:00:00Z",
            finished_at=None if status is StudyRunStatus.RUNNING else "already-finished",
        ),
        plan=plan,
        summary=summary,
    )


def _queue_job(job_id: int, status: QueueJobStatus) -> QueueJob:
    return QueueJob(
        job_id=job_id,
        logical=TrainingJob(TrainingJobCoordinate(STUDY_RUN_ID, TRIAL_ID)),
        status=status,
        dependency_job_id=None,
        retry_not_before=(
            "2026-09-08T10:00:00.000000Z"
            if status is QueueJobStatus.RETRY_WAIT
            else None
        ),
        created_at="2026-09-08T09:00:00.000000Z",
        updated_at="2026-09-08T09:00:00.000000Z",
    )


class _Queue:
    def __init__(self, jobs: tuple[QueueJob, ...] = ()) -> None:
        self.jobs = jobs
        self.calls: list[object] = []
        self.on_inspect = None
        self.on_cancel = None

    def jobs_for_study_run(self, study_run_id: StudyRunId) -> tuple[QueueJob, ...]:
        self.calls.append(("inspect", study_run_id))
        if self.on_inspect is not None:
            self.on_inspect()
        return self.jobs

    def cancel_job(self, job_id: int, *, at: str, close_reason: str | None = None):
        self.calls.append(("cancel", job_id, at, close_reason))
        if self.on_cancel is not None:
            return self.on_cancel(job_id)
        return next((job for job in self.jobs if job.job_id == job_id), None)


def test_running_persists_exact_cancelled_before_queue_and_applies_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _study()
    jobs = tuple(
        _queue_job(index, status)
        for index, status in enumerate(QueueJobStatus, start=1)
    )
    queue = _Queue(jobs)
    events: list[object] = []
    persisted: list[StudyRun] = []

    monkeypatch.setattr(subject, "read_study_run", lambda *args: events.append("read") or current)

    def persist(next_run: StudyRun, *args) -> None:
        events.append("persist")
        persisted.append(next_run)

    monkeypatch.setattr(subject, "persist_study_run_transition", persist)
    queue.on_inspect = lambda: events.append("inspect")
    queue.on_cancel = lambda job_id: events.append(("cancel", job_id))

    result = subject.request_study_run_cancellation(
        STUDY_RUN_ID, FINISHED_AT, QUEUE_AT, object(), object(), queue
    )

    assert result == "accepted"
    assert events[:3] == ["read", "persist", "inspect"]
    assert [event for event in events if isinstance(event, tuple)] == [
        ("cancel", 1),
        ("cancel", 2),
        ("cancel", 4),
    ]
    assert len(persisted) == 1
    cancelled = persisted[0]
    assert cancelled.schema == current.schema
    assert cancelled.id == current.id
    assert cancelled.study == current.study
    assert cancelled.execution.started_at == current.execution.started_at
    assert cancelled.execution.finished_at is FINISHED_AT
    assert cancelled.plan is current.plan
    assert cancelled.status is StudyRunStatus.CANCELLED
    assert cancelled.summary is None


def test_running_with_no_plan_preserves_none_and_does_not_touch_filesystem_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _study(plan=None)
    persisted: list[StudyRun] = []
    queue = _Queue()

    class NoDirectFilesystemAccess:
        def __getattribute__(self, name: str):
            raise AssertionError(f"unexpected direct filesystem access: {name}")

    monkeypatch.setattr(subject, "read_study_run", lambda *args: current)
    monkeypatch.setattr(
        subject,
        "persist_study_run_transition",
        lambda next_run, *args: persisted.append(next_run),
    )

    assert subject.request_study_run_cancellation(
        STUDY_RUN_ID, FINISHED_AT, QUEUE_AT, object(), NoDirectFilesystemAccess(), queue
    ) == "accepted"
    assert persisted[0].plan is None
    assert persisted[0].summary is None


def test_already_cancelled_repairs_lagging_non_active_queue_without_repersist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = tuple(
        _queue_job(index, status)
        for index, status in enumerate(QueueJobStatus, start=1)
    )
    queue = _Queue(jobs)
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study(StudyRunStatus.CANCELLED))
    monkeypatch.setattr(
        subject,
        "persist_study_run_transition",
        lambda *args: pytest.fail("terminal Study Run must not be re-persisted"),
    )

    result = subject.request_study_run_cancellation(
        STUDY_RUN_ID, FINISHED_AT, QUEUE_AT, object(), object(), queue
    )

    assert result == "already_terminal"
    assert queue.calls == [
        ("inspect", STUDY_RUN_ID),
        ("cancel", 1, QUEUE_AT, None),
        ("cancel", 2, QUEUE_AT, None),
        ("cancel", 4, QUEUE_AT, None),
    ]


@pytest.mark.parametrize(
    "status",
    [
        StudyRunStatus.COMPLETED,
        StudyRunStatus.COMPLETED_WITH_FAILURES,
        StudyRunStatus.FAILED,
    ],
)
def test_other_terminal_returns_without_study_or_queue_mutation(
    monkeypatch: pytest.MonkeyPatch,
    status: StudyRunStatus,
) -> None:
    queue = _Queue()
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study(status))
    monkeypatch.setattr(
        subject,
        "persist_study_run_transition",
        lambda *args: pytest.fail("terminal Study Run must not be mutated"),
    )
    assert subject.request_study_run_cancellation(
        STUDY_RUN_ID, FINISHED_AT, QUEUE_AT, object(), object(), queue
    ) == "already_terminal"
    assert queue.calls == []


def test_canonical_persistence_failure_never_touches_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = _Queue()
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study())

    def fail_persist(*args) -> None:
        raise OSError("canonical write failed")

    monkeypatch.setattr(subject, "persist_study_run_transition", fail_persist)

    with pytest.raises(OSError, match="canonical write failed"):
        subject.request_study_run_cancellation(
            STUDY_RUN_ID, FINISHED_AT, QUEUE_AT, object(), object(), queue
        )
    assert queue.calls == []


def test_queue_failure_keeps_canonical_cancelled_and_replay_repairs_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"study": _study()}
    queue = _Queue((_queue_job(2, QueueJobStatus.READY),))
    persist_calls: list[StudyRun] = []
    cancel_attempts = 0

    monkeypatch.setattr(subject, "read_study_run", lambda *args: state["study"])

    def persist(next_run: StudyRun, *args) -> None:
        persist_calls.append(next_run)
        state["study"] = next_run

    def flaky_cancel(job_id: int):
        nonlocal cancel_attempts
        cancel_attempts += 1
        if cancel_attempts == 1:
            raise OSError("queue unavailable")
        return queue.jobs[0]

    monkeypatch.setattr(subject, "persist_study_run_transition", persist)
    queue.on_cancel = flaky_cancel
    with pytest.raises(OSError, match="queue unavailable"):
        subject.request_study_run_cancellation(
            STUDY_RUN_ID, FINISHED_AT, QUEUE_AT, object(), object(), queue
        )

    assert state["study"].status is StudyRunStatus.CANCELLED
    assert len(persist_calls) == 1

    assert subject.request_study_run_cancellation(
        STUDY_RUN_ID, FINISHED_AT, QUEUE_AT, object(), object(), queue
    ) == "already_terminal"
    assert len(persist_calls) == 1
    assert cancel_attempts == 2


def _thread(target, errors: list[BaseException]) -> Thread:
    def run() -> None:
        try:
            target()
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=run, daemon=True)
    thread.start()
    return thread


class _AcquireCancellationQueue(_Queue):
    def __init__(self, cleanup_entered: Event, release_cleanup: Event, acquire_lookup: Event) -> None:
        super().__init__((_queue_job(2, QueueJobStatus.READY),))
        self.cleanup_entered = cleanup_entered
        self.release_cleanup = release_cleanup
        self.acquire_lookup = acquire_lookup

    def jobs_for_study_run(self, study_run_id: StudyRunId) -> tuple[QueueJob, ...]:
        self.calls.append(("inspect", study_run_id))
        self.cleanup_entered.set()
        assert self.release_cleanup.wait(2)
        return self.jobs

    def attempt_by_acquire_token(self, token: str):
        self.acquire_lookup.set()
        return None

    def open_attempt_for_worker(self, worker_id: str):
        return None

    def select_ready_job(self, **kwargs):
        return self.jobs[0]


def test_cancellation_serializes_fresh_acquire_and_terminal_authority_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"study": _study()}
    cleanup_entered = Event()
    release_cleanup = Event()
    acquire_lookup = Event()
    acquire_started = Event()
    acquire_parent_read = Event()
    queue = _AcquireCancellationQueue(cleanup_entered, release_cleanup, acquire_lookup)
    errors: list[BaseException] = []
    acquire_results: list[object] = []

    monkeypatch.setattr(subject, "read_study_run", lambda *args: state["study"])

    def persist(next_run: StudyRun, *args) -> None:
        state["study"] = next_run

    monkeypatch.setattr(subject, "persist_study_run_transition", persist)

    def acquire_read(*args):
        acquire_parent_read.set()
        return state["study"]

    monkeypatch.setattr(acquire_subject, "read_study_run", acquire_read)
    monkeypatch.setattr(
        acquire_subject,
        "read_study_plan",
        lambda *args: pytest.fail("terminal parent must prevent plan read"),
    )
    monkeypatch.setattr(
        acquire_subject,
        "dispatch_training_job",
        lambda *args, **kwargs: pytest.fail("fresh dispatch crossed cancellation"),
    )

    cancellation = _thread(
        lambda: subject.request_study_run_cancellation(
            STUDY_RUN_ID, FINISHED_AT, QUEUE_AT, object(), object(), queue
        ),
        errors,
    )
    assert cleanup_entered.wait(1)
    assert state["study"].status is StudyRunStatus.CANCELLED

    request = AcquireWorkRequest("worker-1", "acquire-after-cancel", frozenset({"training"}))

    def run_acquire() -> None:
        acquire_started.set()
        acquire_results.append(
            acquire_subject.handle_acquire_work(
                request,
                date(2026, 9, 8),
                object(),
                AS_OF,
                "fresh-lease",
                object(),
                object(),
                queue,
                object(),
                SimpleNamespace(),
            )
        )

    acquire = _thread(run_acquire, errors)
    assert acquire_started.wait(1)
    assert not acquire_lookup.wait(0.05)
    assert not acquire_parent_read.is_set()

    release_cleanup.set()
    cancellation.join(2)
    acquire.join(2)

    assert not cancellation.is_alive()
    assert not acquire.is_alive()
    assert errors == []
    assert acquire_lookup.is_set()
    assert acquire_parent_read.is_set()
    assert len(acquire_results) == 1
    assert isinstance(acquire_results[0], AcquireRejection)
    assert acquire_results[0].type == "parent_study_terminal"


class _OutcomeRaceQueue(_Queue):
    def __init__(self, cleanup_entered: Event, release_cleanup: Event, active: QueueJob) -> None:
        super().__init__((active,))
        self.cleanup_entered = cleanup_entered
        self.release_cleanup = release_cleanup
        self.outcome_cancelled = Event()

    def jobs_for_study_run(self, study_run_id: StudyRunId) -> tuple[QueueJob, ...]:
        self.calls.append(("inspect", study_run_id))
        self.cleanup_entered.set()
        assert self.release_cleanup.wait(2)
        return self.jobs
    def cancel_job(self, job_id: int, *, at: str, close_reason: str | None = None):
        self.calls.append(("cancel", job_id, at, close_reason))
        self.outcome_cancelled.set()
        return self.jobs[0]

    def attempts_for_job(self, job_id: int):
        raise AssertionError("cancelled Study must not consult retry attempts")

    def close_attempt(self, *args, **kwargs):
        raise AssertionError("durable cancelled Study must not establish RETRY_WAIT")


class _RetryPolicy:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def after_unsatisfied_attempt(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(retry_not_before="2026-09-08T10:00:00.000000Z")


def test_durable_cancelled_blocks_unsatisfied_retry_wait_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"study": _study()}
    cleanup_entered = Event()
    release_cleanup = Event()
    outcome_started = Event()
    outcome_read = Event()
    active = _queue_job(3, QueueJobStatus.ACTIVE)
    queue = _OutcomeRaceQueue(cleanup_entered, release_cleanup, active)
    policy = _RetryPolicy()
    errors: list[BaseException] = []

    monkeypatch.setattr(subject, "read_study_run", lambda *args: state["study"])

    def persist(next_run: StudyRun, *args) -> None:
        state["study"] = next_run

    monkeypatch.setattr(subject, "persist_study_run_transition", persist)

    def outcome_read_study(*args):
        outcome_read.set()
        return state["study"]

    monkeypatch.setattr(outcome_subject, "read_study_run", outcome_read_study)

    cancellation = _thread(
        lambda: subject.request_study_run_cancellation(
            STUDY_RUN_ID, FINISHED_AT, QUEUE_AT, object(), object(), queue
        ),
        errors,
    )
    assert cleanup_entered.wait(1)
    assert state["study"].status is StudyRunStatus.CANCELLED

    def run_outcome_disposition() -> None:
        outcome_started.set()
        outcome_subject._dispose_unsatisfied(
            active,
            99,
            "failed",
            AS_OF,
            object(),
            object(),
            queue,
            policy,
        )

    outcome = _thread(run_outcome_disposition, errors)
    assert outcome_started.wait(1)
    assert not outcome_read.wait(0.05)

    release_cleanup.set()
    cancellation.join(2)
    outcome.join(2)

    assert not cancellation.is_alive()
    assert not outcome.is_alive()
    assert errors == []
    assert outcome_read.is_set()
    assert policy.calls == []
    assert queue.outcome_cancelled.is_set()
    assert ("cancel", active.job_id, AS_OF, None) in queue.calls


def test_fresh_dispatch_holds_same_exclusion_against_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"study": _study()}
    ready = _queue_job(2, QueueJobStatus.READY)
    entered_dispatch = Event()
    release_dispatch = Event()
    cancellation_started = Event()
    cancellation_read = Event()
    errors: list[BaseException] = []
    acquire_results: list[object] = []

    class SharedQueue(_Queue):
        def attempt_by_acquire_token(self, token: str):
            return None

        def open_attempt_for_worker(self, worker_id: str):
            return None

        def select_ready_job(self, **kwargs):
            return ready

    queue = SharedQueue()
    monkeypatch.setattr(acquire_subject, "read_study_run", lambda *args: state["study"])
    monkeypatch.setattr(acquire_subject, "read_study_plan", lambda *args: object())

    def blocking_dispatch(*args, **kwargs):
        entered_dispatch.set()
        assert release_dispatch.wait(2)
        return "assignment"

    monkeypatch.setattr(acquire_subject, "dispatch_training_job", blocking_dispatch)

    def cancellation_read_study(*args):
        cancellation_read.set()
        return state["study"]

    monkeypatch.setattr(subject, "read_study_run", cancellation_read_study)
    monkeypatch.setattr(
        subject,
        "persist_study_run_transition",
        lambda next_run, *args: state.__setitem__("study", next_run),
    )

    request = AcquireWorkRequest("worker-1", "acquire-before-cancel", frozenset({"training"}))

    acquire = _thread(
        lambda: acquire_results.append(
            acquire_subject.handle_acquire_work(
                request,
                date(2026, 9, 8),
                object(),
                AS_OF,
                "fresh-lease",
                object(),
                object(),
                queue,
                object(),
                SimpleNamespace(),
            )
        ),
        errors,
    )
    assert entered_dispatch.wait(1)

    def run_cancellation() -> None:
        cancellation_started.set()
        subject.request_study_run_cancellation(
            STUDY_RUN_ID, FINISHED_AT, QUEUE_AT, object(), object(), queue
        )

    cancellation = _thread(run_cancellation, errors)
    assert cancellation_started.wait(1)
    assert not cancellation_read.wait(0.05)

    release_dispatch.set()
    acquire.join(2)
    cancellation.join(2)

    assert not acquire.is_alive()
    assert not cancellation.is_alive()
    assert errors == []
    assert acquire_results == ["assignment"]
    assert cancellation_read.is_set()
    assert state["study"].status is StudyRunStatus.CANCELLED
