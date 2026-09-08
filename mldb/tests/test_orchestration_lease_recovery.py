from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import inspect
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mldb.src.common.ids import EvaluationRunId, StudyRunId, TrainingRunId
from mldb.src.evaluation.run import EvaluationRunStatus
from mldb.src.orchestration import heartbeat, outcome, lease_recovery as subject
from mldb.src.orchestration.jobs import (
    EvaluationJob,
    EvaluationJobCoordinate,
    TrainingJob,
    TrainingJobCoordinate,
)
from mldb.src.orchestration.queue import QueueAttempt, QueueJob, QueueJobStatus
from mldb.src.orchestration.retry_policy import RetryDecision
from mldb.src.orchestration.worker_api import AttemptCancelled, HeartbeatRequest
from mldb.src.study.run import StudyRunStatus
from mldb.src.training.run import TrainingRunStatus


AS_OF = "2026-09-08T11:00:00.000000Z"
HEARTBEAT_AT = "2026-09-08T10:30:00.000000Z"
RETRY_AT = "2026-09-08T12:00:00.000000Z"
STUDY = StudyRunId("sr-20260908-001")
TRAINING_RUN = TrainingRunId("tr-20260908-001")
EVALUATION_RUN = EvaluationRunId("ev-20260908-001")


def _job(kind: str, *, status: QueueJobStatus = QueueJobStatus.ACTIVE) -> QueueJob:
    if kind == "training":
        logical = TrainingJob(TrainingJobCoordinate(STUDY, "trial-0001"))
        job_id = 1
    else:
        logical = EvaluationJob(
            EvaluationJobCoordinate(STUDY, "trial-0001", "eval"),
            None,
        )
        job_id = 2
    return QueueJob(job_id, logical, status, None, None, AS_OF, AS_OF)


def _attempt(
    job: QueueJob,
    run_id: object,
    *,
    attempt_id: int = 10,
    lease_token: str = "lease-1",
    lease_until: str = HEARTBEAT_AT,
    finished_at: str | None = None,
) -> QueueAttempt:
    return QueueAttempt(
        attempt_id,
        job.job_id,
        1,
        run_id,
        "worker-1",
        "acquire-1",
        lease_token,
        lease_until,
        HEARTBEAT_AT,
        finished_at,
    )


def _run(run_id: object, status: object) -> SimpleNamespace:
    return SimpleNamespace(id=run_id, status=status)


def _study(status: StudyRunStatus) -> SimpleNamespace:
    return SimpleNamespace(id=STUDY, status=status)


def _configured(
    kind: str = "training",
    *,
    run_id: object | None = None,
    status: QueueJobStatus = QueueJobStatus.ACTIVE,
) -> tuple[MagicMock, MagicMock, QueueJob, QueueAttempt]:
    queue = MagicMock()
    policy = MagicMock()
    policy.after_unsatisfied_attempt.return_value = RetryDecision(None)
    job = _job(kind, status=status)
    if run_id is None:
        run_id = TRAINING_RUN if kind == "training" else EVALUATION_RUN
    current = _attempt(job, run_id)
    queue.expired_open_attempts.return_value = (current,)
    queue.attempt_by_id.return_value = current
    queue.job_by_id.return_value = job
    queue.open_attempt_for_job.return_value = current
    queue.authorized_open_attempt.return_value = None
    queue.attempts_for_job.return_value = (current,)
    return queue, policy, job, current


def _call(queue: object, policy: object) -> None:
    subject.recover_expired_attempts(
        as_of=AS_OF,
        run_finished_at="run-finished",
        layout="layout",
        filesystem="filesystem",
        queue=queue,
        retry_policy=policy,
    )


def _patch_unsatisfied_training(monkeypatch: pytest.MonkeyPatch, run: object) -> None:
    monkeypatch.setattr(subject, "read_training_run", lambda *args: run)
    monkeypatch.setattr(
        subject,
        "read_study_run",
        lambda *args: _study(StudyRunStatus.RUNNING),
    )


def _patch_unsatisfied_evaluation(monkeypatch: pytest.MonkeyPatch, run: object) -> None:
    monkeypatch.setattr(subject, "read_evaluation_run", lambda *args: run)
    monkeypatch.setattr(
        subject,
        "read_study_run",
        lambda *args: _study(StudyRunStatus.RUNNING),
    )


def test_expiry_discovery_and_fresh_authority_order_are_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    queue, policy, job, current = _configured()
    events: list[object] = []

    queue.expired_open_attempts.side_effect = lambda **kwargs: (
        events.append(("expired", kwargs)),
        (current,),
    )[1]
    queue.attempt_by_id.side_effect = lambda attempt_id: (
        events.append(("fresh", attempt_id)),
        current,
    )[1]
    queue.job_by_id.side_effect = lambda job_id: (
        events.append(("job", job_id)),
        job,
    )[1]
    queue.open_attempt_for_job.side_effect = lambda job_id: (
        events.append(("open", job_id)),
        current,
    )[1]
    queue.authorized_open_attempt.side_effect = lambda *args, **kwargs: (
        events.append(("authorized", args, kwargs)),
        None,
    )[1]
    queue.attempts_for_job.side_effect = lambda job_id: (
        events.append(("history", job_id)),
        (current,),
    )[1]
    queue.close_attempt.side_effect = lambda *args, **kwargs: events.append(("close", args, kwargs))
    policy.after_unsatisfied_attempt.side_effect = lambda *args, **kwargs: (
        events.append(("policy", args, kwargs)),
        RetryDecision(None),
    )[1]

    @contextmanager
    def exclusion():
        events.append("enter")
        yield
        events.append("exit")
    import mldb.src.orchestration._coordination as coordination

    monkeypatch.setattr(coordination, "orchestration_exclusion", exclusion)
    monkeypatch.setattr(
        subject,
        "read_training_run",
        lambda *args: (events.append("child"), _run(TRAINING_RUN, TrainingRunStatus.FAILED))[1],
    )
    monkeypatch.setattr(
        subject,
        "read_study_run",
        lambda *args: (events.append("study"), _study(StudyRunStatus.RUNNING))[1],
    )

    _call(queue, policy)

    assert events[:7] == [
        ("expired", {"as_of": AS_OF}),
        "enter",
        ("fresh", current.attempt_id),
        ("job", job.job_id),
        ("open", job.job_id),
        ("authorized", (current.attempt_id, current.lease_token), {"as_of": AS_OF}),
        "child",
    ]
    assert events.index("child") < events.index("study") < next(
        i for i, event in enumerate(events) if isinstance(event, tuple) and event[0] == "close"
    )


def test_fresh_heartbeat_extension_authorized_at_cutoff_skips_child(monkeypatch: pytest.MonkeyPatch) -> None:
    queue, policy, _, current = _configured()
    extended = replace(
        current,
        lease_until="2026-09-08T12:30:00.000000Z",
    )
    queue.attempt_by_id.return_value = extended
    queue.open_attempt_for_job.return_value = extended
    queue.authorized_open_attempt.return_value = extended
    monkeypatch.setattr(
        subject,
        "read_training_run",
        lambda *args: (_ for _ in ()).throw(AssertionError("child read must be skipped")),
    )

    _call(queue, policy)

    queue.expired_open_attempts.assert_called_once_with(as_of=AS_OF)
    queue.authorized_open_attempt.assert_called_once_with(
        current.attempt_id,
        current.lease_token,
        as_of=AS_OF,
    )
    queue.close_attempt.assert_not_called()
    policy.after_unsatisfied_attempt.assert_not_called()


def test_fresh_closed_attempt_is_skipped() -> None:
    queue, policy, _, current = _configured()
    queue.attempt_by_id.return_value = replace(current, finished_at=AS_OF)
    _call(queue, policy)
    queue.job_by_id.assert_not_called()
    queue.authorized_open_attempt.assert_not_called()


def test_missing_fresh_attempt_is_operation_failure() -> None:
    queue, policy, _, _ = _configured()
    queue.attempt_by_id.return_value = None
    with pytest.raises(RuntimeError, match="disappeared"):
        _call(queue, policy)
    queue.job_by_id.assert_not_called()


@pytest.mark.parametrize("field", ["job_id", "run_id", "lease_token"])
def test_discovery_fresh_identity_conflict_is_operation_failure(field: str) -> None:
    queue, policy, _, current = _configured()
    changes = {
        "job_id": current.job_id + 1,
        "run_id": TrainingRunId("tr-20260908-999"),
        "lease_token": "lease-other",
    }
    queue.attempt_by_id.return_value = replace(current, **{field: changes[field]})
    with pytest.raises(RuntimeError, match="disagrees"):
        _call(queue, policy)
    queue.job_by_id.assert_not_called()


def test_missing_parent_is_operation_failure() -> None:
    queue, policy, _, _ = _configured()
    queue.job_by_id.return_value = None
    with pytest.raises(RuntimeError, match="no parent"):
        _call(queue, policy)
    queue.open_attempt_for_job.assert_not_called()


def test_non_active_parent_is_operation_failure() -> None:
    queue, policy, _, current = _configured()
    queue.job_by_id.return_value = _job("training", status=QueueJobStatus.READY)
    with pytest.raises(RuntimeError, match="not ACTIVE"):
        _call(queue, policy)
    queue.open_attempt_for_job.assert_not_called()
    queue.authorized_open_attempt.assert_not_called()


def test_wrong_current_open_attempt_is_operation_failure() -> None:
    queue, policy, _, current = _configured()
    queue.open_attempt_for_job.return_value = replace(current, attempt_id=99)
    with pytest.raises(RuntimeError, match="not the current"):
        _call(queue, policy)
    queue.authorized_open_attempt.assert_not_called()


def test_authorization_contradictory_identity_is_operation_failure() -> None:
    queue, policy, _, current = _configured()
    queue.authorized_open_attempt.return_value = replace(current, attempt_id=99)
    with pytest.raises(RuntimeError, match="contradictory"):
        _call(queue, policy)


class _OpaqueRunId:
    def __str__(self) -> str:
        raise AssertionError("Run ID text must not be inspected")


def test_training_parent_selects_typed_read_without_string_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    opaque = _OpaqueRunId()
    queue, policy, _, _ = _configured("training", run_id=opaque)
    reader = MagicMock(return_value=_run(opaque, TrainingRunStatus.FAILED))
    monkeypatch.setattr(subject, "read_training_run", reader)
    monkeypatch.setattr(
        subject,
        "read_evaluation_run",
        lambda *args: (_ for _ in ()).throw(AssertionError("wrong typed read")),
    )
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study(StudyRunStatus.RUNNING))

    _call(queue, policy)

    reader.assert_called_once_with(opaque, "layout", "filesystem")


def test_evaluation_parent_selects_read_evaluation_run(monkeypatch: pytest.MonkeyPatch) -> None:
    queue, policy, _, _ = _configured("evaluation")
    reader = MagicMock(return_value=_run(EVALUATION_RUN, EvaluationRunStatus.FAILED))
    monkeypatch.setattr(subject, "read_evaluation_run", reader)
    monkeypatch.setattr(
        subject,
        "read_training_run",
        lambda *args: (_ for _ in ()).throw(AssertionError("wrong typed read")),
    )
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study(StudyRunStatus.RUNNING))

    _call(queue, policy)

    reader.assert_called_once_with(EVALUATION_RUN, "layout", "filesystem")


def test_canonical_child_identity_mismatch_is_operation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    queue, policy, _, _ = _configured("training")
    monkeypatch.setattr(
        subject,
        "read_training_run",
        lambda *args: _run(TrainingRunId("tr-20260908-999"), TrainingRunStatus.FAILED),
    )
    with pytest.raises(RuntimeError, match="Canonical child identity"):
        _call(queue, policy)
    policy.after_unsatisfied_attempt.assert_not_called()
    queue.close_attempt.assert_not_called()


@pytest.mark.parametrize(
    ("kind", "status", "failure_name"),
    [
        ("training", TrainingRunStatus.RUNNING, "fail_training_run"),
        ("evaluation", EvaluationRunStatus.RUNNING, "fail_evaluation_run"),
    ],
)
def test_running_child_fails_with_none_before_queue(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    status: object,
    failure_name: str,
) -> None:
    queue, policy, _, current = _configured(kind)
    run_id = current.run_id
    run = _run(run_id, status)
    events: list[str] = []
    if kind == "training":
        monkeypatch.setattr(subject, "read_training_run", lambda *args: run)
    else:
        monkeypatch.setattr(subject, "read_evaluation_run", lambda *args: run)
    fail = MagicMock(side_effect=lambda *args, **kwargs: events.append("canonical"))
    monkeypatch.setattr(subject, failure_name, fail)
    monkeypatch.setattr(
        subject,
        "read_study_run",
        lambda *args: (events.append("study"), _study(StudyRunStatus.RUNNING))[1],
    )
    policy.after_unsatisfied_attempt.side_effect = lambda *args, **kwargs: (
        events.append("policy"),
        RetryDecision(None),
    )[1]
    queue.close_attempt.side_effect = lambda *args, **kwargs: events.append("queue")

    _call(queue, policy)

    assert events == ["canonical", "study", "policy", "queue"]
    assert fail.call_args.kwargs["failure"] is None
    assert fail.call_args.args[1:] == ("run-finished", "layout", "filesystem")
    queue.close_attempt.assert_called_once_with(
        current.attempt_id,
        target_status=QueueJobStatus.FAILED,
        finished_at=AS_OF,
    )


@pytest.mark.parametrize("kind", ["training", "evaluation"])
def test_completed_child_satisfies_after_required_canonical_work(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    queue, policy, _, current = _configured(kind)
    events: list[str] = []
    if kind == "training":
        run = _run(TRAINING_RUN, TrainingRunStatus.COMPLETED)
        monkeypatch.setattr(subject, "read_training_run", lambda *args: run)
        monkeypatch.setattr(
            subject,
            "ensure_model_for_completed_training_run",
            lambda *args: events.append("model"),
        )
    else:
        run = _run(EVALUATION_RUN, EvaluationRunStatus.COMPLETED)
        monkeypatch.setattr(subject, "read_evaluation_run", lambda *args: run)
    queue.close_attempt.side_effect = lambda *args, **kwargs: events.append("queue")

    _call(queue, policy)

    assert events == (["model", "queue"] if kind == "training" else ["queue"])
    queue.close_attempt.assert_called_once_with(
        current.attempt_id,
        target_status=QueueJobStatus.SATISFIED,
        finished_at=AS_OF,
    )
    policy.after_unsatisfied_attempt.assert_not_called()


@pytest.mark.parametrize(
    ("kind", "status", "expected_outcome"),
    [
        ("training", TrainingRunStatus.FAILED, "failed"),
        ("training", TrainingRunStatus.CANCELLED, "cancelled"),
        ("evaluation", EvaluationRunStatus.COMPLETED_PARTIAL, "completed_partial"),
        ("evaluation", EvaluationRunStatus.FAILED, "failed"),
        ("evaluation", EvaluationRunStatus.CANCELLED, "cancelled"),
    ],
)
def test_terminal_unsatisfied_child_uses_retry_policy(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    status: object,
    expected_outcome: str,
) -> None:
    queue, policy, job, current = _configured(kind)
    run = _run(current.run_id, status)
    if kind == "training":
        _patch_unsatisfied_training(monkeypatch, run)
    else:
        _patch_unsatisfied_evaluation(monkeypatch, run)

    _call(queue, policy)

    policy.after_unsatisfied_attempt.assert_called_once_with(
        job,
        (current,),
        expected_outcome,
        at=AS_OF,
    )


@pytest.mark.parametrize(
    ("decision", "target"),
    [(RetryDecision(RETRY_AT), QueueJobStatus.RETRY_WAIT), (RetryDecision(None), QueueJobStatus.FAILED)],
)
def test_retry_decision_maps_mechanically_to_queue(
    monkeypatch: pytest.MonkeyPatch,
    decision: RetryDecision,
    target: QueueJobStatus,
) -> None:
    queue, policy, _, current = _configured("training")
    run = _run(TRAINING_RUN, TrainingRunStatus.FAILED)
    _patch_unsatisfied_training(monkeypatch, run)
    policy.after_unsatisfied_attempt.return_value = decision

    _call(queue, policy)

    expected = {
        "target_status": target,
        "finished_at": AS_OF,
    }
    if decision.retry_not_before is not None:
        expected["retry_not_before"] = RETRY_AT
    queue.close_attempt.assert_called_once_with(current.attempt_id, **expected)


def test_cancelled_study_suppresses_retry_policy_and_cancels_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue, policy, job, current = _configured("evaluation")
    monkeypatch.setattr(
        subject,
        "read_evaluation_run",
        lambda *args: _run(EVALUATION_RUN, EvaluationRunStatus.FAILED),
    )
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study(StudyRunStatus.CANCELLED))

    _call(queue, policy)

    policy.after_unsatisfied_attempt.assert_not_called()
    queue.cancel_job.assert_called_once_with(job.job_id, at=AS_OF)
    queue.close_attempt.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [StudyRunStatus.COMPLETED, StudyRunStatus.COMPLETED_WITH_FAILURES, StudyRunStatus.FAILED],
)
def test_other_terminal_study_is_operation_failure_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    status: StudyRunStatus,
) -> None:
    queue, policy, _, _ = _configured("training")
    monkeypatch.setattr(
        subject,
        "read_training_run",
        lambda *args: _run(TRAINING_RUN, TrainingRunStatus.FAILED),
    )
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study(status))

    with pytest.raises(RuntimeError, match="terminal Study Run"):
        _call(queue, policy)

    policy.after_unsatisfied_attempt.assert_not_called()
    queue.close_attempt.assert_not_called()
    queue.cancel_job.assert_not_called()


def test_model_ensure_failure_propagates_before_queue_close(monkeypatch: pytest.MonkeyPatch) -> None:
    queue, policy, _, _ = _configured("training")
    completed = _run(TRAINING_RUN, TrainingRunStatus.COMPLETED)
    monkeypatch.setattr(subject, "read_training_run", lambda *args: completed)
    monkeypatch.setattr(
        subject,
        "ensure_model_for_completed_training_run",
        lambda *args: (_ for _ in ()).throw(OSError("model store")),
    )

    with pytest.raises(OSError, match="model store"):
        _call(queue, policy)

    queue.close_attempt.assert_not_called()


@pytest.mark.parametrize("failure_site", ["repository", "queue"])
def test_repository_and_queue_failures_propagate(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    queue, policy, _, _ = _configured("evaluation")
    if failure_site == "repository":
        monkeypatch.setattr(
            subject,
            "read_evaluation_run",
            lambda *args: (_ for _ in ()).throw(OSError("repository unavailable")),
        )
        with pytest.raises(OSError, match="repository unavailable"):
            _call(queue, policy)
        queue.close_attempt.assert_not_called()
        return

    monkeypatch.setattr(
        subject,
        "read_evaluation_run",
        lambda *args: _run(EVALUATION_RUN, EvaluationRunStatus.COMPLETED),
    )
    queue.close_attempt.side_effect = OSError("queue unavailable")
    with pytest.raises(OSError, match="queue unavailable"):
        _call(queue, policy)


def test_recovery_creates_no_attempt_and_invents_no_worker_failure_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue, policy, _, _ = _configured("training")
    running = _run(TRAINING_RUN, TrainingRunStatus.RUNNING)
    fail = MagicMock()
    monkeypatch.setattr(subject, "read_training_run", lambda *args: running)
    monkeypatch.setattr(subject, "fail_training_run", fail)
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study(StudyRunStatus.RUNNING))

    _call(queue, policy)

    assert fail.call_args.kwargs == {"failure": None}
    queue.activate_attempt.assert_not_called()
    queue.defer_ready_job.assert_not_called()
    queue.fail_ready_job.assert_not_called()


def _thread_call(errors: list[BaseException], fn) -> None:
    try:
        fn()
    except BaseException as error:
        errors.append(error)


def test_heartbeat_extension_serializes_before_recovery_fresh_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job("training")
    discovered_attempt = _attempt(
        job,
        TRAINING_RUN,
        lease_until="2026-09-08T10:45:00.000000Z",
    )
    extended_attempt = replace(
        discovered_attempt,
        lease_until="2026-09-08T11:30:00.000000Z",
    )
    heartbeat_updating = Event()
    release_heartbeat = Event()
    discovery_called = Event()
    release_discovery = Event()
    recovery_fresh_read = Event()

    class SharedQueue:
        def __init__(self) -> None:
            self.current = discovered_attempt
            self.close_calls = 0

        def authorized_open_attempt(self, attempt_id, lease_token, *, as_of):
            if as_of == HEARTBEAT_AT:
                return self.current
            if as_of == AS_OF and self.current.lease_until > AS_OF:
                return self.current
            return None

        def job_by_id(self, job_id):
            return job

        def heartbeat_attempt(self, attempt_id, lease_token, *, accepted_at):
            heartbeat_updating.set()
            assert release_heartbeat.wait(1)
            self.current = extended_attempt
            return self.current

        def expired_open_attempts(self, *, as_of):
            assert as_of == AS_OF
            discovery_called.set()
            assert release_discovery.wait(1)
            return (discovered_attempt,)

        def attempt_by_id(self, attempt_id):
            recovery_fresh_read.set()
            return self.current

        def open_attempt_for_job(self, job_id):
            return self.current

        def close_attempt(self, *args, **kwargs):
            self.close_calls += 1
            raise AssertionError("freshly authorized attempt must not close")
    queue = SharedQueue()
    policy = MagicMock()
    errors: list[BaseException] = []
    monkeypatch.setattr(heartbeat, "read_study_run", lambda *args: _study(StudyRunStatus.RUNNING))
    monkeypatch.setattr(
        subject,
        "read_training_run",
        lambda *args: (_ for _ in ()).throw(AssertionError("recovery child read must be skipped")),
    )

    heartbeat_thread = Thread(
        target=_thread_call,
        args=(errors, lambda: heartbeat.handle_heartbeat(
            HeartbeatRequest(discovered_attempt.attempt_id, discovered_attempt.lease_token),
            HEARTBEAT_AT,
            "layout",
            "filesystem",
            queue,
        )),
    )
    heartbeat_thread.start()
    assert heartbeat_updating.wait(1)

    recovery_thread = Thread(target=_thread_call, args=(errors, lambda: _call(queue, policy)))
    recovery_thread.start()
    assert discovery_called.wait(1)
    release_discovery.set()
    assert not recovery_fresh_read.wait(0.05)
    release_heartbeat.set()
    heartbeat_thread.join(1)
    recovery_thread.join(1)
    assert recovery_fresh_read.is_set()
    assert queue.close_calls == 0
    assert errors == []


def test_outcome_authority_waits_while_recovery_owns_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_job = _job("training")
    current = _attempt(active_job, TRAINING_RUN)
    recovery_inside = Event()
    release_recovery = Event()
    outcome_authorized = Event()

    class SharedQueue:
        def __init__(self) -> None:
            self.current = current
            self.job = active_job
            self.authorization_calls = 0

        def expired_open_attempts(self, *, as_of):
            return (current,)

        def attempt_by_id(self, attempt_id):
            if not recovery_inside.is_set():
                recovery_inside.set()
                assert release_recovery.wait(1)
            return self.current

        def job_by_id(self, job_id):
            return self.job

        def open_attempt_for_job(self, job_id):
            return self.current

        def authorized_open_attempt(self, attempt_id, lease_token, *, as_of):
            self.authorization_calls += 1
            if self.authorization_calls == 1:
                return None
            outcome_authorized.set()
            return None

        def attempts_for_job(self, job_id):
            return (self.current,)

        def close_attempt(self, attempt_id, *, target_status, finished_at, **kwargs):
            self.current = replace(self.current, finished_at=finished_at)
            self.job = replace(self.job, status=target_status, updated_at=finished_at)
            return self.job, self.current

    queue = SharedQueue()
    policy = MagicMock()
    policy.after_unsatisfied_attempt.return_value = RetryDecision(None)
    canonical = _run(TRAINING_RUN, TrainingRunStatus.FAILED)
    canonical.failure = None
    monkeypatch.setattr(subject, "read_training_run", lambda *args: canonical)
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study(StudyRunStatus.RUNNING))
    monkeypatch.setattr(outcome, "read_training_run", lambda *args: canonical)
    errors: list[BaseException] = []

    recovery_thread = Thread(target=_thread_call, args=(errors, lambda: _call(queue, policy)))
    recovery_thread.start()
    assert recovery_inside.wait(1)

    outcome_thread = Thread(
        target=_thread_call,
        args=(errors, lambda: outcome.handle_attempt_outcome(
            AttemptCancelled(current.attempt_id, current.lease_token),
            "run-finished",
            AS_OF,
            "layout",
            "filesystem",
            queue,
            MagicMock(),
            policy,
        )),
    )
    outcome_thread.start()
    assert not outcome_authorized.wait(0.05)
    release_recovery.set()
    recovery_thread.join(1)
    outcome_thread.join(1)

    assert outcome_authorized.is_set()
    assert errors == []


def test_signature_is_frozen() -> None:
    signature = inspect.signature(subject.recover_expired_attempts)
    assert list(signature.parameters) == [
        "as_of",
        "run_finished_at",
        "layout",
        "filesystem",
        "queue",
        "retry_policy",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
