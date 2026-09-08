from __future__ import annotations

from threading import Event, Thread
from types import SimpleNamespace

import pytest

from mldb.src.orchestration import heartbeat as subject
from mldb.src.orchestration._coordination import orchestration_exclusion
from mldb.src.orchestration.jobs import TrainingJob, TrainingJobCoordinate
from mldb.src.orchestration.queue import QueueAttempt, QueueJob, QueueJobStatus
from mldb.src.orchestration.worker_api import HeartbeatAccepted, HeartbeatRejection, HeartbeatRequest
from mldb.src.study.run import StudyRunStatus


AS_OF = "2026-09-08T09:00:00.000000Z"
STUDY = "sr-20260908-001"


def _attempt(*, finished: str | None = None) -> QueueAttempt:
    return QueueAttempt(
        11, 3, 1, "tr-20260908-001", "worker-1", "acquire-1", "lease-1",
        "2026-09-08T10:00:00.000000Z", AS_OF, finished_at=finished,
    )


def _job() -> QueueJob:
    return QueueJob(
        3, TrainingJob(TrainingJobCoordinate(STUDY, "trial-0001")),
        QueueJobStatus.ACTIVE, None, None, AS_OF, AS_OF,
    )


class _Queue:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.authorized = _attempt()
        self.parent = _job()
        self.updated = _attempt()

    def authorized_open_attempt(self, attempt_id: int, lease_token: str, **kwargs):
        self.calls.append(("authorize", attempt_id, lease_token, kwargs))
        return self.authorized

    def job_by_id(self, job_id: int):
        self.calls.append(("job", job_id))
        return self.parent

    def heartbeat_attempt(self, attempt_id: int, lease_token: str, **kwargs):
        self.calls.append(("heartbeat", attempt_id, lease_token, kwargs))
        return self.updated


def _request() -> HeartbeatRequest:
    return HeartbeatRequest(11, "lease-1")


def _study(status: StudyRunStatus) -> object:
    return SimpleNamespace(id=STUDY, status=status)


@pytest.mark.parametrize("reason", ["unauthorized", "stale", "expired"])
def test_unauthorized_stale_expired_do_not_mutate(reason: str) -> None:
    queue = _Queue()
    queue.authorized = None
    result = subject.handle_heartbeat(_request(), AS_OF, "layout", "fs", queue)
    assert isinstance(result, HeartbeatRejection)
    assert queue.calls == [("authorize", 11, "lease-1", {"as_of": AS_OF})]


def test_missing_parent_rejects_without_heartbeat() -> None:
    queue = _Queue()
    queue.parent = None
    result = subject.handle_heartbeat(_request(), AS_OF, "layout", "fs", queue)
    assert isinstance(result, HeartbeatRejection)
    assert [call[0] for call in queue.calls] == ["authorize", "job"]


@pytest.mark.parametrize(
    ("status", "cancel_requested"),
    [(StudyRunStatus.RUNNING, False), (StudyRunStatus.CANCELLED, True)],
)
def test_running_and_cancelled_extend_exact_queue_lease(monkeypatch: pytest.MonkeyPatch, status: StudyRunStatus, cancel_requested: bool) -> None:
    queue = _Queue()
    queue.updated = SimpleNamespace(
        attempt_id=11, job_id=3, lease_token="lease-1", finished_at=None,
        lease_until="2026-09-08T10:00:01.123456Z",
    )
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study(status))
    result = subject.handle_heartbeat(_request(), AS_OF, "layout", "fs", queue)
    assert result == HeartbeatAccepted("2026-09-08T10:00:01.123456Z", cancel_requested)
    assert [call[0] for call in queue.calls] == ["authorize", "job", "heartbeat"]
    assert queue.calls[-1] == ("heartbeat", 11, "lease-1", {"accepted_at": AS_OF})


@pytest.mark.parametrize(
    "status",
    [StudyRunStatus.COMPLETED, StudyRunStatus.COMPLETED_WITH_FAILURES, StudyRunStatus.FAILED],
)
def test_non_cancelled_terminal_study_rejects_without_extension(monkeypatch: pytest.MonkeyPatch, status: StudyRunStatus) -> None:
    queue = _Queue()
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study(status))
    result = subject.handle_heartbeat(_request(), AS_OF, "layout", "fs", queue)
    assert isinstance(result, HeartbeatRejection)
    assert [call[0] for call in queue.calls] == ["authorize", "job"]


def test_parent_identity_inconsistency_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _Queue()
    monkeypatch.setattr(subject, "read_study_run", lambda *args: SimpleNamespace(id="sr-20260908-999", status=StudyRunStatus.RUNNING))
    result = subject.handle_heartbeat(_request(), AS_OF, "layout", "fs", queue)
    assert isinstance(result, HeartbeatRejection)
    assert [call[0] for call in queue.calls] == ["authorize", "job"]


def test_infrastructure_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _Queue()
    monkeypatch.setattr(subject, "read_study_run", lambda *args: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        subject.handle_heartbeat(_request(), AS_OF, "layout", "fs", queue)
    assert [call[0] for call in queue.calls] == ["authorize", "job"]


def test_heartbeat_and_recovery_style_work_share_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _Queue()
    entered_update = Event()
    release_update = Event()
    recovery_entered = Event()
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study(StudyRunStatus.RUNNING))

    def blocking_update(attempt_id: int, lease_token: str, **kwargs):
        entered_update.set()
        assert release_update.wait(1)
        return queue.updated

    queue.heartbeat_attempt = blocking_update
    heartbeat = Thread(target=lambda: subject.handle_heartbeat(_request(), AS_OF, "layout", "fs", queue))

    def recovery_style() -> None:
        with orchestration_exclusion():
            recovery_entered.set()

    heartbeat.start(); assert entered_update.wait(1)
    recovery = Thread(target=recovery_style); recovery.start()
    assert not recovery_entered.wait(0.05)
    release_update.set(); heartbeat.join(1); recovery.join(1)
    assert recovery_entered.is_set()
