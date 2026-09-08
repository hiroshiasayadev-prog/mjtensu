from __future__ import annotations

from datetime import date
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from mldb.src.orchestration import acquire as subject
from mldb.src.orchestration import dispatch as dispatch_subject
from mldb.src.orchestration.jobs import (
    EvaluationJob,
    EvaluationJobCoordinate,
    TrainingJob,
    TrainingJobCoordinate,
)
from mldb.src.orchestration.queue import QueueAttempt, QueueJob, QueueJobStatus
from mldb.src.orchestration.retry_policy import RetryDecision
from mldb.src.orchestration.worker_api import AcquireRejection, AcquireWorkRequest, NoWork
from mldb.src.study.plan import StudyPlanEvaluation, StudyPlanRow, StudyPlanTraining
from mldb.src.study.run import StudyRunStatus


DAY = date(2026, 9, 8)
AS_OF = "2026-09-08T09:00:00.000000Z"
STUDY = "sr-20260908-001"
TRIAL = "trial-0001"


def _request(token: str = "acquire-1", accepts: frozenset[str] = frozenset({"training"})) -> AcquireWorkRequest:
    return AcquireWorkRequest("worker-1", token, accepts)


def _training_job(status: QueueJobStatus = QueueJobStatus.READY) -> QueueJob:
    return QueueJob(
        1,
        TrainingJob(TrainingJobCoordinate(STUDY, TRIAL)),
        status,
        None,
        None,
        AS_OF,
        AS_OF,
    )


def _evaluation_job(status: QueueJobStatus = QueueJobStatus.READY) -> QueueJob:
    return QueueJob(
        2,
        EvaluationJob(EvaluationJobCoordinate(STUDY, TRIAL, "holdout"), None),
        status,
        None,
        None,
        AS_OF,
        AS_OF,
    )


def _plan(training: bool = True):
    stage = StudyPlanEvaluation("holdout", "eval-corpus-v1", "eval-protocol-v1", {"batch": 8})
    if training:
        source = StudyPlanTraining("arch-v1", "train-corpus-v1", "train-protocol-v1", 7, {"epochs": 3})
        return (StudyPlanRow(TRIAL, (stage,), training=source),)
    return (StudyPlanRow(TRIAL, (stage,), model="mdl-20260908-001"),)


def _study(status: StudyRunStatus = StudyRunStatus.RUNNING, *, has_plan: bool = True) -> object:
    return SimpleNamespace(id=STUDY, status=status, plan=SimpleNamespace() if has_plan else None)


class _Queue:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.replay = None
        self.open_worker = None
        self.selected = None
        self.authorized = None
        self.parent = None
        self.attempts = ()

    def attempt_by_acquire_token(self, token: str):
        self.calls.append(("token", token))
        return self.replay

    def open_attempt_for_worker(self, worker: str):
        self.calls.append(("worker", worker))
        return self.open_worker

    def select_ready_job(self, **kwargs):
        self.calls.append(("select", kwargs))
        return self.selected

    def job_by_id(self, job_id: int):
        self.calls.append(("job", job_id))
        return self.parent

    def authorized_open_attempt(self, attempt_id: int, lease_token: str, **kwargs):
        self.calls.append(("authorize", attempt_id, lease_token, kwargs))
        return self.authorized

    def attempts_for_job(self, job_id: int):
        self.calls.append(("attempts", job_id))
        return self.attempts

    def defer_ready_job(self, job_id: int, **kwargs):
        self.calls.append(("defer", job_id, kwargs))
        return self.selected

    def fail_ready_job(self, job_id: int, **kwargs):
        self.calls.append(("fail", job_id, kwargs))
        return self.selected


def _run_acquire(queue: object, retry_policy: object, request: AcquireWorkRequest | object = None):
    return subject.handle_acquire_work(
        request if request is not None else _request(), DAY, object(), AS_OF, "fresh-lease",
        "layout", "fs", queue, "assets", retry_policy,
    )


def test_malformed_request_has_no_queue_or_canonical_mutation() -> None:
    invalid = object.__new__(AcquireWorkRequest)
    object.__setattr__(invalid, "worker_id", "")
    object.__setattr__(invalid, "acquire_token", "token")
    object.__setattr__(invalid, "accepts", frozenset({"training"}))
    queue = _Queue()
    result = _run_acquire(queue, SimpleNamespace(), invalid)
    assert isinstance(result, AcquireRejection)
    assert queue.calls == []


def test_replay_lookup_is_first_and_closed_token_rejects() -> None:
    queue = _Queue()
    queue.replay = QueueAttempt(
        9, 1, 1, "tr-20260908-001", "worker-1", "acquire-1", "persisted-lease",
        "2026-09-08T10:00:00.000000Z", AS_OF, finished_at=AS_OF,
    )
    result = _run_acquire(queue, SimpleNamespace())
    assert isinstance(result, AcquireRejection)
    assert queue.calls == [("token", "acquire-1")]


def _open_attempt(job_id: int, run_id: str, *, worker: str = "worker-1") -> QueueAttempt:
    return QueueAttempt(
        9, job_id, 1, run_id, worker, "acquire-1", "persisted-lease",
        "2026-09-08T10:00:00.000000Z", AS_OF,
    )


def test_valid_training_replay_uses_persisted_lease_and_never_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _Queue()
    attempt = _open_attempt(1, "tr-20260908-001")
    queue.replay = queue.authorized = attempt
    queue.parent = _training_job(QueueJobStatus.ACTIVE)
    run = SimpleNamespace(
        id=attempt.run_id, status=subject.TrainingRunStatus.RUNNING,
        study=SimpleNamespace(run=STUDY, trial=TRIAL), corpus="c", architecture="a",
        train_protocol="p", execution=SimpleNamespace(seed=7), parameters={"x": 1},
    )
    preflight = SimpleNamespace(
        corpus=SimpleNamespace(metadata=SimpleNamespace(id="c")),
        architecture=SimpleNamespace(metadata=SimpleNamespace(id="a")),
        protocol=SimpleNamespace(metadata=SimpleNamespace(id="p")), seed=7, parameters={"x": 1},
    )
    monkeypatch.setattr(subject, "read_training_run", lambda *args: run)
    monkeypatch.setattr(subject, "preflight_training", lambda *args: preflight)
    monkeypatch.setattr(subject, "project_training_assignment", lambda *args: "replayed")
    monkeypatch.setattr(subject, "dispatch_training_job", lambda *args, **kwargs: pytest.fail("fresh dispatch called"))

    assert _run_acquire(queue, SimpleNamespace()) == "replayed"
    authorize = next(call for call in queue.calls if call[0] == "authorize")
    assert authorize[2] == "persisted-lease"


def test_valid_evaluation_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _Queue()
    attempt = _open_attempt(2, "ev-20260908-001")
    queue.replay = queue.authorized = attempt
    queue.parent = _evaluation_job(QueueJobStatus.ACTIVE)
    run = SimpleNamespace(
        id=attempt.run_id, status=subject.EvaluationRunStatus.RUNNING,
        study=SimpleNamespace(run=STUDY, trial=TRIAL, stage="holdout"),
        model="m", corpus="c", evaluation_protocol="p", parameters={"x": 1},
    )
    preflight = SimpleNamespace(
        model=SimpleNamespace(metadata=SimpleNamespace(id="m")),
        corpus=SimpleNamespace(metadata=SimpleNamespace(id="c")),
        protocol=SimpleNamespace(metadata=SimpleNamespace(id="p")), parameters={"x": 1},
    )
    monkeypatch.setattr(subject, "read_evaluation_run", lambda *args: run)
    monkeypatch.setattr(subject, "preflight_evaluation", lambda *args: preflight)
    monkeypatch.setattr(subject, "project_evaluation_assignment", lambda *args: "eval-replay")
    monkeypatch.setattr(subject, "dispatch_evaluation_job", lambda *args, **kwargs: pytest.fail("fresh dispatch called"))
    assert _run_acquire(queue, SimpleNamespace(), _request(accepts=frozenset({"evaluation"}))) == "eval-replay"


@pytest.mark.parametrize("mode", ["stale", "worker", "capability", "lineage"])
def test_replay_rejections(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    queue = _Queue()
    attempt = _open_attempt(1, "tr-20260908-001", worker="other" if mode == "worker" else "worker-1")
    queue.replay = attempt
    queue.authorized = None if mode == "stale" else attempt
    queue.parent = _training_job(QueueJobStatus.ACTIVE)
    request = _request(accepts=frozenset({"evaluation"}) if mode == "capability" else frozenset({"training"}))
    if mode == "lineage":
        run = SimpleNamespace(
            id=attempt.run_id, status=subject.TrainingRunStatus.RUNNING,
            study=SimpleNamespace(run=STUDY, trial="trial-9999"), corpus="c", architecture="a",
            train_protocol="p", execution=SimpleNamespace(seed=7), parameters={},
        )
        monkeypatch.setattr(subject, "read_training_run", lambda *args: run)
    result = _run_acquire(queue, SimpleNamespace(), request)
    assert isinstance(result, AcquireRejection)


def test_one_open_worker_capability_filtering_and_no_work() -> None:
    queue = _Queue()
    queue.open_worker = _open_attempt(99, "tr-20260908-001")
    assert isinstance(_run_acquire(queue, SimpleNamespace()), AcquireRejection)

    queue = _Queue()
    result = _run_acquire(queue, SimpleNamespace(), _request(accepts=frozenset({"evaluation"})))
    assert isinstance(result, NoWork)
    select = next(call for call in queue.calls if call[0] == "select")
    assert select[1]["accepts_training"] is False
    assert select[1]["accepts_evaluation"] is True


def test_terminal_study_refuses_fresh_child(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _Queue()
    queue.selected = _training_job()
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study(StudyRunStatus.FAILED))
    monkeypatch.setattr(subject, "read_study_plan", lambda *args: pytest.fail("plan should not be read"))
    monkeypatch.setattr(subject, "dispatch_training_job", lambda *args, **kwargs: pytest.fail("dispatch called"))
    assert isinstance(_run_acquire(queue, SimpleNamespace()), AcquireRejection)


@pytest.mark.parametrize(
    ("job_factory", "accepts", "dispatch_name", "expected"),
    [
        (_training_job, frozenset({"training"}), "dispatch_training_job", "training-assignment"),
        (_evaluation_job, frozenset({"evaluation"}), "dispatch_evaluation_job", "evaluation-assignment"),
    ],
)
def test_fresh_success_exact_forwarding(monkeypatch: pytest.MonkeyPatch, job_factory, accepts, dispatch_name: str, expected: str) -> None:
    queue = _Queue()
    queue.selected = job_factory()
    plan = _plan(training=dispatch_name == "dispatch_training_job")
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study())
    monkeypatch.setattr(subject, "read_study_plan", lambda *args: plan)
    seen: list[object] = []
    monkeypatch.setattr(subject, dispatch_name, lambda *args, **kwargs: seen.append((args, kwargs)) or expected)
    result = _run_acquire(queue, SimpleNamespace(), _request(accepts=accepts))
    assert result == expected
    assert seen[0][1] == {
        "worker_id": "worker-1", "acquire_token": "acquire-1",
        "lease_token": "fresh-lease", "activated_at": AS_OF,
    }


class _RetryPolicy:
    def __init__(self, retry_not_before: str | None) -> None:
        self.decision = RetryDecision(retry_not_before)
        self.calls: list[object] = []

    def after_preflight_failure(self, job, attempts, *, at: str):
        self.calls.append((job, attempts, at))
        return self.decision


@pytest.mark.parametrize(
    ("retry_at", "mutation"),
    [("2026-09-08T09:01:00.000000Z", "defer"), (None, "fail")],
)
def test_confirmed_pre_run_failure_uses_retry_policy(monkeypatch: pytest.MonkeyPatch, retry_at: str | None, mutation: str) -> None:
    queue = _Queue()
    queue.selected = _training_job()
    prior = _open_attempt(1, "tr-20260907-001")
    queue.attempts = (prior,)
    policy = _RetryPolicy(retry_at)
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study())
    monkeypatch.setattr(subject, "read_study_plan", lambda *args: _plan())
    monkeypatch.setattr(dispatch_subject, "preflight_training", lambda *args: (_ for _ in ()).throw(ValueError("preflight")))

    result = _run_acquire(queue, policy)
    assert isinstance(result, AcquireRejection)
    assert policy.calls == [(queue.selected, (prior,), AS_OF)]
    mutation_call = next(call for call in queue.calls if call[0] == mutation)
    assert mutation_call[1] == queue.selected.job_id


def test_allocator_or_ambiguous_failure_never_runs_retry_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _Queue()
    queue.selected = _training_job()
    policy = _RetryPolicy("2026-09-08T09:01:00.000000Z")
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study())
    monkeypatch.setattr(subject, "read_study_plan", lambda *args: _plan())
    training = _plan()[0].training
    assert training is not None
    monkeypatch.setattr(dispatch_subject, "preflight_training", lambda *args: SimpleNamespace(parameters=training.parameters))
    monkeypatch.setattr(dispatch_subject, "allocate_training_run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("allocator")))
    with pytest.raises(RuntimeError, match="allocator"):
        _run_acquire(queue, policy)
    assert policy.calls == []
    assert not any(call[0] in {"attempts", "defer", "fail"} for call in queue.calls)

    queue = _Queue()
    queue.selected = _training_job()
    policy = _RetryPolicy(None)
    monkeypatch.setattr(subject, "dispatch_training_job", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("ambiguous")))
    with pytest.raises(ValueError, match="ambiguous"):
        _run_acquire(queue, policy)
    assert policy.calls == []
    assert not any(call[0] in {"attempts", "defer", "fail"} for call in queue.calls)


def test_concurrent_fresh_acquire_is_serialized_without_sleep_race(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _Queue()
    queue.selected = _training_job()
    entered_dispatch = Event()
    release_dispatch = Event()
    second_lookup = Event()
    original_lookup = queue.attempt_by_acquire_token

    def lookup(token: str):
        if token == "acquire-2":
            second_lookup.set()
        return original_lookup(token)

    queue.attempt_by_acquire_token = lookup
    monkeypatch.setattr(subject, "read_study_run", lambda *args: _study())
    monkeypatch.setattr(subject, "read_study_plan", lambda *args: _plan())

    def blocking_dispatch(*args, **kwargs):
        if kwargs["acquire_token"] == "acquire-1":
            entered_dispatch.set()
            assert release_dispatch.wait(1)
        return "assignment"

    monkeypatch.setattr(subject, "dispatch_training_job", blocking_dispatch)
    results: list[object] = []
    first = Thread(target=lambda: results.append(_run_acquire(queue, SimpleNamespace(), _request("acquire-1"))))
    second = Thread(target=lambda: results.append(_run_acquire(queue, SimpleNamespace(), _request("acquire-2"))))
    first.start(); assert entered_dispatch.wait(1); second.start()
    assert not second_lookup.wait(0.05)
    release_dispatch.set(); first.join(1); second.join(1)
    assert second_lookup.is_set()
    assert results == ["assignment", "assignment"]
