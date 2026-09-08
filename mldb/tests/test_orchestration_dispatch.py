from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from mldb.src.common.errors import NotFoundError
from mldb.src.orchestration import dispatch as subject
from mldb.src.orchestration.jobs import (
    EvaluationJob,
    EvaluationJobCoordinate,
    TrainingJob,
    TrainingJobCoordinate,
)
from mldb.src.orchestration.queue import QueueJob, QueueJobStatus
from mldb.src.study.plan import StudyPlanEvaluation, StudyPlanRow, StudyPlanTraining
from mldb.src.training.run import TrainingRunStatus


DAY = date(2026, 9, 8)
STARTED = object()
AS_OF = "2026-09-08T09:00:00.000000Z"
STUDY = "sr-20260908-001"
TRIAL = "trial-0001"


def _training() -> StudyPlanTraining:
    return StudyPlanTraining(
        architecture="arch-v1",
        corpus="train-corpus-v1",
        protocol="train-protocol-v1",
        seed=7,
        parameters={"epochs": 3},
    )


def _stage(stage: str = "holdout") -> StudyPlanEvaluation:
    return StudyPlanEvaluation(
        stage=stage,
        corpus="eval-corpus-v1",
        protocol="eval-protocol-v1",
        parameters={"batch": 8},
    )


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


def _evaluation_job(
    *,
    dependency: TrainingJobCoordinate | None,
    status: QueueJobStatus = QueueJobStatus.READY,
    stage: str = "holdout",
) -> QueueJob:
    logical = EvaluationJob(
        EvaluationJobCoordinate(STUDY, TRIAL, stage),
        dependency,
    )
    return QueueJob(2, logical, status, 1 if dependency else None, None, AS_OF, AS_OF)


def _queue(calls: list[object], attempt: object) -> object:
    def activate(*args: object, **kwargs: object) -> object:
        calls.append(("activate", args, kwargs))
        return attempt

    return SimpleNamespace(activate_attempt=activate)


def _success_common(monkeypatch: pytest.MonkeyPatch, calls: list[object]) -> tuple[object, object]:
    run = SimpleNamespace(id="tr-20260908-001")
    attempt = SimpleNamespace(attempt_id=1)
    monkeypatch.setattr(subject, "project_training_assignment", lambda *args: calls.append(("project", args)) or "assignment")
    return run, attempt


def test_training_success_exact_order_and_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    training = _training()
    plan = (StudyPlanRow(TRIAL, (_stage(),), training=training),)
    preflight = SimpleNamespace(parameters=training.parameters)
    run, attempt = _success_common(monkeypatch, calls)

    monkeypatch.setattr(subject, "preflight_training", lambda *args: calls.append(("preflight", args)) or preflight)
    monkeypatch.setattr(
        subject,
        "allocate_training_run",
        lambda *args, **kwargs: calls.append(("allocate", args, kwargs)) or run,
    )
    queue = _queue(calls, attempt)

    result = subject.dispatch_training_job(
        _training_job(), plan, DAY, STARTED, "layout", "fs", queue, "assets",
        worker_id="worker", acquire_token="acquire", lease_token="lease", activated_at=AS_OF,
    )

    assert result == "assignment"
    assert [call[0] for call in calls] == ["preflight", "allocate", "activate", "project"]
    assert calls[0][1] == (training.corpus, training.architecture, training.protocol, training.seed, training.parameters, "layout", "fs")
    assert calls[1][1] == (preflight, DAY, STARTED, "layout", "fs")
    assert calls[1][2]["study"] == subject.TrainingRunStudyLineage(STUDY, TRIAL)
    assert calls[2][1] == (1, run.id, "worker", "acquire", "lease")
    assert calls[2][2] == {"activated_at": AS_OF}


def test_existing_model_evaluation_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    stage = _stage()
    plan = (StudyPlanRow(TRIAL, (stage,), model="mdl-20260908-009"),)
    preflight = SimpleNamespace(parameters=stage.parameters)
    run = SimpleNamespace(id="ev-20260908-001")
    attempt = SimpleNamespace(attempt_id=2)
    monkeypatch.setattr(subject, "preflight_evaluation", lambda *args: calls.append(("preflight", args)) or preflight)
    monkeypatch.setattr(subject, "allocate_evaluation_run", lambda *args, **kwargs: calls.append(("allocate", args, kwargs)) or run)
    monkeypatch.setattr(subject, "project_evaluation_assignment", lambda *args: calls.append(("project", args)) or "assignment")

    result = subject.dispatch_evaluation_job(
        _evaluation_job(dependency=None), plan, DAY, STARTED, "layout", "fs", _queue(calls, attempt), "assets",
        worker_id="worker", acquire_token="acquire", lease_token="lease", activated_at=AS_OF,
    )

    assert result == "assignment"
    assert [call[0] for call in calls] == ["preflight", "allocate", "activate", "project"]
    assert calls[0][1] == ("mdl-20260908-009", stage.corpus, stage.protocol, stage.parameters, "layout", "fs")
    assert calls[1][2]["study"] == subject.EvaluationRunStudyLineage(STUDY, TRIAL, stage.stage)


def _completed_upstream(training: StudyPlanTraining, *, suffix: str = "001") -> object:
    return SimpleNamespace(
        id=f"tr-20260908-{suffix}", status=TrainingRunStatus.COMPLETED,
        study=SimpleNamespace(run=STUDY, trial=TRIAL), corpus=training.corpus,
        architecture=training.architecture, train_protocol=training.protocol,
        execution=SimpleNamespace(seed=training.seed), parameters=training.parameters,
    )


def test_training_derived_evaluation_uses_only_completed_canonical_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    training = _training()
    stage = _stage()
    dependency = TrainingJobCoordinate(STUDY, TRIAL)
    plan = (StudyPlanRow(TRIAL, (stage,), training=training),)
    completed = _completed_upstream(training, suffix="007")
    ignored = SimpleNamespace(status=TrainingRunStatus.FAILED, study=completed.study)
    seen: list[object] = []

    monkeypatch.setattr(subject, "list_training_runs", lambda *args: (ignored, completed))
    monkeypatch.setattr(subject, "preflight_evaluation", lambda *args: seen.append(args) or SimpleNamespace(parameters=stage.parameters))
    monkeypatch.setattr(subject, "allocate_evaluation_run", lambda *args, **kwargs: SimpleNamespace(id="ev-20260908-003"))
    monkeypatch.setattr(subject, "project_evaluation_assignment", lambda *args: "assignment")

    queue = SimpleNamespace(activate_attempt=lambda *args, **kwargs: SimpleNamespace())
    assert subject.dispatch_evaluation_job(
        _evaluation_job(dependency=dependency), plan, DAY, STARTED, "layout", "fs", queue, "assets",
        worker_id="worker", acquire_token="acquire", lease_token="lease", activated_at=AS_OF,
    ) == "assignment"
    assert seen[0][0] == "mdl-20260908-007"


@pytest.mark.parametrize("plan", [(), (StudyPlanRow(TRIAL, (_stage(),), training=_training()),) * 2])
def test_training_rejects_missing_or_duplicate_row(plan: object) -> None:
    with pytest.raises(ValueError):
        subject.dispatch_training_job(
            _training_job(), plan, DAY, STARTED, "layout", "fs", SimpleNamespace(), "assets",
            worker_id="w", acquire_token="a", lease_token="l", activated_at=AS_OF,
        )


def test_evaluation_rejects_missing_or_duplicate_stage() -> None:
    row = StudyPlanRow(TRIAL, (_stage("x"), _stage("x")), model="mdl-20260908-001")
    with pytest.raises(ValueError):
        subject.dispatch_evaluation_job(
            _evaluation_job(dependency=None, stage="x"), (row,), DAY, STARTED, "layout", "fs", SimpleNamespace(), "assets",
            worker_id="w", acquire_token="a", lease_token="l", activated_at=AS_OF,
        )


def test_wrong_job_status_type_and_dependency_are_rejected() -> None:
    training_plan = (StudyPlanRow(TRIAL, (_stage(),), training=_training()),)
    with pytest.raises(ValueError):
        subject.dispatch_training_job(
            _training_job(QueueJobStatus.ACTIVE), training_plan, DAY, STARTED, "l", "f", SimpleNamespace(), "a",
            worker_id="w", acquire_token="a", lease_token="l", activated_at=AS_OF,
        )
    with pytest.raises(ValueError):
        subject.dispatch_training_job(
            _evaluation_job(dependency=None), training_plan, DAY, STARTED, "l", "f", SimpleNamespace(), "a",
            worker_id="w", acquire_token="a", lease_token="l", activated_at=AS_OF,
        )
    with pytest.raises(ValueError):
        subject.dispatch_evaluation_job(
            _evaluation_job(dependency=TrainingJobCoordinate(STUDY, "trial-9999")), training_plan,
            DAY, STARTED, "l", "f", SimpleNamespace(), "a",
            worker_id="w", acquire_token="a", lease_token="l", activated_at=AS_OF,
        )


def test_preflight_parameter_mismatch_creates_no_run_or_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    training = _training()
    plan = (StudyPlanRow(TRIAL, (_stage(),), training=training),)
    monkeypatch.setattr(subject, "preflight_training", lambda *args: SimpleNamespace(parameters={"epochs": 4}))
    monkeypatch.setattr(subject, "allocate_training_run", lambda *args, **kwargs: calls.append("allocate"))
    queue = SimpleNamespace(activate_attempt=lambda *args, **kwargs: calls.append("activate"))

    with pytest.raises(ValueError):
        subject.dispatch_training_job(
            _training_job(), plan, DAY, STARTED, "l", "f", queue, "a",
            worker_id="w", acquire_token="a", lease_token="l", activated_at=AS_OF,
        )
    assert calls == []


@pytest.mark.parametrize("runs", [(), (_completed_upstream(_training()), _completed_upstream(_training(), suffix="002"))])
def test_training_derived_requires_exactly_one_completed_upstream(monkeypatch: pytest.MonkeyPatch, runs: tuple[object, ...]) -> None:
    training = _training()
    plan = (StudyPlanRow(TRIAL, (_stage(),), training=training),)
    monkeypatch.setattr(subject, "list_training_runs", lambda *args: runs)
    with pytest.raises(ValueError):
        subject.dispatch_evaluation_job(
            _evaluation_job(dependency=TrainingJobCoordinate(STUDY, TRIAL)), plan, DAY, STARTED,
            "l", "f", SimpleNamespace(), "a", worker_id="w", acquire_token="a",
            lease_token="l", activated_at=AS_OF,
        )


def test_upstream_input_mismatch_and_missing_model_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    training = _training()
    plan = (StudyPlanRow(TRIAL, (_stage(),), training=training),)
    bad = _completed_upstream(training)
    bad.parameters = {"epochs": 99}
    monkeypatch.setattr(subject, "list_training_runs", lambda *args: (bad,))
    with pytest.raises(ValueError):
        subject.dispatch_evaluation_job(
            _evaluation_job(dependency=TrainingJobCoordinate(STUDY, TRIAL)), plan, DAY, STARTED,
            "l", "f", SimpleNamespace(), "a", worker_id="w", acquire_token="a", lease_token="l", activated_at=AS_OF,
        )

    good = _completed_upstream(training)
    monkeypatch.setattr(subject, "list_training_runs", lambda *args: (good,))
    monkeypatch.setattr(subject, "preflight_evaluation", lambda *args: (_ for _ in ()).throw(NotFoundError("model")))
    with pytest.raises(NotFoundError):
        subject.dispatch_evaluation_job(
            _evaluation_job(dependency=TrainingJobCoordinate(STUDY, TRIAL)), plan, DAY, STARTED,
            "l", "f", SimpleNamespace(), "a", worker_id="w", acquire_token="a", lease_token="l", activated_at=AS_OF,
        )


def test_allocation_activation_and_projection_failures_do_not_compensate(monkeypatch: pytest.MonkeyPatch) -> None:
    training = _training()
    plan = (StudyPlanRow(TRIAL, (_stage(),), training=training),)
    monkeypatch.setattr(subject, "preflight_training", lambda *args: SimpleNamespace(parameters=training.parameters))
    queue_calls: list[str] = []
    queue = SimpleNamespace(activate_attempt=lambda *args, **kwargs: queue_calls.append("activate"))
    monkeypatch.setattr(subject, "allocate_training_run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("allocate")))
    with pytest.raises(RuntimeError, match="allocate"):
        subject.dispatch_training_job(
            _training_job(), plan, DAY, STARTED, "l", "f", queue, "a",
            worker_id="w", acquire_token="a", lease_token="l", activated_at=AS_OF,
        )
    assert queue_calls == []

    run = SimpleNamespace(id="tr-20260908-010")
    monkeypatch.setattr(subject, "allocate_training_run", lambda *args, **kwargs: run)
    queue.activate_attempt = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("activate"))
    with pytest.raises(RuntimeError, match="activate"):
        subject.dispatch_training_job(
            _training_job(), plan, DAY, STARTED, "l", "f", queue, "a",
            worker_id="w", acquire_token="a", lease_token="l", activated_at=AS_OF,
        )

    attempt = SimpleNamespace(attempt_id=3)
    queue.activate_attempt = lambda *args, **kwargs: attempt
    monkeypatch.setattr(subject, "project_training_assignment", lambda *args: (_ for _ in ()).throw(RuntimeError("project")))
    with pytest.raises(RuntimeError, match="project"):
        subject.dispatch_training_job(
            _training_job(), plan, DAY, STARTED, "l", "f", queue, "a",
            worker_id="w", acquire_token="a", lease_token="l", activated_at=AS_OF,
        )
