from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from mldb.src.api import controller
from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    EvaluationRunId,
    ModelId,
    StudyId,
    TaskId,
    TrainingRunId,
    TrainProtocolId,
)
from mldb.src.common.parameters import PublicParameterDeclaration
from mldb.src.evaluation.protocol import (
    EvaluationOutputs,
    EvaluationProtocol,
    EvaluationProtocolImplementation,
    EvaluationProtocolStatus,
)
from mldb.src.evaluation.run import (
    EvaluationRun,
    EvaluationRunExecution,
    EvaluationRunResult,
    EvaluationRunStatus,
    EvaluationRunStudyLineage,
)
from mldb.src.model.identity import model_id_for_training_run
from mldb.src.model.persistence import ensure_model_for_completed_training_run
from mldb.src.orchestration import acquire, lease_recovery, outcome, reconciliation, study_cancellation
from mldb.src.orchestration._sqlite_queue import SQLiteQueue
from mldb.src.orchestration.jobs import EvaluationJob, TrainingJob, derive_study_jobs
from mldb.src.orchestration.queue import QueueJobStatus, queue_database_path
from mldb.src.orchestration.retry_policy import RetryDecision
from mldb.src.orchestration.study_launch import StudyExecutionSetupError
from mldb.src.orchestration.worker_api import (
    AcquireWorkRequest,
    AttemptFailed,
    NoWork,
    OutcomeAcknowledgement,
)
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.run_persistence import (
    allocate_evaluation_run,
    allocate_study_run,
    allocate_training_run,
    finalize_study_plan,
    list_evaluation_runs,
    list_training_runs,
    persist_evaluation_run_transition,
    persist_study_run_transition,
    persist_training_run_transition,
    read_evaluation_run,
    read_study_run,
    read_training_run,
)
from mldb.src.study.definition import (
    Study,
    StudyEvaluationStage,
    StudyExistingModelSource,
    StudyStatus,
)
from mldb.src.study.plan import StudyPlanEvaluation, StudyPlanRow, StudyPlanTraining
from mldb.src.study.preflight import PreparedStudyExecution
from mldb.src.study.progress import get_study_run_progress
from mldb.src.study.run import StudyRunExecution, StudyRunStatus
from mldb.src.training.run import (
    TrainingRun,
    TrainingRunExecution,
    TrainingRunResult,
    TrainingRunStatus,
    TrainingRunStudyLineage,
)
from mldb.src.training.weights import CanonicalWeightsArtifact


DAY = date(2026, 9, 8)
STARTED = datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc)
CHILD_FINISHED = datetime(2026, 9, 8, 9, 45, tzinfo=timezone.utc)
STUDY_FINISHED = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
T0 = "2026-09-08T09:00:00.000000Z"
T_HALF = "2026-09-08T09:30:00.000000Z"
T_EXPIRED = "2026-09-08T10:00:00.000000Z"
T_BEFORE_RETRY = "2026-09-08T10:59:59.999999Z"
T_RETRY = "2026-09-08T11:00:00.000000Z"
T_LATE = "2026-09-08T12:00:00.000000Z"


class _Policy:
    def __init__(self, retry_not_before: str | None) -> None:
        self.retry_not_before = retry_not_before
        self.calls: list[tuple[object, tuple[object, ...], str, str]] = []

    def after_unsatisfied_attempt(self, job, attempt_history, outcome_name, *, at: str):
        self.calls.append((job, tuple(attempt_history), outcome_name, at))
        return RetryDecision(self.retry_not_before)


class _UnusedCandidates:
    def read(self, *args, **kwargs):
        raise AssertionError("candidate store must not be used for failure outcomes")


def _existing_plan(*, stages: tuple[str, ...] = ("quality",), model: ModelId = ModelId("mdl-20260908-900")):
    return (
        StudyPlanRow(
            trial="trial-0001",
            model=model,
            evaluations=tuple(
                StudyPlanEvaluation(
                    stage=stage,
                    corpus=CorpusId("corpus-eval"),
                    protocol=EvaluationProtocolId("eval-proto"),
                    parameters={"batch": 8},
                )
                for stage in stages
            ),
        ),
    )


def _training_plan(*, trials: int = 1):
    return tuple(
        StudyPlanRow(
            trial=f"trial-{index:04d}",
            training=StudyPlanTraining(
                architecture=ArchitectureId("arch-a"),
                corpus=CorpusId("corpus-train"),
                protocol=TrainProtocolId("train-proto"),
                seed=40 + index,
                parameters={"epochs": 3},
            ),
            evaluations=(
                StudyPlanEvaluation(
                    stage="quality",
                    corpus=CorpusId("corpus-eval"),
                    protocol=EvaluationProtocolId("eval-proto"),
                    parameters={"batch": 8},
                ),
            ),
        )
        for index in range(1, trials + 1)
    )


def _new_study(layout: RepositoryLayout, filesystem: LocalFilesystem, plan):
    prepared = SimpleNamespace(
        study=SimpleNamespace(metadata=SimpleNamespace(id=StudyId("study.restart-e2e")))
    )
    allocated = allocate_study_run(prepared, DAY, STARTED, layout, filesystem)
    return finalize_study_plan(allocated.id, plan, layout, filesystem)


def _admit(study, plan, queue: SQLiteQueue):
    return queue.admit_study_jobs(
        study.id,
        derive_study_jobs(study.id, plan),
        admitted_at=T0,
    )


def _allocate_training(study, row: StudyPlanRow, layout, filesystem) -> TrainingRun:
    assert row.training is not None
    preflight = SimpleNamespace(
        corpus=SimpleNamespace(metadata=SimpleNamespace(id=row.training.corpus)),
        architecture=SimpleNamespace(metadata=SimpleNamespace(id=row.training.architecture)),
        protocol=SimpleNamespace(metadata=SimpleNamespace(id=row.training.protocol)),
        parameters=row.training.parameters,
        seed=row.training.seed,
    )
    return allocate_training_run(
        preflight,
        DAY,
        STARTED,
        layout,
        filesystem,
        study=TrainingRunStudyLineage(study.id, row.trial),
    )


def _allocate_evaluation(
    study,
    row: StudyPlanRow,
    layout,
    filesystem,
    *,
    stage_index: int = 0,
    model: ModelId | None = None,
) -> EvaluationRun:
    evaluation = row.evaluations[stage_index]
    resolved_model = model if model is not None else row.model
    assert resolved_model is not None
    preflight = SimpleNamespace(
        model=SimpleNamespace(metadata=SimpleNamespace(id=resolved_model)),
        corpus=SimpleNamespace(metadata=SimpleNamespace(id=evaluation.corpus)),
        protocol=SimpleNamespace(metadata=SimpleNamespace(id=evaluation.protocol)),
        parameters=evaluation.parameters,
    )
    return allocate_evaluation_run(
        preflight,
        DAY,
        STARTED,
        layout,
        filesystem,
        study=EvaluationRunStudyLineage(study.id, row.trial, evaluation.stage),
    )


def _complete_training(run: TrainingRun, layout, filesystem) -> TrainingRun:
    payload = b"restart-e2e-weights"
    paths = layout.training_run_paths(run.id)
    filesystem.replace_bytes(paths.weights_path, payload)
    completed = replace(
        run,
        status=TrainingRunStatus.COMPLETED,
        execution=TrainingRunExecution(
            seed=run.execution.seed,
            started_at=run.execution.started_at,
            finished_at=CHILD_FINISHED,
        ),
        result=TrainingRunResult(
            CanonicalWeightsArtifact(
                "pytorch-state-dict",
                "artifacts/weights.pt",
                hashlib.sha256(payload).hexdigest(),
                len(payload),
            )
        ),
        failure=None,
    )
    persist_training_run_transition(completed, layout, filesystem)
    return read_training_run(run.id, layout, filesystem)


def _complete_evaluation(run: EvaluationRun, layout, filesystem) -> EvaluationRun:
    completed = replace(
        run,
        status=EvaluationRunStatus.COMPLETED,
        execution=EvaluationRunExecution(
            started_at=run.execution.started_at,
            finished_at=CHILD_FINISHED,
        ),
        result=EvaluationRunResult(metrics={"score": 1.0}, artifacts={}),
        unavailable_outputs=(),
        validation_issues=(),
        failure=None,
    )
    persist_evaluation_run_transition(completed, layout, filesystem)
    return read_evaluation_run(run.id, layout, filesystem)


def _drop_queue_database(root: Path) -> None:
    database = queue_database_path(root)
    for path in (database, Path(str(database) + "-wal"), Path(str(database) + "-shm")):
        path.unlink(missing_ok=True)


def _canonical_bytes(study, layout, filesystem):
    paths = layout.study_run_paths(study.id)
    return (
        filesystem.read_bytes(paths.metadata_path),
        filesystem.read_bytes(paths.plan_path),
    )


def _reconcile(study, layout, filesystem, queue, policy, *, queue_at: str = T_HALF):
    return reconciliation.reconcile_study_run(
        study.id,
        STUDY_FINISHED,
        queue_at,
        layout,
        filesystem,
        queue,
        policy,
    )


def test_ordinary_restart_preserves_full_durable_queue_projection(tmp_path: Path) -> None:
    layout = RepositoryLayout(tmp_path)
    filesystem = LocalFilesystem()
    queue = SQLiteQueue(tmp_path)

    ready_plan = _training_plan()
    ready_study = _new_study(layout, filesystem, ready_plan)
    _admit(ready_study, ready_plan, queue)

    retry_plan = _existing_plan()
    retry_study = _new_study(layout, filesystem, retry_plan)
    retry_row = _admit(retry_study, retry_plan, queue)[0]
    queue.defer_ready_job(retry_row.job_id, retry_not_before=T_RETRY, at=T_HALF)

    satisfied_plan = _existing_plan()
    satisfied_study = _new_study(layout, filesystem, satisfied_plan)
    satisfied_row = _admit(satisfied_study, satisfied_plan, queue)[0]
    satisfied_attempt = queue.activate_attempt(
        satisfied_row.job_id,
        EvaluationRunId("ev-20260908-901"),
        "worker-satisfied",
        "acquire-satisfied",
        "lease-satisfied",
        activated_at=T0,
    )
    queue.close_attempt(
        satisfied_attempt.attempt_id,
        target_status=QueueJobStatus.SATISFIED,
        finished_at=T_HALF,
    )

    failed_plan = _existing_plan()
    failed_study = _new_study(layout, filesystem, failed_plan)
    failed_row = _admit(failed_study, failed_plan, queue)[0]
    failed_attempt = queue.activate_attempt(
        failed_row.job_id,
        EvaluationRunId("ev-20260908-902"),
        "worker-failed",
        "acquire-failed",
        "lease-failed",
        activated_at=T0,
    )
    queue.close_attempt(
        failed_attempt.attempt_id,
        target_status=QueueJobStatus.FAILED,
        finished_at=T_HALF,
    )

    active_plan = _existing_plan()
    active_study = _new_study(layout, filesystem, active_plan)
    active_row = _admit(active_study, active_plan, queue)[0]
    active_attempt = queue.activate_attempt(
        active_row.job_id,
        EvaluationRunId("ev-20260908-903"),
        "worker-active",
        "acquire-active",
        "lease-active",
        activated_at=T0,
    )

    studies = (ready_study, retry_study, satisfied_study, failed_study, active_study)
    canonical_before = {study.id: _canonical_bytes(study, layout, filesystem) for study in studies}
    jobs_before = {study.id: queue.jobs_for_study_run(study.id) for study in studies}
    attempts_before = {
        job.job_id: queue.attempts_for_job(job.job_id)
        for rows in jobs_before.values()
        for job in rows
    }

    reopened = SQLiteQueue(tmp_path)

    assert {study.id: reopened.jobs_for_study_run(study.id) for study in studies} == jobs_before
    assert {
        job.job_id: reopened.attempts_for_job(job.job_id)
        for rows in jobs_before.values()
        for job in rows
    } == attempts_before
    assert {study.id: _canonical_bytes(study, layout, filesystem) for study in studies} == canonical_before
    reopened_active = reopened.attempt_by_id(active_attempt.attempt_id)
    assert reopened_active == active_attempt
    assert (
        reopened_active.attempt_no,
        reopened_active.worker_id,
        reopened_active.acquire_token,
        reopened_active.lease_token,
        reopened_active.lease_until,
        reopened_active.finished_at,
    ) == (1, "worker-active", "acquire-active", "lease-active", T_EXPIRED, None)


def test_healthy_active_attempt_survives_restart_and_reconciliation(tmp_path: Path) -> None:
    layout = RepositoryLayout(tmp_path)
    filesystem = LocalFilesystem()
    plan = _existing_plan()
    study = _new_study(layout, filesystem, plan)
    queue = SQLiteQueue(tmp_path)
    row = _admit(study, plan, queue)[0]
    run = _allocate_evaluation(study, plan[0], layout, filesystem)
    attempt = queue.activate_attempt(
        row.job_id,
        run.id,
        "worker-1",
        "acquire-1",
        "lease-1",
        activated_at=T0,
    )
    run_bytes = filesystem.read_bytes(layout.evaluation_run_paths(run.id).metadata_path)

    reopened = SQLiteQueue(tmp_path)
    policy = _Policy(None)
    result = _reconcile(study, layout, filesystem, reopened, policy, queue_at=T_HALF)

    assert result.status is StudyRunStatus.RUNNING
    assert reopened.job_by_id(row.job_id).status is QueueJobStatus.ACTIVE
    assert reopened.open_attempt_for_job(row.job_id) == attempt
    assert reopened.attempt_by_id(attempt.attempt_id) == attempt
    assert read_evaluation_run(run.id, layout, filesystem) == run
    assert filesystem.read_bytes(layout.evaluation_run_paths(run.id).metadata_path) == run_bytes
    assert policy.calls == []


def test_expired_active_after_restart_fails_canonical_then_retries_durably(tmp_path: Path) -> None:
    layout = RepositoryLayout(tmp_path)
    filesystem = LocalFilesystem()
    plan = _existing_plan()
    study = _new_study(layout, filesystem, plan)
    queue = SQLiteQueue(tmp_path)
    row = _admit(study, plan, queue)[0]
    old_run = _allocate_evaluation(study, plan[0], layout, filesystem)
    old_attempt = queue.activate_attempt(
        row.job_id,
        old_run.id,
        "worker-1",
        "acquire-1",
        "lease-1",
        activated_at=T0,
    )

    queue_close_observed: list[EvaluationRunStatus] = []

    class CanonicalFirstQueue(SQLiteQueue):
        def close_attempt(self, *args, **kwargs):
            current = read_evaluation_run(old_run.id, layout, filesystem)
            queue_close_observed.append(current.status)
            assert current.status is EvaluationRunStatus.FAILED
            assert current.failure is None
            return super().close_attempt(*args, **kwargs)

    reopened = CanonicalFirstQueue(tmp_path)
    policy = _Policy(T_RETRY)
    lease_recovery.recover_expired_attempts(
        as_of=T_EXPIRED,
        run_finished_at=CHILD_FINISHED,
        layout=layout,
        filesystem=filesystem,
        queue=reopened,
        retry_policy=policy,
    )

    failed = read_evaluation_run(old_run.id, layout, filesystem)
    repaired = reopened.job_by_id(row.job_id)
    closed = reopened.attempt_by_id(old_attempt.attempt_id)
    assert failed.status is EvaluationRunStatus.FAILED
    assert failed.failure is None
    assert repaired.status is QueueJobStatus.RETRY_WAIT
    assert repaired.retry_not_before == T_RETRY
    assert closed.finished_at == T_EXPIRED
    assert queue_close_observed == [EvaluationRunStatus.FAILED]
    assert reopened.authorized_open_attempt(old_attempt.attempt_id, old_attempt.lease_token, as_of=T_EXPIRED) is None
    assert len(policy.calls) == 1

    restarted_again = SQLiteQueue(tmp_path)
    assert restarted_again.job_by_id(row.job_id) == repaired
    assert restarted_again.select_ready_job(
        accepts_training=False, accepts_evaluation=True, as_of=T_BEFORE_RETRY
    ) is None
    due = restarted_again.select_ready_job(
        accepts_training=False, accepts_evaluation=True, as_of=T_RETRY
    )
    assert due.job_id == row.job_id
    assert due.status is QueueJobStatus.READY

    fresh_run = _allocate_evaluation(study, plan[0], layout, filesystem)
    fresh_attempt = restarted_again.activate_attempt(
        row.job_id,
        fresh_run.id,
        "worker-2",
        "acquire-2",
        "lease-2",
        activated_at=T_RETRY,
    )
    assert fresh_run.id != old_run.id
    assert fresh_attempt.attempt_no == 2
    assert read_evaluation_run(old_run.id, layout, filesystem).status is EvaluationRunStatus.FAILED
    assert restarted_again.attempt_by_id(old_attempt.attempt_id).finished_at == T_EXPIRED


def test_retry_outcome_state_and_exact_replay_survive_restart(tmp_path: Path) -> None:
    layout = RepositoryLayout(tmp_path)
    filesystem = LocalFilesystem()
    plan = _existing_plan()
    study = _new_study(layout, filesystem, plan)
    queue = SQLiteQueue(tmp_path)
    row = _admit(study, plan, queue)[0]
    run = _allocate_evaluation(study, plan[0], layout, filesystem)
    attempt = queue.activate_attempt(
        row.job_id,
        run.id,
        "worker-1",
        "acquire-1",
        "lease-1",
        activated_at=T0,
    )
    policy = _Policy(T_RETRY)
    exact = AttemptFailed(attempt.attempt_id, attempt.lease_token, "worker-error", "boom")

    assert outcome.handle_attempt_outcome(
        exact,
        CHILD_FINISHED,
        T_HALF,
        layout,
        filesystem,
        queue,
        _UnusedCandidates(),
        policy,
    ) is OutcomeAcknowledgement.ACCEPTED

    canonical_before = filesystem.read_bytes(layout.evaluation_run_paths(run.id).metadata_path)
    reopened = SQLiteQueue(tmp_path)
    retry_job = reopened.job_by_id(row.job_id)
    history = reopened.attempts_for_job(row.job_id)
    assert retry_job.status is QueueJobStatus.RETRY_WAIT
    assert retry_job.retry_not_before == T_RETRY
    assert len(history) == 1 and history[0].finished_at == T_HALF

    assert outcome.handle_attempt_outcome(
        exact,
        CHILD_FINISHED,
        T_HALF,
        layout,
        filesystem,
        reopened,
        _UnusedCandidates(),
        policy,
    ) is OutcomeAcknowledgement.ALREADY_FINALIZED
    conflicting = AttemptFailed(attempt.attempt_id, attempt.lease_token, "worker-error", "different")
    assert outcome.handle_attempt_outcome(
        conflicting,
        CHILD_FINISHED,
        T_HALF,
        layout,
        filesystem,
        reopened,
        _UnusedCandidates(),
        policy,
    ) is OutcomeAcknowledgement.REJECTED
    assert filesystem.read_bytes(layout.evaluation_run_paths(run.id).metadata_path) == canonical_before
    assert reopened.job_by_id(row.job_id) == retry_job
    assert reopened.attempts_for_job(row.job_id) == history
    assert len(policy.calls) == 1
    assert reopened.select_ready_job(
        accepts_training=False, accepts_evaluation=True, as_of=T_BEFORE_RETRY
    ) is None
    assert reopened.select_ready_job(
        accepts_training=False, accepts_evaluation=True, as_of=T_RETRY
    ).status is QueueJobStatus.READY


def test_queue_loss_completed_training_is_reconstructed_without_retraining(tmp_path: Path) -> None:
    layout = RepositoryLayout(tmp_path)
    filesystem = LocalFilesystem()
    plan = _training_plan()
    study = _new_study(layout, filesystem, plan)
    queue = SQLiteQueue(tmp_path)
    _admit(study, plan, queue)
    completed = _complete_training(
        _allocate_training(study, plan[0], layout, filesystem),
        layout,
        filesystem,
    )
    model = ensure_model_for_completed_training_run(completed, layout, filesystem)
    run_path = layout.training_run_paths(completed.id).metadata_path
    model_path = layout.model_metadata_path(model.id)
    run_before = filesystem.read_bytes(run_path)
    model_before = filesystem.read_bytes(model_path)

    _drop_queue_database(tmp_path)
    rebuilt = SQLiteQueue(tmp_path)
    policy = _Policy(None)
    first = _reconcile(study, layout, filesystem, rebuilt, policy)
    rows = rebuilt.jobs_for_study_run(study.id)
    training = next(row for row in rows if isinstance(row.logical, TrainingJob))
    evaluation = next(row for row in rows if isinstance(row.logical, EvaluationJob))

    assert first.status is StudyRunStatus.RUNNING
    assert training.status is QueueJobStatus.SATISFIED
    assert evaluation.status is QueueJobStatus.READY
    assert rebuilt.attempts_for_job(training.job_id) == ()
    assert len(list_training_runs(layout, filesystem)) == 1
    assert filesystem.read_bytes(run_path) == run_before
    assert filesystem.read_bytes(model_path) == model_before

    before_second = rebuilt.jobs_for_study_run(study.id)
    second = _reconcile(study, layout, filesystem, rebuilt, policy)
    assert second.status is StudyRunStatus.RUNNING
    assert rebuilt.jobs_for_study_run(study.id) == before_second
    assert len(list_training_runs(layout, filesystem)) == 1
    assert filesystem.read_bytes(run_path) == run_before
    assert filesystem.read_bytes(model_path) == model_before
    assert policy.calls == []


@pytest.mark.parametrize("model_already_present", [True, False])
def test_completed_training_stale_active_repairs_model_gap_before_queue_satisfaction(
    tmp_path: Path,
    model_already_present: bool,
) -> None:
    layout = RepositoryLayout(tmp_path)
    filesystem = LocalFilesystem()
    plan = _training_plan()
    study = _new_study(layout, filesystem, plan)
    queue = SQLiteQueue(tmp_path)
    rows = _admit(study, plan, queue)
    training_row = next(row for row in rows if isinstance(row.logical, TrainingJob))
    completed = _complete_training(
        _allocate_training(study, plan[0], layout, filesystem),
        layout,
        filesystem,
    )
    model_id = model_id_for_training_run(completed.id)
    if model_already_present:
        ensure_model_for_completed_training_run(completed, layout, filesystem)
    model_path = layout.model_metadata_path(model_id)
    model_before = filesystem.read_bytes(model_path) if model_already_present else None
    run_path = layout.training_run_paths(completed.id).metadata_path
    run_before = filesystem.read_bytes(run_path)
    attempt = queue.activate_attempt(
        training_row.job_id,
        completed.id,
        "worker-stale",
        "acquire-stale",
        "lease-stale",
        activated_at=T0,
    )

    model_visible_before_queue_close: list[bool] = []

    class ModelFirstQueue(SQLiteQueue):
        def close_attempt(self, *args, **kwargs):
            model_visible_before_queue_close.append(filesystem.file_exists(model_path))
            assert filesystem.file_exists(model_path)
            return super().close_attempt(*args, **kwargs)

    restarted = ModelFirstQueue(tmp_path)
    policy = _Policy(None)
    result = _reconcile(study, layout, filesystem, restarted, policy)

    assert result.status is StudyRunStatus.RUNNING
    assert model_visible_before_queue_close == [True]
    assert restarted.job_by_id(training_row.job_id).status is QueueJobStatus.SATISFIED
    assert restarted.attempt_by_id(attempt.attempt_id).finished_at == T_HALF
    evaluation = next(
        row for row in restarted.jobs_for_study_run(study.id) if isinstance(row.logical, EvaluationJob)
    )
    assert evaluation.status is QueueJobStatus.READY
    assert filesystem.file_exists(model_path)
    assert filesystem.read_text(model_path, encoding="utf-8") == (
        f"schema: mjtensu.mldb/model/v1\n"
        f"id: {model_id}\n"
        f"training_run: {completed.id}\n"
    )
    if model_before is not None:
        assert filesystem.read_bytes(model_path) == model_before
    assert filesystem.read_bytes(run_path) == run_before
    assert len(list_training_runs(layout, filesystem)) == 1

    rows_before = restarted.jobs_for_study_run(study.id)
    model_after_first = filesystem.read_bytes(model_path)
    _reconcile(study, layout, filesystem, restarted, policy)
    assert restarted.jobs_for_study_run(study.id) == rows_before
    assert filesystem.read_bytes(model_path) == model_after_first
    assert filesystem.read_bytes(run_path) == run_before
    assert len(list_training_runs(layout, filesystem)) == 1
    assert policy.calls == []


def test_queue_loss_completed_evaluation_finalizes_and_progress_stays_canonical(tmp_path: Path) -> None:
    layout = RepositoryLayout(tmp_path)
    filesystem = LocalFilesystem()
    plan = _existing_plan()
    study = _new_study(layout, filesystem, plan)
    queue = SQLiteQueue(tmp_path)
    _admit(study, plan, queue)
    completed = _complete_evaluation(
        _allocate_evaluation(study, plan[0], layout, filesystem),
        layout,
        filesystem,
    )
    run_path = layout.evaluation_run_paths(completed.id).metadata_path
    run_before = filesystem.read_bytes(run_path)

    _drop_queue_database(tmp_path)
    rebuilt = SQLiteQueue(tmp_path)
    policy = _Policy(None)
    result = _reconcile(study, layout, filesystem, rebuilt, policy, queue_at=T_LATE)
    job = rebuilt.jobs_for_study_run(study.id)[0]

    assert result.status is StudyRunStatus.COMPLETED
    assert result.execution.finished_at == STUDY_FINISHED
    assert result.summary is None
    assert job.status is QueueJobStatus.SATISFIED
    assert rebuilt.attempts_for_job(job.job_id) == ()
    assert len(list_evaluation_runs(layout, filesystem)) == 1
    assert filesystem.read_bytes(run_path) == run_before

    canonical_after_first = read_study_run(study.id, layout, filesystem)
    jobs_after_first = rebuilt.jobs_for_study_run(study.id)
    second = _reconcile(canonical_after_first, layout, filesystem, rebuilt, policy, queue_at=T_LATE)
    assert second == canonical_after_first
    assert rebuilt.jobs_for_study_run(study.id) == jobs_after_first
    assert len(list_evaluation_runs(layout, filesystem)) == 1
    assert filesystem.read_bytes(run_path) == run_before

    progress = get_study_run_progress(study.id, T_LATE, layout, filesystem, rebuilt)
    assert progress.study_run == canonical_after_first
    assert progress.training is None
    assert progress.evaluation.total == 1
    assert progress.evaluation.satisfied == 1
    assert progress.evaluation_incomplete == ()
    assert not hasattr(progress, "attempts")


def test_queue_loss_running_orphan_fails_without_synthetic_attempt_and_retries(tmp_path: Path) -> None:
    layout = RepositoryLayout(tmp_path)
    filesystem = LocalFilesystem()
    plan = _existing_plan()
    study = _new_study(layout, filesystem, plan)
    queue = SQLiteQueue(tmp_path)
    _admit(study, plan, queue)
    running = _allocate_evaluation(study, plan[0], layout, filesystem)

    _drop_queue_database(tmp_path)
    rebuilt = SQLiteQueue(tmp_path)
    policy = _Policy(T_RETRY)
    first = _reconcile(study, layout, filesystem, rebuilt, policy)
    job = rebuilt.jobs_for_study_run(study.id)[0]
    failed = read_evaluation_run(running.id, layout, filesystem)

    assert first.status is StudyRunStatus.RUNNING
    assert failed.status is EvaluationRunStatus.FAILED
    assert failed.failure is None
    assert job.status is QueueJobStatus.RETRY_WAIT
    assert job.retry_not_before == T_RETRY
    assert rebuilt.attempts_for_job(job.job_id) == ()
    assert rebuilt.open_attempt_for_job(job.job_id) is None
    assert len(list_evaluation_runs(layout, filesystem)) == 1
    assert len(policy.calls) == 1

    jobs_before = rebuilt.jobs_for_study_run(study.id)
    run_before = filesystem.read_bytes(layout.evaluation_run_paths(running.id).metadata_path)
    second = _reconcile(study, layout, filesystem, rebuilt, policy)
    assert second.status is StudyRunStatus.RUNNING
    assert rebuilt.jobs_for_study_run(study.id) == jobs_before
    assert filesystem.read_bytes(layout.evaluation_run_paths(running.id).metadata_path) == run_before
    assert rebuilt.attempts_for_job(job.job_id) == ()
    assert len(policy.calls) == 1


def test_completed_evaluation_stale_active_restart_repairs_queue_only_idempotently(tmp_path: Path) -> None:
    layout = RepositoryLayout(tmp_path)
    filesystem = LocalFilesystem()
    plan = _existing_plan()
    study = _new_study(layout, filesystem, plan)
    queue = SQLiteQueue(tmp_path)
    row = _admit(study, plan, queue)[0]
    completed = _complete_evaluation(
        _allocate_evaluation(study, plan[0], layout, filesystem),
        layout,
        filesystem,
    )
    run_path = layout.evaluation_run_paths(completed.id).metadata_path
    run_before = filesystem.read_bytes(run_path)
    attempt = queue.activate_attempt(
        row.job_id,
        completed.id,
        "worker-stale",
        "acquire-stale",
        "lease-stale",
        activated_at=T0,
    )

    restarted = SQLiteQueue(tmp_path)
    policy = _Policy(None)
    first = _reconcile(study, layout, filesystem, restarted, policy)
    repaired = restarted.job_by_id(row.job_id)
    closed = restarted.attempt_by_id(attempt.attempt_id)

    assert first.status is StudyRunStatus.COMPLETED
    assert repaired.status is QueueJobStatus.SATISFIED
    assert closed.finished_at == T_HALF
    assert filesystem.read_bytes(run_path) == run_before
    assert len(list_evaluation_runs(layout, filesystem)) == 1

    canonical = read_study_run(study.id, layout, filesystem)
    queue_before = restarted.jobs_for_study_run(study.id)
    history_before = restarted.attempts_for_job(row.job_id)
    second = _reconcile(canonical, layout, filesystem, restarted, policy)
    assert second == canonical
    assert restarted.jobs_for_study_run(study.id) == queue_before
    assert restarted.attempts_for_job(row.job_id) == history_before
    assert filesystem.read_bytes(run_path) == run_before
    assert len(list_evaluation_runs(layout, filesystem)) == 1
    assert policy.calls == []


def test_cancellation_commit_gap_replay_and_unsatisfied_outcome_cannot_reestablish_retry(tmp_path: Path) -> None:
    layout = RepositoryLayout(tmp_path)
    filesystem = LocalFilesystem()
    plan = _training_plan(trials=4)
    study = _new_study(layout, filesystem, plan)
    queue = SQLiteQueue(tmp_path)
    rows = _admit(study, plan, queue)
    training_rows = sorted(
        (row for row in rows if isinstance(row.logical, TrainingJob)),
        key=lambda row: row.logical.coordinate.trial,
    )
    queue.defer_ready_job(training_rows[1].job_id, retry_not_before=T_RETRY, at=T_HALF)
    active_run = _allocate_training(study, plan[2], layout, filesystem)
    active_attempt = queue.activate_attempt(
        training_rows[2].job_id,
        active_run.id,
        "worker-active",
        "acquire-active",
        "lease-active",
        activated_at=T0,
    )

    cancelled = replace(
        read_study_run(study.id, layout, filesystem),
        status=StudyRunStatus.CANCELLED,
        execution=StudyRunExecution(started_at=study.execution.started_at, finished_at=STUDY_FINISHED),
        summary=None,
    )
    persist_study_run_transition(cancelled, layout, filesystem)
    canonical_cancelled_bytes = filesystem.read_bytes(layout.study_run_paths(study.id).metadata_path)

    restarted = SQLiteQueue(tmp_path)
    assert study_cancellation.request_study_run_cancellation(
        study.id,
        STUDY_FINISHED,
        T_HALF,
        layout,
        filesystem,
        restarted,
    ) == "already_terminal"

    repaired = restarted.jobs_for_study_run(study.id)
    by_id = {row.job_id: row for row in repaired}
    assert by_id[training_rows[0].job_id].status is QueueJobStatus.CANCELLED
    assert by_id[training_rows[1].job_id].status is QueueJobStatus.CANCELLED
    assert by_id[training_rows[2].job_id].status is QueueJobStatus.ACTIVE
    assert all(
        row.status is QueueJobStatus.CANCELLED
        for row in repaired
        if isinstance(row.logical, EvaluationJob)
    )
    assert filesystem.read_bytes(layout.study_run_paths(study.id).metadata_path) == canonical_cancelled_bytes

    no_retry_policy = _Policy(T_RETRY)
    fresh = acquire.handle_acquire_work(
        AcquireWorkRequest("worker-fresh", "acquire-fresh", frozenset({"training", "evaluation"})),
        DAY,
        STARTED,
        T_HALF,
        "lease-fresh",
        layout,
        filesystem,
        restarted,
        object(),
        no_retry_policy,
    )
    assert isinstance(fresh, NoWork)
    assert no_retry_policy.calls == []

    assert study_cancellation.request_study_run_cancellation(
        study.id,
        STUDY_FINISHED,
        T_HALF,
        layout,
        filesystem,
        restarted,
    ) == "already_terminal"
    assert filesystem.read_bytes(layout.study_run_paths(study.id).metadata_path) == canonical_cancelled_bytes
    assert restarted.job_by_id(training_rows[2].job_id).status is QueueJobStatus.ACTIVE

    unsatisfied_policy = _Policy(T_RETRY)
    result = outcome.handle_attempt_outcome(
        AttemptFailed(active_attempt.attempt_id, active_attempt.lease_token, "worker-error", "after cancel"),
        CHILD_FINISHED,
        T_HALF,
        layout,
        filesystem,
        restarted,
        _UnusedCandidates(),
        unsatisfied_policy,
    )
    assert result is OutcomeAcknowledgement.ACCEPTED
    assert unsatisfied_policy.calls == []
    assert restarted.job_by_id(training_rows[2].job_id).status is QueueJobStatus.CANCELLED
    assert all(row.status is not QueueJobStatus.RETRY_WAIT for row in restarted.jobs_for_study_run(study.id))
    failed_child = read_training_run(active_run.id, layout, filesystem)
    assert failed_child.status is TrainingRunStatus.FAILED
    assert failed_child.failure.type == "worker-error"
    assert failed_child.failure.message == "after cancel"
    assert read_study_run(study.id, layout, filesystem).status is StudyRunStatus.CANCELLED


def _controller_prepared() -> PreparedStudyExecution:
    task_id = TaskId("task-v1")
    evaluation_protocol = EvaluationProtocol(
        schema="mjtensu.mldb/evaluation-protocol/v1",
        id=EvaluationProtocolId("eval-v1"),
        status=EvaluationProtocolStatus.SEALED,
        task=task_id,
        name="evaluation",
        description="evaluation",
        implementation=EvaluationProtocolImplementation(entrypoint="evaluate", sha256="b" * 64),
        parameters={"batch_size": PublicParameterDeclaration(default=32)},
        outputs=EvaluationOutputs(metrics={}, artifacts={}),
    )
    study = Study(
        schema="mjtensu.mldb/study/v1",
        id=StudyId("launch-existing-v1"),
        status=StudyStatus.SEALED,
        name="restart e2e",
        description="restart e2e",
        model=StudyExistingModelSource(models=(ModelId("mdl-20260908-901"),)),
        evaluations=(
            StudyEvaluationStage(
                stage="holdout",
                corpus=CorpusId("eval-corpus-v1"),
                protocol=evaluation_protocol.id,
                parameters={"batch_size": 64},
            ),
        ),
    )
    return PreparedStudyExecution(
        study=SimpleNamespace(metadata=study),
        task=SimpleNamespace(metadata=SimpleNamespace(id=task_id)),
        train_protocol=None,
        evaluation_protocols=MappingProxyType(
            {evaluation_protocol.id: SimpleNamespace(metadata=evaluation_protocol)}
        ),
    )


def test_public_execute_study_setup_failure_keeps_exact_allocated_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = RepositoryLayout(tmp_path)
    filesystem = LocalFilesystem()
    prepared = _controller_prepared()

    class FailingAdmissionQueue:
        def admit_study_jobs(self, *args, **kwargs):
            raise OSError("injected admission failure")

    monkeypatch.setattr(controller, "_resolve_study", lambda *args: prepared.study)
    monkeypatch.setattr(controller, "_preflight_study_execution", lambda *args: prepared)

    with pytest.raises(StudyExecutionSetupError) as caught:
        controller.execute_study(
            prepared.study.metadata.id,
            DAY,
            STARTED,
            layout,
            filesystem,
            FailingAdmissionQueue(),
            admitted_at=T0,
            failed_at=STUDY_FINISHED,
        )

    allocated_id = caught.value.study_run_id
    canonical = read_study_run(allocated_id, layout, filesystem)
    assert canonical.id == allocated_id
    assert canonical.status is StudyRunStatus.FAILED
    assert canonical.execution.finished_at == STUDY_FINISHED
    assert isinstance(caught.value.__cause__, OSError), repr(caught.value.__cause__)
    assert canonical.plan is not None
    assert filesystem.file_exists(layout.study_run_paths(allocated_id).metadata_path)
    assert filesystem.file_exists(layout.study_run_paths(allocated_id).plan_path)
    study_directories = tuple(
        child
        for child in filesystem.list_directory(layout.entity_directory(controller.EntityKind.STUDY_RUN))
        if child.is_dir()
    )
    assert [child.name for child in study_directories] == [allocated_id]
