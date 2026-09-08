from __future__ import annotations

from ..common.errors import LifecycleConflictError
from ..evaluation.run import (
    EvaluationRun,
    EvaluationRunExecution,
    EvaluationRunFailure,
    EvaluationRunStatus,
)
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..training.run import (
    TrainingRun,
    TrainingRunExecution,
    TrainingRunFailure,
    TrainingRunStatus,
)
from .run_persistence import (
    persist_evaluation_run_transition,
    persist_training_run_transition,
    read_evaluation_run,
    read_training_run,
)


def fail_training_run(
    run: TrainingRun,
    finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    *,
    failure: TrainingRunFailure | None = None,
) -> TrainingRun:
    current = _fresh_training(run, layout, filesystem)
    failed = TrainingRun(
        schema=current.schema,
        id=current.id,
        status=TrainingRunStatus.FAILED,
        corpus=current.corpus,
        architecture=current.architecture,
        train_protocol=current.train_protocol,
        parameters=current.parameters,
        execution=TrainingRunExecution(
            seed=current.execution.seed,
            started_at=current.execution.started_at,
            finished_at=finished_at,
        ),
        result=None,
        failure=failure,
        study=current.study,
        environment=current.environment,
        work=current.work,
    )
    persist_training_run_transition(failed, layout, filesystem)
    return failed


def cancel_training_run(
    run: TrainingRun,
    finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> TrainingRun:
    current = _fresh_training(run, layout, filesystem)
    cancelled = TrainingRun(
        schema=current.schema,
        id=current.id,
        status=TrainingRunStatus.CANCELLED,
        corpus=current.corpus,
        architecture=current.architecture,
        train_protocol=current.train_protocol,
        parameters=current.parameters,
        execution=TrainingRunExecution(
            seed=current.execution.seed,
            started_at=current.execution.started_at,
            finished_at=finished_at,
        ),
        result=None,
        failure=None,
        study=current.study,
        environment=current.environment,
        work=current.work,
    )
    persist_training_run_transition(cancelled, layout, filesystem)
    return cancelled


def fail_evaluation_run(
    run: EvaluationRun,
    finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    *,
    failure: EvaluationRunFailure | None = None,
) -> EvaluationRun:
    current = _fresh_evaluation(run, layout, filesystem)
    failed = EvaluationRun(
        schema=current.schema,
        id=current.id,
        status=EvaluationRunStatus.FAILED,
        model=current.model,
        corpus=current.corpus,
        evaluation_protocol=current.evaluation_protocol,
        parameters=current.parameters,
        execution=EvaluationRunExecution(
            started_at=current.execution.started_at,
            finished_at=finished_at,
        ),
        result=None,
        unavailable_outputs=(),
        validation_issues=(),
        failure=failure,
        study=current.study,
        environment=current.environment,
    )
    persist_evaluation_run_transition(failed, layout, filesystem)
    return failed


def cancel_evaluation_run(
    run: EvaluationRun,
    finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> EvaluationRun:
    current = _fresh_evaluation(run, layout, filesystem)
    cancelled = EvaluationRun(
        schema=current.schema,
        id=current.id,
        status=EvaluationRunStatus.CANCELLED,
        model=current.model,
        corpus=current.corpus,
        evaluation_protocol=current.evaluation_protocol,
        parameters=current.parameters,
        execution=EvaluationRunExecution(
            started_at=current.execution.started_at,
            finished_at=finished_at,
        ),
        result=None,
        unavailable_outputs=(),
        validation_issues=(),
        failure=None,
        study=current.study,
        environment=current.environment,
    )
    persist_evaluation_run_transition(cancelled, layout, filesystem)
    return cancelled


def _fresh_training(
    supplied: TrainingRun,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> TrainingRun:
    current = read_training_run(supplied.id, layout, filesystem)
    if current.status is not TrainingRunStatus.RUNNING or current != supplied:
        raise LifecycleConflictError("Training Run is stale or already terminal")
    return current


def _fresh_evaluation(
    supplied: EvaluationRun,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> EvaluationRun:
    current = read_evaluation_run(supplied.id, layout, filesystem)
    if current.status is not EvaluationRunStatus.RUNNING or current != supplied:
        raise LifecycleConflictError("Evaluation Run is stale or already terminal")
    return current
