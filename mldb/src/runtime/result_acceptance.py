"""Controller-side acceptance of successful Worker result candidates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from ..catalog.architecture import ArchitectureStatus
from ..evaluation.interface import EvaluationResult
from ..evaluation.preflight import preflight_evaluation
from ..evaluation.result_validation import accept_evaluation_result
from ..evaluation.run import (
    EvaluationRun,
    EvaluationRunExecution,
    EvaluationRunResult,
    EvaluationRunStatus,
)
from ..model.persistence import ensure_model_for_completed_training_run
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from .executable_loader import load_architecture_build
from .resolution import resolve_architecture
from .run_persistence import (
    persist_evaluation_run_transition,
    persist_training_run_transition,
    read_evaluation_run,
    read_training_run,
)
from ..training.run import (
    TrainingRun,
    TrainingRunExecution,
    TrainingRunResult,
    TrainingRunStatus,
)
from ..training.weights import CanonicalWeightsArtifact, load_canonical_weights

if TYPE_CHECKING:
    from ..orchestration.candidates import CandidateStore
    from ..orchestration.worker_api import (
        EvaluationSuccessCandidate,
        TrainingSuccessCandidate,
    )


def _read_verified_candidate_bytes(
    attempt_id: int,
    ref: object,
    store: object,
) -> bytes:
    from ..orchestration.candidates import read_verified_candidate_bytes

    return read_verified_candidate_bytes(attempt_id, ref, store)


def _require_exact_running_training_run(
    run: TrainingRun,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> None:
    current = read_training_run(run.id, layout, filesystem)
    if run.status is not TrainingRunStatus.RUNNING or current != run:
        raise ValueError("Supplied Training Run is not the exact current canonical RUNNING Run.")


def _stage_training_candidate(
    run: TrainingRun,
    payload: bytes,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> Path:
    run_paths = layout.training_run_paths(run.id)
    filesystem.ensure_directory(run_paths.work_dir)
    path = run_paths.work_dir / "result-acceptance-weights.pt"
    filesystem.replace_bytes(path, payload)
    if filesystem.read_bytes(path) != payload:
        raise ValueError("Training candidate staging changed exact bytes.")
    return path


def accept_training_success(
    run: TrainingRun,
    attempt_id: int,
    candidate: TrainingSuccessCandidate,
    finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    candidate_store: CandidateStore,
) -> TrainingRun:
    """Accept one authorized Training success candidate into canonical MLDB state."""

    _require_exact_running_training_run(run, layout, filesystem)
    payload = _read_verified_candidate_bytes(attempt_id, candidate.weights, candidate_store)

    architecture = resolve_architecture(run.architecture, layout, filesystem)
    if architecture.metadata.status is not ArchitectureStatus.SEALED:
        raise ValueError("Current Training Architecture is not sealed.")
    if architecture.metadata.implementation.sha256 is None:
        raise ValueError("Current sealed Training Architecture has no implementation SHA-256.")

    architecture_build = load_architecture_build(architecture)
    staged_path = _stage_training_candidate(run, payload, layout, filesystem)
    load_canonical_weights(staged_path, architecture_build)

    run_paths = layout.training_run_paths(run.id)
    filesystem.ensure_directory(run_paths.artifacts_dir)
    if filesystem.file_exists(run_paths.weights_path):
        existing = filesystem.read_bytes(run_paths.weights_path)
        if existing != payload:
            raise ValueError("Existing canonical Training weights conflict with candidate bytes.")
    else:
        filesystem.replace_bytes(run_paths.weights_path, payload)

    canonical_bytes = filesystem.read_bytes(run_paths.weights_path)
    if canonical_bytes != payload:
        raise ValueError("Canonical Training weights do not equal verified candidate bytes.")

    weights_artifact = CanonicalWeightsArtifact(
        format="pytorch-state-dict",
        path="artifacts/weights.pt",
        sha256=hashlib.sha256(canonical_bytes).hexdigest(),
        bytes=len(canonical_bytes),
    )
    completed = TrainingRun(
        schema=run.schema,
        id=run.id,
        status=TrainingRunStatus.COMPLETED,
        corpus=run.corpus,
        architecture=run.architecture,
        train_protocol=run.train_protocol,
        parameters=run.parameters,
        execution=TrainingRunExecution(
            seed=run.execution.seed,
            started_at=run.execution.started_at,
            finished_at=finished_at,
        ),
        result=TrainingRunResult(weights=weights_artifact),
        failure=None,
        study=run.study,
        environment=run.environment,
        work=run.work,
    )
    persist_training_run_transition(completed, layout, filesystem)
    ensure_model_for_completed_training_run(completed, layout, filesystem)
    return completed


def _stage_evaluation_candidates(
    run: EvaluationRun,
    attempt_id: int,
    candidate: EvaluationSuccessCandidate,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    candidate_store: CandidateStore,
) -> dict[str, Path]:
    verified = [
        (key, _read_verified_candidate_bytes(attempt_id, ref, candidate_store))
        for key, ref in candidate.artifacts.items()
    ]

    run_paths = layout.evaluation_run_paths(run.id)
    staging_dir = run_paths.work_dir / "result-acceptance"
    filesystem.ensure_directory(staging_dir)
    staged: dict[str, Path] = {}
    for index, (key, payload) in enumerate(verified):
        path = staging_dir / f"artifact-{index:04d}.bin"
        filesystem.replace_bytes(path, payload)
        if filesystem.read_bytes(path) != payload:
            raise ValueError(f"Evaluation candidate staging changed bytes for {key!r}.")
        staged[key] = path
    return staged


def accept_evaluation_success(
    run: EvaluationRun,
    attempt_id: int,
    candidate: EvaluationSuccessCandidate,
    finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    candidate_store: CandidateStore,
) -> EvaluationRun:
    """Accept one authorized Evaluation success candidate into canonical MLDB state."""

    current = read_evaluation_run(run.id, layout, filesystem)
    if run.status is not EvaluationRunStatus.RUNNING or current != run:
        raise ValueError("Supplied Evaluation Run is not the exact current canonical RUNNING Run.")

    preflight = preflight_evaluation(
        run.model,
        run.corpus,
        run.evaluation_protocol,
        run.parameters,
        layout,
        filesystem,
    )
    if preflight.parameters != run.parameters:
        raise ValueError("Fresh Evaluation preflight parameters disagree with persisted Run input.")

    staged_artifacts = _stage_evaluation_candidates(
        run,
        attempt_id,
        candidate,
        layout,
        filesystem,
        candidate_store,
    )
    reconstructed = EvaluationResult(
        metrics=candidate.metrics,
        artifacts=staged_artifacts,
        unavailable_outputs=candidate.unavailable_outputs,
    )
    run_paths = layout.evaluation_run_paths(run.id)
    accepted = accept_evaluation_result(
        preflight.protocol.metadata.outputs,
        reconstructed,
        preflight.task,
        run_paths.work_dir,
        run_paths.artifacts_dir,
    )
    partial = bool(accepted.unavailable_outputs) or bool(accepted.validation_issues)
    terminal = EvaluationRun(
        schema=run.schema,
        id=run.id,
        status=(
            EvaluationRunStatus.COMPLETED_PARTIAL
            if partial
            else EvaluationRunStatus.COMPLETED
        ),
        model=run.model,
        corpus=run.corpus,
        evaluation_protocol=run.evaluation_protocol,
        parameters=run.parameters,
        execution=EvaluationRunExecution(
            started_at=run.execution.started_at,
            finished_at=finished_at,
        ),
        result=EvaluationRunResult(
            metrics=accepted.metrics,
            artifacts=accepted.artifacts,
        ),
        unavailable_outputs=accepted.unavailable_outputs,
        validation_issues=accepted.validation_issues,
        failure=None,
        study=run.study,
        environment=run.environment,
    )
    persist_evaluation_run_transition(terminal, layout, filesystem)
    return terminal
