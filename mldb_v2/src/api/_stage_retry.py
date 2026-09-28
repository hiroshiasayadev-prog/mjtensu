"""Narrow synchronous same-StudyResult retry for one failed evaluation stage."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from mldb_v2.src.backend.backend_port import BackendPort
from mldb_v2.src.backend.candidate_outcome import BackendObservation, StageKey, TerminalCandidate
from mldb_v2.src.common.ids import (
    EntityKind,
    EvaluationCoordinateId,
    EvaluationResultId,
    StudyResultId,
    TrialId,
    _canonical_json_bytes,
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
    _validate_typed_reference,
)
from mldb_v2.src.repository.canonical_writes import CanonicalRepositoryWriter
from mldb_v2.src.repository.mutation_coordination import StudyResultMutationCoordinator
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.results.study_result import StudyResult, _validate_study_result
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study._plan_build import _validate_study_plan
from mldb_v2.src.study.execution_readiness import (
    _materialize_evaluation_retry_stage_input,
    _validate_result_plan_lineage,
)
from mldb_v2.src.study.study_driver import _project_backend_summary_best_effort
from mldb_v2.src.verification.result_acceptance import (
    AcceptedResultRecordValidator,
    ResultAcceptor,
)


class _StageRetryBackendFailure(RuntimeError):
    pass


class _StageRetryUnsupported(RuntimeError):
    pass


def _load_context(
    resolver: CanonicalRepositoryResolver,
    *,
    study_result_id: StudyResultId,
) -> tuple[StudyResult, dict[str, object]]:
    result = _validate_study_result(
        resolver.resolve(kind=EntityKind.STUDY_RESULT, entity_id=study_result_id)
    )
    plan = _validate_study_plan(
        resolver.resolve(kind=EntityKind.STUDY_PLAN, entity_id=result["plan"])
    )
    _validate_result_plan_lineage(result, plan)
    return result, cast(dict[str, object], plan)


def _target_slot(
    result: StudyResult,
    *,
    trial: TrialId,
    coordinate: EvaluationCoordinateId,
) -> dict[str, object]:
    trials = [item for item in result["trials"] if item["trial"] == trial]
    if len(trials) != 1:
        raise FileNotFoundError(f"retry trial not found: {trial}")
    slots = [
        item for item in trials[0]["evaluations"]
        if item["coordinate"] == coordinate
    ]
    if len(slots) != 1:
        raise FileNotFoundError(f"retry evaluation coordinate not found: {coordinate}")
    return cast(dict[str, object], slots[0])


def _evaluation_id(
    study_result: StudyResultId,
    trial: TrialId,
    coordinate: EvaluationCoordinateId,
) -> EvaluationResultId:
    return EvaluationResultId(
        _validate_typed_reference(f"{study_result}-{trial}-{coordinate}")
    )


def _attempt_execution_ids(evaluation: Mapping[str, object]) -> list[str]:
    attempts = evaluation.get("attempts")
    if type(attempts) is not list or not attempts:
        raise ValueError("canonical EvaluationResult has no retryable attempt history")
    ids: list[str] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError("canonical EvaluationResult attempt is malformed")
        execution_id = attempt.get("execution_id")
        if type(execution_id) is not str or not execution_id:
            raise ValueError("canonical EvaluationResult attempt execution id is malformed")
        ids.append(execution_id)
    if len(set(ids)) != len(ids):
        raise ValueError("canonical EvaluationResult attempt execution ids are ambiguous")
    return ids


def _stage_key_from_input(stage_input: Mapping[str, object]) -> StageKey:
    return cast(StageKey, {
        "study_result": stage_input["study_result"],
        "plan": stage_input["plan"],
        "trial": stage_input["trial"],
        "kind": "evaluation",
        "coordinate": stage_input["coordinate"],
        "source_commit": stage_input["source_commit"],
    })


def _validate_observation(
    observation: object,
    *,
    expected_key: StageKey,
) -> BackendObservation:
    if type(observation) is not dict or observation.get("stage_key") != expected_key:
        raise _StageRetryBackendFailure(
            "backend retry observation does not match requested logical stage"
        )
    if observation.get("state") not in {"active", "terminal"}:
        raise _StageRetryBackendFailure("backend retry observation has invalid state")
    return cast(BackendObservation, observation)


def _candidate_history(
    candidate: TerminalCandidate,
    *,
    previous: Mapping[str, object],
) -> None:
    current_attempts = previous["attempts"]
    retry_attempts = candidate["attempts"]
    if type(current_attempts) is not list or type(retry_attempts) is not list:
        raise ValueError("retry attempt history is malformed")
    if len(retry_attempts) != len(current_attempts) + 1:
        raise ValueError("retry candidate must append exactly one physical attempt")
    if _canonical_json_bytes(retry_attempts[: len(current_attempts)]) != _canonical_json_bytes(
        current_attempts
    ):
        raise ValueError("retry candidate edited or reordered prior attempt history")
    execution_ids = [item["execution_id"] for item in retry_attempts]
    if len(set(execution_ids)) != len(execution_ids):
        raise ValueError("retry candidate contains duplicate physical execution ids")


def _all_dispositions(result: StudyResult) -> list[str]:
    values: list[str] = []
    for trial in result["trials"]:
        if trial["training"] is not None:
            values.append(trial["training"]["disposition"])
        values.extend(slot["disposition"] for slot in trial["evaluations"])
    return values


def _parent_after_retry(
    current: StudyResult,
    *,
    trial: TrialId,
    coordinate: EvaluationCoordinateId,
    disposition: str,
) -> StudyResult:
    replacement = copy.deepcopy(current)
    slot = _target_slot(replacement, trial=trial, coordinate=coordinate)
    if slot["disposition"] != "failed":
        raise ValueError("lifecycle conflict: retry target is no longer failed")
    slot["disposition"] = disposition
    slot["reason"] = None
    dispositions = _all_dispositions(replacement)
    replacement["status"] = (
        "completed"
        if dispositions and all(item == "completed" for item in dispositions)
        else "completed_with_failures"
    )
    replacement["diagnostic"] = None
    return replacement


def _recover_parent_from_completed_child(
    *,
    writer: CanonicalRepositoryWriter,
    current: StudyResult,
    trial: TrialId,
    coordinate: EvaluationCoordinateId,
) -> StudyResult:
    replacement = _parent_after_retry(
        current,
        trial=trial,
        coordinate=coordinate,
        disposition="completed",
    )
    stored = writer.replace_study_result_after_evaluation_retry(
        entity_id=current["id"],
        trial=trial,
        coordinate=coordinate,
        replacement=replacement,
    )
    return _validate_study_result(stored)


def retry_failed_evaluation_stage(
    *,
    repository_root: str | Path,
    study_result_id: StudyResultId | str,
    trial: TrialId | str,
    coordinate: EvaluationCoordinateId | str,
    backend: BackendPort,
    object_bytes: _ObjectByteAccess,
    wait: Callable[[], None],
) -> StudyResult:
    """Synchronously retry exactly one failed evaluation and persist its new attempt."""
    repository_root = Path(repository_root)
    mldb_data_root = repository_root / "mldb_data"
    result_id = StudyResultId(_validate_typed_reference(study_result_id))
    trial_id = TrialId(_validate_trial_id(trial))
    coordinate_id = EvaluationCoordinateId(_validate_evaluation_coordinate_id(coordinate))
    resolver = CanonicalRepositoryResolver(mldb_data_root)
    validator = AcceptedResultRecordValidator(mldb_data_root=mldb_data_root)
    writer = CanonicalRepositoryWriter(repository_root, record_validator=validator)
    coordinator = StudyResultMutationCoordinator(repository_root)
    acceptor = ResultAcceptor(mldb_data_root=mldb_data_root, object_bytes=object_bytes)

    # Serialize only the canonical preflight snapshot. Do not hold a mutation lock
    # while backend retry admission or remote execution runs.
    with coordinator.acquire(study_result_id=result_id):
        current, plan = _load_context(resolver, study_result_id=result_id)
        if current["status"] != "completed_with_failures":
            raise ValueError(
                "lifecycle conflict: retry-stage requires completed_with_failures StudyResult"
            )
        slot = _target_slot(current, trial=trial_id, coordinate=coordinate_id)
        if slot["disposition"] != "failed":
            raise ValueError("lifecycle conflict: retry-stage requires a failed evaluation")
        expected_evaluation_id = _evaluation_id(result_id, trial_id, coordinate_id)
        if slot["result"] != expected_evaluation_id:
            raise ValueError("retry target result identity does not match StudyResult slot")
        evaluation = resolver.resolve(
            kind=EntityKind.EVALUATION_RESULT,
            entity_id=expected_evaluation_id,
        )
        validator.validate(
            kind=EntityKind.EVALUATION_RESULT,
            entity_id=expected_evaluation_id,
            document=evaluation,
        )
        if evaluation["study_result"] != result_id or evaluation["plan"] != current["plan"]:
            raise ValueError("retry EvaluationResult lineage does not match StudyResult")
        if evaluation["status"] == "completed":
            return _recover_parent_from_completed_child(
                writer=writer,
                current=current,
                trial=trial_id,
                coordinate=coordinate_id,
            )
        if evaluation["status"] != "failed":
            raise ValueError("lifecycle conflict: only failed EvaluationResult may be retried")

        stage_input = _materialize_evaluation_retry_stage_input(
            plan=plan,
            result=current,
            trial=trial_id,
            coordinate=coordinate_id,
            mldb_data_root=mldb_data_root,
        )
        prior_execution_ids = _attempt_execution_ids(evaluation)
        retry_operation = getattr(backend, "retry_stage", None)
        if not callable(retry_operation):
            raise _StageRetryUnsupported("configured backend does not support stage retry")


    # Backend calls are deliberately outside the canonical mutation lock. Two
    # callers may race here; the backend retry key makes this admission idempotent.
    # Same-run retry must recover the existing stage/controller lineage from the
    # failed logical stage itself; it must not run generic Study execution ensure,
    # which can create/recover unrelated duplicate controllers.
    try:
        observation = retry_operation(
            stage_input=stage_input,
            prior_execution_ids=prior_execution_ids,
        )
    except Exception as error:
        raise _StageRetryBackendFailure("backend retry admission failed") from error

    expected_key = _stage_key_from_input(stage_input)
    observation = _validate_observation(observation, expected_key=expected_key)
    while observation["state"] == "active":
        wait()
        try:
            observed = backend.observe(stage_key=expected_key)
        except Exception as error:
            raise _StageRetryBackendFailure("backend retry observation failed") from error
        if observed is None:
            raise _StageRetryBackendFailure("backend retry became unobservable")
        observation = _validate_observation(observed, expected_key=expected_key)

    try:
        candidate = backend.collect(stage_key=expected_key)
    except Exception as error:
        raise _StageRetryBackendFailure("backend retry collection failed") from error
    if candidate is None:
        raise _StageRetryBackendFailure("terminal backend retry has no collectable candidate")
    if candidate["stage_key"] != expected_key or candidate["state"] != "terminal":
        raise _StageRetryBackendFailure("backend retry candidate identity is invalid")
    _candidate_history(candidate, previous=evaluation)

    accepted = acceptor.accept_evaluation(
        request={
            "study_result": current,
            "plan": cast(object, plan),
            "stage_input": stage_input,
            "candidate": candidate,
        }
    )["evaluation_result"]
    if accepted["status"] == "cancelled":
        # A cancelled physical retry is audit-visible but does not make the logical
        # evaluation cancelled; the pre-existing logical failure remains failed.
        accepted = copy.deepcopy(accepted)
        accepted["status"] = "failed"
    if accepted["status"] not in {"failed", "completed"}:
        raise ValueError("retry acceptance produced unsupported logical status")

    with coordinator.acquire(study_result_id=result_id):
        latest, latest_plan = _load_context(resolver, study_result_id=result_id)
        if latest_plan["id"] != plan["id"]:
            raise ValueError("lifecycle conflict: StudyResult Plan changed during retry")
        latest_slot = _target_slot(latest, trial=trial_id, coordinate=coordinate_id)
        latest_evaluation = resolver.resolve(
            kind=EntityKind.EVALUATION_RESULT,
            entity_id=expected_evaluation_id,
        )
        if latest_slot["disposition"] == "completed":
            validator.validate(
                kind=EntityKind.EVALUATION_RESULT,
                entity_id=expected_evaluation_id,
                document=latest_evaluation,
            )
            if latest_evaluation["status"] != "completed":
                raise ValueError("lifecycle conflict: completed retry slot has non-completed child")
            return latest
        if latest_slot["disposition"] != "failed":
            raise ValueError("lifecycle conflict: retry target changed during execution")
        if _canonical_json_bytes(latest_evaluation) != _canonical_json_bytes(evaluation):
            if _canonical_json_bytes(latest_evaluation) != _canonical_json_bytes(accepted):
                raise ValueError(
                    "lifecycle conflict: EvaluationResult changed concurrently during retry"
                )
        else:
            latest_evaluation = writer.replace_failed_evaluation_result(
                entity_id=expected_evaluation_id,
                replacement=accepted,
            )

        logical_status = cast(str, latest_evaluation["status"])
        replacement = _parent_after_retry(
            latest,
            trial=trial_id,
            coordinate=coordinate_id,
            disposition=logical_status,
        )
        stored = writer.replace_study_result_after_evaluation_retry(
            entity_id=result_id,
            trial=trial_id,
            coordinate=coordinate_id,
            replacement=replacement,
        )
        final = _validate_study_result(stored)

    _project_backend_summary_best_effort(
        backend=backend,
        resolver=resolver,
        result=final,
    )
    return final
