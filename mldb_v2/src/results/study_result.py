"""MLDB v2 canonical Study Result public shape and exact validation."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Literal, TypeAlias, TypedDict, cast

from mldb_v2.src.common.diagnostic import Diagnostic, _validate_diagnostic
from mldb_v2.src.common.ids import (
    EvaluationCoordinateId,
    EvaluationResultId,
    ExecutionKey,
    StudyId,
    StudyPlanId,
    StudyResultId,
    TrainingResultId,
    TrialId,
    _validate_canonical_json_value,
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
    _validate_typed_reference,
)

StudyResultStatus: TypeAlias = Literal[
    "submitted",
    "cancelling",
    "completed",
    "completed_with_failures",
    "failed",
    "cancelled",
]
StageDisposition: TypeAlias = Literal[
    "pending",
    "completed",
    "failed",
    "cancelled",
    "skipped",
]
SkipReason: TypeAlias = Literal[
    "upstream_failed",
    "upstream_cancelled",
    "study_cancelled",
    "global_failure",
]


class TrainingSlot(TypedDict):
    disposition: StageDisposition
    result: TrainingResultId | None
    reason: SkipReason | None


class EvaluationSlot(TypedDict):
    coordinate: EvaluationCoordinateId
    stage: str
    disposition: StageDisposition
    result: EvaluationResultId | None
    reason: SkipReason | None


class StudyResultTrial(TypedDict):
    trial: TrialId
    training: TrainingSlot | None
    evaluations: list[EvaluationSlot]


class StudyResult(TypedDict):
    schema: Literal["mjtensu.mldb-v2/study-result/v1"]
    id: StudyResultId
    execution_key: ExecutionKey
    plan: StudyPlanId
    study: StudyId
    source_commit: str
    backend: str
    created_at: str
    status: StudyResultStatus
    diagnostic: Diagnostic | None
    trials: list[StudyResultTrial]


_SCHEMA = "mjtensu.mldb-v2/study-result/v1"
_TOP_LEVEL_FIELDS = {
    "schema",
    "id",
    "execution_key",
    "plan",
    "study",
    "source_commit",
    "backend",
    "created_at",
    "status",
    "diagnostic",
    "trials",
}
_TRIAL_FIELDS = {"trial", "training", "evaluations"}
_TRAINING_SLOT_FIELDS = {"disposition", "result", "reason"}
_EVALUATION_SLOT_FIELDS = {"coordinate", "stage", "disposition", "result", "reason"}
_STATUSES = {
    "submitted",
    "cancelling",
    "completed",
    "completed_with_failures",
    "failed",
    "cancelled",
}
_DISPOSITIONS = {"pending", "completed", "failed", "cancelled", "skipped"}
_SKIP_REASONS = {
    "upstream_failed",
    "upstream_cancelled",
    "study_cancelled",
    "global_failure",
}
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
_EXECUTION_KEY_RE = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_RFC3339_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\Z",
    re.ASCII,
)


def _validate_created_at(value: object) -> str:
    if type(value) is not str or _RFC3339_UTC_RE.fullmatch(value) is None:
        raise ValueError("StudyResult created_at must be RFC3339 UTC using Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("StudyResult created_at must be RFC3339 UTC using Z") from error
    return value


def _validate_execution_identity(value: object) -> ExecutionKey:
    if type(value) is not str or _EXECUTION_KEY_RE.fullmatch(value) is None:
        raise ValueError("invalid StudyResult execution_key")
    parsed = uuid.UUID(hex=value)
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise ValueError("StudyResult execution_key must encode UUID4")
    return ExecutionKey(value)


def _expected_training_result_id(study_result_id: str, trial_id: str) -> TrainingResultId:
    return TrainingResultId(_validate_typed_reference(f"{study_result_id}-{trial_id}-train"))


def _expected_evaluation_result_id(
    study_result_id: str,
    trial_id: str,
    coordinate_id: str,
) -> EvaluationResultId:
    return EvaluationResultId(
        _validate_typed_reference(f"{study_result_id}-{trial_id}-{coordinate_id}")
    )


def _validate_slot_payload(
    *,
    disposition: object,
    result: object,
    reason: object,
    expected_result: str,
) -> tuple[StageDisposition, str | None, SkipReason | None]:
    if type(disposition) is not str or disposition not in _DISPOSITIONS:
        raise ValueError("invalid StudyResult stage disposition")
    if disposition == "pending":
        if result is not None or reason is not None:
            raise ValueError("pending StudyResult slot requires null result and reason")
        return cast(StageDisposition, disposition), None, None
    if disposition in {"completed", "failed", "cancelled"}:
        if reason is not None:
            raise ValueError("terminal attempted StudyResult slot reason must be null")
        validated_result = _validate_typed_reference(result)
        if validated_result != expected_result:
            raise ValueError("StudyResult slot result does not match deterministic child identity")
        return cast(StageDisposition, disposition), validated_result, None
    if result is not None:
        raise ValueError("skipped StudyResult slot result must be null")
    if type(reason) is not str or reason not in _SKIP_REASONS:
        raise ValueError("skipped StudyResult slot requires a stable reason")
    return cast(StageDisposition, disposition), None, cast(SkipReason, reason)


def _validate_training_slot(
    value: object,
    *,
    study_result_id: str,
    trial_id: str,
) -> TrainingSlot:
    if type(value) is not dict or set(value) != _TRAINING_SLOT_FIELDS:
        raise ValueError("StudyResult training slot fields do not match schema")
    disposition, result, reason = _validate_slot_payload(
        disposition=value["disposition"],
        result=value["result"],
        reason=value["reason"],
        expected_result=str(_expected_training_result_id(study_result_id, trial_id)),
    )
    return {
        "disposition": disposition,
        "result": None if result is None else TrainingResultId(result),
        "reason": reason,
    }


def _validate_evaluation_slot(
    value: object,
    *,
    study_result_id: str,
    trial_id: str,
    expected_index: int,
) -> EvaluationSlot:
    if type(value) is not dict or set(value) != _EVALUATION_SLOT_FIELDS:
        raise ValueError("StudyResult evaluation slot fields do not match schema")
    coordinate = _validate_evaluation_coordinate_id(value["coordinate"])
    if coordinate != f"eval-{expected_index:04d}":
        raise ValueError("StudyResult evaluation order does not match canonical sequence")
    stage = value["stage"]
    if type(stage) is not str or not stage:
        raise ValueError("StudyResult evaluation stage must be non-empty")
    disposition, result, reason = _validate_slot_payload(
        disposition=value["disposition"],
        result=value["result"],
        reason=value["reason"],
        expected_result=str(
            _expected_evaluation_result_id(study_result_id, trial_id, str(coordinate))
        ),
    )
    return {
        "coordinate": coordinate,
        "stage": stage,
        "disposition": disposition,
        "result": None if result is None else EvaluationResultId(result),
        "reason": reason,
    }


def _validate_status_diagnostic(status: str, diagnostic: Diagnostic | None) -> None:
    if status in {"submitted", "completed", "completed_with_failures"} and diagnostic is not None:
        raise ValueError(f"{status} StudyResult diagnostic must be null")
    if status == "failed" and diagnostic is None:
        raise ValueError("failed StudyResult requires diagnostic")


def _validate_closure(status: str, trials: list[StudyResultTrial]) -> None:
    dispositions: list[str] = []
    for trial in trials:
        if trial["training"] is not None:
            dispositions.append(trial["training"]["disposition"])
        dispositions.extend(slot["disposition"] for slot in trial["evaluations"])
    if status in {"completed", "completed_with_failures", "failed", "cancelled"}:
        if any(disposition == "pending" for disposition in dispositions):
            raise ValueError("terminal StudyResult requires every planned stage to be terminal")
    if status == "completed" and any(disposition != "completed" for disposition in dispositions):
        raise ValueError("completed StudyResult requires every planned stage to be completed")
    if status == "completed_with_failures" and all(
        disposition == "completed" for disposition in dispositions
    ):
        raise ValueError("completed_with_failures requires a non-completed planned stage")


def _validate_study_result(value: object) -> StudyResult:
    """Validate one exact schema-v1 StudyResult value without persistence or mutation."""
    if type(value) is not dict or set(value) != _TOP_LEVEL_FIELDS:
        raise ValueError("StudyResult fields do not match schema")
    _validate_canonical_json_value(value)
    if value["schema"] != _SCHEMA:
        raise ValueError("unsupported StudyResult schema")

    study_result_id = _validate_typed_reference(value["id"])
    execution_key = _validate_execution_identity(value["execution_key"])
    namespace, local_id = study_result_id.split("/", 1)
    if local_id != f"run-{execution_key}":
        raise ValueError("StudyResult id does not match execution_key")

    plan = _validate_typed_reference(value["plan"])
    study = _validate_typed_reference(value["study"])
    if plan.split("/", 1)[0] != namespace or study.split("/", 1)[0] != namespace:
        raise ValueError("StudyResult, Study, and Plan namespaces must match")

    source_commit = value["source_commit"]
    if type(source_commit) is not str or _COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError("StudyResult source_commit must be a full Git object id")
    backend = value["backend"]
    if type(backend) is not str or not backend:
        raise ValueError("StudyResult backend must be a non-empty generic type name")
    created_at = _validate_created_at(value["created_at"])

    status = value["status"]
    if type(status) is not str or status not in _STATUSES:
        raise ValueError("invalid StudyResult status")
    diagnostic = _validate_diagnostic(value["diagnostic"])
    _validate_status_diagnostic(status, diagnostic)

    raw_trials = value["trials"]
    if type(raw_trials) is not list or not raw_trials:
        raise ValueError("StudyResult trials must be a non-empty list")
    trials: list[StudyResultTrial] = []
    for index, raw_trial in enumerate(raw_trials, start=1):
        if type(raw_trial) is not dict or set(raw_trial) != _TRIAL_FIELDS:
            raise ValueError("StudyResult trial fields do not match schema")
        trial_id = _validate_trial_id(raw_trial["trial"])
        if trial_id != f"trial-{index:04d}":
            raise ValueError("StudyResult trial order does not match canonical sequence")
        raw_training = raw_trial["training"]
        training = (
            None
            if raw_training is None
            else _validate_training_slot(
                raw_training,
                study_result_id=study_result_id,
                trial_id=str(trial_id),
            )
        )
        raw_evaluations = raw_trial["evaluations"]
        if type(raw_evaluations) is not list or not raw_evaluations:
            raise ValueError("StudyResult evaluations must be a non-empty list")
        evaluations = [
            _validate_evaluation_slot(
                raw_slot,
                study_result_id=study_result_id,
                trial_id=str(trial_id),
                expected_index=evaluation_index,
            )
            for evaluation_index, raw_slot in enumerate(raw_evaluations, start=1)
        ]
        trials.append(
            {
                "trial": trial_id,
                "training": training,
                "evaluations": evaluations,
            }
        )

    _validate_closure(status, trials)
    return cast(
        StudyResult,
        {
            "schema": _SCHEMA,
            "id": StudyResultId(study_result_id),
            "execution_key": execution_key,
            "plan": StudyPlanId(plan),
            "study": StudyId(study),
            "source_commit": source_commit,
            "backend": backend,
            "created_at": created_at,
            "status": status,
            "diagnostic": diagnostic,
            "trials": trials,
        },
    )
