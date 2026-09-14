"""Immutable TrainingResult public shape and validation."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, TypedDict, cast

from mldb_v2.src.common.diagnostic import Diagnostic, _validate_diagnostic
from mldb_v2.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EntityKind,
    ModelId,
    StudyPlanId,
    StudyResultId,
    TaskId,
    TrainProtocolId,
    TrainingResultId,
    TrialId,
    _validate_canonical_json_value,
    _validate_trial_id,
    _validate_typed_reference,
)
from mldb_v2.src.common.parameters import (
    ResolvedPublicParameters,
    _validate_public_parameter_value,
    _validate_training_seed,
)
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.training.canonical_weights import (
    CanonicalWeightsArtifactRef,
    _validate_canonical_weights_artifact_ref,
)


class TrainingResultPayload(TypedDict):
    weights: CanonicalWeightsArtifactRef
    model: ModelId


class TrainingResult(TypedDict):
    schema: Literal["mjtensu.mldb-v2/training-result/v1"]
    id: TrainingResultId
    study_result: StudyResultId
    plan: StudyPlanId
    trial: TrialId
    task: TaskId
    architecture: ArchitectureId
    corpus: CorpusId
    train_protocol: TrainProtocolId
    parameters: ResolvedPublicParameters
    seed: int
    source_commit: str
    attempts: list[dict[str, object]]
    status: Literal["completed", "failed", "cancelled"]
    diagnostic: Diagnostic | None
    result: TrainingResultPayload | None


_SCHEMA = "mjtensu.mldb-v2/training-result/v1"
_FIELDS = {
    "schema",
    "id",
    "study_result",
    "plan",
    "trial",
    "task",
    "architecture",
    "corpus",
    "train_protocol",
    "parameters",
    "seed",
    "source_commit",
    "attempts",
    "status",
    "diagnostic",
    "result",
}
_ATTEMPT_FIELDS = {
    "backend", "execution_id", "status", "started_at", "ended_at", "diagnostic"
}
_RESULT_FIELDS = {"weights", "model"}
_COMMIT_RE  = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
_RFC3339_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\Z",
    re.ASCII,
)
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _parse_utc_timestamp(value: object, *, label: str) -> datetime | None:
    if value is None:
        return None
    if type(value) is not str or _RFC3339_UTC_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be RFC3339 UTC using Z")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be RFC3339 UTC using Z") from error


def _validate_attempt_summary(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _ATTEMPT_FIELDS:
        raise ValueError("attempt summary fields do not match schema")
    backend = value["backend"]
    execution_id = value["execution_id"]
    status = value["status"]
    if type(backend) is not str or not backend:
        raise ValueError("attempt backend must be a non-empty string")
    if type(execution_id) is not str or not execution_id:
        raise ValueError("attempt execution_id must be a non-empty string")
    if type(status) is not str or status not in _TERMINAL_STATUSES:
        raise ValueError("invalid attempt status")
    started = _parse_utc_timestamp(value["started_at"], label="attempt started_at")
    ended = _parse_utc_timestamp(value["ended_at"], label="attempt ended_at")
    if started is not None and ended is not None and ended < started:
        raise ValueError("attempt ended_at precedes started_at")
    diagnostic = _validate_diagnostic(value["diagnostic"])
    if status == "completed" and diagnostic is not None:
        raise ValueError("completed attempt diagnostic must be null")
    if status != "completed" and diagnostic is None:
        raise ValueError("non-completed attempt requires diagnostic")
    return value


def _validate_training_result(
    value: object,
    *,
    expected_id: str | None = None,
) -> TrainingResult:
    if type(value) is not dict or set(value) != _FIELDS:
        raise ValueError("TrainingResult fields do not match schema")
    _validate_canonical_json_value(value)
    if value["schema"] != _SCHEMA:
        raise ValueError("unsupported TrainingResult schema")

    training_result_id = _validate_typed_reference(value["id"])
    if expected_id is not None and training_result_id != expected_id:
        raise ValueError("TrainingResult id does not match canonical path identity")
    study_result = _validate_typed_reference(value["study_result"])
    plan = _validate_typed_reference(value["plan"])
    trial = _validate_trial_id(value["trial"])
    task = _validate_typed_reference(value["task"])
    architecture = _validate_typed_reference(value["architecture"])
    corpus = _validate_typed_reference(value["corpus"])
    train_protocol = _validate_typed_reference(value["train_protocol"])

    parameters = value["parameters"]
    if type(parameters) is not dict:
        raise ValueError("TrainingResult parameters must be a mapping")
    for key, item in parameters.items():
        if type(key) is not str:
            raise ValueError("TrainingResult parameter keys must be strings")
        _validate_public_parameter_value(item)

    seed = _validate_training_seed(value["seed"])
    source_commit = value["source_commit"]
    if type(source_commit) is not str or _COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError("TrainingResult source_commit must be a full Git object id")

    attempts = value["attempts"]
    if type(attempts) is not list or not attempts:
        raise ValueError("TrainingResult attempts must be a non-empty list")
    for attempt in attempts:
        _validate_attempt_summary(attempt)

    status = value["status"]
    if type(status) is not str or status not in _TERMINAL_STATUSES:
        raise ValueError("invalid TrainingResult status")
    diagnostic = _validate_diagnostic(value["diagnostic"])
    result = value["result"]

    if status == "completed":
        if diagnostic is not None:
            raise ValueError("completed TrainingResult diagnostic must be null")
        if type(result) is not dict or set(result) != _RESULT_FIELDS:
            raise ValueError("completed TrainingResult requires exact result payload")
        weights = _validate_canonical_weights_artifact_ref(result["weights"])
        model = _validate_typed_reference(result["model"])
        parsed_result: TrainingResultPayload | None = {
            "weights": weights,
            "model": ModelId(model),
        }
    else:
        if diagnostic is None:
            raise ValueError("failed/cancelled TrainingResult requires diagnostic")
        if result is not None:
            raise ValueError("failed/cancelled TrainingResult result must be null")
        parsed_result = None

    parsed: TrainingResult = {
        "schema": _SCHEMA,
        "id": TrainingResultId(training_result_id),
        "study_result": StudyResultId(study_result),
        "plan": StudyPlanId(plan),
        "trial": TrialId(trial),
        "task": TaskId(task),
        "architecture": ArchitectureId(architecture),
        "corpus": CorpusId(corpus),
        "train_protocol": TrainProtocolId(train_protocol),
        "parameters": parameters,
        "seed": seed,
        "source_commit": source_commit,
        "attempts": attempts,
        "status": status,  # type: ignore[typeddict-item]
        "diagnostic": diagnostic,
        "result": parsed_result,
    }
    return cast(TrainingResult, parsed)


def _load_training_result(
    resolver: CanonicalRepositoryResolver,
    training_result_id: TrainingResultId | str,
) -> TrainingResult:
    expected_id = _validate_typed_reference(training_result_id)
    document = resolver.resolve(
        kind=EntityKind.TRAINING_RESULT,
        entity_id=TrainingResultId(expected_id),
    )
    return _validate_training_result(document, expected_id=expected_id)
