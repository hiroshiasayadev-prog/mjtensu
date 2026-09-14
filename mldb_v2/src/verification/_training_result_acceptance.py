"""Persistence-free acceptance for one terminal training candidate."""

from __future__ import annotations

import copy
import re
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

from mldb_v2.src.backend.candidate_outcome import (
    CompletedTrainingCandidate,
    FailedTrainingCandidate,
)
from mldb_v2.src.backend.stage_input import TrainingStageInput
from mldb_v2.src.common.diagnostic import Diagnostic, _validate_diagnostic
from mldb_v2.src.common.ids import (
    ModelId,
    StudyResultId,
    TrainingResultId,
    TrialId,
    _canonical_json_bytes,
    _validate_trial_id,
    _validate_typed_reference,
)
from mldb_v2.src.study._plan_build import _validate_study_plan
from mldb_v2.src.study.plan import StudyPlan
from mldb_v2.src.training.canonical_weights import (
    CanonicalWeightsArtifactRef,
    _load_canonical_state_dict_bytes,
    _load_state_into_fresh_architecture,
    _validate_canonical_weights_artifact_ref,
)
from mldb_v2.src.training.model import Model, _validate_model
from mldb_v2.src.training.training_result import (
    TrainingResult,
    _validate_attempt_summary,
    _validate_training_result,
)

TrainingTerminalCandidate = CompletedTrainingCandidate | FailedTrainingCandidate


class _TrainingAcceptanceResult(TypedDict):
    training_result: TrainingResult
    model: Model | None


_STUDY_RESULT_FIELDS = {
    "schema", "id", "execution_key", "plan", "study", "source_commit", "backend",
    "created_at", "status", "diagnostic", "trials",
}
_STAGE_INPUT_FIELDS = {
    "schema", "study_result", "plan", "plan_sha256", "trial", "kind", "coordinate",
    "source_commit", "pins", "stage", "runtime_model",
}
_TRAINING_STAGE_FIELDS = {
    "task", "corpus", "architecture", "train_protocol", "parameters", "seed",
}
_CANDIDATE_FIELDS = {"state", "stage_key", "attempts", "status", "diagnostic", "result"}
_STAGE_KEY_FIELDS = {"study_result", "plan", "trial", "kind", "coordinate", "source_commit"}
_STUDY_RESULT_SCHEMA = "mjtensu.mldb-v2/study-result/v1"
_STAGE_INPUT_SCHEMA = "mjtensu.mldb-v2/stage-input/v1"
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
_EXECUTION_KEY_RE = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_RFC3339_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\Z",
    re.ASCII,
)
_STUDY_STATUSES = {
    "submitted", "cancelling", "completed", "completed_with_failures", "failed", "cancelled",
}
_CANDIDATE_STATUSES = {"completed", "failed", "cancelled"}


def _acceptance_diagnostic() -> Diagnostic:
    return {
        "code": "training_acceptance_failed",
        "message": "Training candidate failed canonical acceptance.",
    }


def _parse_created_at(value: object) -> None:
    if type(value) is not str or _RFC3339_UTC_RE.fullmatch(value) is None:
        raise ValueError("StudyResult created_at must be RFC3339 UTC using Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("StudyResult created_at must be RFC3339 UTC using Z") from error


def _validate_study_result_anchor(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _STUDY_RESULT_FIELDS:
        raise ValueError("StudyResult fields do not match schema")
    if value["schema"] != _STUDY_RESULT_SCHEMA:
        raise ValueError("unsupported StudyResult schema")

    study_result_id = _validate_typed_reference(value["id"])
    execution_key = value["execution_key"]
    if type(execution_key) is not str or _EXECUTION_KEY_RE.fullmatch(execution_key) is None:
        raise ValueError("invalid StudyResult execution_key")
    parsed = uuid.UUID(hex=execution_key)
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise ValueError("StudyResult execution_key must encode UUID4")
    namespace, local_id = study_result_id.split("/", 1)
    if local_id != f"run-{execution_key}":
        raise ValueError("StudyResult id does not match execution_key")

    plan_id = _validate_typed_reference(value["plan"])
    study_id = _validate_typed_reference(value["study"])
    if plan_id.split("/", 1)[0] != namespace or study_id.split("/", 1)[0] != namespace:
        raise ValueError("StudyResult, Study, and Plan namespaces must match")
    source_commit = value["source_commit"]
    if type(source_commit) is not str or _COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError("StudyResult source_commit must be a full Git object id")
    if type(value["backend"]) is not str or not value["backend"]:
        raise ValueError("StudyResult backend must be a non-empty string")
    _parse_created_at(value["created_at"])
    if value["status"] not in _STUDY_STATUSES:
        raise ValueError("invalid StudyResult status")
    _validate_diagnostic(value["diagnostic"])

    trials = value["trials"]
    if type(trials) is not list:
        raise ValueError("StudyResult trials must be a list")
    for index, trial in enumerate(trials, start=1):
        if type(trial) is not dict or set(trial) != {"trial", "training", "evaluations"}:
            raise ValueError("StudyResult trial fields do not match schema")
        if _validate_trial_id(trial["trial"]) != f"trial-{index:04d}":
            raise ValueError("StudyResult trial order does not match canonical sequence")
        if trial["training"] is not None and (
            type(trial["training"]) is not dict
            or set(trial["training"]) != {"disposition", "result", "reason"}
        ):
            raise ValueError("StudyResult training slot fields do not match schema")
        evaluations = trial["evaluations"]
        if type(evaluations) is not list:
            raise ValueError("StudyResult evaluations must be a list")
        for evaluation_index, slot in enumerate(evaluations, start=1):
            if type(slot) is not dict or set(slot) != {
                "coordinate", "stage", "disposition", "result", "reason"
            }:
                raise ValueError("StudyResult evaluation slot fields do not match schema")
            if slot["coordinate"] != f"eval-{evaluation_index:04d}":
                raise ValueError("StudyResult evaluation order does not match canonical sequence")
            if type(slot["stage"]) is not str or not slot["stage"]:
                raise ValueError("StudyResult evaluation stage must be non-empty")
    return value


def _validate_parent_plan_lineage(
    study_result: dict[str, object], plan: StudyPlan
) -> None:
    if study_result["plan"] != plan["id"]:
        raise ValueError("StudyResult plan does not match request Plan")
    if study_result["study"] != plan["study"]:
        raise ValueError("StudyResult study does not match request Plan")
    if study_result["source_commit"] != plan["source_commit"]:
        raise ValueError("StudyResult source_commit does not match request Plan")

    result_trials = cast(list[dict[str, object]], study_result["trials"])
    if len(result_trials) != len(plan["trials"]):
        raise ValueError("StudyResult trial topology does not match Plan")
    for result_trial, plan_trial in zip(result_trials, plan["trials"]):
        if result_trial["trial"] != plan_trial["trial"]:
            raise ValueError("StudyResult trial identity does not match Plan")
        expects_training = plan_trial["source"]["kind"] == "training"
        if (result_trial["training"] is not None) != expects_training:
            raise ValueError("StudyResult training topology does not match Plan")
        result_evaluations = cast(list[dict[str, object]], result_trial["evaluations"])
        if len(result_evaluations) != len(plan_trial["evaluations"]):
            raise ValueError("StudyResult evaluation topology does not match Plan")
        for result_evaluation, plan_evaluation in zip(
            result_evaluations, plan_trial["evaluations"]
        ):
            if result_evaluation["coordinate"] != plan_evaluation["coordinate"]:
                raise ValueError("StudyResult evaluation coordinate does not match Plan")
            if result_evaluation["stage"] != plan_evaluation["stage"]:
                raise ValueError("StudyResult evaluation stage does not match Plan")


def _training_source(plan: StudyPlan, trial_id: TrialId) -> dict[str, object]:
    for trial in plan["trials"]:
        if trial["trial"] == trial_id:
            source = trial["source"]
            if source["kind"] != "training":
                raise ValueError("requested Plan trial is not a training-source trial")
            return cast(dict[str, object], source)
    raise ValueError("TrainingStageInput trial does not exist in Plan")


def _derive_child_ids(
    study_result_id: str, trial_id: TrialId
) -> tuple[TrainingResultId, ModelId]:
    parent = _validate_typed_reference(study_result_id)
    trial = _validate_trial_id(trial_id)
    training_result_id = _validate_typed_reference(f"{parent}-{trial}-train")
    model_id = _validate_typed_reference(f"{parent}-{trial}-model")
    return TrainingResultId(training_result_id), ModelId(model_id)


def _validate_stage_input_lineage(
    stage_input: object,
    *,
    study_result: dict[str, object],
    plan: StudyPlan,
    trial_id: TrialId,
    source: dict[str, object],
) -> TrainingStageInput:
    if type(stage_input) is not dict or set(stage_input) != _STAGE_INPUT_FIELDS:
        raise ValueError("TrainingStageInput fields do not match schema")
    if stage_input["schema"] != _STAGE_INPUT_SCHEMA:
        raise ValueError("unsupported TrainingStageInput schema")
    if stage_input["study_result"] != study_result["id"]:
        raise ValueError("TrainingStageInput StudyResult mismatch")
    if stage_input["plan"] != plan["id"]:
        raise ValueError("TrainingStageInput Plan mismatch")
    if stage_input["plan_sha256"] != plan["content_sha256"]:
        raise ValueError("TrainingStageInput Plan digest mismatch")
    if stage_input["trial"] != trial_id:
        raise ValueError("TrainingStageInput trial mismatch")
    if stage_input["kind"] != "training" or stage_input["coordinate"] is not None:
        raise ValueError("TrainingStageInput is not the exact training variant")
    if stage_input["source_commit"] != plan["source_commit"]:
        raise ValueError("TrainingStageInput source_commit mismatch")
    if _canonical_json_bytes(stage_input["pins"]) != _canonical_json_bytes(plan["pins"]):
        raise ValueError("TrainingStageInput pins do not exactly match Plan")
    if stage_input["runtime_model"] is not None:
        raise ValueError("training StageInput runtime_model must be null")

    expected_stage = {key: value for key, value in source.items() if key != "kind"}
    stage = stage_input["stage"]
    if type(stage) is not dict or set(stage) != _TRAINING_STAGE_FIELDS:
        raise ValueError("TrainingStage fields do not match schema")
    if _canonical_json_bytes(stage) != _canonical_json_bytes(expected_stage):
        raise ValueError("TrainingStage does not exactly match Plan trial source")
    return cast(TrainingStageInput, stage_input)


def _validated_attempts(candidate: object) -> list[dict[str, object]]:
    if type(candidate) is not dict or set(candidate) != _CANDIDATE_FIELDS:
        raise ValueError("training candidate fields do not match schema")
    if candidate["state"] != "terminal":
        raise ValueError("training candidate must be terminal")
    status = candidate["status"]
    if type(status) is not str or status not in _CANDIDATE_STATUSES:
        raise ValueError("invalid training candidate status")
    attempts = candidate["attempts"]
    if type(attempts) is not list or not attempts:
        raise ValueError("training candidate attempts must be a non-empty list")
    for attempt in attempts:
        _validate_attempt_summary(attempt)
    return copy.deepcopy(attempts)


def _validate_candidate_lineage(
    candidate: TrainingTerminalCandidate,
    *,
    stage_input: TrainingStageInput,
) -> CanonicalWeightsArtifactRef | None:
    stage_key = candidate["stage_key"]
    if type(stage_key) is not dict or set(stage_key) != _STAGE_KEY_FIELDS:
        raise ValueError("training candidate StageKey fields do not match schema")
    expected_key = {
        "study_result": stage_input["study_result"],
        "plan": stage_input["plan"],
        "trial": stage_input["trial"],
        "kind": "training",
        "coordinate": None,
        "source_commit": stage_input["source_commit"],
    }
    if stage_key != expected_key:
        raise ValueError("training candidate StageKey does not exactly match StageInput")

    status = candidate["status"]
    if status == "completed":
        if candidate["diagnostic"] is not None:
            raise ValueError("completed training candidate diagnostic must be null")
        result = candidate["result"]
        if type(result) is not dict or set(result) != {"weights"}:
            raise ValueError("completed training candidate result fields do not match schema")
        return _validate_canonical_weights_artifact_ref(result["weights"])

    diagnostic = _validate_diagnostic(candidate["diagnostic"])
    if diagnostic is None:
        raise ValueError("failed/cancelled training candidate requires diagnostic")
    if candidate["result"] is not None:
        raise ValueError("failed/cancelled training candidate result must be null")
    return None


def _formal_training_result(
    *,
    study_result: dict[str, object],
    plan: StudyPlan,
    trial_id: TrialId,
    source: dict[str, object],
    training_result_id: TrainingResultId,
    model_id: ModelId,
    attempts: list[dict[str, object]],
    status: Literal["completed", "failed", "cancelled"],
    diagnostic: Diagnostic | None,
    weights: CanonicalWeightsArtifactRef | None,
) -> TrainingResult:
    payload = None
    if status == "completed":
        if weights is None:
            raise ValueError("completed formal result requires accepted weights")
        payload = {"weights": weights, "model": model_id}
    value = {
        "schema": "mjtensu.mldb-v2/training-result/v1",
        "id": training_result_id,
        "study_result": study_result["id"],
        "plan": plan["id"],
        "trial": trial_id,
        "task": source["task"],
        "architecture": source["architecture"],
        "corpus": source["corpus"],
        "train_protocol": source["train_protocol"],
        "parameters": copy.deepcopy(source["parameters"]),
        "seed": source["seed"],
        "source_commit": plan["source_commit"],
        "attempts": copy.deepcopy(attempts),
        "status": status,
        "diagnostic": copy.deepcopy(diagnostic),
        "result": copy.deepcopy(payload),
    }
    return _validate_training_result(value, expected_id=str(training_result_id))


def _formal_model(
    *, training_result_id: TrainingResultId, model_id: ModelId
) -> Model:
    return _validate_model(
        {
            "schema": "mjtensu.mldb-v2/model/v1",
            "id": model_id,
            "training_result": training_result_id,
        },
        expected_id=str(model_id),
    )


def _validate_relevant_plan_pins(
    *, plan: StudyPlan, source: dict[str, object]
) -> None:
    expected = {
        "task": source["task"],
        "corpus": source["corpus"],
        "architecture": source["architecture"],
        "train_protocol": source["train_protocol"],
    }
    for kind, entity_id in expected.items():
        matches = [
            pin for pin in plan["pins"]
            if pin["kind"] == kind and pin["id"] == entity_id
        ]
        if len(matches) != 1:
            raise ValueError(f"Plan does not contain exact {kind} pin")


from mldb_v2.src.storage.object_bytes import _ObjectByteAccess


def _validated_candidate_attempts(
    candidate: object, *, expected_backend: str
) -> list[dict[str, object]]:
    attempts = _validated_attempts(candidate)
    status = cast(dict[str, object], candidate)["status"]
    for attempt in attempts:
        if attempt["backend"] != expected_backend:
            raise ValueError("training candidate attempt backend does not match StudyResult backend")
    if attempts[-1]["status"] != status:
        raise ValueError("last training attempt status does not match candidate status")
    return attempts


def _accept_training_candidate(
    *,
    study_result: object,
    plan: Mapping[str, object],
    stage_input: object,
    candidate: object,
    mldb_data_root: str | Path,
    object_bytes: _ObjectByteAccess,
) -> _TrainingAcceptanceResult:
    """Accept one terminal training candidate without canonical persistence."""
    validated_plan = _validate_study_plan(plan)
    validated_study_result = _validate_study_result_anchor(study_result)
    _validate_parent_plan_lineage(validated_study_result, validated_plan)

    if type(stage_input) is not dict:
        raise ValueError("TrainingStageInput must be a mapping")
    trial_id = _validate_trial_id(stage_input.get("trial"))
    source = _training_source(validated_plan, trial_id)
    _validate_relevant_plan_pins(plan=validated_plan, source=source)
    validated_stage_input = _validate_stage_input_lineage(
        stage_input,
        study_result=validated_study_result,
        plan=validated_plan,
        trial_id=trial_id,
        source=source,
    )
    training_result_id, model_id = _derive_child_ids(
        str(validated_study_result["id"]), trial_id
    )
    attempts = _validated_candidate_attempts(
        candidate,
        expected_backend=cast(str, validated_study_result["backend"]),
    )
    typed_candidate = cast(TrainingTerminalCandidate, candidate)

    if typed_candidate["status"] in {"failed", "cancelled"}:
        _validate_candidate_lineage(
            typed_candidate,
            stage_input=validated_stage_input,
        )
        diagnostic = _validate_diagnostic(typed_candidate["diagnostic"])
        if diagnostic is None:
            raise ValueError("terminal failed/cancelled candidate requires diagnostic")
        training_result = _formal_training_result(
            study_result=validated_study_result,
            plan=validated_plan,
            trial_id=trial_id,
            source=source,
            training_result_id=training_result_id,
            model_id=model_id,
            attempts=attempts,
            status=cast(Literal["failed", "cancelled"], typed_candidate["status"]),
            diagnostic=diagnostic,
            weights=None,
        )
        return {"training_result": training_result, "model": None}

    try:
        weights = _validate_candidate_lineage(
            typed_candidate,
            stage_input=validated_stage_input,
        )
        if weights is None:
            raise ValueError("completed candidate did not provide weights")
        data = object_bytes.read_verified(weights)
        state = _load_canonical_state_dict_bytes(data, ref=weights)
        _load_state_into_fresh_architecture(
            mldb_data_root,
            cast(str, source["architecture"]),
            state,
        )
    except (ValueError, OSError, KeyError):
        training_result = _formal_training_result(
            study_result=validated_study_result,
            plan=validated_plan,
            trial_id=trial_id,
            source=source,
            training_result_id=training_result_id,
            model_id=model_id,
            attempts=attempts,
            status="failed",
            diagnostic=_acceptance_diagnostic(),
            weights=None,
        )
        return {"training_result": training_result, "model": None}

    accepted_weights = copy.deepcopy(weights)
    training_result = _formal_training_result(
        study_result=validated_study_result,
        plan=validated_plan,
        trial_id=trial_id,
        source=source,
        training_result_id=training_result_id,
        model_id=model_id,
        attempts=attempts,
        status="completed",
        diagnostic=None,
        weights=accepted_weights,
    )
    model = _formal_model(
        training_result_id=training_result_id,
        model_id=model_id,
    )
    return {"training_result": training_result, "model": model}
