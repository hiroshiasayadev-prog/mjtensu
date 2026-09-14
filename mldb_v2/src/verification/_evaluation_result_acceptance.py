"""Persistence-free acceptance for one terminal Evaluation candidate."""

from __future__ import annotations

import copy
import hashlib
import math
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping, TypedDict, cast

from mldb_v2.src.backend.candidate_outcome import (
    CompletedEvaluationCandidate,
    FailedEvaluationCandidate,
)
from mldb_v2.src.backend.stage_input import EvaluationStageInput
from mldb_v2.src.catalog._executable_definition_loading import _resolve_document
from mldb_v2.src.common.diagnostic import Diagnostic, _validate_diagnostic
from mldb_v2.src.common.ids import (
    EntityKind,
    EvaluationResultId,
    TrialId,
    _canonical_json_bytes,
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
    _validate_typed_reference,
)
from mldb_v2.src.evaluation.evaluation_protocol import (
    EvaluationProtocol,
    _load_evaluation_protocol_definition,
)
from mldb_v2.src.evaluation.evaluation_result import (
    EvaluationArtifactRef,
    EvaluationResult,
)
from mldb_v2.src.storage.artifact_reference import _validate_artifact_ref
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study._plan_build import _validate_study_plan
from mldb_v2.src.study.plan import StudyPlan
from mldb_v2.src.training.canonical_weights import (
    _validate_canonical_weights_artifact_ref,
)
from mldb_v2.src.training.model import _resolve_model_lineage
from mldb_v2.src.training.training_result import _validate_attempt_summary


EvaluationTerminalCandidate = CompletedEvaluationCandidate | FailedEvaluationCandidate


class _EvaluationAcceptanceResult(TypedDict):
    evaluation_result: EvaluationResult


_REQUEST_FIELDS = {"study_result", "plan", "stage_input", "candidate"}
_STUDY_RESULT_FIELDS = {
    "schema", "id", "execution_key", "plan", "study", "source_commit", "backend",
    "created_at", "status", "diagnostic", "trials",
}
_STAGE_INPUT_FIELDS = {
    "schema", "study_result", "plan", "plan_sha256", "trial", "kind", "coordinate",
    "source_commit", "pins", "stage", "runtime_model",
}
_STAGE_FIELDS = {"name", "task", "corpus", "evaluation_protocol", "parameters"}
_RUNTIME_MODEL_FIELDS = {"model", "training_result", "task", "architecture", "weights"}
_CANDIDATE_FIELDS = {"state", "stage_key", "attempts", "status", "diagnostic", "result"}
_STAGE_KEY_FIELDS = {"study_result", "plan", "trial", "kind", "coordinate", "source_commit"}
_COMPLETED_RESULT_FIELDS = {"metrics", "artifacts"}
_EVALUATION_ARTIFACT_FIELDS = {"uri", "bytes", "sha256", "format", "schema"}
_EXECUTION_KEY_RE = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
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
        "code": "evaluation_acceptance_failed",
        "message": "Evaluation candidate failed canonical acceptance.",
    }


def _same_canonical_value(left: object, right: object) -> bool:
    try:
        return _canonical_json_bytes(left) == _canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


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
    if value["schema"] != "mjtensu.mldb-v2/study-result/v1":
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
    if type(value["trials"]) is not list:
        raise ValueError("StudyResult trials must be a list")
    return value


def _validate_parent_plan_lineage(study_result: dict[str, object], plan: StudyPlan) -> None:
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
        if type(result_trial) is not dict or set(result_trial) != {"trial", "training", "evaluations"}:
            raise ValueError("StudyResult trial fields do not match schema")
        if result_trial["trial"] != plan_trial["trial"]:
            raise ValueError("StudyResult trial identity does not match Plan")
        result_evaluations = result_trial["evaluations"]
        if type(result_evaluations) is not list or len(result_evaluations) != len(plan_trial["evaluations"]):
            raise ValueError("StudyResult evaluation topology does not match Plan")
        for result_slot, plan_coordinate in zip(result_evaluations, plan_trial["evaluations"]):
            if type(result_slot) is not dict or set(result_slot) != {
                "coordinate", "stage", "disposition", "result", "reason"
            }:
                raise ValueError("StudyResult evaluation slot fields do not match schema")
            if result_slot["coordinate"] != plan_coordinate["coordinate"]:
                raise ValueError("StudyResult evaluation coordinate does not match Plan")
            if result_slot["stage"] != plan_coordinate["stage"]:
                raise ValueError("StudyResult evaluation stage does not match Plan")


def _select_target(plan: StudyPlan, stage_input: object) -> tuple[TrialId, dict[str, object], dict[str, object]]:
    if type(stage_input) is not dict:
        raise ValueError("EvaluationStageInput must be a mapping")
    trial_id = _validate_trial_id(stage_input.get("trial"))
    coordinate_id = _validate_evaluation_coordinate_id(stage_input.get("coordinate"))
    for trial in plan["trials"]:
        if trial["trial"] != trial_id:
            continue
        for coordinate in trial["evaluations"]:
            if coordinate["coordinate"] == coordinate_id:
                return TrialId(trial_id), cast(dict[str, object], trial), cast(dict[str, object], coordinate)
        raise ValueError("EvaluationStageInput coordinate does not exist in Plan trial")
    raise ValueError("EvaluationStageInput trial does not exist in Plan")


def _expected_model_id(study_result_id: str, trial: dict[str, object]) -> str:
    source = cast(dict[str, object], trial["source"])
    if source["kind"] == "existing_model":
        return _validate_typed_reference(source["model"])
    if source["kind"] == "training":
        return _validate_typed_reference(f"{study_result_id}-{trial['trial']}-model")
    raise ValueError("unsupported Plan trial source")


def _result_id(study_result_id: str, trial: str, coordinate: str) -> EvaluationResultId:
    return EvaluationResultId(_validate_typed_reference(f"{study_result_id}-{trial}-{coordinate}"))


def _validate_stage_input(
    stage_input: object,
    *,
    study_result: dict[str, object],
    plan: StudyPlan,
    trial: dict[str, object],
    coordinate: dict[str, object],
    mldb_data_root: str | Path,
) -> EvaluationStageInput:
    if type(stage_input) is not dict or set(stage_input) != _STAGE_INPUT_FIELDS:
        raise ValueError("EvaluationStageInput fields do not match schema")
    if stage_input["schema"] != "mjtensu.mldb-v2/stage-input/v1":
        raise ValueError("unsupported EvaluationStageInput schema")
    if stage_input["study_result"] != study_result["id"]:
        raise ValueError("EvaluationStageInput StudyResult mismatch")
    if stage_input["plan"] != plan["id"]:
        raise ValueError("EvaluationStageInput Plan mismatch")
    if stage_input["plan_sha256"] != plan["content_sha256"]:
        raise ValueError("EvaluationStageInput Plan digest mismatch")
    if stage_input["trial"] != trial["trial"]:
        raise ValueError("EvaluationStageInput trial mismatch")
    if stage_input["kind"] != "evaluation":
        raise ValueError("EvaluationStageInput kind must be evaluation")
    if stage_input["coordinate"] != coordinate["coordinate"]:
        raise ValueError("EvaluationStageInput coordinate mismatch")
    if stage_input["source_commit"] != plan["source_commit"]:
        raise ValueError("EvaluationStageInput source_commit mismatch")
    if not _same_canonical_value(stage_input["pins"], plan["pins"]):
        raise ValueError("EvaluationStageInput pins do not exactly match Plan")

    stage = stage_input["stage"]
    if type(stage) is not dict or set(stage) != _STAGE_FIELDS:
        raise ValueError("EvaluationStage fields do not match schema")
    expected_stage = {
        "name": coordinate["stage"],
        "task": coordinate["task"],
        "corpus": coordinate["corpus"],
        "evaluation_protocol": coordinate["evaluation_protocol"],
        "parameters": coordinate["parameters"],
    }
    if not _same_canonical_value(stage, expected_stage):
        raise ValueError("EvaluationStage does not exactly match Plan coordinate")

    runtime_model = stage_input["runtime_model"]
    if type(runtime_model) is not dict or set(runtime_model) != _RUNTIME_MODEL_FIELDS:
        raise ValueError("EvaluationStageInput runtime_model fields do not match schema")
    expected_model = _expected_model_id(str(study_result["id"]), trial)
    if runtime_model["model"] != expected_model:
        raise ValueError("runtime Model id does not match Plan lineage")
    _validate_typed_reference(runtime_model["training_result"])
    _validate_typed_reference(runtime_model["task"])
    _validate_typed_reference(runtime_model["architecture"])
    _validate_canonical_weights_artifact_ref(runtime_model["weights"])

    lineage = _resolve_model_lineage(mldb_data_root, expected_model)
    expected_runtime = {
        "model": lineage.model["id"],
        "training_result": lineage.training_result["id"],
        "task": lineage.training_result["task"],
        "architecture": lineage.training_result["architecture"],
        "weights": lineage.weights,
    }
    if not _same_canonical_value(runtime_model, expected_runtime):
        raise ValueError("runtime Model lineage does not match canonical Model")
    if runtime_model["task"] != coordinate["task"]:
        raise ValueError("runtime Model Task does not match Evaluation Task")
    return cast(EvaluationStageInput, stage_input)


def _validate_protocol_pin(
    *,
    mldb_data_root: str | Path,
    plan: StudyPlan,
    protocol_id: str,
) -> EvaluationProtocol:
    matching = [
        pin for pin in plan["pins"]
        if pin["kind"] == "evaluation_protocol" and pin["id"] == protocol_id
    ]
    if len(matching) != 1:
        raise ValueError("Plan does not contain exact EvaluationProtocol pin")
    pin = matching[0]
    protocol = _load_evaluation_protocol_definition(mldb_data_root, protocol_id)
    if protocol["status"] != "sealed":
        raise ValueError("EvaluationProtocol is not sealed")
    if protocol["task"] is None:
        raise ValueError("EvaluationProtocol task is invalid")

    _document, yaml_path = _resolve_document(
        mldb_data_root,
        kind=EntityKind.EVALUATION_PROTOCOL,
        entity_id=protocol_id,
    )
    if hashlib.sha256(yaml_path.read_bytes()).hexdigest() != pin["yaml_sha256"]:
        raise ValueError("EvaluationProtocol YAML does not match Plan pin")
    companion = yaml_path.with_suffix(".py")
    if not companion.is_file():
        raise ValueError("EvaluationProtocol companion is missing")
    companion_sha = hashlib.sha256(companion.read_bytes()).hexdigest()
    if companion_sha != pin["companion_sha256"]:
        raise ValueError("EvaluationProtocol companion does not match Plan pin")
    implementation = protocol["implementation"]
    if implementation.get("sha256") != companion_sha:
        raise ValueError("EvaluationProtocol implementation digest is inconsistent")
    if not _same_canonical_value(implementation.get("sources", []), pin["sources"]):
        raise ValueError("EvaluationProtocol sources do not match Plan pin")
    return protocol


def _validate_candidate_common(
    candidate: object,
    *,
    expected_backend: str,
) -> tuple[str, list[dict[str, object]]]:
    if type(candidate) is not dict or set(candidate) != _CANDIDATE_FIELDS:
        raise ValueError("evaluation candidate fields do not match schema")
    if candidate["state"] != "terminal":
        raise ValueError("evaluation candidate must be terminal")
    status = candidate["status"]
    if type(status) is not str or status not in _CANDIDATE_STATUSES:
        raise ValueError("invalid evaluation candidate status")
    attempts = candidate["attempts"]
    if type(attempts) is not list or not attempts:
        raise ValueError("evaluation candidate attempts must be non-empty")
    validated: list[dict[str, object]] = []
    for raw in attempts:
        attempt = _validate_attempt_summary(raw)
        if attempt["backend"] != expected_backend:
            raise ValueError("candidate attempt backend does not match StudyResult backend")
        validated.append(attempt)
    if validated[-1]["status"] != status:
        raise ValueError("last attempt status does not match candidate status")
    return status, copy.deepcopy(validated)


def _validate_candidate_stage_key(
    candidate: EvaluationTerminalCandidate,
    *,
    stage_input: EvaluationStageInput,
) -> None:
    stage_key = candidate["stage_key"]
    if type(stage_key) is not dict or set(stage_key) != _STAGE_KEY_FIELDS:
        raise ValueError("evaluation candidate StageKey fields do not match schema")
    expected = {
        "study_result": stage_input["study_result"],
        "plan": stage_input["plan"],
        "trial": stage_input["trial"],
        "kind": "evaluation",
        "coordinate": stage_input["coordinate"],
        "source_commit": stage_input["source_commit"],
    }
    if stage_key != expected:
        raise ValueError("evaluation candidate StageKey does not exactly match StageInput")


def _validate_metric_value(value: object, declaration: Mapping[str, object]) -> int | float:
    metric_type = declaration["type"]
    if metric_type == "integer":
        if type(value) is not int:
            raise ValueError("integer metric must be an exact integer")
        return value
    if metric_type == "number":
        if type(value) is int:
            return value
        if type(value) is not float:
            raise ValueError("number metric must be an exact int or float")
        if not math.isfinite(value):
            raise ValueError("number metric must be finite")
        return value
    raise ValueError("unsupported metric declaration type")


def _validate_metrics(
    value: object,
    declarations: Mapping[str, Mapping[str, object]],
) -> dict[str, int | float]:
    if type(value) is not dict:
        raise ValueError("evaluation metrics must be a mapping")
    undeclared = set(value) - set(declarations)
    if undeclared:
        raise ValueError("evaluation candidate contains undeclared metric")
    missing = [name for name, declaration in declarations.items() if declaration["required"] and name not in value]
    if missing:
        raise ValueError("evaluation candidate is missing required metric")
    accepted: dict[str, int | float] = {}
    for name, raw in value.items():
        accepted[name] = _validate_metric_value(raw, declarations[name])
    return accepted


def _validate_artifacts(
    value: object,
    declarations: Mapping[str, Mapping[str, object]],
    *,
    object_bytes: _ObjectByteAccess,
) -> dict[str, EvaluationArtifactRef]:
    if type(value) is not dict:
        raise ValueError("evaluation artifacts must be a mapping")
    undeclared = set(value) - set(declarations)
    if undeclared:
        raise ValueError("evaluation candidate contains undeclared artifact")
    missing = [name for name, declaration in declarations.items() if declaration["required"] and name not in value]
    if missing:
        raise ValueError("evaluation candidate is missing required artifact")
    accepted: dict[str, EvaluationArtifactRef] = {}
    for name, raw in value.items():
        if type(raw) is not dict or set(raw) != _EVALUATION_ARTIFACT_FIELDS:
            raise ValueError("evaluation artifact fields do not match schema")
        declaration = declarations[name]
        if raw["format"] != declaration["format"]:
            raise ValueError("evaluation artifact format does not match declaration")
        if raw["schema"] != declaration["schema"]:
            raise ValueError("evaluation artifact schema does not match declaration")
        base_ref = _validate_artifact_ref(raw)
        object_bytes.read_verified(base_ref)
        accepted[name] = cast(EvaluationArtifactRef, copy.deepcopy(raw))
    return accepted


def _formal_evaluation_result(
    *,
    study_result: dict[str, object],
    plan: StudyPlan,
    trial: TrialId,
    coordinate: dict[str, object],
    model_id: str,
    attempts: list[dict[str, object]],
    status: Literal["completed", "failed", "cancelled"],
    diagnostic: Diagnostic | None,
    metrics: Mapping[str, int | float] | None = None,
    artifacts: Mapping[str, EvaluationArtifactRef] | None = None,
) -> EvaluationResult:
    payload = None
    if status == "completed":
        if diagnostic is not None or metrics is None or artifacts is None:
            raise ValueError("completed EvaluationResult requires accepted success payload")
        payload = {
            "metrics": copy.deepcopy(dict(metrics)),
            "artifacts": copy.deepcopy(dict(artifacts)),
        }
    else:
        if diagnostic is None:
            raise ValueError("failed/cancelled EvaluationResult requires diagnostic")
    value: EvaluationResult = {
        "schema": "mjtensu.mldb-v2/evaluation-result/v1",
        "id": _result_id(str(study_result["id"]), str(trial), str(coordinate["coordinate"])),
        "study_result": study_result["id"],
        "plan": plan["id"],
        "trial": trial,
        "coordinate": coordinate["coordinate"],
        "stage": coordinate["stage"],
        "model": model_id,
        "task": coordinate["task"],
        "corpus": coordinate["corpus"],
        "evaluation_protocol": coordinate["evaluation_protocol"],
        "parameters": copy.deepcopy(cast(dict[str, object], coordinate["parameters"])),
        "source_commit": plan["source_commit"],
        "attempts": copy.deepcopy(attempts),
        "status": status,
        "diagnostic": copy.deepcopy(diagnostic),
        "result": payload,
    }
    return value


def _accept_evaluation_result(
    *,
    request: Mapping[str, object],
    mldb_data_root: str | Path,
    object_bytes: _ObjectByteAccess,
) -> _EvaluationAcceptanceResult:
    if type(request) is not dict or set(request) != _REQUEST_FIELDS:
        raise ValueError("evaluation acceptance request fields do not match schema")

    plan = _validate_study_plan(cast(Mapping[str, object], request["plan"]))
    study_result = _validate_study_result_anchor(request["study_result"])
    _validate_parent_plan_lineage(study_result, plan)
    trial_id, trial, coordinate = _select_target(plan, request["stage_input"])
    expected_model = _expected_model_id(str(study_result["id"]), trial)
    status, attempts = _validate_candidate_common(
        request["candidate"], expected_backend=cast(str, study_result["backend"])
    )
    candidate = cast(EvaluationTerminalCandidate, request["candidate"])
    stage_input = _validate_stage_input(
        request["stage_input"],
        study_result=study_result,
        plan=plan,
        trial=trial,
        coordinate=coordinate,
        mldb_data_root=mldb_data_root,
    )

    try:
        _validate_candidate_stage_key(candidate, stage_input=stage_input)
        protocol = _validate_protocol_pin(
            mldb_data_root=mldb_data_root,
            plan=plan,
            protocol_id=str(coordinate["evaluation_protocol"]),
        )
        if protocol["task"] != coordinate["task"]:
            raise ValueError("EvaluationProtocol Task does not match Plan coordinate")

        if status == "completed":
            if candidate["diagnostic"] is not None:
                raise ValueError("completed evaluation candidate diagnostic must be null")
            result = candidate["result"]
            if type(result) is not dict or set(result) != _COMPLETED_RESULT_FIELDS:
                raise ValueError("completed evaluation candidate result fields do not match schema")
            metrics = _validate_metrics(result["metrics"], protocol["metrics"])
            artifacts = _validate_artifacts(
                result["artifacts"], protocol["artifacts"], object_bytes=object_bytes
            )
            return {
                "evaluation_result": _formal_evaluation_result(
                    study_result=study_result, plan=plan, trial=trial_id,
                    coordinate=coordinate, model_id=expected_model, attempts=attempts,
                    status="completed", diagnostic=None, metrics=metrics, artifacts=artifacts,
                )
            }

        diagnostic = _validate_diagnostic(candidate["diagnostic"])
        if diagnostic is None:
            raise ValueError("failed/cancelled evaluation candidate requires diagnostic")
        if candidate["result"] is not None:
            raise ValueError("failed/cancelled evaluation candidate result must be null")
        return {
            "evaluation_result": _formal_evaluation_result(
                study_result=study_result,
                plan=plan,
                trial=trial_id,
                coordinate=coordinate,
                model_id=expected_model,
                attempts=attempts,
                status=cast(Literal["failed", "cancelled"], status),
                diagnostic=diagnostic,
            )
        }
    except Exception:
        if status != "completed":
            raise
        return {
            "evaluation_result": _formal_evaluation_result(
                study_result=study_result,
                plan=plan,
                trial=trial_id,
                coordinate=coordinate,
                model_id=expected_model,
                attempts=attempts,
                status="failed",
                diagnostic=_acceptance_diagnostic(),
            )
        }
