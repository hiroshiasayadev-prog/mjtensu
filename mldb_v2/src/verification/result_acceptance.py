"""Runtime result-acceptance composition and canonical child-record validation."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict, cast

from mldb_v2.src.backend.candidate_outcome import (
    CompletedEvaluationCandidate,
    CompletedTrainingCandidate,
    FailedEvaluationCandidate,
    FailedTrainingCandidate,
)
from mldb_v2.src.backend.stage_input import EvaluationStageInput, TrainingStageInput
from mldb_v2.src.common.diagnostic import _validate_diagnostic
from mldb_v2.src.common.ids import (
    EntityKind,
    EvaluationResultId,
    ModelId,
    TrainingResultId,
    _canonical_json_bytes,
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
    _validate_typed_reference,
)
from mldb_v2.src.common.parameters import _validate_public_parameter_value
from mldb_v2.src.evaluation.evaluation_result import EvaluationResult
from mldb_v2.src.repository.canonical_writes import (
    CanonicalDocument,
    ImmutableCanonicalId,
    ImmutableCanonicalKind,
)
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.results.study_result import StudyResult
from mldb_v2.src.storage.artifact_reference import _validate_artifact_ref
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study._plan_build import _validate_study_plan
from mldb_v2.src.study.plan import StudyPlan
from mldb_v2.src.training.model import Model, _resolve_model_lineage, _validate_model
from mldb_v2.src.training.training_result import (
    TrainingResult,
    _validate_attempt_summary,
    _validate_training_result,
)
from mldb_v2.src.verification._evaluation_result_acceptance import (
    _accept_evaluation_result,
    _validate_metrics,
    _validate_protocol_pin,
)
from mldb_v2.src.verification._training_result_acceptance import _accept_training_candidate

TrainingTerminalCandidate: TypeAlias = CompletedTrainingCandidate | FailedTrainingCandidate
EvaluationTerminalCandidate: TypeAlias = CompletedEvaluationCandidate | FailedEvaluationCandidate


class TrainingAcceptanceRequest(TypedDict):
    study_result: StudyResult
    plan: StudyPlan
    stage_input: TrainingStageInput
    candidate: TrainingTerminalCandidate


class EvaluationAcceptanceRequest(TypedDict):
    study_result: StudyResult
    plan: StudyPlan
    stage_input: EvaluationStageInput
    candidate: EvaluationTerminalCandidate


class TrainingAcceptanceResult(TypedDict):
    training_result: TrainingResult
    model: Model | None


class EvaluationAcceptanceResult(TypedDict):
    evaluation_result: EvaluationResult


class ResultAcceptor:
    """Compose the frozen acceptance boundary over the persistence-free implementations."""

    def __init__(
        self,
        *,
        mldb_data_root: str | Path,
        object_bytes: _ObjectByteAccess,
    ) -> None:
        self._mldb_data_root = Path(mldb_data_root)
        self._object_bytes = object_bytes

    def accept_training(
        self,
        *,
        request: TrainingAcceptanceRequest,
    ) -> TrainingAcceptanceResult:
        return cast(
            TrainingAcceptanceResult,
            _accept_training_candidate(
                study_result=request["study_result"],
                plan=request["plan"],
                stage_input=request["stage_input"],
                candidate=request["candidate"],
                mldb_data_root=self._mldb_data_root,
                object_bytes=self._object_bytes,
            ),
        )

    def accept_evaluation(
        self,
        *,
        request: EvaluationAcceptanceRequest,
    ) -> EvaluationAcceptanceResult:
        return cast(
            EvaluationAcceptanceResult,
            _accept_evaluation_result(
                request=request,
                mldb_data_root=self._mldb_data_root,
                object_bytes=self._object_bytes,
            ),
        )


_EVALUATION_FIELDS = {
    "schema",
    "id",
    "study_result",
    "plan",
    "trial",
    "coordinate",
    "stage",
    "model",
    "task",
    "corpus",
    "evaluation_protocol",
    "parameters",
    "source_commit",
    "attempts",
    "status",
    "diagnostic",
    "result",
}
_EVALUATION_RESULT_FIELDS = {"metrics", "artifacts"}
_EVALUATION_ARTIFACT_FIELDS = {"uri", "bytes", "sha256", "format", "schema"}
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)


def _same_canonical_value(left: object, right: object) -> bool:
    try:
        return _canonical_json_bytes(left) == _canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _find_plan_trial_and_coordinate(
    plan: StudyPlan,
    *,
    trial_id: str,
    coordinate_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    matches = [trial for trial in plan["trials"] if trial["trial"] == trial_id]
    if len(matches) != 1:
        raise ValueError("EvaluationResult trial does not exist in StudyPlan")
    trial = cast(dict[str, object], matches[0])
    coordinates = [
        coordinate
        for coordinate in cast(list[dict[str, object]], trial["evaluations"])
        if coordinate["coordinate"] == coordinate_id
    ]
    if len(coordinates) != 1:
        raise ValueError("EvaluationResult coordinate does not exist in StudyPlan trial")
    return trial, cast(dict[str, object], coordinates[0])


def _validate_training_record(
    resolver: CanonicalRepositoryResolver,
    *,
    entity_id: TrainingResultId,
    document: CanonicalDocument,
) -> None:
    training = _validate_training_result(document, expected_id=str(entity_id))
    expected_id = f"{training['study_result']}-{training['trial']}-train"
    if training["id"] != expected_id:
        raise ValueError("TrainingResult identity does not match StudyResult/trial lineage")

    plan = _validate_study_plan(
        resolver.resolve(kind=EntityKind.STUDY_PLAN, entity_id=training["plan"])
    )
    if training["plan"] != plan["id"] or training["source_commit"] != plan["source_commit"]:
        raise ValueError("TrainingResult Plan/source lineage mismatch")
    trials = [trial for trial in plan["trials"] if trial["trial"] == training["trial"]]
    if len(trials) != 1 or trials[0]["source"]["kind"] != "training":
        raise ValueError("TrainingResult trial is not a training-source Plan trial")
    source = trials[0]["source"]
    expected = {
        "task": source["task"],
        "architecture": source["architecture"],
        "corpus": source["corpus"],
        "train_protocol": source["train_protocol"],
        "parameters": source["parameters"],
        "seed": source["seed"],
    }
    actual = {
        "task": training["task"],
        "architecture": training["architecture"],
        "corpus": training["corpus"],
        "train_protocol": training["train_protocol"],
        "parameters": training["parameters"],
        "seed": training["seed"],
    }
    if not _same_canonical_value(actual, expected):
        raise ValueError("TrainingResult does not exactly match StudyPlan training source")

    if training["status"] == "completed":
        if training["result"] is None:
            raise ValueError("completed TrainingResult requires result")
        expected_model = f"{training['study_result']}-{training['trial']}-model"
        if training["result"]["model"] != expected_model:
            raise ValueError("TrainingResult Model identity is not deterministic")


def _validate_model_record(
    resolver: CanonicalRepositoryResolver,
    *,
    entity_id: ModelId,
    document: CanonicalDocument,
) -> None:
    model = _validate_model(document, expected_id=str(entity_id))
    training_document = resolver.resolve(
        kind=EntityKind.TRAINING_RESULT,
        entity_id=model["training_result"],
    )
    _validate_training_record(
        resolver,
        entity_id=model["training_result"],
        document=training_document,
    )
    training = _validate_training_result(
        training_document,
        expected_id=str(model["training_result"]),
    )
    if training["status"] != "completed" or training["result"] is None:
        raise ValueError("Model requires a completed TrainingResult")
    expected_model = f"{training['study_result']}-{training['trial']}-model"
    if model["id"] != expected_model or training["result"]["model"] != model["id"]:
        raise ValueError("Model identity does not match completed TrainingResult lineage")


def _validate_evaluation_artifacts_without_io(
    value: object,
    declarations: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("EvaluationResult artifacts must be a mapping")
    undeclared = set(value) - set(declarations)
    if undeclared:
        raise ValueError("EvaluationResult contains undeclared artifact")
    missing = [
        name
        for name, declaration in declarations.items()
        if declaration["required"] and name not in value
    ]
    if missing:
        raise ValueError("EvaluationResult is missing required artifact")

    accepted: dict[str, object] = {}
    for name, raw in value.items():
        if type(raw) is not dict or set(raw) != _EVALUATION_ARTIFACT_FIELDS:
            raise ValueError("EvaluationResult artifact fields do not match schema")
        declaration = declarations[name]
        if raw["format"] != declaration["format"]:
            raise ValueError("EvaluationResult artifact format does not match declaration")
        if raw["schema"] != declaration["schema"]:
            raise ValueError("EvaluationResult artifact schema does not match declaration")
        _validate_artifact_ref(raw)
        accepted[name] = raw
    return accepted


def _validate_evaluation_record(
    mldb_data_root: Path,
    resolver: CanonicalRepositoryResolver,
    *,
    entity_id: EvaluationResultId,
    document: CanonicalDocument,
) -> None:
    if type(document) is not dict or set(document) != _EVALUATION_FIELDS:
        raise ValueError("EvaluationResult fields do not match schema")
    _canonical_json_bytes(document)
    if document["schema"] != "mjtensu.mldb-v2/evaluation-result/v1":
        raise ValueError("unsupported EvaluationResult schema")

    result_id = _validate_typed_reference(document["id"])
    if result_id != str(entity_id):
        raise ValueError("EvaluationResult id does not match canonical path identity")
    study_result = _validate_typed_reference(document["study_result"])
    plan_id = _validate_typed_reference(document["plan"])
    trial_id = str(_validate_trial_id(document["trial"]))
    coordinate_id = str(_validate_evaluation_coordinate_id(document["coordinate"]))
    expected_id = f"{study_result}-{trial_id}-{coordinate_id}"
    if result_id != expected_id:
        raise ValueError("EvaluationResult identity does not match StudyResult coordinate lineage")

    stage = document["stage"]
    if type(stage) is not str or not stage:
        raise ValueError("EvaluationResult stage must be non-empty")
    model = _validate_typed_reference(document["model"])
    task = _validate_typed_reference(document["task"])
    corpus = _validate_typed_reference(document["corpus"])
    protocol_id = _validate_typed_reference(document["evaluation_protocol"])
    parameters = document["parameters"]
    if type(parameters) is not dict:
        raise ValueError("EvaluationResult parameters must be a mapping")
    for key, item in parameters.items():
        if type(key) is not str:
            raise ValueError("EvaluationResult parameter keys must be strings")
        _validate_public_parameter_value(item)
    source_commit = document["source_commit"]
    if type(source_commit) is not str or _COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError("EvaluationResult source_commit must be a full Git object id")

    attempts = document["attempts"]
    if type(attempts) is not list or not attempts:
        raise ValueError("EvaluationResult attempts must be a non-empty list")
    for attempt in attempts:
        _validate_attempt_summary(attempt)

    status = document["status"]
    if type(status) is not str or status not in _TERMINAL_STATUSES:
        raise ValueError("invalid EvaluationResult status")
    diagnostic = _validate_diagnostic(document["diagnostic"])

    plan = _validate_study_plan(
        resolver.resolve(kind=EntityKind.STUDY_PLAN, entity_id=plan_id)
    )
    if plan["source_commit"] != source_commit:
        raise ValueError("EvaluationResult source commit does not match StudyPlan")
    trial, coordinate = _find_plan_trial_and_coordinate(
        plan,
        trial_id=trial_id,
        coordinate_id=coordinate_id,
    )
    expected_fields = {
        "stage": coordinate["stage"],
        "task": coordinate["task"],
        "corpus": coordinate["corpus"],
        "evaluation_protocol": coordinate["evaluation_protocol"],
        "parameters": coordinate["parameters"],
    }
    actual_fields = {
        "stage": stage,
        "task": task,
        "corpus": corpus,
        "evaluation_protocol": protocol_id,
        "parameters": parameters,
    }
    if not _same_canonical_value(actual_fields, expected_fields):
        raise ValueError("EvaluationResult does not exactly match StudyPlan coordinate")

    source = cast(dict[str, object], trial["source"])
    if source["kind"] == "training":
        expected_model = f"{study_result}-{trial_id}-model"
    elif source["kind"] == "existing_model":
        expected_model = _validate_typed_reference(source["model"])
    else:
        raise ValueError("unsupported StudyPlan trial source")
    if model != expected_model:
        raise ValueError("EvaluationResult Model does not match StudyPlan lineage")
    lineage = _resolve_model_lineage(mldb_data_root, expected_model)
    if lineage.model["id"] != model or lineage.training_result["task"] != task:
        raise ValueError("EvaluationResult canonical Model lineage is invalid")

    payload = document["result"]
    if status == "completed":
        if diagnostic is not None:
            raise ValueError("completed EvaluationResult diagnostic must be null")
        if type(payload) is not dict or set(payload) != _EVALUATION_RESULT_FIELDS:
            raise ValueError("completed EvaluationResult requires exact result payload")
        protocol = _validate_protocol_pin(
            mldb_data_root=mldb_data_root,
            plan=plan,
            protocol_id=protocol_id,
        )
        if protocol["task"] != task:
            raise ValueError("EvaluationProtocol Task does not match EvaluationResult")
        _validate_metrics(payload["metrics"], protocol["metrics"])
        _validate_evaluation_artifacts_without_io(
            payload["artifacts"],
            protocol["artifacts"],
        )
    else:
        if diagnostic is None:
            raise ValueError("failed/cancelled EvaluationResult requires diagnostic")
        if payload is not None:
            raise ValueError("failed/cancelled EvaluationResult result must be null")


class AcceptedResultRecordValidator:
    """Kind-dispatched persistence validator for accepted terminal child records."""

    def __init__(self, *, mldb_data_root: str | Path) -> None:
        self._mldb_data_root = Path(mldb_data_root)
        self._resolver = CanonicalRepositoryResolver(self._mldb_data_root)

    def validate(
        self,
        *,
        kind: ImmutableCanonicalKind,
        entity_id: ImmutableCanonicalId,
        document: CanonicalDocument,
    ) -> None:
        if kind == EntityKind.TRAINING_RESULT:
            _validate_training_record(
                self._resolver,
                entity_id=cast(TrainingResultId, entity_id),
                document=document,
            )
            return
        if kind == EntityKind.MODEL:
            _validate_model_record(
                self._resolver,
                entity_id=cast(ModelId, entity_id),
                document=document,
            )
            return
        if kind == EntityKind.EVALUATION_RESULT:
            _validate_evaluation_record(
                self._mldb_data_root,
                self._resolver,
                entity_id=cast(EvaluationResultId, entity_id),
                document=document,
            )
            return
        raise ValueError("accepted-result validator only accepts terminal child record kinds")
