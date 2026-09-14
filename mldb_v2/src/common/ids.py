"""MLDB v2 common identity runtime values and internal validators."""

from __future__ import annotations

import json as _json
import math as _math
import re as _re
from enum import Enum
from typing import NewType, TypeAlias

NamespaceId = NewType("NamespaceId", str)
TaskId = NewType("TaskId", str)
CorpusId = NewType("CorpusId", str)
ArchitectureId = NewType("ArchitectureId", str)
TrainProtocolId = NewType("TrainProtocolId", str)
EvaluationProtocolId = NewType("EvaluationProtocolId", str)
StudyId = NewType("StudyId", str)
StudyPlanId = NewType("StudyPlanId", str)
StudyResultId = NewType("StudyResultId", str)
TrainingResultId = NewType("TrainingResultId", str)
ModelId = NewType("ModelId", str)
EvaluationResultId = NewType("EvaluationResultId", str)
TrialId = NewType("TrialId", str)
EvaluationCoordinateId = NewType("EvaluationCoordinateId", str)
ExecutionKey = NewType("ExecutionKey", str)
TypedEntityId: TypeAlias = (
    TaskId
    | CorpusId
    | ArchitectureId
    | TrainProtocolId
    | EvaluationProtocolId
    | StudyId
    | StudyPlanId
    | StudyResultId
    | TrainingResultId
    | ModelId
    | EvaluationResultId
)


class EntityKind(str, Enum):
    NAMESPACE = "namespace"
    TASK = "task"
    CORPUS = "corpus"
    ARCHITECTURE = "architecture"
    TRAIN_PROTOCOL = "train_protocol"
    EVALUATION_PROTOCOL = "evaluation_protocol"
    STUDY = "study"
    STUDY_PLAN = "study_plan"
    TRAINING_RESULT = "training_result"
    MODEL = "model"
    EVALUATION_RESULT = "evaluation_result"
    STUDY_RESULT = "study_result"


class DefinitionKind(str, Enum):
    TASK = "task"
    CORPUS = "corpus"
    ARCHITECTURE = "architecture"
    TRAIN_PROTOCOL = "train_protocol"
    EVALUATION_PROTOCOL = "evaluation_protocol"
    STUDY = "study"


_LOCAL_SEGMENT_RE = _re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", _re.ASCII)
_TRIAL_ID_RE = _re.compile(r"trial-([0-9]{4})", _re.ASCII)
_EVALUATION_COORDINATE_ID_RE = _re.compile(r"eval-([0-9]{4})", _re.ASCII)


def _validate_local_segment(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _LOCAL_SEGMENT_RE.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")
    return value


def _validate_namespace_id(value: object) -> NamespaceId:
    return NamespaceId(_validate_local_segment(value, label="namespace id"))


def _validate_typed_reference(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid typed entity reference")
    parts = value.split("/")
    if len(parts) != 2:
        raise ValueError("invalid typed entity reference")
    namespace, local_id = parts
    _validate_local_segment(namespace, label="reference namespace")
    _validate_local_segment(local_id, label="reference local id")
    return value


def _validate_sequence_id(value: object, pattern: _re.Pattern[str], *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid {label}")
    match = pattern.fullmatch(value)
    if match is None or int(match.group(1)) < 1:
        raise ValueError(f"invalid {label}")
    return value


def _validate_trial_id(value: object) -> TrialId:
    return TrialId(_validate_sequence_id(value, _TRIAL_ID_RE, label="trial id"))


def _validate_evaluation_coordinate_id(value: object) -> EvaluationCoordinateId:
    return EvaluationCoordinateId(
        _validate_sequence_id(value, _EVALUATION_COORDINATE_ID_RE, label="evaluation coordinate id")
    )


def _validate_canonical_json_value(value: object) -> None:
    value_type = type(value)
    if value is None or value_type in {bool, str, int}:
        return
    if value_type is float:
        if not _math.isfinite(value):
            raise ValueError("canonical JSON numbers must be finite")
        return
    if value_type is list:
        for item in value:
            _validate_canonical_json_value(item)
        return
    if value_type is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical JSON object keys must be strings")
            _validate_canonical_json_value(item)
        return
    raise ValueError("value is not JSON-compatible")


def _canonical_json_bytes(value: object) -> bytes:
    _validate_canonical_json_value(value)
    text = _json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")
