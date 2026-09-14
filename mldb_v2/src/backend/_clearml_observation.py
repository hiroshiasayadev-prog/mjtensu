"""ClearML active observation and terminal candidate recovery."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, cast

from mldb_v2.src.backend._clearml_admission import (
    ClearMLTaskRecord,
    _HARNESS_SYMBOL,
    _metadata as _admission_metadata,
    _ownership_key as _admission_ownership_key,
    _restore_stage_input,
    _study_id_from_pins,
)
from mldb_v2.src.backend.candidate_outcome import (
    ActiveBackendObservation,
    BackendObservation,
    StageKey,
    TerminalCandidate,
)
from mldb_v2.src.backend.stage_input import StageInput
from mldb_v2.src.common.diagnostic import _validate_diagnostic
from mldb_v2.src.common.ids import (
    _canonical_json_bytes,
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
    _validate_typed_reference,
)
from mldb_v2.src.results.attempt_summary import AttemptSummary
from mldb_v2.src.storage.artifact_reference import _validate_artifact_ref
from mldb_v2.src.training.canonical_weights import (
    _validate_canonical_weights_artifact_ref,
)

_STAGE_KEY_FIELDS = {
    "study_result", "plan", "trial", "kind", "coordinate", "source_commit"
}
_CANDIDATE_FIELDS = {
    "state", "stage_key", "attempts", "status", "diagnostic", "result"
}
_ATTEMPT_FIELDS = {
    "backend", "execution_id", "status", "started_at", "ended_at", "diagnostic"
}
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_EVALUATION_ARTIFACT_FIELDS = {"uri", "bytes", "sha256", "format", "schema"}
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
_RFC3339_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\Z",
    re.ASCII,
)


class ClearMLObservationError(RuntimeError):
    """Bounded ClearML observation/recovery failure."""


@dataclass(frozen=True)
class ClearMLRuntimeProjection:
    """SDK-neutral ordered runtime projection for one logical stage."""

    state: Literal["active", "terminal"]
    execution_ids: Sequence[str]
    terminal_candidates: Sequence[object]


class ClearMLObservationClient(Protocol):
    """Minimal mocked/production seam for observation and collection."""

    def search_tasks(
        self, *, project: str, ownership_key: str
    ) -> Sequence[ClearMLTaskRecord]: ...

    def read_runtime_projection(
        self, *, task_id: str
    ) -> ClearMLRuntimeProjection: ...


def _namespace_of(reference: object) -> str:
    return _validate_typed_reference(reference).split("/", 1)[0]


def _validate_stage_key(value: object) -> StageKey:
    if type(value) is not dict or set(value) != _STAGE_KEY_FIELDS:
        raise ValueError("StageKey fields do not match schema")
    study_result = _validate_typed_reference(value["study_result"])
    plan = _validate_typed_reference(value["plan"])
    trial = str(_validate_trial_id(value["trial"]))
    source_commit = value["source_commit"]
    if type(source_commit) is not str or _COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError("StageKey source_commit must be a full Git object id")
    kind = value["kind"]
    if kind == "training":
        if value["coordinate"] is not None:
            raise ValueError("training StageKey coordinate must be null")
        coordinate = None
    elif kind == "evaluation":
        coordinate = str(_validate_evaluation_coordinate_id(value["coordinate"]))
    else:
        raise ValueError("StageKey kind must be training or evaluation")
    return cast(StageKey, {
        "study_result": study_result,
        "plan": plan,
        "trial": trial,
        "kind": kind,
        "coordinate": coordinate,
        "source_commit": source_commit,
    })


def _ownership_key_for_stage_key(stage_key: StageKey) -> str:
    validated = _validate_stage_key(stage_key)
    payload = {
        "study_result": validated["study_result"],
        "trial": validated["trial"],
        "kind": validated["kind"],
        "coordinate": validated["coordinate"],
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return f"mldb-v2-stage:{digest}"


def _stage_key_from_input(stage_input: StageInput) -> StageKey:
    return _validate_stage_key(
        {
            "study_result": stage_input["study_result"],
            "plan": stage_input["plan"],
            "trial": stage_input["trial"],
            "kind": stage_input["kind"],
            "coordinate": stage_input["coordinate"],
            "source_commit": stage_input["source_commit"],
        }
    )



def _validate_owned_record(
    record: ClearMLTaskRecord,
    *,
    requested: StageKey,
    ownership_key: str,
) -> StageInput:
    if type(record.task_id) is not str or not record.task_id:
        raise ClearMLObservationError("ClearML Task ID must be a non-empty opaque string")
    transport = record.configuration.get("mldb.stage_input")
    try:
        stage_input = _restore_stage_input(cast(str, transport))
    except Exception as error:
        raise ClearMLObservationError("ClearML StageInput projection is malformed") from error
    recovered_key = _stage_key_from_input(stage_input)
    if recovered_key != requested:
        raise ClearMLObservationError("ClearML recovered StageKey does not match request")
    if _admission_ownership_key(stage_input) != ownership_key:
        raise ClearMLObservationError("ClearML ownership key does not match recovered StageInput")
    study_id = _study_id_from_pins(stage_input)
    expected_metadata = _admission_metadata(
        stage_input, study_id=study_id, ownership_key=ownership_key
    )
    if any(record.metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ClearMLObservationError("ClearML searchable metadata does not match StageKey lineage")
    if record.configuration.get("mldb.ownership_key") != ownership_key:
        raise ClearMLObservationError("ClearML configuration ownership mismatch")
    if record.configuration.get("mldb.source_commit") != requested["source_commit"]:
        raise ClearMLObservationError("ClearML source_commit mismatch")
    if record.configuration.get("mldb.harness") != _HARNESS_SYMBOL:
        raise ClearMLObservationError("ClearML common-harness projection mismatch")
    if record.configuration.get("mldb.public_parameters") != stage_input["stage"]["parameters"]:
        raise ClearMLObservationError("ClearML public parameter projection mismatch")
    return stage_input


def _parse_utc(value: object, *, label: str) -> datetime | None:
    if value is None:
        return None
    if type(value) is not str or _RFC3339_UTC_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be RFC3339 UTC using Z")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be RFC3339 UTC using Z") from error


def _validate_attempt(value: object, *, execution_id: str) -> AttemptSummary:
    if type(value) is not dict or set(value) != _ATTEMPT_FIELDS:
        raise ValueError("attempt summary fields do not match schema")
    if value["backend"] != "clearml":
        raise ValueError("attempt backend must be clearml")
    if value["execution_id"] != execution_id:
        raise ValueError("attempt execution_id does not match backend attempt order")
    status = value["status"]
    if type(status) is not str or status not in _TERMINAL_STATUSES:
        raise ValueError("invalid attempt status")
    started = _parse_utc(value["started_at"], label="attempt started_at")
    ended = _parse_utc(value["ended_at"], label="attempt ended_at")
    if started is not None and ended is not None and ended < started:
        raise ValueError("attempt ended_at precedes started_at")
    diagnostic = _validate_diagnostic(value["diagnostic"])
    if status == "completed" and diagnostic is not None:
        raise ValueError("completed attempt diagnostic must be null")
    if status != "completed" and diagnostic is None:
        raise ValueError("non-completed attempt requires diagnostic")
    return cast(AttemptSummary, value)


def _validate_evaluation_result(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"metrics", "artifacts"}:
        raise ValueError("completed evaluation result fields do not match schema")
    metrics = value["metrics"]
    if type(metrics) is not dict or any(type(key) is not str for key in metrics):
        raise ValueError("evaluation metrics must be a string-keyed mapping")
    for metric in metrics.values():
        if type(metric) not in {int, float}:
            raise ValueError("evaluation metric values must be exact int or float")
        if type(metric) is float and not math.isfinite(metric):
            raise ValueError("evaluation metric values must be finite")
    artifacts = value["artifacts"]
    if type(artifacts) is not dict or any(type(key) is not str for key in artifacts):
        raise ValueError("evaluation artifacts must be a string-keyed mapping")
    for artifact in artifacts.values():
        if type(artifact) is not dict or set(artifact) != _EVALUATION_ARTIFACT_FIELDS:
            raise ValueError("evaluation ArtifactRef fields do not match schema")
        _validate_artifact_ref(artifact)
        if type(artifact["format"]) is not str or not artifact["format"]:
            raise ValueError("evaluation artifact format must be non-empty")
        if type(artifact["schema"]) is not str or not artifact["schema"]:
            raise ValueError("evaluation artifact schema must be non-empty")
    return value


def _validate_completed_result(*, kind: str, value: object) -> object:
    if kind == "training":
        if type(value) is not dict or set(value) != {"weights"}:
            raise ValueError("completed training result fields do not match schema")
        _validate_canonical_weights_artifact_ref(value["weights"])
        return value
    return _validate_evaluation_result(value)


def _validate_terminal_candidate(
    value: object,
    *,
    requested: StageKey,
    execution_id: str,
) -> tuple[TerminalCandidate, AttemptSummary]:
    if type(value) is not dict or set(value) != _CANDIDATE_FIELDS:
        raise ValueError("terminal candidate fields do not match schema")
    if value["state"] != "terminal":
        raise ValueError("stored harness payload must be terminal")
    candidate_key = _validate_stage_key(value["stage_key"])
    if candidate_key != requested:
        raise ValueError("stored harness candidate StageKey mismatch")
    attempts = value["attempts"]
    if type(attempts) is not list or len(attempts) != 1:
        raise ValueError("each stored harness candidate must contain exactly one attempt")
    attempt = _validate_attempt(attempts[0], execution_id=execution_id)
    status = value["status"]
    if type(status) is not str or status not in _TERMINAL_STATUSES:
        raise ValueError("invalid terminal candidate status")
    if status != attempt["status"]:
        raise ValueError("terminal candidate status contradicts attempt status")
    diagnostic = _validate_diagnostic(value["diagnostic"])
    if status == "completed":
        if diagnostic is not None:
            raise ValueError("completed terminal candidate diagnostic must be null")
        _validate_completed_result(kind=requested["kind"], value=value["result"])
    else:
        if diagnostic is None:
            raise ValueError("failed/cancelled terminal candidate requires diagnostic")
        if value["result"] is not None:
            raise ValueError("failed/cancelled terminal candidate result must be null")
    return cast(TerminalCandidate, value), attempt


def _validate_execution_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("execution_ids must be an ordered sequence")
    ids = tuple(value)
    if not ids or any(type(item) is not str or not item for item in ids):
        raise ValueError("execution_ids must contain non-empty opaque strings")
    if len(set(ids)) != len(ids):
        raise ValueError("execution_ids must not contain duplicates")
    return ids


def _validate_projection(
    projection: ClearMLRuntimeProjection,
    *,
    owner_task_id: str,
    requested: StageKey,
) -> ActiveBackendObservation | TerminalCandidate:
    if projection.state not in {"active", "terminal"}:
        raise ValueError("ClearML runtime projection state must be active or terminal")
    execution_ids = _validate_execution_ids(projection.execution_ids)
    if execution_ids[0] != owner_task_id:
        raise ValueError("ClearML runtime projection does not start with recovered owner Task")
    if isinstance(projection.terminal_candidates, (str, bytes)):
        raise ValueError("terminal_candidates must be an ordered sequence")
    payloads = tuple(projection.terminal_candidates)
    if projection.state == "active":
        if len(payloads) >= len(execution_ids):
            raise ValueError("active projection must retain at least one non-terminal attempt")
        for index, payload in enumerate(payloads):
            candidate, _attempt = _validate_terminal_candidate(
                payload,
                requested=requested,
                execution_id=execution_ids[index],
            )
            if candidate["status"] == "completed":
                raise ValueError("completed attempt cannot be followed by an active retry")
        return ActiveBackendObservation(
            state="active",
            stage_key=requested,
            backend="clearml",
            execution_ids=list(execution_ids),
        )

    if len(payloads) != len(execution_ids):
        raise ValueError("terminal projection requires one harness payload per attempt")
    attempts: list[AttemptSummary] = []
    validated_candidates: list[TerminalCandidate] = []
    previous_started: datetime | None = None
    for index, payload in enumerate(payloads):
        candidate, attempt = _validate_terminal_candidate(
            payload,
            requested=requested,
            execution_id=execution_ids[index],
        )
        started = _parse_utc(attempt["started_at"], label="attempt started_at")
        if previous_started is not None and started is not None and started < previous_started:
            raise ValueError("attempt history is not ordered oldest-to-newest")
        if started is not None:
            previous_started = started
        if index < len(payloads) - 1 and candidate["status"] == "completed":
            raise ValueError("completed attempt cannot be followed by another retry")
        attempts.append(attempt)
        validated_candidates.append(candidate)

    final = validated_candidates[-1]
    return cast(TerminalCandidate, {
        "state": "terminal",
        "stage_key": requested,
        "attempts": attempts,
        "status": final["status"],
        "diagnostic": final["diagnostic"],
        "result": final["result"],
    })


class ClearMLObservationService:
    """Recover ClearML work as frozen backend-neutral observations/candidates."""

    def __init__(self, *, client: ClearMLObservationClient) -> None:
        self._client = client

    def _recover(
        self, *, stage_key: StageKey
    ) -> tuple[StageKey, ClearMLTaskRecord, ClearMLRuntimeProjection] | None:
        requested = _validate_stage_key(stage_key)
        ownership_key = _ownership_key_for_stage_key(requested)
        project = f"mldb/{_namespace_of(requested['study_result'])}"
        try:
            records = tuple(
                self._client.search_tasks(
                    project=project,
                    ownership_key=ownership_key,
                )
            )
        except Exception as error:
            raise ClearMLObservationError("ClearML ownership search failed") from error
        if not records:
            return None
        if len(records) != 1:
            raise ClearMLObservationError(
                "multiple ClearML Tasks claim one logical ownership key"
            )
        record = records[0]
        try:
            _validate_owned_record(
                record,
                requested=requested,
                ownership_key=ownership_key,
            )
        except ClearMLObservationError:
            raise
        except Exception as error:
            raise ClearMLObservationError("ClearML ownership projection is malformed") from error
        try:
            projection = self._client.read_runtime_projection(task_id=record.task_id)
        except Exception as error:
            raise ClearMLObservationError("ClearML runtime projection read failed") from error
        if not isinstance(projection, ClearMLRuntimeProjection):
            raise ClearMLObservationError("ClearML runtime projection has invalid shape")
        return requested, record, projection

    def observe(self, *, stage_key: StageKey) -> BackendObservation | None:
        recovered = self._recover(stage_key=stage_key)
        if recovered is None:
            return None
        requested, record, projection = recovered
        try:
            return _validate_projection(
                projection,
                owner_task_id=record.task_id,
                requested=requested,
            )
        except Exception as error:
            raise ClearMLObservationError("ClearML observation projection is malformed") from error

    def collect(self, *, stage_key: StageKey) -> TerminalCandidate | None:
        recovered = self._recover(stage_key=stage_key)
        if recovered is None:
            return None
        requested, record, projection = recovered
        if projection.state == "active":
            try:
                _validate_projection(
                    projection,
                    owner_task_id=record.task_id,
                    requested=requested,
                )
            except Exception as error:
                raise ClearMLObservationError("ClearML observation projection is malformed") from error
            return None
        try:
            candidate = _validate_projection(
                projection,
                owner_task_id=record.task_id,
                requested=requested,
            )
        except Exception as error:
            raise ClearMLObservationError("ClearML terminal candidate projection is malformed") from error
        if candidate["state"] != "terminal":
            raise ClearMLObservationError("ClearML terminal projection did not yield terminal candidate")
        return cast(TerminalCandidate, candidate)
