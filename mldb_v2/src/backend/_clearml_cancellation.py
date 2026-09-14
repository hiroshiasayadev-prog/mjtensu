"""ClearML Study cancellation and optional read-only log access."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from mldb_v2.src.common.ids import (
    StudyResultId,
    _canonical_json_bytes,
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
    _validate_typed_reference,
)

_STUDY_RESULT_METADATA_KEY = "mldb.study_result"


class ClearMLCancellationError(RuntimeError):
    """Bounded failure while requesting cancellation of owned ClearML work."""


class ClearMLCancellationState(str, Enum):
    """SDK-neutral cancellation classification supplied by the activation seam."""

    ACTIVE = "active"
    TERMINAL = "terminal"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ClearMLOwnedTask:
    """SDK-neutral searchable projection used by cancellation/log reads."""

    task_id: str
    metadata: Mapping[str, str]
    cancellation_state: ClearMLCancellationState


@dataclass(frozen=True)
class ClearMLLogData:
    """Safe log payload returned by an optional client capability."""

    chunks: tuple[str, ...]
    locator: str | None = None

    def __post_init__(self) -> None:
        if any(type(chunk) is not str for chunk in self.chunks):
            raise ValueError("log chunks must be strings")
        if self.locator is not None and (type(self.locator) is not str or not self.locator):
            raise ValueError("log locator must be null or a non-empty string")


@dataclass(frozen=True)
class ClearMLLogRecord:
    """Immutable observational log result with no backend configuration."""

    task_id: str
    chunks: tuple[str, ...]
    locator: str | None = None


class ClearMLTaskSearchClient(Protocol):
    """Search seam that must apply the requested searchable metadata filter."""

    def search_tasks_by_metadata(
        self, *, project: str, key: str, value: str
    ) -> Sequence[ClearMLOwnedTask]: ...


class ClearMLCancellationClient(ClearMLTaskSearchClient, Protocol):
    """Required cancellation seam; request_cancellation is idempotent."""

    def request_cancellation(self, *, task_id: str) -> None: ...


class ClearMLLogCapability(Protocol):
    """Optional read-only capability discovered at runtime."""

    def read_task_logs(self, *, task_id: str) -> ClearMLLogData | None: ...


def _study_identity(study_result: StudyResultId) -> tuple[str, str]:
    value = _validate_typed_reference(study_result)
    namespace = value.split("/", 1)[0]
    return value, namespace


def _expected_ownership_key(
    *, study_result: str, trial: str, kind: str, coordinate: str | None
) -> str:
    payload = {
        "study_result": study_result,
        "trial": trial,
        "kind": kind,
        "coordinate": coordinate,
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return f"mldb-v2-stage:{digest}"


def _is_exact_owned_task(
    task: ClearMLOwnedTask, *, study_result: str, namespace: str
) -> bool:
    if type(task.task_id) is not str or not task.task_id:
        return False
    if not isinstance(task.metadata, Mapping):
        return False
    metadata = task.metadata
    if metadata.get("mldb.namespace") != namespace:
        return False
    if metadata.get(_STUDY_RESULT_METADATA_KEY) != study_result:
        return False

    try:
        trial = str(_validate_trial_id(metadata.get("mldb.trial")))
    except ValueError:
        return False
    kind = metadata.get("mldb.stage_kind")
    if kind == "training":
        if "mldb.evaluation_coordinate" in metadata or "mldb.evaluation_stage" in metadata:
            return False
        coordinate = None
    elif kind == "evaluation":
        stage_name = metadata.get("mldb.evaluation_stage")
        if type(stage_name) is not str or not stage_name:
            return False
        try:
            coordinate = str(
                _validate_evaluation_coordinate_id(
                    metadata.get("mldb.evaluation_coordinate")
                )
            )
        except ValueError:
            return False
    else:
        return False

    ownership_key = metadata.get("mldb.ownership_key")
    if type(ownership_key) is not str:
        return False
    return ownership_key == _expected_ownership_key(
        study_result=study_result,
        trial=trial,
        kind=kind,
        coordinate=coordinate,
    )


def _owned_tasks(
    *, client: ClearMLTaskSearchClient, study_result: StudyResultId
) -> tuple[ClearMLOwnedTask, ...]:
    exact_study_result, namespace = _study_identity(study_result)
    project = f"mldb/{namespace}"
    try:
        records = client.search_tasks_by_metadata(
            project=project,
            key=_STUDY_RESULT_METADATA_KEY,
            value=exact_study_result,
        )
    except Exception as error:
        raise ClearMLCancellationError("ClearML Study ownership search failed") from error

    owned: list[ClearMLOwnedTask] = []
    seen_task_ids: set[str] = set()
    for task in tuple(records):
        if not _is_exact_owned_task(
            task, study_result=exact_study_result, namespace=namespace
        ):
            continue
        if task.task_id in seen_task_ids:
            continue
        seen_task_ids.add(task.task_id)
        owned.append(task)
    return tuple(owned)


class ClearMLCancellationService:
    """Request backend-only cancellation for one exact StudyResult."""

    def __init__(self, *, client: ClearMLCancellationClient) -> None:
        self._client = client

    def cancel_study(self, *, study_result: StudyResultId) -> None:
        for task in _owned_tasks(client=self._client, study_result=study_result):
            if task.cancellation_state is not ClearMLCancellationState.ACTIVE:
                continue
            try:
                self._client.request_cancellation(task_id=task.task_id)
            except Exception as error:
                raise ClearMLCancellationError(
                    "ClearML cancellation request failed"
                ) from error


class ClearMLLogService:
    """Optional observational log reader for exact owned ClearML Tasks."""

    def __init__(self, *, client: ClearMLTaskSearchClient) -> None:
        self._client = client

    def read_task_logs(
        self, *, study_result: StudyResultId, task_id: str
    ) -> ClearMLLogRecord | None:
        if type(task_id) is not str or not task_id:
            raise ValueError("task_id must be a non-empty string")
        try:
            tasks = _owned_tasks(client=self._client, study_result=study_result)
        except ClearMLCancellationError:
            return None
        if not any(task.task_id == task_id for task in tasks):
            return None

        reader = getattr(self._client, "read_task_logs", None)
        if not callable(reader):
            return None
        try:
            data = reader(task_id=task_id)
        except Exception:
            return None
        if data is None or not isinstance(data, ClearMLLogData):
            return None
        return ClearMLLogRecord(
            task_id=task_id,
            chunks=tuple(data.chunks),
            locator=data.locator,
        )
