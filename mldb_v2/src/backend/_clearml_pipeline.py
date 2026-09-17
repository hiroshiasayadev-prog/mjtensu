"""ClearML StudyResult -> Pipeline controller creation/recovery seam."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from mldb_v2.src.backend.study_execution import (
    BackendStudyExecutionObservation,
    StudyExecutionKey,
)
from mldb_v2.src.common.ids import _canonical_json_bytes, _validate_typed_reference
from mldb_v2.src.results.study_result import StudyResult, _validate_study_result
from mldb_v2.src.study._plan_build import _validate_study_plan
from mldb_v2.src.study.plan import StudyPlan


class ClearMLPipelineError(RuntimeError):
    """Bounded Pipeline creation/recovery/projection failure."""


@dataclass(frozen=True)
class ClearMLPipelineRecord:
    task_id: str
    metadata: Mapping[str, str]
    configuration: Mapping[str, object]
    status: str


@dataclass(frozen=True)
class ClearMLPipelineCreateRequest:
    project: str
    task_name: str
    metadata: Mapping[str, str]
    configuration: Mapping[str, object]
    source_commit: str


class ClearMLPipelineClient(Protocol):
    def search_pipeline_runs(
        self, *, project: str, ownership_key: str
    ) -> Sequence[ClearMLPipelineRecord]: ...

    def create_pipeline_run(
        self, request: ClearMLPipelineCreateRequest
    ) -> str | None: ...


def _namespace(reference: object) -> str:
    return _validate_typed_reference(reference).split("/", 1)[0]


def _execution_key(plan: StudyPlan, result: StudyResult) -> StudyExecutionKey:
    return {
        "study_result": result["id"],
        "plan": plan["id"],
        "source_commit": plan["source_commit"],
    }


def _validate_pair(plan: object, study_result: object) -> tuple[StudyPlan, StudyResult]:
    validated_plan = _validate_study_plan(cast(Mapping[str, object], plan))
    validated_result = _validate_study_result(study_result)
    if validated_result["plan"] != validated_plan["id"]:
        raise ValueError("StudyResult plan does not match StudyPlan")
    if validated_result["study"] != validated_plan["study"]:
        raise ValueError("StudyResult study does not match StudyPlan")
    if validated_result["source_commit"] != validated_plan["source_commit"]:
        raise ValueError("StudyResult source_commit does not match StudyPlan")
    return validated_plan, validated_result


def _pipeline_ownership_key(plan: StudyPlan, result: StudyResult) -> str:
    payload = {
        "study_result": result["id"],
        "plan": plan["id"],
        "plan_sha256": plan["content_sha256"],
        "source_commit": plan["source_commit"],
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return f"mldb-v2-pipeline:{digest}"


def _step_name(*, trial: str, kind: str, coordinate: str | None) -> str:
    suffix = "train" if kind == "training" else cast(str, coordinate)
    return f"{trial}-{suffix}"


def _pipeline_topology(plan: StudyPlan, result: StudyResult) -> dict[str, object]:
    steps: list[dict[str, object]] = []
    for plan_trial, result_trial in zip(plan["trials"], result["trials"]):
        trial = str(plan_trial["trial"])
        source = plan_trial["source"]
        training_name: str | None = None
        if source["kind"] == "training":
            training_name = _step_name(trial=trial, kind="training", coordinate=None)
            steps.append(
                {
                    "name": training_name,
                    "trial": trial,
                    "kind": "training",
                    "coordinate": None,
                    "stage": "training",
                    "parents": [],
                }
            )
        for evaluation in plan_trial["evaluations"]:
            coordinate = str(evaluation["coordinate"])
            steps.append(
                {
                    "name": _step_name(
                        trial=trial, kind="evaluation", coordinate=coordinate
                    ),
                    "trial": trial,
                    "kind": "evaluation",
                    "coordinate": coordinate,
                    "stage": str(evaluation["stage"]),
                    "parents": [] if training_name is None else [training_name],
                }
            )
    return {
        "schema": "mjtensu.mldb-v2/clearml-pipeline-topology/v1",
        "study_result": str(result["id"]),
        "plan": str(plan["id"]),
        "plan_sha256": plan["content_sha256"],
        "study": str(plan["study"]),
        "source_commit": plan["source_commit"],
        "steps": steps,
    }


def _metadata(
    plan: StudyPlan, result: StudyResult, *, ownership_key: str
) -> dict[str, str]:
    return {
        "mldb.namespace": _namespace(result["id"]),
        "mldb.study": str(plan["study"]),
        "mldb.plan": str(plan["id"]),
        "mldb.study_result": str(result["id"]),
        "mldb.source_commit": plan["source_commit"],
        "mldb.pipeline_ownership_key": ownership_key,
    }


def _configuration(
    plan: StudyPlan, result: StudyResult, *, ownership_key: str
) -> dict[str, object]:
    immutable_result_identity = {
        "id": str(result["id"]),
        "execution_key": str(result["execution_key"]),
        "plan": str(result["plan"]),
        "study": str(result["study"]),
        "source_commit": result["source_commit"],
        "backend": result["backend"],
        "created_at": result["created_at"],
    }
    return {
        "mldb.pipeline_ownership_key": ownership_key,
        "mldb.plan": json.loads(_canonical_json_bytes(plan).decode("utf-8")),
        "mldb.study_result_identity": immutable_result_identity,
        "mldb.topology": _pipeline_topology(plan, result),
        "mldb.cache_executed_step": False,
    }


def _matches(
    record: ClearMLPipelineRecord,
    *,
    metadata: Mapping[str, str],
    configuration: Mapping[str, object],
) -> bool:
    if type(record.task_id) is not str or not record.task_id:
        return False
    if any(record.metadata.get(key) != value for key, value in metadata.items()):
        return False
    return record.configuration == configuration


def _select_existing(
    records: Sequence[ClearMLPipelineRecord],
    *,
    metadata: Mapping[str, str],
    configuration: Mapping[str, object],
) -> ClearMLPipelineRecord | None:
    records = tuple(records)
    if not records:
        return None
    if len(records) != 1:
        raise ClearMLPipelineError(
            "multiple ClearML Pipeline controllers claim one logical Study execution"
        )
    record = records[0]
    if not _matches(record, metadata=metadata, configuration=configuration):
        raise ClearMLPipelineError(
            "ClearML Pipeline ownership metadata or configuration mismatch"
        )
    return record


def _observation(
    *,
    record: ClearMLPipelineRecord,
    plan: StudyPlan,
    result: StudyResult,
) -> BackendStudyExecutionObservation:
    active_statuses = {"created", "queued", "in_progress", "publishing"}
    failed_statuses = {"failed"}
    cancelled_statuses = {"stopped"}
    completed_statuses = {"completed", "published", "closed"}
    raw = record.status
    if raw in active_statuses:
        state: Literal["active", "terminal"] = "active"
        status: Literal["active", "completed", "failed", "cancelled"] = "active"
    elif raw in failed_statuses:
        state, status = "terminal", "failed"
    elif raw in cancelled_statuses:
        state, status = "terminal", "cancelled"
    elif raw in completed_statuses:
        state, status = "terminal", "completed"
    else:
        raise ClearMLPipelineError(f"unsupported ClearML Pipeline status: {raw}")
    return {
        "state": state,
        "status": status,
        "key": _execution_key(plan, result),
        "backend": "clearml",
        "execution_id": record.task_id,
    }


class ClearMLPipelineService:
    """Idempotent creation/recovery of one ClearML controller per StudyResult."""

    def __init__(
        self,
        *,
        client: ClearMLPipelineClient,
        recovery_search_attempts: int = 3,
    ) -> None:
        if type(recovery_search_attempts) is not int or recovery_search_attempts < 1:
            raise ValueError("recovery_search_attempts must be a positive integer")
        self._client = client
        self._recovery_search_attempts = recovery_search_attempts

    def _search(
        self,
        *,
        project: str,
        ownership_key: str,
        metadata: Mapping[str, str],
        configuration: Mapping[str, object],
    ) -> ClearMLPipelineRecord | None:
        try:
            records = self._client.search_pipeline_runs(
                project=project, ownership_key=ownership_key
            )
        except Exception as error:
            raise ClearMLPipelineError("ClearML Pipeline ownership search failed") from error
        return _select_existing(
            records,
            metadata=metadata,
            configuration=configuration,
        )

    def ensure(
        self, *, plan: StudyPlan, study_result: StudyResult
    ) -> BackendStudyExecutionObservation:
        plan, study_result = _validate_pair(plan, study_result)
        project = f"mldb/{_namespace(study_result['id'])}"
        ownership = _pipeline_ownership_key(plan, study_result)
        metadata = _metadata(plan, study_result, ownership_key=ownership)
        configuration = _configuration(plan, study_result, ownership_key=ownership)
        existing = self._search(
            project=project,
            ownership_key=ownership,
            metadata=metadata,
            configuration=configuration,
        )
        if existing is not None:
            return _observation(record=existing, plan=plan, result=study_result)

        request = ClearMLPipelineCreateRequest(
            project=project,
            task_name=f"{str(plan['study']).split('/', 1)[1]} | {study_result['id'].split('/', 1)[1]}",
            metadata=metadata,
            configuration=configuration,
            source_commit=plan["source_commit"],
        )
        create_id: str | None = None
        create_error: Exception | None = None
        try:
            create_id = self._client.create_pipeline_run(request)
            if create_id is not None and (type(create_id) is not str or not create_id):
                raise TypeError("ClearML Pipeline create response must be null or non-empty ID")
        except Exception as error:
            create_error = error
            create_id = None

        found: ClearMLPipelineRecord | None = None
        for _ in range(self._recovery_search_attempts):
            found = self._search(
                project=project,
                ownership_key=ownership,
                metadata=metadata,
                configuration=configuration,
            )
            if found is not None:
                break
        if found is None:
            if create_error is not None:
                raise ClearMLPipelineError(
                    "ClearML Pipeline create outcome is ambiguous and ownership could not be recovered"
                ) from create_error
            raise ClearMLPipelineError(
                "ClearML Pipeline creation did not yield a recoverable ownership record"
            )
        if create_id is not None and found.task_id != create_id:
            raise ClearMLPipelineError(
                "ClearML Pipeline create response does not match recovered ownership"
            )
        return _observation(record=found, plan=plan, result=study_result)

    def project_summary(
        self,
        *,
        execution_id: str,
        summary: Mapping[str, object],
    ) -> None:
        if type(execution_id) is not str or not execution_id:
            raise ValueError("execution_id must be a non-empty string")
        if not isinstance(summary, Mapping):
            raise ValueError("summary must be a mapping")
        project = getattr(self._client, "project_pipeline_summary", None)
        if not callable(project):
            return
        try:
            project(execution_id=execution_id, summary=dict(summary))
        except Exception as error:
            raise ClearMLPipelineError("ClearML Pipeline summary projection failed") from error

    def cancel_execution(self, *, execution_id: str) -> None:
        if type(execution_id) is not str or not execution_id:
            raise ValueError("execution_id must be a non-empty string")
        cancel = getattr(self._client, "request_cancellation", None)
        if not callable(cancel):
            raise ClearMLPipelineError("ClearML Pipeline cancellation capability is unavailable")
        try:
            cancel(task_id=execution_id)
        except Exception as error:
            raise ClearMLPipelineError("ClearML Pipeline cancellation failed") from error

    def observe(
        self, *, plan: StudyPlan, study_result: StudyResult
    ) -> BackendStudyExecutionObservation | None:
        plan, study_result = _validate_pair(plan, study_result)
        project = f"mldb/{_namespace(study_result['id'])}"
        ownership = _pipeline_ownership_key(plan, study_result)
        metadata = _metadata(plan, study_result, ownership_key=ownership)
        configuration = _configuration(plan, study_result, ownership_key=ownership)
        record = self._search(
            project=project,
            ownership_key=ownership,
            metadata=metadata,
            configuration=configuration,
        )
        if record is None:
            return None
        return _observation(record=record, plan=plan, result=study_result)
