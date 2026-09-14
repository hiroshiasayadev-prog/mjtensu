"""Read-only composition for MLDB v2 discovery, progress, observation, logs, and diagnosis."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Collection, Mapping, cast

from mldb_v2.src.backend.backend_port import BackendPort
from mldb_v2.src.backend.candidate_outcome import BackendObservation, StageKey
from mldb_v2.src.common.ids import (
    EntityKind,
    EvaluationCoordinateId,
    NamespaceId,
    StudyId,
    StudyResultId,
    TrialId,
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
)
from mldb_v2.src.repository.listing import CanonicalListing, CanonicalRepositoryListing
from mldb_v2.src.repository.resolution import (
    CanonicalDocument,
    CanonicalEntityId,
    CanonicalRepositoryResolver,
)
from mldb_v2.src.results.study_result import (
    StageDisposition,
    StudyResult,
    StudyResultStatus,
    _validate_study_result,
)
from mldb_v2.src.verification.definition_lifecycle import DefinitionLifecycleStatus

from .query_interface import (
    BackendLogChunk,
    BackendLogRequest,
    DiagnosisCheck,
    EntityResource,
    ProgressCounter,
    StudyObservation,
    StudyProgress,
    StudyResultView,
)


class _InvalidQueryRequest(ValueError):
    """Bounded invalid query/filter request for later public error mapping."""


class _UnsupportedQueryCapability(RuntimeError):
    """Bounded optional-capability failure for later public error mapping."""


class _QueryBackendFailure(RuntimeError):
    """Bounded backend-observation contract failure."""


_DEFINITION_KINDS: tuple[EntityKind, ...] = (
    EntityKind.TASK,
    EntityKind.CORPUS,
    EntityKind.ARCHITECTURE,
    EntityKind.TRAIN_PROTOCOL,
    EntityKind.EVALUATION_PROTOCOL,
    EntityKind.STUDY,
)
_DEFINITION_KIND_INDEX = {kind: index for index, kind in enumerate(_DEFINITION_KINDS)}
_LIFECYCLE_STATUSES = frozenset({"draft", "sealed"})
_DISPOSITIONS: tuple[StageDisposition, ...] = (
    "pending",
    "completed",
    "failed",
    "cancelled",
    "skipped",
)
_LOG_REQUEST_FIELDS = {"study_result", "trial", "coordinate", "failed_only", "follow"}


def _diagnostic(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _validate_lifecycle_filter(
    status: DefinitionLifecycleStatus | None,
) -> DefinitionLifecycleStatus | None:
    if status is None:
        return None
    if type(status) is not str or status not in _LIFECYCLE_STATUSES:
        raise _InvalidQueryRequest("status must be exactly 'draft' or 'sealed'")
    return cast(DefinitionLifecycleStatus, status)


def _deduplicate_issues(issues: Iterable[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    result: list[Mapping[str, object]] = []
    seen: set[tuple[object, object]] = set()
    for issue in issues:
        key = (issue.get("code"), issue.get("message"))
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return tuple(result)


def _filter_lifecycle(
    listing: CanonicalListing,
    *,
    status: DefinitionLifecycleStatus | None,
) -> CanonicalListing:
    if status is None:
        return listing
    items: list[CanonicalDocument] = []
    issues: list[Mapping[str, object]] = list(listing["issues"])
    for document in listing["items"]:
        current = document.get("status")
        if type(current) is not str or current not in _LIFECYCLE_STATUSES:
            issues.append(
                _diagnostic(
                    "query_definition_status_invalid",
                    f"{document.get('id')!r}: reusable definition has invalid lifecycle status",
                )
            )
            continue
        if current == status:
            items.append(document)
    return {
        "items": tuple(items),
        "issues": cast(tuple, _deduplicate_issues(issues)),
    }


def _empty_counter() -> ProgressCounter:
    return {
        "planned": 0,
        "pending": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "skipped": 0,
    }


def _finish_counter(counter: ProgressCounter) -> ProgressCounter:
    counter["planned"] = sum(counter[name] for name in _DISPOSITIONS)
    return counter


def _derive_progress(result: StudyResult) -> StudyProgress:
    training = _empty_counter()
    evaluations = _empty_counter()
    for trial in result["trials"]:
        training_slot = trial["training"]
        if training_slot is not None:
            training[training_slot["disposition"]] += 1
        for evaluation in trial["evaluations"]:
            evaluations[evaluation["disposition"]] += 1

    _finish_counter(training)
    _finish_counter(evaluations)
    total = _empty_counter()
    for name in _DISPOSITIONS:
        total[name] = training[name] + evaluations[name]
    _finish_counter(total)
    return {"training": training, "evaluations": evaluations, "total": total}


def _stage_key(
    result: StudyResult,
    *,
    trial: TrialId,
    kind: str,
    coordinate: EvaluationCoordinateId | None,
) -> StageKey:
    if kind == "training":
        return {
            "study_result": result["id"],
            "plan": result["plan"],
            "trial": trial,
            "kind": "training",
            "coordinate": None,
            "source_commit": result["source_commit"],
        }
    return {
        "study_result": result["id"],
        "plan": result["plan"],
        "trial": trial,
        "kind": "evaluation",
        "coordinate": cast(EvaluationCoordinateId, coordinate),
        "source_commit": result["source_commit"],
    }


def _pending_stage_keys(result: StudyResult) -> list[StageKey]:
    keys: list[StageKey] = []
    for trial in result["trials"]:
        training = trial["training"]
        if training is not None and training["disposition"] == "pending":
            keys.append(
                _stage_key(
                    result,
                    trial=trial["trial"],
                    kind="training",
                    coordinate=None,
                )
            )
        for evaluation in trial["evaluations"]:
            if evaluation["disposition"] != "pending":
                continue
            keys.append(
                _stage_key(
                    result,
                    trial=trial["trial"],
                    kind="evaluation",
                    coordinate=evaluation["coordinate"],
                )
            )
    return keys


def _selected_stage_keys(
    result: StudyResult,
    *,
    trial_filter: TrialId | None,
    coordinate_filter: EvaluationCoordinateId | None,
) -> list[StageKey]:
    keys: list[StageKey] = []
    for trial in result["trials"]:
        if trial_filter is not None and trial["trial"] != trial_filter:
            continue
        if coordinate_filter is None and trial["training"] is not None:
            keys.append(
                _stage_key(
                    result,
                    trial=trial["trial"],
                    kind="training",
                    coordinate=None,
                )
            )
        for evaluation in trial["evaluations"]:
            if coordinate_filter is not None and evaluation["coordinate"] != coordinate_filter:
                continue
            keys.append(
                _stage_key(
                    result,
                    trial=trial["trial"],
                    kind="evaluation",
                    coordinate=evaluation["coordinate"],
                )
            )
    return keys


def _validate_observation(
    observed: BackendObservation | None,
    *,
    requested: StageKey,
) -> BackendObservation | None:
    if observed is None:
        return None
    if not isinstance(observed, dict) or observed.get("stage_key") != requested:
        raise _QueryBackendFailure("backend observation does not match requested StageKey")
    return observed


def _execution_ids(
    observation: BackendObservation,
    *,
    failed_only: bool,
) -> list[str]:
    if observation["state"] == "active":
        if failed_only:
            return []
        raw_ids = observation.get("execution_ids")
        if type(raw_ids) is not list:
            raise _QueryBackendFailure("active backend observation has invalid execution_ids")
        if any(type(value) is not str or not value for value in raw_ids):
            raise _QueryBackendFailure("backend execution IDs must be non-empty strings")
        return list(raw_ids)

    attempts = observation.get("attempts")
    if type(attempts) is not list:
        raise _QueryBackendFailure("terminal backend observation has invalid attempts")
    result: list[str] = []
    for attempt in attempts:
        if type(attempt) is not dict:
            raise _QueryBackendFailure("terminal backend attempt is invalid")
        if failed_only and attempt.get("status") != "failed":
            continue
        execution_id = attempt.get("execution_id")
        if execution_id is None:
            continue
        if type(execution_id) is not str or not execution_id:
            raise _QueryBackendFailure("backend execution ID must be null or non-empty")
        result.append(execution_id)
    return result


def _normalize_log_chunks(value: object, *, execution_id: str | None) -> tuple[BackendLogChunk, ...]:
    if value is None:
        return ()
    chunks: object
    if isinstance(value, Mapping):
        chunks = value.get("chunks")
    else:
        chunks = getattr(value, "chunks", None)
    if not isinstance(chunks, (tuple, list)):
        raise _QueryBackendFailure("backend log capability returned invalid chunks")
    result: list[BackendLogChunk] = []
    for chunk in chunks:
        if type(chunk) is not str:
            raise _QueryBackendFailure("backend log chunk must be text")
        result.append({"execution_id": execution_id, "text": chunk})
    return tuple(result)


def _validate_direct_log_chunks(value: object) -> tuple[BackendLogChunk, ...]:
    try:
        values = tuple(cast(Iterable[object], value))
    except TypeError as error:
        raise _QueryBackendFailure("backend log capability must return an iterable") from error
    result: list[BackendLogChunk] = []
    for chunk in values:
        if type(chunk) is not dict or set(chunk) != {"execution_id", "text"}:
            raise _QueryBackendFailure("backend log chunk fields do not match query contract")
        execution_id = chunk["execution_id"]
        text = chunk["text"]
        if execution_id is not None and (type(execution_id) is not str or not execution_id):
            raise _QueryBackendFailure("backend log execution_id must be null or non-empty")
        if type(text) is not str:
            raise _QueryBackendFailure("backend log text must be a string")
        result.append({"execution_id": execution_id, "text": text})
    return tuple(result)


class ReadOnlyQueryService:
    """Concrete read-only QueryInterface composition over W001 and BackendPort."""

    def __init__(
        self,
        *,
        mldb_data_root: str | Path,
        backend: BackendPort,
        diagnostic_probes: Sequence[Callable[[], DiagnosisCheck]] = (),
    ) -> None:
        self._root = Path(mldb_data_root)
        self._listing = CanonicalRepositoryListing(self._root)
        self._resolver = CanonicalRepositoryResolver(self._root)
        self._backend = backend
        self._diagnostic_probes = tuple(diagnostic_probes)

    def list_entities(
        self,
        *,
        resource: EntityResource,
        namespace: NamespaceId | None = None,
        status: DefinitionLifecycleStatus | None = None,
    ) -> CanonicalListing:
        lifecycle = _validate_lifecycle_filter(status)
        if resource == "definitions":
            records: list[tuple[str, int, str, CanonicalDocument]] = []
            issues: list[Mapping[str, object]] = []
            for kind in _DEFINITION_KINDS:
                listing = self._listing.list_entities(kind=kind, namespace=namespace)
                filtered = _filter_lifecycle(listing, status=lifecycle)
                issues.extend(filtered["issues"])
                for document in filtered["items"]:
                    entity_id = str(document["id"])
                    namespace_id = entity_id.split("/", 1)[0]
                    records.append(
                        (
                            namespace_id,
                            _DEFINITION_KIND_INDEX[kind],
                            entity_id,
                            document,
                        )
                    )
            records.sort(key=lambda item: (item[0], item[1], item[2]))
            return {
                "items": tuple(record[3] for record in records),
                "issues": cast(tuple, _deduplicate_issues(issues)),
            }

        if not isinstance(resource, EntityKind):
            raise _InvalidQueryRequest(f"unsupported entity resource: {resource!r}")
        if lifecycle is not None and resource not in _DEFINITION_KINDS:
            raise _InvalidQueryRequest(
                f"lifecycle status filter is unsupported for resource {resource.value!r}"
            )
        listing = self._listing.list_entities(kind=resource, namespace=namespace)
        return _filter_lifecycle(listing, status=lifecycle)

    def get_entity(
        self,
        *,
        kind: EntityKind,
        entity_id: CanonicalEntityId,
    ) -> CanonicalDocument:
        return self._resolver.resolve(kind=kind, entity_id=entity_id)

    def list_study_results(
        self,
        *,
        namespace: NamespaceId | None = None,
        study: StudyId | None = None,
        statuses: Collection[StudyResultStatus] | None = None,
        created_at_from: datetime | None = None,
        created_at_to: datetime | None = None,
        limit: int | None = None,
    ) -> CanonicalListing:
        for name, value in (
            ("created_at_from", created_at_from),
            ("created_at_to", created_at_to),
        ):
            if value is not None and (
                not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None
            ):
                raise _InvalidQueryRequest(f"{name} must be timezone-aware")
        if (
            created_at_from is not None
            and created_at_to is not None
            and created_at_from > created_at_to
        ):
            raise _InvalidQueryRequest("created_at_from must not be after created_at_to")
        return self._listing.list_study_results(
            namespace=namespace,
            study=study,
            statuses=statuses,
            created_at_from=created_at_from,
            created_at_to=created_at_to,
            limit=limit,
        )

    def get_study_result(
        self,
        *,
        study_result: StudyResultId,
    ) -> StudyResultView:
        document = self._resolver.resolve(
            kind=EntityKind.STUDY_RESULT,
            entity_id=study_result,
        )
        result = _validate_study_result(document)
        return {"study_result": result, "progress": _derive_progress(result)}

    def observe_study(
        self,
        *,
        study_result: StudyResultId,
    ) -> StudyObservation:
        view = self.get_study_result(study_result=study_result)
        result = view["study_result"]
        observations: list[BackendObservation] = []
        for key in _pending_stage_keys(result):
            observed = _validate_observation(
                self._backend.observe(stage_key=copy.deepcopy(key)),
                requested=key,
            )
            if observed is not None:
                observations.append(copy.deepcopy(observed))
        return {
            "study_result": result,
            "progress": view["progress"],
            "backend_observations": observations,
        }

    def read_backend_logs(
        self,
        *,
        request: BackendLogRequest,
    ) -> Iterable[BackendLogChunk]:
        if type(request) is not dict or set(request) != _LOG_REQUEST_FIELDS:
            raise _InvalidQueryRequest("backend log request fields do not match contract")
        if type(request["failed_only"]) is not bool or type(request["follow"]) is not bool:
            raise _InvalidQueryRequest("failed_only and follow must be booleans")
        trial = request["trial"]
        coordinate = request["coordinate"]
        if trial is not None:
            try:
                trial = _validate_trial_id(trial)
            except ValueError as error:
                raise _InvalidQueryRequest("invalid trial filter") from error
        if coordinate is not None:
            try:
                coordinate = _validate_evaluation_coordinate_id(coordinate)
            except ValueError as error:
                raise _InvalidQueryRequest("invalid evaluation coordinate filter") from error

        direct_reader = getattr(self._backend, "read_backend_logs", None)
        if callable(direct_reader):
            value = direct_reader(request=copy.deepcopy(request))
            return _validate_direct_log_chunks(value)

        task_reader = getattr(self._backend, "read_task_logs", None)
        if not callable(task_reader):
            raise _UnsupportedQueryCapability("backend does not support log access")
        if request["follow"]:
            raise _UnsupportedQueryCapability(
                "backend log capability does not support follow mode"
            )

        view = self.get_study_result(study_result=request["study_result"])
        result = view["study_result"]
        stage_keys = _selected_stage_keys(
            result,
            trial_filter=cast(TrialId | None, trial),
            coordinate_filter=cast(EvaluationCoordinateId | None, coordinate),
        )
        chunks: list[BackendLogChunk] = []
        seen: set[str] = set()
        for key in stage_keys:
            observed = _validate_observation(
                self._backend.observe(stage_key=copy.deepcopy(key)),
                requested=key,
            )
            if observed is None:
                continue
            for execution_id in _execution_ids(
                observed,
                failed_only=request["failed_only"],
            ):
                if execution_id in seen:
                    continue
                seen.add(execution_id)
                record = task_reader(
                    study_result=result["id"],
                    task_id=execution_id,
                )
                chunks.extend(
                    _normalize_log_chunks(record, execution_id=execution_id)
                )
        return tuple(chunks)

    def diagnose(self) -> list[DiagnosisCheck]:
        checks: list[DiagnosisCheck] = []
        if self._root.is_dir():
            checks.append(
                {"name": "repository", "status": "ok", "diagnostic": None}
            )
        else:
            checks.append(
                {
                    "name": "repository",
                    "status": "error",
                    "diagnostic": _diagnostic(
                        "repository_unavailable",
                        "canonical repository root is unavailable",
                    ),
                }
            )

        log_reader = getattr(self._backend, "read_backend_logs", None)
        task_reader = getattr(self._backend, "read_task_logs", None)
        if callable(log_reader) or callable(task_reader):
            checks.append(
                {"name": "backend_logs", "status": "ok", "diagnostic": None}
            )
        else:
            checks.append(
                {
                    "name": "backend_logs",
                    "status": "unsupported",
                    "diagnostic": _diagnostic(
                        "unsupported_capability",
                        "backend log access is unavailable",
                    ),
                }
            )

        for index, probe in enumerate(self._diagnostic_probes, start=1):
            try:
                check = probe()
            except Exception:
                checks.append(
                    {
                        "name": f"probe-{index}",
                        "status": "error",
                        "diagnostic": _diagnostic(
                            "diagnostic_probe_failed",
                            "diagnostic probe failed",
                        ),
                    }
                )
                continue
            if (
                type(check) is not dict
                or set(check) != {"name", "status", "diagnostic"}
                or type(check["name"]) is not str
                or not check["name"]
                or check["status"] not in {"ok", "warning", "error", "unsupported"}
            ):
                checks.append(
                    {
                        "name": f"probe-{index}",
                        "status": "error",
                        "diagnostic": _diagnostic(
                            "diagnostic_probe_invalid",
                            "diagnostic probe returned an invalid check",
                        ),
                    }
                )
                continue
            checks.append(copy.deepcopy(check))
        return checks
