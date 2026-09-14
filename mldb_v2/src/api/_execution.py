"""Application-side Study execution composition over injected W006 authority."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, TypedDict

from mldb_v2.src.backend.candidate_outcome import StageKey
from mldb_v2.src.common.ids import (
    EntityKind,
    EvaluationResultId,
    ExecutionKey,
    ModelId,
    StudyId,
    StudyPlanId,
    StudyResultId,
    TrainingResultId,
    _validate_typed_reference,
)
from mldb_v2.src.repository.canonical_writes import CanonicalRepositoryWriter
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.results.study_result import StudyResult, StudyResultStatus, _validate_study_result
from mldb_v2.src.study._plan_build import _validate_study_plan
from mldb_v2.src.study.plan import StudyPlan

from ._errors import _ApplicationBoundaryError, _application_error

FinalizedResultId: TypeAlias = TrainingResultId | EvaluationResultId


class StudyAdvanceResponse(TypedDict):
    study_result: StudyResultId
    status: StudyResultStatus
    changed: bool
    admitted: list[StageKey]
    finalized_results: list[FinalizedResultId]
    finalized_models: list[ModelId]
    active: list[StageKey]
    terminal: bool


CancelStudyOutcome: TypeAlias = Literal["accepted", "already_cancelling", "already_terminal"]


class CancelStudyResponse(TypedDict):
    study_result: StudyResultId
    outcome: CancelStudyOutcome
    status: StudyResultStatus


_STUDY_STATUSES = {
    "submitted", "cancelling", "completed", "completed_with_failures", "failed", "cancelled"
}
_TERMINAL_STATUSES = {"completed", "completed_with_failures", "failed", "cancelled"}
_CANCEL_OUTCOMES = {"accepted", "already_cancelling", "already_terminal"}


class _PlanStudy(Protocol):
    def __call__(self, *, study: StudyId) -> StudyPlan: ...


class _AdvanceOnePass(Protocol):
    def __call__(self, *, study_result: StudyResultId) -> StudyAdvanceResponse: ...


class _CancelRequest(Protocol):
    def __call__(self, *, study_result: StudyResultId) -> Mapping[str, object]: ...


class _StudyResultOnlyValidator:
    """Guard: this shell owns no immutable child-record persistence."""

    def validate(self, *, kind, entity_id, document) -> None:
        raise AssertionError("execution shell must not create immutable child records")


def _utc_created_at() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _invalid_request(message: str) -> _ApplicationBoundaryError:
    return _ApplicationBoundaryError(_application_error("invalid_request", message))


def _validate_request_reference(value: object, *, label: str) -> str:
    try:
        return _validate_typed_reference(value)
    except ValueError as error:
        raise _invalid_request(f"{label} identity is invalid") from error


def _validate_backend_name(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("backend must be a non-empty generic backend name")
    return value


def _validate_request_backend_name(value: object) -> str:
    try:
        return _validate_backend_name(value)
    except ValueError as error:
        raise _invalid_request("backend name is invalid") from error


def _validate_execution_key(value: object) -> ExecutionKey:
    if type(value) is not str or len(value) != 32:
        raise ValueError("execution key must be 32 lowercase hexadecimal characters")
    try:
        parsed = uuid.UUID(hex=value)
    except (ValueError, AttributeError) as error:
        raise ValueError("execution key must encode UUID4") from error
    if parsed.hex != value or parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise ValueError("execution key must encode UUID4")
    return ExecutionKey(value)


def _validate_request_execution_key(value: object) -> ExecutionKey:
    try:
        return _validate_execution_key(value)
    except ValueError as error:
        raise _invalid_request("execution key is invalid") from error


def _adapt_cancel_response(
    value: Mapping[str, object], *, expected_id: StudyResultId
) -> CancelStudyResponse:
    if type(value) is not dict or set(value) != {"study_result", "outcome", "status"}:
        raise ValueError("cancellation seam returned invalid response shape")
    study_result = StudyResultId(_validate_typed_reference(value["study_result"]))
    if study_result != expected_id:
        raise ValueError("cancellation seam returned different StudyResult identity")
    outcome = value["outcome"]
    status = value["status"]
    if type(outcome) is not str or outcome not in _CANCEL_OUTCOMES:
        raise ValueError("cancellation seam returned invalid outcome")
    if type(status) is not str or status not in _STUDY_STATUSES:
        raise ValueError("cancellation seam returned invalid StudyResult status")
    return {
        "study_result": study_result,
        "outcome": outcome,  # type: ignore[typeddict-item]
        "status": status,  # type: ignore[typeddict-item]
    }


def _study_result_id(plan: StudyPlan, execution_key: ExecutionKey) -> StudyResultId:
    namespace = str(plan["study"]).split("/", 1)[0]
    return StudyResultId(
        _validate_typed_reference(f"{namespace}/run-{execution_key}")
    )


def _initial_study_result(
    *,
    plan: StudyPlan,
    backend: str,
    execution_key: ExecutionKey,
    created_at: str,
) -> StudyResult:
    result_id = _study_result_id(plan, execution_key)
    trials: list[dict[str, object]] = []
    for plan_trial in plan["trials"]:
        source = plan_trial["source"]
        training = None
        if source["kind"] == "training":
            training = {"disposition": "pending", "result": None, "reason": None}
        evaluations = [
            {
                "coordinate": item["coordinate"],
                "stage": item["stage"],
                "disposition": "pending",
                "result": None,
                "reason": None,
            }
            for item in plan_trial["evaluations"]
        ]
        trials.append(
            {"trial": plan_trial["trial"], "training": training, "evaluations": evaluations}
        )

    return _validate_study_result(
        {
            "schema": "mjtensu.mldb-v2/study-result/v1",
            "id": result_id,
            "execution_key": execution_key,
            "plan": plan["id"],
            "study": plan["study"],
            "source_commit": plan["source_commit"],
            "backend": backend,
            "created_at": created_at,
            "status": "submitted",
            "diagnostic": None,
            "trials": trials,
        }
    )


def _same_start_identity(
    existing: StudyResult,
    *,
    plan: StudyPlan,
    backend: str,
    execution_key: ExecutionKey,
) -> bool:
    return (
        existing["execution_key"] == execution_key
        and existing["plan"] == plan["id"]
        and existing["study"] == plan["study"]
        and existing["source_commit"] == plan["source_commit"]
        and existing["backend"] == backend
    )


class _ExecutionCompositionShell:
    """Thin Application execution shell; W006 remains the progression authority."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        plan_study: _PlanStudy,
        advance_one_pass: _AdvanceOnePass,
        request_cancel: _CancelRequest,
        wait: Callable[[], None] = lambda: None,
        execution_key_factory: Callable[[], uuid.UUID] = uuid.uuid4,
        created_at_factory: Callable[[], str] = _utc_created_at,
    ) -> None:
        self._repository_root = Path(repository_root)
        self._resolver = CanonicalRepositoryResolver(self._repository_root / "mldb_data")
        self._writer = CanonicalRepositoryWriter(
            self._repository_root,
            record_validator=_StudyResultOnlyValidator(),
        )
        self._plan_study = plan_study
        self._advance_one_pass = advance_one_pass
        self._request_cancel = request_cancel
        self._wait = wait
        self._execution_key_factory = execution_key_factory
        self._created_at_factory = created_at_factory

    def _load_plan(self, plan_id: StudyPlanId | str) -> StudyPlan:
        validated_id = StudyPlanId(_validate_typed_reference(plan_id))
        document = self._resolver.resolve(kind=EntityKind.STUDY_PLAN, entity_id=validated_id)
        plan = _validate_study_plan(document)
        if plan["id"] != validated_id:
            raise ValueError("resolved StudyPlan identity mismatch")
        return plan

    def _load_result(self, study_result: StudyResultId | str) -> StudyResult:
        validated_id = StudyResultId(_validate_typed_reference(study_result))
        document = self._resolver.resolve(
            kind=EntityKind.STUDY_RESULT,
            entity_id=validated_id,
        )
        result = _validate_study_result(document)
        if result["id"] != validated_id:
            raise ValueError("resolved StudyResult identity mismatch")
        return result

    def _fresh_execution_key(self) -> ExecutionKey:
        value = self._execution_key_factory()
        if not isinstance(value, uuid.UUID):
            raise TypeError("execution_key_factory must return UUID")
        return _validate_execution_key(value.hex)

    def _start_study_validated(
        self,
        *,
        exact_plan: StudyPlan,
        backend_name: str,
        key: ExecutionKey,
    ) -> StudyResult:
        result_id = _study_result_id(exact_plan, key)

        try:
            existing = self._load_result(result_id)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not _same_start_identity(
                existing,
                plan=exact_plan,
                backend=backend_name,
                execution_key=key,
            ):
                raise ValueError(
                    "lifecycle conflict: execution key already used with different Plan/backend"
                )
            return existing

        initial = _initial_study_result(
            plan=exact_plan,
            backend=backend_name,
            execution_key=key,
            created_at=self._created_at_factory(),
        )
        try:
            stored = self._writer.create_study_result(
                entity_id=result_id,
                document=initial,
            )
        except ValueError as error:
            if "lifecycle conflict" not in str(error):
                raise
            existing = self._load_result(result_id)
            if not _same_start_identity(
                existing,
                plan=exact_plan,
                backend=backend_name,
                execution_key=key,
            ):
                raise
            return existing
        return _validate_study_result(stored)

    def start_study(
        self,
        *,
        plan: StudyPlanId | str,
        backend: str,
        execution_key: ExecutionKey | str,
    ) -> StudyResult:
        plan_id = StudyPlanId(_validate_request_reference(plan, label="Plan"))
        backend_name = _validate_request_backend_name(backend)
        key = _validate_request_execution_key(execution_key)
        exact_plan = self._load_plan(plan_id)
        return self._start_study_validated(
            exact_plan=exact_plan,
            backend_name=backend_name,
            key=key,
        )

    def advance_study(self, *, study_result: StudyResultId | str) -> StudyAdvanceResponse:
        result_id = StudyResultId(
            _validate_request_reference(study_result, label="StudyResult")
        )
        return self._advance_one_pass(study_result=result_id)

    def _drive_to_terminal(self, study_result: StudyResultId) -> StudyResult:
        while True:
            response = self.advance_study(study_result=study_result)
            if response["terminal"]:
                return self._load_result(study_result)
            self._wait()

    def run_study(self, *, study: StudyId | str, backend: str) -> StudyResult:
        study_id = StudyId(_validate_request_reference(study, label="Study"))
        backend_name = _validate_request_backend_name(backend)
        plan = _validate_study_plan(self._plan_study(study=study_id))
        if plan["study"] != study_id:
            raise ValueError("planned Study does not match requested Study")
        started = self._start_study_validated(
            exact_plan=plan,
            backend_name=backend_name,
            key=self._fresh_execution_key(),
        )
        return self._drive_to_terminal(started["id"])

    def resume_study(self, *, study_result: StudyResultId | str) -> StudyResult:
        result_id = StudyResultId(
            _validate_request_reference(study_result, label="StudyResult")
        )
        current = self._load_result(result_id)
        if current["status"] in _TERMINAL_STATUSES:
            raise ValueError("lifecycle conflict: cannot resume terminal StudyResult")
        return self._drive_to_terminal(current["id"])

    def rerun_study(
        self,
        *,
        source: StudyResultId | str,
        backend: str | None = None,
    ) -> StudyResult:
        source_id = StudyResultId(_validate_request_reference(source, label="StudyResult"))
        backend_override = (
            None if backend is None else _validate_request_backend_name(backend)
        )
        source_result = self._load_result(source_id)
        exact_plan = self._load_plan(source_result["plan"])
        if (
            exact_plan["study"] != source_result["study"]
            or exact_plan["source_commit"] != source_result["source_commit"]
        ):
            raise ValueError("source StudyResult does not match its referenced immutable Plan")
        selected_backend = (
            source_result["backend"] if backend_override is None else backend_override
        )
        started = self._start_study_validated(
            exact_plan=exact_plan,
            backend_name=selected_backend,
            key=self._fresh_execution_key(),
        )
        return self._drive_to_terminal(started["id"])

    def cancel_study(self, *, study_result: StudyResultId | str) -> CancelStudyResponse:
        result_id = StudyResultId(
            _validate_request_reference(study_result, label="StudyResult")
        )
        return _adapt_cancel_response(
            self._request_cancel(study_result=result_id),
            expected_id=result_id,
        )
