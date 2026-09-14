"""Private bounded translation into the stable ApplicationError value."""

from __future__ import annotations

from mldb_v2.src.study._plan_build import _StudyPlanError
from mldb_v2.src.study._planning_preflight import _PlanningPreflightError
from mldb_v2.src.study._source_pinning import _SourcePinningError

from .errors import ApplicationError, ApplicationErrorCode


class _ApplicationBoundaryError(RuntimeError):
    """Raised by component services with one already-bounded public error value."""

    def __init__(self, error: ApplicationError) -> None:
        self.error = error
        super().__init__(error["message"])


def _application_error(code: ApplicationErrorCode, message: str) -> ApplicationError:
    return {"code": code, "message": message}


def _code_for_lower_error(error: BaseException) -> ApplicationErrorCode:
    if isinstance(error, FileNotFoundError):
        return "not_found"
    if isinstance(error, _PlanningPreflightError):
        if error.code == "study_not_sealed":
            return "not_sealed"
        if error.code == "study_unavailable":
            return "not_found"
        return "validation_failed"
    if isinstance(error, _SourcePinningError):
        return "source_not_pinned"
    if isinstance(error, _StudyPlanError):
        return "validation_failed"
    if isinstance(error, ValueError):
        text = str(error).lower()
        if "lifecycle conflict" in text or "not a draft" in text or "became stale" in text:
            return "lifecycle_conflict"
        return "validation_failed"
    return "internal_failure"


def _message_for_code(code: ApplicationErrorCode) -> str:
    return {
        "not_found": "requested canonical object was not found",
        "invalid_request": "application request is invalid",
        "validation_failed": "canonical validation rejected the operation",
        "not_sealed": "requested Study must be sealed",
        "source_not_pinned": "required committed source inputs are not pinned",
        "lifecycle_conflict": "requested mutation conflicts with canonical lifecycle state",
        "backend_unavailable": "required backend operation is unavailable",
        "unsupported_capability": "requested capability is unsupported",
        "result_rejected": "backend result was rejected by MLDB acceptance",
        "internal_failure": "unexpected application failure",
    }[code]


def _translate_application_error(error: BaseException) -> ApplicationError:
    if isinstance(error, _ApplicationBoundaryError):
        return error.error
    code = _code_for_lower_error(error)
    return _application_error(code, _message_for_code(code))


def _raise_application_error(error: BaseException) -> None:
    raise _ApplicationBoundaryError(_translate_application_error(error)) from error
