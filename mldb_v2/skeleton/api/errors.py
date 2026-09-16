"""Stable transport-independent MLDB v2 application error value."""

from typing import Literal, TypeAlias, TypedDict

ApplicationErrorCode: TypeAlias = Literal[
    "not_found",
    "invalid_request",
    "validation_failed",
    "not_sealed",
    "source_not_pinned",
    "lifecycle_conflict",
    "backend_unavailable",
    "unsupported_capability",
    "result_rejected",
    "internal_failure",
]


class ApplicationError(TypedDict):
    code: ApplicationErrorCode
    message: str
