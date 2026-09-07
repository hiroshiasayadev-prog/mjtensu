"""Common MLDB validation results and expected application/domain errors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationIssue:
    """One machine-identifiable validation problem."""

    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Ordered validation issues produced for one validation scope."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether validation produced no issues."""
        return not self.issues


class MldbError(Exception):
    """Base for expected MLDB application/domain errors crossing shared boundaries."""


class NotFoundError(MldbError):
    """A requested typed MLDB entity or execution record does not exist."""


class InvalidRequestError(MldbError):
    """A required request shape or supplied ID/kind combination is invalid."""


class ValidationFailedError(MldbError):
    """A mutation/launch cannot proceed because domain validation failed."""

    report: ValidationReport

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("domain validation failed")


class UnsupportedOperationError(MldbError):
    """The requested application operation is not defined for the target kind."""


class LifecycleConflictError(MldbError):
    """A requested mutation conflicts with immutable or terminal lifecycle state."""
