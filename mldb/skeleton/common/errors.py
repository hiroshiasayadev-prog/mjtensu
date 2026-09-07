"""Common MLDB validation results and expected application/domain errors.

This skeleton fixes public signatures only. It intentionally excludes transport,
repository-I/O, worker-communication, and feature-specific execution failures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationIssue:
    """One machine-identifiable validation problem.

    ``code`` is a stable machine-readable string owned by the validator or
    contract that reports the issue. MLDB v1 does not define one global issue
    code enum.

    ``message`` is a concise human-readable explanation.

    ``path`` optionally identifies the logical field or reference location
    within the value being validated. It is not a filesystem path or transport
    address, and no global path grammar is fixed by MLDB v1.
    """

    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Ordered validation issues produced for one validation scope.

    An empty ``issues`` tuple represents successful validation. Every issue in
    this v1 report is validation-significant; severity levels are intentionally
    not part of the common contract.
    """

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether validation produced no issues."""
        ...


class MldbError(Exception):
    """Base for expected MLDB application/domain errors crossing shared boundaries.

    This is not a catch-all wrapper for infrastructure failures, arbitrary
    protocol exceptions, worker communication failures, or feature-specific
    execution details.
    """

    ...


class NotFoundError(MldbError):
    """A requested typed MLDB entity or execution record does not exist."""

    ...


class InvalidRequestError(MldbError):
    """A required request shape or supplied ID/kind combination is invalid."""

    ...


class ValidationFailedError(MldbError):
    """A mutation/launch cannot proceed because domain validation failed.

    Read-only validation returns ``ValidationReport`` directly instead of
    raising this exception.
    """

    report: ValidationReport

    def __init__(self, report: ValidationReport) -> None:
        ...


class UnsupportedOperationError(MldbError):
    """The requested application operation is not defined for the target kind."""

    ...


class LifecycleConflictError(MldbError):
    """A requested mutation conflicts with immutable or terminal lifecycle state."""

    ...
