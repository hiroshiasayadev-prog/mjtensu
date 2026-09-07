"""MLDB Study Run record and lifecycle implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Literal

from ..common.errors import ValidationIssue, ValidationReport
from ..common.ids import StudyId, StudyRunId


class StudyRunStatus(str, Enum):
    """Persisted Study Run lifecycle state."""

    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class StudyRunExecution:
    """Execution facts owned directly by one Study Run."""

    started_at: object
    finished_at: object | None = None


@dataclass(frozen=True, slots=True)
class StudyRunPlan:
    """Integrity and count metadata for one complete immutable Study Run plan."""

    path: Literal["plan.jsonl"]
    sha256: str
    bytes: int
    trials: int
    evaluation_jobs: int


@dataclass(frozen=True, slots=True)
class StudyRunTrainingSummary:
    """Derived final outcome counts for planned training coordinates."""

    completed: int
    failed: int
    cancelled: int


@dataclass(frozen=True, slots=True)
class StudyRunEvaluationSummary:
    """Derived final outcome counts for planned evaluation coordinates."""

    completed: int
    completed_partial: int
    failed: int
    cancelled: int
    blocked: int


@dataclass(frozen=True, slots=True)
class StudyRunSummary:
    """Optional derived Study Run summary over final planned-coordinate outcomes."""

    training: StudyRunTrainingSummary | None = None
    evaluation: StudyRunEvaluationSummary | None = None


@dataclass(frozen=True, slots=True)
class StudyRun:
    """One persisted MLDB Study execution event."""

    schema: Literal["mjtensu.mldb/study-run/v1"]
    id: StudyRunId
    status: StudyRunStatus
    study: StudyId
    execution: StudyRunExecution
    plan: StudyRunPlan | None = None
    summary: StudyRunSummary | None = None


def validate_study_run(run: StudyRun) -> ValidationReport:
    """Validate Study Run metadata-local v1 invariants without external I/O."""

    issues: list[ValidationIssue] = []

    if run.schema != "mjtensu.mldb/study-run/v1":
        _add_issue(issues, "study_run.schema", "unsupported Study Run schema", "schema")

    if not isinstance(run.id, str) or _STUDY_RUN_ID.fullmatch(run.id) is None:
        _add_issue(
            issues,
            "study_run.id",
            "Study Run id must follow sr-YYYYMMDD-NNN",
            "id",
        )

    if not isinstance(run.status, StudyRunStatus):
        _add_issue(issues, "study_run.status", "invalid Study Run status", "status")
        status: StudyRunStatus | None = None
    else:
        status = run.status

    if status in _TERMINAL_STATUSES and run.execution.finished_at is None:
        _add_issue(
            issues,
            "study_run.execution.finished_at",
            "terminal Study Run requires execution.finished_at",
            "execution.finished_at",
        )

    if run.plan is not None:
        _validate_plan_metadata(run.plan, issues)

    if status in {
        StudyRunStatus.COMPLETED,
        StudyRunStatus.COMPLETED_WITH_FAILURES,
    } and run.plan is None:
        _add_issue(
            issues,
            "study_run.plan.required",
            "completed Study Run requires finalized plan metadata",
            "plan",
        )

    if run.summary is not None:
        _validate_summary(run.summary, issues)

    return ValidationReport(tuple(issues))


def validate_study_run_transition(
    source: StudyRunStatus,
    target: StudyRunStatus,
) -> ValidationReport:
    """Validate one requested Study Run lifecycle status transition."""

    if not isinstance(source, StudyRunStatus) or not isinstance(target, StudyRunStatus):
        return ValidationReport(
            (
                ValidationIssue(
                    code="study_run.transition.status",
                    message="source and target must be StudyRunStatus values",
                ),
            )
        )

    if source is StudyRunStatus.RUNNING and target in _TERMINAL_STATUSES:
        return ValidationReport()

    return ValidationReport(
        (
            ValidationIssue(
                code="study_run.transition.invalid",
                message=f"invalid Study Run transition: {source.value} -> {target.value}",
            ),
        )
    )


_STUDY_RUN_ID = re.compile(r"sr-[0-9]{8}-(?!000)[0-9]{3}\Z")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_TERMINAL_STATUSES = frozenset(
    {
        StudyRunStatus.COMPLETED,
        StudyRunStatus.COMPLETED_WITH_FAILURES,
        StudyRunStatus.FAILED,
        StudyRunStatus.CANCELLED,
    }
)


def _add_issue(
    issues: list[ValidationIssue],
    code: str,
    message: str,
    path: str | None = None,
) -> None:
    issues.append(ValidationIssue(code=code, message=message, path=path))


def _validate_plan_metadata(
    plan: object,
    issues: list[ValidationIssue],
) -> None:
    if not isinstance(plan, StudyRunPlan):
        _add_issue(
            issues,
            "study_run.plan.shape",
            "plan must be StudyRunPlan metadata",
            "plan",
        )
        return

    if plan.path != "plan.jsonl":
        _add_issue(
            issues,
            "study_run.plan.path",
            "plan.path must be exactly plan.jsonl",
            "plan.path",
        )
    if not isinstance(plan.sha256, str) or _SHA256.fullmatch(plan.sha256) is None:
        _add_issue(
            issues,
            "study_run.plan.sha256",
            "plan.sha256 must be a 64-character hexadecimal SHA-256",
            "plan.sha256",
        )
    _validate_non_negative_integer(plan.bytes, issues, "plan.bytes")
    _validate_non_negative_integer(plan.trials, issues, "plan.trials")
    _validate_non_negative_integer(plan.evaluation_jobs, issues, "plan.evaluation_jobs")


def _validate_summary(
    summary: object,
    issues: list[ValidationIssue],
) -> None:
    if not isinstance(summary, StudyRunSummary):
        _add_issue(
            issues,
            "study_run.summary.shape",
            "summary must be StudyRunSummary",
            "summary",
        )
        return

    if summary.training is not None:
        if not isinstance(summary.training, StudyRunTrainingSummary):
            _add_issue(
                issues,
                "study_run.summary.training.shape",
                "training summary must be StudyRunTrainingSummary",
                "summary.training",
            )
        else:
            _validate_non_negative_integer(
                summary.training.completed,
                issues,
                "summary.training.completed",
            )
            _validate_non_negative_integer(
                summary.training.failed,
                issues,
                "summary.training.failed",
            )
            _validate_non_negative_integer(
                summary.training.cancelled,
                issues,
                "summary.training.cancelled",
            )

    if summary.evaluation is not None:
        if not isinstance(summary.evaluation, StudyRunEvaluationSummary):
            _add_issue(
                issues,
                "study_run.summary.evaluation.shape",
                "evaluation summary must be StudyRunEvaluationSummary",
                "summary.evaluation",
            )
        else:
            _validate_non_negative_integer(
                summary.evaluation.completed,
                issues,
                "summary.evaluation.completed",
            )
            _validate_non_negative_integer(
                summary.evaluation.completed_partial,
                issues,
                "summary.evaluation.completed_partial",
            )
            _validate_non_negative_integer(
                summary.evaluation.failed,
                issues,
                "summary.evaluation.failed",
            )
            _validate_non_negative_integer(
                summary.evaluation.cancelled,
                issues,
                "summary.evaluation.cancelled",
            )
            _validate_non_negative_integer(
                summary.evaluation.blocked,
                issues,
                "summary.evaluation.blocked",
            )


def _validate_non_negative_integer(
    value: object,
    issues: list[ValidationIssue],
    path: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _add_issue(
            issues,
            "study_run.non_negative_integer",
            "value must be a non-negative integer and not boolean",
            path,
        )
