"""Training Run domain implementation for MLDB Wave I1-B."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Literal, Mapping

from ..common.errors import ValidationIssue, ValidationReport
from ..common.ids import (
    ArchitectureId,
    CorpusId,
    StudyRunId,
    TrainingRunId,
    TrainProtocolId,
)
from ..common.parameters import (
    PublicParameterValue,
    ResolvedPublicParameters,
    is_public_parameter_value,
)
from .weights import CanonicalWeightsArtifact


class TrainingRunStatus(str, Enum):
    """Persisted Training Run lifecycle state."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TrainingRunExecution:
    """Execution facts owned directly by one Training Run."""

    seed: int
    started_at: object
    finished_at: object | None = None


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Successful Training Run result metadata."""

    weights: CanonicalWeightsArtifact


@dataclass(frozen=True, slots=True)
class TrainingRunFailure:
    """Concise historical failure fact for a failed Training Run."""

    type: str
    message: str


@dataclass(frozen=True, slots=True)
class TrainingRunStudyLineage:
    """Optional downstream-to-upstream Study lineage for one Training Run."""

    run: StudyRunId
    trial: str


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """One persisted MLDB training execution attempt."""

    schema: Literal["mjtensu.mldb/training-run/v1"]
    id: TrainingRunId
    status: TrainingRunStatus
    corpus: CorpusId
    architecture: ArchitectureId
    train_protocol: TrainProtocolId
    parameters: ResolvedPublicParameters
    execution: TrainingRunExecution
    result: TrainingRunResult | None = None
    failure: TrainingRunFailure | None = None
    study: TrainingRunStudyLineage | None = None
    environment: Mapping[str, object] | None = None
    work: Mapping[str, object] | None = None


def validate_training_run(run: TrainingRun) -> ValidationReport:
    """Validate Training Run metadata-local v1 invariants without external I/O."""
    issues: list[ValidationIssue] = []

    if run.schema != "mjtensu.mldb/training-run/v1":
        issues.append(
            ValidationIssue(
                code="training_run.schema.unsupported",
                message="Training Run schema must be 'mjtensu.mldb/training-run/v1'.",
                path="schema",
            )
        )

    if type(run.id) is not str or not _TRAINING_RUN_ID_PATTERN.fullmatch(run.id):
        issues.append(
            ValidationIssue(
                code="training_run.id.invalid",
                message="Training Run id must match 'tr-YYYYMMDD-NNN' with a positive sequence.",
                path="id",
            )
        )

    if not isinstance(run.status, TrainingRunStatus):
        issues.append(
            ValidationIssue(
                code="training_run.status.invalid",
                message="Training Run status is outside the v1 lifecycle.",
                path="status",
            )
        )

    if type(run.execution.seed) is not int:
        issues.append(
            ValidationIssue(
                code="training_run.execution.seed.invalid",
                message="Training Run execution.seed must be an integer and must not be boolean.",
                path="execution.seed",
            )
        )

    for name, value in run.parameters.items():
        if type(name) is not str:
            issues.append(
                ValidationIssue(
                    code="training_run.parameter.name.invalid",
                    message="Training Run parameter names must be strings.",
                    path="parameters",
                )
            )
            continue
        if not is_public_parameter_value(value):
            issues.append(
                ValidationIssue(
                    code="training_run.parameter.value.invalid",
                    message=f"Training Run parameter {name!r} is outside the JSON-compatible value domain.",
                    path=f"parameters[{name!r}]",
                )
            )

    if run.status is TrainingRunStatus.RUNNING:
        if run.execution.finished_at is not None:
            issues.append(
                ValidationIssue(
                    code="training_run.execution.finished_at.forbidden",
                    message="A running Training Run must not have execution.finished_at.",
                    path="execution.finished_at",
                )
            )
    elif isinstance(run.status, TrainingRunStatus):
        if run.execution.finished_at is None:
            issues.append(
                ValidationIssue(
                    code="training_run.execution.finished_at.required",
                    message="A terminal Training Run requires execution.finished_at.",
                    path="execution.finished_at",
                )
            )

    if run.status is TrainingRunStatus.COMPLETED:
        if run.result is None:
            issues.append(
                ValidationIssue(
                    code="training_run.result.required",
                    message="A completed Training Run requires canonical result metadata.",
                    path="result",
                )
            )
    elif run.result is not None:
        issues.append(
            ValidationIssue(
                code="training_run.result.forbidden",
                message="Only a completed Training Run may carry canonical result metadata.",
                path="result",
            )
        )

    if run.result is not None:
        weights = run.result.weights
        if weights.format != "pytorch-state-dict":
            issues.append(
                ValidationIssue(
                    code="training_run.result.weights.format.invalid",
                    message="Canonical Training Run weights format must be 'pytorch-state-dict'.",
                    path="result.weights.format",
                )
            )
        if weights.path != "artifacts/weights.pt":
            issues.append(
                ValidationIssue(
                    code="training_run.result.weights.path.invalid",
                    message="Canonical Training Run weights path must be 'artifacts/weights.pt'.",
                    path="result.weights.path",
                )
            )
        if not _is_sha256(weights.sha256):
            issues.append(
                ValidationIssue(
                    code="training_run.result.weights.sha256.invalid",
                    message="Canonical Training Run weights sha256 must be a 64-character hexadecimal digest.",
                    path="result.weights.sha256",
                )
            )
        if type(weights.bytes) is not int or weights.bytes < 0:
            issues.append(
                ValidationIssue(
                    code="training_run.result.weights.bytes.invalid",
                    message="Canonical Training Run weights bytes must be a non-negative integer.",
                    path="result.weights.bytes",
                )
            )

    if run.failure is not None:
        if type(run.failure.type) is not str or run.failure.type == "":
            issues.append(
                ValidationIssue(
                    code="training_run.failure.type.invalid",
                    message="Training Run failure.type must be a non-empty string when present.",
                    path="failure.type",
                )
            )
        if type(run.failure.message) is not str or run.failure.message == "":
            issues.append(
                ValidationIssue(
                    code="training_run.failure.message.invalid",
                    message="Training Run failure.message must be a non-empty string when present.",
                    path="failure.message",
                )
            )

    if run.study is not None:
        if type(run.study.run) is not str or not _STUDY_RUN_ID_PATTERN.fullmatch(
            run.study.run
        ):
            issues.append(
                ValidationIssue(
                    code="training_run.study.run.invalid",
                    message="Training Run study.run must match 'sr-YYYYMMDD-NNN' with a positive sequence.",
                    path="study.run",
                )
            )
        if type(run.study.trial) is not str or not _TRIAL_ID_PATTERN.fullmatch(
            run.study.trial
        ):
            issues.append(
                ValidationIssue(
                    code="training_run.study.trial.invalid",
                    message="Training Run study.trial must match 'trial-NNNN' with a positive sequence.",
                    path="study.trial",
                )
            )

    return ValidationReport(tuple(issues))


def validate_training_run_transition(
    source: TrainingRunStatus,
    target: TrainingRunStatus,
) -> ValidationReport:
    """Validate one requested Training Run lifecycle status transition."""
    if (
        isinstance(source, TrainingRunStatus)
        and isinstance(target, TrainingRunStatus)
        and source is TrainingRunStatus.RUNNING
        and target
        in {
            TrainingRunStatus.COMPLETED,
            TrainingRunStatus.FAILED,
            TrainingRunStatus.CANCELLED,
        }
    ):
        return ValidationReport()

    return ValidationReport(
        (
            ValidationIssue(
                code="training_run.transition.invalid",
                message="Training Run v1 permits only RUNNING to a terminal status transition.",
                path="status",
            ),
        )
    )


_TRAINING_RUN_ID_PATTERN = re.compile(r"tr-[0-9]{8}-(?!000)[0-9]{3}\Z")
_STUDY_RUN_ID_PATTERN = re.compile(r"sr-[0-9]{8}-(?!000)[0-9]{3}\Z")
_TRIAL_ID_PATTERN = re.compile(r"trial-(?!0000)[0-9]{4}\Z")


def _is_sha256(value: object) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)
