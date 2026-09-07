"""Public Python signatures for MLDB Evaluation Run records and lifecycle rules."""

from __future__ import annotations

import datetime as _datetime
import math
import re
from collections.abc import Mapping as _Mapping, Sequence as _Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Literal, Mapping, Sequence

from ..common.errors import ValidationIssue, ValidationReport
from ..common.ids import (
    CorpusId,
    EvaluationProtocolId,
    EvaluationRunId,
    ModelId,
    StudyRunId,
)
from ..common.parameters import (
    PublicParameterValue,
    ResolvedPublicParameters,
    is_public_parameter_value,
)
from .interface import UnavailableOutput
from .result_validation import AcceptedEvaluationArtifact, EvaluationValidationIssue


class EvaluationRunStatus(str, Enum):
    """Persisted Evaluation Run lifecycle state."""

    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_PARTIAL = "completed_partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class EvaluationRunExecution:
    """Execution facts owned directly by one Evaluation Run."""

    started_at: object
    finished_at: object | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """Persisted accepted formal result surface for one Evaluation Run."""

    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, AcceptedEvaluationArtifact]


@dataclass(frozen=True, slots=True)
class EvaluationRunFailure:
    """Concise historical failure fact for an Evaluation Run when safely available."""

    type: str
    message: str


@dataclass(frozen=True, slots=True)
class EvaluationRunStudyLineage:
    """Optional downstream-to-upstream Study lineage for one Evaluation Run."""

    run: StudyRunId
    trial: str
    stage: str


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """One persisted MLDB evaluation execution attempt."""

    schema: Literal["mjtensu.mldb/evaluation-run/v1"]
    id: EvaluationRunId
    status: EvaluationRunStatus
    model: ModelId
    corpus: CorpusId
    evaluation_protocol: EvaluationProtocolId
    parameters: ResolvedPublicParameters
    execution: EvaluationRunExecution
    result: EvaluationRunResult | None = None
    unavailable_outputs: Sequence[UnavailableOutput] = ()
    validation_issues: Sequence[EvaluationValidationIssue] = ()
    failure: EvaluationRunFailure | None = None
    study: EvaluationRunStudyLineage | None = None
    environment: Mapping[str, object] | None = None


_EVALUATION_RUN_ID_RE = re.compile(r"^ev-([0-9]{8})-([0-9]{3})$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SUPPORTED_ARTIFACT_FORMATS = frozenset({"jsonl", "csv", "json"})
_TERMINAL_STATUSES = frozenset(
    {
        EvaluationRunStatus.COMPLETED,
        EvaluationRunStatus.COMPLETED_PARTIAL,
        EvaluationRunStatus.FAILED,
        EvaluationRunStatus.CANCELLED,
    }
)


def _append_issue(
    issues: list[ValidationIssue],
    code: str,
    message: str,
    path: str | None = None,
) -> None:
    issues.append(ValidationIssue(code=code, message=message, path=path))


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_metric_scalar(value: object) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _valid_evaluation_run_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    match = _EVALUATION_RUN_ID_RE.fullmatch(value)
    if match is None or match.group(2) == "000":
        return False
    try:
        _datetime.datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError:
        return False
    return True


def _valid_persisted_artifact_path(path: object) -> bool:
    if not isinstance(path, str) or not path:
        return False
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        return False
    return len(pure.parts) >= 2 and pure.parts[0] == "artifacts"


def _valid_output_reference(reference: object) -> bool:
    if not isinstance(reference, str):
        return False
    namespace, separator, key = reference.partition(".")
    return separator == "." and namespace in {"metrics", "artifacts"} and bool(key)


def _validate_accepted_artifact(
    issues: list[ValidationIssue],
    key: object,
    artifact: object,
) -> None:
    base = f"result.artifacts.{key}"
    if not _non_empty_string(key):
        _append_issue(
            issues,
            "evaluation_run.invalid_artifact_key",
            "Accepted artifact key must be a non-empty string.",
            base,
        )
    if not isinstance(artifact, AcceptedEvaluationArtifact):
        _append_issue(
            issues,
            "evaluation_run.invalid_artifact_metadata",
            "Accepted artifact metadata has an invalid shape.",
            base,
        )
        return
    if not _valid_persisted_artifact_path(artifact.path):
        _append_issue(
            issues,
            "evaluation_run.invalid_artifact_path",
            "Accepted artifact path must be Run-relative beneath artifacts/.",
            f"{base}.path",
        )
    if artifact.format not in _SUPPORTED_ARTIFACT_FORMATS:
        _append_issue(
            issues,
            "evaluation_run.invalid_artifact_format",
            "Accepted artifact format must be jsonl, csv, or json.",
            f"{base}.format",
        )
    if not _non_empty_string(artifact.schema):
        _append_issue(
            issues,
            "evaluation_run.invalid_artifact_schema",
            "Accepted artifact schema must be a non-empty string.",
            f"{base}.schema",
        )
    if not isinstance(artifact.sha256, str) or _SHA256_RE.fullmatch(artifact.sha256) is None:
        _append_issue(
            issues,
            "evaluation_run.invalid_artifact_sha256",
            "Accepted artifact sha256 must be a 64-character hexadecimal value.",
            f"{base}.sha256",
        )
    if (
        isinstance(artifact.bytes, bool)
        or not isinstance(artifact.bytes, int)
        or artifact.bytes < 0
    ):
        _append_issue(
            issues,
            "evaluation_run.invalid_artifact_bytes",
            "Accepted artifact bytes must be a non-negative integer.",
            f"{base}.bytes",
        )


def _validate_incompleteness(
    issues: list[ValidationIssue],
    unavailable_outputs: object,
    validation_issues: object,
) -> None:
    if isinstance(unavailable_outputs, (str, bytes)) or not isinstance(
        unavailable_outputs, _Sequence
    ):
        _append_issue(
            issues,
            "evaluation_run.invalid_unavailable_outputs",
            "unavailable_outputs must be a sequence.",
            "unavailable_outputs",
        )
    else:
        seen: set[str] = set()
        for index, item in enumerate(unavailable_outputs):
            path = f"unavailable_outputs[{index}]"
            if not isinstance(item, UnavailableOutput):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_unavailable_output",
                    "Unavailable output has an invalid shape.",
                    path,
                )
                continue
            if not _valid_output_reference(item.output):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_unavailable_reference",
                    "Unavailable output must use metrics.<key> or artifacts.<key>.",
                    f"{path}.output",
                )
            elif item.output in seen:
                _append_issue(
                    issues,
                    "evaluation_run.duplicate_unavailable_output",
                    "Unavailable output reference must not be duplicated.",
                    f"{path}.output",
                )
            else:
                seen.add(item.output)
            if not _non_empty_string(item.type):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_unavailable_type",
                    "Unavailable output type must be a non-empty string.",
                    f"{path}.type",
                )
            if not _non_empty_string(item.message):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_unavailable_message",
                    "Unavailable output message must be a non-empty string.",
                    f"{path}.message",
                )

    if isinstance(validation_issues, (str, bytes)) or not isinstance(
        validation_issues, _Sequence
    ):
        _append_issue(
            issues,
            "evaluation_run.invalid_validation_issues",
            "validation_issues must be a sequence.",
            "validation_issues",
        )
    else:
        for index, item in enumerate(validation_issues):
            path = f"validation_issues[{index}]"
            if not isinstance(item, EvaluationValidationIssue):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_validation_issue",
                    "Evaluation validation issue has an invalid shape.",
                    path,
                )
                continue
            if not _valid_output_reference(item.output):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_validation_issue_reference",
                    "Validation issue output must use metrics.<key> or artifacts.<key>.",
                    f"{path}.output",
                )
            if not _non_empty_string(item.type):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_validation_issue_type",
                    "Validation issue type must be a non-empty string.",
                    f"{path}.type",
                )
            if not _non_empty_string(item.message):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_validation_issue_message",
                    "Validation issue message must be a non-empty string.",
                    f"{path}.message",
                )


def validate_evaluation_run(run: EvaluationRun) -> ValidationReport:
    """Validate Evaluation Run metadata-local v1 invariants without external I/O."""

    issues: list[ValidationIssue] = []

    if run.schema != "mjtensu.mldb/evaluation-run/v1":
        _append_issue(
            issues,
            "evaluation_run.unsupported_schema",
            "schema must be 'mjtensu.mldb/evaluation-run/v1'.",
            "schema",
        )
    if not _valid_evaluation_run_id(run.id):
        _append_issue(
            issues,
            "evaluation_run.invalid_id",
            "id must match ev-YYYYMMDD-NNN.",
            "id",
        )
    if run.status not in {
        EvaluationRunStatus.RUNNING,
        EvaluationRunStatus.COMPLETED,
        EvaluationRunStatus.COMPLETED_PARTIAL,
        EvaluationRunStatus.FAILED,
        EvaluationRunStatus.CANCELLED,
    }:
        _append_issue(
            issues,
            "evaluation_run.invalid_status",
            "status is not a valid Evaluation Run lifecycle state.",
            "status",
        )

    for value, path in (
        (run.model, "model"),
        (run.corpus, "corpus"),
        (run.evaluation_protocol, "evaluation_protocol"),
    ):
        if not _non_empty_string(value):
            _append_issue(
                issues,
                "evaluation_run.invalid_reference",
                f"{path} must be a non-empty ID.",
                path,
            )

    if not isinstance(run.parameters, _Mapping):
        _append_issue(
            issues,
            "evaluation_run.invalid_parameters",
            "parameters must be a mapping.",
            "parameters",
        )
    else:
        for key, value in run.parameters.items():
            if not _non_empty_string(key):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_parameter_key",
                    "Parameter keys must be non-empty strings.",
                    f"parameters.{key}",
                )
            if not is_public_parameter_value(value):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_parameter_value",
                    "Parameter value is outside the JSON-compatible public value domain.",
                    f"parameters.{key}",
                )

    if run.execution.started_at is None:
        _append_issue(
            issues,
            "evaluation_run.missing_started_at",
            "execution.started_at is required.",
            "execution.started_at",
        )
    if run.status == EvaluationRunStatus.RUNNING:
        if run.execution.finished_at is not None:
            _append_issue(
                issues,
                "evaluation_run.running_has_finished_at",
                "A running Evaluation Run must not have execution.finished_at.",
                "execution.finished_at",
            )
    elif run.status in _TERMINAL_STATUSES and run.execution.finished_at is None:
        _append_issue(
            issues,
            "evaluation_run.missing_finished_at",
            "Every terminal Evaluation Run requires execution.finished_at.",
            "execution.finished_at",
        )

    result_metrics: object = None
    result_artifacts: object = None
    if run.result is not None:
        if not isinstance(run.result, EvaluationRunResult):
            _append_issue(
                issues,
                "evaluation_run.invalid_result",
                "result has an invalid EvaluationRunResult shape.",
                "result",
            )
        else:
            result_metrics = run.result.metrics
            result_artifacts = run.result.artifacts
            if not isinstance(result_metrics, _Mapping):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_result_metrics",
                    "result.metrics must be a mapping.",
                    "result.metrics",
                )
            else:
                for key, value in result_metrics.items():
                    if not _non_empty_string(key):
                        _append_issue(
                            issues,
                            "evaluation_run.invalid_metric_key",
                            "Accepted metric keys must be non-empty strings.",
                            f"result.metrics.{key}",
                        )
                    if not _is_metric_scalar(value):
                        _append_issue(
                            issues,
                            "evaluation_run.invalid_metric_value",
                            "Accepted metric must be a finite int or float, not bool.",
                            f"result.metrics.{key}",
                        )
            if not isinstance(result_artifacts, _Mapping):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_result_artifacts",
                    "result.artifacts must be a mapping.",
                    "result.artifacts",
                )
            else:
                for key, artifact in result_artifacts.items():
                    _validate_accepted_artifact(issues, key, artifact)

    if run.status in {
        EvaluationRunStatus.COMPLETED,
        EvaluationRunStatus.COMPLETED_PARTIAL,
    } and run.result is None:
        _append_issue(
            issues,
            "evaluation_run.missing_result",
            "Completed Evaluation Runs require result.metrics and result.artifacts.",
            "result",
        )

    _validate_incompleteness(
        issues,
        run.unavailable_outputs,
        run.validation_issues,
    )

    unavailable_count = (
        len(run.unavailable_outputs)
        if isinstance(run.unavailable_outputs, _Sequence)
        and not isinstance(run.unavailable_outputs, (str, bytes))
        else 0
    )
    validation_issue_count = (
        len(run.validation_issues)
        if isinstance(run.validation_issues, _Sequence)
        and not isinstance(run.validation_issues, (str, bytes))
        else 0
    )

    if run.status == EvaluationRunStatus.COMPLETED:
        if unavailable_count or validation_issue_count:
            _append_issue(
                issues,
                "evaluation_run.completed_has_incompleteness",
                "A completed Evaluation Run must have no incompleteness records.",
                "status",
            )

    if run.status == EvaluationRunStatus.COMPLETED_PARTIAL:
        accepted_count = 0
        if isinstance(result_metrics, _Mapping):
            accepted_count += len(result_metrics)
        if isinstance(result_artifacts, _Mapping):
            accepted_count += len(result_artifacts)
        if accepted_count == 0:
            _append_issue(
                issues,
                "evaluation_run.partial_without_output",
                "A completed_partial Run must retain at least one accepted formal output.",
                "result",
            )
        if unavailable_count + validation_issue_count == 0:
            _append_issue(
                issues,
                "evaluation_run.partial_without_reason",
                "A completed_partial Run requires at least one incompleteness record.",
                "status",
            )

    if run.failure is not None:
        if not isinstance(run.failure, EvaluationRunFailure):
            _append_issue(
                issues,
                "evaluation_run.invalid_failure",
                "failure has an invalid shape.",
                "failure",
            )
        else:
            if not _non_empty_string(run.failure.type):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_failure_type",
                    "failure.type must be a non-empty string.",
                    "failure.type",
                )
            if not _non_empty_string(run.failure.message):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_failure_message",
                    "failure.message must be a non-empty string.",
                    "failure.message",
                )

    if run.study is not None:
        if not isinstance(run.study, EvaluationRunStudyLineage):
            _append_issue(
                issues,
                "evaluation_run.invalid_study_lineage",
                "study has an invalid lineage shape.",
                "study",
            )
        else:
            if not _non_empty_string(run.study.run):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_study_run",
                    "study.run must be a non-empty Study Run ID.",
                    "study.run",
                )
            if not _non_empty_string(run.study.trial):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_study_trial",
                    "study.trial must be a non-empty Study-local trial string.",
                    "study.trial",
                )
            if not _non_empty_string(run.study.stage):
                _append_issue(
                    issues,
                    "evaluation_run.invalid_study_stage",
                    "study.stage must be a non-empty Study-local stage string.",
                    "study.stage",
                )

    if run.environment is not None and not isinstance(run.environment, _Mapping):
        _append_issue(
            issues,
            "evaluation_run.invalid_environment",
            "environment must be a mapping when present.",
            "environment",
        )

    return ValidationReport(tuple(issues))


def validate_evaluation_run_transition(
    source: EvaluationRunStatus,
    target: EvaluationRunStatus,
) -> ValidationReport:
    """Validate one requested Evaluation Run lifecycle status transition."""

    if source == EvaluationRunStatus.RUNNING and target in _TERMINAL_STATUSES:
        return ValidationReport()
    return ValidationReport(
        (
            ValidationIssue(
                code="evaluation_run.invalid_transition",
                message=(
                    "Only RUNNING -> COMPLETED, COMPLETED_PARTIAL, FAILED, or "
                    "CANCELLED is a valid Evaluation Run status transition."
                ),
                path="status",
            ),
        )
    )
