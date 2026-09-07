"""Public Python signatures for MLDB Evaluation Protocol metadata.

Implementation of the frozen Evaluation Protocol v1 metadata contract. This module
performs metadata-local validation only and deliberately performs no repository I/O,
implementation loading, hashing of sibling files, or runtime resolution.
"""

from __future__ import annotations

import re
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Mapping

from ..common.errors import ValidationIssue, ValidationReport
from ..common.ids import EvaluationProtocolId, TaskId
from ..common.parameters import (
    PublicParameterDeclarations,
    is_public_parameter_value,
)


class EvaluationProtocolStatus(str, Enum):
    """Persisted Evaluation Protocol lifecycle state."""

    DRAFT = "draft"
    SEALED = "sealed"


@dataclass(frozen=True, slots=True)
class EvaluationProtocolImplementation:
    """Evaluation Protocol v1 executable-implementation identity metadata."""

    entrypoint: Literal["evaluate"]
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationMetricDeclaration:
    """One protocol-local declared scalar metric."""

    type: Literal["number"]
    description: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationArtifactDeclaration:
    """One declared formal structured Evaluation Protocol artifact."""

    format: Literal["jsonl", "csv", "json"]
    schema: str
    required: bool


@dataclass(frozen=True, slots=True)
class EvaluationOutputs:
    """Declared formal output surface of one Evaluation Protocol."""

    metrics: Mapping[str, EvaluationMetricDeclaration]
    artifacts: Mapping[str, EvaluationArtifactDeclaration]


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """One reusable post-training evaluation definition for exactly one Task."""

    schema: Literal["mjtensu.mldb/evaluation-protocol/v1"]
    id: EvaluationProtocolId
    status: EvaluationProtocolStatus
    task: TaskId
    name: str
    description: str
    implementation: EvaluationProtocolImplementation
    parameters: PublicParameterDeclarations
    outputs: EvaluationOutputs


_VERSIONED_ID_RE = re.compile(r"^.+-v[1-9][0-9]*$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SUPPORTED_ARTIFACT_FORMATS = frozenset({"jsonl", "csv", "json"})
_MISSING = object()


def _append_issue(
    issues: list[ValidationIssue],
    code: str,
    message: str,
    path: str | None = None,
) -> None:
    issues.append(ValidationIssue(code=code, message=message, path=path))


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_mapping_key(
    issues: list[ValidationIssue],
    key: object,
    path: str,
) -> bool:
    if not _non_empty_string(key):
        _append_issue(
            issues,
            "evaluation_protocol.invalid_mapping_key",
            "Mapping keys must be non-empty strings.",
            path,
        )
        return False
    return True


def validate_evaluation_protocol_metadata(
    protocol: EvaluationProtocol,
) -> ValidationReport:
    """Validate static Evaluation Protocol metadata invariants without external I/O."""

    issues: list[ValidationIssue] = []

    if protocol.schema != "mjtensu.mldb/evaluation-protocol/v1":
        _append_issue(
            issues,
            "evaluation_protocol.unsupported_schema",
            "schema must be 'mjtensu.mldb/evaluation-protocol/v1'.",
            "schema",
        )

    if not isinstance(protocol.id, str) or _VERSIONED_ID_RE.fullmatch(protocol.id) is None:
        _append_issue(
            issues,
            "evaluation_protocol.invalid_id",
            "id must end in '-v<positive-integer>'.",
            "id",
        )

    if protocol.status not in (
        EvaluationProtocolStatus.DRAFT,
        EvaluationProtocolStatus.SEALED,
    ):
        _append_issue(
            issues,
            "evaluation_protocol.invalid_status",
            "status must be draft or sealed.",
            "status",
        )

    if not _non_empty_string(protocol.task):
        _append_issue(
            issues,
            "evaluation_protocol.invalid_task",
            "task must be a non-empty Task ID.",
            "task",
        )
    if not _non_empty_string(protocol.name):
        _append_issue(
            issues,
            "evaluation_protocol.invalid_name",
            "name must be a non-empty string.",
            "name",
        )
    if not _non_empty_string(protocol.description):
        _append_issue(
            issues,
            "evaluation_protocol.invalid_description",
            "description must be a non-empty string.",
            "description",
        )

    implementation = protocol.implementation
    entrypoint = getattr(implementation, "entrypoint", _MISSING)
    sha256 = getattr(implementation, "sha256", _MISSING)
    if entrypoint != "evaluate":
        _append_issue(
            issues,
            "evaluation_protocol.invalid_entrypoint",
            "implementation.entrypoint must be 'evaluate'.",
            "implementation.entrypoint",
        )

    if protocol.status == EvaluationProtocolStatus.SEALED and sha256 is None:
        _append_issue(
            issues,
            "evaluation_protocol.missing_sha256",
            "A sealed Evaluation Protocol requires implementation.sha256.",
            "implementation.sha256",
        )
    if sha256 is not None and (
        not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None
    ):
        _append_issue(
            issues,
            "evaluation_protocol.invalid_sha256",
            "implementation.sha256 must be a 64-character hexadecimal SHA-256 value.",
            "implementation.sha256",
        )

    if not isinstance(protocol.parameters, _Mapping):
        _append_issue(
            issues,
            "evaluation_protocol.invalid_parameters",
            "parameters must be a mapping.",
            "parameters",
        )
    else:
        for key, declaration in protocol.parameters.items():
            key_path = f"parameters.{key}"
            _validate_mapping_key(issues, key, key_path)
            default = getattr(declaration, "default", _MISSING)
            if default is _MISSING:
                _append_issue(
                    issues,
                    "evaluation_protocol.missing_parameter_default",
                    "Every public parameter declaration requires default.",
                    key_path,
                )
            elif not is_public_parameter_value(default):
                _append_issue(
                    issues,
                    "evaluation_protocol.invalid_parameter_default",
                    "Public parameter default is outside the JSON-compatible value domain.",
                    f"{key_path}.default",
                )

    outputs = protocol.outputs
    metrics = getattr(outputs, "metrics", _MISSING)
    artifacts = getattr(outputs, "artifacts", _MISSING)

    if not isinstance(metrics, _Mapping):
        _append_issue(
            issues,
            "evaluation_protocol.invalid_metrics",
            "outputs.metrics must be a mapping.",
            "outputs.metrics",
        )
    else:
        for key, declaration in metrics.items():
            key_path = f"outputs.metrics.{key}"
            _validate_mapping_key(issues, key, key_path)
            if getattr(declaration, "type", _MISSING) != "number":
                _append_issue(
                    issues,
                    "evaluation_protocol.invalid_metric_type",
                    "Metric declaration type must be 'number'.",
                    f"{key_path}.type",
                )
            description = getattr(declaration, "description", None)
            if description is not None and not isinstance(description, str):
                _append_issue(
                    issues,
                    "evaluation_protocol.invalid_metric_description",
                    "Metric description must be a string when present.",
                    f"{key_path}.description",
                )

    if not isinstance(artifacts, _Mapping):
        _append_issue(
            issues,
            "evaluation_protocol.invalid_artifacts",
            "outputs.artifacts must be a mapping.",
            "outputs.artifacts",
        )
    else:
        for key, declaration in artifacts.items():
            key_path = f"outputs.artifacts.{key}"
            _validate_mapping_key(issues, key, key_path)
            artifact_format = getattr(declaration, "format", _MISSING)
            artifact_schema = getattr(declaration, "schema", _MISSING)
            required = getattr(declaration, "required", _MISSING)

            if artifact_format not in _SUPPORTED_ARTIFACT_FORMATS:
                _append_issue(
                    issues,
                    "evaluation_protocol.invalid_artifact_format",
                    "Artifact format must be jsonl, csv, or json.",
                    f"{key_path}.format",
                )
            if not _non_empty_string(artifact_schema):
                _append_issue(
                    issues,
                    "evaluation_protocol.invalid_artifact_schema",
                    "Artifact schema must be a non-empty versioned schema identifier.",
                    f"{key_path}.schema",
                )
            if not isinstance(required, bool):
                _append_issue(
                    issues,
                    "evaluation_protocol.invalid_artifact_required",
                    "Artifact required must be boolean.",
                    f"{key_path}.required",
                )

    return ValidationReport(tuple(issues))
