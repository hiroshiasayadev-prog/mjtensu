"""Architecture metadata implementation for MLDB Wave I1-A."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
import re
from typing import Literal, TypeAlias

import torch

from ..common.errors import ValidationIssue, ValidationReport
from ..common.ids import ArchitectureId, TaskId


class ArchitectureStatus(str, Enum):
    """Persisted Architecture lifecycle state."""

    DRAFT = "draft"
    SEALED = "sealed"


@dataclass(frozen=True, slots=True)
class ArchitectureImplementation:
    """Architecture v1 executable-implementation identity metadata."""

    framework: Literal["pytorch"]
    entrypoint: Literal["build"]
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ArchitectureInterface:
    """Coarse model input/output compatibility metadata."""

    input: Mapping[str, object]
    output: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ArchitectureStructure:
    """Human/search-oriented summary of the executable model structure."""

    summary: str
    traits: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class Architecture:
    """One versioned unweighted model structure for exactly one Task."""

    schema: Literal["mjtensu.mldb/architecture/v1"]
    id: ArchitectureId
    status: ArchitectureStatus
    task: TaskId
    name: str
    family: str
    description: str
    implementation: ArchitectureImplementation
    interface: ArchitectureInterface
    structure: ArchitectureStructure
    parameters: object | None = None


ArchitectureBuild: TypeAlias = Callable[[], torch.nn.Module]
"""Resolved Architecture v1 ``build`` callable."""


def validate_architecture_metadata(architecture: Architecture) -> ValidationReport:
    """Validate static Architecture metadata invariants without external I/O."""
    issues: list[ValidationIssue] = []

    if architecture.schema != "mjtensu.mldb/architecture/v1":
        issues.append(
            ValidationIssue(
                code="architecture.schema.unsupported",
                message="Architecture schema must be 'mjtensu.mldb/architecture/v1'.",
                path="schema",
            )
        )

    if type(architecture.id) is not str or not _ARCHITECTURE_ID_PATTERN.fullmatch(
        architecture.id
    ):
        issues.append(
            ValidationIssue(
                code="architecture.id.invalid",
                message="Architecture id must end in '-vN' with N a positive integer.",
                path="id",
            )
        )

    if not isinstance(architecture.status, ArchitectureStatus):
        issues.append(
            ValidationIssue(
                code="architecture.status.invalid",
                message="Architecture status must be draft or sealed.",
                path="status",
            )
        )

    if type(architecture.family) is not str or architecture.family == "":
        issues.append(
            ValidationIssue(
                code="architecture.family.invalid",
                message="Architecture family must be a non-empty string.",
                path="family",
            )
        )

    if architecture.implementation.framework != "pytorch":
        issues.append(
            ValidationIssue(
                code="architecture.implementation.framework.unsupported",
                message="Architecture v1 framework must be 'pytorch'.",
                path="implementation.framework",
            )
        )

    if architecture.implementation.entrypoint != "build":
        issues.append(
            ValidationIssue(
                code="architecture.implementation.entrypoint.invalid",
                message="Architecture v1 entrypoint must be 'build'.",
                path="implementation.entrypoint",
            )
        )

    implementation_sha256 = architecture.implementation.sha256
    if architecture.status is ArchitectureStatus.SEALED and implementation_sha256 is None:
        issues.append(
            ValidationIssue(
                code="architecture.implementation.sha256.required",
                message="A sealed Architecture requires implementation.sha256.",
                path="implementation.sha256",
            )
        )
    elif implementation_sha256 is not None and not _is_sha256(implementation_sha256):
        issues.append(
            ValidationIssue(
                code="architecture.implementation.sha256.invalid",
                message="Architecture implementation sha256 must be a 64-character hexadecimal digest.",
                path="implementation.sha256",
            )
        )

    output_kind = architecture.interface.output.get("kind")
    if type(output_kind) is not str or output_kind == "":
        issues.append(
            ValidationIssue(
                code="architecture.interface.output.kind.invalid",
                message="Architecture interface output kind must be a non-empty string.",
                path="interface.output.kind",
            )
        )

    if type(architecture.structure.summary) is not str or architecture.structure.summary == "":
        issues.append(
            ValidationIssue(
                code="architecture.structure.summary.invalid",
                message="Architecture structure summary must be a non-empty string.",
                path="structure.summary",
            )
        )

    return ValidationReport(tuple(issues))


_ARCHITECTURE_ID_PATTERN = re.compile(r".+-v[1-9][0-9]*\Z")


def _is_sha256(value: object) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)
