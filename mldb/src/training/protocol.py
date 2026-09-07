"""Train Protocol metadata implementation for MLDB Wave I1-B."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Literal, TYPE_CHECKING, TypeAlias

import torch

from ..common.errors import ValidationIssue, ValidationReport
from ..common.ids import TaskId, TrainProtocolId
from ..common.parameters import (
    PublicParameterDeclarations,
    PublicParameterValue,
    ResolvedPublicParameters,
    is_public_parameter_value,
)

if TYPE_CHECKING:
    from ..runtime.catalog_handles import ArchitectureHandle, CorpusHandle, TaskHandle


class TrainProtocolStatus(str, Enum):
    """Persisted Train Protocol lifecycle state."""

    DRAFT = "draft"
    SEALED = "sealed"


@dataclass(frozen=True, slots=True)
class TrainProtocolImplementation:
    """Train Protocol v1 executable-implementation identity metadata."""

    entrypoint: Literal["train"]
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class TrainProtocol:
    """One reusable executable training definition for exactly one Task."""

    schema: Literal["mjtensu.mldb/train-protocol/v1"]
    id: TrainProtocolId
    status: TrainProtocolStatus
    task: TaskId
    name: str
    description: str
    implementation: TrainProtocolImplementation
    parameters: PublicParameterDeclarations
    notes: object | None = None


@dataclass(frozen=True, slots=True)
class TrainContext:
    """Resolved request supplied to one Train Protocol ``train`` entrypoint."""

    task: TaskHandle
    corpus: CorpusHandle
    architecture: ArchitectureHandle
    seed: int
    parameters: ResolvedPublicParameters
    work_dir: Path


TrainEntrypoint: TypeAlias = Callable[[TrainContext], torch.nn.Module]
"""Resolved Train Protocol v1 ``train`` callable."""


def validate_train_protocol_metadata(protocol: TrainProtocol) -> ValidationReport:
    """Validate static Train Protocol metadata invariants without external I/O."""
    issues: list[ValidationIssue] = []

    if protocol.schema != "mjtensu.mldb/train-protocol/v1":
        issues.append(
            ValidationIssue(
                code="train_protocol.schema.unsupported",
                message="Train Protocol schema must be 'mjtensu.mldb/train-protocol/v1'.",
                path="schema",
            )
        )

    if type(protocol.id) is not str or not _TRAIN_PROTOCOL_ID_PATTERN.fullmatch(
        protocol.id
    ):
        issues.append(
            ValidationIssue(
                code="train_protocol.id.invalid",
                message="Train Protocol id must end in '-vN' with N a positive integer.",
                path="id",
            )
        )

    if not isinstance(protocol.status, TrainProtocolStatus):
        issues.append(
            ValidationIssue(
                code="train_protocol.status.invalid",
                message="Train Protocol status must be draft or sealed.",
                path="status",
            )
        )

    if type(protocol.name) is not str or protocol.name == "":
        issues.append(
            ValidationIssue(
                code="train_protocol.name.invalid",
                message="Train Protocol name must be a non-empty string.",
                path="name",
            )
        )

    if protocol.implementation.entrypoint != "train":
        issues.append(
            ValidationIssue(
                code="train_protocol.implementation.entrypoint.invalid",
                message="Train Protocol v1 entrypoint must be 'train'.",
                path="implementation.entrypoint",
            )
        )

    implementation_sha256 = protocol.implementation.sha256
    if protocol.status is TrainProtocolStatus.SEALED and implementation_sha256 is None:
        issues.append(
            ValidationIssue(
                code="train_protocol.implementation.sha256.required",
                message="A sealed Train Protocol requires implementation.sha256.",
                path="implementation.sha256",
            )
        )
    elif implementation_sha256 is not None and not _is_sha256(implementation_sha256):
        issues.append(
            ValidationIssue(
                code="train_protocol.implementation.sha256.invalid",
                message="Train Protocol implementation sha256 must be a 64-character hexadecimal digest.",
                path="implementation.sha256",
            )
        )

    for name, declaration in protocol.parameters.items():
        if type(name) is not str:
            issues.append(
                ValidationIssue(
                    code="train_protocol.parameter.name.invalid",
                    message="Train Protocol public parameter names must be strings.",
                    path="parameters",
                )
            )
            continue
        if not is_public_parameter_value(declaration.default):
            issues.append(
                ValidationIssue(
                    code="train_protocol.parameter.default.invalid",
                    message=f"Public parameter {name!r} has a default outside the JSON-compatible value domain.",
                    path=f"parameters[{name!r}].default",
                )
            )

    return ValidationReport(tuple(issues))


_TRAIN_PROTOCOL_ID_PATTERN = re.compile(r".+-v[1-9][0-9]*\Z")


def _is_sha256(value: object) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)
