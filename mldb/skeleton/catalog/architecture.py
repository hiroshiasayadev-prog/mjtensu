"""Public Python signatures for MLDB Architecture definitions.

This skeleton fixes the Architecture metadata and executable-construction boundary.
It intentionally does not load sibling Python files, resolve repository paths, run
pytest, perform sealing mutations, train models, or load learned state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

import torch

from ..common.errors import ValidationReport
from ..common.ids import ArchitectureId, TaskId


class ArchitectureStatus(str, Enum):
    """Persisted Architecture lifecycle state."""

    DRAFT = "draft"
    SEALED = "sealed"


@dataclass(frozen=True, slots=True)
class ArchitectureImplementation:
    """Architecture v1 executable-implementation identity metadata.

    ``framework`` and ``entrypoint`` are fixed by the Architecture v1 contract.
    ``sha256`` identifies the exact sibling Architecture Python bytes. It is
    required when the containing Architecture is ``SEALED`` and may be omitted
    while the Architecture is ``DRAFT``.

    Hash calculation and comparison with filesystem bytes are runtime/sealing
    responsibilities rather than behavior of this metadata value.
    """

    framework: Literal["pytorch"]
    entrypoint: Literal["build"]
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ArchitectureInterface:
    """Coarse model input/output compatibility metadata.

    Architecture v1 intentionally does not define one universal tensor or
    ``forward()`` result schema. The nested mappings therefore retain
    Architecture-specific compatibility facts without turning this skeleton into
    a model-graph or tensor-schema DSL.
    """

    input: Mapping[str, object]
    output: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ArchitectureStructure:
    """Human/search-oriented summary of the executable model structure."""

    summary: str
    traits: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class Architecture:
    """One versioned unweighted model structure for exactly one Task.

    The object represents Architecture YAML metadata only. It does not embed a
    loaded Python module, constructed ``torch.nn.Module``, learned weights,
    training policy, repository path, pytest result, or sealing operation.

    ``parameters`` is opaque Architecture-specific summary metadata. Generic
    MLDB code must not reinterpret it as caller-supplied constructor arguments.
    """

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
"""Resolved Architecture v1 ``build`` callable.

The callable accepts no positional or keyword arguments and returns one
``torch.nn.Module``. Architecture-specific topology parameters are fixed inside
that Architecture's sibling implementation. Loading the sibling module and
obtaining this callable belong to the runtime executable loader, not this
feature module.
"""


def validate_architecture_metadata(architecture: Architecture) -> ValidationReport:
    """Validate static Architecture metadata invariants without external I/O.

    This validation owns format-intrinsic checks such as the Architecture v1
    declaration, terminal ``-vN`` identity grammar, lifecycle/hash conditional,
    fixed PyTorch/build declaration, and other metadata-local format rules.

    It does not resolve ``architecture.task``, compare the ID with a filesystem
    basename, read or hash sibling Python bytes, import the implementation,
    inspect or invoke ``build``, run asset pytest, or perform a draft-to-sealed
    mutation. Those checks belong to asset resolution, executable loading,
    verification, and sealing respectively.
    """

    ...
