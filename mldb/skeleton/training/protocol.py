"""Public Python signatures for MLDB Train Protocol definitions and execution.

This skeleton fixes the Train Protocol metadata boundary and the callable shape used
by resolved training execution. It intentionally does not resolve repository assets,
load sibling Python implementations, perform launch preflight, allocate Training Runs,
mutate lifecycle state, train models, or serialize learned weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, TypeAlias

import torch

from ..common.errors import ValidationReport
from ..common.ids import TaskId, TrainProtocolId
from ..common.parameters import (
    PublicParameterDeclarations,
    PublicParameterValue,
    ResolvedPublicParameters,
)
from ..runtime.catalog_handles import ArchitectureHandle, CorpusHandle, TaskHandle


class TrainProtocolStatus(str, Enum):
    """Persisted Train Protocol lifecycle state."""

    DRAFT = "draft"
    SEALED = "sealed"


@dataclass(frozen=True, slots=True)
class TrainProtocolImplementation:
    """Train Protocol v1 executable-implementation identity metadata.

    ``entrypoint`` is fixed to ``"train"`` by the v1 contract. ``sha256`` identifies
    the exact sibling Train Protocol Python bytes. It is required when the containing
    protocol is ``SEALED`` and may be omitted while the protocol is ``DRAFT``.

    Resolving the sibling path, calculating or comparing hashes, importing the module,
    and exposing the callable are runtime/sealing responsibilities rather than behavior
    of this metadata value.
    """

    entrypoint: Literal["train"]
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class TrainProtocol:
    """One reusable executable training definition for exactly one Task.

    The protocol is permanently bound only to ``task``. A concrete Corpus and
    Architecture are selected later for one Training Run and supplied through
    :class:`TrainContext` after launch preflight succeeds.

    ``parameters`` uses the shared Wave 0 public-parameter declaration contract
    directly. That normalized declaration intentionally retains only the required
    executable ``default`` value. Optional authored fields such as ``description``,
    ``type``, ``minimum``, ``maximum``, or ``suggested`` have no generic MLDB v1
    semantics and are therefore not promoted into a second Train-Protocol-specific
    parameter schema here.

    ``notes`` preserves optional supplemental metadata without assigning a generic
    structure or executable meaning to it.
    """

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
    """Resolved request supplied to one Train Protocol ``train`` entrypoint.

    All referenced Catalog handles and the complete public-parameter mapping have
    already passed launch preflight before this value is constructed. ``seed`` is the
    validated Training Run seed: it is an ``int``, boolean is invalid, no common numeric
    range exists, and MLDB performs no coercion. The seed is separate from public
    Train-Protocol parameters.

    ``work_dir`` is the concrete Training Run ``work/`` directory available for
    protocol-owned checkpoints, logs, generated configuration, and temporary files. Its
    contents do not become canonical MLDB artifacts merely by being written there.
    """

    task: TaskHandle
    corpus: CorpusHandle
    architecture: ArchitectureHandle
    seed: int
    parameters: ResolvedPublicParameters
    work_dir: Path


TrainEntrypoint: TypeAlias = Callable[[TrainContext], torch.nn.Module]
"""Resolved Train Protocol v1 ``train`` callable.

Each sibling Train Protocol implementation exposes a function named ``train`` matching
this callable shape. A successful call accepts exactly one :class:`TrainContext` and
returns one trained ``torch.nn.Module`` whose selected learned state is already restored.
Loading the sibling implementation, validating callable presence, invoking it, checking
the returned module against a fresh selected Architecture, and serializing canonical
weights belong to later training execution/runtime boundaries.
"""


def validate_train_protocol_metadata(protocol: TrainProtocol) -> ValidationReport:
    """Validate static Train Protocol metadata invariants without external I/O.

    This validation owns format-intrinsic rules decidable from the normalized in-memory
    value, including the v1 schema declaration, terminal positive-integer ``-vN`` ID
    grammar, lifecycle/hash conditional, fixed ``train`` entrypoint, the non-empty
    ``name`` requirement, and validity of each represented public-parameter default in
    the shared JSON-compatible value domain.

    Raw YAML parsing and normalization must reject malformed declaration shapes such as
    a parameter entry missing ``default`` before constructing
    :class:`PublicParameterDeclaration` values; this skeleton does not define a YAML
    representation or parser.

    This function does not compare the ID with a filesystem basename, resolve the
    referenced Task, resolve Corpus or Architecture selections, read or hash sibling
    Python bytes, load the implementation, resolve caller parameter overrides, validate
    a concrete training seed, check cross-asset Task/interface compatibility, enforce
    execution-time sealed requirements, allocate a Training Run, or perform sealing.
    Those checks belong to repository/runtime resolution, launch preflight, executable
    loading, training execution, or lifecycle tooling as applicable.
    """

    ...
