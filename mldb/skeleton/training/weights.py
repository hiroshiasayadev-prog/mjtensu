"""Public signatures for MLDB canonical learned-weight artifacts.

This module fixes the Training Run-owned canonical PyTorch state boundary. It
models the direct tensor state mapping, the completed-artifact metadata reused by
Training Run records, and the runtime operations that accept, serialize, and load
canonical learned weights.

It intentionally does not allocate Training Runs, create Run directories, resolve
repository placement, load Architecture implementation files, choose checkpoints,
invoke Train Protocols, mutate Run lifecycle state, or define Model identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import torch

from ..catalog.architecture import ArchitectureBuild


CanonicalWeightsState: TypeAlias = Mapping[str, torch.Tensor]
"""Canonical MLDB v1 learned state.

The runtime representation is a plain mapping from string state keys to detached
CPU ``torch.Tensor`` values. Wrapper checkpoints, optimizer/scheduler state,
trainer objects, counters, protocol configuration, and non-tensor extra state are
outside this type's contract.
"""


@dataclass(frozen=True, slots=True)
class CanonicalWeightsArtifact:
    """Persisted metadata for one completed Training Run's canonical weights.

    ``path`` is always the Training Run-relative canonical artifact path rather
    than an absolute repository path. ``sha256`` and ``bytes`` describe the exact
    bytes persisted for this Training Run; they do not define a semantic hash of
    tensor values or promise byte-identical independent serialization.

    This value is the canonical ``result.weights`` shape that later Training Run
    metadata should reuse instead of redefining format/path/hash/size fields.
    """

    format: Literal["pytorch-state-dict"]
    path: Literal["artifacts/weights.pt"]
    sha256: str
    bytes: int


def accept_trained_state(
    trained_module: torch.nn.Module,
    architecture_build: ArchitectureBuild,
) -> CanonicalWeightsState:
    """Accept a Train Protocol result and produce canonical CPU tensor state.

    Acceptance requires the supplied value to satisfy the trained-module
    boundary, every ``state_dict`` key/value to satisfy the plain string-to-tensor
    contract, every persisted tensor to be detached and represented on CPU, and
    the resulting state to load strictly into a fresh module returned by
    ``architecture_build()``.

    Architecture implementation loading is deliberately outside this function;
    callers supply the already-loaded/resolved ``ArchitectureBuild`` callable.
    Failure is an operation failure rather than a new public result-monad or
    feature-specific exception hierarchy in this skeleton.
    """

    ...


def serialize_canonical_weights(
    state: CanonicalWeightsState,
    weights_path: Path,
) -> CanonicalWeightsArtifact:
    """Persist canonical state directly and return exact artifact metadata.

    ``weights_path`` is the concrete filesystem destination already resolved by
    the Training Run runtime. The operation serializes ``state`` directly as the
    ``pytorch-state-dict`` payload at that destination and returns metadata whose
    recorded ``path`` remains the Run-relative ``artifacts/weights.pt`` contract.

    Successful return requires SHA-256 and byte size to describe the exact bytes
    that were persisted. Directory creation, repository placement calculation,
    Run mutation, and lifecycle finalization are outside this operation.
    """

    ...


def load_canonical_weights(
    weights_path: Path,
    architecture_build: ArchitectureBuild,
) -> torch.nn.Module:
    """Load canonical weights into a fresh selected Architecture strictly.

    Loading uses the canonical CPU/restricted-weights contract, validates the
    loaded object as a plain string-to-``torch.Tensor`` mapping before module
    loading, constructs a fresh module through ``architecture_build()``, and
    applies the state with strict matching.

    The operation does not resolve an ``ArchitectureHandle`` or dynamically import
    its sibling implementation. Payload repair, key rewriting, non-strict matching,
    and arbitrary checkpoint coercion are outside the canonical v1 boundary.
    """

    ...
