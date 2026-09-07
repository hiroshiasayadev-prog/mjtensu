"""Canonical learned-weight implementation for MLDB Wave I1-B."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal, TypeAlias

import torch

from ..catalog.architecture import ArchitectureBuild


CanonicalWeightsState: TypeAlias = Mapping[str, torch.Tensor]
"""Canonical MLDB v1 learned state."""


@dataclass(frozen=True, slots=True)
class CanonicalWeightsArtifact:
    """Persisted metadata for one completed Training Run's canonical weights."""

    format: Literal["pytorch-state-dict"]
    path: Literal["artifacts/weights.pt"]
    sha256: str
    bytes: int


def accept_trained_state(
    trained_module: torch.nn.Module,
    architecture_build: ArchitectureBuild,
) -> CanonicalWeightsState:
    """Accept a Train Protocol result and produce canonical CPU tensor state."""
    if not isinstance(trained_module, torch.nn.Module):
        raise TypeError("trained_module must be a torch.nn.Module")

    raw_state = trained_module.state_dict()
    _validate_tensor_state(raw_state, require_canonical=False)

    canonical_state = {
        key: value.detach().cpu()
        for key, value in raw_state.items()
    }
    _validate_tensor_state(canonical_state, require_canonical=True)

    fresh_module = architecture_build()
    if not isinstance(fresh_module, torch.nn.Module):
        raise TypeError("architecture_build() must return a torch.nn.Module")
    fresh_module.load_state_dict(canonical_state, strict=True)

    return canonical_state


def serialize_canonical_weights(
    state: CanonicalWeightsState,
    weights_path: Path,
) -> CanonicalWeightsArtifact:
    """Persist canonical state directly and return exact artifact metadata."""
    _validate_tensor_state(state, require_canonical=True)

    torch.save(state, weights_path)

    byte_count = weights_path.stat().st_size
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    return CanonicalWeightsArtifact(
        format="pytorch-state-dict",
        path="artifacts/weights.pt",
        sha256=digest,
        bytes=byte_count,
    )


def load_canonical_weights(
    weights_path: Path,
    architecture_build: ArchitectureBuild,
) -> torch.nn.Module:
    """Load canonical weights into a fresh selected Architecture strictly."""
    state = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    _validate_tensor_state(state, require_canonical=True)

    module = architecture_build()
    if not isinstance(module, torch.nn.Module):
        raise TypeError("architecture_build() must return a torch.nn.Module")
    module.load_state_dict(state, strict=True)
    return module


def _validate_tensor_state(state: object, *, require_canonical: bool) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("canonical weights state must be a mapping")

    for key, value in state.items():
        if type(key) is not str:
            raise TypeError("canonical weights state keys must be strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"canonical weights state value for {key!r} must be a torch.Tensor"
            )
        if require_canonical:
            if value.device.type != "cpu":
                raise ValueError(
                    f"canonical weights tensor {key!r} must be on CPU"
                )
            if value.requires_grad:
                raise ValueError(
                    f"canonical weights tensor {key!r} must be detached"
                )
