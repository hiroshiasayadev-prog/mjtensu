"""Canonical PyTorch learned-state runtime helpers."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from mldb_v2.src.catalog.architecture_build import _load_architecture_build
from mldb_v2.src.common.ids import ArchitectureId
from mldb_v2.src.storage.artifact_reference import (
    ArtifactRef,
    _validate_artifact_ref,
    _verify_artifact_bytes,
)

if TYPE_CHECKING:
    import torch
    import torch.nn


CanonicalStateDict: TypeAlias = Mapping[str, "torch.Tensor"]


class CanonicalWeightsArtifactRef(ArtifactRef):
    format: Literal["pytorch-state-dict/v1"]


_CANONICAL_WEIGHTS_FORMAT = "pytorch-state-dict/v1"
_CANONICAL_WEIGHT_REF_FIELDS = {"uri", "bytes", "sha256", "format"}


def _validate_canonical_weights_artifact_ref(value: object) -> CanonicalWeightsArtifactRef:
    if type(value) is not dict or set(value) != _CANONICAL_WEIGHT_REF_FIELDS:
        raise ValueError("canonical weights ArtifactRef fields do not match schema")
    _validate_artifact_ref(value)
    if value["format"] != _CANONICAL_WEIGHTS_FORMAT:
        raise ValueError("unsupported canonical weights format")
    return cast(CanonicalWeightsArtifactRef, value)


def _canonicalize_state_dict(value: object) -> dict[str, "torch.Tensor"]:
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required for canonical weights") from error

    if not isinstance(value, Mapping):
        raise ValueError("canonical weights must be a mapping")
    canonical: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if type(key) is not str:
            raise ValueError("canonical state keys must be exact strings")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("canonical state values must be torch.Tensor")
        canonical[key] = tensor.detach().cpu()
    return canonical


def _serialize_canonical_state_dict(value: object) -> bytes:
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required for canonical weights") from error

    canonical = _canonicalize_state_dict(value)
    stream = BytesIO()
    torch.save(canonical, stream)
    return stream.getvalue()


def _load_canonical_state_dict_bytes(
    data: bytes,
    *,
    ref: CanonicalWeightsArtifactRef | None = None,
) -> dict[str, "torch.Tensor"]:
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required for canonical weights") from error

    if type(data) is not bytes:
        raise ValueError("canonical weight data must be exact bytes")
    verified = data
    if ref is not None:
        validated_ref = _validate_canonical_weights_artifact_ref(ref)
        verified = _verify_artifact_bytes(ref=validated_ref, data=data)
    try:
        loaded = torch.load(BytesIO(verified), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("invalid canonical weight bytes") from error
    return _canonicalize_state_dict(loaded)


def _build_fresh_architecture_module(
    mldb_data_root: str | Path,
    architecture_id: ArchitectureId | str,
) -> "torch.nn.Module":
    try:
        import torch.nn as nn
    except ImportError as error:
        raise ValueError("PyTorch is required to build Architecture runtime") from error

    build = _load_architecture_build(mldb_data_root, architecture_id)
    try:
        module = build()
    except Exception as error:
        raise ValueError("Architecture build() raised during runtime construction") from error
    if not isinstance(module, nn.Module):
        raise ValueError("Architecture build() must return torch.nn.Module")
    return module


def _strict_load_state_dict(module: "torch.nn.Module", state: object) -> None:
    canonical = _canonicalize_state_dict(state)
    try:
        incompatible = module.load_state_dict(canonical, strict=True)
    except Exception as error:
        raise ValueError("canonical state is not strictly compatible with Architecture") from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("canonical state is not strictly compatible with Architecture")


def _load_state_into_fresh_architecture(
    mldb_data_root: str | Path,
    architecture_id: ArchitectureId | str,
    state: object,
) -> "torch.nn.Module":
    module = _build_fresh_architecture_module(mldb_data_root, architecture_id)
    _strict_load_state_dict(module, state)
    return module


def _canonicalize_trained_module_state(
    mldb_data_root: str | Path,
    architecture_id: ArchitectureId | str,
    trained_module: object,
) -> dict[str, "torch.Tensor"]:
    try:
        import torch.nn as nn
    except ImportError as error:
        raise ValueError("PyTorch is required for trained-module validation") from error

    if not isinstance(trained_module, nn.Module):
        raise ValueError("Train Protocol must return torch.nn.Module")
    state = _canonicalize_state_dict(trained_module.state_dict())
    fresh = _build_fresh_architecture_module(mldb_data_root, architecture_id)
    _strict_load_state_dict(fresh, state)
    return state
