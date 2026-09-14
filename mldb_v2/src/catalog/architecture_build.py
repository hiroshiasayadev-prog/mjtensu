"""MLDB v2 Architecture companion build boundary."""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeAlias, cast

from mldb_v2.src.common.ids import ArchitectureId, EntityKind

from ._executable_definition_loading import _load_executable_callable
from .architecture import _load_architecture_definition

if TYPE_CHECKING:
    import torch.nn


ArchitectureBuild: TypeAlias = Callable[[], "torch.nn.Module"]


def _load_architecture_build(
    mldb_data_root: str | Path, architecture_id: ArchitectureId | str
) -> ArchitectureBuild:
    _load_architecture_definition(mldb_data_root, architecture_id)
    callable_value = _load_executable_callable(
        mldb_data_root,
        kind=EntityKind.ARCHITECTURE,
        entity_id=str(architecture_id),
        entrypoint="build",
        arity=0,
    )
    try:
        import torch.nn as nn
    except ImportError as error:
        raise ValueError("PyTorch is required to verify Architecture build") from error
    try:
        first = callable_value()
        second = callable_value()
    except Exception as error:
        raise ValueError("Architecture build() raised during verification") from error
    if not isinstance(first, nn.Module) or not isinstance(second, nn.Module):
        raise ValueError("Architecture build() must return torch.nn.Module")
    if first is second:
        raise ValueError("Architecture build() must return a fresh module")
    return cast(ArchitectureBuild, callable_value)
