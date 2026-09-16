"""MLDB v2 Architecture companion build boundary."""

from typing import TYPE_CHECKING, Callable, TypeAlias

if TYPE_CHECKING:
    import torch.nn


ArchitectureBuild: TypeAlias = Callable[[], "torch.nn.Module"]
