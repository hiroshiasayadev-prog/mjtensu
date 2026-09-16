"""MLDB v2 canonical learned-weight public shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping, TypeAlias

from mldb_v2.skeleton.storage.artifact_reference import ArtifactRef

if TYPE_CHECKING:
    import torch


CanonicalStateDict: TypeAlias = Mapping[str, "torch.Tensor"]


class CanonicalWeightsArtifactRef(ArtifactRef):
    format: Literal["pytorch-state-dict/v1"]
