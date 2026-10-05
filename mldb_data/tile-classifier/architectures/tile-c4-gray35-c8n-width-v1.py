from torch import nn

from ..lib.cyclic_equivariant_tile_shape_classifier_v1 import (
    build_cyclic_equivariant_tile_shape_classifier,
)


def build() -> nn.Module:
    return build_cyclic_equivariant_tile_shape_classifier(
        group_size=4,
        fields=(8, 16, 32, 64),
    )
