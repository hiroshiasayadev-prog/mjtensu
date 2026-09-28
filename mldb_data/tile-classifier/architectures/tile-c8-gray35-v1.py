from torch import nn

from ..lib.c8_tile_shape_classifier_v1 import build_c8_tile_shape_classifier


def build() -> nn.Module:
    return build_c8_tile_shape_classifier()
