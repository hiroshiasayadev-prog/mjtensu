from torch import nn

from ..lib.c8_rgb_full_width_classifier_v1 import build_c8_rgb_full_width_classifier


def build() -> nn.Module:
    return build_c8_rgb_full_width_classifier(fields=(8, 16, 32, 64))
