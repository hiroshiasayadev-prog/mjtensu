from torch import nn

from ..lib.c8_rgb_aggressive_partial_dw_classifier_v1 import build_c8_rgb_aggressive_partial_dw_classifier


def build() -> nn.Module:
    return build_c8_rgb_aggressive_partial_dw_classifier(fields=(6, 12, 24, 48), dw_stages=(4,), dw_multiplier=1)
