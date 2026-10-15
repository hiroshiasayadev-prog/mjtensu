from torch import nn

from ..lib.c8_rgb_redfeature_stage234_dw222_v1 import build_c8_rgb_redfeature_stage234_dw222_classifier


def build() -> nn.Module:
    return build_c8_rgb_redfeature_stage234_dw222_classifier(feature_kind='normalized-red', fields=(6, 12, 24, 48))
