from torch import nn

from ..lib.c8_rgb_full_stride_classifier_v1 import build_c8_rgb_full_stride_classifier


def build() -> nn.Module:
    return build_c8_rgb_full_stride_classifier(fields=(6, 12, 24, 48), stage_strides=(1, 2, 2, 2))
