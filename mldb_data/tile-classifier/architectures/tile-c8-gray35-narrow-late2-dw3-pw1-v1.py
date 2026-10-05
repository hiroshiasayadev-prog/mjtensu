from torch import nn

from ..lib.c8_narrow_late2_dw3_pw1_classifier_v1 import (
    build_c8_narrow_late2_dw3_pw1_classifier,
)


def build() -> nn.Module:
    return build_c8_narrow_late2_dw3_pw1_classifier()
