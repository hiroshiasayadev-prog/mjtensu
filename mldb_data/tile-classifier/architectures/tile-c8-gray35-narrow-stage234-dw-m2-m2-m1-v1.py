from torch import nn

from ..lib.c8_narrow_stage234_dw_multiplier_classifier_v1 import (
    build_c8_narrow_stage234_dw_multiplier_classifier,
)


def build() -> nn.Module:
    return build_c8_narrow_stage234_dw_multiplier_classifier(
        stage3_multiplier=2,
        stage4_multiplier=1,
    )
