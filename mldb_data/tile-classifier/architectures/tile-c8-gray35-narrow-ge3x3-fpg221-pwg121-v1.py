from torch import nn

from ..lib.c8_narrow_stage234_grouped_dwpw_classifier_v1 import (
    build_c8_narrow_stage234_grouped_dwpw_classifier,
)


def build() -> nn.Module:
    return build_c8_narrow_stage234_grouped_dwpw_classifier(
        dw_fields_per_group=(2, 2, 1),
        pw_groups=(1, 2, 1),
    )
