from torch import nn

from ..lib.plain_stage_kernel_ablation_v1 import build_plain_stage_kernel_classifier


def build() -> nn.Module:
    return build_plain_stage_kernel_classifier(stage_kernels=(5, 3, 1, 1))
