from torch import nn
from ..lib.nanodet_plus_aabb_v1 import build_nanodet_plus_aabb

def build() -> nn.Module:
    return build_nanodet_plus_aabb(use_depthwise=False, use_res=False, num_blocks=1)
