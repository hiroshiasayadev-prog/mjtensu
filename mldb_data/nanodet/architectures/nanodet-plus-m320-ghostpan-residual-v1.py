from torch import nn
from ..lib.nanodet_plus_aabb_v1 import build_nanodet_plus_aabb

def build() -> nn.Module:
    return build_nanodet_plus_aabb(use_depthwise=True, use_res=True, num_blocks=1)
