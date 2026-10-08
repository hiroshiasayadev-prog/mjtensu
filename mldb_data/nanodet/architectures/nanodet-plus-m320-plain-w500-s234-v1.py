from torch import nn
from ..lib.plain_m320_ghostpan_backbone_v1 import build_plain_m320_detector

def build() -> nn.Module:
    return build_plain_m320_detector(output_start_stride=2)
