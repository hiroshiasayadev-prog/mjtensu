from __future__ import annotations

from copy import deepcopy
from torch import nn

from .nanodet_plus_aabb_v1 import build_nanodet_plus_aabb
from nanodet_aabb_optimized.model.fpn.ghost_pan import GhostPAN


def _conv(in_ch: int, out_ch: int, kernel: int, *, groups: int = 1, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


class PlainFeaturePyramid(nn.Module):
    """Preserves Plain w500 late-DW/PW256 spatial stages, with RGB stem and optional S16."""
    def __init__(self, *, output_start_stride: int):
        super().__init__()
        if output_start_stride not in (2, 4):
            raise ValueError('output_start_stride must be 2 or 4')
        self.output_start_stride = output_start_stride
        self.stage1 = _conv(3, 16, 5)
        self.stage2 = _conv(16, 32, 3)
        self.stage3 = _conv(32, 64, 3)
        self.stage4 = nn.Sequential(_conv(64, 96, 3), _conv(96, 96, 3, groups=96), _conv(96, 256, 1))
        self.pool = nn.MaxPool2d(2, 2)
        if output_start_stride == 4:
            self.stage5 = nn.Sequential(_conv(256, 256, 3, groups=256, stride=2), _conv(256, 128, 1))

    def forward(self, images):
        x = self.pool(self.stage1(images))  # S2
        s2 = self.stage2(x)
        x = self.pool(s2)                    # S4
        s3 = self.stage3(x)
        x = self.pool(s3)                    # S8
        s4 = self.stage4(x)
        if self.output_start_stride == 2:
            return (s2, s3, s4)
        return (s3, s4, self.stage5(s4))


def build_plain_m320_detector(*, output_start_stride: int) -> nn.Module:
    """Preserve NanoDet Plus/GhostPAN/head while switching only backbone and four head strides."""
    if output_start_stride not in (2, 4):
        raise ValueError(output_start_stride)
    model = build_nanodet_plus_aabb(use_depthwise=True, use_res=False, num_blocks=1)
    model.backbone = PlainFeaturePyramid(output_start_stride=output_start_stride)
    channels = [32, 64, 256] if output_start_stride == 2 else [64, 256, 128]
    model.fpn = GhostPAN(
        in_channels=channels, out_channels=96, kernel_size=5,
        num_blocks=1, use_res=False, num_extra_level=1,
        use_depthwise=True, activation='LeakyReLU',
    )
    model.aux_fpn = deepcopy(model.fpn)
    strides = [output_start_stride * (2 ** i) for i in range(4)]
    model.head.strides = strides
    model.aux_head.strides = strides
    return model
