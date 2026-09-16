from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x0_5

STRIDES = (2, 4, 8)
STEM_STRIDE = 1
USE_MAXPOOL = False
INCLUDE_P2 = False


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


def make_normalization(channels: int, normalization: str) -> nn.Module:
    if normalization == "batch":
        return nn.BatchNorm2d(channels)
    if normalization == "group":
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(normalization)


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, channels: int, normalization: str = "batch") -> None:
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            make_normalization(channels, normalization),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            make_normalization(channels, normalization),
            nn.SiLU(inplace=True),
        )


class ShuffleNetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        source = shufflenet_v2_x0_5(weights=None)
        self.conv1 = source.conv1
        self.conv1[0].stride = (STEM_STRIDE, STEM_STRIDE)
        self.maxpool = source.maxpool if USE_MAXPOOL else nn.Identity()
        self.stage2 = source.stage2
        self.stage3 = source.stage3
        self.stage4 = source.stage4
        self.out_channels = (24, 48, 96, 192) if INCLUDE_P2 else (48, 96, 192)

    def forward(self, x: torch.Tensor):
        stem = self.conv1(x)
        pooled = self.maxpool(stem)
        c3 = self.stage2(pooled)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        if INCLUDE_P2:
            return pooled, c3, c4, c5
        return c3, c4, c5


class TinyFPN(nn.Module):
    def __init__(self, in_channels, out_channels: int = 64) -> None:
        super().__init__()
        self.laterals = nn.ModuleList([ConvBNAct(int(value), out_channels) for value in in_channels])
        self.outputs = nn.ModuleList([DepthwiseSeparable(out_channels, "batch") for _ in in_channels])

    def forward(self, features):
        laterals = [layer(feature) for layer, feature in zip(self.laterals, features)]
        merged = [None] * len(laterals)
        merged[-1] = laterals[-1]
        for index in range(len(laterals) - 2, -1, -1):
            merged[index] = laterals[index] + F.interpolate(
                merged[index + 1], size=laterals[index].shape[-2:], mode="nearest"
            )
        return tuple(layer(feature) for layer, feature in zip(self.outputs, merged))


class RotatedFCOSHead(nn.Module):
    def __init__(self, channels: int = 64, stacked_convs: int = 2) -> None:
        super().__init__()
        self.tower = nn.Sequential(*(DepthwiseSeparable(channels, "group") for _ in range(stacked_convs)))
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.centerness = nn.Conv2d(channels, 1, 1)
        self.regression = nn.Conv2d(channels, 6, 1)
        nn.init.constant_(self.objectness.bias, -4.59511985013459)
        nn.init.zeros_(self.centerness.bias)
        nn.init.zeros_(self.regression.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        hidden = self.tower(feature)
        return torch.cat((self.objectness(hidden), self.centerness(hidden), self.regression(hidden)), dim=1)


class RotatedFCOS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.strides = STRIDES
        self.backbone = ShuffleNetBackbone()
        self.fpn = TinyFPN(self.backbone.out_channels, 64)
        self.head = RotatedFCOSHead(64, 2)

    def forward(self, images: torch.Tensor):
        features = self.fpn(self.backbone(images))
        return tuple(self.head(feature) for feature in features)


def build() -> nn.Module:
    return RotatedFCOS()
