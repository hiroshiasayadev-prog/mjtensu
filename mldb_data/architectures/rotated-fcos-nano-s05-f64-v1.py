from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights,
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
)

STRIDES = (8, 16, 32)

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
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
    raise ValueError(f"Unsupported normalization: {normalization}")

class DepthwiseSeparable(nn.Sequential):
    def __init__(self, channels: int, *, normalization: str = "batch") -> None:
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            make_normalization(channels, normalization),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            make_normalization(channels, normalization),
            nn.SiLU(inplace=True),
        )

class ShuffleNetBackbone(nn.Module):
    def __init__(self, variant: str, *, pretrained: bool) -> None:
        super().__init__()
        if variant == "shufflenet_v2_x0_5":
            weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
            model = shufflenet_v2_x0_5(weights=weights)
        elif variant == "shufflenet_v2_x1_0":
            weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
            model = shufflenet_v2_x1_0(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {variant}")
        self.conv1 = model.conv1
        self.maxpool = model.maxpool
        self.stage2 = model.stage2
        self.stage3 = model.stage3
        self.stage4 = model.stage4
        self.out_channels = self._infer_channels()

    def _infer_channels(self) -> tuple[int, int, int]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            sample = torch.zeros(1, 3, 320, 320)
            c3, c4, c5 = self.forward(sample)
        self.train(was_training)
        return (int(c3.shape[1]), int(c4.shape[1]), int(c5.shape[1]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.maxpool(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5

class TinyFPN(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("TinyFPN requires exactly three backbone feature levels")
        self.lateral3 = ConvBNAct(int(in_channels[0]), out_channels)
        self.lateral4 = ConvBNAct(int(in_channels[1]), out_channels)
        self.lateral5 = ConvBNAct(int(in_channels[2]), out_channels)
        self.output3 = DepthwiseSeparable(out_channels)
        self.output4 = DepthwiseSeparable(out_channels)
        self.output5 = DepthwiseSeparable(out_channels)

    def forward(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return self.output3(p3), self.output4(p4), self.output5(p5)

class RotatedFCOSHead(nn.Module):
    def __init__(self, channels: int, *, stacked_convs: int, normalization: str) -> None:
        super().__init__()
        if stacked_convs < 1:
            raise ValueError("stacked_convs must be positive")
        self.tower = nn.Sequential(
            *(DepthwiseSeparable(channels, normalization=normalization) for _ in range(stacked_convs))
        )
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.centerness = nn.Conv2d(channels, 1, 1)
        # dx, dy, log(w/stride), log(h/stride), sin(2theta), cos(2theta)
        self.regression = nn.Conv2d(channels, 6, 1)
        nn.init.constant_(self.objectness.bias, -4.59511985013459)
        nn.init.constant_(self.centerness.bias, 0.0)
        nn.init.zeros_(self.regression.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        hidden = self.tower(feature)
        return torch.cat(
            (self.objectness(hidden), self.centerness(hidden), self.regression(hidden)),
            dim=1,
        )

class RotatedFCOSNano(nn.Module):
    def __init__(
        self,
        *,
        backbone: str = "shufflenet_v2_x0_5",
        fpn_channels: int = 64,
        head_convs: int = 2,
        head_normalization: str = "batch",
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.fpn_channels = int(fpn_channels)
        self.head_convs = int(head_convs)
        self.head_normalization = str(head_normalization)
        self.backbone = ShuffleNetBackbone(backbone, pretrained=pretrained_backbone)
        self.fpn = TinyFPN(self.backbone.out_channels, self.fpn_channels)
        self.head = RotatedFCOSHead(
            self.fpn_channels,
            stacked_convs=self.head_convs,
            normalization=self.head_normalization,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.fpn(self.backbone(x))
        return tuple(self.head(feature) for feature in features)  # type: ignore[return-value]

    def config(self) -> dict[str, Any]:
        return {
            "name": "rotated_fcos_nano",
            "backbone": self.backbone_name,
            "fpn_channels": self.fpn_channels,
            "head_convs": self.head_convs,
            "head_normalization": self.head_normalization,
            "strides": list(STRIDES),
            "regression": ["dx", "dy", "logw", "logh", "sin2theta", "cos2theta"],
        }

def build() -> nn.Module:
    return RotatedFCOSNano(
        backbone="shufflenet_v2_x0_5",
        fpn_channels=64,
        head_convs=2,
        head_normalization="group",
        pretrained_backbone=False,
    )
