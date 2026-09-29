from __future__ import annotations

import torch
from torch import nn


def build_plain_stage_kernel_classifier(
    *,
    stage_kernels: tuple[int, int, int, int],
    channels: tuple[int, int, int, int] = (16, 32, 64, 96),
    late_channels: int = 256,
    class_count: int = 35,
) -> nn.Module:
    if len(stage_kernels) != 4 or any(kernel not in (1, 3, 5) for kernel in stage_kernels):
        raise ValueError("stage_kernels must contain four values drawn from 1, 3, or 5")
    layers: list[nn.Module] = []
    in_channels = 1
    for index, (out_channels, kernel) in enumerate(zip(channels, stage_kernels, strict=True)):
        layers.extend([
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                padding=kernel // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        ])
        if index < len(channels) - 1:
            layers.append(nn.MaxPool2d(2, 2))
        in_channels = out_channels
    layers.extend([
        nn.Conv2d(channels[-1], late_channels, 1, bias=False),
        nn.BatchNorm2d(late_channels),
        nn.SiLU(inplace=True),
    ])
    return _PlainStageKernelClassifier(
        features=nn.Sequential(*layers),
        late_channels=late_channels,
        class_count=class_count,
    )


class _PlainStageKernelClassifier(nn.Module):
    def __init__(self, *, features: nn.Sequential, late_channels: int, class_count: int) -> None:
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(late_channels, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, class_count),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)))
