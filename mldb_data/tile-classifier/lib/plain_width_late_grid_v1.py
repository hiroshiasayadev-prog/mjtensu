from __future__ import annotations

import torch
from torch import nn


def build_plain_width_late_classifier(
    channels: tuple[int, int, int, int],
    *,
    late_channels: int | None,
    class_count: int = 35,
) -> nn.Module:
    layers: list[nn.Module] = []
    in_channels = 1
    for index, out_channels in enumerate(channels):
        layers.extend([
            nn.Conv2d(
                in_channels,
                out_channels,
                5 if index == 0 else 3,
                padding=2 if index == 0 else 1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        ])
        if index < len(channels) - 1:
            layers.append(nn.MaxPool2d(2, 2))
        in_channels = out_channels
    head_channels = channels[-1]
    if late_channels is not None:
        layers.extend([
            nn.Conv2d(head_channels, late_channels, 1, bias=False),
            nn.BatchNorm2d(late_channels),
            nn.SiLU(inplace=True),
        ])
        head_channels = late_channels
    return _PlainWidthLateClassifier(
        features=nn.Sequential(*layers),
        head_channels=head_channels,
        class_count=class_count,
    )


class _PlainWidthLateClassifier(nn.Module):
    def __init__(self, *, features: nn.Sequential, head_channels: int, class_count: int) -> None:
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_channels, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, class_count),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)))
