from __future__ import annotations

import torch
from torch import nn


class PlainStage2DepthwiseSeparableClassifier(nn.Module):
    def __init__(self, class_count: int = 35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),

            nn.Conv2d(96, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, class_count),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)))


def build() -> nn.Module:
    return PlainStage2DepthwiseSeparableClassifier(35)
