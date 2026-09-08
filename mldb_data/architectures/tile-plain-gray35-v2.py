import torch
from torch import nn


class PlainTileShapeClassifier(nn.Module):
    def __init__(self, class_count: int = 35) -> None:
        super().__init__()
        channels = (32, 64, 128, 192)
        layers = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 5 if index == 0 else 3,
                          padding=2 if index == 0 else 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            ])
            if index < len(channels) - 1:
                layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(192, 256),
                                        nn.SiLU(inplace=True), nn.Linear(256, class_count))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)))


def build() -> nn.Module:
    return PlainTileShapeClassifier(35)
