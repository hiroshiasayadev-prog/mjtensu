import torch
from torch import nn


class PlainStridedTileShapeClassifier(nn.Module):
    def __init__(self, class_count: int = 35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(24), nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.SiLU(inplace=True),
            nn.Conv2d(96, 144, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(144), nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(144, 256),
                                        nn.SiLU(inplace=True), nn.Linear(256, class_count))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)))


def build() -> nn.Module:
    return PlainStridedTileShapeClassifier(35)
