import torch
from torch import nn
from torchvision.models import shufflenet_v2_x0_5


class TileShuffleNetV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = shufflenet_v2_x0_5(weights=None)
        out_channels = base.conv1[0].out_channels
        base.conv1[0] = nn.Conv2d(1, out_channels, 3, stride=2, padding=1, bias=False)
        nn.init.kaiming_normal_(base.conv1[0].weight, mode="fan_out")
        self.features = nn.Sequential(
            base.conv1,
            base.maxpool,
            base.stage2,
            base.stage3,
            base.stage4,
            base.conv5,
        )
        self.classifier = nn.Linear(base.fc.in_features, 35)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features.mean(dim=(2, 3)))


def build() -> nn.Module:
    return TileShuffleNetV2()
