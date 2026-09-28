from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn


CLASS_COUNT = 35
WIDTH_MULT = 1.0


def _make_divisible(
    value: float,
    divisor: int = 8,
    min_value: int | None = None,
) -> int:
    minimum = divisor if min_value is None else min_value
    rounded = max(minimum, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return int(rounded)


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation_layer: Callable[[], nn.Module] | None = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ) -> None:
        padding = (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_channels),
        ]
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.scale_activation = nn.Hardsigmoid(inplace=True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(input_tensor)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return input_tensor * scale


@dataclass(frozen=True)
class MobileNetV3BlockConfig:
    kernel: int
    input_channels: int
    expanded_channels: int
    out_channels: int
    use_se: bool
    activation: str
    stride: int


class MobileNetV3InvertedResidual(nn.Module):
    def __init__(
        self,
        config: MobileNetV3BlockConfig,
        *,
        norm_layer: Callable[[int], nn.Module],
    ) -> None:
        super().__init__()
        if config.stride not in (1, 2):
            raise ValueError("MobileNetV3 stride must be 1 or 2")
        if config.activation == "HS":
            activation_factory: Callable[[], nn.Module] = (
                lambda: nn.Hardswish(inplace=True)
            )
        elif config.activation == "RE":
            activation_factory = lambda: nn.ReLU(inplace=True)
        else:
            raise ValueError(
                f"Unsupported MobileNetV3 activation: {config.activation}"
            )

        layers: list[nn.Module] = []
        if config.expanded_channels != config.input_channels:
            layers.append(
                ConvBNActivation(
                    config.input_channels,
                    config.expanded_channels,
                    kernel_size=1,
                    activation_layer=activation_factory,
                    norm_layer=norm_layer,
                )
            )
        layers.append(
            ConvBNActivation(
                config.expanded_channels,
                config.expanded_channels,
                kernel_size=config.kernel,
                stride=config.stride,
                groups=config.expanded_channels,
                activation_layer=activation_factory,
                norm_layer=norm_layer,
            )
        )
        if config.use_se:
            squeeze_channels = _make_divisible(
                config.expanded_channels // 4,
                8,
            )
            layers.append(
                SqueezeExcitation(
                    config.expanded_channels,
                    squeeze_channels,
                )
            )
        layers.append(
            ConvBNActivation(
                config.expanded_channels,
                config.out_channels,
                kernel_size=1,
                activation_layer=None,
                norm_layer=norm_layer,
            )
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = (
            config.stride == 1
            and config.input_channels == config.out_channels
        )
        self.config = config

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        result = self.block(input_tensor)
        if self.use_residual:
            result = result + input_tensor
        return result


def _configs(
    width_mult: float = WIDTH_MULT,
) -> tuple[list[MobileNetV3BlockConfig], int]:
    if width_mult != 1.0:
        raise ValueError("This production definition is fixed to width_mult=1.0")

    def adjust(value: int) -> int:
        return _make_divisible(value * width_mult, 8)

    raw: Sequence[tuple[int, int, int, bool, str, int]] = (
        (3, 16, 16, True, "RE", 2),
        (3, 72, 24, False, "RE", 2),
        (3, 88, 24, False, "RE", 1),
        (5, 96, 40, True, "HS", 2),
        (5, 240, 40, True, "HS", 1),
        (5, 240, 40, True, "HS", 1),
        (5, 120, 48, True, "HS", 1),
        (5, 144, 48, True, "HS", 1),
        (5, 288, 96, True, "HS", 2),
        (5, 576, 96, True, "HS", 1),
        (5, 576, 96, True, "HS", 1),
    )
    input_channels = adjust(16)
    configs: list[MobileNetV3BlockConfig] = []
    for kernel, expanded, output, use_se, activation, stride in raw:
        config = MobileNetV3BlockConfig(
            kernel=kernel,
            input_channels=input_channels,
            expanded_channels=adjust(expanded),
            out_channels=adjust(output),
            use_se=use_se,
            activation=activation,
            stride=stride,
        )
        configs.append(config)
        input_channels = config.out_channels
    return configs, adjust(1024)


class MobileNetV3SmallTileClassifier(nn.Module):
    def __init__(
        self,
        *,
        class_count: int = CLASS_COUNT,
        width_mult: float = WIDTH_MULT,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        configs, last_channel = _configs(width_mult)
        norm_layer: Callable[[int], nn.Module] = lambda channels: nn.BatchNorm2d(
            channels,
            eps=0.001,
            momentum=0.01,
        )
        first_channels = configs[0].input_channels
        layers: list[nn.Module] = [
            ConvBNActivation(
                1,
                first_channels,
                kernel_size=3,
                stride=2,
                activation_layer=lambda: nn.Hardswish(inplace=True),
                norm_layer=norm_layer,
            )
        ]
        layers.extend(
            MobileNetV3InvertedResidual(
                config,
                norm_layer=norm_layer,
            )
            for config in configs
        )
        last_block_channels = configs[-1].out_channels
        last_conv_channels = 6 * last_block_channels
        layers.append(
            ConvBNActivation(
                last_block_channels,
                last_conv_channels,
                kernel_size=1,
                activation_layer=lambda: nn.Hardswish(inplace=True),
                norm_layer=norm_layer,
            )
        )
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, class_count),
        )
        self.width_mult = float(width_mult)
        self.block_configs = tuple(configs)
        self.last_conv_channels = int(last_conv_channels)
        self.last_channel = int(last_channel)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(
                f"Expected grayscale NCHW tensor [N,1,H,W], got {tuple(images.shape)}"
            )
        features = self.features(images)
        features = self.avgpool(features).flatten(1)
        return self.classifier(features)


def build_mobilenet_v3_small_tile_classifier() -> nn.Module:
    return MobileNetV3SmallTileClassifier(
        class_count=CLASS_COUNT,
        width_mult=WIDTH_MULT,
    )
