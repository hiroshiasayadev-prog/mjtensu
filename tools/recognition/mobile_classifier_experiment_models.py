from __future__ import annotations

"""INV-011 mobile-oriented base-tile classifier candidates.

The implementations are intentionally local to the investigation so the accepted
INV-007/008 Plain/C8 implementations stay reproducible.  The topologies follow the
standard ShuffleNetV2 and MobileNetV3-Small designs, adapted only for one grayscale
input channel, 64x64 inputs, and a 35-class output head.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn


DEFAULT_CLASS_COUNT = 35
SHUFFLENET_V2_STAGE_REPEATS = (4, 8, 4)
SHUFFLENET_V2_CHANNELS = {
    0.5: (24, 48, 96, 192, 1024),
    1.0: (24, 116, 232, 464, 1024),
}


@dataclass(frozen=True)
class MobileClassifierDescription:
    name: str
    family: str
    width_mult: float
    parameter_count: int
    trainable_parameter_count: int
    details: dict[str, object]


def _make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    minimum = divisor if min_value is None else min_value
    rounded = max(minimum, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return int(rounded)


def channel_shuffle(input_tensor: torch.Tensor, groups: int) -> torch.Tensor:
    if input_tensor.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got {tuple(input_tensor.shape)}")
    batch, channels, height, width = input_tensor.shape
    if channels % groups != 0:
        raise ValueError(f"Channel count {channels} is not divisible by groups={groups}")
    channels_per_group = channels // groups
    shuffled = input_tensor.reshape(batch, groups, channels_per_group, height, width)
    shuffled = shuffled.transpose(1, 2).contiguous()
    return shuffled.reshape(batch, channels, height, width)


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


class ShuffleNetV2InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if stride not in (1, 2, 3):
            raise ValueError("ShuffleNetV2 stride must be 1, 2, or 3")
        branch_channels = out_channels // 2
        if stride == 1 and in_channels != branch_channels * 2:
            raise ValueError(
                "Stride-1 ShuffleNetV2 block requires input channels == output channels"
            )
        if out_channels % 2 != 0:
            raise ValueError("ShuffleNetV2 output channels must be even")

        if stride > 1:
            self.branch1 = nn.Sequential(
                self._depthwise_conv(in_channels, in_channels, stride=stride),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Identity()

        branch2_input = in_channels if stride > 1 else branch_channels
        self.branch2 = nn.Sequential(
            nn.Conv2d(branch2_input, branch_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            self._depthwise_conv(branch_channels, branch_channels, stride=stride),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(branch_channels, branch_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.stride = int(stride)

    @staticmethod
    def _depthwise_conv(
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
    ) -> nn.Conv2d:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            left, right = input_tensor.chunk(2, dim=1)
            output = torch.cat((left, self.branch2(right)), dim=1)
        else:
            output = torch.cat((self.branch1(input_tensor), self.branch2(input_tensor)), dim=1)
        return channel_shuffle(output, 2)


class ShuffleNetV2TileClassifier(nn.Module):
    def __init__(
        self,
        *,
        class_count: int = DEFAULT_CLASS_COUNT,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        if width_mult not in SHUFFLENET_V2_CHANNELS:
            raise ValueError(
                f"Unsupported ShuffleNetV2 width {width_mult}; expected one of "
                f"{sorted(SHUFFLENET_V2_CHANNELS)}"
            )
        stage_channels = SHUFFLENET_V2_CHANNELS[width_mult]
        input_channels = stage_channels[0]
        self.conv1 = ConvBNActivation(
            1,
            input_channels,
            kernel_size=3,
            stride=2,
            activation_layer=lambda: nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stages: list[nn.Module] = []
        for repeats, output_channels in zip(
            SHUFFLENET_V2_STAGE_REPEATS, stage_channels[1:4]
        ):
            blocks = [ShuffleNetV2InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                blocks.append(
                    ShuffleNetV2InvertedResidual(output_channels, output_channels, 1)
                )
            stages.append(nn.Sequential(*blocks))
            input_channels = output_channels
        self.stage2, self.stage3, self.stage4 = stages
        self.conv5 = ConvBNActivation(
            input_channels,
            stage_channels[-1],
            kernel_size=1,
            activation_layer=lambda: nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(stage_channels[-1], class_count)
        self.width_mult = float(width_mult)
        self.stage_channels = tuple(int(value) for value in stage_channels)
        self.stage_repeats = SHUFFLENET_V2_STAGE_REPEATS
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
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
        features = self.conv1(images)
        features = self.maxpool(features)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.stage4(features)
        features = self.conv5(features)
        features = features.mean(dim=(2, 3))
        return self.classifier(features)


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
        activation_factory: Callable[[], nn.Module]
        if config.activation == "HS":
            activation_factory = lambda: nn.Hardswish(inplace=True)
        elif config.activation == "RE":
            activation_factory = lambda: nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported MobileNetV3 activation: {config.activation}")

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
            squeeze_channels = _make_divisible(config.expanded_channels // 4, 8)
            layers.append(SqueezeExcitation(config.expanded_channels, squeeze_channels))
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
            config.stride == 1 and config.input_channels == config.out_channels
        )
        self.out_channels = config.out_channels
        self.config = config

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        result = self.block(input_tensor)
        if self.use_residual:
            result = result + input_tensor
        return result


def _mobilenet_v3_small_configs(width_mult: float) -> tuple[list[MobileNetV3BlockConfig], int]:
    if width_mult not in (0.5, 1.0):
        raise ValueError("MobileNetV3-Small width must be 0.5 or 1.0")

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
        class_count: int = DEFAULT_CLASS_COUNT,
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        configs, last_channel = _mobilenet_v3_small_configs(width_mult)
        norm_layer: Callable[[int], nn.Module] = lambda channels: nn.BatchNorm2d(
            channels, eps=0.001, momentum=0.01
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
            MobileNetV3InvertedResidual(config, norm_layer=norm_layer)
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


def build_mobile_classifier(
    model_name: str,
    *,
    class_count: int = DEFAULT_CLASS_COUNT,
) -> nn.Module:
    normalized = model_name.strip().lower()
    if normalized == "shufflenet-v2-0.5x":
        return ShuffleNetV2TileClassifier(class_count=class_count, width_mult=0.5)
    if normalized == "shufflenet-v2-1.0x":
        return ShuffleNetV2TileClassifier(class_count=class_count, width_mult=1.0)
    if normalized == "mobilenet-v3-small-0.5x":
        return MobileNetV3SmallTileClassifier(class_count=class_count, width_mult=0.5)
    if normalized == "mobilenet-v3-small-1.0x":
        return MobileNetV3SmallTileClassifier(class_count=class_count, width_mult=1.0)
    raise ValueError(
        f"Unsupported mobile classifier {model_name!r}; expected ShuffleNetV2 0.5x/1.0x "
        "or MobileNetV3-Small 0.5x/1.0x"
    )


def describe_mobile_classifier(
    model: nn.Module,
    model_name: str,
) -> MobileClassifierDescription:
    normalized = model_name.strip().lower()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    if isinstance(model, ShuffleNetV2TileClassifier):
        family = "shufflenet-v2"
        details: dict[str, object] = {
            "stage_channels": list(model.stage_channels),
            "stage_repeats": list(model.stage_repeats),
            "activation": "ReLU",
            "input_channels": 1,
            "input_size": 64,
            "class_count": int(model.classifier.out_features),
        }
    elif isinstance(model, MobileNetV3SmallTileClassifier):
        family = "mobilenet-v3-small"
        details = {
            "blocks": [
                {
                    "kernel": config.kernel,
                    "input_channels": config.input_channels,
                    "expanded_channels": config.expanded_channels,
                    "out_channels": config.out_channels,
                    "use_se": config.use_se,
                    "activation": config.activation,
                    "stride": config.stride,
                }
                for config in model.block_configs
            ],
            "last_conv_channels": model.last_conv_channels,
            "last_channel": model.last_channel,
            "activation": "ReLU/Hardswish",
            "input_channels": 1,
            "input_size": 64,
            "class_count": int(model.classifier[-1].out_features),
        }
    else:
        raise TypeError(f"Unexpected mobile classifier type for {normalized}: {type(model)!r}")

    return MobileClassifierDescription(
        name=normalized,
        family=family,
        width_mult=float(getattr(model, "width_mult")),
        parameter_count=int(parameter_count),
        trainable_parameter_count=int(trainable_parameter_count),
        details=details,
    )
