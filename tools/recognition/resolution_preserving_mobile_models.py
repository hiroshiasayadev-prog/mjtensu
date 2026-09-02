from __future__ import annotations

"""INV-012 resolution-preserving MobileNetV3 tile classifiers.

The candidate family keeps the MobileNetV3-Small inverted-residual/SE operator style
from INV-011 but makes the 64x64 spatial schedule explicit.  The standard ImageNet
stride pattern reaches 2x2 before global pooling; these variants stop at 8x8 or 4x4
and spend the remaining budget on independent same-resolution 96-channel blocks.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn

try:
    from mobile_classifier_experiment_models import (
        ConvBNActivation,
        MobileNetV3BlockConfig,
        MobileNetV3InvertedResidual,
    )
except ModuleNotFoundError:  # package-style import used by tests
    from tools.recognition.mobile_classifier_experiment_models import (
        ConvBNActivation,
        MobileNetV3BlockConfig,
        MobileNetV3InvertedResidual,
    )


DEFAULT_CLASS_COUNT = 35
SUPPORTED_FINAL_RESOLUTIONS = (4, 8)
SUPPORTED_LATE_REPEATS = (1, 2, 3)


@dataclass(frozen=True)
class ResolutionPreservingMobileDescription:
    name: str
    family: str
    final_feature_resolution: int
    late_repeats: int
    parameter_count: int
    trainable_parameter_count: int
    details: dict[str, object]


def _make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    minimum = divisor if min_value is None else min_value
    rounded = max(minimum, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return int(rounded)


def _build_configs(
    *,
    final_feature_resolution: int,
    late_repeats: int,
    width_mult: float,
) -> list[MobileNetV3BlockConfig]:
    if final_feature_resolution not in SUPPORTED_FINAL_RESOLUTIONS:
        raise ValueError(
            f"final_feature_resolution must be one of {SUPPORTED_FINAL_RESOLUTIONS}"
        )
    if late_repeats not in SUPPORTED_LATE_REPEATS:
        raise ValueError(f"late_repeats must be one of {SUPPORTED_LATE_REPEATS}")
    if width_mult != 1.0:
        raise ValueError("INV-012 fixes width_mult=1.0 to isolate spatial/depth effects")

    def adjust(value: int) -> int:
        return _make_divisible(value * width_mult, 8)

    # Standard MobileNetV3-Small block families through the 48-channel stage.
    # Only the 24->40 stage stride changes for the 8x8 endpoint.  The later
    # 48->96 transition is always stride=1, so neither candidate family ever
    # downsamples below its declared endpoint.
    raw_prefix: Sequence[tuple[int, int, int, bool, str, int]] = (
        (3, 16, 16, True, "RE", 2),
        (3, 72, 24, False, "RE", 2),
        (3, 88, 24, False, "RE", 1),
        (5, 96, 40, True, "HS", 1 if final_feature_resolution == 8 else 2),
        (5, 240, 40, True, "HS", 1),
        (5, 240, 40, True, "HS", 1),
        (5, 120, 48, True, "HS", 1),
        (5, 144, 48, True, "HS", 1),
    )

    configs: list[MobileNetV3BlockConfig] = []
    input_channels = adjust(16)
    for kernel, expanded, output, use_se, activation, stride in raw_prefix:
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

    # The first terminal block performs only a channel transition 48->96.
    # Additional repeats have independent weights and retain 96 channels.
    configs.append(
        MobileNetV3BlockConfig(
            kernel=5,
            input_channels=input_channels,
            expanded_channels=adjust(288),
            out_channels=adjust(96),
            use_se=True,
            activation="HS",
            stride=1,
        )
    )
    input_channels = adjust(96)
    for _ in range(late_repeats - 1):
        configs.append(
            MobileNetV3BlockConfig(
                kernel=5,
                input_channels=input_channels,
                expanded_channels=adjust(576),
                out_channels=adjust(96),
                use_se=True,
                activation="HS",
                stride=1,
            )
        )
    return configs


class ResolutionPreservingMobileNetV3TileClassifier(nn.Module):
    def __init__(
        self,
        *,
        class_count: int = DEFAULT_CLASS_COUNT,
        final_feature_resolution: int,
        late_repeats: int,
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        configs = _build_configs(
            final_feature_resolution=final_feature_resolution,
            late_repeats=late_repeats,
            width_mult=width_mult,
        )
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
        last_channel = _make_divisible(1024 * width_mult, 8)

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, class_count),
        )
        self.width_mult = float(width_mult)
        self.final_feature_resolution = int(final_feature_resolution)
        self.late_repeats = int(late_repeats)
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

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(
                f"Expected grayscale NCHW tensor [N,1,H,W], got {tuple(images.shape)}"
            )
        features = self.features(images)
        expected = self.final_feature_resolution
        if tuple(features.shape[-2:]) != (expected, expected):
            raise RuntimeError(
                f"INV-012 spatial contract violated: got {tuple(features.shape[-2:])}, "
                f"expected {(expected, expected)}"
            )
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(images)
        pooled = self.avgpool(features).flatten(1)
        return self.classifier(pooled)


def parse_resolution_preserving_condition(
    model_name: str,
) -> tuple[int, int]:
    normalized = model_name.strip().lower()
    prefix = "mobile-tile-f"
    if not normalized.startswith(prefix) or "-r" not in normalized:
        raise ValueError(
            f"Unsupported INV-012 model {model_name!r}; expected mobile-tile-f4-r1..r3 "
            "or mobile-tile-f8-r1..r3"
        )
    resolution_text, repeat_text = normalized[len(prefix) :].split("-r", 1)
    try:
        resolution = int(resolution_text)
        repeats = int(repeat_text)
    except ValueError as error:
        raise ValueError(f"Invalid INV-012 model name: {model_name!r}") from error
    if resolution not in SUPPORTED_FINAL_RESOLUTIONS or repeats not in SUPPORTED_LATE_REPEATS:
        raise ValueError(f"Invalid INV-012 model name: {model_name!r}")
    return resolution, repeats


def build_resolution_preserving_mobile_classifier(
    model_name: str,
    *,
    class_count: int = DEFAULT_CLASS_COUNT,
) -> ResolutionPreservingMobileNetV3TileClassifier:
    resolution, repeats = parse_resolution_preserving_condition(model_name)
    return ResolutionPreservingMobileNetV3TileClassifier(
        class_count=class_count,
        final_feature_resolution=resolution,
        late_repeats=repeats,
        width_mult=1.0,
    )


def describe_resolution_preserving_mobile_classifier(
    model: ResolutionPreservingMobileNetV3TileClassifier,
    model_name: str,
) -> ResolutionPreservingMobileDescription:
    resolution, repeats = parse_resolution_preserving_condition(model_name)
    if model.final_feature_resolution != resolution or model.late_repeats != repeats:
        raise ValueError("Model topology does not match its INV-012 condition name")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return ResolutionPreservingMobileDescription(
        name=model_name.strip().lower(),
        family="mobilenet-v3-small-tile",
        final_feature_resolution=resolution,
        late_repeats=repeats,
        parameter_count=int(parameter_count),
        trainable_parameter_count=int(trainable_parameter_count),
        details={
            "input_channels": 1,
            "input_size": 64,
            "class_count": int(model.classifier[-1].out_features),
            "width_mult": model.width_mult,
            "final_feature_resolution": resolution,
            "late_repeats": repeats,
            "last_conv_channels": model.last_conv_channels,
            "last_channel": model.last_channel,
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
        },
    )
