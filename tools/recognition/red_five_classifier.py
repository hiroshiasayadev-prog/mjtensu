from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


DEFAULT_C8_FIELDS = (8, 16, 32, 64)
SUPPORTED_INPUT_MODES = ("rgb", "cr", "ycr")
INPUT_CHANNELS = {
    "rgb": 3,
    "cr": 1,
    "ycr": 2,
}


@dataclass(frozen=True)
class ModelDescription:
    name: str
    input_mode: str
    input_channels: int
    parameter_count: int
    trainable_parameter_count: int
    c8_fields: tuple[int, ...]


class C8RedFiveClassifier(nn.Module):
    """C8 rotation-equivariant binary red-five classifier.

    Input channels are scalar fields under spatial rotation: RGB channels do not
    permute when the tile rotates, and Y/Cr are scalar functions of RGB.  Each
    channel is therefore represented by one trivial representation.  The
    convolutional backbone remains rotation-equivariant, then GroupPooling
    removes the orientation coordinate before the binary classification head.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        fields: Sequence[int] = DEFAULT_C8_FIELDS,
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        field_counts = tuple(int(value) for value in fields)
        if len(field_counts) < 2 or any(value < 1 for value in field_counts):
            raise ValueError("fields must contain at least two positive field counts")

        try:
            from escnn import gspaces
            from escnn import nn as enn
        except ImportError as error:  # pragma: no cover - Linux training dependency
            raise RuntimeError(
                "C8RedFiveClassifier requires escnn in the training environment."
            ) from error

        self.input_channels = int(input_channels)
        self.field_counts = field_counts
        self._enn = enn
        self.gspace = gspaces.rot2dOnR2(8)
        self.input_type = enn.FieldType(
            self.gspace,
            [self.gspace.trivial_repr] * self.input_channels,
        )

        layers: list[nn.Module] = []
        in_type = self.input_type
        for block_index, field_count in enumerate(field_counts):
            out_type = enn.FieldType(
                self.gspace,
                [self.gspace.regular_repr] * field_count,
            )
            kernel_size = 5 if block_index == 0 else 3
            padding = kernel_size // 2
            layers.extend(
                [
                    enn.R2Conv(
                        in_type,
                        out_type,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    ),
                    enn.InnerBatchNorm(out_type),
                    enn.ReLU(out_type, inplace=True),
                ]
            )
            if block_index < len(field_counts) - 1:
                layers.append(
                    enn.PointwiseMaxPool(
                        out_type,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )
            in_type = out_type

        self.equivariant_backbone = enn.SequentialModule(*layers)
        self.group_pool = enn.GroupPooling(in_type)
        invariant_channels = self.group_pool.out_type.size
        hidden = max(64, invariant_channels * 2)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(invariant_channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected NCHW with {self.input_channels} channels, got {tuple(images.shape)}"
            )
        geometric = self._enn.GeometricTensor(images, self.input_type)
        features = self.equivariant_backbone(geometric)
        invariant = self.group_pool(features).tensor
        pooled = self.spatial_pool(invariant)
        return self.classifier(pooled)


def normalize_input_mode(input_mode: str) -> str:
    normalized = input_mode.lower().strip()
    if normalized not in SUPPORTED_INPUT_MODES:
        raise ValueError(
            f"Unsupported input mode: {input_mode!r}; expected one of {SUPPORTED_INPUT_MODES}"
        )
    return normalized


def build_model(
    input_mode: str,
    *,
    c8_fields: Sequence[int] = DEFAULT_C8_FIELDS,
) -> C8RedFiveClassifier:
    normalized = normalize_input_mode(input_mode)
    return C8RedFiveClassifier(
        input_channels=INPUT_CHANNELS[normalized],
        fields=c8_fields,
    )


def describe_model(
    model: C8RedFiveClassifier,
    input_mode: str,
) -> ModelDescription:
    normalized = normalize_input_mode(input_mode)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return ModelDescription(
        name="c8_red_five",
        input_mode=normalized,
        input_channels=INPUT_CHANNELS[normalized],
        parameter_count=parameter_count,
        trainable_parameter_count=trainable,
        c8_fields=tuple(model.field_counts),
    )
