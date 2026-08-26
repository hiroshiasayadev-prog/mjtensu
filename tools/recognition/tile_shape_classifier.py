from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


DEFAULT_CLASS_COUNT = 34
DEFAULT_C8_FIELDS = (8, 16, 32, 64)


@dataclass(frozen=True)
class ModelDescription:
    name: str
    parameter_count: int
    trainable_parameter_count: int
    c8_fields: tuple[int, ...] | None


class C8TileShapeClassifier(nn.Module):
    """C8 rotation-equivariant grayscale classifier.

    Each regular representation carries eight orientation channels. GroupPooling
    removes the orientation coordinate only after the equivariant convolutional
    backbone, producing a rotation-invariant descriptor for the tile class head.
    """

    def __init__(
        self,
        *,
        class_count: int = DEFAULT_CLASS_COUNT,
        fields: Sequence[int] = DEFAULT_C8_FIELDS,
    ) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("class_count must be at least 2")
        field_counts = tuple(int(value) for value in fields)
        if len(field_counts) < 2 or any(value < 1 for value in field_counts):
            raise ValueError("fields must contain at least two positive field counts")

        try:
            from escnn import gspaces
            from escnn import nn as enn
        except ImportError as error:  # pragma: no cover - Linux training dependency
            raise RuntimeError(
                "C8TileShapeClassifier requires escnn. Install it in the Linux "
                "training environment (for example: pip install escnn==1.0.11)."
            ) from error

        self.class_count = int(class_count)
        self.field_counts = field_counts
        self._enn = enn
        self.gspace = gspaces.rot2dOnR2(8)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])

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
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(invariant_channels, max(128, invariant_channels * 2)),
            nn.SiLU(inplace=True),
            nn.Linear(max(128, invariant_channels * 2), self.class_count),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(
                f"Expected grayscale NCHW tensor with shape [N,1,H,W], got {tuple(images.shape)}"
            )
        geometric = self._enn.GeometricTensor(images, self.input_type)
        features = self.equivariant_backbone(geometric)
        invariant = self.group_pool(features).tensor
        pooled = self.spatial_pool(invariant)
        return self.classifier(pooled)


class PlainTileShapeClassifier(nn.Module):
    """Small conventional CNN kept as an optional reference baseline."""

    def __init__(self, *, class_count: int = DEFAULT_CLASS_COUNT) -> None:
        super().__init__()
        channels = (32, 64, 128, 192)
        layers: list[nn.Module] = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=5 if index == 0 else 3,
                        padding=2 if index == 0 else 1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            if index < len(channels) - 1:
                layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, class_count),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)))


def build_model(
    model_name: str,
    *,
    class_count: int = DEFAULT_CLASS_COUNT,
    c8_fields: Sequence[int] = DEFAULT_C8_FIELDS,
) -> nn.Module:
    normalized = model_name.lower().strip()
    if normalized == "c8":
        return C8TileShapeClassifier(class_count=class_count, fields=c8_fields)
    if normalized == "plain":
        return PlainTileShapeClassifier(class_count=class_count)
    raise ValueError(f"Unsupported model: {model_name!r}; expected 'c8' or 'plain'")


def describe_model(model: nn.Module, model_name: str) -> ModelDescription:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    fields = (
        tuple(model.field_counts)
        if isinstance(model, C8TileShapeClassifier)
        else None
    )
    return ModelDescription(
        name=model_name,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable,
        c8_fields=fields,
    )
