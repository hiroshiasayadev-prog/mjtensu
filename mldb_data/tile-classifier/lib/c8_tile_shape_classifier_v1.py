from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


C8_GROUP_SIZE = 8
C8_FIELDS = (8, 16, 32, 64)
CLASS_COUNT = 35


class C8TileShapeClassifier(nn.Module):
    """C8 rotation-equivariant grayscale tile classifier."""

    def __init__(
        self,
        *,
        class_count: int = CLASS_COUNT,
        fields: Sequence[int] = C8_FIELDS,
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
        except ImportError as error:
            raise RuntimeError(
                "C8TileShapeClassifier requires escnn==1.0.11-compatible semantics."
            ) from error

        self.class_count = int(class_count)
        self.field_counts = field_counts
        self._enn = enn
        self.gspace = gspaces.rot2dOnR2(C8_GROUP_SIZE)
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
        hidden_channels = max(128, invariant_channels * 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(invariant_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, self.class_count),
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


def build_c8_tile_shape_classifier() -> nn.Module:
    return C8TileShapeClassifier(class_count=CLASS_COUNT, fields=C8_FIELDS)
