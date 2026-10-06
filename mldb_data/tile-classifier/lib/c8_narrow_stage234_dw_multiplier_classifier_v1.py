from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


CLASS_COUNT = 35
GROUP_SIZE = 8
FIELD_COUNTS = (4, 8, 16, 32)


class C8NarrowStage234DepthwiseMultiplierClassifier(nn.Module):
    """C8 narrow classifier with field-wise DW3x3 multipliers on stages 2-4."""

    def __init__(
        self,
        *,
        stage_multipliers: Mapping[int, int],
        class_count: int = CLASS_COUNT,
    ) -> None:
        super().__init__()
        try:
            from escnn import gspaces
            from escnn import nn as enn
        except ImportError as error:
            raise RuntimeError(
                "C8NarrowStage234DepthwiseMultiplierClassifier requires escnn==1.0.11-compatible semantics."
            ) from error

        normalized = {int(stage): int(multiplier) for stage, multiplier in stage_multipliers.items()}
        if set(normalized) != {2, 3, 4}:
            raise ValueError("stage_multipliers must define exactly stages 2, 3, and 4")
        if any(multiplier < 1 for multiplier in normalized.values()):
            raise ValueError("stage multipliers must be positive integers")

        self.group_size = GROUP_SIZE
        self.class_count = int(class_count)
        self.field_counts = FIELD_COUNTS
        self.tensor_channels = tuple(GROUP_SIZE * count for count in FIELD_COUNTS)
        self.stage_multipliers = normalized
        self._enn = enn
        self.gspace = gspaces.rot2dOnR2(GROUP_SIZE)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])

        layers: list[nn.Module] = []
        in_type = self.input_type
        for block_index, field_count in enumerate(FIELD_COUNTS):
            stage_number = block_index + 1
            out_type = enn.FieldType(
                self.gspace,
                [self.gspace.regular_repr] * field_count,
            )

            if stage_number in {2, 3, 4}:
                multiplier = normalized[stage_number]
                depthwise_fields = len(in_type) * multiplier
                depthwise_type = enn.FieldType(
                    self.gspace,
                    [self.gspace.regular_repr] * depthwise_fields,
                )
                layers.extend(
                    [
                        enn.R2Conv(
                            in_type,
                            depthwise_type,
                            kernel_size=3,
                            padding=1,
                            groups=len(in_type),
                            bias=False,
                        ),
                        enn.InnerBatchNorm(depthwise_type),
                        enn.ReLU(depthwise_type, inplace=True),
                        enn.R2Conv(
                            depthwise_type,
                            out_type,
                            kernel_size=1,
                            padding=0,
                            bias=False,
                        ),
                        enn.InnerBatchNorm(out_type),
                        enn.ReLU(out_type, inplace=True),
                    ]
                )
            else:
                layers.extend(
                    [
                        enn.R2Conv(
                            in_type,
                            out_type,
                            kernel_size=5,
                            padding=2,
                            bias=False,
                        ),
                        enn.InnerBatchNorm(out_type),
                        enn.ReLU(out_type, inplace=True),
                    ]
                )

            if block_index < len(FIELD_COUNTS) - 1:
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
        invariant_channels = int(self.group_pool.out_type.size)
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
                f"Expected grayscale NCHW tensor [N,1,H,W], got {tuple(images.shape)}"
            )
        geometric = self._enn.GeometricTensor(images, self.input_type)
        features = self.equivariant_backbone(geometric)
        invariant = self.group_pool(features).tensor
        pooled = self.spatial_pool(invariant)
        return self.classifier(pooled)


def build_c8_narrow_stage234_dw_multiplier_classifier(
    *,
    stage3_multiplier: int,
    stage4_multiplier: int,
) -> nn.Module:
    return C8NarrowStage234DepthwiseMultiplierClassifier(
        stage_multipliers={
            2: 2,
            3: stage3_multiplier,
            4: stage4_multiplier,
        },
        class_count=CLASS_COUNT,
    )
