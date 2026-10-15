from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

GROUP_SIZE = 8
CLASS_COUNT = 2


class C8RGBAggressivePartialDWClassifier(nn.Module):
    def __init__(
        self,
        *,
        fields: Sequence[int],
        dw_stages: Sequence[int],
        dw_multiplier: int,
        class_count: int = CLASS_COUNT,
    ) -> None:
        super().__init__()
        field_counts = tuple(int(v) for v in fields)
        selected_dw_stages = tuple(sorted(int(v) for v in dw_stages))
        multiplier = int(dw_multiplier)
        if len(field_counts) != 4 or any(v < 1 for v in field_counts):
            raise ValueError("fields must contain exactly four positive field counts")
        if any(stage not in (2, 3, 4) for stage in selected_dw_stages):
            raise ValueError("dw_stages may contain only stages 2, 3, or 4")
        if multiplier not in (1, 2):
            raise ValueError("dw_multiplier must be 1 or 2")
        try:
            from escnn import gspaces
            from escnn import nn as enn
        except ImportError as error:
            raise RuntimeError("C8RGBAggressivePartialDWClassifier requires escnn==1.0.11-compatible semantics") from error

        self.group_size = GROUP_SIZE
        self.class_count = int(class_count)
        self.field_counts = field_counts
        self.stage_strides = (1, 2, 2, 2)
        self.dw_stages = selected_dw_stages
        self.dw_multiplier = multiplier
        self.tensor_channels = tuple(GROUP_SIZE * count for count in field_counts)
        self._enn = enn
        self.gspace = gspaces.rot2dOnR2(GROUP_SIZE)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)

        layers: list[nn.Module] = []
        in_type = self.input_type
        for stage, (field_count, stride) in enumerate(zip(field_counts, self.stage_strides, strict=True), start=1):
            out_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * field_count)
            if stage in selected_dw_stages:
                depthwise_type = enn.FieldType(
                    self.gspace,
                    [self.gspace.regular_repr] * (len(in_type) * multiplier),
                )
                layers.extend(
                    [
                        enn.R2Conv(
                            in_type,
                            depthwise_type,
                            kernel_size=3,
                            stride=stride,
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
                kernel_size = 5 if stage == 1 else 3
                layers.extend(
                    [
                        enn.R2Conv(
                            in_type,
                            out_type,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=kernel_size // 2,
                            bias=False,
                        ),
                        enn.InnerBatchNorm(out_type),
                        enn.ReLU(out_type, inplace=True),
                    ]
                )
            if stage == 1:
                layers.append(enn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1))
            in_type = out_type

        self.equivariant_backbone = enn.SequentialModule(*layers)
        self.group_pool = enn.GroupPooling(in_type)
        invariant_channels = int(self.group_pool.out_type.size)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = max(64, invariant_channels * 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(invariant_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, self.class_count),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB NCHW tensor [N,3,H,W], got {tuple(images.shape)}")
        x = self._enn.GeometricTensor(images, self.input_type)
        x = self.equivariant_backbone(x)
        x = self.group_pool(x).tensor
        return self.classifier(self.spatial_pool(x))


def build_c8_rgb_aggressive_partial_dw_classifier(
    *, fields: Sequence[int], dw_stages: Sequence[int], dw_multiplier: int
) -> nn.Module:
    return C8RGBAggressivePartialDWClassifier(
        fields=fields,
        dw_stages=dw_stages,
        dw_multiplier=dw_multiplier,
        class_count=CLASS_COUNT,
    )
