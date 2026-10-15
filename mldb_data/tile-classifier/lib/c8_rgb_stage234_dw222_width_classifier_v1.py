from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

GROUP_SIZE = 8
CLASS_COUNT = 2


class C8RGBStage234DW222WidthClassifier(nn.Module):
    def __init__(self, *, fields: Sequence[int], class_count: int = CLASS_COUNT) -> None:
        super().__init__()
        field_counts = tuple(int(v) for v in fields)
        if len(field_counts) != 4 or any(v < 1 for v in field_counts):
            raise ValueError("fields must contain exactly four positive field counts")
        try:
            from escnn import gspaces
            from escnn import nn as enn
        except ImportError as error:
            raise RuntimeError("C8RGBStage234DW222WidthClassifier requires escnn==1.0.11-compatible semantics") from error
        self.group_size = GROUP_SIZE
        self.class_count = int(class_count)
        self.field_counts = field_counts
        self.tensor_channels = tuple(GROUP_SIZE * count for count in field_counts)
        self.stage_multipliers = {2: 2, 3: 2, 4: 2}
        self._enn = enn
        self.gspace = gspaces.rot2dOnR2(GROUP_SIZE)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        layers: list[nn.Module] = []
        in_type = self.input_type
        for block_index, field_count in enumerate(field_counts):
            stage = block_index + 1
            out_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * field_count)
            if stage == 1:
                layers.extend([
                    enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
                    enn.InnerBatchNorm(out_type),
                    enn.ReLU(out_type, inplace=True),
                ])
            else:
                depthwise_type = enn.FieldType(
                    self.gspace, [self.gspace.regular_repr] * (len(in_type) * 2)
                )
                layers.extend([
                    enn.R2Conv(in_type, depthwise_type, kernel_size=3, padding=1, groups=len(in_type), bias=False),
                    enn.InnerBatchNorm(depthwise_type),
                    enn.ReLU(depthwise_type, inplace=True),
                    enn.R2Conv(depthwise_type, out_type, kernel_size=1, padding=0, bias=False),
                    enn.InnerBatchNorm(out_type),
                    enn.ReLU(out_type, inplace=True),
                ])
            if block_index < 3:
                layers.append(enn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1))
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
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB NCHW tensor [N,3,H,W], got {tuple(images.shape)}")
        geometric = self._enn.GeometricTensor(images, self.input_type)
        features = self.equivariant_backbone(geometric)
        invariant = self.group_pool(features).tensor
        return self.classifier(self.spatial_pool(invariant))


def build_c8_rgb_stage234_dw222_width_classifier(*, fields: Sequence[int]) -> nn.Module:
    return C8RGBStage234DW222WidthClassifier(fields=fields, class_count=CLASS_COUNT)
