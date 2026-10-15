from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

GROUP_SIZE = 8
CLASS_COUNT = 2


class C8RGBFullStrideClassifier(nn.Module):
    def __init__(
        self,
        *,
        fields: Sequence[int],
        stage_strides: Sequence[int],
        class_count: int = CLASS_COUNT,
    ) -> None:
        super().__init__()
        field_counts = tuple(int(v) for v in fields)
        strides = tuple(int(v) for v in stage_strides)
        if len(field_counts) != 4 or any(v < 1 for v in field_counts):
            raise ValueError("fields must contain exactly four positive field counts")
        if len(strides) != 4 or any(v not in (1, 2) for v in strides):
            raise ValueError("stage_strides must contain exactly four values in {1,2}")
        if strides[0] != 1:
            raise ValueError("stage1 stride must remain 1; stage1 max-pool owns the first downsample")
        try:
            from escnn import gspaces
            from escnn import nn as enn
        except ImportError as error:
            raise RuntimeError("C8RGBFullStrideClassifier requires escnn==1.0.11-compatible semantics") from error

        self.group_size = GROUP_SIZE
        self.class_count = int(class_count)
        self.field_counts = field_counts
        self.stage_strides = strides
        self.tensor_channels = tuple(GROUP_SIZE * count for count in field_counts)
        self._enn = enn
        self.gspace = gspaces.rot2dOnR2(GROUP_SIZE)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)

        layers: list[nn.Module] = []
        in_type = self.input_type
        for block_index, (field_count, stride) in enumerate(zip(field_counts, strides, strict=True)):
            out_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * field_count)
            kernel_size = 5 if block_index == 0 else 3
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
            if block_index == 0:
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


def build_c8_rgb_full_stride_classifier(
    *, fields: Sequence[int], stage_strides: Sequence[int]
) -> nn.Module:
    return C8RGBFullStrideClassifier(
        fields=fields,
        stage_strides=stage_strides,
        class_count=CLASS_COUNT,
    )
