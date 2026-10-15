from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

GROUP_SIZE = 8
CLASS_COUNT = 2
SUPPORTED_FEATURES = ("cr", "normalized-red", "exr", "nrg")


def append_red_feature(rgb: torch.Tensor, *, feature_kind: str) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB NCHW tensor [N,3,H,W], got {tuple(rgb.shape)}")
    if feature_kind not in SUPPORTED_FEATURES:
        raise ValueError(f"unsupported red feature: {feature_kind}")
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    if feature_kind == "cr":
        feature = (0.5 * r - 0.418688 * g - 0.081312 * b + 0.5).clamp(0.0, 1.0)
    elif feature_kind == "normalized-red":
        denom = (r + g + b).clamp_min(1.0e-6)
        feature = r / denom
    elif feature_kind == "exr":
        denom = (r + g + b).clamp_min(1.0e-6)
        feature = 1.4 * (r / denom) - (g / denom)
    else:
        denom = (r + g).clamp_min(1.0e-6)
        feature = (r - g) / denom
    return torch.cat((rgb, feature), dim=1)


class C8RGBRedFeatureStage234DW222Classifier(nn.Module):
    def __init__(self, *, feature_kind: str, fields: Sequence[int], class_count: int = CLASS_COUNT) -> None:
        super().__init__()
        if feature_kind not in SUPPORTED_FEATURES:
            raise ValueError(f"unsupported red feature: {feature_kind}")
        field_counts = tuple(int(v) for v in fields)
        if len(field_counts) != 4 or any(v < 1 for v in field_counts):
            raise ValueError("fields must contain exactly four positive field counts")
        try:
            from escnn import gspaces
            from escnn import nn as enn
        except ImportError as error:
            raise RuntimeError("C8RGBRedFeatureStage234DW222Classifier requires escnn==1.0.11-compatible semantics") from error
        self.feature_kind = feature_kind
        self.group_size = GROUP_SIZE
        self.class_count = int(class_count)
        self.field_counts = field_counts
        self.tensor_channels = tuple(GROUP_SIZE * count for count in field_counts)
        self.stage_multipliers = {2: 2, 3: 2, 4: 2}
        self.register_buffer("input_mean", torch.zeros(4, dtype=torch.float32))
        self.register_buffer("input_std", torch.ones(4, dtype=torch.float32))
        self._enn = enn
        self.gspace = gspaces.rot2dOnR2(GROUP_SIZE)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 4)
        layers: list[nn.Module] = []
        in_type = self.input_type
        for block_index, field_count in enumerate(field_counts):
            stage = block_index + 1
            out_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * field_count)
            if stage == 1:
                layers.extend([enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False), enn.InnerBatchNorm(out_type), enn.ReLU(out_type, inplace=True)])
            else:
                depthwise_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * (len(in_type) * 2))
                layers.extend([
                    enn.R2Conv(in_type, depthwise_type, kernel_size=3, padding=1, groups=len(in_type), bias=False),
                    enn.InnerBatchNorm(depthwise_type), enn.ReLU(depthwise_type, inplace=True),
                    enn.R2Conv(depthwise_type, out_type, kernel_size=1, padding=0, bias=False),
                    enn.InnerBatchNorm(out_type), enn.ReLU(out_type, inplace=True),
                ])
            if block_index < 3:
                layers.append(enn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1))
            in_type = out_type
        self.equivariant_backbone = enn.SequentialModule(*layers)
        self.group_pool = enn.GroupPooling(in_type)
        invariant_channels = int(self.group_pool.out_type.size)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = max(128, invariant_channels * 2)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(invariant_channels, hidden_channels), nn.SiLU(inplace=True), nn.Linear(hidden_channels, self.class_count))

    def set_input_normalization(self, mean: Sequence[float], std: Sequence[float]) -> None:
        if len(mean) != 4 or len(std) != 4 or any(float(v) <= 0.0 for v in std):
            raise ValueError("input normalization requires four channels and positive std")
        self.input_mean.copy_(torch.tensor(tuple(mean), dtype=self.input_mean.dtype, device=self.input_mean.device))
        self.input_std.copy_(torch.tensor(tuple(std), dtype=self.input_std.dtype, device=self.input_std.device))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        features = append_red_feature(rgb, feature_kind=self.feature_kind)
        features = (features - self.input_mean.view(1, 4, 1, 1)) / self.input_std.view(1, 4, 1, 1)
        geometric = self._enn.GeometricTensor(features, self.input_type)
        encoded = self.equivariant_backbone(geometric)
        invariant = self.group_pool(encoded).tensor
        return self.classifier(self.spatial_pool(invariant))


def build_c8_rgb_redfeature_stage234_dw222_classifier(*, feature_kind: str, fields: Sequence[int]) -> nn.Module:
    return C8RGBRedFeatureStage234DW222Classifier(feature_kind=feature_kind, fields=fields, class_count=CLASS_COUNT)
