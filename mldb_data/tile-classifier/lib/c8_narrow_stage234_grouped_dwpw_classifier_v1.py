from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


CLASS_COUNT = 35
GROUP_SIZE = 8
FIELD_COUNTS = (4, 8, 16, 32)
STAGES = (2, 3, 4)


class C8NarrowStage234GroupedDWPWClassifier(nn.Module):
    """C8 narrow classifier with independently grouped DW3x3 and PW1x1 stages."""

    def __init__(
        self,
        *,
        stage_multipliers: Mapping[int, int],
        dw_fields_per_group: Mapping[int, int],
        pw_groups: Mapping[int, int],
        class_count: int = CLASS_COUNT,
    ) -> None:
        super().__init__()
        try:
            from escnn import gspaces
            from escnn import nn as enn
        except ImportError as error:
            raise RuntimeError(
                "C8NarrowStage234GroupedDWPWClassifier requires escnn==1.0.11-compatible semantics."
            ) from error

        multipliers = {int(stage): int(value) for stage, value in stage_multipliers.items()}
        dw_fpg = {int(stage): int(value) for stage, value in dw_fields_per_group.items()}
        pw_group_counts = {int(stage): int(value) for stage, value in pw_groups.items()}
        for name, values in (
            ("stage_multipliers", multipliers),
            ("dw_fields_per_group", dw_fpg),
            ("pw_groups", pw_group_counts),
        ):
            if set(values) != set(STAGES):
                raise ValueError(f"{name} must define exactly stages 2, 3, and 4")
            if any(value < 1 for value in values.values()):
                raise ValueError(f"{name} values must be positive integers")

        self.group_size = GROUP_SIZE
        self.class_count = int(class_count)
        self.field_counts = FIELD_COUNTS
        self.tensor_channels = tuple(GROUP_SIZE * count for count in FIELD_COUNTS)
        self.stage_multipliers = multipliers
        self.dw_fields_per_group = dw_fpg
        self.pw_groups = pw_group_counts
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

            if stage_number in STAGES:
                multiplier = multipliers[stage_number]
                fields_per_group = dw_fpg[stage_number]
                if len(in_type) % fields_per_group != 0:
                    raise ValueError(
                        f"stage {stage_number} input fields {len(in_type)} are not divisible "
                        f"by DW fields/group {fields_per_group}"
                    )
                dw_groups = len(in_type) // fields_per_group
                depthwise_fields = len(in_type) * multiplier
                if depthwise_fields % dw_groups != 0:
                    raise ValueError(
                        f"stage {stage_number} DW output fields {depthwise_fields} are not "
                        f"divisible by DW groups {dw_groups}"
                    )
                depthwise_type = enn.FieldType(
                    self.gspace,
                    [self.gspace.regular_repr] * depthwise_fields,
                )

                pointwise_groups = pw_group_counts[stage_number]
                if depthwise_fields % pointwise_groups != 0:
                    raise ValueError(
                        f"stage {stage_number} DW fields {depthwise_fields} are not divisible "
                        f"by PW groups {pointwise_groups}"
                    )
                if len(out_type) % pointwise_groups != 0:
                    raise ValueError(
                        f"stage {stage_number} output fields {len(out_type)} are not divisible "
                        f"by PW groups {pointwise_groups}"
                    )

                layers.extend(
                    [
                        enn.R2Conv(
                            in_type,
                            depthwise_type,
                            kernel_size=3,
                            padding=1,
                            groups=dw_groups,
                            bias=False,
                        ),
                        enn.InnerBatchNorm(depthwise_type),
                        enn.ReLU(depthwise_type, inplace=True),
                        enn.R2Conv(
                            depthwise_type,
                            out_type,
                            kernel_size=1,
                            padding=0,
                            groups=pointwise_groups,
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


def _stage_map(values: Sequence[int], *, name: str) -> dict[int, int]:
    normalized = tuple(int(value) for value in values)
    if len(normalized) != len(STAGES):
        raise ValueError(f"{name} must contain exactly three values for stages 2/3/4")
    return dict(zip(STAGES, normalized, strict=True))


def build_c8_narrow_stage234_grouped_dwpw_classifier(
    *,
    dw_fields_per_group: Sequence[int],
    pw_groups: Sequence[int],
) -> nn.Module:
    return C8NarrowStage234GroupedDWPWClassifier(
        stage_multipliers={2: 2, 3: 2, 4: 2},
        dw_fields_per_group=_stage_map(dw_fields_per_group, name="dw_fields_per_group"),
        pw_groups=_stage_map(pw_groups, name="pw_groups"),
        class_count=CLASS_COUNT,
    )
