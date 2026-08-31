from __future__ import annotations

"""Experiment-only rotation-robust classifier architectures.

These models are intentionally isolated from the production classifier module. C8 and
Plain reuse the accepted implementations. RIC-CNN and SConv are compact-backbone
operator comparisons; RotEqNet preserves its method-specific public vector-field
topology. All three research paths are kept exportable through the repository's
PyTorch 1.13 / ONNX opset-16 toolchain.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

try:
    from tile_shape_classifier import C8TileShapeClassifier, PlainTileShapeClassifier
except ModuleNotFoundError:  # package-style import used by tests
    from tools.recognition.tile_shape_classifier import (
        C8TileShapeClassifier,
        PlainTileShapeClassifier,
    )


DEFAULT_CLASS_COUNT = 35
DEFAULT_CHANNELS = (32, 64, 128, 192)
DEFAULT_ROTEQNET_CHANNELS = (6, 16, 32)
DEFAULT_ROTEQNET_ANGLES = 17


@dataclass(frozen=True)
class ExperimentModelDescription:
    name: str
    parameter_count: int
    trainable_parameter_count: int
    details: dict[str, object]


def _rotation_grids(kernel_size: int, angles_rad: Sequence[float]) -> torch.Tensor:
    grids: list[torch.Tensor] = []
    for angle in angles_rad:
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        theta = torch.tensor(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0)
        grid = F.affine_grid(
            theta,
            size=(1, 1, kernel_size, kernel_size),
            align_corners=True,
        )[0]
        grids.append(grid)
    return torch.stack(grids, dim=0)


def _rotate_filter_bank(weight: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    out_channels, in_channels, height, width = weight.shape
    flattened = weight.reshape(out_channels * in_channels, 1, height, width)
    expanded_grid = grid.unsqueeze(0).expand(flattened.shape[0], -1, -1, -1)
    rotated = F.grid_sample(
        flattened,
        expanded_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return rotated.reshape(out_channels, in_channels, height, width)


class RotEqConv2d(nn.Module):
    """RotEqNet-style rotating convolution with vector-field orientation pooling.

    Mode 1 accepts a scalar feature map.  Mode 2 accepts a `(u, v)` vector field and
    rotates both learned vector-filter components as in the public RotEqNet PyTorch
    remake.  The maximum response orientation becomes a vector-field angle while the
    rectified maximum response becomes its magnitude.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        n_angles: int = DEFAULT_ROTEQNET_ANGLES,
        mode: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("RotEqConv2d requires an odd kernel size")
        if n_angles < 2:
            raise ValueError("n_angles must be at least 2")
        if mode not in (1, 2):
            raise ValueError("mode must be 1 (scalar) or 2 (vector field)")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.n_angles = int(n_angles)
        self.mode = int(mode)
        self.padding = self.kernel_size // 2 if padding is None else int(padding)

        angles = [2.0 * math.pi * index / self.n_angles for index in range(self.n_angles)]
        self.register_buffer(
            "angles_rad",
            torch.tensor(angles, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "rotation_grids",
            _rotation_grids(self.kernel_size, angles),
            persistent=True,
        )

        self.weight_u = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        )
        if self.mode == 2:
            self.weight_v = nn.Parameter(
                torch.empty(
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
        else:
            self.register_parameter("weight_v", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match the public RotEqNet PyTorch implementation: U(-1/sqrt(fan_in), +...).
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = 1.0 / math.sqrt(float(fan_in))
        nn.init.uniform_(self.weight_u, -bound, bound)
        if self.weight_v is not None:
            nn.init.uniform_(self.weight_v, -bound, bound)

    def _response_for_angle(
        self,
        input_value: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        angle_index: int,
    ) -> torch.Tensor:
        grid = self.rotation_grids[angle_index]
        rotated_u = _rotate_filter_bank(self.weight_u, grid)
        if self.mode == 1:
            if not isinstance(input_value, torch.Tensor):
                raise TypeError("mode=1 RotEqConv2d expects a scalar tensor")
            return F.conv2d(input_value, rotated_u, padding=self.padding)

        if not isinstance(input_value, tuple) or len(input_value) != 2:
            raise TypeError("mode=2 RotEqConv2d expects a (u, v) vector field")
        if self.weight_v is None:
            raise RuntimeError("mode=2 RotEqConv2d has no v-component weights")
        input_u, input_v = input_value
        rotated_v = _rotate_filter_bank(self.weight_v, grid)
        angle = self.angles_rad[angle_index]
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        filter_for_u = cosine * rotated_u - sine * rotated_v
        filter_for_v = sine * rotated_u + cosine * rotated_v
        return F.conv2d(input_u, filter_for_u, padding=self.padding) + F.conv2d(
            input_v,
            filter_for_v,
            padding=self.padding,
        )

    def forward(
        self,
        input_value: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        best_strength: torch.Tensor | None = None
        best_index: torch.Tensor | None = None
        for angle_index in range(self.n_angles):
            response = self._response_for_angle(input_value, angle_index)
            if best_strength is None:
                best_strength = response
                best_index = torch.zeros_like(response, dtype=torch.int64)
                continue
            if best_index is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("RotEqNet orientation index state is missing")
            wins = response > best_strength
            best_strength = torch.maximum(best_strength, response)
            replacement = torch.full_like(best_index, angle_index)
            best_index = torch.where(wins, replacement, best_index)

        if best_strength is None or best_index is None:  # pragma: no cover
            raise RuntimeError("RotEqNet evaluated no orientations")
        angle_map = best_index.to(dtype=best_strength.dtype) * (
            2.0 * math.pi / float(self.n_angles)
        )
        magnitude = F.relu(best_strength)
        return magnitude * torch.cos(angle_map), magnitude * torch.sin(angle_map)


class VectorBatchNorm2d(nn.Module):
    """Magnitude-only batch normalization matching the public RotEqNet layer."""

    def __init__(self, channels: int, *, eps: float = 1.0e-5, momentum: float = 0.5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight = nn.Parameter(torch.empty(1, channels, 1, 1))
        self.register_buffer("running_var", torch.ones(1, channels, 1, 1))
        nn.init.uniform_(self.weight)

    def forward(
        self, input_value: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = input_value
        if self.training:
            variance = (u.square() + v.square()).mean(dim=(0, 2, 3), keepdim=True)
            std = torch.sqrt(variance)
            with torch.no_grad():
                self.running_var.mul_(1.0 - self.momentum).add_(
                    variance.detach() * self.momentum
                )
            scale = self.weight / (std + self.eps)
        else:
            scale = self.weight / torch.sqrt(self.running_var + self.eps)
        return u * scale, v * scale


class VectorMaxPool2d(nn.Module):
    """Pool the vector located at the maximum-magnitude spatial position."""

    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)

    def forward(
        self, input_value: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = input_value
        magnitude_sq = u.square() + v.square()
        _, indices = F.max_pool2d(
            magnitude_sq,
            kernel_size=self.kernel_size,
            stride=self.stride,
            return_indices=True,
        )
        batch, channels, pooled_h, pooled_w = indices.shape
        flat_indices = indices.reshape(batch, channels, pooled_h * pooled_w)
        flat_u = u.reshape(batch, channels, -1)
        flat_v = v.reshape(batch, channels, -1)
        pooled_u = torch.gather(flat_u, 2, flat_indices).reshape(
            batch, channels, pooled_h, pooled_w
        )
        pooled_v = torch.gather(flat_v, 2, flat_indices).reshape(
            batch, channels, pooled_h, pooled_w
        )
        return pooled_u, pooled_v


class RotEqNetTileClassifier(nn.Module):
    """64x64 adaptation of the public RotEqNet MNIST-rot topology.

    The method-specific feature extractor is kept intact: 9x9 RotConv 1->6, pool, VBN;
    9x9 RotConv 6->16, pool, VBN; 9x9 RotConv 16->32 with padding=1; magnitude;
    1x1 Conv 32->128, BN, ReLU, Dropout2d(0.7), 1x1 class Conv.  The author's 28x28
    input naturally collapses to 1x1 after the third RotConv.  Our fixed 64x64 classifier
    contract leaves a 10x10 class map, so only an AdaptiveAvgPool2d(1) is added before
    flattening to 35 logits.
    """

    def __init__(
        self,
        *,
        class_count: int = DEFAULT_CLASS_COUNT,
        channels: Sequence[int] = DEFAULT_ROTEQNET_CHANNELS,
        n_angles: int = DEFAULT_ROTEQNET_ANGLES,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        if len(widths) != 3:
            raise ValueError("RotEqNet experiment expects the public 3-stage channel schedule")
        self.rot1 = RotEqConv2d(
            1, widths[0], 9, n_angles=n_angles, mode=1, padding=4
        )
        self.pool1 = VectorMaxPool2d(2, 2)
        self.norm1 = VectorBatchNorm2d(widths[0])
        self.rot2 = RotEqConv2d(
            widths[0], widths[1], 9, n_angles=n_angles, mode=2, padding=4
        )
        self.pool2 = VectorMaxPool2d(2, 2)
        self.norm2 = VectorBatchNorm2d(widths[1])
        self.rot3 = RotEqConv2d(
            widths[1], widths[2], 9, n_angles=n_angles, mode=2, padding=1
        )
        self.head = nn.Sequential(
            nn.Conv2d(widths[2], 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.7),
            nn.Conv2d(128, class_count, kernel_size=1),
        )
        self.output_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = widths
        self.n_angles = int(n_angles)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        vector = self.rot1(images)
        vector = self.pool1(vector)
        vector = self.norm1(vector)
        vector = self.rot2(vector)
        vector = self.pool2(vector)
        vector = self.norm2(vector)
        vector = self.rot3(vector)
        u, v = vector
        magnitude = torch.sqrt(u.square() + v.square())
        logits_map = self.head(magnitude)
        return self.output_pool(logits_map).flatten(1)


def _normalized_grid(x: torch.Tensor, y: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if width > 1:
        x_normalized = x * (2.0 / float(width - 1)) - 1.0
    else:  # pragma: no cover - spatial sizes in this experiment are >1
        x_normalized = torch.zeros_like(x)
    if height > 1:
        y_normalized = y * (2.0 / float(height - 1)) - 1.0
    else:  # pragma: no cover
        y_normalized = torch.zeros_like(y)
    return torch.stack((x_normalized, y_normalized), dim=-1)


def build_ric_sampling_grid(height: int, width: int) -> torch.Tensor:
    """Build the 3x3 RIC-C neighborhood for each feature-map location.

    The eight non-center samples lie on a unit-radius ring whose zero direction is the
    radial direction from the feature-map center.  This is the tensor equivalent of the
    fixed deformable-convolution offsets used by the author implementation.
    """

    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    center_x = (float(width) - 1.0) / 2.0
    center_y = (float(height) - 1.0) / 2.0
    # Match the author implementation literally. Its `grid_x` is the row axis and
    # `grid_y` is the column axis, so theta=atan2(delta_col, delta_row), then each
    # sampled row/column displacement is (cos(theta+k*pi/4), sin(...)).
    theta = torch.remainder(
        torch.atan2(x - center_x, y - center_y), 2.0 * math.pi
    )
    theta = torch.round(theta * 10000.0) / 10000.0
    samples: list[torch.Tensor] = []
    for index in range(8):
        angle = theta + index * (math.pi / 4.0)
        sample_y = y + torch.cos(angle)
        sample_x = x + torch.sin(angle)
        samples.append(_normalized_grid(sample_x, sample_y, height, width))
    center_grid = _normalized_grid(x, y, height, width)
    ordered = samples[:4] + [center_grid] + samples[4:]
    return torch.stack(ordered, dim=0)


class RICConv2d(nn.Module):
    """RIC-C 3x3 convolution expressed with fixed GridSample neighborhoods."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        height: int,
        width: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.height = int(height)
        self.width = int(width)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.register_buffer(
            "sampling_grid",
            build_ric_sampling_grid(self.height, self.width),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        if self.bias is not None:
            fan_in = self.in_channels * 9
            bound = 1.0 / math.sqrt(float(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        sampled: list[torch.Tensor] = []
        for sample_index in range(9):
            grid = self.sampling_grid[sample_index].unsqueeze(0).expand(
                input_tensor.shape[0], -1, -1, -1
            )
            sampled.append(
                F.grid_sample(
                    input_tensor,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
            )
        stacked = torch.stack(sampled, dim=2)
        batch, channels, points, height, width = stacked.shape
        packed = stacked.reshape(batch, channels * points, height, width)
        packed_weight = self.weight.reshape(self.out_channels, self.in_channels * 9, 1, 1)
        return F.conv2d(packed, packed_weight, bias=self.bias)


class RICCNNTileClassifier(nn.Module):
    def __init__(
        self,
        *,
        class_count: int = DEFAULT_CLASS_COUNT,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        if len(widths) != 4:
            raise ValueError("RIC-CNN experiment expects four channel stages")
        resolutions = (image_size, image_size // 2, image_size // 4, image_size // 8)
        layers: list[nn.Module] = []
        in_channels = 1
        for index, (out_channels, resolution) in enumerate(zip(widths, resolutions)):
            layers.extend(
                [
                    RICConv2d(
                        in_channels,
                        out_channels,
                        height=resolution,
                        width=resolution,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            if index < len(widths) - 1:
                layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1], 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, class_count),
        )
        self.channels = widths

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)))


def build_square_ring_row_major_gather(kernel_size: int) -> torch.Tensor:
    """Map [center, sorted ring1, sorted ring2, ...] into square-kernel row-major order.

    The SConv paper sorts each Chebyshev-distance ring independently and then places
    the sorted values back into the corresponding square-grid ring in row-major order.
    Keeping this mapping explicit prevents the center/ring values from being paired with
    the wrong learned kernel weights.
    """
    if kernel_size % 2 != 1 or kernel_size < 3:
        raise ValueError("SConv kernel_size must be an odd value >= 3")
    center = kernel_size // 2
    gather = [-1] * (kernel_size * kernel_size)
    gather[center * kernel_size + center] = 0
    source_index = 1
    for radius in range(1, center + 1):
        ring_positions: list[int] = []
        for row in range(kernel_size):
            for col in range(kernel_size):
                if max(abs(row - center), abs(col - center)) == radius:
                    ring_positions.append(row * kernel_size + col)
        expected = 8 * radius
        if len(ring_positions) != expected:
            raise RuntimeError(
                f"SConv square ring radius={radius} has {len(ring_positions)} positions, expected {expected}"
            )
        for offset, flat_position in enumerate(ring_positions):
            gather[flat_position] = source_index + offset
        source_index += expected
    if any(value < 0 for value in gather):  # pragma: no cover - construction invariant
        raise RuntimeError("SConv row-major ring mapping is incomplete")
    return torch.tensor(gather, dtype=torch.int64)


def build_polar_ring_grids(height: int, width: int, kernel_size: int) -> list[torch.Tensor]:
    if kernel_size % 2 != 1 or kernel_size < 3:
        raise ValueError("SConv kernel_size must be an odd value >= 3")
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    rings: list[torch.Tensor] = []
    max_radius = kernel_size // 2
    for radius in range(1, max_radius + 1):
        point_count = 8 * radius
        ring_samples: list[torch.Tensor] = []
        for index in range(point_count):
            angle = 2.0 * math.pi * index / point_count
            sample_x = x + float(radius) * math.cos(angle)
            sample_y = y + float(radius) * math.sin(angle)
            ring_samples.append(
                _normalized_grid(sample_x, sample_y, height, width)
            )
        rings.append(torch.stack(ring_samples, dim=0))
    return rings


class SConv2d(nn.Module):
    """Sorting Convolution using polar rings, bilinear sampling and ring-wise TopK sort."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        height: int,
        width: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.height = int(height)
        self.width = int(width)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        ring_grids = build_polar_ring_grids(height, width, kernel_size)
        self.ring_point_counts: tuple[int, ...] = tuple(grid.shape[0] for grid in ring_grids)
        for index, grid in enumerate(ring_grids):
            self.register_buffer(f"ring_grid_{index}", grid, persistent=True)
        self.register_buffer(
            "row_major_gather_index",
            build_square_ring_row_major_gather(self.kernel_size),
            persistent=True,
        )
        expected_points = 1 + sum(self.ring_point_counts)
        if expected_points != kernel_size * kernel_size:
            raise ValueError(
                f"Polar SConv point count {expected_points} != kernel area {kernel_size**2}"
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1.0 / math.sqrt(float(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        ordered_samples: list[torch.Tensor] = [input_tensor.unsqueeze(2)]
        for ring_index, point_count in enumerate(self.ring_point_counts):
            grid_bank = getattr(self, f"ring_grid_{ring_index}")
            sampled_ring: list[torch.Tensor] = []
            for sample_index in range(point_count):
                grid = grid_bank[sample_index].unsqueeze(0).expand(
                    input_tensor.shape[0], -1, -1, -1
                )
                sampled_ring.append(
                    F.grid_sample(
                        input_tensor,
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )
                )
            ring = torch.stack(sampled_ring, dim=2)
            sorted_ring = torch.topk(
                ring,
                k=point_count,
                dim=2,
                largest=False,
                sorted=True,
            ).values
            ordered_samples.append(sorted_ring)

        samples_by_ring = torch.cat(ordered_samples, dim=2)
        samples = torch.index_select(
            samples_by_ring, 2, self.row_major_gather_index
        )
        batch, channels, points, height, width = samples.shape
        packed = samples.reshape(batch, channels * points, height, width)
        packed_weight = self.weight.reshape(
            self.out_channels,
            self.in_channels * self.kernel_size * self.kernel_size,
            1,
            1,
        )
        return F.conv2d(packed, packed_weight, bias=self.bias)


class SConvTileClassifier(nn.Module):
    def __init__(
        self,
        *,
        class_count: int = DEFAULT_CLASS_COUNT,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        if len(widths) != 4:
            raise ValueError("SConv experiment expects four channel stages")
        kernels = (5, 3, 3, 3)
        resolutions = (image_size, image_size // 2, image_size // 4, image_size // 8)
        layers: list[nn.Module] = []
        in_channels = 1
        for index, (out_channels, kernel_size, resolution) in enumerate(
            zip(widths, kernels, resolutions)
        ):
            layers.extend(
                [
                    SConv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        height=resolution,
                        width=resolution,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            if index < len(widths) - 1:
                layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1], 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, class_count),
        )
        self.channels = widths

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)))


def build_experiment_model(
    model_name: str,
    *,
    class_count: int = DEFAULT_CLASS_COUNT,
    image_size: int = 64,
) -> nn.Module:
    normalized = model_name.strip().lower()
    if normalized == "c8":
        return C8TileShapeClassifier(class_count=class_count)
    if normalized == "plain":
        return PlainTileShapeClassifier(class_count=class_count)
    if normalized == "roteqnet":
        return RotEqNetTileClassifier(class_count=class_count)
    if normalized == "riccnn":
        return RICCNNTileClassifier(class_count=class_count, image_size=image_size)
    if normalized == "sconv":
        return SConvTileClassifier(class_count=class_count, image_size=image_size)
    raise ValueError(
        f"Unsupported experiment model {model_name!r}; expected c8/plain/roteqnet/riccnn/sconv"
    )


def describe_experiment_model(model: nn.Module, model_name: str) -> ExperimentModelDescription:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    normalized = model_name.strip().lower()
    details: dict[str, object] = {}
    if normalized == "c8":
        details = {"c8_fields": list(getattr(model, "field_counts", (8, 16, 32, 64)))}
    elif normalized == "plain":
        details = {"channels": list(DEFAULT_CHANNELS), "kernels": [5, 3, 3, 3]}
    elif normalized == "roteqnet":
        details = {
            "channels": list(getattr(model, "channels", DEFAULT_ROTEQNET_CHANNELS)),
            "kernels": [9, 9, 9],
            "paddings": [4, 4, 1],
            "orientations": int(getattr(model, "n_angles", DEFAULT_ROTEQNET_ANGLES)),
            "representation": "public RotEqNet vector-field topology; adaptive class-map pooling added for 64x64 input",
        }
    elif normalized == "riccnn":
        details = {
            "channels": list(getattr(model, "channels", DEFAULT_CHANNELS)),
            "kernels": [3, 3, 3, 3],
            "sampling": "center-radial fixed 8-neighbor ring + center via GridSample",
        }
    elif normalized == "sconv":
        details = {
            "channels": list(getattr(model, "channels", DEFAULT_CHANNELS)),
            "kernels": [5, 3, 3, 3],
            "sampling": "polar rings with 8*r bilinear samples and ring-wise TopK sorting",
        }
    return ExperimentModelDescription(
        name=normalized,
        parameter_count=int(parameters),
        trainable_parameter_count=int(trainable),
        details=details,
    )
