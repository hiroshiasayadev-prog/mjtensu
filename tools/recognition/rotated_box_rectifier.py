from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

if __package__:
    from .rotated_fcos_nano import canonicalize_obb, obb_corners, polygon_area, polygon_clip
else:
    from tools.recognition.rotated_fcos_nano import (  # type: ignore[no-redef]
        canonicalize_obb,
        obb_corners,
        polygon_area,
        polygon_clip,
    )


@dataclass(frozen=True)
class CropWindow:
    center_x: float
    center_y: float
    side: float

    @property
    def left(self) -> float:
        return self.center_x - self.side / 2.0

    @property
    def top(self) -> float:
        return self.center_y - self.side / 2.0

    @property
    def right(self) -> float:
        return self.center_x + self.side / 2.0

    @property
    def bottom(self) -> float:
        return self.center_y + self.side / 2.0


class RotatedBoxRectifier(nn.Module):
    """Tiny crop-level regressor that converts an HBB crop into a canonical OBB.

    Output channels are:
      dx, dy, log(width/crop_side), log(height/crop_side), sin(2theta), cos(2theta)
    where dx/dy are normalized by crop_side. The target OBB is canonicalized so width <= height.
    """

    def __init__(self, *, input_size: int = 96) -> None:
        super().__init__()
        if input_size < 32:
            raise ValueError("input_size must be at least 32")
        self.input_size = int(input_size)
        channels = (16, 32, 64, 96)
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 9, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(128, 6),
        )
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB NCHW input, got {tuple(images.shape)}")
        return self.head(self.pool(self.features(images)))

    def config(self) -> dict[str, int | str]:
        return {
            "name": "rotated_box_rectifier",
            "input_size": self.input_size,
            "output": "dx_dy_logw_logh_sin2theta_cos2theta",
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def hbb_from_obb(obb: Sequence[float]) -> tuple[float, float, float, float]:
    corners = obb_corners(obb)
    left = min(point[0] for point in corners)
    top = min(point[1] for point in corners)
    right = max(point[0] for point in corners)
    bottom = max(point[1] for point in corners)
    return (left, top, right - left, bottom - top)


def default_crop_window(obb: Sequence[float], *, context_scale: float = 1.40) -> CropWindow:
    if not math.isfinite(context_scale) or context_scale <= 1.0:
        raise ValueError("context_scale must be finite and > 1")
    left, top, width, height = hbb_from_obb(obb)
    side = max(width, height) * context_scale
    return CropWindow(
        center_x=left + width / 2.0,
        center_y=top + height / 2.0,
        side=side,
    )


def window_contains_obb(window: CropWindow, obb: Sequence[float], *, epsilon: float = 1.0e-5) -> bool:
    return all(
        window.left - epsilon <= x <= window.right + epsilon
        and window.top - epsilon <= y <= window.bottom + epsilon
        for x, y in obb_corners(obb)
    )


def encode_target(obb: Sequence[float], window: CropWindow) -> torch.Tensor:
    cx, cy, width, height, angle_deg = canonicalize_obb(obb)
    if window.side <= 0.0:
        raise ValueError("crop side must be positive")
    angle = math.radians(angle_deg)
    return torch.tensor(
        [
            (cx - window.center_x) / window.side,
            (cy - window.center_y) / window.side,
            math.log(max(width, 1.0e-6) / window.side),
            math.log(max(height, 1.0e-6) / window.side),
            math.sin(2.0 * angle),
            math.cos(2.0 * angle),
        ],
        dtype=torch.float32,
    )


def decode_prediction(prediction: Sequence[float], window: CropWindow) -> tuple[float, float, float, float, float]:
    if len(prediction) != 6:
        raise ValueError("prediction must contain six values")
    dx, dy, logw, logh, sin2theta, cos2theta = (float(value) for value in prediction)
    cx = window.center_x + dx * window.side
    cy = window.center_y + dy * window.side
    width = math.exp(max(-5.0, min(1.0, logw))) * window.side
    height = math.exp(max(-5.0, min(1.0, logh))) * window.side
    angle_deg = 0.5 * math.degrees(math.atan2(sin2theta, cos2theta))
    return canonicalize_obb((cx, cy, width, height, angle_deg))


def rotated_overlap_metrics(
    prediction: Sequence[float],
    target: Sequence[float],
) -> tuple[float, float, float]:
    predicted_polygon = obb_corners(prediction)
    target_polygon = obb_corners(target)
    intersection = polygon_area(polygon_clip(predicted_polygon, target_polygon))
    predicted_area = max(0.0, float(prediction[2])) * max(0.0, float(prediction[3]))
    target_area = max(0.0, float(target[2])) * max(0.0, float(target[3]))
    union = predicted_area + target_area - intersection
    iou = 0.0 if union <= 0.0 else intersection / union
    coverage = 0.0 if target_area <= 0.0 else intersection / target_area
    purity = 0.0 if predicted_area <= 0.0 else intersection / predicted_area
    return iou, coverage, purity


def angle_error_deg(first: float, second: float) -> float:
    first_canonical = canonicalize_obb((0.0, 0.0, 1.0, 2.0, first))[4]
    second_canonical = canonicalize_obb((0.0, 0.0, 1.0, 2.0, second))[4]
    delta = abs((first_canonical - second_canonical + 90.0) % 180.0 - 90.0)
    return delta
