from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights,
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
)


STRIDES = (8, 16, 32)
DEFAULT_SIZE_RANGES = ((0.0, 64.0), (48.0, 128.0), (96.0, 1.0e8))


@dataclass(frozen=True)
class Detection:
    score: float
    cx: float
    cy: float
    width: float
    height: float
    angle_deg: float

    @property
    def obb(self) -> tuple[float, float, float, float, float]:
        return (self.cx, self.cy, self.width, self.height, self.angle_deg)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


def make_normalization(channels: int, normalization: str) -> nn.Module:
    if normalization == "batch":
        return nn.BatchNorm2d(channels)
    if normalization == "group":
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported normalization: {normalization}")


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, channels: int, *, normalization: str = "batch") -> None:
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            make_normalization(channels, normalization),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            make_normalization(channels, normalization),
            nn.SiLU(inplace=True),
        )


class ShuffleNetBackbone(nn.Module):
    def __init__(self, variant: str, *, pretrained: bool) -> None:
        super().__init__()
        if variant == "shufflenet_v2_x0_5":
            weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
            model = shufflenet_v2_x0_5(weights=weights)
        elif variant == "shufflenet_v2_x1_0":
            weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
            model = shufflenet_v2_x1_0(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {variant}")
        self.conv1 = model.conv1
        self.maxpool = model.maxpool
        self.stage2 = model.stage2
        self.stage3 = model.stage3
        self.stage4 = model.stage4
        self.out_channels = self._infer_channels()

    def _infer_channels(self) -> tuple[int, int, int]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            sample = torch.zeros(1, 3, 320, 320)
            c3, c4, c5 = self.forward(sample)
        self.train(was_training)
        return (int(c3.shape[1]), int(c4.shape[1]), int(c5.shape[1]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.maxpool(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5


class TinyFPN(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("TinyFPN requires exactly three backbone feature levels")
        self.lateral3 = ConvBNAct(int(in_channels[0]), out_channels)
        self.lateral4 = ConvBNAct(int(in_channels[1]), out_channels)
        self.lateral5 = ConvBNAct(int(in_channels[2]), out_channels)
        self.output3 = DepthwiseSeparable(out_channels)
        self.output4 = DepthwiseSeparable(out_channels)
        self.output5 = DepthwiseSeparable(out_channels)

    def forward(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return self.output3(p3), self.output4(p4), self.output5(p5)


class RotatedFCOSHead(nn.Module):
    def __init__(self, channels: int, *, stacked_convs: int, normalization: str) -> None:
        super().__init__()
        if stacked_convs < 1:
            raise ValueError("stacked_convs must be positive")
        self.tower = nn.Sequential(
            *(DepthwiseSeparable(channels, normalization=normalization) for _ in range(stacked_convs))
        )
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.centerness = nn.Conv2d(channels, 1, 1)
        # dx, dy, log(w/stride), log(h/stride), sin(2theta), cos(2theta)
        self.regression = nn.Conv2d(channels, 6, 1)
        nn.init.constant_(self.objectness.bias, -4.59511985013459)
        nn.init.constant_(self.centerness.bias, 0.0)
        nn.init.zeros_(self.regression.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        hidden = self.tower(feature)
        return torch.cat(
            (self.objectness(hidden), self.centerness(hidden), self.regression(hidden)),
            dim=1,
        )


class RotatedFCOSNano(nn.Module):
    def __init__(
        self,
        *,
        backbone: str = "shufflenet_v2_x0_5",
        fpn_channels: int = 64,
        head_convs: int = 2,
        head_normalization: str = "batch",
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.fpn_channels = int(fpn_channels)
        self.head_convs = int(head_convs)
        self.head_normalization = str(head_normalization)
        self.backbone = ShuffleNetBackbone(backbone, pretrained=pretrained_backbone)
        self.fpn = TinyFPN(self.backbone.out_channels, self.fpn_channels)
        self.head = RotatedFCOSHead(
            self.fpn_channels,
            stacked_convs=self.head_convs,
            normalization=self.head_normalization,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.fpn(self.backbone(x))
        return tuple(self.head(feature) for feature in features)  # type: ignore[return-value]

    def config(self) -> dict[str, Any]:
        return {
            "name": "rotated_fcos_nano",
            "backbone": self.backbone_name,
            "fpn_channels": self.fpn_channels,
            "head_convs": self.head_convs,
            "head_normalization": self.head_normalization,
            "strides": list(STRIDES),
            "regression": ["dx", "dy", "logw", "logh", "sin2theta", "cos2theta"],
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_points(
    outputs: Sequence[torch.Tensor],
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(outputs) != len(STRIDES):
        raise ValueError(f"Expected {len(STRIDES)} output levels, found {len(outputs)}")
    point_chunks: list[torch.Tensor] = []
    stride_chunks: list[torch.Tensor] = []
    level_chunks: list[torch.Tensor] = []
    for level, (output, stride) in enumerate(zip(outputs, STRIDES, strict=True)):
        height, width = output.shape[-2:]
        target_device = output.device if device is None else device
        ys = (torch.arange(height, device=target_device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(width, device=target_device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        point_chunks.append(points)
        stride_chunks.append(
            torch.full((points.shape[0],), float(stride), device=target_device)
        )
        level_chunks.append(
            torch.full((points.shape[0],), level, device=target_device, dtype=torch.long)
        )
    return (
        torch.cat(point_chunks, dim=0),
        torch.cat(stride_chunks, dim=0),
        torch.cat(level_chunks, dim=0),
    )


def flatten_outputs(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    chunks = [output.permute(0, 2, 3, 1).reshape(output.shape[0], -1, 8) for output in outputs]
    return torch.cat(chunks, dim=1)


def build_targets(
    boxes: torch.Tensor,
    points: torch.Tensor,
    strides: torch.Tensor,
    levels: torch.Tensor,
    *,
    center_radius: float = 1.5,
    size_ranges: Sequence[tuple[float, float]] = DEFAULT_SIZE_RANGES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    point_count = points.shape[0]
    device = points.device
    objectness = torch.zeros(point_count, device=device)
    centerness = torch.zeros(point_count, device=device)
    regression = torch.zeros(point_count, 6, device=device)
    if boxes.numel() == 0:
        return objectness, centerness, regression
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError(f"Expected [N,5] OBB tensor, found {tuple(boxes.shape)}")

    boxes = boxes.to(device=device, dtype=torch.float32)
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    widths = boxes[:, 2].clamp_min(1.0e-4)
    heights = boxes[:, 3].clamp_min(1.0e-4)
    angles = torch.deg2rad(boxes[:, 4])
    cosine = torch.cos(angles)
    sine = torch.sin(angles)

    dx = points[:, 0, None] - cx[None, :]
    dy = points[:, 1, None] - cy[None, :]
    local_x = dx * cosine[None, :] + dy * sine[None, :]
    local_y = -dx * sine[None, :] + dy * cosine[None, :]
    half_width = widths[None, :] / 2.0
    half_height = heights[None, :] / 2.0
    inside = (local_x.abs() <= half_width) & (local_y.abs() <= half_height)

    radius = center_radius * strides[:, None]
    center_inside = (
        (local_x.abs() <= torch.minimum(half_width, radius))
        & (local_y.abs() <= torch.minimum(half_height, radius))
    )
    max_size = torch.maximum(widths, heights)
    ranges = torch.tensor(size_ranges, device=device, dtype=torch.float32)
    range_low = ranges[levels, 0][:, None]
    range_high = ranges[levels, 1][:, None]
    level_match = (max_size[None, :] >= range_low) & (max_size[None, :] <= range_high)
    candidates = inside & center_inside & level_match

    areas = (widths * heights)[None, :].expand(point_count, -1).clone()
    areas[~candidates] = float("inf")
    assigned_area, assigned_index = areas.min(dim=1)
    positive = torch.isfinite(assigned_area)
    if not positive.any():
        return objectness, centerness, regression

    positive_indices = positive.nonzero(as_tuple=False).squeeze(1)
    gt_indices = assigned_index[positive]
    objectness[positive] = 1.0

    selected_local_x = local_x[positive_indices, gt_indices]
    selected_local_y = local_y[positive_indices, gt_indices]
    selected_width = widths[gt_indices]
    selected_height = heights[gt_indices]
    selected_stride = strides[positive]
    left = selected_width / 2.0 + selected_local_x
    right = selected_width / 2.0 - selected_local_x
    top = selected_height / 2.0 + selected_local_y
    bottom = selected_height / 2.0 - selected_local_y
    lr = torch.minimum(left, right) / torch.maximum(left, right).clamp_min(1.0e-6)
    tb = torch.minimum(top, bottom) / torch.maximum(top, bottom).clamp_min(1.0e-6)
    centerness[positive] = torch.sqrt((lr * tb).clamp_min(0.0))

    selected_cx = cx[gt_indices]
    selected_cy = cy[gt_indices]
    selected_angle = angles[gt_indices]
    regression[positive, 0] = (selected_cx - points[positive, 0]) / selected_stride
    regression[positive, 1] = (selected_cy - points[positive, 1]) / selected_stride
    regression[positive, 2] = torch.log(selected_width / selected_stride)
    regression[positive, 3] = torch.log(selected_height / selected_stride)
    regression[positive, 4] = torch.sin(2.0 * selected_angle)
    regression[positive, 5] = torch.cos(2.0 * selected_angle)
    return objectness, centerness, regression


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    binary_cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * (1.0 - p_t).pow(gamma) * binary_cross_entropy


def compute_loss(
    outputs: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    *,
    center_radius: float = 1.5,
) -> dict[str, torch.Tensor]:
    predictions = flatten_outputs(outputs)
    points, strides, levels = make_points(outputs)
    batch_size, point_count, _channels = predictions.shape
    if len(targets) != batch_size:
        raise ValueError(f"Expected {batch_size} target tensors, found {len(targets)}")

    objectness_targets = torch.zeros(batch_size, point_count, device=predictions.device)
    centerness_targets = torch.zeros(batch_size, point_count, device=predictions.device)
    regression_targets = torch.zeros(batch_size, point_count, 6, device=predictions.device)
    for batch_index, boxes in enumerate(targets):
        obj, ctr, reg = build_targets(
            boxes,
            points,
            strides,
            levels,
            center_radius=center_radius,
        )
        objectness_targets[batch_index] = obj
        centerness_targets[batch_index] = ctr
        regression_targets[batch_index] = reg

    positive = objectness_targets > 0.5
    positive_count = positive.sum().clamp_min(1).to(predictions.dtype)
    objectness_loss = sigmoid_focal_loss(
        predictions[..., 0], objectness_targets
    ).sum() / positive_count

    if positive.any():
        centerness_loss = F.binary_cross_entropy_with_logits(
            predictions[..., 1][positive],
            centerness_targets[positive],
            reduction="sum",
        ) / positive_count
        weights = centerness_targets[positive].clamp_min(0.05)
        predicted_regression = predictions[..., 2:][positive]
        target_regression = regression_targets[positive]
        center_size = F.smooth_l1_loss(
            predicted_regression[:, :4],
            target_regression[:, :4],
            reduction="none",
            beta=0.25,
        ).mean(dim=1)
        center_size_loss = (center_size * weights).sum() / weights.sum().clamp_min(1.0)

        predicted_angle = F.normalize(predicted_regression[:, 4:6], dim=1, eps=1.0e-6)
        target_angle = F.normalize(target_regression[:, 4:6], dim=1, eps=1.0e-6)
        angle_distance = 1.0 - (predicted_angle * target_angle).sum(dim=1)
        angle_loss = (angle_distance * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        zero = predictions.sum() * 0.0
        centerness_loss = zero
        center_size_loss = zero
        angle_loss = zero

    total = objectness_loss + centerness_loss + 2.0 * center_size_loss + angle_loss
    return {
        "total": total,
        "objectness": objectness_loss,
        "centerness": centerness_loss,
        "center_size": center_size_loss,
        "angle": angle_loss,
        "positive_count": positive_count.detach(),
    }


def decode_batch(
    outputs: Sequence[torch.Tensor],
    *,
    score_threshold: float = 0.20,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 64,
    pre_nms_topk: int = 300,
) -> list[list[Detection]]:
    predictions = flatten_outputs(outputs)
    points, strides, _levels = make_points(outputs)
    results: list[list[Detection]] = []
    for batch_index in range(predictions.shape[0]):
        item = predictions[batch_index]
        scores = torch.sigmoid(item[:, 0]) * torch.sigmoid(item[:, 1])
        candidate_indices = (scores >= score_threshold).nonzero(as_tuple=False).squeeze(1)
        if candidate_indices.numel() == 0:
            results.append([])
            continue
        candidate_scores = scores[candidate_indices]
        if candidate_indices.numel() > pre_nms_topk:
            candidate_scores, order = torch.topk(candidate_scores, pre_nms_topk)
            candidate_indices = candidate_indices[order]
        regression = item[candidate_indices, 2:]
        candidate_points = points[candidate_indices]
        candidate_strides = strides[candidate_indices]
        centers = candidate_points + regression[:, :2] * candidate_strides[:, None]
        sizes = torch.exp(regression[:, 2:4].clamp(-4.0, 4.0)) * candidate_strides[:, None]
        angle = 0.5 * torch.atan2(regression[:, 4], regression[:, 5]) * 180.0 / math.pi

        detections = []
        for index in range(candidate_indices.numel()):
            cx = float(centers[index, 0].detach().cpu())
            cy = float(centers[index, 1].detach().cpu())
            width = float(sizes[index, 0].detach().cpu())
            height = float(sizes[index, 1].detach().cpu())
            angle_deg = float(angle[index].detach().cpu())
            cx, cy, width, height, angle_deg = canonicalize_obb(
                (cx, cy, width, height, angle_deg)
            )
            detections.append(
                Detection(
                    score=float(candidate_scores[index].detach().cpu()),
                    cx=cx,
                    cy=cy,
                    width=width,
                    height=height,
                    angle_deg=angle_deg,
                )
            )
        results.append(rotated_nms(detections, nms_iou_threshold, max_detections))
    return results


def rotated_nms(
    detections: Sequence[Detection],
    iou_threshold: float,
    max_detections: int,
) -> list[Detection]:
    remaining = sorted(detections, key=lambda detection: detection.score, reverse=True)
    kept: list[Detection] = []
    while remaining and len(kept) < max_detections:
        winner = remaining.pop(0)
        kept.append(winner)
        remaining = [
            candidate
            for candidate in remaining
            if rotated_iou(winner.obb, candidate.obb) < iou_threshold
        ]
    return kept


def rotated_iou(
    first: Sequence[float], second: Sequence[float]
) -> float:
    first_polygon = obb_corners(first)
    second_polygon = obb_corners(second)
    intersection_polygon = polygon_clip(first_polygon, second_polygon)
    intersection = polygon_area(intersection_polygon)
    first_area = max(0.0, float(first[2])) * max(0.0, float(first[3]))
    second_area = max(0.0, float(second[2])) * max(0.0, float(second[3]))
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def match_detections(
    detections: Sequence[Detection],
    ground_truths: Sequence[Sequence[float]],
    *,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    unmatched = set(range(len(ground_truths)))
    matches: list[tuple[Detection, Sequence[float], float]] = []
    false_positives = 0
    for detection in sorted(detections, key=lambda value: value.score, reverse=True):
        candidates = [
            (index, rotated_iou(detection.obb, ground_truths[index])) for index in unmatched
        ]
        if not candidates:
            false_positives += 1
            continue
        gt_index, iou = max(candidates, key=lambda value: value[1])
        if iou < iou_threshold:
            false_positives += 1
            continue
        unmatched.remove(gt_index)
        matches.append((detection, ground_truths[gt_index], iou))
    angle_errors = [angle_error_deg(match[0].angle_deg, float(match[1][4])) for match in matches]
    return {
        "tp": len(matches),
        "fp": false_positives,
        "fn": len(unmatched),
        "ious": [match[2] for match in matches],
        "angle_errors": angle_errors,
    }


def angle_error_deg(first: float, second: float) -> float:
    delta = abs(normalize_angle_180(first - second))
    return min(delta, 180.0 - delta)


def canonicalize_obb(
    obb: Sequence[float],
) -> tuple[float, float, float, float, float]:
    cx, cy, width, height, angle_deg = (float(value) for value in obb)
    if width > height:
        width, height = height, width
        angle_deg += 90.0
    return (cx, cy, width, height, normalize_angle_180(angle_deg))


def normalize_angle_180(angle_deg: float) -> float:
    angle = (angle_deg + 90.0) % 180.0 - 90.0
    if angle >= 90.0:
        angle -= 180.0
    return angle


def obb_corners(obb: Sequence[float]) -> list[tuple[float, float]]:
    cx, cy, width, height, angle_deg = (float(value) for value in obb)
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    half_width = width / 2.0
    half_height = height / 2.0
    return [
        (
            cx + local_x * cosine - local_y * sine,
            cy + local_x * sine + local_y * cosine,
        )
        for local_x, local_y in (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        )
    ]


def polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    ) / 2.0


def polygon_clip(
    subject: Sequence[tuple[float, float]],
    clip_polygon: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    output = list(subject)
    if not output:
        return []
    orientation = signed_polygon_area(clip_polygon)
    for index in range(len(clip_polygon)):
        edge_start = clip_polygon[index]
        edge_end = clip_polygon[(index + 1) % len(clip_polygon)]
        input_polygon = output
        output = []
        if not input_polygon:
            break
        previous = input_polygon[-1]
        previous_inside = point_inside_edge(previous, edge_start, edge_end, orientation)
        for current in input_polygon:
            current_inside = point_inside_edge(current, edge_start, edge_end, orientation)
            if current_inside:
                if not previous_inside:
                    output.append(line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(line_intersection(previous, current, edge_start, edge_end))
            previous = current
            previous_inside = current_inside
    return output


def signed_polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    return sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    ) / 2.0


def point_inside_edge(
    point: tuple[float, float],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
    orientation: float,
) -> bool:
    cross = (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    )
    return cross >= -1.0e-9 if orientation >= 0.0 else cross <= 1.0e-9


def line_intersection(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = first_start
    x2, y2 = first_end
    x3, y3 = second_start
    x4, y4 = second_end
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1.0e-12:
        return first_end
    determinant_first = x1 * y2 - y1 * x2
    determinant_second = x3 * y4 - y3 * x4
    return (
        (determinant_first * (x3 - x4) - (x1 - x2) * determinant_second) / denominator,
        (determinant_first * (y3 - y4) - (y1 - y2) * determinant_second) / denominator,
    )


def mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return None if not materialized else sum(materialized) / len(materialized)
