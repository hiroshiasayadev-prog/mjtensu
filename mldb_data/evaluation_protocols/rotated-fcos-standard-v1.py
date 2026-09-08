from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from mldb.src.evaluation.interface import EvaluationResult
from mldb.src.model.loading import load_model

STRIDES = (8, 16, 32)
IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)

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

def _load_split(database: Path, split: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT image_rgb_u8, annotations_json FROM sample WHERE split = ? ORDER BY sample_id",
            (split,),
        ).fetchall()
    if not rows:
        raise ValueError(f"Corpus split is empty: {split}")
    images: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for payload, annotations_raw in rows:
        image = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone().reshape(3, 320, 320)
        annotations = json.loads(annotations_raw)
        boxes = torch.tensor(
            [[float(value) for value in item["obb"]] for item in annotations],
            dtype=torch.float32,
        )
        if boxes.numel() == 0:
            boxes = torch.empty((0, 5), dtype=torch.float32)
        images.append(image)
        targets.append(boxes)
    return torch.stack(images), targets


def evaluate(context):
    parameters = context.parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(parameters["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(parameters["tf32"])
    model = load_model(context.model).to(device)
    images_u8, targets = _load_split(context.corpus.artifact_path, "val")
    batch_size = int(parameters["batch_size"])
    amp_enabled = bool(parameters["amp"]) and device.type == "cuda"
    mean = IMAGE_MEAN.to(device)
    std = IMAGE_STD.to(device)
    tp = fp = fn = 0
    ious: list[float] = []
    angle_errors: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, images_u8.shape[0], batch_size):
            batch = images_u8[start : start + batch_size].to(device).float().mul_(1.0 / 255.0)
            batch = (batch - mean) / std
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(batch)
            detections = decode_batch(
                outputs,
                score_threshold=float(parameters["score_threshold"]),
                nms_iou_threshold=float(parameters["nms_iou_threshold"]),
                max_detections=int(parameters["max_detections"]),
            )
            batch_targets = targets[start : start + batch_size]
            for prediction, ground_truth in zip(detections, batch_targets, strict=True):
                matched = match_detections(
                    prediction,
                    ground_truth.tolist(),
                    iou_threshold=float(parameters["match_iou_threshold"]),
                )
                tp += int(matched["tp"])
                fp += int(matched["fp"])
                fn += int(matched["fn"])
                ious.extend(float(value) for value in matched["ious"])
                angle_errors.extend(float(value) for value in matched["angle_errors"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "rotated_iou_mean": float(sum(ious) / len(ious)) if ious else 0.0,
        "angle_error_deg_mean": float(sum(angle_errors) / len(angle_errors)) if angle_errors else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
    return EvaluationResult(metrics=metrics, artifacts={}, unavailable_outputs=())
