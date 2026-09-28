from __future__ import annotations

import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate


EXPECTED_LABELS = (
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "east", "south", "west", "north", "white", "green", "red", "invalid",
)
MANZU_FOCUS_LABELS = ("5m", "6m", "7m")
IMAGE_SIZE = 64
SPLIT = "manual_val"


@dataclass(frozen=True)
class GeometryCase:
    name: str
    angle_deg: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    perspective_yaw: float = 0.0
    perspective_pitch: float = 0.0
    bbox_center_x: float = 0.0
    bbox_center_y: float = 0.0
    bbox_scale_x: float = 1.0
    bbox_scale_y: float = 1.0
    detector_style_recrop: bool = False


GEOMETRY_CASES = (
    GeometryCase("front-facing"),
    GeometryCase("affine-x-compress-0p85", scale_x=0.85),
    GeometryCase("affine-x-compress-0p75", scale_x=0.75),
    GeometryCase("affine-y-compress-0p85", scale_y=0.85),
    GeometryCase("perspective-yaw-left-0p10", perspective_yaw=-0.10),
    GeometryCase("perspective-yaw-right-0p10", perspective_yaw=0.10),
    GeometryCase("perspective-yaw-right-0p15", perspective_yaw=0.15),
    GeometryCase("perspective-pitch-0p10", perspective_pitch=0.10),
    GeometryCase(
        "recrop-yaw-left-0p10-angle20",
        angle_deg=20.0,
        perspective_yaw=-0.10,
        detector_style_recrop=True,
    ),
    GeometryCase(
        "recrop-yaw-right-0p10-angle20",
        angle_deg=20.0,
        perspective_yaw=0.10,
        detector_style_recrop=True,
    ),
    GeometryCase(
        "recrop-yaw-right-0p15-jitter",
        perspective_yaw=0.15,
        bbox_center_x=0.04,
        bbox_center_y=-0.03,
        bbox_scale_x=1.05,
        bbox_scale_y=0.96,
        detector_style_recrop=True,
    ),
)
SHIFT_CASES = (
    ("shift-x-minus-2", -2, 0),
    ("shift-x-plus-2", 2, 0),
    ("shift-y-minus-2", 0, -2),
    ("shift-y-plus-2", 0, 2),
)


def _load_metadata_and_normalization(database: Path) -> tuple[list[str], float, float]:
    with sqlite3.connect(database) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM experiment_metadata"))
        labels = [str(value) for value in json.loads(metadata["base_labels"])]
        pixel_sum = 0.0
        pixel_sq_sum = 0.0
        pixel_count = 0
        for (payload,) in connection.execute("SELECT image_gray_u8 FROM sample WHERE split='train'"):
            values = np.frombuffer(payload, dtype=np.uint8).astype(np.float64) / 255.0
            pixel_sum += float(values.sum())
            pixel_sq_sum += float(np.square(values).sum())
            pixel_count += int(values.size)
    if pixel_count == 0:
        raise ValueError("Corpus train split is empty")
    mean = pixel_sum / pixel_count
    variance = max(pixel_sq_sum / pixel_count - mean * mean, 0.0)
    return labels, mean, max(math.sqrt(variance), 1.0 / 255.0)


def _load_rows(database: Path, labels: Sequence[str]) -> list[dict[str, Any]]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT sample_id, base_label, class_index, image_gray_u8,
                   original_width, original_height, source, capture_id,
                   layout_id, region, source_image_path
            FROM sample
            WHERE split=?
            ORDER BY class_index, sample_id
            """,
            (SPLIT,),
        ).fetchall()
    result: list[dict[str, Any]] = []
    counts = {label: 0 for label in labels}
    for row in rows:
        class_index = int(row[2])
        base_label = str(row[1])
        if class_index < 0 or class_index >= len(labels):
            raise ValueError(f"manual_val class_index out of range: {class_index}")
        if labels[class_index] != base_label:
            raise ValueError(
                f"manual_val label/index mismatch: label={base_label!r} index={class_index} "
                f"expected={labels[class_index]!r}"
            )
        counts[base_label] += 1
        result.append(
            {
                "sample_id": str(row[0]),
                "base_label": base_label,
                "class_index": class_index,
                "image": np.frombuffer(row[3], dtype=np.uint8)
                .copy()
                .reshape(IMAGE_SIZE, IMAGE_SIZE),
                "original_width": int(row[4]),
                "original_height": int(row[5]),
                "source": str(row[6]),
                "capture_id": None if row[7] is None else str(row[7]),
                "layout_id": None if row[8] is None else str(row[8]),
                "region": None if row[9] is None else str(row[9]),
                "source_image_path": str(row[10]),
            }
        )
    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"manual_val must cover every classifier class; missing={missing}")
    return result


def _content_extent(width: int, height: int) -> tuple[float, float]:
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = max(1, min(IMAGE_SIZE, int(math.floor(width * scale + 0.5))))
    resized_height = max(1, min(IMAGE_SIZE, int(math.floor(height * scale + 0.5))))
    return resized_width / IMAGE_SIZE, resized_height / IMAGE_SIZE


def _rectangle_corners(
    batch: int,
    extent_x: float,
    extent_y: float,
    *,
    device,
    dtype,
) -> torch.Tensor:
    corners = torch.tensor(
        [
            [-extent_x, -extent_y],
            [extent_x, -extent_y],
            [extent_x, extent_y],
            [-extent_x, extent_y],
        ],
        device=device,
        dtype=dtype,
    )
    return corners.unsqueeze(0).expand(batch, -1, -1).clone()


def _destination_quad(source: torch.Tensor, case: GeometryCase) -> torch.Tensor:
    destination = source.clone()
    extent_x = source[:, :, 0].abs().amax(dim=1)
    extent_y = source[:, :, 1].abs().amax(dim=1)
    yaw = torch.full_like(extent_y, case.perspective_yaw) * extent_y
    pitch = torch.full_like(extent_x, case.perspective_pitch) * extent_x
    destination[:, 0, 1] -= yaw
    destination[:, 1, 1] += yaw
    destination[:, 2, 1] -= yaw
    destination[:, 3, 1] += yaw
    destination[:, 0, 0] += pitch
    destination[:, 1, 0] -= pitch
    destination[:, 2, 0] += pitch
    destination[:, 3, 0] -= pitch

    radians = math.radians(case.angle_deg)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    affine = torch.tensor(
        [
            [cosine * case.scale_x, -sine * case.scale_y],
            [sine * case.scale_x, cosine * case.scale_y],
        ],
        device=source.device,
        dtype=source.dtype,
    ).unsqueeze(0).expand(source.shape[0], -1, -1)
    return torch.bmm(destination, affine.transpose(1, 2))


def _solve_homography(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    x, y = source[:, :, 0], source[:, :, 1]
    u, v = destination[:, :, 0], destination[:, :, 1]
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    first = torch.stack(
        (x, y, ones, zeros, zeros, zeros, -u * x, -u * y),
        dim=-1,
    )
    second = torch.stack(
        (zeros, zeros, zeros, x, y, ones, -v * x, -v * y),
        dim=-1,
    )
    matrix = torch.stack((first, second), dim=2).reshape(source.shape[0], 8, 8)
    right = torch.stack((u, v), dim=2).reshape(source.shape[0], 8, 1)
    solution = torch.linalg.solve(matrix, right).squeeze(-1)
    result = torch.zeros(
        (source.shape[0], 3, 3),
        device=source.device,
        dtype=source.dtype,
    )
    result[:, 0, :3] = solution[:, :3]
    result[:, 1, :3] = solution[:, 3:6]
    result[:, 2, :2] = solution[:, 6:8]
    result[:, 2, 2] = 1.0
    return result


def _transform_points(homography: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(
        (*points.shape[:2], 1),
        device=points.device,
        dtype=points.dtype,
    )
    homogeneous = torch.cat((points, ones), dim=-1)
    transformed = torch.bmm(homogeneous, homography.transpose(1, 2))
    denominator = transformed[:, :, 2:3]
    sign = torch.where(
        denominator < 0,
        -torch.ones_like(denominator),
        torch.ones_like(denominator),
    )
    denominator = torch.where(
        denominator.abs() < 1.0e-6,
        sign * 1.0e-6,
        denominator,
    )
    return transformed[:, :, :2] / denominator


def _normalized_grid(
    batch: int,
    height: int,
    width: int,
    *,
    device,
    dtype,
) -> torch.Tensor:
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def _warp(images: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    batch, _channels, height, width = images.shape
    grid = _normalized_grid(
        batch,
        height,
        width,
        device=images.device,
        dtype=images.dtype,
    )
    inverse = torch.linalg.inv(homography)
    source = _transform_points(inverse, grid.reshape(batch, -1, 2))
    return F.grid_sample(
        images,
        source.reshape(batch, height, width, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def _border_median(images: torch.Tensor) -> torch.Tensor:
    top = images[:, :, 0, :]
    bottom = images[:, :, -1, :]
    left = images[:, :, 1:-1, 0]
    right = images[:, :, 1:-1, -1]
    return torch.cat((top, bottom, left, right), dim=-1).median(dim=-1).values


def _sample_letterboxed_bbox(
    source: torch.Tensor,
    homography: torch.Tensor,
    bbox_center: torch.Tensor,
    bbox_size: torch.Tensor,
    fill: torch.Tensor,
) -> torch.Tensor:
    batch = source.shape[0]
    grid = _normalized_grid(
        batch,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=source.device,
        dtype=source.dtype,
    )
    qx, qy = grid[:, :, :, 0], grid[:, :, :, 1]
    width = bbox_size[:, 0].clamp_min(1.0e-4)
    height = bbox_size[:, 1].clamp_min(1.0e-4)
    ratio_x = torch.minimum(torch.ones_like(width), width / height)
    ratio_y = torch.minimum(torch.ones_like(height), height / width)
    mask = (qx.abs() <= ratio_x[:, None, None]) & (
        qy.abs() <= ratio_y[:, None, None]
    )
    local_x = qx / ratio_x[:, None, None].clamp_min(1.0e-4)
    local_y = qy / ratio_y[:, None, None].clamp_min(1.0e-4)
    warped_x = bbox_center[:, 0, None, None] + 0.5 * width[:, None, None] * local_x
    warped_y = bbox_center[:, 1, None, None] + 0.5 * height[:, None, None] * local_y
    warped = torch.stack((warped_x, warped_y), dim=-1)
    inverse = torch.linalg.inv(homography)
    source_points = _transform_points(inverse, warped.reshape(batch, -1, 2))
    sampled = F.grid_sample(
        source,
        source_points.reshape(batch, IMAGE_SIZE, IMAGE_SIZE, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return torch.where(mask[:, None], sampled, fill[:, :, None, None])


def _apply_geometry_case(
    images: torch.Tensor,
    rows: Sequence[dict[str, Any]],
    case: GeometryCase,
) -> torch.Tensor:
    if case.name == "front-facing":
        return images
    batch = images.shape[0]
    if not case.detector_style_recrop:
        source = _rectangle_corners(
            batch,
            1.0,
            1.0,
            device=images.device,
            dtype=images.dtype,
        )
        destination = _destination_quad(source, case)
        return _warp(images, _solve_homography(source, destination))

    canvas_scale = 1.5
    canvas_height = int(round(IMAGE_SIZE * canvas_scale))
    canvas_width = int(round(IMAGE_SIZE * canvas_scale))
    pad_y = canvas_height - IMAGE_SIZE
    pad_x = canvas_width - IMAGE_SIZE
    canvas = F.pad(
        images,
        (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2),
        mode="replicate",
    )
    extents = [
        _content_extent(row["original_width"], row["original_height"])
        for row in rows
    ]
    extent_x = torch.tensor(
        [x for x, _y in extents],
        device=images.device,
        dtype=images.dtype,
    )
    extent_y = torch.tensor(
        [y for _x, y in extents],
        device=images.device,
        dtype=images.dtype,
    )
    extent_x = extent_x * (IMAGE_SIZE / canvas_width)
    extent_y = extent_y * (IMAGE_SIZE / canvas_height)
    source = _rectangle_corners(
        batch,
        1.0,
        1.0,
        device=images.device,
        dtype=images.dtype,
    )
    source[:, :, 0] *= extent_x[:, None]
    source[:, :, 1] *= extent_y[:, None]
    destination = _destination_quad(source, case)
    homography = _solve_homography(source, destination)
    minimum, maximum = destination.amin(dim=1), destination.amax(dim=1)
    center = (minimum + maximum) * 0.5
    size = (maximum - minimum).clamp_min(1.0e-3)
    center[:, 0] += case.bbox_center_x * size[:, 0]
    center[:, 1] += case.bbox_center_y * size[:, 1]
    size[:, 0] *= max(case.bbox_scale_x, 0.25)
    size[:, 1] *= max(case.bbox_scale_y, 0.25)
    return _sample_letterboxed_bbox(
        canvas,
        homography,
        center,
        size,
        _border_median(images),
    )


def _shift_images(images: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    y_source = (
        torch.arange(IMAGE_SIZE, device=images.device)
        .sub(dy)
        .clamp(0, IMAGE_SIZE - 1)
    )
    x_source = (
        torch.arange(IMAGE_SIZE, device=images.device)
        .sub(dx)
        .clamp(0, IMAGE_SIZE - 1)
    )
    return images.index_select(2, y_source).index_select(3, x_source)


def _infer(
    model,
    images_01: torch.Tensor,
    *,
    mean: float,
    std: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, images_01.shape[0], batch_size):
            batch = images_01[start : start + batch_size].to(
                device,
                non_blocking=device.type == "cuda",
            )
            logits = model(batch.sub(mean).div(std))
            if not isinstance(logits, torch.Tensor):
                raise TypeError("Classifier must return a logits tensor")
            outputs.append(logits.detach().float().cpu())
    return torch.cat(outputs, dim=0)


def _true_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(logits.shape[0])
    true_values = logits[rows, targets]
    masked = logits.clone()
    masked[rows, targets] = -torch.inf
    return true_values - masked.max(dim=1).values


def _condition_summary(
    logits: torch.Tensor,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    if logits.ndim != 2 or logits.shape[0] != len(rows) or logits.shape[1] != len(labels):
        raise ValueError(
            "Classifier logits must have shape "
            f"({len(rows)}, {len(labels)}); got {tuple(logits.shape)}"
        )
    targets = torch.tensor([row["class_index"] for row in rows], dtype=torch.long)
    predictions = logits.argmax(dim=1)
    probabilities = logits.softmax(dim=1)
    margins = _true_margin(logits, targets)
    confusion = torch.zeros(
        (len(labels), len(labels)),
        dtype=torch.int64,
    )
    for true_index, predicted_index in zip(
        targets.tolist(),
        predictions.tolist(),
        strict=True,
    ):
        confusion[true_index, predicted_index] += 1

    per_class: list[dict[str, Any]] = []
    for class_index, label in enumerate(labels):
        mask = targets == class_index
        sample_count = int(mask.sum())
        correct_count = int((predictions[mask] == class_index).sum())
        recall = float(correct_count / sample_count)
        wrong_counts = confusion[class_index].clone()
        wrong_counts[class_index] = 0
        top_wrong_count = int(wrong_counts.max())
        top_wrong_index = int(wrong_counts.argmax()) if top_wrong_count else -1
        per_class.append(
            {
                "class_index": class_index,
                "label": label,
                "sample_count": sample_count,
                "correct_count": correct_count,
                "recall": recall,
                "error_rate": float(1.0 - recall),
                "mean_true_probability": float(
                    probabilities[mask, class_index].mean()
                ),
                "mean_true_margin": float(margins[mask].mean()),
                "top_wrong_label": "" if top_wrong_count == 0 else labels[top_wrong_index],
                "top_wrong_count": top_wrong_count,
                "top_wrong_rate": float(top_wrong_count / sample_count),
            }
        )

    six_index = labels.index("6m")
    five_index = labels.index("5m")
    seven_index = labels.index("7m")
    six_mask = targets == six_index
    six_neighbor_count = int(
        ((predictions[six_mask] == five_index) | (predictions[six_mask] == seven_index)).sum()
    )
    six_count = int(six_mask.sum())
    accuracy = float((predictions == targets).float().mean())
    macro_recall = float(np.mean([row["recall"] for row in per_class]))
    return {
        "count": len(rows),
        "accuracy": accuracy,
        "error_rate": float(1.0 - accuracy),
        "macro_recall": macro_recall,
        "worst_class_recall": float(min(row["recall"] for row in per_class)),
        "mean_true_probability": float(
            probabilities[torch.arange(len(rows)), targets].mean()
        ),
        "mean_true_margin": float(margins.mean()),
        "true_6m_to_5m_or_7m_confusion_rate": float(
            six_neighbor_count / six_count
        ),
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def _topk(
    logits: torch.Tensor,
    labels: Sequence[str],
    count: int = 3,
) -> list[dict[str, Any]]:
    probability = logits.softmax(dim=0)
    indices = logits.argsort(descending=True)[:count]
    return [
        {
            "label": labels[int(index)],
            "probability": float(probability[index]),
            "logit": float(logits[index]),
        }
        for index in indices
    ]


def _select_class_inspection(
    rows: Sequence[dict[str, Any]],
    condition_logits: dict[str, torch.Tensor],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    targets = torch.tensor([row["class_index"] for row in rows], dtype=torch.long)
    margins = {
        name: _true_margin(logits, targets)
        for name, logits in condition_logits.items()
    }
    predictions = {
        name: logits.argmax(dim=1)
        for name, logits in condition_logits.items()
    }
    front = margins["front-facing"]
    selected: list[dict[str, Any]] = []
    for class_index, label in enumerate(labels):
        indices = [
            index
            for index, row in enumerate(rows)
            if int(row["class_index"]) == class_index
        ]
        front_index = min(indices, key=lambda index: float(front[index]))
        worst_condition, worst_index = min(
            (
                (condition, index)
                for condition in condition_logits
                for index in indices
            ),
            key=lambda item: float(margins[item[0]][item[1]]),
        )
        selected.append(
            {
                "class_index": class_index,
                "label": label,
                "sample_count": len(indices),
                "front_sample_index": front_index,
                "front_sample_id": rows[front_index]["sample_id"],
                "front_prediction": labels[
                    int(predictions["front-facing"][front_index])
                ],
                "front_true_margin": float(front[front_index]),
                "worst_condition": worst_condition,
                "worst_sample_index": worst_index,
                "worst_sample_id": rows[worst_index]["sample_id"],
                "worst_prediction": labels[
                    int(predictions[worst_condition][worst_index])
                ],
                "worst_true_margin": float(
                    margins[worst_condition][worst_index]
                ),
            }
        )
    return selected


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plotly(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _confusion_plot(
    summary: dict[str, Any],
    labels: Sequence[str],
) -> dict[str, Any]:
    matrix = summary["confusion"]
    row_rates: list[list[float]] = []
    for row in matrix:
        total = max(sum(row), 1)
        row_rates.append([float(value / total) for value in row])
    return {
        "data": [
            {
                "type": "heatmap",
                "x": list(labels),
                "y": list(labels),
                "z": matrix,
                "customdata": row_rates,
                "colorscale": "Blues",
                "hovertemplate": (
                    "true=%{y}<br>predicted=%{x}<br>count=%{z}"
                    "<br>true-class row share=%{customdata:.1%}<extra></extra>"
                ),
            }
        ],
        "layout": {
            "title": "Front-facing 35-class confusion counts",
            "xaxis": {"title": "predicted class", "tickangle": -45},
            "yaxis": {"title": "true class", "autorange": "reversed"},
        },
    }


def _accuracy_recall_plot(
    condition_order: Sequence[str],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "overall accuracy",
                "x": list(condition_order),
                "y": [summaries[name]["accuracy"] for name in condition_order],
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "macro recall (35 classes)",
                "x": list(condition_order),
                "y": [summaries[name]["macro_recall"] for name in condition_order],
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "worst-class recall",
                "x": list(condition_order),
                "y": [
                    summaries[name]["worst_class_recall"]
                    for name in condition_order
                ],
            },
        ],
        "layout": {
            "title": (
                "Full-class robustness: accuracy and class-balanced recall "
                "by perturbation"
            ),
            "xaxis": {"title": "deterministic perturbation condition"},
            "yaxis": {"title": "rate", "range": [0, 1]},
        },
    }


def _true_margin_plot(
    condition_order: Sequence[str],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "mean true-class logit margin",
                "x": list(condition_order),
                "y": [
                    summaries[name]["mean_true_margin"]
                    for name in condition_order
                ],
            }
        ],
        "layout": {
            "title": (
                "Full-class robustness: mean true-class logit margin "
                "(model-dependent scale)"
            ),
            "xaxis": {"title": "deterministic perturbation condition"},
            "yaxis": {"title": "true-class logit margin"},
        },
    }


def _class_condition_heatmap(
    condition_order: Sequence[str],
    summaries: dict[str, dict[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    recall_rows: list[list[float]] = []
    count_rows: list[list[int]] = []
    for class_index, _label in enumerate(labels):
        recall_rows.append(
            [
                float(summaries[name]["per_class"][class_index]["recall"])
                for name in condition_order
            ]
        )
        count_rows.append(
            [
                int(summaries[name]["per_class"][class_index]["sample_count"])
                for name in condition_order
            ]
        )
    return {
        "data": [
            {
                "type": "heatmap",
                "x": list(condition_order),
                "y": list(labels),
                "z": recall_rows,
                "customdata": count_rows,
                "zmin": 0,
                "zmax": 1,
                "colorscale": "Viridis",
                "hovertemplate": (
                    "class=%{y}<br>condition=%{x}<br>recall=%{z:.3f}"
                    "<br>samples=%{customdata}<extra></extra>"
                ),
            }
        ],
        "layout": {
            "title": "Per-class recall by perturbation condition (all 35 classes)",
            "xaxis": {
                "title": "deterministic perturbation condition",
                "tickangle": -45,
            },
            "yaxis": {"title": "true class", "autorange": "reversed"},
        },
    }


def _tensor_to_image(image_01: torch.Tensor, size: int) -> Image.Image:
    array = (
        image_01.squeeze(0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="L").convert("RGB").resize(
        (size, size),
        Image.Resampling.NEAREST,
    )


def _contact_sheet(
    path: Path,
    rows: Sequence[dict[str, Any]],
    selections: Sequence[dict[str, Any]],
    condition_images: dict[str, torch.Tensor],
) -> None:
    columns = 5
    panel_width = 310
    panel_height = 180
    rows_count = math.ceil(len(selections) / columns)
    image_size = 88
    sheet = Image.new(
        "RGB",
        (panel_width * columns, panel_height * rows_count),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for position, item in enumerate(selections):
        grid_y, grid_x = divmod(position, columns)
        x0 = grid_x * panel_width
        y0 = grid_y * panel_height
        front_index = int(item["front_sample_index"])
        worst_index = int(item["worst_sample_index"])
        worst_condition = str(item["worst_condition"])
        front_image = _tensor_to_image(
            condition_images["front-facing"][front_index],
            image_size,
        )
        worst_image = _tensor_to_image(
            condition_images[worst_condition][worst_index],
            image_size,
        )
        sheet.paste(front_image, (x0 + 10, y0 + 48))
        sheet.paste(worst_image, (x0 + 108, y0 + 48))
        draw.text(
            (x0 + 10, y0 + 8),
            f"{int(item['class_index']):02d} {item['label']} | n={item['sample_count']}",
            fill="black",
        )
        draw.text(
            (x0 + 10, y0 + 24),
            (
                f"front-hard: pred={item['front_prediction']} "
                f"margin={item['front_true_margin']:.2f}"
            ),
            fill="black",
        )
        worst_fill = (
            "black"
            if item["worst_prediction"] == item["label"]
            else "red"
        )
        draw.text(
            (x0 + 10, y0 + 138),
            f"worst #{list(condition_images).index(worst_condition):02d} "
            f"{worst_condition[:25]}",
            fill="black",
        )
        draw.text(
            (x0 + 10, y0 + 154),
            (
                f"pred={item['worst_prediction']} "
                f"margin={item['worst_true_margin']:.2f}"
            ),
            fill=worst_fill,
        )
        draw.text((x0 + 35, y0 + 136 - 14), "front", fill="gray")
        draw.text((x0 + 135, y0 + 136 - 14), "worst", fill="gray")
    sheet.save(path)


def _build_rows(
    condition_order: Sequence[str],
    summaries: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    condition_rows: list[dict[str, Any]] = []
    class_condition_rows: list[dict[str, Any]] = []
    manzu_rows: list[dict[str, Any]] = []
    for condition_index, name in enumerate(condition_order):
        summary = summaries[name]
        condition_rows.append(
            {
                "condition_index": condition_index,
                "condition": name,
                "sample_count": summary["count"],
                "overall_accuracy": summary["accuracy"],
                "overall_error_rate": summary["error_rate"],
                "macro_recall": summary["macro_recall"],
                "worst_class_recall": summary["worst_class_recall"],
                "mean_true_probability": summary["mean_true_probability"],
                "mean_true_margin": summary["mean_true_margin"],
                "true_6m_to_5m_or_7m_confusion_rate": summary[
                    "true_6m_to_5m_or_7m_confusion_rate"
                ],
            }
        )
        for per_class in summary["per_class"]:
            class_condition_rows.append(
                {
                    "condition_index": condition_index,
                    "condition": name,
                    **per_class,
                }
            )
        by_label = {
            row["label"]: row
            for row in summary["per_class"]
        }
        manzu_rows.append(
            {
                "condition_index": condition_index,
                "condition": name,
                "recall_5m": by_label["5m"]["recall"],
                "recall_6m": by_label["6m"]["recall"],
                "recall_7m": by_label["7m"]["recall"],
                "true_6m_to_5m_or_7m_confusion_rate": summary[
                    "true_6m_to_5m_or_7m_confusion_rate"
                ],
            }
        )
    return condition_rows, class_condition_rows, manzu_rows


def evaluate(context):
    batch_size = int(context.parameters["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    database = context.corpus.root / "dataset.sqlite"
    labels, mean, std = _load_metadata_and_normalization(database)
    if tuple(labels) != EXPECTED_LABELS:
        raise ValueError(
            "Corpus base_labels do not match tile-shape-classification-35-v1"
        )
    rows = _load_rows(database, labels)
    images = (
        torch.from_numpy(np.stack([row["image"] for row in rows]))
        .float()
        .unsqueeze(1)
        .mul(1.0 / 255.0)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = context.model.module.to(device)
    model.eval()

    condition_images: dict[str, torch.Tensor] = {}
    condition_logits: dict[str, torch.Tensor] = {}
    condition_summaries: dict[str, dict[str, Any]] = {}
    for case in GEOMETRY_CASES:
        transformed = _apply_geometry_case(images, rows, case)
        condition_images[case.name] = transformed
        logits = _infer(
            model,
            transformed,
            mean=mean,
            std=std,
            batch_size=batch_size,
            device=device,
        )
        condition_logits[case.name] = logits
        condition_summaries[case.name] = _condition_summary(
            logits,
            rows,
            labels,
        )
    for name, dx, dy in SHIFT_CASES:
        transformed = _shift_images(images, dx, dy)
        condition_images[name] = transformed
        logits = _infer(
            model,
            transformed,
            mean=mean,
            std=std,
            batch_size=batch_size,
            device=device,
        )
        condition_logits[name] = logits
        condition_summaries[name] = _condition_summary(
            logits,
            rows,
            labels,
        )
    condition_order = list(condition_logits)

    for condition_index, name in enumerate(condition_order):
        summary = condition_summaries[name]
        context.telemetry.report_scalar(
            group="classifier/full_class_robustness",
            series="overall_accuracy",
            value=float(summary["accuracy"]),
            step=condition_index,
        )
        context.telemetry.report_scalar(
            group="classifier/full_class_robustness",
            series="macro_recall",
            value=float(summary["macro_recall"]),
            step=condition_index,
        )
        context.telemetry.report_scalar(
            group="classifier/full_class_robustness",
            series="mean_true_margin",
            value=float(summary["mean_true_margin"]),
            step=condition_index,
        )
        context.telemetry.report_scalar(
            group="classifier/manzu_focus",
            series="true_6m_to_5m_or_7m_confusion_rate",
            value=float(summary["true_6m_to_5m_or_7m_confusion_rate"]),
            step=condition_index,
        )

    condition_rows, class_condition_rows, manzu_rows = _build_rows(
        condition_order,
        condition_summaries,
    )
    selections = _select_class_inspection(
        rows,
        condition_logits,
        labels,
    )

    per_sample: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        conditions: dict[str, Any] = {}
        for name in condition_order:
            logits = condition_logits[name][index]
            probabilities = logits.softmax(dim=0)
            prediction = int(logits.argmax())
            conditions[name] = {
                "prediction": labels[prediction],
                "correct": prediction == int(row["class_index"]),
                "true_probability": float(
                    probabilities[int(row["class_index"])]
                ),
                "true_margin": float(
                    _true_margin(
                        logits.unsqueeze(0),
                        torch.tensor([int(row["class_index"])]),
                    )[0]
                ),
                "top3": _topk(logits, labels),
            }
        per_sample.append(
            {
                "sample_id": row["sample_id"],
                "true_label": row["base_label"],
                "class_index": row["class_index"],
                "source": row["source"],
                "capture_id": row["capture_id"],
                "layout_id": row["layout_id"],
                "region": row["region"],
                "source_image_path": row["source_image_path"],
                "conditions": conditions,
            }
        )

    work_dir = context.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    condition_table_path = work_dir / "full-class-condition-summary.csv"
    class_condition_table_path = work_dir / "full-class-per-class-condition.csv"
    confusion_path = work_dir / "front-facing-35-class-confusion.plotly.json"
    accuracy_recall_path = work_dir / "full-class-robustness-accuracy-recall.plotly.json"
    true_margin_path = work_dir / "full-class-robustness-true-margin.plotly.json"
    class_heatmap_path = work_dir / "full-class-condition-recall-heatmap.plotly.json"
    contact_sheet_path = work_dir / "full-class-error-inspection-contact-sheet.png"
    manzu_focus_path = work_dir / "manzu-5m-6m-7m-focus.csv"
    per_sample_path = work_dir / "full-class-per-sample.jsonl"
    report_path = work_dir / "full-class-diagnostic-report.json"

    _write_csv(
        condition_table_path,
        (
            "condition_index",
            "condition",
            "sample_count",
            "overall_accuracy",
            "overall_error_rate",
            "macro_recall",
            "worst_class_recall",
            "mean_true_probability",
            "mean_true_margin",
            "true_6m_to_5m_or_7m_confusion_rate",
        ),
        condition_rows,
    )
    _write_csv(
        class_condition_table_path,
        (
            "condition_index",
            "condition",
            "class_index",
            "label",
            "sample_count",
            "correct_count",
            "recall",
            "error_rate",
            "mean_true_probability",
            "mean_true_margin",
            "top_wrong_label",
            "top_wrong_count",
            "top_wrong_rate",
        ),
        class_condition_rows,
    )
    _write_csv(
        manzu_focus_path,
        (
            "condition_index",
            "condition",
            "recall_5m",
            "recall_6m",
            "recall_7m",
            "true_6m_to_5m_or_7m_confusion_rate",
        ),
        manzu_rows,
    )
    _write_plotly(
        confusion_path,
        _confusion_plot(condition_summaries["front-facing"], labels),
    )
    _write_plotly(
        accuracy_recall_path,
        _accuracy_recall_plot(condition_order, condition_summaries),
    )
    _write_plotly(
        true_margin_path,
        _true_margin_plot(condition_order, condition_summaries),
    )
    _write_plotly(
        class_heatmap_path,
        _class_condition_heatmap(condition_order, condition_summaries, labels),
    )
    _contact_sheet(
        contact_sheet_path,
        rows,
        selections,
        condition_images,
    )
    with per_sample_path.open("w", encoding="utf-8") as handle:
        for item in per_sample:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    front = condition_summaries["front-facing"]
    accuracy_values = [
        float(condition_summaries[name]["accuracy"])
        for name in condition_order
    ]
    macro_recall_values = [
        float(condition_summaries[name]["macro_recall"])
        for name in condition_order
    ]
    true_margin_values = [
        float(condition_summaries[name]["mean_true_margin"])
        for name in condition_order
    ]
    class_condition_recalls = [
        float(row["recall"])
        for row in class_condition_rows
    ]
    metrics: dict[str, int | float] = {
        "front_accuracy": float(front["accuracy"]),
        "front_macro_recall": float(front["macro_recall"]),
        "front_worst_class_recall": float(front["worst_class_recall"]),
        "mean_condition_accuracy": float(np.mean(accuracy_values)),
        "worst_condition_accuracy": float(min(accuracy_values)),
        "mean_condition_macro_recall": float(np.mean(macro_recall_values)),
        "worst_condition_macro_recall": float(min(macro_recall_values)),
        "worst_class_condition_recall": float(min(class_condition_recalls)),
        "front_true_margin_mean": float(front["mean_true_margin"]),
        "mean_condition_true_margin": float(np.mean(true_margin_values)),
        "true_6m_to_5m_or_7m_confusion_max": float(
            max(
                condition_summaries[name][
                    "true_6m_to_5m_or_7m_confusion_rate"
                ]
                for name in condition_order
            )
        ),
    }

    report = {
        "schema": "mjtensu.recognition/tile-full-class-diagnostic-report/v1",
        "split": SPLIT,
        "labels": list(labels),
        "sample_count": len(rows),
        "normalization": {"mean": mean, "std": std},
        "condition_order": condition_order,
        "metrics": metrics,
        "condition_summary": condition_rows,
        "per_class_condition": class_condition_rows,
        "class_error_inspection_selection": selections,
        "manzu_focus": {
            "labels": list(MANZU_FOCUS_LABELS),
            "semantics": (
                "Secondary focused slice. "
                "true_6m_to_5m_or_7m_confusion_rate measures only true 6m "
                "samples predicted as 5m or 7m."
            ),
            "conditions": manzu_rows,
        },
        "telemetry": {
            "cadence": "once per completed deterministic perturbation condition",
            "step": "zero-based index in condition_order",
            "groups": {
                "classifier/full_class_robustness": [
                    "overall_accuracy",
                    "macro_recall",
                    "mean_true_margin",
                ],
                "classifier/manzu_focus": [
                    "true_6m_to_5m_or_7m_confusion_rate"
                ],
            },
        },
        "notes": {
            "true_margin": (
                "Raw-logit true-class margin is model-scale dependent. "
                "Use the separate margin artifact primarily to diagnose "
                "within-model condition sensitivity, not as a calibrated "
                "cross-architecture score."
            ),
            "contact_sheet": (
                "Every class contributes one panel. Left is that class's "
                "lowest-margin front-facing sample. Right is the lowest-margin "
                "sample-condition event for that class across all evaluated "
                "perturbations."
            ),
            "acceptance": (
                "All metrics and artifacts are diagnostic outputs only; "
                "they do not select or accept a model."
            ),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return EvaluationCandidate(
        metrics=metrics,
        artifacts={
            "condition_summary_table": condition_table_path,
            "per_class_condition_table": class_condition_table_path,
            "front_confusion_matrix": confusion_path,
            "robustness_accuracy_recall_plot": accuracy_recall_path,
            "robustness_true_margin_plot": true_margin_path,
            "class_condition_recall_heatmap": class_heatmap_path,
            "class_error_contact_sheet": contact_sheet_path,
            "manzu_focus_table": manzu_focus_path,
            "per_sample_details": per_sample_path,
            "diagnostic_report": report_path,
        },
    )
