from __future__ import annotations

import base64
import csv
import io
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


TARGET_LABELS = ("5m", "6m", "7m")
IMAGE_SIZE = 64


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
    GeometryCase("recrop-yaw-left-0p10-angle20", angle_deg=20.0, perspective_yaw=-0.10, detector_style_recrop=True),
    GeometryCase("recrop-yaw-right-0p10-angle20", angle_deg=20.0, perspective_yaw=0.10, detector_style_recrop=True),
    GeometryCase("recrop-yaw-right-0p15-jitter", perspective_yaw=0.15, bbox_center_x=0.04, bbox_center_y=-0.03, bbox_scale_x=1.05, bbox_scale_y=0.96, detector_style_recrop=True),
)

SHIFT_CASES = (("shift-x-minus-2", -2, 0), ("shift-x-plus-2", 2, 0), ("shift-y-minus-2", 0, -2), ("shift-y-plus-2", 0, 2))


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


def _load_rows(database: Path, split: str) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in TARGET_LABELS)
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            f"""
            SELECT sample_id, base_label, class_index, image_gray_u8,
                   original_width, original_height, source, capture_id,
                   layout_id, region, source_image_path
            FROM sample
            WHERE split=? AND base_label IN ({placeholders})
            ORDER BY base_label, sample_id
            """,
            (split, *TARGET_LABELS),
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append({
            "sample_id": str(row[0]),
            "base_label": str(row[1]),
            "class_index": int(row[2]),
            "image": np.frombuffer(row[3], dtype=np.uint8).copy().reshape(IMAGE_SIZE, IMAGE_SIZE),
            "original_width": int(row[4]),
            "original_height": int(row[5]),
            "source": str(row[6]),
            "capture_id": None if row[7] is None else str(row[7]),
            "layout_id": None if row[8] is None else str(row[8]),
            "region": None if row[9] is None else str(row[9]),
            "source_image_path": str(row[10]),
        })
    counts = {label: sum(row["base_label"] == label for row in result) for label in TARGET_LABELS}
    if min(counts.values(), default=0) < 3:
        raise ValueError(f"Diagnostic split requires at least three samples per target class: {counts}")
    return result


def _content_extent(width: int, height: int) -> tuple[float, float]:
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = max(1, min(IMAGE_SIZE, int(math.floor(width * scale + 0.5))))
    resized_height = max(1, min(IMAGE_SIZE, int(math.floor(height * scale + 0.5))))
    return resized_width / IMAGE_SIZE, resized_height / IMAGE_SIZE


def _rectangle_corners(batch: int, extent_x: float, extent_y: float, *, device, dtype) -> torch.Tensor:
    corners = torch.tensor(
        [[-extent_x, -extent_y], [extent_x, -extent_y], [extent_x, extent_y], [-extent_x, extent_y]],
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
        [[cosine * case.scale_x, -sine * case.scale_y],
         [sine * case.scale_x, cosine * case.scale_y]],
        device=source.device,
        dtype=source.dtype,
    ).unsqueeze(0).expand(source.shape[0], -1, -1)
    return torch.bmm(destination, affine.transpose(1, 2))


def _solve_homography(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    x, y = source[:, :, 0], source[:, :, 1]
    u, v = destination[:, :, 0], destination[:, :, 1]
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    first = torch.stack((x, y, ones, zeros, zeros, zeros, -u * x, -u * y), dim=-1)
    second = torch.stack((zeros, zeros, zeros, x, y, ones, -v * x, -v * y), dim=-1)
    matrix = torch.stack((first, second), dim=2).reshape(source.shape[0], 8, 8)
    right = torch.stack((u, v), dim=2).reshape(source.shape[0], 8, 1)
    solution = torch.linalg.solve(matrix, right).squeeze(-1)
    result = torch.zeros((source.shape[0], 3, 3), device=source.device, dtype=source.dtype)
    result[:, 0, :3] = solution[:, :3]
    result[:, 1, :3] = solution[:, 3:6]
    result[:, 2, :2] = solution[:, 6:8]
    result[:, 2, 2] = 1.0
    return result


def _transform_points(homography: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((*points.shape[:2], 1), device=points.device, dtype=points.dtype)
    homogeneous = torch.cat((points, ones), dim=-1)
    transformed = torch.bmm(homogeneous, homography.transpose(1, 2))
    denominator = transformed[:, :, 2:3]
    sign = torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator))
    denominator = torch.where(denominator.abs() < 1.0e-6, sign * 1.0e-6, denominator)
    return transformed[:, :, :2] / denominator


def _normalized_grid(batch: int, height: int, width: int, *, device, dtype) -> torch.Tensor:
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def _warp(images: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    batch, _channels, height, width = images.shape
    grid = _normalized_grid(batch, height, width, device=images.device, dtype=images.dtype)
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
    grid = _normalized_grid(batch, IMAGE_SIZE, IMAGE_SIZE, device=source.device, dtype=source.dtype)
    qx, qy = grid[:, :, :, 0], grid[:, :, :, 1]
    width = bbox_size[:, 0].clamp_min(1.0e-4)
    height = bbox_size[:, 1].clamp_min(1.0e-4)
    ratio_x = torch.minimum(torch.ones_like(width), width / height)
    ratio_y = torch.minimum(torch.ones_like(height), height / width)
    mask = (qx.abs() <= ratio_x[:, None, None]) & (qy.abs() <= ratio_y[:, None, None])
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


def _apply_geometry_case(images: torch.Tensor, rows: Sequence[dict[str, Any]], case: GeometryCase) -> torch.Tensor:
    if case.name == "front-facing":
        return images
    batch = images.shape[0]
    if not case.detector_style_recrop:
        source = _rectangle_corners(batch, 1.0, 1.0, device=images.device, dtype=images.dtype)
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
    extents = [_content_extent(row["original_width"], row["original_height"]) for row in rows]
    extent_x = torch.tensor([x for x, _y in extents], device=images.device, dtype=images.dtype)
    extent_y = torch.tensor([y for _x, y in extents], device=images.device, dtype=images.dtype)
    extent_x = extent_x * (IMAGE_SIZE / canvas_width)
    extent_y = extent_y * (IMAGE_SIZE / canvas_height)
    source = _rectangle_corners(batch, 1.0, 1.0, device=images.device, dtype=images.dtype)
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
    return _sample_letterboxed_bbox(canvas, homography, center, size, _border_median(images))


def _shift_images(images: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    y_source = torch.arange(IMAGE_SIZE, device=images.device).sub(dy).clamp(0, IMAGE_SIZE - 1)
    x_source = torch.arange(IMAGE_SIZE, device=images.device).sub(dx).clamp(0, IMAGE_SIZE - 1)
    return images.index_select(2, y_source).index_select(3, x_source)


def _infer(model, images_01: torch.Tensor, *, mean: float, std: float, batch_size: int, device: torch.device) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, images_01.shape[0], batch_size):
            batch = images_01[start : start + batch_size].to(device, non_blocking=device.type == "cuda")
            logits = model(batch.sub(mean).div(std))
            outputs.append(logits.detach().float().cpu())
    return torch.cat(outputs, dim=0)


def _true_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(logits.shape[0])
    true_values = logits[rows, targets]
    masked = logits.clone()
    masked[rows, targets] = -torch.inf
    return true_values - masked.max(dim=1).values


def _condition_summary(logits: torch.Tensor, rows: Sequence[dict[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    targets = torch.tensor([row["class_index"] for row in rows], dtype=torch.long)
    predictions = logits.argmax(dim=1)
    probabilities = logits.softmax(dim=1)
    margins = _true_margin(logits, targets)
    entropy = -(probabilities.clamp_min(1.0e-12) * probabilities.clamp_min(1.0e-12).log()).sum(dim=1)
    confusion = {true: {pred: 0 for pred in (*TARGET_LABELS, "other")} for true in TARGET_LABELS}
    six_mix = 0
    six_count = 0
    for index, row in enumerate(rows):
        predicted_label = labels[int(predictions[index])]
        bucket = predicted_label if predicted_label in TARGET_LABELS else "other"
        confusion[row["base_label"]][bucket] += 1
        if row["base_label"] == "6m":
            six_count += 1
            if predicted_label in {"5m", "7m"}:
                six_mix += 1
    return {
        "count": len(rows),
        "accuracy": float((predictions == targets).float().mean()),
        "mean_true_probability": float(probabilities[torch.arange(len(rows)), targets].mean()),
        "mean_true_margin": float(margins.mean()),
        "mean_entropy_nats": float(entropy.mean()),
        "six_neighbor_confusion_rate": float(six_mix / max(six_count, 1)),
        "confusion": confusion,
    }


def _topk(logits: torch.Tensor, labels: Sequence[str], count: int = 3) -> list[dict[str, Any]]:
    probability = logits.softmax(dim=0)
    indices = logits.argsort(descending=True)[:count]
    return [
        {"label": labels[int(index)], "probability": float(probability[index]), "logit": float(logits[index])}
        for index in indices
    ]


def _select_representatives(
    rows: Sequence[dict[str, Any]],
    condition_logits: dict[str, torch.Tensor],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    targets = torch.tensor([row["class_index"] for row in rows], dtype=torch.long)
    front_margins = _true_margin(condition_logits["front-facing"], targets)
    condition_margins = {name: _true_margin(logits, targets) for name, logits in condition_logits.items()}
    selected: list[dict[str, Any]] = []
    for label in TARGET_LABELS:
        indices = [index for index, row in enumerate(rows) if row["base_label"] == label]
        hard = min(indices, key=lambda index: float(front_margins[index]))
        failures: list[tuple[float, int, str, str]] = []
        for case_name, logits in condition_logits.items():
            predictions = logits.argmax(dim=1)
            for index in indices:
                predicted = labels[int(predictions[index])]
                if predicted in TARGET_LABELS and predicted != label:
                    failures.append((float(condition_margins[case_name][index]), index, case_name, predicted))
        failures.sort(key=lambda item: item[0])
        fragile_event = next((item for item in failures if item[1] != hard), None)
        fragile = fragile_event[1] if fragile_event is not None else hard
        remaining = [index for index in indices if index not in {hard, fragile}]
        stable = max(
            remaining or indices,
            key=lambda index: min(float(margins[index]) for margins in condition_margins.values()),
        )
        picks = [("baseline-hard", hard, None), ("neighbor-fragile", fragile, fragile_event), ("view-stable", stable, None)]
        used: set[int] = set()
        for role, index, event in picks:
            if index in used:
                index = next(candidate for candidate in indices if candidate not in used)
                event = None
            used.add(index)
            selected.append({
                "role": role,
                "row_index": index,
                "baseline_true_margin": float(front_margins[index]),
                "neighbor_failure": None if event is None else {
                    "condition": event[2], "prediction": event[3], "true_margin": event[0]
                },
            })
    return selected


def _occlusion_result(
    model,
    image_01: torch.Tensor,
    target_index: int,
    *,
    mean: float,
    std: float,
    patch: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    baseline = _infer(model, image_01.unsqueeze(0), mean=mean, std=std, batch_size=1, device=device)
    baseline_margin = float(_true_margin(baseline, torch.tensor([target_index]))[0])
    variants: list[torch.Tensor] = []
    cells: list[tuple[int, int]] = []
    for y in range(0, IMAGE_SIZE, patch):
        for x in range(0, IMAGE_SIZE, patch):
            variant = image_01.clone()
            variant[:, y : y + patch, x : x + patch] = mean
            variants.append(variant)
            cells.append((y, x))
    logits = _infer(model, torch.stack(variants), mean=mean, std=std, batch_size=batch_size, device=device)
    margins = _true_margin(logits, torch.full((len(variants),), target_index, dtype=torch.long))
    drops = baseline_margin - margins.numpy()
    positive = np.maximum(drops, 0.0)
    total = float(positive.sum())
    square_sum = float(np.square(positive).sum())
    stats = {
        "top_patch_share": 0.0 if total <= 1.0e-12 else float(positive.max() / total),
        "effective_patch_count": 0.0 if square_sum <= 1.0e-12 else float(total * total / square_sum),
        "peak_drop": float(max(float(positive.max(initial=0.0)), 0.0)),
    }
    heat = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for (y, x), drop in zip(cells, drops, strict=True):
        heat[y : y + patch, x : x + patch] = float(drop)
    return heat, stats


def _centroid_diagnostic(feature: torch.Tensor, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    feature = F.normalize(feature.float(), dim=1)
    row_labels = [row["base_label"] for row in rows]
    predictions: list[str] = []
    margins: list[float] = []
    for index in range(len(rows)):
        similarities: dict[str, float] = {}
        for label in TARGET_LABELS:
            members = [i for i, value in enumerate(row_labels) if value == label and i != index]
            centroid = F.normalize(feature[members].mean(dim=0), dim=0)
            similarities[label] = float(torch.dot(feature[index], centroid))
        ranked = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        predictions.append(ranked[0][0])
        true_value = similarities[row_labels[index]]
        competitor = max(value for label, value in similarities.items() if label != row_labels[index])
        margins.append(true_value - competitor)
    centroids: dict[str, torch.Tensor] = {}
    for label in TARGET_LABELS:
        indices = [i for i, value in enumerate(row_labels) if value == label]
        centroids[label] = F.normalize(feature[indices].mean(dim=0), dim=0)
    return {
        "loo_centroid_accuracy": float(np.mean([pred == true for pred, true in zip(predictions, row_labels, strict=True)])),
        "mean_true_centroid_margin": float(np.mean(margins)),
        "cos_5m_6m": float(torch.dot(centroids["5m"], centroids["6m"])),
        "cos_6m_7m": float(torch.dot(centroids["6m"], centroids["7m"])),
        "cos_5m_7m": float(torch.dot(centroids["5m"], centroids["7m"])),
        "predictions": predictions,
        "margins": margins,
    }


def _representation_diagnostic(
    model,
    images: torch.Tensor,
    rows: Sequence[dict[str, Any]],
    *,
    mean: float,
    std: float,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    features = getattr(model, "features", None)
    if not isinstance(features, torch.nn.Sequential):
        return []
    captured: dict[int, list[torch.Tensor]] = {index: [] for index, _module in enumerate(features)}
    shapes: dict[int, str] = {}
    hooks = []

    def hook_for(index: int):
        def capture(_module, _inputs, output) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                return
            shapes[index] = "x".join(str(value) for value in output.shape[1:])
            captured[index].append(output.detach().float().mean(dim=(2, 3)).cpu())
        return capture

    for index, module in enumerate(features):
        hooks.append(module.register_forward_hook(hook_for(index)))
    try:
        _infer(model, images, mean=mean, std=std, batch_size=batch_size, device=device)
    finally:
        for hook in hooks:
            hook.remove()

    result: list[dict[str, Any]] = []
    for index, module in enumerate(features):
        batches = captured[index]
        if not batches:
            continue
        diagnostic = _centroid_diagnostic(torch.cat(batches, dim=0), rows)
        result.append({
            "layer_index": index,
            "layer": f"features.{index}:{type(module).__name__}",
            "shape": shapes[index],
            **diagnostic,
        })
    return result


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plotly(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def _occlusion_overlay(image_u8: np.ndarray, heat: np.ndarray) -> Image.Image:
    base = np.repeat(image_u8[:, :, None], 3, axis=2).astype(np.float32)
    scale = max(float(np.max(np.abs(heat))), 1.0e-6)
    strength = np.clip(np.abs(heat) / scale, 0.0, 1.0)[:, :, None]
    positive = np.zeros_like(base)
    positive[:, :, 0] = 255.0
    negative = np.zeros_like(base)
    negative[:, :, 2] = 255.0
    color = np.where(heat[:, :, None] >= 0, positive, negative)
    mixed = base * (1.0 - 0.55 * strength) + color * (0.55 * strength)
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), mode="RGB")


def _contact_sheet(
    path: Path,
    rows: Sequence[dict[str, Any]],
    representatives: Sequence[dict[str, Any]],
    front_logits: torch.Tensor,
    labels: Sequence[str],
    heatmaps: dict[int, np.ndarray],
) -> None:
    cell_width, cell_height = 540, 290
    sheet = Image.new("RGB", (cell_width * 3, cell_height * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for position, pick in enumerate(representatives):
        row_number, column = divmod(position, 3)
        x0, y0 = column * cell_width, row_number * cell_height
        index = int(pick["row_index"])
        row = rows[index]
        source = Image.fromarray(row["image"], mode="L").convert("RGB").resize((200, 200), Image.Resampling.NEAREST)
        overlay = _occlusion_overlay(row["image"], heatmaps[index]).resize((200, 200), Image.Resampling.NEAREST)
        sheet.paste(source, (x0 + 12, y0 + 45))
        sheet.paste(overlay, (x0 + 228, y0 + 45))
        top = _topk(front_logits[index], labels, count=2)
        draw.text(
            (x0 + 12, y0 + 10),
            f"{row['base_label']} {pick['role']} | pred={top[0]['label']} p={top[0]['probability']:.3f}",
            fill="black",
        )
        draw.text((x0 + 228, y0 + 250), "red: hide hurts | blue: hide helps", fill="black")
    sheet.save(path)


def _robustness_plot(condition_order: Sequence[str], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "data": [
            {"type": "scatter", "mode": "lines+markers", "name": "accuracy", "x": list(condition_order), "y": [summaries[name]["accuracy"] for name in condition_order]},
            {"type": "scatter", "mode": "lines+markers", "name": "true margin", "x": list(condition_order), "y": [summaries[name]["mean_true_margin"] for name in condition_order], "yaxis": "y2"},
            {"type": "scatter", "mode": "lines+markers", "name": "6m→5m/7m", "x": list(condition_order), "y": [summaries[name]["six_neighbor_confusion_rate"] for name in condition_order]},
        ],
        "layout": {
            "title": "5m/6m/7m robustness by fixed view/crop condition",
            "xaxis": {"title": "condition"},
            "yaxis": {"title": "accuracy / confusion rate", "range": [0, 1]},
            "yaxis2": {"title": "mean true margin", "overlaying": "y", "side": "right"},
        },
    }


def _confusion_plot(summary: dict[str, Any]) -> dict[str, Any]:
    columns = [*TARGET_LABELS, "other"]
    matrix = [[summary["confusion"][true][pred] for pred in columns] for true in TARGET_LABELS]
    return {
        "data": [{
            "type": "heatmap",
            "x": columns,
            "y": list(TARGET_LABELS),
            "z": matrix,
            "text": matrix,
            "texttemplate": "%{text}",
            "colorscale": "Blues",
        }],
        "layout": {"title": "Front-facing 5m/6m/7m confusion", "xaxis": {"title": "predicted"}, "yaxis": {"title": "true"}},
    }


def _layer_plot(layers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = [row["layer"] for row in layers]
    return {
        "data": [
            {"type": "scatter", "mode": "lines+markers", "name": "LOO centroid accuracy", "x": names, "y": [row["loo_centroid_accuracy"] for row in layers]},
            {"type": "scatter", "mode": "lines+markers", "name": "true-centroid margin", "x": names, "y": [row["mean_true_centroid_margin"] for row in layers], "yaxis": "y2"},
        ],
        "layout": {
            "title": "Where 5m/6m/7m separate inside model.features",
            "xaxis": {"title": "feature stage"},
            "yaxis": {"title": "centroid accuracy", "range": [0, 1]},
            "yaxis2": {"title": "centroid margin", "overlaying": "y", "side": "right"},
        },
    }


def _html_report(payload: dict[str, Any], contact_sheet: Path) -> str:
    image_b64 = base64.b64encode(contact_sheet.read_bytes()).decode("ascii")
    condition_rows = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in (
            row["condition_index"], row["condition"], f"{row['accuracy']:.3f}",
            f"{row['mean_true_margin']:.3f}", f"{row['mean_entropy_nats']:.3f}",
            f"{row['six_neighbor_confusion_rate']:.3f}",
        )) + "</tr>"
        for row in payload["summary_rows"]
    )
    layer_rows = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in (
            row["layer_index"], row["layer"], row["shape"], f"{row['loo_centroid_accuracy']:.3f}",
            f"{row['mean_true_centroid_margin']:.3f}",
        )) + "</tr>"
        for row in payload["representation"]
    )
    return f"""<!doctype html><meta charset='utf-8'><title>5m/6m/7m diagnostic</title>
<style>body{{font:15px system-ui;margin:28px;max-width:1500px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:5px 8px}}.note{{background:#f3f3f3;padding:12px;border-radius:6px}}img{{max-width:100%;image-rendering:pixelated}}</style>
<h1>5m / 6m / 7m diagnostic</h1>
<div class='note'>Aggregate metrics use every target sample in the requested split. The contact sheet is explanatory only: three deterministic samples per class. Red occlusion means hiding the patch lowers the true-class logit margin; blue means hiding it raises the margin. Layer centroid geometry is diagnostic, not causal proof.</div>
<h2>Robustness conditions</h2><table><tr><th>#</th><th>condition</th><th>accuracy</th><th>true margin</th><th>entropy</th><th>6m→5/7</th></tr>{condition_rows}</table>
<h2>Occlusion concentration</h2><p>top-patch share={payload['occlusion']['top_patch_share_mean']:.3f}; effective patch count={payload['occlusion']['effective_patch_count_mean']:.3f}; peak drop={payload['occlusion']['peak_drop_mean']:.3f}</p>
<h2>Nine concrete crops</h2><img alt='original and occlusion exemplars' src='data:image/png;base64,{image_b64}'>
<h2>Internal representation</h2><table><tr><th>#</th><th>layer</th><th>shape</th><th>centroid acc</th><th>centroid margin</th></tr>{layer_rows}</table>"""


def evaluate(context):
    parameters = context.parameters
    split = str(parameters["split"])
    batch_size = int(parameters["batch_size"])
    patch = int(parameters["occlusion_patch"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if patch < 1 or IMAGE_SIZE % patch != 0:
        raise ValueError("occlusion_patch must divide 64 exactly")

    database = context.corpus.root / "dataset.sqlite"
    labels, mean, std = _load_metadata_and_normalization(database)
    rows = _load_rows(database, split)
    images = torch.from_numpy(np.stack([row["image"] for row in rows])).float().unsqueeze(1).mul(1.0 / 255.0)
    targets = torch.tensor([row["class_index"] for row in rows], dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = context.model.module.to(device)
    model.eval()

    condition_images: dict[str, torch.Tensor] = {}
    condition_logits: dict[str, torch.Tensor] = {}
    condition_summaries: dict[str, dict[str, Any]] = {}
    for case in GEOMETRY_CASES:
        transformed = _apply_geometry_case(images, rows, case)
        condition_images[case.name] = transformed
        logits = _infer(model, transformed, mean=mean, std=std, batch_size=batch_size, device=device)
        condition_logits[case.name] = logits
        condition_summaries[case.name] = _condition_summary(logits, rows, labels)
    for name, dx, dy in SHIFT_CASES:
        transformed = _shift_images(images, dx, dy)
        condition_images[name] = transformed
        logits = _infer(model, transformed, mean=mean, std=std, batch_size=batch_size, device=device)
        condition_logits[name] = logits
        condition_summaries[name] = _condition_summary(logits, rows, labels)
    condition_order = list(condition_logits)

    representatives = _select_representatives(rows, condition_logits, labels)
    representative_indices = {int(item["row_index"]) for item in representatives}
    heatmaps: dict[int, np.ndarray] = {}
    occlusion_stats: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        heat, stats = _occlusion_result(
            model,
            images[index],
            int(row["class_index"]),
            mean=mean,
            std=std,
            patch=patch,
            batch_size=batch_size,
            device=device,
        )
        occlusion_stats.append(stats)
        if index in representative_indices:
            heatmaps[index] = heat

    representation = _representation_diagnostic(
        model,
        images,
        rows,
        mean=mean,
        std=std,
        batch_size=batch_size,
        device=device,
    )

    summary_rows: list[dict[str, Any]] = []
    for condition_index, name in enumerate(condition_order):
        summary = condition_summaries[name]
        summary_rows.append({
            "condition_index": condition_index,
            "condition": name,
            "accuracy": summary["accuracy"],
            "mean_true_probability": summary["mean_true_probability"],
            "mean_true_margin": summary["mean_true_margin"],
            "mean_entropy_nats": summary["mean_entropy_nats"],
            "six_neighbor_confusion_rate": summary["six_neighbor_confusion_rate"],
        })
        context.telemetry.report_scalar(group="manzu/robustness", series="accuracy", value=float(summary["accuracy"]), step=condition_index)
        context.telemetry.report_scalar(group="manzu/robustness", series="true_margin", value=float(summary["mean_true_margin"]), step=condition_index)
        context.telemetry.report_scalar(group="manzu/robustness", series="6m_to_5m_or_7m", value=float(summary["six_neighbor_confusion_rate"]), step=condition_index)

    occlusion_aggregate = {
        "top_patch_share_mean": float(np.mean([row["top_patch_share"] for row in occlusion_stats])),
        "effective_patch_count_mean": float(np.mean([row["effective_patch_count"] for row in occlusion_stats])),
        "peak_drop_mean": float(np.mean([row["peak_drop"] for row in occlusion_stats])),
    }
    for series, value in occlusion_aggregate.items():
        context.telemetry.report_scalar(group="manzu/occlusion", series=series, value=float(value), step=0)
    for layer_index, layer in enumerate(representation):
        context.telemetry.report_scalar(
            group="manzu/representation",
            series="centroid_accuracy",
            value=float(layer["loo_centroid_accuracy"]),
            step=layer_index,
        )

    perturbation_rows: list[dict[str, Any]] = []
    front_logits = condition_logits["front-facing"]
    for pick in representatives:
        index = int(pick["row_index"])
        row = rows[index]
        for name in condition_order:
            logits = condition_logits[name][index]
            probability = logits.softmax(dim=0)
            prediction = int(logits.argmax())
            margin = float(_true_margin(logits.unsqueeze(0), torch.tensor([row["class_index"]]))[0])
            top = _topk(logits, labels, count=2)
            perturbation_rows.append({
                "sample_id": row["sample_id"],
                "true_label": row["base_label"],
                "role": pick["role"],
                "condition": name,
                "prediction": labels[prediction],
                "true_probability": float(probability[row["class_index"]]),
                "true_margin": margin,
                "top2": ";".join(f"{item['label']}:{item['probability']:.6f}" for item in top),
            })

    per_sample: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        conditions: dict[str, Any] = {}
        for name in condition_order:
            logits = condition_logits[name][index]
            probability = logits.softmax(dim=0)
            conditions[name] = {
                "prediction": labels[int(logits.argmax())],
                "true_probability": float(probability[row["class_index"]]),
                "true_margin": float(_true_margin(logits.unsqueeze(0), torch.tensor([row["class_index"]]))[0]),
                "top3": _topk(logits, labels),
            }
        per_sample.append({
            "sample_id": row["sample_id"],
            "true_label": row["base_label"],
            "source": row["source"],
            "capture_id": row["capture_id"],
            "layout_id": row["layout_id"],
            "region": row["region"],
            "source_image_path": row["source_image_path"],
            "occlusion": occlusion_stats[index],
            "conditions": conditions,
        })

    work_dir = context.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_path = work_dir / "manzu-summary.csv"
    perturbation_path = work_dir / "manzu-exemplar-perturbations.csv"
    robustness_plot_path = work_dir / "manzu-robustness.plotly.json"
    confusion_plot_path = work_dir / "manzu-confusion.plotly.json"
    layer_table_path = work_dir / "manzu-layer-separation.csv"
    layer_plot_path = work_dir / "manzu-layer-separation.plotly.json"
    contact_sheet_path = work_dir / "manzu-contact-sheet.png"
    per_sample_path = work_dir / "manzu-per-sample.jsonl"
    report_json_path = work_dir / "manzu-report.json"
    report_html_path = work_dir / "manzu-report.html"

    _write_csv(
        summary_path,
        ("condition_index", "condition", "accuracy", "mean_true_probability", "mean_true_margin", "mean_entropy_nats", "six_neighbor_confusion_rate"),
        summary_rows,
    )
    _write_csv(
        perturbation_path,
        ("sample_id", "true_label", "role", "condition", "prediction", "true_probability", "true_margin", "top2"),
        perturbation_rows,
    )
    layer_table_rows = [
        {
            "layer_index": row["layer_index"],
            "layer": row["layer"],
            "shape": row["shape"],
            "loo_centroid_accuracy": row["loo_centroid_accuracy"],
            "mean_true_centroid_margin": row["mean_true_centroid_margin"],
            "cos_5m_6m": row["cos_5m_6m"],
            "cos_6m_7m": row["cos_6m_7m"],
            "cos_5m_7m": row["cos_5m_7m"],
        }
        for row in representation
    ]
    _write_csv(
        layer_table_path,
        ("layer_index", "layer", "shape", "loo_centroid_accuracy", "mean_true_centroid_margin", "cos_5m_6m", "cos_6m_7m", "cos_5m_7m"),
        layer_table_rows,
    )
    _write_plotly(robustness_plot_path, _robustness_plot(condition_order, condition_summaries))
    _write_plotly(confusion_plot_path, _confusion_plot(condition_summaries["front-facing"]))
    _write_plotly(layer_plot_path, _layer_plot(representation))
    _contact_sheet(contact_sheet_path, rows, representatives, front_logits, labels, heatmaps)
    with per_sample_path.open("w", encoding="utf-8") as handle:
        for row in per_sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = {
        "schema": "mjtensu.recognition/manzu-diagnostic-report/v1",
        "split": split,
        "target_labels": list(TARGET_LABELS),
        "sample_count": len(rows),
        "normalization": {"mean": mean, "std": std},
        "condition_order": condition_order,
        "summary_rows": summary_rows,
        "occlusion": occlusion_aggregate,
        "representatives": representatives,
        "representation": representation,
        "notes": {
            "condition_step": "manzu/robustness telemetry step is the zero-based index in condition_order.",
            "occlusion": "Positive/red means mean-value occlusion lowers the true-class margin. Concentration metrics use every target sample, not only exemplars.",
            "representation": "Leave-one-out nearest-centroid geometry over 5m/6m/7m is diagnostic and not causal attribution.",
        },
    }
    report_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_html_path.write_text(_html_report(payload, contact_sheet_path), encoding="utf-8")

    accuracy_values = [float(condition_summaries[name]["accuracy"]) for name in condition_order]
    margin_values = [float(condition_summaries[name]["mean_true_margin"]) for name in condition_order]
    metrics: dict[str, int | float] = {
        "front_accuracy": float(condition_summaries["front-facing"]["accuracy"]),
        "mean_condition_accuracy": float(np.mean(accuracy_values)),
        "worst_condition_accuracy": float(min(accuracy_values)),
        "front_true_margin_mean": float(condition_summaries["front-facing"]["mean_true_margin"]),
        "mean_condition_true_margin": float(np.mean(margin_values)),
        "six_neighbor_confusion_max": float(max(condition_summaries[name]["six_neighbor_confusion_rate"] for name in condition_order)),
        "occlusion_top_patch_share_mean": occlusion_aggregate["top_patch_share_mean"],
        "occlusion_effective_patch_count_mean": occlusion_aggregate["effective_patch_count_mean"],
        "occlusion_peak_drop_mean": occlusion_aggregate["peak_drop_mean"],
    }
    if representation:
        metrics["representation_best_centroid_accuracy"] = float(max(row["loo_centroid_accuracy"] for row in representation))

    return EvaluationCandidate(
        metrics=metrics,
        artifacts={
            "summary_table": summary_path,
            "perturbation_table": perturbation_path,
            "robustness_plot": robustness_plot_path,
            "confusion_plot": confusion_plot_path,
            "layer_separation_table": layer_table_path,
            "layer_separation_plot": layer_plot_path,
            "contact_sheet": contact_sheet_path,
            "per_sample_details": per_sample_path,
            "report_json": report_json_path,
            "report_html": report_html_path,
        },
    )
