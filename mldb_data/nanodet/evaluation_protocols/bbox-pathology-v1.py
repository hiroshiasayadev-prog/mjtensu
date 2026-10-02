from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate


INPUT_SIZE = 320
BGR_MEAN = torch.tensor((103.53, 116.28, 123.675), dtype=torch.float32).view(1, 3, 1, 1)
BGR_STD = torch.tensor((57.375, 57.12, 58.395), dtype=torch.float32).view(1, 3, 1, 1)


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_xywh(cls, value: Sequence[float]) -> "Box":
        if len(value) != 4:
            raise ValueError("bbox must have four values")
        x, y, width, height = (float(item) for item in value)
        if not all(math.isfinite(item) for item in (x, y, width, height)):
            raise ValueError("bbox values must be finite")
        if width <= 0.0 or height <= 0.0:
            raise ValueError("bbox dimensions must be positive")
        return cls(x, y, x + width, y + height)

    def xywh(self) -> list[float]:
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source_domain: str
    source_file_name: str
    image: torch.Tensor
    ground_truths: tuple[Box, ...]


@dataclass(frozen=True)
class PathologyResult:
    duplicate_gt_count: int
    duplicate_extra_prediction_count: int
    multi_gt_prediction_count: int
    spurious_prediction_count: int
    missed_gt_count: int
    tangled_component_count: int
    maximum_gt_multiplicity: int
    maximum_pred_gt_degree: int
    count_delta: int
    duplicate_score: float
    multi_gt_score: float
    spurious_score: float

    @property
    def affected(self) -> bool:
        return bool(
            self.duplicate_gt_count
            or self.multi_gt_prediction_count
            or self.spurious_prediction_count
            or self.missed_gt_count
        )


class CompositeDataset(Dataset):
    def __init__(self, path: Path, split: str) -> None:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                """
                SELECT sample_id, source_domain, source_file_name,
                       annotations_json, image_chw_u8
                FROM sample
                WHERE split=?
                ORDER BY sample_id
                """,
                (split,),
            ).fetchall()
        if not rows:
            raise ValueError(f"corpus split is empty: {split}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Sample:
        sample_id, source_domain, source_file_name, annotations_raw, image_raw = self.rows[index]
        image = torch.from_numpy(
            np.frombuffer(image_raw, dtype=np.uint8).copy().reshape(3, INPUT_SIZE, INPUT_SIZE)
        )
        annotations = json.loads(annotations_raw)
        ground_truths = tuple(Box.from_xywh(item["bbox"]) for item in annotations)
        return Sample(
            sample_id=str(sample_id),
            source_domain=str(source_domain),
            source_file_name=str(source_file_name),
            image=image,
            ground_truths=ground_truths,
        )


def _collate(samples: Sequence[Sample]) -> tuple[torch.Tensor, list[Sample]]:
    return torch.stack([sample.image for sample in samples]), list(samples)


def _preprocess(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    bgr = images[:, [2, 1, 0]].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    mean = BGR_MEAN.to(device)
    std = BGR_STD.to(device)
    return (bgr - mean) / std


def _model_spec(model: torch.nn.Module) -> tuple[tuple[int, ...], int, int]:
    head = getattr(model, "head", None)
    raw_strides = getattr(head, "strides", (8, 16, 32, 64))
    strides = tuple(int(value) for value in raw_strides)
    reg_max = int(getattr(head, "reg_max", 7))
    num_classes = int(getattr(head, "num_classes", 1))
    if not strides or any(value <= 0 for value in strides):
        raise ValueError("NanoDet strides must be positive")
    if reg_max < 1:
        raise ValueError("NanoDet reg_max must be positive")
    if num_classes != 1:
        raise ValueError("bbox-pathology-v1 requires one detector class")
    return strides, reg_max, num_classes


def _center_priors(strides: Sequence[int], device: torch.device) -> torch.Tensor:
    chunks = []
    for stride in strides:
        size = math.ceil(INPUT_SIZE / stride)
        coordinates = torch.arange(size, dtype=torch.float32, device=device) * float(stride)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        stride_column = torch.full_like(xx.reshape(-1), float(stride))
        chunks.append(torch.stack((xx.reshape(-1), yy.reshape(-1), stride_column), dim=1))
    return torch.cat(chunks, dim=0)


def _decode_batch(
    output: torch.Tensor,
    *,
    strides: Sequence[int],
    reg_max: int,
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> list[list[Detection]]:
    if not isinstance(output, torch.Tensor) or output.ndim != 3:
        raise TypeError("NanoDet model must return [batch, points, channels] tensor")
    expected_points = sum(math.ceil(INPUT_SIZE / stride) ** 2 for stride in strides)
    expected_channels = 1 + 4 * (reg_max + 1)
    if output.shape[1:] != (expected_points, expected_channels):
        raise ValueError(
            f"unexpected NanoDet output shape {tuple(output.shape)}; "
            f"expected [batch,{expected_points},{expected_channels}]"
        )

    priors = _center_priors(strides, output.device)
    class_scores = output[..., 0].sigmoid()
    regression = output[..., 1:].reshape(output.shape[0], expected_points, 4, reg_max + 1)
    probabilities = regression.softmax(dim=-1)
    bins = torch.arange(reg_max + 1, dtype=output.dtype, device=output.device)
    distances = (probabilities * bins).sum(dim=-1) * priors[None, :, 2, None]
    x1 = (priors[None, :, 0] - distances[..., 0]).clamp(0.0, float(INPUT_SIZE))
    y1 = (priors[None, :, 1] - distances[..., 1]).clamp(0.0, float(INPUT_SIZE))
    x2 = (priors[None, :, 0] + distances[..., 2]).clamp(0.0, float(INPUT_SIZE))
    y2 = (priors[None, :, 1] + distances[..., 3]).clamp(0.0, float(INPUT_SIZE))

    results: list[list[Detection]] = []
    for batch_index in range(output.shape[0]):
        indices = torch.nonzero(class_scores[batch_index] > score_threshold, as_tuple=False).squeeze(1)
        candidates = []
        for index in indices.tolist():
            if x2[batch_index, index] <= x1[batch_index, index] or y2[batch_index, index] <= y1[batch_index, index]:
                continue
            candidates.append(
                Detection(
                    box=Box(
                        float(x1[batch_index, index]),
                        float(y1[batch_index, index]),
                        float(x2[batch_index, index]),
                        float(y2[batch_index, index]),
                    ),
                    score=float(class_scores[batch_index, index]),
                )
            )
        results.append(_nms(candidates, nms_iou_threshold, max_detections))
    return results


def _box_area(box: Box) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def _intersection_area(left: Box, right: Box) -> float:
    return max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1)) * max(
        0.0, min(left.y2, right.y2) - max(left.y1, right.y1)
    )


def _iou(left: Box, right: Box) -> float:
    intersection = _intersection_area(left, right)
    union = _box_area(left) + _box_area(right) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _overlap_over_smaller(left: Box, right: Box) -> float:
    smaller = min(_box_area(left), _box_area(right))
    return 0.0 if smaller <= 0.0 else _intersection_area(left, right) / smaller


def _nms(
    detections: Sequence[Detection], iou_threshold: float, max_detections: int
) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.score, reverse=True):
        if any(_iou(candidate.box, accepted.box) > iou_threshold for accepted in kept):
            continue
        kept.append(candidate)
        if len(kept) >= max_detections:
            break
    return kept


_FIXED_REGIONS: tuple[tuple[str, Box], ...] = (
    ("completed_hand", Box(7.0, 0.0, 313.0, 72.0)),
    ("dora_indicators", Box(7.0, 74.0, 313.0, 146.0)),
    ("melds", Box(74.0, 148.0, 246.0, 320.0)),
)


def _clip(left: Box, right: Box) -> Box | None:
    result = Box(
        max(left.x1, right.x1),
        max(left.y1, right.y1),
        min(left.x2, right.x2),
        min(left.y2, right.y2),
    )
    return None if result.x2 <= result.x1 or result.y2 <= result.y1 else result


def _assign_region(box: Box) -> tuple[str, Box] | None:
    center_x = 0.5 * (box.x1 + box.x2)
    center_y = 0.5 * (box.y1 + box.y2)
    for name, region in _FIXED_REGIONS:
        if region.x1 <= center_x < region.x2 and region.y1 <= center_y < region.y2:
            clipped = _clip(box, region)
            if clipped is not None:
                return name, clipped
    return None


def _suppress_product_duplicates(
    detections: Sequence[Detection], overlap_threshold: float
) -> list[Detection]:
    by_region: defaultdict[str, list[tuple[Detection, Box]]] = defaultdict(list)
    for detection in detections:
        assigned = _assign_region(detection.box)
        if assigned is not None:
            region, clipped = assigned
            by_region[region].append((detection, clipped))

    winners: list[Detection] = []
    for group in by_region.values():
        candidates: list[tuple[Detection, Box]] = []
        for candidate_index, candidate in enumerate(group):
            _, candidate_box = candidate
            candidate_area = _box_area(candidate_box)
            smaller = [
                other
                for other_index, other in enumerate(group)
                if other_index != candidate_index
                and _box_area(other[1]) < candidate_area
                and _overlap_over_smaller(candidate_box, other[1]) >= overlap_threshold
            ]
            bridge = any(
                _overlap_over_smaller(smaller[left][1], smaller[right][1]) < overlap_threshold
                for left in range(len(smaller))
                for right in range(left + 1, len(smaller))
            )
            if not bridge:
                candidates.append(candidate)

        candidates.sort(key=lambda item: item[0].score, reverse=True)
        kept: list[tuple[Detection, Box]] = []
        for candidate in candidates:
            if any(
                _overlap_over_smaller(candidate[1], accepted[1]) >= overlap_threshold
                for accepted in kept
            ):
                continue
            kept.append(candidate)
        winners.extend(detection for detection, _ in kept)
    return winners


def _pair_details(prediction: Box, ground_truth: Box) -> dict[str, float]:
    intersection = _intersection_area(prediction, ground_truth)
    pred_area = _box_area(prediction)
    gt_area = _box_area(ground_truth)
    union = pred_area + gt_area - intersection
    return {
        "iou": 0.0 if union <= 0.0 else intersection / union,
        "gt_coverage": 0.0 if gt_area <= 0.0 else intersection / gt_area,
        "pred_coverage": 0.0 if pred_area <= 0.0 else intersection / pred_area,
        "overlap_over_smaller": (
            0.0 if min(pred_area, gt_area) <= 0.0 else intersection / min(pred_area, gt_area)
        ),
    }


def _tangled_components(
    gt_neighbors: Sequence[Sequence[int]], pred_neighbors: Sequence[Sequence[int]]
) -> int:
    seen_gt: set[int] = set()
    seen_pred: set[int] = set()
    count = 0
    for start, neighbors in enumerate(gt_neighbors):
        if start in seen_gt or not neighbors:
            continue
        pending: list[tuple[str, int]] = [("gt", start)]
        component_gt: set[int] = set()
        component_pred: set[int] = set()
        while pending:
            kind, index = pending.pop()
            if kind == "gt":
                if index in seen_gt:
                    continue
                seen_gt.add(index)
                component_gt.add(index)
                pending.extend(("pred", pred) for pred in gt_neighbors[index])
            else:
                if index in seen_pred:
                    continue
                seen_pred.add(index)
                component_pred.add(index)
                pending.extend(("gt", gt) for gt in pred_neighbors[index])
        if len(component_gt) >= 2 and len(component_pred) >= 2:
            count += 1
    return count


def _analyze(
    ground_truths: Sequence[Box],
    detections: Sequence[Detection],
    overlap_threshold: float,
) -> tuple[PathologyResult, dict[str, Any]]:
    pair_details = [
        [_pair_details(detection.box, ground_truth) for ground_truth in ground_truths]
        for detection in detections
    ]
    pred_neighbors = [
        [
            gt_index
            for gt_index, values in enumerate(row)
            if values["overlap_over_smaller"] >= overlap_threshold
        ]
        for row in pair_details
    ]
    gt_neighbors = [
        [
            pred_index
            for pred_index, row in enumerate(pair_details)
            if row[gt_index]["overlap_over_smaller"] >= overlap_threshold
        ]
        for gt_index in range(len(ground_truths))
    ]

    duplicate_score = sum(
        sorted(
            (
                pair_details[pred_index][gt_index]["overlap_over_smaller"]
                for pred_index in neighbors
            ),
            reverse=True,
        )[1]
        for gt_index, neighbors in enumerate(gt_neighbors)
        if len(neighbors) >= 2
    )
    multi_gt_score = sum(
        sorted(
            (pair_details[pred_index][gt]["overlap_over_smaller"] for gt in neighbors),
            reverse=True,
        )[1]
        for pred_index, neighbors in enumerate(pred_neighbors)
        if len(neighbors) >= 2
    )
    result = PathologyResult(
        duplicate_gt_count=sum(len(row) >= 2 for row in gt_neighbors),
        duplicate_extra_prediction_count=sum(max(0, len(row) - 1) for row in gt_neighbors),
        multi_gt_prediction_count=sum(len(row) >= 2 for row in pred_neighbors),
        spurious_prediction_count=sum(len(row) == 0 for row in pred_neighbors),
        missed_gt_count=sum(len(row) == 0 for row in gt_neighbors),
        tangled_component_count=_tangled_components(gt_neighbors, pred_neighbors),
        maximum_gt_multiplicity=max((len(row) for row in gt_neighbors), default=0),
        maximum_pred_gt_degree=max((len(row) for row in pred_neighbors), default=0),
        count_delta=len(detections) - len(ground_truths),
        duplicate_score=float(duplicate_score),
        multi_gt_score=float(multi_gt_score),
        spurious_score=float(
            sum(
                detection.score
                for detection, neighbors in zip(detections, pred_neighbors, strict=True)
                if not neighbors
            )
        ),
    )
    detail = {
        "ground_truths": [
            {
                "gt_index": index,
                "box": box.xywh(),
                "prediction_indices": gt_neighbors[index],
            }
            for index, box in enumerate(ground_truths)
        ],
        "predictions": [
            {
                "prediction_index": pred_index,
                "box": detection.box.xywh(),
                "score": detection.score,
                "gt_indices": pred_neighbors[pred_index],
                "overlaps": [
                    {"gt_index": gt_index, **values}
                    for gt_index, values in enumerate(pair_details[pred_index])
                    if values["overlap_over_smaller"] > 0.0
                ],
            }
            for pred_index, detection in enumerate(detections)
        ],
    }
    return result, detail


def _match_counts(
    ground_truths: Sequence[Box], detections: Sequence[Detection], threshold: float = 0.5
) -> tuple[int, int, int]:
    unmatched = set(range(len(ground_truths)))
    true_positives = 0
    false_positives = 0
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        candidates = [(index, _iou(detection.box, ground_truths[index])) for index in unmatched]
        if not candidates:
            false_positives += 1
            continue
        index, value = max(candidates, key=lambda item: item[1])
        if value < threshold:
            false_positives += 1
            continue
        unmatched.remove(index)
        true_positives += 1
    return true_positives, false_positives, len(unmatched)


def _summary(
    rows: Sequence[tuple[int, int, PathologyResult]]
) -> dict[str, int | float]:
    image_count = len(rows)
    gt_count = sum(gt for gt, _pred, _result in rows)
    pred_count = sum(pred for _gt, pred, _result in rows)
    duplicate_gt = sum(result.duplicate_gt_count for _gt, _pred, result in rows)
    duplicate_extra = sum(result.duplicate_extra_prediction_count for _gt, _pred, result in rows)
    multi_gt = sum(result.multi_gt_prediction_count for _gt, _pred, result in rows)
    spurious = sum(result.spurious_prediction_count for _gt, _pred, result in rows)
    missed = sum(result.missed_gt_count for _gt, _pred, result in rows)
    affected = sum(result.affected for _gt, _pred, result in rows)
    positive_delta = sum(result.count_delta > 0 for _gt, _pred, result in rows)
    return {
        "image_count": image_count,
        "ground_truth_count": gt_count,
        "prediction_count": pred_count,
        "duplicate_gt_count": duplicate_gt,
        "duplicate_gt_rate": duplicate_gt / gt_count if gt_count else 0.0,
        "duplicate_extra_prediction_count": duplicate_extra,
        "duplicate_extra_per_gt": duplicate_extra / gt_count if gt_count else 0.0,
        "multi_gt_prediction_count": multi_gt,
        "multi_gt_prediction_rate": multi_gt / pred_count if pred_count else 0.0,
        "spurious_prediction_count": spurious,
        "spurious_prediction_rate": spurious / pred_count if pred_count else 0.0,
        "missed_gt_count": missed,
        "missed_gt_rate": missed / gt_count if gt_count else 0.0,
        "affected_image_count": affected,
        "affected_image_rate": affected / image_count if image_count else 0.0,
        "positive_count_delta_image_count": positive_delta,
        "positive_count_delta_image_rate": positive_delta / image_count if image_count else 0.0,
        "maximum_gt_multiplicity": max(
            (result.maximum_gt_multiplicity for _gt, _pred, result in rows), default=0
        ),
        "maximum_pred_gt_degree": max(
            (result.maximum_pred_gt_degree for _gt, _pred, result in rows), default=0
        ),
        "tangled_component_count": sum(
            result.tangled_component_count for _gt, _pred, result in rows
        ),
    }


def _detection_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _pathology_key(result: PathologyResult, category: str) -> tuple[float, ...]:
    if category == "duplicate":
        return (
            float(result.duplicate_extra_prediction_count),
            result.duplicate_score,
            float(result.maximum_gt_multiplicity),
        )
    if category == "multi_gt":
        return (
            float(result.multi_gt_prediction_count),
            result.multi_gt_score,
            float(result.maximum_pred_gt_degree),
        )
    if category == "spurious":
        return (float(result.spurious_prediction_count), result.spurious_score)
    if category == "miss":
        return (float(result.missed_gt_count),)
    return (
        float(
            result.duplicate_extra_prediction_count
            + result.multi_gt_prediction_count
            + result.spurious_prediction_count
            + result.missed_gt_count
        ),
        float(abs(result.count_delta)),
        result.duplicate_score + result.multi_gt_score + result.spurious_score,
    )


def _select_worst(
    rows: Sequence[dict[str, Any]], category: str, count: int
) -> list[dict[str, Any]]:
    def relevant(row: dict[str, Any]) -> bool:
        result: PathologyResult = row["pathology"]
        if category == "duplicate":
            return result.duplicate_gt_count > 0
        if category == "multi_gt":
            return result.multi_gt_prediction_count > 0
        if category == "spurious":
            return result.spurious_prediction_count > 0
        if category == "miss":
            return result.missed_gt_count > 0
        return result.affected

    selected = [row for row in rows if relevant(row)]
    return sorted(
        selected,
        key=lambda row: _pathology_key(row["pathology"], category),
        reverse=True,
    )[:count]


def _overlay(
    image: torch.Tensor,
    ground_truths: Sequence[Box],
    detections: Sequence[Detection],
    overlap_threshold: float,
) -> Image.Image:
    array = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    rendered = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(rendered)
    pair_details = [
        [_pair_details(detection.box, ground_truth) for ground_truth in ground_truths]
        for detection in detections
    ]
    pred_neighbors = [
        [
            gt_index
            for gt_index, values in enumerate(row)
            if values["overlap_over_smaller"] >= overlap_threshold
        ]
        for row in pair_details
    ]
    gt_neighbors = [
        [
            pred_index
            for pred_index, row in enumerate(pair_details)
            if row[gt_index]["overlap_over_smaller"] >= overlap_threshold
        ]
        for gt_index in range(len(ground_truths))
    ]

    for gt_index, ground_truth in enumerate(ground_truths):
        degree = len(gt_neighbors[gt_index])
        draw.rectangle(
            (ground_truth.x1, ground_truth.y1, ground_truth.x2, ground_truth.y2),
            outline=(0, 255, 0),
            width=2,
        )
        if degree != 1:
            draw.text(
                (ground_truth.x1 + 1, ground_truth.y1 + 1),
                f"G{gt_index} p={degree}",
                fill=(0, 255, 0),
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )

    for pred_index, detection in enumerate(detections):
        degree = len(pred_neighbors[pred_index])
        duplicate_member = (
            degree == 1
            and len(gt_neighbors[pred_neighbors[pred_index][0]]) >= 2
        )
        if degree == 0:
            color = (255, 0, 255)
        elif degree >= 2:
            color = (255, 165, 0)
        elif duplicate_member:
            color = (0, 255, 255)
        else:
            color = (255, 0, 0)
        draw.rectangle(
            (detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2),
            outline=color,
            width=2,
        )
        if degree != 1 or duplicate_member:
            draw.text(
                (detection.box.x1 + 1, detection.box.y2 - 11),
                f"P{pred_index} {detection.score:.2f} g={degree}",
                fill=color,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
    return rendered


def _contact_sheet(
    destination: Path,
    rows: Sequence[dict[str, Any]],
    *,
    overlap_threshold: float,
) -> None:
    if not rows:
        image = Image.new("RGB", (640, 80), "white")
        ImageDraw.Draw(image).text((10, 10), "No matching pathology samples", fill="black")
        image.save(destination)
        image.close()
        return
    columns = min(4, len(rows))
    cell_width, cell_height = 340, 385
    sheet = Image.new(
        "RGB",
        (columns * cell_width, math.ceil(len(rows) / columns) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for ordinal, row in enumerate(rows):
        result: PathologyResult = row["pathology"]
        visual = _overlay(
            row["image"],
            row["ground_truths"],
            row["detections"],
            overlap_threshold,
        ).resize((320, 320), Image.Resampling.BILINEAR)
        x = (ordinal % columns) * cell_width + 10
        y = (ordinal // columns) * cell_height + 54
        sheet.paste(visual, (x, y))
        visual.close()
        file_name = Path(row["source_file_name"]).name
        if len(file_name) > 48:
            file_name = f"{file_name[:24]}...{file_name[-21:]}"
        draw.text((x, y - 50), file_name, fill="black")
        draw.text(
            (x, y - 36),
            (
                f"dup={result.duplicate_extra_prediction_count} "
                f"multi={result.multi_gt_prediction_count} "
                f"stray={result.spurious_prediction_count} miss={result.missed_gt_count}"
            ),
            fill="black",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    sheet.close()


def evaluate(context):
    parameters = context.parameters
    split = str(parameters["split"])
    batch_size = int(parameters["batch_size"])
    workers = int(parameters["workers"])
    score_threshold = float(parameters["score_threshold"])
    nms_iou_threshold = float(parameters["nms_iou_threshold"])
    max_detections = int(parameters["max_detections"])
    pathology_overlap_threshold = float(parameters["pathology_overlap_threshold"])
    duplicate_overlap_threshold = float(parameters["duplicate_overlap_threshold"])
    worst_count = int(parameters["worst_count"])

    database = context.corpus.root / "dataset.sqlite"
    dataset = CompositeDataset(database, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = context.model.module.to(device).eval()
    strides, reg_max, _num_classes = _model_spec(model)

    nms_rows: list[dict[str, Any]] = []
    product_rows: list[dict[str, Any]] = []
    nms_pathology_summary_rows: list[tuple[int, int, PathologyResult]] = []
    product_pathology_summary_rows: list[tuple[int, int, PathologyResult]] = []
    nms_tp = nms_fp = nms_fn = 0
    product_tp = product_fp = product_fn = 0

    with torch.inference_mode():
        for images, samples in loader:
            output = model(_preprocess(images, device))
            if isinstance(output, (tuple, list)) and len(output) == 1:
                output = output[0]
            decoded = _decode_batch(
                output,
                strides=strides,
                reg_max=reg_max,
                score_threshold=score_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
            )
            for sample, detections in zip(samples, decoded, strict=True):
                product_detections = _suppress_product_duplicates(
                    detections, duplicate_overlap_threshold
                )
                nms_pathology, nms_detail = _analyze(
                    sample.ground_truths,
                    detections,
                    pathology_overlap_threshold,
                )
                product_pathology, product_detail = _analyze(
                    sample.ground_truths,
                    product_detections,
                    pathology_overlap_threshold,
                )
                tp, fp, fn = _match_counts(sample.ground_truths, detections)
                nms_tp += tp
                nms_fp += fp
                nms_fn += fn
                tp, fp, fn = _match_counts(sample.ground_truths, product_detections)
                product_tp += tp
                product_fp += fp
                product_fn += fn

                shared = {
                    "sample_id": sample.sample_id,
                    "source_domain": sample.source_domain,
                    "source_file_name": sample.source_file_name,
                    "image": sample.image,
                    "ground_truths": sample.ground_truths,
                }
                nms_rows.append(
                    {
                        **shared,
                        "detections": detections,
                        "pathology": nms_pathology,
                        "detail": nms_detail,
                    }
                )
                product_rows.append(
                    {
                        **shared,
                        "detections": product_detections,
                        "pathology": product_pathology,
                        "detail": product_detail,
                    }
                )
                nms_pathology_summary_rows.append(
                    (len(sample.ground_truths), len(detections), nms_pathology)
                )
                product_pathology_summary_rows.append(
                    (
                        len(sample.ground_truths),
                        len(product_detections),
                        product_pathology,
                    )
                )

    nms_summary = _summary(nms_pathology_summary_rows)
    product_summary = _summary(product_pathology_summary_rows)
    nms_detection = _detection_metrics(nms_tp, nms_fp, nms_fn)
    product_detection = _detection_metrics(product_tp, product_fp, product_fn)

    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_path = work_dir / "bbox-pathology-summary.json"
    per_image_path = work_dir / "bbox-pathology-per-image.jsonl"

    summary_payload = {
        "schema": "mjtensu.nanodet/bbox-pathology-summary/v1",
        "split": split,
        "model": context.model.definition.get("id"),
        "parameters": {
            "score_threshold": score_threshold,
            "nms_iou_threshold": nms_iou_threshold,
            "max_detections": max_detections,
            "pathology_overlap_metric": "intersection / min(area(prediction), area(ground_truth))",
            "pathology_overlap_threshold": pathology_overlap_threshold,
            "duplicate_overlap_threshold": duplicate_overlap_threshold,
        },
        "definitions": {
            "duplicate_gt": "GT connected to two or more predictions.",
            "multi_gt_prediction": "Prediction connected to two or more GT boxes.",
            "spurious_prediction": "Prediction connected to no GT box.",
            "missed_gt": "GT connected to no prediction.",
            "tangled_component": "Connected component containing at least two GTs and two predictions.",
            "positive_count_delta_image": "Image where prediction count exceeds GT count.",
        },
        "overlay_legend": {
            "ground_truth": "green",
            "normal_prediction": "red",
            "duplicate_member_prediction": "cyan",
            "multi_gt_prediction": "orange",
            "spurious_prediction": "magenta",
        },
        "after_nms": {
            **nms_summary,
            **nms_detection,
            "true_positive_count": nms_tp,
            "false_positive_count": nms_fp,
            "false_negative_count": nms_fn,
        },
        "after_product_postprocess": {
            **product_summary,
            **product_detection,
            "true_positive_count": product_tp,
            "false_positive_count": product_fp,
            "false_negative_count": product_fn,
        },
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with per_image_path.open("w", encoding="utf-8") as handle:
        for nms_row, product_row in zip(nms_rows, product_rows, strict=True):
            payload = {
                "sample_id": nms_row["sample_id"],
                "source_domain": nms_row["source_domain"],
                "source_file_name": nms_row["source_file_name"],
                "ground_truth_count": len(nms_row["ground_truths"]),
                "after_nms": {
                    **asdict(nms_row["pathology"]),
                    **nms_row["detail"],
                },
                "after_product_postprocess": {
                    **asdict(product_row["pathology"]),
                    **product_row["detail"],
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    artifacts: dict[str, Path] = {
        "summary_json": summary_path,
        "per_image_details": per_image_path,
    }
    for category in ("duplicate", "multi_gt", "spurious", "miss", "overall"):
        path = work_dir / f"after-nms-{category}-worst.png"
        _contact_sheet(
            path,
            _select_worst(nms_rows, category, worst_count),
            overlap_threshold=pathology_overlap_threshold,
        )
        artifacts[f"after_nms_{category}_contact_sheet"] = path

    product_overall_path = work_dir / "after-product-overall-worst.png"
    _contact_sheet(
        product_overall_path,
        _select_worst(product_rows, "overall", worst_count),
        overlap_threshold=pathology_overlap_threshold,
    )
    artifacts["after_product_overall_contact_sheet"] = product_overall_path

    metrics: dict[str, int | float] = {
        "precision_after_nms": nms_detection["precision"],
        "recall_after_nms": nms_detection["recall"],
        "f1_after_nms": nms_detection["f1"],
        "duplicate_gt_rate_after_nms": float(nms_summary["duplicate_gt_rate"]),
        "multi_gt_prediction_rate_after_nms": float(
            nms_summary["multi_gt_prediction_rate"]
        ),
        "spurious_prediction_rate_after_nms": float(
            nms_summary["spurious_prediction_rate"]
        ),
        "missed_gt_rate_after_nms": float(nms_summary["missed_gt_rate"]),
        "affected_image_rate_after_nms": float(nms_summary["affected_image_rate"]),
        "positive_count_delta_image_rate_after_nms": float(
            nms_summary["positive_count_delta_image_rate"]
        ),
        "precision_after_product": product_detection["precision"],
        "recall_after_product": product_detection["recall"],
        "f1_after_product": product_detection["f1"],
        "duplicate_gt_rate_after_product": float(product_summary["duplicate_gt_rate"]),
        "multi_gt_prediction_rate_after_product": float(
            product_summary["multi_gt_prediction_rate"]
        ),
        "spurious_prediction_rate_after_product": float(
            product_summary["spurious_prediction_rate"]
        ),
        "missed_gt_rate_after_product": float(product_summary["missed_gt_rate"]),
        "affected_image_rate_after_product": float(
            product_summary["affected_image_rate"]
        ),
        "positive_count_delta_image_rate_after_product": float(
            product_summary["positive_count_delta_image_rate"]
        ),
        "ground_truth_count": int(nms_summary["ground_truth_count"]),
        "prediction_count_after_nms": int(nms_summary["prediction_count"]),
        "prediction_count_after_product": int(product_summary["prediction_count"]),
    }
    return EvaluationCandidate(metrics=metrics, artifacts=artifacts)
