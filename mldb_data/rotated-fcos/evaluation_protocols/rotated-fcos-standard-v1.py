from __future__ import annotations

import json
import math
import sqlite3

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


class DetectionDataset(Dataset):
    def __init__(self, path, split):
        with sqlite3.connect(path) as connection:
            self.rows = connection.execute(
                "SELECT image_chw_u8, annotations_json FROM sample WHERE split=? ORDER BY sample_id",
                (split,),
            ).fetchall()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_raw, annotations_raw = self.rows[index]
        image = np.frombuffer(image_raw, dtype=np.uint8).copy().reshape(3, 320, 320)
        tensor = torch.from_numpy(image).float().mul_(1.0 / 255.0)
        tensor = tensor.sub(IMAGE_MEAN).div(IMAGE_STD)
        annotations = json.loads(annotations_raw)
        boxes = torch.tensor([item["obb"] for item in annotations], dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = torch.empty((0, 5), dtype=torch.float32)
        return tensor, boxes


def collate(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def flatten_outputs(outputs):
    return torch.cat(
        [output.permute(0, 2, 3, 1).reshape(output.shape[0], -1, 8) for output in outputs],
        dim=1,
    )


def make_points(outputs, strides):
    point_chunks, stride_chunks = [], []
    for output, stride in zip(outputs, strides):
        height, width = output.shape[-2:]
        ys = (torch.arange(height, device=output.device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(width, device=output.device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        point_chunks.append(points)
        stride_chunks.append(torch.full((points.shape[0],), float(stride), device=output.device))
    return torch.cat(point_chunks), torch.cat(stride_chunks)


def normalize_angle(angle):
    value = (angle + 90.0) % 180.0 - 90.0
    return value - 180.0 if value >= 90.0 else value


def canonicalize(obb):
    cx, cy, width, height, angle = [float(value) for value in obb]
    if width > height:
        width, height = height, width
        angle += 90.0
    return (cx, cy, width, height, normalize_angle(angle))


def corners(obb):
    cx, cy, width, height, angle = obb
    rad = math.radians(angle)
    cosine, sine = math.cos(rad), math.sin(rad)
    result = []
    for local_x, local_y in ((-width/2,-height/2),(width/2,-height/2),(width/2,height/2),(-width/2,height/2)):
        result.append((cx + local_x*cosine - local_y*sine, cy + local_x*sine + local_y*cosine))
    return result


def signed_area(polygon):
    return sum(
        polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
        - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
        for i in range(len(polygon))
    ) / 2.0


def polygon_area(polygon):
    return abs(signed_area(polygon)) if len(polygon) >= 3 else 0.0


def inside(point, a, b, orientation):
    cross = (b[0]-a[0])*(point[1]-a[1]) - (b[1]-a[1])*(point[0]-a[0])
    return cross >= -1.0e-9 if orientation >= 0.0 else cross <= 1.0e-9


def intersection(p1, p2, q1, q2):
    x1,y1=p1; x2,y2=p2; x3,y3=q1; x4,y4=q2
    denominator=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if abs(denominator) < 1.0e-12:
        return p2
    d1=x1*y2-y1*x2; d2=x3*y4-y3*x4
    return ((d1*(x3-x4)-(x1-x2)*d2)/denominator, (d1*(y3-y4)-(y1-y2)*d2)/denominator)


def clip(subject, clip_polygon):
    output = list(subject)
    orientation = signed_area(clip_polygon)
    for index in range(len(clip_polygon)):
        a, b = clip_polygon[index], clip_polygon[(index + 1) % len(clip_polygon)]
        input_polygon = output
        output = []
        if not input_polygon:
            break
        previous = input_polygon[-1]
        previous_inside = inside(previous, a, b, orientation)
        for current in input_polygon:
            current_inside = inside(current, a, b, orientation)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current, a, b))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current, a, b))
            previous, previous_inside = current, current_inside
    return output


def rotated_iou(first, second):
    a, b = corners(first), corners(second)
    inter = polygon_area(clip(a, b))
    union = first[2]*first[3] + second[2]*second[3] - inter
    return 0.0 if union <= 0.0 else inter / union


def angle_error(first, second):
    delta = abs(normalize_angle(first - second))
    return min(delta, 180.0 - delta)


def decode(outputs, strides, score_threshold, nms_iou_threshold, max_detections):
    predictions = flatten_outputs(outputs)
    points, stride_values = make_points(outputs, strides)
    results = []
    for batch_index in range(predictions.shape[0]):
        item = predictions[batch_index]
        scores = torch.sigmoid(item[:, 0]) * torch.sigmoid(item[:, 1])
        indices = (scores >= score_threshold).nonzero(as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            results.append([])
            continue
        candidate_scores = scores[indices]
        if indices.numel() > 300:
            candidate_scores, order = torch.topk(candidate_scores, 300)
            indices = indices[order]
        regression = item[indices, 2:]
        candidate_points = points[indices]
        candidate_strides = stride_values[indices]
        centers = candidate_points + regression[:, :2] * candidate_strides[:, None]
        sizes = torch.exp(regression[:, 2:4].clamp(-4.0, 4.0)) * candidate_strides[:, None]
        angles = 0.5 * torch.atan2(regression[:, 4], regression[:, 5]) * 180.0 / math.pi
        detections = []
        for index in range(indices.numel()):
            obb = canonicalize((
                float(centers[index,0]), float(centers[index,1]),
                float(sizes[index,0]), float(sizes[index,1]), float(angles[index]),
            ))
            detections.append((float(candidate_scores[index]), obb))
        detections.sort(key=lambda value: value[0], reverse=True)
        kept = []
        while detections and len(kept) < max_detections:
            winner = detections.pop(0)
            kept.append(winner)
            detections = [candidate for candidate in detections if rotated_iou(winner[1], candidate[1]) < nms_iou_threshold]
        results.append(kept)
    return results


def match(detections, ground_truth, threshold):
    unmatched = set(range(len(ground_truth)))
    ious, angles = [], []
    fp = 0
    gt = [canonicalize(value.tolist()) for value in ground_truth]
    for _score, obb in detections:
        candidates = [(index, rotated_iou(obb, gt[index])) for index in unmatched]
        if not candidates:
            fp += 1
            continue
        index, iou = max(candidates, key=lambda value: value[1])
        if iou < threshold:
            fp += 1
            continue
        unmatched.remove(index)
        ious.append(iou)
        angles.append(angle_error(obb[4], gt[index][4]))
    return len(ious), fp, len(unmatched), ious, angles


def evaluate(context):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rotated FCOS evaluation")
    parameters = context.parameters
    device = torch.device("cuda")
    dataset = DetectionDataset(context.corpus.root / "dataset.sqlite", str(parameters["split"]))
    loader = DataLoader(
        dataset,
        batch_size=int(parameters["batch_size"]),
        shuffle=False,
        num_workers=int(parameters["workers"]),
        pin_memory=True,
        persistent_workers=int(parameters["workers"]) > 0,
        collate_fn=collate,
    )
    model = context.model.module.to(device).eval()
    strides = tuple(int(value) for value in model.strides)
    tp = fp = fn = 0
    all_ious, all_angles = [], []
    with torch.inference_mode():
        for images, boxes in loader:
            outputs = model(images.to(device))
            detections = decode(
                outputs, strides,
                float(parameters["score_threshold"]),
                float(parameters["nms_iou_threshold"]),
                int(parameters["max_detections"]),
            )
            for predicted, target in zip(detections, boxes):
                item_tp, item_fp, item_fn, ious, angles = match(
                    predicted, target, float(parameters["match_iou_threshold"])
                )
                tp += item_tp; fp += item_fp; fn += item_fn
                all_ious.extend(ious); all_angles.extend(angles)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "rotated_iou_mean": float(np.mean(all_ious)) if all_ious else 0.0,
        "rotated_iou_p10": float(np.percentile(all_ious, 10.0)) if all_ious else 0.0,
        "angle_error_mean_deg": float(np.mean(all_angles)) if all_angles else 0.0,
        "angle_error_p90_deg": float(np.percentile(all_angles, 90.0)) if all_angles else 0.0,
        "matched_count": int(len(all_ious)),
    }
    return EvaluationCandidate(metrics=metrics, artifacts={})
