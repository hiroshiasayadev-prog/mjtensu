from __future__ import annotations

import json
import math
import random
import sqlite3

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ShuffleNet_V2_X0_5_Weights, shufflenet_v2_x0_5


IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


class DetectionDataset(Dataset):
    def __init__(self, path, split: str):
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                "SELECT image_chw_u8, annotations_json FROM sample WHERE split=? ORDER BY sample_id",
                (split,),
            ).fetchall()
        self.rows = rows

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


def make_points(outputs, strides):
    points = []
    stride_values = []
    levels = []
    for level, (output, stride) in enumerate(zip(outputs, strides)):
        height, width = output.shape[-2:]
        ys = (torch.arange(height, device=output.device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(width, device=output.device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        level_points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        points.append(level_points)
        stride_values.append(torch.full((level_points.shape[0],), float(stride), device=output.device))
        levels.append(torch.full((level_points.shape[0],), level, dtype=torch.long, device=output.device))
    return torch.cat(points), torch.cat(stride_values), torch.cat(levels)


def size_ranges(strides):
    ranges = []
    for index, stride in enumerate(strides):
        low = 0.0 if index == 0 else float(strides[index - 1]) * 6.0
        high = 1.0e8 if index == len(strides) - 1 else float(stride) * 8.0
        ranges.append((low, high))
    return ranges


def build_targets(boxes, points, strides, levels, center_radius: float, ranges):
    count = points.shape[0]
    device = points.device
    objectness = torch.zeros(count, device=device)
    centerness = torch.zeros(count, device=device)
    regression = torch.zeros(count, 6, device=device)
    if boxes.numel() == 0:
        return objectness, centerness, regression
    boxes = boxes.to(device=device, dtype=torch.float32)
    cx, cy = boxes[:, 0], boxes[:, 1]
    widths = boxes[:, 2].clamp_min(1.0e-4)
    heights = boxes[:, 3].clamp_min(1.0e-4)
    angles = torch.deg2rad(boxes[:, 4])
    cosine, sine = torch.cos(angles), torch.sin(angles)
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
    range_tensor = torch.tensor(ranges, device=device, dtype=torch.float32)
    low = range_tensor[levels, 0][:, None]
    high = range_tensor[levels, 1][:, None]
    candidates = inside & center_inside & (max_size[None, :] >= low) & (max_size[None, :] <= high)
    areas = (widths * heights)[None, :].expand(count, -1).clone()
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
    regression[positive, 0] = (cx[gt_indices] - points[positive, 0]) / selected_stride
    regression[positive, 1] = (cy[gt_indices] - points[positive, 1]) / selected_stride
    regression[positive, 2] = torch.log(selected_width / selected_stride)
    regression[positive, 3] = torch.log(selected_height / selected_stride)
    regression[positive, 4] = torch.sin(2.0 * angles[gt_indices])
    regression[positive, 5] = torch.cos(2.0 * angles[gt_indices])
    return objectness, centerness, regression


def focal(logits, targets, alpha=0.25, gamma=2.0):
    probabilities = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * (1.0 - p_t).pow(gamma) * bce


def compute_loss(outputs, targets, model_strides, center_radius):
    predictions = torch.cat(
        [output.permute(0, 2, 3, 1).reshape(output.shape[0], -1, 8) for output in outputs],
        dim=1,
    )
    points, strides, levels = make_points(outputs, model_strides)
    ranges = size_ranges(model_strides)
    batch_size, point_count, _ = predictions.shape
    objectness_targets = torch.zeros(batch_size, point_count, device=predictions.device)
    centerness_targets = torch.zeros(batch_size, point_count, device=predictions.device)
    regression_targets = torch.zeros(batch_size, point_count, 6, device=predictions.device)
    for index, boxes in enumerate(targets):
        obj, ctr, reg = build_targets(
            boxes, points, strides, levels, center_radius, ranges
        )
        objectness_targets[index] = obj
        centerness_targets[index] = ctr
        regression_targets[index] = reg
    positive = objectness_targets > 0.5
    positive_count = positive.sum().clamp_min(1).to(predictions.dtype)
    objectness_loss = focal(predictions[..., 0], objectness_targets).sum() / positive_count
    if positive.any():
        centerness_loss = F.binary_cross_entropy_with_logits(
            predictions[..., 1][positive], centerness_targets[positive], reduction="sum"
        ) / positive_count
        weights = centerness_targets[positive].clamp_min(0.05)
        predicted = predictions[..., 2:][positive]
        target = regression_targets[positive]
        center_size = F.smooth_l1_loss(
            predicted[:, :4], target[:, :4], reduction="none", beta=0.25
        ).mean(dim=1)
        center_size_loss = (center_size * weights).sum() / weights.sum().clamp_min(1.0)
        predicted_angle = F.normalize(predicted[:, 4:6], dim=1, eps=1.0e-6)
        target_angle = F.normalize(target[:, 4:6], dim=1, eps=1.0e-6)
        angle_distance = 1.0 - (predicted_angle * target_angle).sum(dim=1)
        angle_loss = (angle_distance * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        zero = predictions.sum() * 0.0
        centerness_loss = zero
        center_size_loss = zero
        angle_loss = zero
    return objectness_loss + centerness_loss + 2.0 * center_size_loss + angle_loss


def flatten_outputs(outputs):
    return torch.cat(
        [output.permute(0, 2, 3, 1).reshape(output.shape[0], -1, 8) for output in outputs],
        dim=1,
    )


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
    return [
        (cx + x*cosine - y*sine, cy + x*sine + y*cosine)
        for x, y in ((-width/2,-height/2),(width/2,-height/2),(width/2,height/2),(-width/2,height/2))
    ]


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
    return (
        (d1*(x3-x4)-(x1-x2)*d2)/denominator,
        (d1*(y3-y4)-(y1-y2)*d2)/denominator,
    )


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
    first_polygon = corners(first)
    second_polygon = corners(second)
    intersection_area = polygon_area(clip(first_polygon, second_polygon))
    union = first[2]*first[3] + second[2]*second[3] - intersection_area
    return 0.0 if union <= 0.0 else intersection_area / union


def decode(outputs, strides, score_threshold, nms_iou_threshold, max_detections):
    predictions = flatten_outputs(outputs)
    points, stride_values, _levels = make_points(outputs, strides)
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
                float(centers[index,0].detach().cpu()),
                float(centers[index,1].detach().cpu()),
                float(sizes[index,0].detach().cpu()),
                float(sizes[index,1].detach().cpu()),
                float(angles[index].detach().cpu()),
            ))
            detections.append((float(candidate_scores[index].detach().cpu()), obb))
        detections.sort(key=lambda value: value[0], reverse=True)
        kept = []
        while detections and len(kept) < max_detections:
            winner = detections.pop(0)
            kept.append(winner)
            detections = [
                candidate for candidate in detections
                if rotated_iou(winner[1], candidate[1]) < nms_iou_threshold
            ]
        results.append(kept)
    return results


def match(detections, ground_truth, threshold):
    unmatched = set(range(len(ground_truth)))
    gt = [canonicalize(value.tolist()) for value in ground_truth]
    ious = []
    fp = 0
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
    return len(ious), fp, len(unmatched), ious


def validation_key(model, loader, *, use_amp, center_radius, score_threshold,
                   nms_iou_threshold, match_iou_threshold, max_detections):
    model.eval()
    tp = fp = fn = 0
    ious = []
    loss_total = 0.0
    batches = 0
    with torch.inference_mode():
        for images, boxes in loader:
            images = images.cuda(non_blocking=True)
            targets = [value.cuda(non_blocking=True) for value in boxes]
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = compute_loss(
                    outputs,
                    targets,
                    tuple(int(value) for value in model.strides),
                    center_radius,
                )
            loss_total += float(loss.detach().cpu())
            batches += 1
            detections = decode(
                outputs,
                tuple(int(value) for value in model.strides),
                score_threshold,
                nms_iou_threshold,
                max_detections,
            )
            for predicted, target in zip(detections, boxes):
                item_tp, item_fp, item_fn, item_ious = match(
                    predicted, target, match_iou_threshold
                )
                tp += item_tp; fp += item_fp; fn += item_fn
                ious.extend(item_ious)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    mean_loss = loss_total / max(1, batches)
    return (float(f1), float(recall), float(mean_iou), -float(mean_loss))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(context):
    parameters = context.parameters
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rotated FCOS training")
    seed_everything(int(context.seed))
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    train_dataset = DetectionDataset(context.corpus.root / "dataset.sqlite", str(parameters["train_split"]))
    val_dataset = DetectionDataset(context.corpus.root / "dataset.sqlite", str(parameters["validation_split"]))
    generator = torch.Generator().manual_seed(int(context.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(parameters["batch_size"]),
        shuffle=True,
        num_workers=int(parameters["workers"]),
        pin_memory=True,
        persistent_workers=int(parameters["workers"]) > 0,
        collate_fn=collate,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(parameters["batch_size"]),
        shuffle=False,
        num_workers=int(parameters["workers"]),
        pin_memory=True,
        persistent_workers=int(parameters["workers"]) > 0,
        collate_fn=collate,
    )
    model = context.model.to(device)
    if bool(parameters["pretrained_backbone"]):
        reference = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
        model.backbone.load_state_dict(reference.state_dict(), strict=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    epochs = int(parameters["epochs"])
    warmup_epochs = float(parameters["warmup_epochs"])
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, round(warmup_epochs * len(train_loader)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            max(0.05, (step + 1) / warmup_steps)
            if step < warmup_steps
            else 0.05 + 0.95 * 0.5 * (
                1.0 + math.cos(math.pi * min(max((step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0), 1.0))
            )
        ),
    )
    use_amp = bool(parameters["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    freeze_epochs = int(parameters["freeze_backbone_epochs"])
    patience = int(parameters["early_stop_patience"])
    best_key = (-1.0, -1.0, -1.0, -float("inf"))
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(epochs):
        frozen = epoch < freeze_epochs
        for parameter in model.backbone.parameters():
            parameter.requires_grad = not frozen
        model.train()
        if frozen:
            model.backbone.eval()
        for images, boxes in train_loader:
            images = images.to(device, non_blocking=True)
            targets = [value.to(device, non_blocking=True) for value in boxes]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = compute_loss(
                    outputs,
                    targets,
                    tuple(int(value) for value in model.strides),
                    float(parameters["center_radius"]),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        key = validation_key(
            model,
            val_loader,
            use_amp=use_amp,
            center_radius=float(parameters["center_radius"]),
            score_threshold=float(parameters["score_threshold"]),
            nms_iou_threshold=float(parameters["nms_iou_threshold"]),
            match_iou_threshold=float(parameters["match_iou_threshold"]),
            max_detections=int(parameters["max_detections"]),
        )
        if key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training produced no best model state")
    model.load_state_dict(best_state)
    return model
