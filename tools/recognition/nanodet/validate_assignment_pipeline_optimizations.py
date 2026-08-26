from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate exact or tolerance-bounded equivalence of NanoDet "
            "assignment-pipeline optimizations and run CUDA microbenchmarks."
        )
    )
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def original_inside_mask(
    priors: torch.Tensor, gt_bboxes: torch.Tensor
) -> torch.Tensor:
    prior_center = priors[:, :2]
    lt_ = prior_center[:, None] - gt_bboxes[:, :2]
    rb_ = gt_bboxes[:, 2:] - prior_center[:, None]
    deltas = torch.cat([lt_, rb_], dim=-1)
    is_in_gts = deltas.min(dim=-1).values > 0
    return is_in_gts.sum(dim=1) > 0


def optimized_inside_mask(
    priors: torch.Tensor, gt_bboxes: torch.Tensor
) -> torch.Tensor:
    prior_center = priors[:, :2]
    lt_ = prior_center[:, None] - gt_bboxes[:, :2]
    rb_ = gt_bboxes[:, 2:] - prior_center[:, None]
    return ((lt_.amin(dim=-1) > 0) & (rb_.amin(dim=-1) > 0)).any(dim=1)


def original_cls_cost(
    pred_scores: torch.Tensor,
    gt_labels: torch.Tensor,
    pairwise_ious: torch.Tensor,
) -> torch.Tensor:
    num_valid = pred_scores.size(0)
    num_gt = gt_labels.size(0)
    gt_onehot = (
        F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
        .float()
        .unsqueeze(0)
        .repeat(num_valid, 1, 1)
    )
    repeated_scores = pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
    soft_label = gt_onehot * pairwise_ious[..., None]
    scale_factor = soft_label - repeated_scores.sigmoid()
    return (
        F.binary_cross_entropy_with_logits(
            repeated_scores, soft_label, reduction="none"
        )
        * scale_factor.abs().pow(2.0)
    ).sum(dim=-1)


def optimized_cls_cost(
    pred_scores: torch.Tensor,
    gt_labels: torch.Tensor,
    pairwise_ious: torch.Tensor,
) -> torch.Tensor:
    num_valid = pred_scores.size(0)
    num_gt = gt_labels.size(0)
    gt_onehot = (
        F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
        .float()
        .unsqueeze(0)
        .expand(num_valid, -1, -1)
    )
    expanded_scores = pred_scores.unsqueeze(1).expand(-1, num_gt, -1)
    soft_label = gt_onehot * pairwise_ious[..., None]
    scale_factor = soft_label - expanded_scores.sigmoid()
    return (
        F.binary_cross_entropy_with_logits(
            expanded_scores, soft_label, reduction="none"
        )
        * scale_factor.abs().pow(2.0)
    ).sum(dim=-1)


def original_conflict_resolution(
    matching_matrix: torch.Tensor, cost: torch.Tensor
) -> torch.Tensor:
    result = matching_matrix.clone()
    conflict_mask = result.sum(1) > 1
    if conflict_mask.sum() > 0:
        _, cost_argmin = torch.min(cost[conflict_mask, :], dim=1)
        result[conflict_mask, :] *= 0.0
        result[conflict_mask, cost_argmin] = 1.0
    return result


def optimized_conflict_resolution(
    matching_matrix: torch.Tensor, cost: torch.Tensor
) -> torch.Tensor:
    result = matching_matrix.clone()
    conflict_mask = result.sum(1) > 1
    conflict_cost = cost[conflict_mask, :]
    cost_argmin = torch.argmin(conflict_cost, dim=1)
    result[conflict_mask, :] = 0.0
    result[conflict_mask, cost_argmin] = 1.0
    return result


def make_numpy_targets(
    batch_size: int, max_gt: int, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, max_gt + 1, size=batch_size)
    boxes: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for count in counts:
        xy1 = rng.random((int(count), 2), dtype=np.float32) * 200.0
        wh = rng.random((int(count), 2), dtype=np.float32) * 100.0 + 1.0
        boxes.append(np.concatenate([xy1, xy1 + wh], axis=1).astype(np.float32))
        labels.append(np.zeros((int(count),), dtype=np.int64))
    return boxes, labels


def original_target_transfer(
    boxes: list[np.ndarray], labels: list[np.ndarray], device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    device_boxes = [torch.from_numpy(value).to(device) for value in boxes]
    device_labels = [torch.from_numpy(value).to(device) for value in labels]
    return device_boxes, device_labels


def packed_target_transfer(
    boxes: list[np.ndarray], labels: list[np.ndarray], device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    counts = [value.shape[0] for value in boxes]
    total = sum(counts)
    if total > 0:
        packed_boxes = torch.from_numpy(np.concatenate(boxes, axis=0)).to(device)
        packed_labels = torch.from_numpy(np.concatenate(labels, axis=0)).to(device)
    else:
        packed_boxes = torch.empty((0, 4), dtype=torch.float32, device=device)
        packed_labels = torch.empty((0,), dtype=torch.long, device=device)
    return list(packed_boxes.split(counts)), list(packed_labels.split(counts))


def assert_equivalent(device: torch.device) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(7001)

    priors = torch.rand((2025, 4), generator=generator, device=device) * 320.0
    xy1 = torch.rand((100, 2), generator=generator, device=device) * 200.0
    wh = torch.rand((100, 2), generator=generator, device=device) * 100.0 + 1.0
    gt_bboxes = torch.cat([xy1, xy1 + wh], dim=1)
    torch.testing.assert_close(
        optimized_inside_mask(priors, gt_bboxes),
        original_inside_mask(priors, gt_bboxes),
        rtol=0,
        atol=0,
    )

    pred_scores = torch.randn((2025, 1), generator=generator, device=device)
    gt_labels = torch.zeros((100,), dtype=torch.long, device=device)
    pairwise_ious = torch.rand(
        (2025, 100), generator=generator, device=device
    )
    torch.testing.assert_close(
        optimized_cls_cost(pred_scores, gt_labels, pairwise_ious),
        original_cls_cost(pred_scores, gt_labels, pairwise_ious),
        rtol=1e-6,
        atol=1e-7,
    )

    matching = torch.zeros((2025, 100), device=device)
    selected_rows = torch.randperm(2025, generator=generator, device=device)[:300]
    first_cols = torch.randint(
        0, 100, (300,), generator=generator, device=device
    )
    second_cols = (first_cols + 1) % 100
    matching[selected_rows, first_cols] = 1.0
    matching[selected_rows[:150], second_cols[:150]] = 1.0
    cost = torch.rand((2025, 100), generator=generator, device=device)
    torch.testing.assert_close(
        optimized_conflict_resolution(matching, cost),
        original_conflict_resolution(matching, cost),
        rtol=0,
        atol=0,
    )

    boxes, labels = make_numpy_targets(batch_size=96, max_gt=118, seed=7002)
    original_boxes, original_labels = original_target_transfer(boxes, labels, device)
    packed_boxes, packed_labels = packed_target_transfer(boxes, labels, device)
    for original, packed in zip(original_boxes, packed_boxes):
        torch.testing.assert_close(packed, original, rtol=0, atol=0)
    for original, packed in zip(original_labels, packed_labels):
        torch.testing.assert_close(packed, original, rtol=0, atol=0)

    value = torch.tensor(0.25, device=device)
    original_factor = max(value.item(), 1.0)
    optimized_factor = value.clamp_min(1.0)
    torch.testing.assert_close(
        optimized_factor,
        torch.tensor(original_factor, device=device),
        rtol=0,
        atol=0,
    )

    print(f"equivalent: device={device}")


def benchmark(
    name: str,
    original,
    optimized,
    device: torch.device,
    repeats: int,
) -> None:
    for _ in range(5):
        original()
        optimized()
    synchronize(device)

    start = time.perf_counter()
    for _ in range(repeats):
        original()
    synchronize(device)
    original_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(repeats):
        optimized()
    synchronize(device)
    optimized_seconds = time.perf_counter() - start

    original_ms = original_seconds * 1000.0 / repeats
    optimized_ms = optimized_seconds * 1000.0 / repeats
    print(f"{name}_original_ms:  {original_ms:.3f}")
    print(f"{name}_optimized_ms: {optimized_ms:.3f}")
    print(f"{name}_speedup:       {original_ms / optimized_ms:.2f}x")


def main() -> None:
    args = parse_args()
    assert_equivalent(torch.device("cpu"))

    if not torch.cuda.is_available():
        print("CUDA unavailable; skipped CUDA equivalence and benchmarks")
        return

    device = torch.device("cuda")
    assert_equivalent(device)

    boxes, labels = make_numpy_targets(batch_size=96, max_gt=118, seed=8001)
    benchmark(
        "target_transfer",
        lambda: original_target_transfer(boxes, labels, device),
        lambda: packed_target_transfer(boxes, labels, device),
        device,
        args.repeats,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(8002)
    pred_scores = torch.randn((2025, 1), generator=generator, device=device)
    gt_labels = torch.zeros((100,), dtype=torch.long, device=device)
    pairwise_ious = torch.rand(
        (2025, 100), generator=generator, device=device
    )
    benchmark(
        "classification_cost",
        lambda: original_cls_cost(pred_scores, gt_labels, pairwise_ious),
        lambda: optimized_cls_cost(pred_scores, gt_labels, pairwise_ious),
        device,
        args.repeats,
    )


if __name__ == "__main__":
    main()
