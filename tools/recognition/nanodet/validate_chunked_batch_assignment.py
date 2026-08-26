from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from nanodet.model.head.assigner.dsl_assigner import DynamicSoftLabelAssigner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the existing per-image DSLA path with the patched "
            "single-class chunked batch-assignment path."
        )
    )
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--num-priors", type=int, default=3549)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--chunks", default="4,8,16")
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_counts(
    annotation_path: Path | None,
    batch_size: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    if annotation_path is None:
        # Deliberately cover empty, ordinary hand-sized, dense, and maximum-like
        # images when no dataset annotation file is supplied.
        population = np.asarray(
            [0, 1, 5, 14, 20, 32, 48, 64, 80, 100, 118], dtype=np.int64
        )
    else:
        with annotation_path.open(encoding="utf-8") as handle:
            coco = json.load(handle)
        counts_by_image: Counter[int] = Counter()
        for annotation in coco["annotations"]:
            counts_by_image[int(annotation["image_id"])] += 1
        population = np.asarray(
            [
                counts_by_image.get(int(image["id"]), 0)
                for image in coco["images"]
            ],
            dtype=np.int64,
        )
    return [
        int(value)
        for value in rng.choice(population, size=batch_size, replace=True)
    ]


def make_case(
    *,
    device: torch.device,
    batch_size: int,
    num_priors: int,
    image_size: int,
    counts: list[int],
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    centers = torch.rand(
        (batch_size, num_priors, 2),
        generator=generator,
        device=device,
    ) * float(image_size)
    strides = torch.full(
        (batch_size, num_priors, 2),
        8.0,
        dtype=torch.float32,
        device=device,
    )
    priors = torch.cat([centers, strides], dim=2)
    pred_scores = torch.randn(
        (batch_size, num_priors, 1),
        generator=generator,
        device=device,
    )

    decoded_half_size = (
        torch.rand(
            (batch_size, num_priors, 2),
            generator=generator,
            device=device,
        )
        * 20.0
        + 2.0
    )
    decoded_bboxes = torch.cat(
        [centers - decoded_half_size, centers + decoded_half_size], dim=2
    )

    gt_bboxes: list[torch.Tensor] = []
    gt_labels: list[torch.Tensor] = []
    for count in counts:
        if count == 0:
            gt_bboxes.append(
                torch.empty((0, 4), dtype=torch.float32, device=device)
            )
            gt_labels.append(
                torch.empty((0,), dtype=torch.long, device=device)
            )
            continue
        xy1 = torch.rand(
            (count, 2), generator=generator, device=device
        ) * (float(image_size) * 0.85)
        wh = (
            torch.rand((count, 2), generator=generator, device=device)
            * (float(image_size) * 0.20)
            + 2.0
        )
        gt_bboxes.append(torch.cat([xy1, xy1 + wh], dim=1))
        gt_labels.append(torch.zeros((count,), dtype=torch.long, device=device))

    return pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels


def pad_chunk(
    gt_bboxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(gt_bboxes)
    max_gt = max((value.size(0) for value in gt_bboxes), default=0)
    device = gt_bboxes[0].device
    dtype = gt_bboxes[0].dtype
    if max_gt == 0:
        return (
            torch.empty((batch_size, 0, 4), device=device, dtype=dtype),
            torch.empty((batch_size, 0), device=device, dtype=torch.long),
            torch.empty((batch_size, 0), device=device, dtype=torch.bool),
        )

    counts = torch.tensor(
        [value.size(0) for value in gt_bboxes],
        device=device,
        dtype=torch.long,
    )
    offsets_list: list[int] = []
    running = 0
    for value in gt_bboxes:
        offsets_list.append(running)
        running += value.size(0)
    offsets = torch.tensor(offsets_list, device=device, dtype=torch.long)
    positions = torch.arange(max_gt, device=device).unsqueeze(0)
    valid_mask = positions < counts.unsqueeze(1)
    indices = offsets.unsqueeze(1) + positions
    indices = indices.masked_fill(~valid_mask, 0)

    packed_bboxes = torch.cat(gt_bboxes, dim=0)
    packed_labels = torch.cat(gt_labels, dim=0)
    padded_bboxes = packed_bboxes[indices]
    padded_labels = packed_labels[indices]
    padded_bboxes = torch.where(
        valid_mask.unsqueeze(2), padded_bboxes, torch.zeros_like(padded_bboxes)
    )
    padded_labels = torch.where(
        valid_mask, padded_labels, torch.zeros_like(padded_labels)
    )
    return padded_bboxes, padded_labels, valid_mask


def run_legacy(
    assigner: DynamicSoftLabelAssigner,
    pred_scores: torch.Tensor,
    priors: torch.Tensor,
    decoded_bboxes: torch.Tensor,
    gt_bboxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_inds = []
    max_overlaps = []
    labels = []
    for index in range(pred_scores.size(0)):
        result = assigner.assign(
            pred_scores[index],
            priors[index],
            decoded_bboxes[index],
            gt_bboxes[index],
            gt_labels[index],
            None,
        )
        gt_inds.append(result.gt_inds)
        max_overlaps.append(result.max_overlaps)
        labels.append(result.labels)
    return (
        torch.stack(gt_inds, dim=0),
        torch.stack(max_overlaps, dim=0),
        torch.stack(labels, dim=0),
    )


def run_chunked(
    assigner: DynamicSoftLabelAssigner,
    pred_scores: torch.Tensor,
    priors: torch.Tensor,
    decoded_bboxes: torch.Tensor,
    gt_bboxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = [[], [], []]
    for start in range(0, pred_scores.size(0), chunk_size):
        end = min(start + chunk_size, pred_scores.size(0))
        padded_bboxes, padded_labels, valid_mask = pad_chunk(
            gt_bboxes[start:end], gt_labels[start:end]
        )
        result = assigner.assign_batch(
            pred_scores[start:end],
            priors[start:end],
            decoded_bboxes[start:end],
            padded_bboxes,
            padded_labels,
            valid_mask,
        )
        for destination, value in zip(outputs, result):
            destination.append(value)
    return tuple(torch.cat(values, dim=0) for values in outputs)


def build_targets(
    assigned_gt_inds: torch.Tensor,
    max_overlaps: torch.Tensor,
    assigned_labels: torch.Tensor,
    priors: torch.Tensor,
    gt_bboxes: list[torch.Tensor],
    reg_max: int = 7,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size, num_priors = assigned_gt_inds.shape
    labels = priors.new_full((batch_size, num_priors), 1, dtype=torch.long)
    label_scores = priors.new_zeros((batch_size, num_priors))
    label_weights = (assigned_gt_inds >= 0).to(dtype=priors.dtype)
    bbox_targets = torch.zeros_like(priors)
    dist_targets = torch.zeros_like(priors)
    num_pos = []

    for image_index in range(batch_size):
        positive_mask = assigned_gt_inds[image_index] > 0
        labels[image_index, positive_mask] = assigned_labels[
            image_index, positive_mask
        ]
        label_scores[image_index, positive_mask] = max_overlaps[
            image_index, positive_mask
        ]
        positive_gt_indices = assigned_gt_inds[
            image_index, positive_mask
        ] - 1
        positive_boxes = gt_bboxes[image_index][positive_gt_indices]
        bbox_targets[image_index, positive_mask] = positive_boxes

        positive_priors = priors[image_index, positive_mask]
        left_top = positive_priors[:, :2] - positive_boxes[:, :2]
        right_bottom = positive_boxes[:, 2:] - positive_priors[:, :2]
        distances = torch.cat([left_top, right_bottom], dim=1)
        distances = distances / positive_priors[:, 2, None]
        dist_targets[image_index, positive_mask] = distances.clamp(
            min=0, max=reg_max - 0.1
        )
        num_pos.append(positive_mask.sum())

    return (
        labels,
        label_scores,
        label_weights,
        bbox_targets,
        dist_targets,
        torch.stack(num_pos),
    )


def assert_results_equal(
    expected: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
    description: str,
) -> None:
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(
        actual[1], expected[1], rtol=1e-6, atol=1e-7, equal_nan=True
    )
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    print(f"equivalent_assignments: {description}")


def assert_equivalent(device: torch.device) -> None:
    assigner = DynamicSoftLabelAssigner(topk=13, iou_factor=3.0)
    cases = [
        ([0, 1, 5, 20], 256, 320),
        ([1, 14, 64, 100], 2025, 320),
    ]
    for case_index, (counts, num_priors, image_size) in enumerate(cases):
        values = make_case(
            device=device,
            batch_size=len(counts),
            num_priors=num_priors,
            image_size=image_size,
            counts=counts,
            seed=9100 + case_index,
        )
        pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels = values
        legacy = run_legacy(
            assigner,
            pred_scores,
            priors,
            decoded_bboxes,
            gt_bboxes,
            gt_labels,
        )
        legacy_targets = build_targets(*legacy, priors, gt_bboxes)

        for chunk_size in (1, 2, 4):
            chunked = run_chunked(
                assigner,
                pred_scores,
                priors,
                decoded_bboxes,
                gt_bboxes,
                gt_labels,
                chunk_size,
            )
            assert_results_equal(
                legacy,
                chunked,
                f"device={device.type} case={case_index} chunk={chunk_size}",
            )
            chunked_targets = build_targets(*chunked, priors, gt_bboxes)
            for target_index, (expected, actual) in enumerate(
                zip(legacy_targets, chunked_targets)
            ):
                tolerance = 0 if target_index in (0, 2, 5) else 1e-6
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=tolerance,
                    atol=tolerance,
                    equal_nan=True,
                )
            print(
                "equivalent_targets: "
                f"device={device.type} case={case_index} chunk={chunk_size}"
            )


def benchmark(
    *,
    device: torch.device,
    annotation_path: Path | None,
    image_size: int,
    num_priors: int,
    batch_size: int,
    chunks: list[int],
    repeats: int,
) -> None:
    if device.type != "cuda":
        return
    counts = make_counts(annotation_path, batch_size, seed=9200)
    values = make_case(
        device=device,
        batch_size=batch_size,
        num_priors=num_priors,
        image_size=image_size,
        counts=counts,
        seed=9201,
    )
    pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels = values
    assigner = DynamicSoftLabelAssigner(topk=13, iou_factor=3.0)

    # Validate every requested chunk size on the exact benchmark dimensions
    # and sampled dataset GT distribution before reporting performance. This
    # exercises full chunks and the final remainder chunk at batch size 96.
    legacy = run_legacy(
        assigner,
        pred_scores,
        priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
    )
    legacy_targets = build_targets(*legacy, priors, gt_bboxes)
    for chunk_size in chunks:
        chunked = run_chunked(
            assigner,
            pred_scores,
            priors,
            decoded_bboxes,
            gt_bboxes,
            gt_labels,
            chunk_size,
        )
        assert_results_equal(
            legacy,
            chunked,
            (
                f"device={device.type} actual_batch={batch_size} "
                f"priors={num_priors} chunk={chunk_size}"
            ),
        )
        chunked_targets = build_targets(*chunked, priors, gt_bboxes)
        for target_index, (expected, actual) in enumerate(
            zip(legacy_targets, chunked_targets)
        ):
            tolerance = 0 if target_index in (0, 2, 5) else 1e-6
            torch.testing.assert_close(
                actual,
                expected,
                rtol=tolerance,
                atol=tolerance,
                equal_nan=True,
            )
        print(
            "equivalent_targets: "
            f"device={device.type} actual_batch={batch_size} "
            f"priors={num_priors} chunk={chunk_size}"
        )

    def measure(function) -> tuple[float, float]:
        function()
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(repeats):
            function()
        synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeats
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        return elapsed_ms, peak_mib

    legacy_ms, legacy_peak = measure(
        lambda: run_legacy(
            assigner,
            pred_scores,
            priors,
            decoded_bboxes,
            gt_bboxes,
            gt_labels,
        )
    )
    print(f"legacy_ms_per_batch: {legacy_ms:.3f}")
    print(f"legacy_peak_allocated_mib: {legacy_peak:.1f}")
    print(f"sampled_gt_mean: {float(np.mean(counts)):.2f}")
    print(f"sampled_gt_max: {max(counts)}")

    for chunk_size in chunks:
        chunk_ms, chunk_peak = measure(
            lambda size=chunk_size: run_chunked(
                assigner,
                pred_scores,
                priors,
                decoded_bboxes,
                gt_bboxes,
                gt_labels,
                size,
            )
        )
        print(f"chunk_{chunk_size}_ms_per_batch: {chunk_ms:.3f}")
        print(f"chunk_{chunk_size}_speedup: {legacy_ms / chunk_ms:.2f}x")
        print(f"chunk_{chunk_size}_peak_allocated_mib: {chunk_peak:.1f}")


def main() -> None:
    args = parse_args()
    chunks = [int(value) for value in args.chunks.split(",") if value]
    assert_equivalent(torch.device("cpu"))

    if not torch.cuda.is_available():
        print("CUDA unavailable; skipped CUDA equivalence and benchmark")
        return

    device = torch.device("cuda")
    assert_equivalent(device)
    benchmark(
        device=device,
        annotation_path=args.annotations,
        image_size=args.image_size,
        num_priors=args.num_priors,
        batch_size=args.batch_size,
        chunks=chunks,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
