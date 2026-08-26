from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch


TOPK = 13


@dataclass(frozen=True)
class MatchResult:
    matched_pred_ious: torch.Tensor
    matched_gt_inds: torch.Tensor
    valid_mask: torch.Tensor


def _finish_matching_original(
    matching_matrix: torch.Tensor,
    cost: torch.Tensor,
    pairwise_ious: torch.Tensor,
    valid_mask: torch.Tensor,
) -> MatchResult:
    prior_match_gt_mask = matching_matrix.sum(1) > 1
    if prior_match_gt_mask.sum() > 0:
        _, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
        matching_matrix[prior_match_gt_mask, :] *= 0.0
        matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0

    fg_mask_inboxes = matching_matrix.sum(1) > 0.0
    valid_mask[valid_mask.clone()] = fg_mask_inboxes

    matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
    matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
    return MatchResult(matched_pred_ious, matched_gt_inds, valid_mask)


def _finish_matching_vectorized(
    matching_matrix: torch.Tensor,
    cost: torch.Tensor,
    pairwise_ious: torch.Tensor,
    valid_mask: torch.Tensor,
) -> MatchResult:
    prior_match_gt_mask = matching_matrix.sum(1) > 1
    cost_argmin = torch.argmin(cost, dim=1)
    matching_matrix[prior_match_gt_mask, :] = 0.0
    matching_matrix[
        prior_match_gt_mask, cost_argmin[prior_match_gt_mask]
    ] = 1.0

    fg_mask_inboxes = matching_matrix.sum(1) > 0.0
    valid_mask[valid_mask.clone()] = fg_mask_inboxes

    matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
    matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
    return MatchResult(matched_pred_ious, matched_gt_inds, valid_mask)


def original_matching(
    cost: torch.Tensor,
    pairwise_ious: torch.Tensor,
    valid_mask: torch.Tensor,
) -> MatchResult:
    num_gt = cost.size(1)
    matching_matrix = torch.zeros_like(cost)
    candidate_topk = min(TOPK, pairwise_ious.size(0))
    topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(
            cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
        )
        matching_matrix[:, gt_idx][pos_idx] = 1.0

    return _finish_matching_original(
        matching_matrix, cost, pairwise_ious, valid_mask
    )


def vectorized_matching(
    cost: torch.Tensor,
    pairwise_ious: torch.Tensor,
    valid_mask: torch.Tensor,
) -> MatchResult:
    matching_matrix = torch.zeros_like(cost)
    candidate_topk = min(TOPK, pairwise_ious.size(0))
    topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

    _, candidate_pos_idx = torch.topk(
        cost, k=candidate_topk, dim=0, largest=False
    )
    candidate_rank = torch.arange(
        candidate_topk, device=cost.device
    ).unsqueeze(1)
    selected_mask = candidate_rank < dynamic_ks.unsqueeze(0)
    matching_matrix.scatter_(
        0,
        candidate_pos_idx,
        selected_mask.to(dtype=matching_matrix.dtype),
    )

    return _finish_matching_vectorized(
        matching_matrix, cost, pairwise_ious, valid_mask
    )


def make_case(
    *,
    device: torch.device,
    num_priors: int,
    num_gt: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Continuous random values plus deterministic perturbations make tied costs
    # vanishingly unlikely. torch.topk does not guarantee stable tie ordering,
    # so exact equivalence is asserted on non-tied inputs.
    cost = torch.rand(
        (num_priors, num_gt), device=device, generator=generator
    )
    row_offset = torch.arange(num_priors, device=device).unsqueeze(1) * 1e-8
    col_offset = torch.arange(num_gt, device=device).unsqueeze(0) * 1e-10
    cost = cost + row_offset + col_offset

    pairwise_ious = torch.rand(
        (num_priors, num_gt), device=device, generator=generator
    )

    outer_size = num_priors + max(17, num_priors // 5)
    valid_mask = torch.zeros(outer_size, dtype=torch.bool, device=device)
    true_positions = torch.randperm(
        outer_size, device=device, generator=generator
    )[:num_priors]
    valid_mask[true_positions] = True
    return cost, pairwise_ious, valid_mask


def assert_equivalent(device: torch.device) -> None:
    cases = [
        (20, 1),
        (64, 5),
        (256, 20),
        (2025, 100),
    ]

    for seed, (num_priors, num_gt) in enumerate(cases, start=100):
        cost, pairwise_ious, valid_mask = make_case(
            device=device,
            num_priors=num_priors,
            num_gt=num_gt,
            seed=seed,
        )
        original = original_matching(
            cost.clone(), pairwise_ious.clone(), valid_mask.clone()
        )
        vectorized = vectorized_matching(
            cost.clone(), pairwise_ious.clone(), valid_mask.clone()
        )

        torch.testing.assert_close(
            vectorized.matched_pred_ious,
            original.matched_pred_ious,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            vectorized.matched_gt_inds,
            original.matched_gt_inds,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            vectorized.valid_mask,
            original.valid_mask,
            rtol=0,
            atol=0,
        )

        print(
            f"equivalent: device={device.type} priors={num_priors} gt={num_gt}"
        )


def benchmark(device: torch.device, repeats: int) -> None:
    if device.type != "cuda":
        return

    cost, pairwise_ious, valid_mask = make_case(
        device=device,
        num_priors=2025,
        num_gt=100,
        seed=999,
    )

    def run(function) -> float:
        for _ in range(5):
            function(cost, pairwise_ious, valid_mask.clone())
        torch.cuda.synchronize()

        started = time.perf_counter()
        for _ in range(repeats):
            function(cost, pairwise_ious, valid_mask.clone())
        torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000 / repeats

    original_ms = run(original_matching)
    vectorized_ms = run(vectorized_matching)

    print(f"original_ms_per_call:   {original_ms:.3f}")
    print(f"vectorized_ms_per_call: {vectorized_ms:.3f}")
    print(f"speedup:                {original_ms / vectorized_ms:.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_equivalent(torch.device("cpu"))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        assert_equivalent(device)
        benchmark(device, args.repeats)
    else:
        print("CUDA unavailable; skipped CUDA equivalence and benchmark")


if __name__ == "__main__":
    main()
