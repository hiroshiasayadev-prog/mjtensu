from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch NanoDet v1.0.0 DSLA dynamic-k matching to replace the "
            "per-ground-truth top-k loop and CUDA .item() synchronizations "
            "with one batched top-k plus a rank mask."
        )
    )
    parser.add_argument(
        "nanodet_root",
        type=Path,
        help="Path to the NanoDet repository root containing nanodet/ and tools/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.nanodet_root.resolve()
    path = root / "nanodet" / "model" / "head" / "assigner" / "dsl_assigner.py"
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")

    old = '''        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
'''

    new = '''        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        # Select the maximum candidate set for all ground truths in one launch.
        # Each column then keeps only its first dynamic_k ranked candidates.
        # This removes one CUDA-synchronizing Tensor.item() and one top-k launch
        # per ground-truth instance.
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

        del (
            topk_ious,
            dynamic_ks,
            candidate_pos_idx,
            candidate_rank,
            selected_mask,
        )

        # Resolve multiply assigned priors without a host-side tensor condition.
        # The original `if prior_match_gt_mask.sum() > 0` synchronizes CUDA for
        # every assignment call. Indexing with an empty mask is already a no-op.
        prior_match_gt_mask = matching_matrix.sum(1) > 1
        cost_argmin = torch.argmin(cost, dim=1)
        matching_matrix[prior_match_gt_mask, :] = 0.0
        matching_matrix[
            prior_match_gt_mask, cost_argmin[prior_match_gt_mask]
        ] = 1.0
        del cost_argmin
'''

    if new in text:
        result = "already_applied"
    elif text.count(old) == 1:
        path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
        result = "patched"
    else:
        raise RuntimeError(
            "Expected the pinned NanoDet v1.0.0 dynamic_k_matching block exactly "
            "once; refusing to patch an unexpected source layout."
        )

    py_compile.compile(str(path), doraise=True)
    print(
        "NanoDet dynamic-k vectorization patch complete:\n"
        f"  result: {result}\n"
        f"  file: {path}\n"
        "  per-GT Tensor.item(): removed\n"
        "  per-GT torch.topk: replaced by one batched torch.topk\n"
        "  host-side conflict condition: replaced by branchless indexing"
    )


if __name__ == "__main__":
    main()
