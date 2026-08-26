from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch NanoDet v1.0.0 target assignment to batch ground-truth "
            "host-to-device transfers, remove repeated tensor copies, and "
            "avoid remaining host-visible CUDA scalar synchronizations."
        )
    )
    parser.add_argument(
        "nanodet_root",
        type=Path,
        help="Path to the NanoDet repository root.",
    )
    return parser.parse_args()


def replace_once(text: str, old: str, new: str, description: str) -> str:
    if new in text:
        print(f"{description}: already applied")
        return text
    if text.count(old) != 1:
        raise RuntimeError(
            f"Expected exactly one source block for {description}; "
            "refusing to patch an unexpected source layout."
        )
    print(f"{description}: patched")
    return text.replace(old, new, 1)


def patch_assigner(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    old_inside_gt = '''        prior_center = priors[:, :2]
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0
        valid_mask = is_in_gts.sum(dim=1) > 0
'''
    new_inside_gt = '''        prior_center = priors[:, :2]
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        # Avoid materializing a [num_priors, num_gt, 4] concatenated tensor.
        # A prior is inside a ground-truth box only when both left/top and
        # right/bottom deltas are strictly positive.
        valid_mask = (
            (lt_.amin(dim=-1) > 0) & (rb_.amin(dim=-1) > 0)
        ).any(dim=1)
        del lt_, rb_
'''
    text = replace_once(
        text,
        old_inside_gt,
        new_inside_gt,
        "inside-ground-truth temporary allocation removal",
    )

    old_repeat = '''        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
'''
    new_repeat = '''        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .expand(num_valid, -1, -1)
        )
        valid_pred_scores = valid_pred_scores.unsqueeze(1).expand(
            -1, num_gt, -1
        )
'''
    text = replace_once(
        text,
        old_repeat,
        new_repeat,
        "assignment broadcast instead of repeat copies",
    )

    old_conflict = '''        prior_match_gt_mask = matching_matrix.sum(1) > 1
        cost_argmin = torch.argmin(cost, dim=1)
        matching_matrix[prior_match_gt_mask, :] = 0.0
        matching_matrix[
            prior_match_gt_mask, cost_argmin[prior_match_gt_mask]
        ] = 1.0
        del cost_argmin
'''
    new_conflict = '''        prior_match_gt_mask = matching_matrix.sum(1) > 1
        conflict_cost = cost[prior_match_gt_mask, :]
        cost_argmin = torch.argmin(conflict_cost, dim=1)
        matching_matrix[prior_match_gt_mask, :] = 0.0
        matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        del conflict_cost, cost_argmin
'''
    text = replace_once(
        text,
        old_conflict,
        new_conflict,
        "conflict-only cost argmin",
    )

    path.write_text(text, encoding="utf-8", newline="\n")
    py_compile.compile(str(path), doraise=True)


def patch_head(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    old_pack_position = '''        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        if aux_preds is not None:
'''
    new_pack_position = '''        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        # The collate function preserves variable-length NumPy arrays as a
        # Python list. Pack the whole batch before one host-to-device transfer,
        # then expose per-image tensor views for the existing assignment loop.
        gt_counts = [boxes.shape[0] for boxes in gt_bboxes]
        total_gt = sum(gt_counts)
        if total_gt > 0:
            packed_gt_bboxes = torch.from_numpy(
                np.concatenate(gt_bboxes, axis=0)
            ).to(device=device, dtype=decoded_bboxes.dtype)
            packed_gt_labels = torch.from_numpy(
                np.concatenate(gt_labels, axis=0)
            ).to(device=device)
        else:
            packed_gt_bboxes = decoded_bboxes.new_empty((0, 4))
            packed_gt_labels = torch.empty(
                (0,), device=device, dtype=torch.long
            )
        gt_bboxes = packed_gt_bboxes.split(gt_counts)
        gt_labels = packed_gt_labels.split(gt_counts)

        if aux_preds is not None:
'''
    text = replace_once(
        text,
        old_pack_position,
        new_pack_position,
        "batched ground-truth host-to-device transfer",
    )

    old_target_conversion = '''        device = center_priors.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)

        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = torch.from_numpy(gt_bboxes_ignore).to(device)
            gt_bboxes_ignore = gt_bboxes_ignore.to(decoded_bboxes.dtype)
'''
    new_target_conversion = '''        device = center_priors.device
        if not torch.is_tensor(gt_bboxes):
            gt_bboxes = torch.from_numpy(gt_bboxes)
        if not torch.is_tensor(gt_labels):
            gt_labels = torch.from_numpy(gt_labels)
        gt_bboxes = gt_bboxes.to(
            device=device, dtype=decoded_bboxes.dtype, non_blocking=True
        )
        gt_labels = gt_labels.to(device=device, non_blocking=True)

        if gt_bboxes_ignore is not None:
            if not torch.is_tensor(gt_bboxes_ignore):
                gt_bboxes_ignore = torch.from_numpy(gt_bboxes_ignore)
            gt_bboxes_ignore = gt_bboxes_ignore.to(
                device=device, dtype=decoded_bboxes.dtype, non_blocking=True
            )
'''
    text = replace_once(
        text,
        old_target_conversion,
        new_target_conversion,
        "tensor-aware per-image target conversion",
    )

    old_num_samples = '''        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0
        )
'''
    new_num_samples = '''        num_total_samples = reduce_mean(
            cls_preds.new_tensor(float(sum(num_pos)))
        ).clamp_min(1.0)
'''
    text = replace_once(
        text,
        old_num_samples,
        new_num_samples,
        "classification average-factor CUDA synchronization removal",
    )

    old_bbox_factor = '''            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)
'''
    new_bbox_factor = '''            bbox_avg_factor = reduce_mean(
                weight_targets.sum()
            ).clamp_min(1.0)
'''
    text = replace_once(
        text,
        old_bbox_factor,
        new_bbox_factor,
        "bbox average-factor CUDA synchronization removal",
    )

    path.write_text(text, encoding="utf-8", newline="\n")
    py_compile.compile(str(path), doraise=True)


def main() -> None:
    args = parse_args()
    root = args.nanodet_root.resolve()
    assigner_path = root / "nanodet" / "model" / "head" / "assigner" / "dsl_assigner.py"
    head_path = root / "nanodet" / "model" / "head" / "nanodet_plus_head.py"

    if not assigner_path.is_file():
        raise FileNotFoundError(assigner_path)
    if not head_path.is_file():
        raise FileNotFoundError(head_path)

    patch_assigner(assigner_path)
    patch_head(head_path)

    print(
        "NanoDet assignment-pipeline optimization patch complete:\n"
        "  ground-truth H2D copies: packed per batch\n"
        "  assignment repeat copies: replaced by broadcast views\n"
        "  host-visible avg-factor scalars: removed\n"
        "  large inside-box concatenation: removed\n"
        "  conflict argmin: restricted to conflicting priors"
    )


if __name__ == "__main__":
    main()
