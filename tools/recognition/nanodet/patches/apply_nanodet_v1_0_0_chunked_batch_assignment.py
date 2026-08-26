from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch NanoDet v1.0.0 NanoDet-Plus single-class target assignment "
            "to evaluate images in configurable CUDA chunks instead of one "
            "Python call per image."
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

    marker = '''    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
'''
    insertion = '''    def assign_batch(
        self,
        pred_scores,
        priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
        gt_valid_mask,
    ):
        """Assign one padded single-class image chunk in batched CUDA ops.

        Args:
            pred_scores (Tensor): [batch, num_priors, 1].
            priors (Tensor): [batch, num_priors, 4].
            decoded_bboxes (Tensor): [batch, num_priors, 4].
            gt_bboxes (Tensor): [batch, max_gt, 4].
            gt_labels (Tensor): [batch, max_gt].
            gt_valid_mask (Tensor): [batch, max_gt].

        Returns:
            Tuple of assigned ground-truth indices, maximum overlaps, and
            assigned labels, each retaining the batch dimension.
        """
        if pred_scores.size(-1) != 1:
            raise ValueError(
                "assign_batch is intentionally restricted to single-class "
                "detectors; use the existing per-image assign path otherwise."
            )

        inf = 100000000
        batch_size, num_bboxes, _ = pred_scores.shape
        max_gt = gt_bboxes.size(1)

        assigned_gt_inds = decoded_bboxes.new_zeros(
            (batch_size, num_bboxes), dtype=torch.long
        )
        assigned_labels = decoded_bboxes.new_full(
            (batch_size, num_bboxes), -1, dtype=torch.long
        )

        if max_gt == 0 or num_bboxes == 0:
            return (
                assigned_gt_inds,
                decoded_bboxes.new_zeros((batch_size, num_bboxes)),
                assigned_labels,
            )

        prior_center = priors[..., :2]
        prior_x = prior_center[..., 0].unsqueeze(2)
        prior_y = prior_center[..., 1].unsqueeze(2)
        is_in_gts = (
            (prior_x > gt_bboxes[:, None, :, 0])
            & (prior_y > gt_bboxes[:, None, :, 1])
            & (prior_x < gt_bboxes[:, None, :, 2])
            & (prior_y < gt_bboxes[:, None, :, 3])
            & gt_valid_mask[:, None, :]
        )
        valid_prior_mask = is_in_gts.any(dim=2)
        del is_in_gts, prior_x, prior_y

        pairwise_ious = bbox_overlaps(decoded_bboxes, gt_bboxes)
        eligible_mask = (
            valid_prior_mask[:, :, None] & gt_valid_mask[:, None, :]
        )
        pairwise_ious = torch.where(
            eligible_mask,
            pairwise_ious,
            torch.zeros((), dtype=pairwise_ious.dtype, device=pairwise_ious.device),
        )

        # This project uses one detector class. Avoid materializing a
        # [batch, priors, gt, classes] one-hot cost tensor.
        expanded_scores = pred_scores[..., 0].unsqueeze(2).expand(
            -1, -1, max_gt
        )
        soft_label = pairwise_ious
        scale_factor = soft_label - expanded_scores.sigmoid()
        cls_cost = F.binary_cross_entropy_with_logits(
            expanded_scores, soft_label, reduction="none"
        ) * scale_factor.abs().pow(2.0)
        cost_matrix = (
            cls_cost - torch.log(pairwise_ious + 1e-7) * self.iou_factor
        ).masked_fill(~eligible_mask, float(inf))

        matching_matrix = self.dynamic_k_matching_batch(
            cost_matrix,
            pairwise_ious,
            gt_valid_mask,
            valid_prior_mask,
        )
        foreground_mask = matching_matrix.sum(dim=2) > 0.0
        matched_gt_inds = matching_matrix.argmax(dim=2)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(dim=2)

        assigned_gt_inds[foreground_mask] = matched_gt_inds[foreground_mask] + 1
        matched_labels = gt_labels.gather(1, matched_gt_inds)
        assigned_labels[foreground_mask] = matched_labels[foreground_mask].long()

        max_overlaps = decoded_bboxes.new_full(
            (batch_size, num_bboxes), -inf, dtype=torch.float32
        )
        max_overlaps[foreground_mask] = matched_pred_ious[foreground_mask]

        # The original early return uses zero overlaps for images with no GT
        # or no prior center inside any GT box.
        active_image_mask = (
            gt_valid_mask.any(dim=1) & valid_prior_mask.any(dim=1)
        )
        max_overlaps[~active_image_mask] = 0.0
        return assigned_gt_inds, max_overlaps, assigned_labels

    def dynamic_k_matching_batch(
        self,
        cost,
        pairwise_ious,
        gt_valid_mask,
        valid_prior_mask,
    ):
        """Batched equivalent of dynamic_k_matching for padded GT chunks."""
        batch_size, num_priors, max_gt = cost.shape
        matching_matrix = torch.zeros_like(cost)
        candidate_topk = min(self.topk, num_priors)

        topk_ious, _ = torch.topk(
            pairwise_ious, candidate_topk, dim=1
        )
        dynamic_ks = torch.clamp(topk_ious.sum(dim=1).int(), min=1)
        active_gt_mask = (
            gt_valid_mask & valid_prior_mask.any(dim=1).unsqueeze(1)
        )
        dynamic_ks = torch.where(
            active_gt_mask, dynamic_ks, torch.zeros_like(dynamic_ks)
        )

        _, candidate_pos_idx = torch.topk(
            cost, k=candidate_topk, dim=1, largest=False
        )
        candidate_rank = torch.arange(
            candidate_topk, device=cost.device
        ).view(1, candidate_topk, 1)
        selected_mask = candidate_rank < dynamic_ks.unsqueeze(1)
        matching_matrix.scatter_(
            1,
            candidate_pos_idx,
            selected_mask.to(dtype=matching_matrix.dtype),
        )

        # Resolve prior-to-multiple-GT conflicts across the flattened batch.
        conflict_mask = matching_matrix.sum(dim=2) > 1
        flat_matching = matching_matrix.reshape(batch_size * num_priors, max_gt)
        flat_cost = cost.reshape(batch_size * num_priors, max_gt)
        conflict_rows = torch.nonzero(
            conflict_mask.reshape(-1), as_tuple=False
        ).squeeze(1)
        conflict_argmin = torch.argmin(flat_cost[conflict_rows], dim=1)
        flat_matching[conflict_rows, :] = 0.0
        flat_matching[conflict_rows, conflict_argmin] = 1.0
        return flat_matching.view(batch_size, num_priors, max_gt)

'''
    if insertion in text:
        print("chunked batched assigner methods: already applied")
    elif text.count(marker) == 1:
        text = text.replace(marker, insertion + marker, 1)
        print("chunked batched assigner methods: patched")
    else:
        raise RuntimeError(
            "Expected exactly one dynamic_k_matching method marker; refusing "
            "to patch an unexpected assigner source layout."
        )

    path.write_text(text, encoding="utf-8", newline="\n")
    py_compile.compile(str(path), doraise=True)


def patch_head(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "import os\n" not in text:
        text = replace_once(
            text,
            "import math\n",
            "import math\nimport os\n",
            "assignment chunk environment import",
        )
    else:
        print("assignment chunk environment import: already applied")

    old_assignment = '''        if aux_preds is not None:
            # use auxiliary head to assign
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            aux_dis_preds = (
                self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds.detach(),
                center_priors,
                aux_decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
            )
        else:
            # use self prediction to assign
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                cls_preds.detach(),
                center_priors,
                decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
            )
'''
    new_assignment = '''        if aux_preds is not None:
            # use auxiliary head to assign
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            aux_dis_preds = (
                self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
            assign_cls_preds = aux_cls_preds.detach()
            assign_decoded_bboxes = aux_decoded_bboxes.detach()
        else:
            # use self prediction to assign
            assign_cls_preds = cls_preds.detach()
            assign_decoded_bboxes = decoded_bboxes.detach()

        assign_chunk_size = int(os.environ.get("NANODET_ASSIGN_CHUNK", "0"))
        ignore_targets_empty = all(
            value is None or value.shape[0] == 0
            for value in gt_bboxes_ignore
        )
        if (
            assign_chunk_size > 0
            and self.num_classes == 1
            and ignore_targets_empty
        ):
            batch_assign_res = self.target_assign_batch_chunks(
                assign_cls_preds,
                center_priors,
                assign_decoded_bboxes,
                packed_gt_bboxes,
                packed_gt_labels,
                gt_counts,
                assign_chunk_size,
            )
        else:
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                assign_cls_preds,
                center_priors,
                assign_decoded_bboxes,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
            )
'''
    text = replace_once(
        text,
        old_assignment,
        new_assignment,
        "chunked assignment dispatch",
    )

    old_loss_prelude = '''        num_total_samples = reduce_mean(
            cls_preds.new_tensor(float(sum(num_pos)))
        ).clamp_min(1.0)

        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        label_weights = torch.cat(label_weights, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
'''
    new_loss_prelude = '''        if torch.is_tensor(num_pos):
            num_total_samples = reduce_mean(
                num_pos.sum().to(dtype=cls_preds.dtype)
            ).clamp_min(1.0)
        else:
            num_total_samples = reduce_mean(
                cls_preds.new_tensor(float(sum(num_pos)))
            ).clamp_min(1.0)

        if torch.is_tensor(labels):
            labels = labels.reshape(-1)
            label_scores = label_scores.reshape(-1)
            label_weights = label_weights.reshape(-1)
            bbox_targets = bbox_targets.reshape(-1, 4)
        else:
            labels = torch.cat(labels, dim=0)
            label_scores = torch.cat(label_scores, dim=0)
            label_weights = torch.cat(label_weights, dim=0)
            bbox_targets = torch.cat(bbox_targets, dim=0)
'''
    text = replace_once(
        text,
        old_loss_prelude,
        new_loss_prelude,
        "batched assignment loss input",
    )

    old_dist_cat = '''            dist_targets = torch.cat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
'''
    new_dist_cat = '''            if torch.is_tensor(dist_targets):
                dist_targets = dist_targets.reshape(-1, 4)
            else:
                dist_targets = torch.cat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
'''
    text = replace_once(
        text,
        old_dist_cat,
        new_dist_cat,
        "batched distance targets",
    )

    marker = '''    @torch.no_grad()
    def target_assign_single_img(
'''
    insertion = '''    @torch.no_grad()
    def target_assign_batch_chunks(
        self,
        cls_preds,
        center_priors,
        decoded_bboxes,
        packed_gt_bboxes,
        packed_gt_labels,
        gt_counts,
        chunk_size,
    ):
        """Build assignment targets using padded CUDA chunks."""
        batch_size = cls_preds.size(0)
        chunk_size = max(1, min(int(chunk_size), batch_size))

        offsets = []
        running_offset = 0
        for count in gt_counts:
            offsets.append(running_offset)
            running_offset += count
        count_tensor = torch.tensor(
            gt_counts, device=cls_preds.device, dtype=torch.long
        )
        offset_tensor = torch.tensor(
            offsets, device=cls_preds.device, dtype=torch.long
        )

        chunk_outputs = [[] for _ in range(6)]
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk_counts = count_tensor[start:end]
            chunk_offsets = offset_tensor[start:end]
            max_gt = max(gt_counts[start:end], default=0)

            if max_gt > 0:
                positions = torch.arange(
                    max_gt, device=cls_preds.device, dtype=torch.long
                ).unsqueeze(0)
                gt_valid_mask = positions < chunk_counts.unsqueeze(1)
                gather_indices = chunk_offsets.unsqueeze(1) + positions
                gather_indices = gather_indices.masked_fill(~gt_valid_mask, 0)
                padded_gt_bboxes = packed_gt_bboxes[gather_indices]
                padded_gt_labels = packed_gt_labels[gather_indices]
                padded_gt_bboxes = torch.where(
                    gt_valid_mask.unsqueeze(2),
                    padded_gt_bboxes,
                    torch.zeros_like(padded_gt_bboxes),
                )
                padded_gt_labels = torch.where(
                    gt_valid_mask,
                    padded_gt_labels,
                    torch.zeros_like(padded_gt_labels),
                )
            else:
                chunk_batch = end - start
                padded_gt_bboxes = decoded_bboxes.new_empty(
                    (chunk_batch, 0, 4)
                )
                padded_gt_labels = torch.empty(
                    (chunk_batch, 0),
                    device=cls_preds.device,
                    dtype=torch.long,
                )
                gt_valid_mask = torch.empty(
                    (chunk_batch, 0),
                    device=cls_preds.device,
                    dtype=torch.bool,
                )

            assigned_gt_inds, max_overlaps, assigned_labels = (
                self.assigner.assign_batch(
                    cls_preds[start:end],
                    center_priors[start:end],
                    decoded_bboxes[start:end],
                    padded_gt_bboxes,
                    padded_gt_labels,
                    gt_valid_mask,
                )
            )

            num_priors = center_priors.size(1)
            positive_mask = assigned_gt_inds > 0
            labels = center_priors.new_full(
                (end - start, num_priors),
                self.num_classes,
                dtype=torch.long,
            )
            labels[positive_mask] = assigned_labels[positive_mask]
            label_scores = center_priors.new_zeros(
                (end - start, num_priors), dtype=torch.float
            )
            label_scores[positive_mask] = max_overlaps[positive_mask]
            label_weights = (assigned_gt_inds >= 0).to(
                dtype=center_priors.dtype
            )
            bbox_targets = torch.zeros_like(center_priors[start:end])
            dist_targets = torch.zeros_like(center_priors[start:end])

            if max_gt > 0:
                matched_gt_inds = (assigned_gt_inds - 1).clamp_min(0)
                matched_gt_bboxes = padded_gt_bboxes.gather(
                    1,
                    matched_gt_inds.unsqueeze(2).expand(-1, -1, 4),
                )
                bbox_targets[positive_mask] = matched_gt_bboxes[positive_mask]

                prior_centers = center_priors[start:end, :, :2]
                left_top = prior_centers - matched_gt_bboxes[..., :2]
                right_bottom = matched_gt_bboxes[..., 2:] - prior_centers
                all_dist_targets = torch.cat(
                    [left_top, right_bottom], dim=2
                ) / center_priors[start:end, :, 2, None]
                all_dist_targets = all_dist_targets.clamp(
                    min=0, max=self.reg_max - 0.1
                )
                dist_targets[positive_mask] = all_dist_targets[positive_mask]

            num_pos = positive_mask.sum(dim=1)
            outputs = (
                labels,
                label_scores,
                label_weights,
                bbox_targets,
                dist_targets,
                num_pos,
            )
            for destination, value in zip(chunk_outputs, outputs):
                destination.append(value)

        return tuple(torch.cat(values, dim=0) for values in chunk_outputs)

'''
    if insertion in text:
        print("chunked batched target builder: already applied")
    elif text.count(marker) == 1:
        text = text.replace(marker, insertion + marker, 1)
        print("chunked batched target builder: patched")
    else:
        raise RuntimeError(
            "Expected exactly one target_assign_single_img marker; refusing "
            "to patch an unexpected head source layout."
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
        "NanoDet chunked batch-assignment patch complete:\n"
        "  activation: NANODET_ASSIGN_CHUNK=<positive integer>\n"
        "  default: 0, retaining the existing per-image path\n"
        "  optimized fast path: single-class and no non-empty ignore targets\n"
        "  fallback: existing per-image assignment\n"
        "  candidate chunks for RTX 3090: 4, 8, 16"
    )


if __name__ == "__main__":
    main()
