from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__:
    from .rotated_fcos_nano import RotatedFCOSNano, decode_batch, match_detections
    from .train_rotated_fcos_nano import (
        RotatedCocoDataset,
        collate_batch,
        distribution_summary,
        resolve_device,
    )
else:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from tools.recognition.rotated_fcos_nano import (  # type: ignore[no-redef]
        RotatedFCOSNano,
        decode_batch,
        match_detections,
    )
    from tools.recognition.train_rotated_fcos_nano import (  # type: ignore[no-redef]
        RotatedCocoDataset,
        collate_batch,
        distribution_summary,
        resolve_device,
    )


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RotatedFCOSNano checkpoint and sweep score thresholds."
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "rotated_fcos_runs"
        / "rfcos_nano_s05_f64_seed42"
        / "model_best.pt",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "rotated_detector_corpus"
        / "annotations"
        / "val.json",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=64)
    parser.add_argument(
        "--head-bn-batch-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic only: keep the model in eval mode but make BatchNorm layers inside "
            "the shared FCOS head use current-batch statistics. A large metric change indicates "
            "that shared BatchNorm running statistics across FPN levels are invalid."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=None,
        help="Explicit thresholds. Defaults to 0.02..0.80 with denser sampling around 0.2-0.5.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    annotations_path = args.annotations.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not annotations_path.is_file():
        raise FileNotFoundError(annotations_path)

    device = resolve_device(str(args.device))
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    payload = torch.load(checkpoint_path, **load_kwargs)
    config = payload["model_config"]
    model = RotatedFCOSNano(
        backbone=str(config["backbone"]),
        fpn_channels=int(config["fpn_channels"]),
        head_convs=int(config["head_convs"]),
        head_normalization=str(config.get("head_normalization", "batch")),
        pretrained_backbone=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval().to(device)
    if bool(args.head_bn_batch_stats):
        for module in model.head.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.train()

    dataset = RotatedCocoDataset(annotations_path, repository_root=repository_root)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=collate_batch,
    )

    cached: list[tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]] = []
    with torch.inference_mode():
        for images, boxes, _metadata in loader:
            images = images.to(device)
            outputs = tuple(item.detach().cpu() for item in model(images))
            cached.append((outputs, [box.cpu() for box in boxes]))

    thresholds = (
        sorted(set(float(value) for value in args.thresholds))
        if args.thresholds
        else default_thresholds()
    )
    rows = [
        evaluate_threshold(
            cached,
            threshold=threshold,
            nms_iou_threshold=float(args.nms_iou_threshold),
            match_iou_threshold=float(args.match_iou_threshold),
            max_detections=int(args.max_detections),
        )
        for threshold in thresholds
    ]
    best_f1 = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    precision_qualified = [row for row in rows if row["precision"] >= 0.50]
    best_recall_with_precision = (
        max(
            precision_qualified,
            key=lambda row: (row["recall"], row["f1"], row["precision"]),
        )
        if precision_qualified
        else None
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "annotations": str(annotations_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "head_bn_batch_stats": bool(args.head_bn_batch_stats),
        "rows": rows,
        "best_f1": best_f1,
        "best_recall_with_precision_ge_0_5": best_recall_with_precision,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def default_thresholds() -> list[float]:
    values = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18]
    values.extend(float(value) for value in np.arange(0.20, 0.51, 0.025))
    values.extend([0.55, 0.60, 0.70, 0.80])
    return sorted(set(round(value, 4) for value in values))


def evaluate_threshold(
    cached: Sequence[tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]],
    *,
    threshold: float,
    nms_iou_threshold: float,
    match_iou_threshold: float,
    max_detections: int,
) -> dict[str, Any]:
    tp = fp = fn = 0
    ious: list[float] = []
    angle_errors: list[float] = []
    for outputs, boxes in cached:
        detections = decode_batch(
            outputs,
            score_threshold=threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
        )
        for prediction, ground_truth in zip(detections, boxes, strict=True):
            matched = match_detections(
                prediction,
                ground_truth.tolist(),
                iou_threshold=match_iou_threshold,
            )
            tp += int(matched["tp"])
            fp += int(matched["fp"])
            fn += int(matched["fn"])
            ious.extend(float(value) for value in matched["ious"])
            angle_errors.extend(float(value) for value in matched["angle_errors"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rotated_iou": distribution_summary(ious),
        "angle_error_deg": distribution_summary(angle_errors),
    }


if __name__ == "__main__":
    raise SystemExit(main())
