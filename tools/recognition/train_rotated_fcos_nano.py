from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .rotated_fcos_nano import (
        RotatedFCOSNano,
        compute_loss,
        count_parameters,
        decode_batch,
        match_detections,
    )
else:
    _repository_root_for_import = Path(__file__).resolve().parents[2]
    if str(_repository_root_for_import) not in sys.path:
        sys.path.insert(0, str(_repository_root_for_import))
    from tools.recognition.rotated_fcos_nano import (  # type: ignore[no-redef]
        RotatedFCOSNano,
        compute_loss,
        count_parameters,
        decode_batch,
        match_detections,
    )


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


class RotatedCocoDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    def __init__(self, annotations_path: Path, *, repository_root: Path) -> None:
        self.annotations_path = annotations_path.resolve()
        self.repository_root = repository_root.resolve()
        payload = load_json(self.annotations_path)
        validate_rotated_coco(payload, self.annotations_path)
        self.images = sorted(payload["images"], key=lambda image: int(image["id"]))
        self.annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            self.annotations_by_image[int(annotation["image_id"])].append(annotation)
        self.mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        record = self.images[index]
        image_path = resolve_repository_image(self.repository_root, str(record["file_name"]))
        with Image.open(image_path) as opened:
            rgb = opened.convert("RGB")
            if rgb.size != (320, 320):
                raise ValueError(f"Expected 320x320 image, found {rgb.size}: {image_path}")
            array = np.array(rgb, dtype=np.uint8, copy=True)
        image = torch.from_numpy(array).permute(2, 0, 1).float().mul_(1.0 / 255.0)
        image = image.sub(self.mean).div(self.std)
        annotations = sorted(
            self.annotations_by_image.get(int(record["id"]), []),
            key=lambda annotation: int(annotation["id"]),
        )
        boxes = torch.tensor(
            [[float(value) for value in annotation["obb"]] for annotation in annotations],
            dtype=torch.float32,
        )
        if boxes.numel() == 0:
            boxes = torch.empty((0, 5), dtype=torch.float32)
        metadata = {
            "image_id": int(record["id"]),
            "file_name": str(record["file_name"]),
            "capture_id": record.get("capture_id"),
            "layout_id": record.get("layout_id"),
            "dataset_origin": record.get("dataset_origin"),
        }
        return image, boxes, metadata


def collate_batch(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]
) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, Any]]]:
    images = torch.stack([item[0] for item in batch], dim=0)
    boxes = [item[1] for item in batch]
    metadata = [item[2] for item in batch]
    return images, boxes, metadata


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny one-class rotated FCOS-style detector for Mahjong tile localization. "
            "The default model uses ImageNet-pretrained ShuffleNetV2 0.5x, a 64-channel FPN, "
            "and direct center/size/sin(2theta)/cos(2theta) regression."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--train-annotations",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "rotated_detector_augmented_dataset"
        / "annotations"
        / "train.json",
    )
    parser.add_argument(
        "--val-annotations",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "rotated_detector_corpus"
        / "annotations"
        / "val.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "rotated_fcos_runs"
        / "rfcos_nano_s05_f64_seed42",
    )
    parser.add_argument(
        "--backbone",
        choices=("shufflenet_v2_x0_5", "shufflenet_v2_x1_0"),
        default="shufflenet_v2_x0_5",
    )
    parser.add_argument("--fpn-channels", type=int, default=64)
    parser.add_argument("--head-convs", type=int, default=2)
    parser.add_argument(
        "--head-normalization",
        choices=("batch", "group"),
        default="group",
        help=(
            "Normalization used by the shared FCOS head. GroupNorm is the default because "
            "BatchNorm running statistics are invalid when one shared head is reused across FPN levels."
        ),
    )
    parser.add_argument(
        "--pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torchvision ImageNet weights. Disable if the training host cannot download them.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    parser.add_argument("--center-radius", type=float, default=1.5)
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=64)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--export-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt a 320x320 ONNX export after training. Export failure does not discard the checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)
    seed_everything(int(args.seed))
    repository_root = args.repository_root.resolve()
    train_annotations = args.train_annotations.resolve()
    val_annotations = args.val_annotations.resolve()
    output_directory = args.output_directory.resolve()
    for path in (train_annotations, val_annotations):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_directory.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(args.device))
    train_dataset = RotatedCocoDataset(train_annotations, repository_root=repository_root)
    val_dataset = RotatedCocoDataset(val_annotations, repository_root=repository_root)
    generator = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.workers) > 0,
        collate_fn=collate_batch,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.workers) > 0,
        collate_fn=collate_batch,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Train and validation loaders must both be non-empty")

    model = RotatedFCOSNano(
        backbone=str(args.backbone),
        fpn_channels=int(args.fpn_channels),
        head_convs=int(args.head_convs),
        head_normalization=str(args.head_normalization),
        pretrained_backbone=bool(args.pretrained_backbone),
    ).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    total_steps = max(1, int(args.epochs) * len(train_loader))
    warmup_steps = max(1, round(float(args.warmup_epochs) * len(train_loader)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        ),
    )
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = make_grad_scaler(use_amp)

    start_epoch = 0
    best_key = (-1.0, -1.0, -1.0, -float("inf"))
    best_epoch = -1
    if args.resume is not None:
        start_epoch, best_key, best_epoch = load_checkpoint(
            args.resume.resolve(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

    run_config = {
        "artifact": "rotated_fcos_nano_training",
        "repository_root": str(repository_root),
        "train_annotations": str(train_annotations),
        "val_annotations": str(val_annotations),
        "output_directory": str(output_directory),
        "model": model.config(),
        "parameter_count": parameter_count,
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_epochs": float(args.warmup_epochs),
            "freeze_backbone_epochs": int(args.freeze_backbone_epochs),
            "center_radius": float(args.center_radius),
            "seed": int(args.seed),
            "amp": use_amp,
        },
        "validation": {
            "score_threshold": float(args.score_threshold),
            "nms_iou_threshold": float(args.nms_iou_threshold),
            "match_iou_threshold": float(args.match_iou_threshold),
            "max_detections": int(args.max_detections),
        },
        "normalization": {"mean": list(IMAGE_MEAN), "std": list(IMAGE_STD)},
        "runtime": {
            "torch": str(torch.__version__),
            "device": str(device),
            "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        },
    }
    atomic_write_json(output_directory / "config.json", run_config)
    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)

    history_path = output_directory / "history.jsonl"
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(start_epoch, int(args.epochs)):
        freeze_backbone = epoch < int(args.freeze_backbone_epochs)
        set_backbone_trainable(model, not freeze_backbone)
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            center_radius=float(args.center_radius),
            backbone_frozen=freeze_backbone,
        )
        val_metrics = validate_epoch(
            model,
            val_loader,
            device=device,
            use_amp=use_amp,
            center_radius=float(args.center_radius),
            score_threshold=float(args.score_threshold),
            nms_iou_threshold=float(args.nms_iou_threshold),
            match_iou_threshold=float(args.match_iou_threshold),
            max_detections=int(args.max_detections),
        )
        record = {
            "epoch": epoch + 1,
            "elapsed_seconds": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "backbone_frozen": freeze_backbone,
            "train": train_metrics,
            "val": val_metrics,
        }
        append_jsonl(history_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

        key = validation_key(val_metrics)
        is_best = key > best_key
        if is_best:
            best_key = key
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = make_checkpoint(
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_key=best_key,
            best_epoch=best_epoch,
            run_config=run_config,
            val_metrics=val_metrics,
        )
        torch.save(checkpoint, output_directory / "last.pt")
        if is_best:
            torch.save(checkpoint, output_directory / "model_best.pt")
            atomic_write_json(
                output_directory / "model_best_metrics.json",
                {"epoch": best_epoch, "validation_key": list(best_key), "val": val_metrics},
            )

        if (
            int(args.early_stop_patience) > 0
            and epochs_without_improvement >= int(args.early_stop_patience)
        ):
            print(
                f"[early-stop] no validation improvement for {epochs_without_improvement} epochs",
                flush=True,
            )
            break

    best_path = output_directory / "model_best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(best_path)
    export_result: dict[str, Any] = {"requested": bool(args.export_onnx), "status": "skipped"}
    if bool(args.export_onnx):
        try:
            export_path = output_directory / "model_best.onnx"
            export_best_onnx(best_path, export_path, device=device)
            export_result = {"requested": True, "status": "completed", "path": str(export_path)}
        except Exception as error:
            export_result = {
                "requested": True,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            print(f"[onnx-export-warning] {type(error).__name__}: {error}", file=sys.stderr)

    summary = {
        "status": "completed",
        "output_directory": str(output_directory),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_validation_key": list(best_key),
        "parameter_count": parameter_count,
        "elapsed_seconds": time.perf_counter() - started,
        "onnx_export": export_result,
    }
    atomic_write_json(output_directory / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "fpn_channels", "head_convs", "max_detections"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.workers) < 0:
        raise ValueError("--workers must be non-negative")
    if float(args.learning_rate) <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if float(args.weight_decay) < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if float(args.center_radius) <= 0.0:
        raise ValueError("--center-radius must be positive")
    for name in ("score_threshold", "nms_iou_threshold", "match_iou_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be within [0,1]")


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def set_backbone_trainable(model: RotatedFCOSNano, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


def make_grad_scaler(enabled: bool) -> Any:
    amp_namespace = getattr(torch, "amp", None)
    grad_scaler = getattr(amp_namespace, "GradScaler", None)
    if grad_scaler is not None:
        return grad_scaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def cuda_autocast(enabled: bool) -> Any:
    amp_namespace = getattr(torch, "amp", None)
    autocast = getattr(amp_namespace, "autocast", None)
    if autocast is not None:
        return autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def learning_rate_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(0.05, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_epoch(
    model: RotatedFCOSNano,
    loader: DataLoader[Any],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
    center_radius: float,
    backbone_frozen: bool,
) -> dict[str, float]:
    model.train()
    if backbone_frozen:
        model.backbone.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    for images, boxes, _metadata in loader:
        images = images.to(device, non_blocking=True)
        targets = [item.to(device, non_blocking=True) for item in boxes]
        optimizer.zero_grad(set_to_none=True)
        with cuda_autocast(use_amp):
            outputs = model(images)
            losses = compute_loss(outputs, targets, center_radius=center_radius)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu())
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def validate_epoch(
    model: RotatedFCOSNano,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    use_amp: bool,
    center_radius: float,
    score_threshold: float,
    nms_iou_threshold: float,
    match_iou_threshold: float,
    max_detections: int,
) -> dict[str, Any]:
    model.eval()
    loss_totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    tp = fp = fn = 0
    ious: list[float] = []
    angle_errors: list[float] = []
    with torch.inference_mode():
        for images, boxes, _metadata in loader:
            images = images.to(device, non_blocking=True)
            targets = [item.to(device, non_blocking=True) for item in boxes]
            with cuda_autocast(use_amp):
                outputs = model(images)
                losses = compute_loss(outputs, targets, center_radius=center_radius)
            detections = decode_batch(
                outputs,
                score_threshold=score_threshold,
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
            for key, value in losses.items():
                loss_totals[key] += float(value.detach().cpu())
            batches += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": {key: value / max(1, batches) for key, value in loss_totals.items()},
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rotated_iou": distribution_summary(ious),
        "angle_error_deg": distribution_summary(angle_errors),
    }


def validation_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    f1 = float(metrics["f1"])
    recall = float(metrics["recall"])
    mean_iou = float(metrics["rotated_iou"]["mean"] or 0.0)
    loss = float(metrics["loss"]["total"])
    return (f1, recall, mean_iou, -loss)


def distribution_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p10": None, "median": None, "p90": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "p10": float(np.percentile(array, 10.0)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "mean": float(np.mean(array)),
    }


def make_checkpoint(
    *,
    epoch: int,
    model: RotatedFCOSNano,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    best_key: tuple[float, float, float, float],
    best_epoch: int,
    run_config: dict[str, Any],
    val_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "model_config": model.config(),
        "run_config": run_config,
        "best_key": list(best_key),
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
    }


def load_checkpoint(
    path: Path,
    *,
    model: RotatedFCOSNano,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    device: torch.device,
) -> tuple[int, tuple[float, float, float, float], int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    # This checkpoint is produced locally by this training script and includes optimizer/
    # scheduler metadata, so load the full trusted payload explicitly when the installed
    # PyTorch supports weights_only. Older training hosts do not expose that argument.
    payload = load_trusted_checkpoint(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    if "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])
    best_key_values = payload.get("best_key", [-1.0, -1.0, -1.0, -float("inf")])
    best_key = tuple(float(value) for value in best_key_values)
    if len(best_key) != 4:
        raise ValueError("Invalid best_key in checkpoint")
    return int(payload["epoch"]), best_key, int(payload.get("best_epoch", -1))  # type: ignore[return-value]


def export_best_onnx(checkpoint_path: Path, output_path: Path, *, device: torch.device) -> None:
    # model_best.pt is a trusted local checkpoint written by this script.
    payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
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
    dummy = torch.zeros(1, 3, 320, 320, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs: dict[str, Any] = {
        "input_names": ["images"],
        "output_names": ["stride8", "stride16", "stride32"],
        "opset_version": 17,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(model, dummy, str(output_path), **export_kwargs)


def load_trusted_checkpoint(path: Path, *, map_location: Any) -> Any:
    load_kwargs: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)


def validate_rotated_coco(payload: dict[str, Any], path: Path) -> None:
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"Invalid field {key}: {path}")
    if not payload["images"] or not payload["annotations"]:
        raise ValueError(f"Empty rotated dataset: {path}")
    image_ids = {int(image["id"]) for image in payload["images"]}
    for image in payload["images"]:
        if int(image.get("width", 0)) != 320 or int(image.get("height", 0)) != 320:
            raise ValueError(f"Dataset image is not 320x320: {image}")
    for annotation in payload["annotations"]:
        if int(annotation["image_id"]) not in image_ids:
            raise ValueError(f"Annotation references missing image: {annotation.get('id')}")
        obb = annotation.get("obb")
        if not isinstance(obb, list) or len(obb) != 5:
            raise ValueError(f"Annotation has no [cx,cy,w,h,angle] OBB: {annotation.get('id')}")
        values = [float(value) for value in obb]
        if not all(math.isfinite(value) for value in values) or values[2] <= 1.0 or values[3] <= 1.0:
            raise ValueError(f"Invalid OBB: {annotation.get('id')} {obb}")


def resolve_repository_image(repository_root: Path, file_name: str) -> Path:
    pure = PurePosixPath(file_name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe repository-relative path: {file_name}")
    path = repository_root.joinpath(*pure.parts)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        output.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
