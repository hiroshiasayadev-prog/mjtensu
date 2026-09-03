from __future__ import annotations

import argparse
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
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .rotated_box_rectifier import (
        CropWindow,
        RotatedBoxRectifier,
        angle_error_deg,
        count_parameters,
        decode_prediction,
        default_crop_window,
        encode_target,
        rotated_overlap_metrics,
        window_contains_obb,
    )
else:
    repository_root_for_import = Path(__file__).resolve().parents[2]
    if str(repository_root_for_import) not in sys.path:
        sys.path.insert(0, str(repository_root_for_import))
    from tools.recognition.rotated_box_rectifier import (  # type: ignore[no-redef]
        CropWindow,
        RotatedBoxRectifier,
        angle_error_deg,
        count_parameters,
        decode_prediction,
        default_crop_window,
        encode_target,
        rotated_overlap_metrics,
        window_contains_obb,
    )


IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.25, 0.25, 0.25)


class RectifierDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    def __init__(
        self,
        annotations_path: Path,
        *,
        repository_root: Path,
        input_size: int,
        training: bool,
        context_scale: float,
        center_jitter: float,
        scale_jitter: float,
    ) -> None:
        self.annotations_path = annotations_path.resolve()
        self.repository_root = repository_root.resolve()
        self.input_size = int(input_size)
        self.training = bool(training)
        self.context_scale = float(context_scale)
        self.center_jitter = float(center_jitter)
        self.scale_jitter = float(scale_jitter)
        payload = load_json(self.annotations_path)
        validate_rotated_coco(payload, self.annotations_path)
        images_by_id = {int(image["id"]): image for image in payload["images"]}
        self.samples: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for annotation in sorted(payload["annotations"], key=lambda item: int(item["id"])):
            image = images_by_id.get(int(annotation["image_id"]))
            if image is None:
                raise ValueError(f"Missing image for annotation {annotation.get('id')}")
            self.samples.append((image, annotation))
        self.mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        image_record, annotation = self.samples[index]
        image_path = resolve_repository_image(self.repository_root, str(image_record["file_name"]))
        target_obb = tuple(float(value) for value in annotation["obb"])
        window = self._sample_window(target_obb)
        with Image.open(image_path) as opened:
            rgb = opened.convert("RGB")
            crop = sample_window(rgb, window, self.input_size)
        if self.training:
            crop = augment_appearance(crop)
        array = np.asarray(crop, dtype=np.uint8).copy()
        crop.close()
        image = torch.from_numpy(array).permute(2, 0, 1).float().mul_(1.0 / 255.0)
        image = image.sub(self.mean).div(self.std)
        target = encode_target(target_obb, window)
        metadata = {
            "window": (window.center_x, window.center_y, window.side),
            "target_obb": target_obb,
            "annotation_id": int(annotation["id"]),
            "image_id": int(image_record["id"]),
            "file_name": str(image_record["file_name"]),
            "region": annotation.get("region"),
        }
        return image, target, metadata

    def _sample_window(self, obb: Sequence[float]) -> CropWindow:
        base = default_crop_window(obb, context_scale=self.context_scale)
        if not self.training or (self.center_jitter <= 0.0 and self.scale_jitter <= 0.0):
            return base
        for _attempt in range(16):
            scale = math.exp(random.uniform(-self.scale_jitter, self.scale_jitter))
            side = base.side * scale
            candidate = CropWindow(
                center_x=base.center_x + random.uniform(-self.center_jitter, self.center_jitter) * base.side,
                center_y=base.center_y + random.uniform(-self.center_jitter, self.center_jitter) * base.side,
                side=side,
            )
            if window_contains_obb(candidate, obb):
                return candidate
        return base


def collate_batch(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    return (
        torch.stack([item[0] for item in batch]),
        torch.stack([item[1] for item in batch]),
        [item[2] for item in batch],
    )


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny HBB-to-OBB crop rectifier. Each known tile HBB is expanded to a square "
            "context crop; the model predicts center correction, oriented size and angle."
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
        / "rotated_box_rectifier_runs"
        / "rgb96_seed42",
    )
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--context-scale", type=float, default=1.40)
    parser.add_argument("--center-jitter", type=float, default=0.06)
    parser.add_argument("--scale-jitter", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-onnx", action=argparse.BooleanOptionalAction, default=True)
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
    train_dataset = RectifierDataset(
        train_annotations,
        repository_root=repository_root,
        input_size=int(args.input_size),
        training=True,
        context_scale=float(args.context_scale),
        center_jitter=float(args.center_jitter),
        scale_jitter=float(args.scale_jitter),
    )
    val_dataset = RectifierDataset(
        val_annotations,
        repository_root=repository_root,
        input_size=int(args.input_size),
        training=False,
        context_scale=float(args.context_scale),
        center_jitter=0.0,
        scale_jitter=0.0,
    )
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

    model = RotatedBoxRectifier(input_size=int(args.input_size)).to(device)
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
            step, total_steps=total_steps, warmup_steps=warmup_steps
        ),
    )
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    parameter_count = count_parameters(model)

    config = {
        "artifact": "rotated_box_rectifier_training",
        "repository_root": str(repository_root),
        "train_annotations": str(train_annotations),
        "val_annotations": str(val_annotations),
        "output_directory": str(output_directory),
        "model": model.config(),
        "parameter_count": parameter_count,
        "dataset": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "context_scale": float(args.context_scale),
            "center_jitter": float(args.center_jitter),
            "scale_jitter": float(args.scale_jitter),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_epochs": float(args.warmup_epochs),
            "seed": int(args.seed),
            "amp": use_amp,
        },
        "normalization": {"mean": list(IMAGE_MEAN), "std": list(IMAGE_STD)},
        "runtime": {
            "torch": str(torch.__version__),
            "device": str(device),
            "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        },
    }
    atomic_write_json(output_directory / "config.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    best_key = (-1.0, -1.0, -float("inf"), -float("inf"))
    best_epoch = -1
    epochs_without_improvement = 0
    started = time.perf_counter()
    history_path = output_directory / "history.jsonl"
    for epoch in range(int(args.epochs)):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        val_metrics = validate_epoch(model, val_loader, device=device, use_amp=use_amp)
        record = {
            "epoch": epoch + 1,
            "elapsed_seconds": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
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
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "config": config,
            "val_metrics": val_metrics,
            "best_key": list(best_key),
            "best_epoch": best_epoch,
        }
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
            export_onnx(best_path, export_path, device=device)
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


def train_epoch(
    model: RotatedBoxRectifier,
    loader: DataLoader[Any],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    for images, targets, _metadata in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            predictions = model(images)
            losses = rectifier_loss(predictions, targets)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu())
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def validate_epoch(
    model: RotatedBoxRectifier,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    loss_totals: defaultdict[str, float] = defaultdict(float)
    ious: list[float] = []
    coverages: list[float] = []
    purities: list[float] = []
    angle_errors: list[float] = []
    center_errors: list[float] = []
    batches = 0
    with torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                predictions = model(images)
                losses = rectifier_loss(predictions, targets_device)
            predictions_cpu = predictions.detach().float().cpu()
            for row, item in zip(predictions_cpu, metadata, strict=True):
                window_values = item["window"]
                window = CropWindow(
                    center_x=float(window_values[0]),
                    center_y=float(window_values[1]),
                    side=float(window_values[2]),
                )
                predicted = decode_prediction(row.tolist(), window)
                target_obb = tuple(float(value) for value in item["target_obb"])
                iou, coverage, purity = rotated_overlap_metrics(predicted, target_obb)
                ious.append(iou)
                coverages.append(coverage)
                purities.append(purity)
                angle_errors.append(angle_error_deg(predicted[4], target_obb[4]))
                center_errors.append(math.hypot(predicted[0] - target_obb[0], predicted[1] - target_obb[1]))
            for key, value in losses.items():
                loss_totals[key] += float(value.detach().cpu())
            batches += 1
    return {
        "loss": {key: value / max(1, batches) for key, value in loss_totals.items()},
        "rotated_iou": distribution_summary(ious),
        "gt_coverage": distribution_summary(coverages),
        "crop_purity": distribution_summary(purities),
        "angle_error_deg": distribution_summary(angle_errors),
        "center_error_px": distribution_summary(center_errors),
    }


def rectifier_loss(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    center = F.smooth_l1_loss(predictions[:, :2], targets[:, :2], beta=0.05)
    size = F.smooth_l1_loss(predictions[:, 2:4], targets[:, 2:4], beta=0.10)
    predicted_angle = F.normalize(predictions[:, 4:6], dim=1, eps=1.0e-6)
    target_angle = F.normalize(targets[:, 4:6], dim=1, eps=1.0e-6)
    angle = (1.0 - (predicted_angle * target_angle).sum(dim=1)).mean()
    total = 2.0 * center + size + angle
    return {"total": total, "center": center, "size": size, "angle": angle}


def validation_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    iou_p10 = float(metrics["rotated_iou"]["p10"] or 0.0)
    iou_median = float(metrics["rotated_iou"]["median"] or 0.0)
    angle_p90 = float(metrics["angle_error_deg"]["p90"] or 180.0)
    loss = float(metrics["loss"]["total"])
    return (iou_p10, iou_median, -angle_p90, -loss)


def sample_window(image: Image.Image, window: CropWindow, input_size: int) -> Image.Image:
    return image.transform(
        (input_size, input_size),
        Image.Transform.EXTENT,
        (window.left, window.top, window.right, window.bottom),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def augment_appearance(image: Image.Image) -> Image.Image:
    if random.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.80, 1.20))
    if random.random() < 0.8:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
    return image


def learning_rate_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(0.05, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


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


def export_onnx(checkpoint_path: Path, output_path: Path, *, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model = RotatedBoxRectifier(input_size=int(config["input_size"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval().to(device)
    dummy = torch.zeros(1, 3, int(config["input_size"]), int(config["input_size"]), device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["geometry"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "input_size"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.workers) < 0:
        raise ValueError("--workers must be non-negative")
    if float(args.context_scale) <= 1.0:
        raise ValueError("--context-scale must be > 1")
    if float(args.center_jitter) < 0.0 or float(args.scale_jitter) < 0.0:
        raise ValueError("jitter values must be non-negative")
    if float(args.learning_rate) <= 0.0:
        raise ValueError("--learning-rate must be positive")


def validate_rotated_coco(payload: dict[str, Any], path: Path) -> None:
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list) or not images or not annotations:
        raise ValueError(f"Invalid or empty rotated dataset: {path}")
    image_ids = {int(image["id"]) for image in images}
    for annotation in annotations:
        if int(annotation["image_id"]) not in image_ids:
            raise ValueError(f"Annotation references missing image: {annotation.get('id')}")
        obb = annotation.get("obb")
        if not isinstance(obb, list) or len(obb) != 5:
            raise ValueError(f"Annotation has invalid OBB: {annotation.get('id')}")


def resolve_repository_image(repository_root: Path, file_name: str) -> Path:
    pure = PurePosixPath(file_name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe repository-relative path: {file_name}")
    path = repository_root.joinpath(*pure.parts)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
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
