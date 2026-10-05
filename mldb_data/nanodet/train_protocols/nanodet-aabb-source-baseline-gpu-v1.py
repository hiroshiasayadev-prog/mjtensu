from __future__ import annotations

import copy
import math
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from nanodet_aabb_optimized.data.batch_process import stack_batch_img
from nanodet_aabb_optimized.data.dataset.coco import CocoDataset
from nanodet_aabb_optimized.util import load_model_weight


_ARCHIVE = "nanodet-training-baseline-v1.tar"
_TRAIN_PIPELINE = {
    "perspective": 0.0,
    "scale": [0.6, 1.4],
    "stretch": [[0.8, 1.2], [0.8, 1.2]],
    "rotation": 0,
    "shear": 0,
    "translate": 0.2,
    "flip": 0.5,
    "brightness": 0.2,
    "contrast": [0.6, 1.4],
    "saturation": [0.5, 1.2],
    "normalize": [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]],
}
_VAL_PIPELINE = {
    "normalize": [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]],
}


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def warn(self, _message):
        pass

    def log(self, _message):
        pass


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_archive(archive: Path, destination: Path) -> Path:
    root = destination / "training-corpus"
    marker = root / ".ready"
    if marker.is_file():
        return root
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as tf:
        for member in tf.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe corpus archive path: {member.name}")
        tf.extractall(root)
    marker.write_text("ok\n", encoding="utf-8")
    return root


def _dataset(root: Path, split: str, *, train: bool):
    if split.startswith("jp_"):
        image_path = root / "images" / split
        annotation = root / "annotations" / f"{split}.coco.json"
    elif split.startswith("composite_"):
        image_path = root / "images" / "composite"
        annotation = root / "annotations" / f"{split}.coco.json"
    else:
        raise ValueError(f"unsupported split: {split}")
    return CocoDataset(
        img_path=str(image_path),
        ann_path=str(annotation),
        input_size=(320, 320),
        pipeline=_TRAIN_PIPELINE if train else _VAL_PIPELINE,
        keep_ratio=False,
        mode="train" if train else "val",
    )


def _datasets(root: Path, recipe: str):
    if recipe == "coco-only-v1":
        train = _dataset(root, "jp_train", train=True)
        val = _dataset(root, "jp_valid", train=False)
    elif recipe == "joint-naive-v1":
        train = ConcatDataset(
            [
                _dataset(root, "jp_train", train=True),
                _dataset(root, "composite_train", train=True),
            ]
        )
        val = ConcatDataset(
            [
                _dataset(root, "jp_valid", train=False),
                _dataset(root, "composite_val", train=False),
            ]
        )
    else:
        raise ValueError(f"unsupported recipe: {recipe}")
    return train, val



def _collate(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        result = {key: _collate([sample[key] for sample in batch]) for key in elem}
        images = result.get("img")
        if isinstance(images, list) and images and all(isinstance(x, torch.Tensor) for x in images):
            result["img"] = stack_batch_img(images, divisible=32)
        return result
    return batch


def _move_images(batch, device):
    batch = dict(batch)
    batch["img"] = batch["img"].to(device=device, non_blocking=True)
    return batch


@torch.no_grad()
def _update_ema(ema_model, model, decay: float) -> None:
    source = model.state_dict()
    for name, target in ema_model.state_dict().items():
        value = source[name].detach()
        if target.dtype.is_floating_point:
            target.mul_(decay).add_(value, alpha=1.0 - decay)
        else:
            target.copy_(value)


def _mean_states(accumulator, states):
    for name, value in states.items():
        accumulator[name] = accumulator.get(name, 0.0) + float(value.detach().mean().cpu())


@torch.no_grad()
def _validate(model, loader, device, amp: bool, epoch: int):
    model.set_epoch(epoch)
    model.eval()
    total = 0.0
    count = 0
    states_total: dict[str, float] = {}
    for batch in loader:
        batch = _move_images(batch, device)
        with torch.cuda.amp.autocast(enabled=amp):
            _predictions, loss, states = model.forward_train(batch)
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise RuntimeError("non-finite validation loss")
        total += value
        count += 1
        _mean_states(states_total, states)
    return total / max(1, count), {k: v / max(1, count) for k, v in states_total.items()}


def train(context):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    p = context.parameters
    seed = int(context.seed)
    _seed_everything(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    root = _extract_archive(context.corpus.root / _ARCHIVE, context.work_dir)
    train_dataset, val_dataset = _datasets(root, str(p["recipe"]))

    workers = int(p["workers"])
    batch_size = int(p["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=_collate,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=_collate,
        drop_last=False,
    )

    device = torch.device("cuda")
    model = context.model
    checkpoint = torch.load(
        root / "pretrained" / "nanodet-plus-m_320.pth",
        map_location="cpu",
        weights_only=True,
    )
    load_model_weight(model, checkpoint, _Logger())
    model.to(device)

    ema_model = copy.deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    base_lr = float(p["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(p["weight_decay"]),
    )
    epochs = int(p["epochs"])
    eta_min = float(p["eta_min"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=eta_min
    )
    warmup_steps = int(p["warmup_steps"])
    warmup_ratio = float(p["warmup_ratio"])
    validation_interval = int(p["validation_interval"])
    ema_decay = float(p["ema_decay"])
    amp = bool(p["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.set_epoch(epoch)
        model.train()
        total = 0.0
        batches = 0
        states_total: dict[str, float] = {}

        for batch in train_loader:
            global_step += 1
            if global_step <= warmup_steps:
                progress = global_step / max(1, warmup_steps)
                factor = warmup_ratio + (1.0 - warmup_ratio) * progress
                for group in optimizer.param_groups:
                    group["lr"] = base_lr * factor

            batch = _move_images(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                _predictions, loss, states = model.forward_train(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(p["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            _update_ema(ema_model, model, ema_decay)

            total += float(loss.detach().cpu())
            batches += 1
            _mean_states(states_total, states)

        scheduler.step()
        completed = epoch + 1
        context.telemetry.report_scalar(
            group="training",
            series="loss",
            value=total / max(1, batches),
            step=completed,
        )
        context.telemetry.report_scalar(
            group="training",
            series="learning_rate",
            value=float(optimizer.param_groups[0]["lr"]),
            step=completed,
        )
        for name, value in states_total.items():
            context.telemetry.report_scalar(
                group="training_components",
                series=name,
                value=value / max(1, batches),
                step=completed,
            )

        if completed % validation_interval == 0 or completed == epochs:
            val_loss, val_states = _validate(ema_model, val_loader, device, amp, epoch)
            context.telemetry.report_scalar(
                group="validation", series="loss", value=val_loss, step=completed
            )
            for name, value in val_states.items():
                context.telemetry.report_scalar(
                    group="validation_components",
                    series=name,
                    value=value,
                    step=completed,
                )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in ema_model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("training produced no validated checkpoint")
    model.load_state_dict(best_state)
    return model
