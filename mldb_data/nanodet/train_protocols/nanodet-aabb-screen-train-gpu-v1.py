from __future__ import annotations

import json
import math
import random
import sqlite3
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


INPUT_SIZE = 320
BGR_MEAN = torch.tensor((103.53, 116.28, 123.675), dtype=torch.float32).view(1, 3, 1, 1)
BGR_STD = torch.tensor((57.375, 57.12, 58.395), dtype=torch.float32).view(1, 3, 1, 1)


class DetectionDataset(Dataset):
    def __init__(self, path: Path, split: str) -> None:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                """
                SELECT image_chw_u8, annotations_json
                FROM sample
                WHERE split=?
                ORDER BY sample_id
                """,
                (split,),
            ).fetchall()
        if not rows:
            raise ValueError(f"corpus split is empty: {split}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image_raw, annotations_raw = self.rows[index]
        image = torch.from_numpy(
            np.frombuffer(image_raw, dtype=np.uint8).copy().reshape(3, INPUT_SIZE, INPUT_SIZE)
        )
        annotations = json.loads(annotations_raw)
        boxes = np.asarray(
            [
                [
                    float(item["bbox"][0]),
                    float(item["bbox"][1]),
                    float(item["bbox"][0] + item["bbox"][2]),
                    float(item["bbox"][1] + item["bbox"][3]),
                ]
                for item in annotations
            ],
            dtype=np.float32,
        ).reshape(-1, 4)
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)
        ignored = np.empty((0, 4), dtype=np.float32)
        return image, boxes, labels, ignored


def _collate(batch):
    return (
        torch.stack([item[0] for item in batch]),
        [item[1] for item in batch],
        [item[2] for item in batch],
        [item[3] for item in batch],
    )


def _preprocess(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    bgr = images[:, [2, 1, 0]].to(device=device, dtype=torch.float32, non_blocking=True)
    return (bgr - BGR_MEAN.to(device)) / BGR_STD.to(device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loss_for_batch(model, images, boxes, labels, ignored, device, amp: bool):
    meta = {
        "img": _preprocess(images, device),
        "gt_bboxes": boxes,
        "gt_labels": labels,
        "gt_bboxes_ignore": ignored,
    }
    with torch.cuda.amp.autocast(enabled=amp):
        _predictions, loss, loss_states = model.forward_train(meta)
    return loss, loss_states


def _validate(model, loader, device, amp: bool) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for images, boxes, labels, ignored in loader:
            loss, _states = _loss_for_batch(
                model, images, boxes, labels, ignored, device, amp
            )
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise RuntimeError("non-finite validation loss")
            total += value
            count += 1
    return total / max(1, count)


def train(context):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NanoDet AABB screen training")

    parameters = context.parameters
    seed = int(context.seed)
    _seed_everything(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    database = context.corpus.root / "dataset.sqlite"
    train_dataset = DetectionDataset(database, str(parameters["train_split"]))
    val_dataset = DetectionDataset(database, str(parameters["validation_split"]))

    generator = torch.Generator().manual_seed(seed)
    workers = int(parameters["workers"])
    batch_size = int(parameters["batch_size"])
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

    model = context.model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )

    epochs = int(parameters["epochs"])
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, int(float(parameters["warmup_epochs"]) * steps_per_epoch))
    base_lr = float(parameters["learning_rate"])
    eta_min = float(parameters["eta_min"])
    floor = min(1.0, eta_min / base_lr)

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            progress = (step + 1) / warmup_steps
            return 1.0e-4 + (1.0 - 1.0e-4) * progress
        progress = min(
            1.0,
            max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)),
        )
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    use_amp = bool(parameters["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    validation_interval = int(parameters["validation_interval"])

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.set_epoch(epoch)
        model.train()
        train_total = 0.0
        train_batches = 0

        for images, boxes, labels, ignored in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _loss_states = _loss_for_batch(
                model, images, boxes, labels, ignored, device, use_amp
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(parameters["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_total += float(loss.detach().cpu())
            train_batches += 1

        completed_epoch = epoch + 1
        context.telemetry.report_scalar(
            group="training",
            series="loss",
            value=train_total / max(1, train_batches),
            step=completed_epoch,
        )
        context.telemetry.report_scalar(
            group="training",
            series="learning_rate",
            value=float(optimizer.param_groups[0]["lr"]),
            step=completed_epoch,
        )

        if completed_epoch % validation_interval == 0 or completed_epoch == epochs:
            val_loss = _validate(model, val_loader, device, use_amp)
            context.telemetry.report_scalar(
                group="validation",
                series="loss",
                value=val_loss,
                step=completed_epoch,
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("training produced no validated model state")
    model.load_state_dict(best_state)
    return model
