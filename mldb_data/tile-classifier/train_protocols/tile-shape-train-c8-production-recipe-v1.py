from __future__ import annotations

import math
import random
import sqlite3

import numpy as np
import torch
import torch.nn.functional as F

from ..lib.c8_tile_shape_classifier_v1 import build_c8_tile_shape_classifier


_C8_ARCHITECTURE_ID = "tile-classifier/tile-c8-gray35-v1"
_IMAGE_SIZE = 64
_CLASS_COUNT = 35


class _Split:
    def __init__(self, images_u8: torch.Tensor, labels: torch.Tensor) -> None:
        self.images_u8 = images_u8
        self.labels = labels

    @property
    def count(self) -> int:
        return int(self.labels.shape[0])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_cuda(*, tf32: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def _load_split(database, split_name: str) -> _Split:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT image_gray_u8, class_index
            FROM sample
            WHERE split = ?
            ORDER BY sample_id
            """,
            (split_name,),
        ).fetchall()
    if not rows:
        raise ValueError(f"Corpus split is empty: {split_name}")
    expected_bytes = _IMAGE_SIZE * _IMAGE_SIZE
    images = np.empty((len(rows), _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.uint8)
    labels = np.empty((len(rows),), dtype=np.int64)
    for index, (payload, class_index) in enumerate(rows):
        raw = bytes(payload)
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{split_name} sample has {len(raw)} gray bytes, expected {expected_bytes}"
            )
        label = int(class_index)
        if not 0 <= label < _CLASS_COUNT:
            raise ValueError(f"class_index must be in [0,{_CLASS_COUNT - 1}], got {label}")
        images[index] = np.frombuffer(raw, dtype=np.uint8).reshape(_IMAGE_SIZE, _IMAGE_SIZE)
        labels[index] = label
    return _Split(torch.from_numpy(images), torch.from_numpy(labels))


def _normalization(train_images_u8: torch.Tensor) -> tuple[float, float]:
    train_np = train_images_u8.numpy()
    mean = float(train_np.mean(dtype=np.float64) / 255.0)
    std = float(train_np.std(dtype=np.float64) / 255.0)
    return mean, max(std, 1.0 / 255.0)


def _resolve_cache_device(requested: str, *, image_bytes: int, fraction: float) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("cache_device must be one of: auto, cpu, cuda")
    if not 0.0 < fraction < 0.8:
        raise ValueError("cache_vram_fraction must be between 0 and 0.8")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cache_device=cuda requires CUDA")
        return "cuda"
    if not torch.cuda.is_available():
        return "cpu"
    free_bytes, _ = torch.cuda.mem_get_info()
    return "cuda" if image_bytes <= int(free_bytes * fraction) else "cpu"


def _cache_split(split: _Split, *, device: torch.device, cache_device: str) -> _Split:
    if cache_device == "cuda":
        split.images_u8 = split.images_u8.to(device=device, non_blocking=False)
        split.labels = split.labels.to(device=device, non_blocking=False)
    elif device.type == "cuda":
        split.images_u8 = split.images_u8.pin_memory()
        split.labels = split.labels.pin_memory()
    return split


def _fetch_batch(
    split: _Split,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    index_cpu = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    if split.images_u8.device.type == "cuda":
        index_device = index_cpu.to(device=device, non_blocking=False)
        return (
            split.images_u8.index_select(0, index_device),
            split.labels.index_select(0, index_device),
        )
    return (
        split.images_u8.index_select(0, index_cpu).to(device=device, non_blocking=True),
        split.labels.index_select(0, index_cpu).to(device=device, non_blocking=True),
    )


def _rotate_batch(images: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    radians = angles_deg.to(dtype=torch.float32) * (math.pi / 180.0)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    theta = torch.zeros(
        (images.shape[0], 2, 3),
        device=images.device,
        dtype=torch.float32,
    )
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def _train_one_epoch(
    model,
    split: _Split,
    *,
    optimizer,
    scaler,
    batch_size: int,
    device: torch.device,
    mean: float,
    std: float,
    rotation_augment_deg: float,
    amp_enabled: bool,
    epoch: int,
    seed: int,
) -> dict[str, float]:
    model.train()
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    order = rng.permutation(split.count)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for start in range(0, split.count, batch_size):
        indices = order[start : start + batch_size]
        images, targets = _fetch_batch(split, indices, device=device)
        images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
        if rotation_augment_deg > 0.0:
            angles = torch.empty(
                (images.shape[0],),
                device=device,
                dtype=torch.float32,
            ).uniform_(-rotation_augment_deg, rotation_augment_deg)
            images = _rotate_batch(images, angles)
        images = images.sub(mean).div(std)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        count = int(targets.shape[0])
        total_loss += float(loss.detach().item()) * count
        total_correct += int((logits.detach().argmax(dim=1) == targets).sum().item())
        total_count += count

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def _angle_key(angle: float) -> str:
    value = float(angle)
    return f"{int(value)}deg" if value.is_integer() else f"{value:g}deg"


def _evaluate_split(
    model,
    split: _Split,
    *,
    angle_deg: float,
    batch_size: int,
    device: torch.device,
    mean: float,
    std: float,
    amp_enabled: bool,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for start in range(0, split.count, batch_size):
            indices = np.arange(start, min(split.count, start + batch_size), dtype=np.int64)
            images, targets = _fetch_batch(split, indices, device=device)
            images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
            if abs(angle_deg) > 1.0e-9:
                angles = torch.full(
                    (images.shape[0],),
                    angle_deg,
                    device=device,
                    dtype=torch.float32,
                )
                images = _rotate_batch(images, angles)
            images = images.sub(mean).div(std)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.numel())
    return correct / max(total, 1)


def _evaluate_all(
    model,
    splits: dict[str, _Split],
    *,
    angles: tuple[float, ...],
    batch_size: int,
    device: torch.device,
    mean: float,
    std: float,
    amp_enabled: bool,
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for split_name in ("manual_val", "jp_val"):
        angle_results: dict[str, float] = {}
        for angle in angles:
            angle_results[_angle_key(angle)] = _evaluate_split(
                model,
                splits[split_name],
                angle_deg=angle,
                batch_size=batch_size,
                device=device,
                mean=mean,
                std=std,
                amp_enabled=amp_enabled,
            )
        result[split_name] = {"angles": angle_results}
    return result


def _checkpoint_score(
    validation: dict[str, dict[str, dict[str, float]]],
    *,
    required_angles: tuple[float, ...],
) -> float | None:
    manual_angles = validation["manual_val"]["angles"]
    keys = [_angle_key(angle) for angle in required_angles]
    if any(key not in manual_angles for key in keys):
        return None
    return float(np.mean([manual_angles[key] for key in keys]))


def _validate_parameters(parameters) -> None:
    if int(parameters["epochs"]) < 1:
        raise ValueError("epochs must be positive")
    if int(parameters["batch_size"]) < 2 or int(parameters["eval_batch_size"]) < 2:
        raise ValueError("batch sizes must be at least 2")
    if float(parameters["learning_rate"]) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(parameters["weight_decay"]) < 0.0:
        raise ValueError("weight_decay must not be negative")
    rotation = float(parameters["rotation_augment_deg"])
    if not 0.0 <= rotation <= 45.0:
        raise ValueError("rotation_augment_deg must be in [0,45]")
    angles = tuple(float(value) for value in parameters["eval_angles"])
    if not angles or not any(abs(angle) < 1.0e-9 for angle in angles):
        raise ValueError("eval_angles must include 0 degrees")
    if int(parameters["angle_eval_every"]) < 1:
        raise ValueError("angle_eval_every must be positive")
    _resolve_cache_device(
        str(parameters["cache_device"]),
        image_bytes=0,
        fraction=float(parameters["cache_vram_fraction"]),
    )


def train(context):
    parameters = context.parameters
    _validate_parameters(parameters)
    architecture_id = context.architecture.get("id")
    if architecture_id != _C8_ARCHITECTURE_ID:
        raise ValueError(
            f"This historical C8 recipe requires {_C8_ARCHITECTURE_ID}, got {architecture_id}"
        )

    seed = int(context.seed)
    _seed_everything(seed)
    _configure_cuda(tf32=bool(parameters["tf32"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    database = context.corpus.root / "dataset.sqlite"
    splits = {
        name: _load_split(database, name)
        for name in ("train", "manual_val", "jp_val")
    }
    mean, std = _normalization(splits["train"].images_u8)
    total_image_bytes = sum(int(split.images_u8.numel()) for split in splits.values())
    cache_device = _resolve_cache_device(
        str(parameters["cache_device"]),
        image_bytes=total_image_bytes,
        fraction=float(parameters["cache_vram_fraction"]),
    )
    for split in splits.values():
        _cache_split(split, device=device, cache_device=cache_device)

    # Historical source seeds before model construction. MLDB constructs context.model
    # before Train Protocol invocation, so rebuild the exact selected Architecture here
    # after seeding. The returned state remains strict-load compatible with context.model.
    model = build_c8_tile_shape_classifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    epochs = int(parameters["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(parameters["learning_rate"]) * 0.05,
    )
    amp_enabled = bool(parameters["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    batch_size = int(parameters["batch_size"])
    eval_batch_size = int(parameters["eval_batch_size"])
    rotation_augment_deg = float(parameters["rotation_augment_deg"])
    eval_angles = tuple(float(value) for value in parameters["eval_angles"])
    angle_eval_every = int(parameters["angle_eval_every"])

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        train_metrics = _train_one_epoch(
            model,
            splits["train"],
            optimizer=optimizer,
            scaler=scaler,
            batch_size=batch_size,
            device=device,
            mean=mean,
            std=std,
            rotation_augment_deg=rotation_augment_deg,
            amp_enabled=amp_enabled,
            epoch=epoch,
            seed=seed,
        )
        full_sweep = (
            epoch == 1
            or epoch == epochs
            or epoch % angle_eval_every == 0
        )
        epoch_angles = eval_angles if full_sweep else (0.0,)
        validation = _evaluate_all(
            model,
            splits,
            angles=epoch_angles,
            batch_size=eval_batch_size,
            device=device,
            mean=mean,
            std=std,
            amp_enabled=amp_enabled,
        )
        scheduler.step()

        context.telemetry.report_scalar(
            group="optimization",
            series="cross_entropy_loss",
            value=float(train_metrics["loss"]),
            step=epoch,
        )
        context.telemetry.report_scalar(
            group="validation",
            series="manual_accuracy_0deg",
            value=float(validation["manual_val"]["angles"][_angle_key(0.0)]),
            step=epoch,
        )
        context.telemetry.report_scalar(
            group="validation",
            series="jp_accuracy_0deg",
            value=float(validation["jp_val"]["angles"][_angle_key(0.0)]),
            step=epoch,
        )

        primary = _checkpoint_score(validation, required_angles=eval_angles)
        if primary is not None:
            context.telemetry.report_scalar(
                group="validation",
                series="checkpoint_manual_angle_mean",
                value=float(primary),
                step=epoch,
            )
            if primary > best_score:
                best_score = primary
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("training produced no full-sweep checkpoint candidate")
    model.load_state_dict(best_state, strict=True)
    return model
