from __future__ import annotations

import hashlib
import math
import random
import sqlite3

import numpy as np
import torch
import torch.nn.functional as F

from ..lib.mobilenet_v3_small_tile_classifier_v1 import (
    build_mobilenet_v3_small_tile_classifier,
)


_ARCHITECTURE_ID = "tile-classifier/tile-mobilenet-v3-small-1x-gray35-v1"
_IMAGE_SIZE = 64
_CLASS_COUNT = 35


class _Split:
    def __init__(
        self,
        sample_ids: tuple[str, ...],
        images_u8: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.sample_ids = sample_ids
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
            SELECT sample_id, image_gray_u8, class_index
            FROM sample
            WHERE split = ?
            ORDER BY sample_id
            """,
            (split_name,),
        ).fetchall()
    if not rows:
        raise ValueError(f"Corpus split is empty: {split_name}")
    expected_bytes = _IMAGE_SIZE * _IMAGE_SIZE
    sample_ids: list[str] = []
    images = np.empty((len(rows), _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.uint8)
    labels = np.empty((len(rows),), dtype=np.int64)
    for index, (sample_id, payload, class_index) in enumerate(rows):
        raw = bytes(payload)
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{split_name} sample has {len(raw)} gray bytes, expected {expected_bytes}"
            )
        label = int(class_index)
        if not 0 <= label < _CLASS_COUNT:
            raise ValueError(
                f"class_index must be in [0,{_CLASS_COUNT - 1}], got {label}"
            )
        sample_ids.append(str(sample_id))
        images[index] = np.frombuffer(raw, dtype=np.uint8).reshape(
            _IMAGE_SIZE,
            _IMAGE_SIZE,
        )
        labels[index] = label
    return _Split(tuple(sample_ids), torch.from_numpy(images), torch.from_numpy(labels))


def _normalization(train_images_u8: torch.Tensor) -> tuple[float, float]:
    train_np = train_images_u8.numpy()
    mean = float(train_np.mean(dtype=np.float64) / 255.0)
    std = float(train_np.std(dtype=np.float64) / 255.0)
    return mean, max(std, 1.0 / 255.0)


def _resolve_cache_device(
    requested: str,
    *,
    image_bytes: int,
    fraction: float,
) -> str:
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


def _cache_split(
    split: _Split,
    *,
    device: torch.device,
    cache_device: str,
) -> _Split:
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
        split.images_u8.index_select(0, index_cpu).to(
            device=device,
            non_blocking=True,
        ),
        split.labels.index_select(0, index_cpu).to(
            device=device,
            non_blocking=True,
        ),
    )


def _deterministic_random360_angles(
    sample_ids: tuple[str, ...],
    *,
    seed: int,
    epoch: int,
) -> np.ndarray:
    result = np.empty((len(sample_ids),), dtype=np.float32)
    prefix = f"{seed}\0{epoch}\0".encode("utf-8")
    denominator = float(2**64)
    for index, sample_id in enumerate(sample_ids):
        digest = hashlib.sha256(prefix + sample_id.encode("utf-8")).digest()
        unit = int.from_bytes(digest[:8], "big") / denominator
        result[index] = np.float32(-180.0 + 360.0 * unit)
    return result


def _rotate_batch(
    images: torch.Tensor,
    angles_deg: torch.Tensor,
) -> torch.Tensor:
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
    amp_enabled: bool,
    epoch: int,
    seed: int,
) -> dict[str, float]:
    model.train()
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    order = rng.permutation(split.count)
    angles = _deterministic_random360_angles(
        split.sample_ids,
        seed=seed,
        epoch=epoch,
    )
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for start in range(0, split.count, batch_size):
        indices = order[start : start + batch_size]
        images, targets = _fetch_batch(split, indices, device=device)
        images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
        batch_angles = torch.from_numpy(angles[indices]).to(
            device=device,
            dtype=torch.float32,
        )
        images = _rotate_batch(images, batch_angles)
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
        total_correct += int(
            (logits.detach().argmax(dim=1) == targets).sum().item()
        )
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
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for start in range(0, split.count, batch_size):
            indices = np.arange(
                start,
                min(split.count, start + batch_size),
                dtype=np.int64,
            )
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
            logits = model(images)
            correct += int(
                (logits.argmax(dim=1) == targets).sum().item()
            )
            total += int(targets.numel())
    return correct / max(total, 1)


def _evaluate_angles(
    model,
    split: _Split,
    *,
    angles: tuple[float, ...],
    batch_size: int,
    device: torch.device,
    mean: float,
    std: float,
) -> dict[str, float]:
    return {
        _angle_key(angle): _evaluate_split(
            model,
            split,
            angle_deg=angle,
            batch_size=batch_size,
            device=device,
            mean=mean,
            std=std,
        )
        for angle in angles
    }


def _validate_parameters(parameters) -> None:
    if int(parameters["epochs"]) < 1:
        raise ValueError("epochs must be positive")
    if int(parameters["batch_size"]) < 2:
        raise ValueError("batch_size must be at least 2")
    if int(parameters["eval_batch_size"]) < 1:
        raise ValueError("eval_batch_size must be positive")
    if float(parameters["learning_rate"]) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(parameters["weight_decay"]) < 0.0:
        raise ValueError("weight_decay must not be negative")
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
    if architecture_id != _ARCHITECTURE_ID:
        raise ValueError(
            f"This INV-011 recipe requires {_ARCHITECTURE_ID}, "
            f"got {architecture_id}"
        )

    seed = int(context.seed)
    _seed_everything(seed)
    _configure_cuda(tf32=bool(parameters["tf32"]))
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    database = context.corpus.root / "dataset.sqlite"
    train_split = _load_split(database, "train")
    manual_val = _load_split(database, "manual_val")
    mean, std = _normalization(train_split.images_u8)
    total_image_bytes = int(
        train_split.images_u8.numel()
        + manual_val.images_u8.numel()
    )
    cache_device = _resolve_cache_device(
        str(parameters["cache_device"]),
        image_bytes=total_image_bytes,
        fraction=float(parameters["cache_vram_fraction"]),
    )
    _cache_split(
        train_split,
        device=device,
        cache_device=cache_device,
    )
    _cache_split(
        manual_val,
        device=device,
        cache_device=cache_device,
    )

    # INV-011 seeds before model construction. MLDB constructs context.model
    # before Train Protocol invocation, so rebuild the selected architecture
    # after seeding and return state-compatible weights.
    model = build_mobilenet_v3_small_tile_classifier().to(device)
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
    eval_angles = tuple(
        float(value) for value in parameters["eval_angles"]
    )
    angle_eval_every = int(parameters["angle_eval_every"])

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        train_metrics = _train_one_epoch(
            model,
            train_split,
            optimizer=optimizer,
            scaler=scaler,
            batch_size=batch_size,
            device=device,
            mean=mean,
            std=std,
            amp_enabled=amp_enabled,
            epoch=epoch,
            seed=seed,
        )
        full_sweep = (
            epoch == 1
            or epoch == epochs
            or epoch % angle_eval_every == 0
        )
        angles = eval_angles if full_sweep else (0.0,)
        validation = _evaluate_angles(
            model,
            manual_val,
            angles=angles,
            batch_size=eval_batch_size,
            device=device,
            mean=mean,
            std=std,
        )

        context.telemetry.report_scalar(
            group="optimization",
            series="cross_entropy_loss",
            value=float(train_metrics["loss"]),
            step=epoch,
        )
        context.telemetry.report_scalar(
            group="optimization",
            series="train_accuracy",
            value=float(train_metrics["accuracy"]),
            step=epoch,
        )
        context.telemetry.report_scalar(
            group="validation",
            series="manual_accuracy_0deg",
            value=float(validation[_angle_key(0.0)]),
            step=epoch,
        )

        if full_sweep:
            score = float(
                np.mean(
                    [
                        validation[_angle_key(angle)]
                        for angle in eval_angles
                    ]
                )
            )
            context.telemetry.report_scalar(
                group="validation",
                series="checkpoint_manual_angle_mean",
                value=score,
                step=epoch,
            )
            if score > best_score:
                best_score = score
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

        scheduler.step()

    if best_state is None:
        raise RuntimeError(
            "training produced no full-sweep checkpoint candidate"
        )
    model.load_state_dict(best_state, strict=True)
    return model
