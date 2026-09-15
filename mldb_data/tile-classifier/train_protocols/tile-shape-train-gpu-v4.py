from __future__ import annotations

import math
import random
import sqlite3
from pathlib import Path

import torch
import torch.nn.functional as F



def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_split(database: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT image_gray_u8, class_index FROM sample WHERE split = ? ORDER BY sample_id",
            (split,),
        ).fetchall()
    if not rows:
        raise ValueError(f"Corpus split is empty: {split}")
    images = torch.stack([
        torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone().reshape(1, 64, 64)
        for payload, _class_index in rows
    ])
    labels = torch.tensor([int(class_index) for _payload, class_index in rows], dtype=torch.long)
    return images, labels


def _normalization(images_u8: torch.Tensor) -> tuple[float, float]:
    values = images_u8.float().mul(1.0 / 255.0)
    mean = float(values.mean())
    std = max(float(values.std(unbiased=False)), 1.0 / 255.0)
    return mean, std


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
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return "cuda" if image_bytes <= int(free_bytes * fraction) else "cpu"


def _cache_tensors(
    images_u8: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    cache_device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_device == "cuda":
        return images_u8.to(device), labels.to(device)
    if device.type == "cuda":
        return images_u8.pin_memory(), labels.pin_memory()
    return images_u8, labels


def _fetch_batch(
    images_u8: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images_u8.device.type == "cuda":
        index_device = indices.to(device)
        return images_u8.index_select(0, index_device), labels.index_select(0, index_device)
    return (
        images_u8.index_select(0, indices).to(device, non_blocking=True),
        labels.index_select(0, indices).to(device, non_blocking=True),
    )


def _rotate_batch(images: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected NCHW images, got {tuple(images.shape)}")
    radians = angles_deg.to(dtype=torch.float32) * (math.pi / 180.0)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    theta = torch.zeros((images.shape[0], 2, 3), device=images.device, dtype=torch.float32)
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


def _projective_augment_batch(
    images: torch.Tensor,
    *,
    max_perspective: float,
    max_shear: float,
    max_stretch: float,
    probability: float,
) -> torch.Tensor:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("projective_augment_probability must be in [0,1]")
    for name, value, limit in (
        ("perspective_augment", max_perspective, 0.25),
        ("shear_augment", max_shear, 0.25),
        ("stretch_augment", max_stretch, 0.30),
    ):
        if not 0.0 <= float(value) <= limit:
            raise ValueError(f"{name} must be in [0,{limit}]")
    if images.shape[0] == 0 or (
        max_perspective <= 0.0 and max_shear <= 0.0 and max_stretch <= 0.0
    ):
        return images

    batch, _channels, height, width = images.shape
    dtype = torch.float32
    x = torch.linspace(-1.0, 1.0, width, device=images.device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, height, device=images.device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base_x = grid_x.unsqueeze(0).expand(batch, -1, -1)
    base_y = grid_y.unsqueeze(0).expand(batch, -1, -1)

    def random_signed(maximum: float) -> torch.Tensor:
        if maximum <= 0.0:
            return torch.zeros((batch, 1, 1), device=images.device, dtype=dtype)
        return torch.empty((batch, 1, 1), device=images.device, dtype=dtype).uniform_(
            -maximum, maximum
        )

    stretch_x = 1.0 + random_signed(float(max_stretch))
    stretch_y = 1.0 + random_signed(float(max_stretch))
    shear_x = random_signed(float(max_shear))
    shear_y = random_signed(float(max_shear))
    perspective_x = random_signed(float(max_perspective))
    perspective_y = random_signed(float(max_perspective))
    denominator = 1.0 + perspective_x * base_x + perspective_y * base_y
    denominator = denominator.clamp_min(0.5)
    sample_x = (stretch_x * base_x + shear_x * base_y) / denominator
    sample_y = (shear_y * base_x + stretch_y * base_y) / denominator
    grid = torch.stack((sample_x, sample_y), dim=-1)
    if probability < 1.0:
        apply_mask = torch.rand((batch, 1, 1, 1), device=images.device) < probability
        identity = torch.stack((base_x, base_y), dim=-1)
        grid = torch.where(apply_mask, grid, identity)
    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _validate_parameters(parameters) -> None:
    if int(parameters["epochs"]) < 1:
        raise ValueError("epochs must be positive")
    if int(parameters["batch_size"]) < 2:
        raise ValueError("batch_size must be at least 2")
    if float(parameters["learning_rate"]) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(parameters["weight_decay"]) < 0.0:
        raise ValueError("weight_decay must not be negative")
    if not 0.0 <= float(parameters["rotation_augment_deg"]) <= 45.0:
        raise ValueError("rotation_augment_deg must be in [0,45]")
    if not 0.0 <= float(parameters["perspective_augment"]) <= 0.25:
        raise ValueError("perspective_augment must be in [0,0.25]")
    if not 0.0 <= float(parameters["shear_augment"]) <= 0.25:
        raise ValueError("shear_augment must be in [0,0.25]")
    if not 0.0 <= float(parameters["stretch_augment"]) <= 0.30:
        raise ValueError("stretch_augment must be in [0,0.30]")
    if not 0.0 <= float(parameters["projective_augment_probability"]) <= 1.0:
        raise ValueError("projective_augment_probability must be in [0,1]")
    _resolve_cache_device(
        str(parameters["cache_device"]),
        image_bytes=0,
        fraction=float(parameters["cache_vram_fraction"]),
    )


def train(context):
    parameters = context.parameters
    _validate_parameters(parameters)
    seed = int(context.seed)
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(parameters["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(parameters["tf32"])

    images_u8, labels = _load_split(context.corpus.root / "dataset.sqlite", "train")
    mean, std = _normalization(images_u8)
    cache_device = _resolve_cache_device(
        str(parameters["cache_device"]),
        image_bytes=int(images_u8.numel()),
        fraction=float(parameters["cache_vram_fraction"]),
    )
    images_u8, labels = _cache_tensors(
        images_u8,
        labels,
        device=device,
        cache_device=cache_device,
    )
    model = context.model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    epochs = int(parameters["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(parameters["learning_rate"]) * 0.05,
    )
    amp_enabled = bool(parameters["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    batch_size = int(parameters["batch_size"])
    rotation = float(parameters["rotation_augment_deg"])
    perspective = float(parameters["perspective_augment"])
    shear = float(parameters["shear_augment"])
    stretch = float(parameters["stretch_augment"])
    projective_probability = float(parameters["projective_augment_probability"])

    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch * 1_000_003)
        order = torch.randperm(images_u8.shape[0], generator=generator)
        model.train()
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        epoch_sample_count = 0
        for start in range(0, order.numel(), batch_size):
            indices = order[start : start + batch_size]
            batch_u8, target = _fetch_batch(
                images_u8,
                labels,
                indices,
                device=device,
            )
            batch = batch_u8.float().mul_(1.0 / 255.0)
            if rotation > 0.0:
                angles = torch.empty(batch.shape[0], device=device).uniform_(-rotation, rotation)
                batch = _rotate_batch(batch, angles)
            if projective_probability > 0.0 and (perspective > 0.0 or shear > 0.0 or stretch > 0.0):
                batch = _projective_augment_batch(
                    batch,
                    max_perspective=perspective,
                    max_shear=shear,
                    max_stretch=stretch,
                    probability=projective_probability,
                )
            batch = batch.sub(mean).div(std)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(batch)
                loss = F.cross_entropy(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sample_count = int(target.shape[0])
            epoch_loss_sum.add_(loss.detach().float(), alpha=sample_count)
            epoch_sample_count += sample_count
        scheduler.step()
        context.telemetry.report_scalar(
            group="optimization",
            series="cross_entropy_loss",
            value=float(epoch_loss_sum.item()) / epoch_sample_count,
            step=epoch + 1,
        )
    return model
