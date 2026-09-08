import torch

from mldb.src.runtime.executable_loader import load_architecture_build
from tools.recognition.train_tile_shape_classifier import (
    configure_cuda,
    load_training_cache,
    seed_everything,
    train_one_epoch,
)


def train(context):
    parameters = context.parameters
    seed_everything(int(context.seed))
    configure_cuda(tf32=bool(parameters["tf32"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tile-shape training")
    device = torch.device("cuda")
    cache = load_training_cache(
        context.corpus.artifact_path,
        device=device,
        cache_device=str(parameters["cache_device"]),
        cache_vram_fraction=float(parameters["cache_vram_fraction"]),
    )
    model = load_architecture_build(context.architecture)().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    epochs = int(parameters["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(parameters["learning_rate"]) * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(parameters["amp"]))
    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model,
            cache.splits["train"],
            optimizer=optimizer,
            scaler=scaler,
            batch_size=int(parameters["batch_size"]),
            device=device,
            mean=cache.mean,
            std=cache.std,
            rotation_augment_deg=float(parameters["rotation_augment_deg"]),
            perspective_augment=float(parameters["perspective_augment"]),
            shear_augment=float(parameters["shear_augment"]),
            stretch_augment=float(parameters["stretch_augment"]),
            projective_augment_probability=float(parameters["projective_augment_probability"]),
            amp=bool(parameters["amp"]),
            epoch=epoch,
            seed=int(context.seed),
        )
        scheduler.step()
    return model
