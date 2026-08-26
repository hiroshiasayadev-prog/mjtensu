from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


TRAIN_PREFETCH_FACTOR = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch NanoDet v1.0.0 so only the training DataLoader keeps workers "
            "alive and prefetches four batches per worker."
        )
    )
    parser.add_argument(
        "nanodet_root",
        type=Path,
        help="Path to the NanoDet repository root containing tools/train.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.nanodet_root.resolve() / "tools" / "train.py"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)

    text = train_path.read_text(encoding="utf-8")

    original = '''    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )
'''

    previous_patch = f'''    dataloader_worker_kwargs = {{}}
    if cfg.device.workers_per_gpu > 0:
        dataloader_worker_kwargs = {{
            "persistent_workers": True,
            "prefetch_factor": {TRAIN_PREFETCH_FACTOR},
        }}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
        **dataloader_worker_kwargs,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
        **dataloader_worker_kwargs,
    )
'''

    target = f'''    train_dataloader_worker_kwargs = {{}}
    if cfg.device.workers_per_gpu > 0:
        train_dataloader_worker_kwargs = {{
            "persistent_workers": True,
            "prefetch_factor": {TRAIN_PREFETCH_FACTOR},
        }}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
        **train_dataloader_worker_kwargs,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )
'''

    if target in text:
        result = "already_applied"
    elif text.count(previous_patch) == 1:
        train_path.write_text(
            text.replace(previous_patch, target, 1), encoding="utf-8", newline="\n"
        )
        result = "updated_previous_patch"
    elif text.count(original) == 1:
        train_path.write_text(
            text.replace(original, target, 1), encoding="utf-8", newline="\n"
        )
        result = "patched"
    else:
        raise RuntimeError(
            "Expected the original or previously patched NanoDet v1.0.0 "
            "DataLoader block exactly once; refusing to modify an unexpected source layout."
        )

    py_compile.compile(str(train_path), doraise=True)
    print(
        "NanoDet training DataLoader prefetch patch complete:\n"
        f"  result: {result}\n"
        "  train_persistent_workers: true\n"
        f"  train_prefetch_factor: {TRAIN_PREFETCH_FACTOR}\n"
        "  validation_persistent_workers: false\n"
        "  workers_per_gpu: unchanged (configured by YAML)"
    )


if __name__ == "__main__":
    main()
