from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nanodet.data.batch_process import stack_batch_img


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    workers: int
    prefetch_factor: int
    warmup_batches: int
    measured_batches: int
    image_size: int


class SyntheticDetectionDataset(Dataset):
    def __init__(self, length: int, image_size: int) -> None:
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        # Use a real allocation per sample so multiprocessing/storage behavior is
        # representative of NanoDet dataset output.
        image = torch.full(
            (3, self.image_size, self.image_size),
            float(index % 251) / 251.0,
            dtype=torch.float32,
        )
        count = 1 + (index % 100)
        boxes = np.zeros((count, 4), dtype=np.float32)
        labels = np.zeros((count,), dtype=np.int64)
        return {
            "img": image,
            "gt_bboxes": boxes,
            "gt_labels": labels,
            "img_info": {"id": index},
        }


def legacy_collate(batch: list[dict]) -> dict:
    elem = batch[0]
    return {
        key: [sample[key] for sample in batch]
        if not isinstance(elem[key], dict)
        else {
            nested_key: [sample[key][nested_key] for sample in batch]
            for nested_key in elem[key]
        }
        for key in elem
    }


def hybrid_collate(batch: list[dict]) -> dict:
    collated = legacy_collate(batch)
    collated["img"] = stack_batch_img(collated["img"], divisible=32)
    return collated


def assert_equivalent() -> None:
    batch = [
        {
            "img": torch.arange(3 * height * width, dtype=torch.float32).reshape(
                3, height, width
            ),
            "gt_bboxes": np.zeros((index + 1, 4), dtype=np.float32),
            "gt_labels": np.zeros((index + 1,), dtype=np.int64),
            "img_info": {"id": index},
        }
        for index, (height, width) in enumerate(
            [(320, 320), (287, 301), (319, 257), (256, 320)]
        )
    ]

    legacy = legacy_collate(batch)
    expected_images = stack_batch_img(legacy["img"], divisible=32)
    optimized = hybrid_collate(batch)

    torch.testing.assert_close(optimized["img"], expected_images, rtol=0, atol=0)
    assert isinstance(optimized["gt_bboxes"], list)
    assert isinstance(optimized["gt_labels"], list)
    assert len(optimized["gt_bboxes"]) == len(batch)
    assert len(optimized["gt_labels"]) == len(batch)
    assert optimized["img_info"]["id"] == list(range(len(batch)))

    print("equivalent: padded image batch")
    print("equivalent: variable-length targets remain lists")
    print("legacy_image_storages:", len(legacy["img"]))
    print("optimized_image_storages: 1")


def preprocess_legacy(batch: dict, device: torch.device) -> torch.Tensor:
    images = [image.to(device, non_blocking=True) for image in batch["img"]]
    return stack_batch_img(images, divisible=32)


def preprocess_optimized(batch: dict, device: torch.device) -> torch.Tensor:
    return batch["img"].to(device, non_blocking=True)


def benchmark_one(
    *,
    config: BenchmarkConfig,
    collate_fn,
    preprocess_fn,
    device: torch.device,
) -> float:
    total_batches = config.warmup_batches + config.measured_batches
    dataset = SyntheticDetectionDataset(
        length=config.batch_size * (total_batches + config.prefetch_factor + 2),
        image_size=config.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
        prefetch_factor=config.prefetch_factor if config.workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True,
    )

    iterator = iter(loader)
    for _ in range(config.warmup_batches):
        output = preprocess_fn(next(iterator), device)
        del output
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(config.measured_batches):
        output = preprocess_fn(next(iterator), device)
        del output
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    del iterator
    del loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--measured-batches", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_equivalent()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        warmup_batches=args.warmup_batches,
        measured_batches=args.measured_batches,
        image_size=args.image_size,
    )

    legacy_seconds = benchmark_one(
        config=config,
        collate_fn=legacy_collate,
        preprocess_fn=preprocess_legacy,
        device=device,
    )
    optimized_seconds = benchmark_one(
        config=config,
        collate_fn=hybrid_collate,
        preprocess_fn=preprocess_optimized,
        device=device,
    )

    images = config.batch_size * config.measured_batches
    print(f"device: {device}")
    print(f"legacy_seconds:    {legacy_seconds:.3f}")
    print(f"optimized_seconds: {optimized_seconds:.3f}")
    print(f"pipeline_speedup:  {legacy_seconds / optimized_seconds:.2f}x")
    print(f"legacy_images_per_second:    {images / legacy_seconds:.1f}")
    print(f"optimized_images_per_second: {images / optimized_seconds:.1f}")


if __name__ == "__main__":
    main()
