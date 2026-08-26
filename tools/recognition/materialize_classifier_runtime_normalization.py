from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np


BASE_RUNTIME_SPEC = "c8-tile-35-v1"
RED_FIVE_RUNTIME_SPEC = "c8-red-five-v1"


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Materialize code-owned classifier normalization constants for the selected "
            "production C8 runtime specs from the exact compact training databases."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--base-database",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "tile_classifier_datasets"
        / "gray35_jp500_seed42_v2.sqlite",
    )
    parser.add_argument(
        "--red-five-database",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "red_five_datasets"
        / "rgb64_binary_jp5000_seed42.sqlite",
    )
    parser.add_argument(
        "--runtime-specs",
        type=Path,
        default=repository_root
        / "product"
        / "frontend"
        / "src"
        / "recognition"
        / "model-runtime"
        / "runtime-specs.ts",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print computed constants without changing runtime-specs.ts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_database = args.base_database.resolve()
    red_five_database = args.red_five_database.resolve()
    runtime_specs = args.runtime_specs.resolve()

    base_mean, base_std = compute_base_normalization(base_database)
    red_mean, red_std = compute_red_five_rgb_normalization(red_five_database)

    print(f"base database: {base_database}")
    print(f"  mean = {base_mean:.17g}")
    print(f"  std  = {base_std:.17g}")
    print(f"red-five database: {red_five_database}")
    print(f"  mean = {[float(value) for value in red_mean]}")
    print(f"  std  = {[float(value) for value in red_std]}")

    if args.check:
        return

    source = runtime_specs.read_text(encoding="utf-8")
    updated = replace_runtime_spec_normalization(
        source,
        runtime_spec=BASE_RUNTIME_SPEC,
        mean=(base_mean,),
        std=(base_std,),
    )
    updated = replace_runtime_spec_normalization(
        updated,
        runtime_spec=RED_FIVE_RUNTIME_SPEC,
        mean=tuple(float(value) for value in red_mean),
        std=tuple(float(value) for value in red_std),
    )
    if updated == source:
        raise RuntimeError("runtime-specs.ts was not changed; expected unmaterialized constants")
    runtime_specs.write_text(updated, encoding="utf-8", newline="\n")
    print(f"updated: {runtime_specs}")


def compute_base_normalization(database: Path) -> tuple[float, float]:
    if not database.is_file():
        raise FileNotFoundError(database)

    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=60)
    try:
        metadata = {
            str(key): str(value)
            for key, value in connection.execute(
                "SELECT key, value FROM experiment_metadata"
            )
        }
        image_size = int(metadata["image_size"])
        count = int(
            connection.execute(
                "SELECT COUNT(*) FROM sample WHERE split = 'train'"
            ).fetchone()[0]
        )
        if count < 1:
            raise ValueError(f"No training images found in {database}")
        images = np.empty((count, image_size, image_size), dtype=np.uint8)
        rows = connection.execute(
            "SELECT image_gray_u8 FROM sample WHERE split = 'train' ORDER BY sample_id"
        )
        expected_bytes = image_size * image_size
        for index, (raw_value,) in enumerate(rows):
            raw = bytes(raw_value)
            if len(raw) != expected_bytes:
                raise ValueError(
                    f"Base training blob has {len(raw)} bytes; expected {expected_bytes}"
                )
            images[index] = np.frombuffer(raw, dtype=np.uint8).reshape(
                image_size, image_size
            )
    finally:
        connection.close()

    # Mirror train_tile_shape_classifier.load_training_cache exactly.
    mean = float(images.mean(dtype=np.float64) / 255.0)
    std = float(images.std(dtype=np.float64) / 255.0)
    std = max(std, 1.0 / 255.0)
    return mean, std


def compute_red_five_rgb_normalization(
    database: Path,
    *,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    if not database.is_file():
        raise FileNotFoundError(database)

    channel_sum = np.zeros((3,), dtype=np.float64)
    channel_square_sum = np.zeros((3,), dtype=np.float64)
    total_weighted_pixels = 0.0
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=60)
    try:
        metadata = {
            str(key): str(value)
            for key, value in connection.execute(
                "SELECT key, value FROM experiment_metadata"
            )
        }
        image_size = int(metadata["image_size"])
        cursor = connection.execute(
            """
            SELECT image_rgb_u8, train_repeat
            FROM sample
            WHERE split = 'train'
            ORDER BY sample_id
            """
        )
        expected_bytes = image_size * image_size * 3
        while True:
            rows = cursor.fetchmany(block_size)
            if not rows:
                break
            images = np.empty((len(rows), image_size, image_size, 3), dtype=np.uint8)
            repeats = np.empty((len(rows),), dtype=np.int64)
            for index, (raw_value, repeat_value) in enumerate(rows):
                raw = bytes(raw_value)
                if len(raw) != expected_bytes:
                    raise ValueError(
                        f"Red-five training blob has {len(raw)} bytes; expected {expected_bytes}"
                    )
                repeat = int(repeat_value)
                if repeat < 1:
                    raise ValueError("train_repeat must be >= 1")
                images[index] = np.frombuffer(raw, dtype=np.uint8).reshape(
                    image_size, image_size, 3
                )
                repeats[index] = repeat

            # Mirror train_red_five_classifier.compute_input_statistics for RGB.
            represented = images.astype(np.float32) * (1.0 / 255.0)
            weights = repeats.astype(np.float64)
            pixel_count = represented.shape[1] * represented.shape[2]
            per_sample_sum = represented.sum(axis=(1, 2), dtype=np.float64)
            per_sample_square_sum = np.square(represented, dtype=np.float64).sum(
                axis=(1, 2), dtype=np.float64
            )
            channel_sum += (per_sample_sum * weights[:, None]).sum(axis=0)
            channel_square_sum += (
                per_sample_square_sum * weights[:, None]
            ).sum(axis=0)
            total_weighted_pixels += float(weights.sum() * pixel_count)
    finally:
        connection.close()

    if total_weighted_pixels <= 0:
        raise ValueError(f"No weighted training pixels found in {database}")
    mean = channel_sum / total_weighted_pixels
    variance = np.maximum(
        channel_square_sum / total_weighted_pixels - mean * mean,
        0.0,
    )
    std = np.maximum(np.sqrt(variance), 1.0 / 255.0)
    return mean, std


def replace_runtime_spec_normalization(
    source: str,
    *,
    runtime_spec: str,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> str:
    block_pattern = re.compile(
        rf"('{re.escape(runtime_spec)}':\s*\{{.*?role:\s*'[^']+',)(.*?)(\n\s*\}},)",
        re.DOTALL,
    )
    match = block_pattern.search(source)
    if match is None:
        raise RuntimeError(f"Runtime spec block not found: {runtime_spec}")

    middle = match.group(2)
    normalization_pattern = re.compile(
        r"\n\s*(?://[^\n]*\n\s*)*classifierNormalization:\s*(?:null|\{.*?\}),",
        re.DOTALL,
    )
    if normalization_pattern.search(middle) is None:
        raise RuntimeError(
            f"classifierNormalization field not found in runtime spec: {runtime_spec}"
        )

    indentation = "    "
    replacement = (
        f"\n{indentation}classifierNormalization: {{\n"
        f"{indentation}  mean: {format_number_array(mean)},\n"
        f"{indentation}  std: {format_number_array(std)},\n"
        f"{indentation}}},"
    )
    replaced_middle = normalization_pattern.sub(replacement, middle, count=1)
    return source[: match.start(2)] + replaced_middle + source[match.end(2) :]


def format_number_array(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(format(value, ".17g") for value in values) + "]"


if __name__ == "__main__":
    main()
