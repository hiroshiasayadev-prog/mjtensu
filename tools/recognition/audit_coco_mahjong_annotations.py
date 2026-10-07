from __future__ import annotations

import argparse
import importlib.util
import csv
import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import onnxruntime as ort
except ImportError:  # allows mapping/correction unit tests without ORT
    ort = None  # type: ignore[assignment]

BASE_LABELS = (
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "east", "south", "west", "north", "white", "green", "red", "invalid",
)
RED_FIVE_BASE = {"red5m": "5m", "red5p": "5p", "red5s": "5s"}
BASE_TO_RED_FIVE = {"5m": "red5m", "5p": "red5p", "5s": "red5s"}
JP_NUMERIC_TILE_LABELS = (
    "1m", "2m", "3m", "4m", "5m", "red5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "red5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "red5s", "6s", "7s", "8s", "9s",
    "east", "south", "west", "north", "white", "green", "red",
)
JP_ALIASES = {"5mr": "red5m", "5pr": "red5p", "5sr": "red5s"}
HONORS = {"east", "south", "west", "north", "white", "green", "red"}
IMAGE_SIZE = 64
BASE_MEAN = 0.6815832403977466
BASE_STD = 0.2725553681973969
RED_MEAN = np.asarray([0.66025093606229934, 0.69172744263865471, 0.6489080530422624], dtype=np.float32)
RED_STD = np.asarray([0.30491469480493394, 0.24924454491506576, 0.27107025824445752], dtype=np.float32)
EXPECTED_BASE_RUNTIME = "gray64-tile-35-v1"
EXPECTED_RED_RUNTIME = "c8-red-five-v1"


@dataclass(frozen=True)
class SourceSpec:
    dataset_id: str
    split: str
    annotations_path: Path
    image_root: Path


@dataclass(frozen=True)
class CategoryEntry:
    category_id: int
    raw_name: str
    semantic_label: str | None
    family: str


@dataclass(frozen=True)
class CropResult:
    image: Image.Image
    pixel_box: tuple[int, int, int, int]
    clipped: bool
    requested_area: float
    visible_area: int


def build_sources(repository_root: Path) -> list[SourceSpec]:
    data = repository_root / "data"
    return [
        SourceSpec("coco_mahjong", "train2017", data / "coco_mahjong/annotations/instances_train2017.json", data / "coco_mahjong/train2017"),
        SourceSpec("coco_mahjong", "val2017", data / "coco_mahjong/annotations/instances_val2017.json", data / "coco_mahjong/val2017"),
        SourceSpec("coco_mahjong_jp_v2", "train", data / "coco_mahjong_jp_v2/train/_annotations.coco.json", data / "coco_mahjong_jp_v2/train"),
        SourceSpec("coco_mahjong_jp_v2", "valid", data / "coco_mahjong_jp_v2/valid/_annotations.coco.json", data / "coco_mahjong_jp_v2/valid"),
        SourceSpec("coco_mahjong_jp_v2", "test", data / "coco_mahjong_jp_v2/test/_annotations.coco.json", data / "coco_mahjong_jp_v2/test"),
    ]


def normalize_category_label(dataset_id: str, raw_name: str) -> str | None:
    name = raw_name.strip()
    if dataset_id == "coco_mahjong":
        for prefix, suit in (("circle_", "p"), ("bamboo_", "s"), ("character_", "m")):
            if name.startswith(prefix):
                number = name[len(prefix):]
                if number.isdigit() and 1 <= int(number) <= 9:
                    return f"{int(number)}{suit}"
        if name in HONORS:
            return name
        raise ValueError(f"Unsupported coco_mahjong category name: {raw_name!r}")

    if dataset_id == "coco_mahjong_jp_v2":
        if name == "mahjong-tiles":
            return None
        if name.isdecimal():
            index = int(name)
            if 0 <= index < len(JP_NUMERIC_TILE_LABELS):
                return JP_NUMERIC_TILE_LABELS[index]
            raise ValueError(f"Unsupported Mahjong-jp numeric category: {raw_name!r}")
        name = JP_ALIASES.get(name, name)
        explicit = {f"{n}{s}" for s in "mps" for n in range(1, 10)} | HONORS | set(RED_FIVE_BASE)
        if name in explicit:
            return name
        raise ValueError(f"Unsupported coco_mahjong_jp_v2 category name: {raw_name!r}")

    raise ValueError(f"Unsupported dataset: {dataset_id}")


def category_family(raw_name: str) -> str:
    return "numeric" if raw_name.strip().isdecimal() else "named"


def build_category_map(payload: dict[str, Any], dataset_id: str) -> tuple[dict[int, CategoryEntry], dict[str, Any]]:
    raw_categories = payload.get("categories")
    if not isinstance(raw_categories, list):
        raise ValueError("COCO categories must be a list")

    by_id: dict[int, CategoryEntry] = {}
    ids: Counter[int] = Counter()
    names: Counter[str] = Counter()
    duplicate_details: dict[int, list[str]] = defaultdict(list)
    semantic_to_ids: dict[str, set[int]] = defaultdict(set)

    for raw in raw_categories:
        if not isinstance(raw, dict):
            raise ValueError("COCO category entry must be an object")
        category_id = int(raw["id"])
        raw_name = str(raw["name"])
        semantic = normalize_category_label(dataset_id, raw_name)
        entry = CategoryEntry(category_id, raw_name, semantic, category_family(raw_name))
        ids[category_id] += 1
        names[raw_name] += 1
        duplicate_details[category_id].append(raw_name)
        if semantic is not None:
            semantic_to_ids[semantic].add(category_id)

        previous = by_id.get(category_id)
        if previous is not None:
            if previous.semantic_label != semantic:
                raise ValueError(
                    f"Ambiguous duplicate category id {category_id}: "
                    f"{previous.raw_name!r}->{previous.semantic_label!r} vs {raw_name!r}->{semantic!r}"
                )
            continue
        by_id[category_id] = entry

    report = {
        "category_count": len(raw_categories),
        "unique_category_id_count": len(by_id),
        "duplicate_category_ids": {
            str(k): {"count": v, "raw_names": duplicate_details[k]}
            for k, v in sorted(ids.items()) if v > 1
        },
        "duplicate_category_names": {k: v for k, v in sorted(names.items()) if v > 1},
        "semantic_label_to_category_ids": {
            k: sorted(v) for k, v in sorted(semantic_to_ids.items())
        },
        "categories": [
            {
                "id": int(raw["id"]),
                "name": str(raw["name"]),
                "semantic_label": normalize_category_label(dataset_id, str(raw["name"])),
                "family": category_family(str(raw["name"])),
            }
            for raw in raw_categories
        ],
    }
    return by_id, report


def safe_image_path(image_root: Path, file_name: str) -> Path:
    pure = PurePosixPath(file_name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe COCO file_name: {file_name}")
    path = image_root.joinpath(*pure.parts).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def extract_coco_crop(image: Image.Image, bbox: Sequence[float]) -> CropResult:
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox shape: {bbox!r}")
    x, y, width, height = (float(v) for v in bbox)
    if width <= 0 or height <= 0 or not all(math.isfinite(v) for v in (x, y, width, height)):
        raise ValueError(f"Invalid bbox values: {bbox!r}")
    raw_left = math.floor(x)
    raw_top = math.floor(y)
    raw_right = math.ceil(x + width)
    raw_bottom = math.ceil(y + height)
    left = max(0, raw_left)
    top = max(0, raw_top)
    right = min(image.width, raw_right)
    bottom = min(image.height, raw_bottom)
    if right <= left or bottom <= top:
        raise ValueError(f"Bounding box has no visible pixels: {bbox!r}, image={image.size}")
    clipped = (left, top, right, bottom) != (raw_left, raw_top, raw_right, raw_bottom)
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    return CropResult(crop, (left, top, right, bottom), clipped, width * height, crop.width * crop.height)


def _resized_size(width: int, height: int, image_size: int = IMAGE_SIZE) -> tuple[int, int]:
    scale = min(image_size / width, image_size / height)
    return (
        max(1, min(image_size, int(math.floor(width * scale + 0.5)))),
        max(1, min(image_size, int(math.floor(height * scale + 0.5)))),
    )


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _clamp_byte(value: float) -> int:
    return max(0, min(255, _round_half_up(value)))


def _lanczos(value: float) -> float:
    distance = abs(value)
    if distance < 1.0e-7:
        return 1.0
    if distance >= 3.0:
        return 0.0
    pi_x = math.pi * distance
    return (math.sin(pi_x) / pi_x) * (math.sin(pi_x / 3.0) / (pi_x / 3.0))


@lru_cache(maxsize=2048)
def _resample_contributions(
    source_size: int,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse runtime-equivalent Lanczos indices/weights (at most six taps)."""
    indices = np.zeros((target_size, 6), dtype=np.intp)
    weights = np.zeros((target_size, 6), dtype=np.float64)
    scale = source_size / target_size
    for target_index in range(target_size):
        source_coordinate = (target_index + 0.5) * scale - 0.5
        minimum = math.ceil(source_coordinate - 3.0 + 1.0)
        maximum = math.floor(source_coordinate + 3.0)
        tap_count = maximum - minimum + 1
        weight_sum = 0.0
        for tap, raw_source_index in enumerate(range(minimum, maximum + 1)):
            indices[target_index, tap] = max(
                0, min(source_size - 1, raw_source_index)
            )
            weight = _lanczos(source_coordinate - raw_source_index)
            weights[target_index, tap] = weight
            weight_sum += weight
        if weight_sum != 0.0:
            weights[target_index, :tap_count] /= weight_sum
    return indices, weights


def _runtime_resample_channel(source: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint8)
    source_height, source_width = source.shape
    if (source_width, source_height) == (target_width, target_height):
        return source.copy()
    x_indices, x_weights = _resample_contributions(source_width, target_width)
    y_indices, y_weights = _resample_contributions(source_height, target_height)

    # Match the frontend's separable Lanczos pass: float64 horizontal
    # accumulation, then float64 vertical accumulation, with byte rounding only
    # after the vertical pass.
    horizontal = (
        source.astype(np.float64)[:, x_indices] * x_weights[None, :, :]
    ).sum(axis=2)
    sampled = (
        horizontal[y_indices, :] * y_weights[:, :, None]
    ).sum(axis=1)
    return np.clip(np.floor(sampled + 0.5), 0, 255).astype(np.uint8)


def _runtime_resample_batch(
    source: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Vectorized runtime-equivalent resampling for an NxHxW equal-shape batch."""
    source = np.asarray(source, dtype=np.uint8)
    if source.ndim != 3:
        raise ValueError(f"Expected NxHxW batch, got {source.shape}")
    _, source_height, source_width = source.shape
    if (source_width, source_height) == (target_width, target_height):
        return source.copy()
    x_indices, x_weights = _resample_contributions(source_width, target_width)
    y_indices, y_weights = _resample_contributions(source_height, target_height)
    horizontal = (
        source.astype(np.float64)[:, :, x_indices] * x_weights[None, None, :, :]
    ).sum(axis=3)
    sampled = (
        horizontal[:, y_indices, :] * y_weights[None, :, :, None]
    ).sum(axis=2)
    return np.clip(np.floor(sampled + 0.5), 0, 255).astype(np.uint8)


def _runtime_rgb_pixels(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB crop, got {rgb.shape}")
        return rgb
    if image.mode == "RGB":
        return np.asarray(image, dtype=np.uint8)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _runtime_gray_pixels(rgb: np.ndarray) -> np.ndarray:
    values = (
        rgb[:, :, 0].astype(np.float64) * 0.299
        + rgb[:, :, 1].astype(np.float64) * 0.587
        + rgb[:, :, 2].astype(np.float64) * 0.114
    )
    return np.clip(np.floor(values + 0.5), 0, 255).astype(np.uint8)


def _border_values(channel: np.ndarray) -> np.ndarray:
    height, width = channel.shape
    if width == 1 or height == 1:
        return channel.reshape(-1)
    return np.concatenate(
        (channel[0, :], channel[-1, :], channel[1:-1, 0], channel[1:-1, -1])
    )


def _border_values_batch(channel: np.ndarray) -> np.ndarray:
    if channel.ndim != 3:
        raise ValueError(f"Expected NxHxW channel batch, got {channel.shape}")
    _, height, width = channel.shape
    if width == 1 or height == 1:
        return channel.reshape(channel.shape[0], -1)
    return np.concatenate(
        (
            channel[:, 0, :],
            channel[:, -1, :],
            channel[:, 1:-1, 0],
            channel[:, 1:-1, -1],
        ),
        axis=1,
    )


def _runtime_letterbox_channel(channel: np.ndarray) -> np.ndarray:
    height, width = channel.shape
    resized_width, resized_height = _resized_size(width, height)
    resized = _runtime_resample_channel(channel, resized_width, resized_height)
    border = _border_values(channel)
    fill = 127 if border.size == 0 else _clamp_byte(float(np.median(border)))
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE), fill, dtype=np.uint8)
    offset_x = (IMAGE_SIZE - resized_width) // 2
    offset_y = (IMAGE_SIZE - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def _runtime_letterbox_batch(channel: np.ndarray) -> np.ndarray:
    if channel.ndim != 3:
        raise ValueError(f"Expected NxHxW channel batch, got {channel.shape}")
    _, height, width = channel.shape
    resized_width, resized_height = _resized_size(width, height)
    resized = _runtime_resample_batch(channel, resized_width, resized_height)
    border = _border_values_batch(channel)
    fill = np.clip(np.floor(np.median(border, axis=1) + 0.5), 0, 255).astype(np.uint8)
    canvas = np.empty((channel.shape[0], IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    canvas[:] = fill[:, None, None]
    offset_x = (IMAGE_SIZE - resized_width) // 2
    offset_y = (IMAGE_SIZE - resized_height) // 2
    canvas[:, offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def preprocess_gray_crop(image: Image.Image | np.ndarray) -> np.ndarray:
    rgb = _runtime_rgb_pixels(image)
    gray = _runtime_gray_pixels(rgb)
    canvas = _runtime_letterbox_channel(gray).astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray((canvas - np.float32(BASE_MEAN)) / np.float32(BASE_STD))


def preprocess_rgb_crop(image: Image.Image | np.ndarray) -> np.ndarray:
    rgb = _runtime_rgb_pixels(image)
    channels = np.stack(
        [_runtime_letterbox_channel(rgb[:, :, index]) for index in range(3)], axis=0
    ).astype(np.float32)
    channels /= np.float32(255.0)
    normalized = (channels - RED_MEAN[:, None, None]) / RED_STD[:, None, None]
    return np.ascontiguousarray(normalized)


def preprocess_gray_batch(
    crops: Sequence[Image.Image | np.ndarray],
    *,
    executor: ThreadPoolExecutor | None,
    vectorized_min_group: int = 4,
    vectorized_chunk: int = 256,
) -> list[np.ndarray]:
    """Use equal-shape vectorization when profitable, scalar preprocessing otherwise."""
    results: list[np.ndarray | None] = [None] * len(crops)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, crop in enumerate(crops):
        if isinstance(crop, np.ndarray):
            height, width = int(crop.shape[0]), int(crop.shape[1])
        else:
            width, height = crop.size
        groups[(height, width)].append(index)

    scalar_indices: list[int] = []
    for indices in groups.values():
        if len(indices) < vectorized_min_group:
            scalar_indices.extend(indices)
            continue
        for start in range(0, len(indices), vectorized_chunk):
            chunk_indices = indices[start : start + vectorized_chunk]
            rgb = np.stack(
                [_runtime_rgb_pixels(crops[index]) for index in chunk_indices],
                axis=0,
            )
            gray_values = (
                rgb[:, :, :, 0].astype(np.float64) * 0.299
                + rgb[:, :, :, 1].astype(np.float64) * 0.587
                + rgb[:, :, :, 2].astype(np.float64) * 0.114
            )
            gray = np.clip(np.floor(gray_values + 0.5), 0, 255).astype(np.uint8)
            canvas = _runtime_letterbox_batch(gray).astype(np.float32)
            canvas /= np.float32(255.0)
            prepared = (canvas - np.float32(BASE_MEAN)) / np.float32(BASE_STD)
            for output_index, value in zip(chunk_indices, prepared, strict=True):
                results[output_index] = np.ascontiguousarray(value)

    if scalar_indices:
        scalar_crops = [crops[index] for index in scalar_indices]
        if executor is None:
            prepared_scalar = [preprocess_gray_crop(crop) for crop in scalar_crops]
        else:
            prepared_scalar = list(executor.map(preprocess_gray_crop, scalar_crops))
        for output_index, value in zip(scalar_indices, prepared_scalar, strict=True):
            results[output_index] = value

    if any(value is None for value in results):
        raise RuntimeError("Preprocessing batch left unresolved crops")
    return [value for value in results if value is not None]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def classify_review_status(
    annotated_label: str | None,
    predicted_label: str,
    confidence: float,
    margin: float,
    *,
    strong_confidence: float,
    strong_margin: float,
    low_confidence: float,
    low_margin: float,
) -> str:
    if annotated_label is None:
        return "unmapped_annotation"
    mismatch = annotated_label != predicted_label
    if mismatch and confidence >= strong_confidence and margin >= strong_margin:
        return "high_confidence_mismatch"
    if mismatch:
        return "ambiguous_mismatch"
    if confidence < low_confidence or margin < low_margin:
        return "low_confidence_match"
    return "match_confident"


def choose_correction_category_id(
    entries: dict[int, CategoryEntry],
    *,
    approved_label: str,
    original_category_id: int,
) -> int:
    if approved_label == "invalid":
        raise ValueError("invalid cannot be written as a tile category correction")
    candidates = [entry for entry in entries.values() if entry.semantic_label == approved_label]
    if not candidates:
        raise ValueError(f"No source category maps to approved label {approved_label!r}")
    original = entries.get(original_category_id)
    if original is not None:
        same_family = [entry for entry in candidates if entry.family == original.family]
        if same_family:
            candidates = same_family
    return min(entry.category_id for entry in candidates)


def atomic_write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            if compact:
                json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
            else:
                json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model_url(url: str) -> str:
    return url.split("?", 1)[0]


def load_production_models(production_root: Path) -> dict[str, Any]:
    model_set_path = production_root / "product/frontend/src/recognition/model-runtime/production-model-set.json"
    payload = load_json(model_set_path)
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"Invalid production model set: {model_set_path}")
    base = models["tile-classifier"]
    red = models["red-five-classifier"]
    if base.get("runtimeSpec") != EXPECTED_BASE_RUNTIME:
        raise ValueError(
            f"Production base runtime changed: {base.get('runtimeSpec')!r}; "
            f"audit preprocessing must be reconciled before running"
        )
    if red.get("runtimeSpec") != EXPECTED_RED_RUNTIME:
        raise ValueError(
            f"Production red-five runtime changed: {red.get('runtimeSpec')!r}; "
            f"audit preprocessing must be reconciled before running"
        )
    base_path = production_root / "vendor/recognition-models" / parse_model_url(str(base["url"]))
    red_path = production_root / "vendor/recognition-models" / parse_model_url(str(red["url"]))
    for path in (base_path, red_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(base_path) != str(base["sha256"]):
        raise ValueError(f"Base classifier SHA-256 mismatch: {base_path}")
    if sha256_file(red_path) != str(red["sha256"]):
        raise ValueError(f"Red-five classifier SHA-256 mismatch: {red_path}")
    return {
        "model_set_path": model_set_path,
        "model_set_version": payload.get("modelSetVersion"),
        "base": {**base, "path": base_path},
        "red": {**red, "path": red_path},
    }


def make_session(path: Path, providers: Sequence[str] | None = None) -> Any:
    if ort is None:
        raise RuntimeError("onnxruntime is required for audit execution")
    available = set(ort.get_available_providers())
    requested = list(providers or ("CUDAExecutionProvider", "CPUExecutionProvider"))
    selected = [provider for provider in requested if provider in available]
    if not selected:
        selected = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=selected)


class TorchSessionAdapter:
    """Small ORT-compatible adapter for the production checkpoint on CUDA."""

    class Input:
        name = "images"

    def __init__(self, model: Any, *, checkpoint: Path, checkpoint_sha256: str) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for the PyTorch audit backend")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.model = model.to(torch.device("cuda")).eval()
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self.input = self.Input()

    def get_inputs(self) -> list[Input]:
        return [self.input]

    def run(self, _outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        inputs = feeds[self.input.name]
        torch = self.torch
        with torch.inference_mode():
            tensor = torch.from_numpy(inputs).to(torch.device("cuda"))
            logits = self.model(tensor)
            return [logits.detach().cpu().numpy()]

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "pytorch-cuda-fp32",
            "device": self.torch.cuda.get_device_name(0),
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "tf32": False,
        }


def load_export_checkpoint(
    metadata_path: Path,
    *,
    expected_onnx_sha256: str,
) -> tuple[dict[str, Any], Path]:
    metadata = load_json(metadata_path)
    onnx = metadata.get("onnx")
    checkpoint = metadata.get("checkpoint")
    if not isinstance(onnx, dict) or not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid export metadata: {metadata_path}")
    if str(onnx.get("sha256")) != expected_onnx_sha256:
        raise ValueError(
            f"Export metadata ONNX SHA does not match production binding: {metadata_path}"
        )
    checkpoint_path = Path(str(checkpoint["path"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_sha256 = str(checkpoint["sha256"])
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint_path}")
    return metadata, checkpoint_path


def _load_architecture_module(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("audit_current_base_architecture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load architecture module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = getattr(module, "build", None)
    if not callable(build):
        raise ValueError(f"Architecture module has no callable build(): {path}")
    return module


def make_torch_sessions(
    models: dict[str, Any],
    *,
    base_architecture_path: Path,
    base_weights_path: Path,
    base_weights_sha256: str,
    red_metadata_path: Path,
) -> tuple[TorchSessionAdapter, TorchSessionAdapter, dict[str, Any]]:
    import torch

    if sha256_file(base_weights_path) != base_weights_sha256:
        raise ValueError(f"Base classifier weights SHA-256 mismatch: {base_weights_path}")
    architecture_module = _load_architecture_module(base_architecture_path)
    base_model = architecture_module.build().eval()
    try:
        base_state = torch.load(base_weights_path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        base_state = torch.load(base_weights_path, map_location="cpu")
    if not isinstance(base_state, dict) or not all(isinstance(key, str) for key in base_state):
        raise ValueError("Base weights are not a canonical pytorch-state-dict/v1 mapping")
    base_model.load_state_dict(base_state, strict=True)
    if base_model.training:
        raise RuntimeError("Base classifier must be in eval mode before the output-shape probe")
    with torch.inference_mode():
        probe = base_model(torch.zeros((2, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32))
    if tuple(probe.shape) != (2, len(BASE_LABELS)):
        raise ValueError(f"Base architecture output mismatch: got {tuple(probe.shape)}")

    red_metadata, red_checkpoint = load_export_checkpoint(
        red_metadata_path,
        expected_onnx_sha256=str(models["red"]["sha256"]),
    )
    red_payload = torch.load(red_checkpoint, map_location="cpu")

    try:
        from red_five_classifier import build_model as build_red_model
        from export_c8_classifiers_onnx import build_exported_model
    except ImportError:
        from tools.recognition.red_five_classifier import build_model as build_red_model
        from tools.recognition.export_c8_classifiers_onnx import build_exported_model

    red_config = red_payload["config"]
    if str(red_config.get("input_mode")) != "rgb":
        raise ValueError("Production red-five checkpoint is not RGB")
    red_fields = tuple(int(value) for value in red_config["model"]["c8_fields"] )
    red_model = build_red_model("rgb", c8_fields=red_fields)
    red_model.load_state_dict(red_payload["model_state_dict"])
    red_exported = build_exported_model(red_model)

    base_session = TorchSessionAdapter(
        base_model,
        checkpoint=base_weights_path,
        checkpoint_sha256=base_weights_sha256,
    )
    red_session = TorchSessionAdapter(
        red_exported,
        checkpoint=red_checkpoint,
        checkpoint_sha256=str(red_metadata["checkpoint"]["sha256"]),
    )
    provenance = {
        "base_architecture": str(base_architecture_path),
        "base_architecture_sha256": sha256_file(base_architecture_path),
        "base_weights_format": "pytorch-state-dict/v1",
        "base_runner": base_session.describe(),
        "red_export_metadata": str(red_metadata_path),
        "red_export_parity": red_metadata.get("parity"),
        "red_runner": red_session.describe(),
    }
    return base_session, red_session, provenance


def iter_image_records(payload: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO images/annotations must be lists")
    image_by_id: dict[int, dict[str, Any]] = {}
    for image in images:
        image_id = int(image["id"])
        if image_id in image_by_id:
            raise ValueError(f"Duplicate image id {image_id}")
        image_by_id[image_id] = image
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_annotation_ids: set[int] = set()
    for annotation in annotations:
        annotation_id = int(annotation["id"])
        if annotation_id in seen_annotation_ids:
            raise ValueError(f"Duplicate annotation id {annotation_id}")
        seen_annotation_ids.add(annotation_id)
        image_id = int(annotation["image_id"])
        if image_id not in image_by_id:
            raise ValueError(f"Annotation {annotation_id} references unknown image {image_id}")
        annotations_by_image[image_id].append(annotation)
    return image_by_id, annotations_by_image


def _record_confusion(summary: dict[str, Any], record: dict[str, Any]) -> None:
    summary["checked"] += 1
    status = record["review_status"]
    summary["status"][status] += 1
    label = record["annotated_label"] or "<unmapped>"
    summary["by_category"][label][status] += 1
    if record["annotated_label"] != record["predicted_label"]:
        summary["confusions"][(label, record["predicted_label"])] += 1
    if record["crop_clipped"]:
        summary["crop_clipped"] += 1


def finalize_batch(
    pending: list[dict[str, Any]],
    *,
    base_session: Any,
    red_session: Any,
    thresholds: dict[str, float],
    preprocess_executor: ThreadPoolExecutor | None = None,
) -> list[dict[str, Any]]:
    if not pending:
        return []
    crops = [record["_crop"] for record in pending]
    gray_inputs = preprocess_gray_batch(crops, executor=preprocess_executor)
    base_input = np.stack(gray_inputs)[:, None, :, :].astype(np.float32, copy=False)
    base_logits = np.asarray(base_session.run(None, {base_session.get_inputs()[0].name: base_input})[0])
    base_probs = softmax(base_logits)
    order = np.argsort(base_probs, axis=1)[:, ::-1]
    red_indices: list[int] = []

    for i, record in enumerate(pending):
        top1 = int(order[i, 0])
        top2 = int(order[i, 1])
        record["base_predicted_label"] = BASE_LABELS[top1]
        record["base_confidence"] = float(base_probs[i, top1])
        record["base_margin"] = float(base_probs[i, top1] - base_probs[i, top2])
        record["base_second_label"] = BASE_LABELS[top2]
        record["base_second_confidence"] = float(base_probs[i, top2])
        if (
            record["dataset_id"] == "coco_mahjong_jp_v2"
            and record["base_predicted_label"] in BASE_TO_RED_FIVE
        ):
            red_indices.append(i)

    red_outputs: dict[int, tuple[bool, float, float]] = {}
    if red_indices:
        red_crops = [pending[i]["_crop"] for i in red_indices]
        if preprocess_executor is None:
            rgb_inputs = [preprocess_rgb_crop(crop) for crop in red_crops]
        else:
            rgb_inputs = list(preprocess_executor.map(preprocess_rgb_crop, red_crops))
        rgb_input = np.stack(rgb_inputs).astype(np.float32, copy=False)
        red_logits = np.asarray(red_session.run(None, {red_session.get_inputs()[0].name: rgb_input})[0])
        red_probs = softmax(red_logits)
        red_order = np.argsort(red_probs, axis=1)[:, ::-1]
        for j, pending_index in enumerate(red_indices):
            top1 = int(red_order[j, 0])
            top2 = int(red_order[j, 1])
            red_outputs[pending_index] = (
                top1 == 1,
                float(red_probs[j, top1]),
                float(red_probs[j, top1] - red_probs[j, top2]),
            )

    for i, record in enumerate(pending):
        base_label = record["base_predicted_label"]
        predicted = base_label
        confidence = float(record["base_confidence"])
        margin = float(record["base_margin"])
        record["red_five_predicted"] = None
        record["red_five_confidence"] = None
        record["red_five_margin"] = None
        if i in red_outputs:
            is_red, red_conf, red_margin = red_outputs[i]
            record["red_five_predicted"] = "red" if is_red else "normal"
            record["red_five_confidence"] = red_conf
            record["red_five_margin"] = red_margin
            if record["dataset_id"] == "coco_mahjong_jp_v2":
                if is_red:
                    predicted = BASE_TO_RED_FIVE[base_label]
                confidence = min(confidence, red_conf)
                margin = min(margin, red_margin)
        record["predicted_label"] = predicted
        record["confidence"] = confidence
        record["margin"] = margin
        record["review_status"] = classify_review_status(
            record["annotated_label"],
            predicted,
            confidence,
            margin,
            strong_confidence=thresholds["strong_confidence"],
            strong_margin=thresholds["strong_margin"],
            low_confidence=thresholds["low_confidence"],
            low_margin=thresholds["low_margin"],
        )
        record.pop("_crop", None)
    return pending


def _new_summary() -> dict[str, Any]:
    return {
        "checked": 0,
        "status": Counter(),
        "by_category": defaultdict(Counter),
        "confusions": Counter(),
        "crop_clipped": 0,
    }


def audit_source(
    source: SourceSpec,
    *,
    base_session: Any,
    red_session: Any,
    batch_size: int,
    thresholds: dict[str, float],
    flagged_csv: Any,
    flagged_jsonl: Any,
    all_jsonl: Any | None,
    contact_candidates: list[dict[str, Any]],
    preprocess_workers: int,
) -> dict[str, Any]:
    payload = load_json(source.annotations_path)
    source_annotation_sha256 = sha256_file(source.annotations_path)
    categories, schema_report = build_category_map(payload, source.dataset_id)
    image_by_id, annotations_by_image = iter_image_records(payload)
    summary = _new_summary()
    pending: list[dict[str, Any]] = []
    retained_by_status: Counter[str] = Counter()
    next_progress = 100_000

    fieldnames = FLAGGED_COLUMNS

    def consume(records: list[dict[str, Any]]) -> None:
        nonlocal next_progress
        for record in records:
            _record_confusion(summary, record)
            if all_jsonl is not None:
                all_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["review_status"] == "match_confident":
                continue
            flagged_csv.writerow({key: record.get(key) for key in fieldnames})
            flagged_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            status = record["review_status"]
            retained_by_status[status] += 1
            # Keep review rendering bounded during million-annotation audits.
            if retained_by_status[status] <= 1000:
                contact_candidates.append(dict(record))
        if summary["checked"] >= next_progress:
            print(
                f"[audit] {source.dataset_id}/{source.split} "
                f"checked={summary['checked']}",
                flush=True,
            )
            while next_progress <= summary["checked"]:
                next_progress += 100_000

    preprocess_executor = (
        ThreadPoolExecutor(max_workers=preprocess_workers) if preprocess_workers > 1 else None
    )
    try:
        for image_id, image_record in image_by_id.items():
            image_path = safe_image_path(source.image_root, str(image_record["file_name"]))
            with Image.open(image_path) as opened:
                opened.load()
                image = ImageOps.exif_transpose(opened).convert("RGB")
            expected = (int(image_record.get("width", image.width)), int(image_record.get("height", image.height)))
            if image.size != expected:
                raise ValueError(f"Image size mismatch {image_path}: JSON={expected}, actual={image.size}")

            for annotation in annotations_by_image.get(image_id, []):
                category_id = int(annotation["category_id"])
                category = categories.get(category_id)
                if category is None:
                    raise ValueError(f"Unknown category id {category_id} in {source.annotations_path}")
                crop = extract_coco_crop(image, annotation["bbox"])
                annotated_label = category.semantic_label
                record = {
                    "dataset_id": source.dataset_id,
                    "split": source.split,
                    "source_annotation_sha256": source_annotation_sha256,
                    "annotation_id": int(annotation["id"]),
                    "image_id": image_id,
                    "image_file": str(image_record["file_name"]),
                    "image_path": str(image_path),
                    "category_id": category_id,
                    "raw_category_name": category.raw_name,
                    "annotated_label": annotated_label,
                    "annotated_base_label": None if annotated_label is None else RED_FIVE_BASE.get(annotated_label, annotated_label),
                    "bbox": [float(v) for v in annotation["bbox"]],
                    "crop_pixel_box": list(crop.pixel_box),
                    "crop_width": crop.image.width,
                    "crop_height": crop.image.height,
                    "crop_clipped": crop.clipped,
                    "crop_visible_over_requested_area": (
                        float(crop.visible_area / crop.requested_area) if crop.requested_area > 0 else 0.0
                    ),
                    "_crop": crop.image,
                }
                pending.append(record)
                if len(pending) >= batch_size:
                    consume(finalize_batch(
                        pending, base_session=base_session, red_session=red_session,
                        thresholds=thresholds, preprocess_executor=preprocess_executor,
                    ))
                    pending = []

        if pending:
            consume(finalize_batch(
                pending, base_session=base_session, red_session=red_session,
                thresholds=thresholds, preprocess_executor=preprocess_executor,
            ))
    finally:
        if preprocess_executor is not None:
            preprocess_executor.shutdown(wait=True)

    return {
        "dataset_id": source.dataset_id,
        "split": source.split,
        "annotation_path": str(source.annotations_path),
        "annotation_sha256": source_annotation_sha256,
        "image_root": str(source.image_root),
        "schema": schema_report,
        "checked": summary["checked"],
        "status": dict(summary["status"]),
        "crop_clipped": summary["crop_clipped"],
        "by_category": {k: dict(v) for k, v in sorted(summary["by_category"].items())},
        "confusions": [
            {"annotated": a, "predicted": p, "count": count}
            for (a, p), count in summary["confusions"].most_common(50)
        ],
        "contact_sheet_candidate_counts": dict(retained_by_status),
    }


FLAGGED_COLUMNS = (
    "dataset_id", "split", "source_annotation_sha256", "annotation_id",
    "image_id", "image_file", "image_path",
    "category_id", "raw_category_name", "annotated_label", "annotated_base_label",
    "predicted_label", "confidence", "margin",
    "base_predicted_label", "base_confidence", "base_margin", "base_second_label", "base_second_confidence",
    "red_five_predicted", "red_five_confidence", "red_five_margin",
    "review_status", "bbox", "crop_pixel_box", "crop_width", "crop_height",
    "crop_clipped", "crop_visible_over_requested_area",
)


def write_correction_template(flagged_jsonl_path: Path, output_path: Path) -> int:
    rows: list[dict[str, Any]] = []
    with flagged_jsonl_path.open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["review_status"] != "high_confidence_mismatch":
                continue
            if record["predicted_label"] == "invalid":
                continue
            rows.append(record)
    rows.sort(key=lambda row: (-float(row["confidence"]), -float(row["margin"])))
    columns = (
        "approve", "approved_label", "review_note", "dataset_id", "split",
        "source_annotation_sha256", "annotation_id", "annotated_label",
        "proposed_label", "confidence", "margin", "image_file", "bbox",
    )
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "approve": "",
                "approved_label": "",
                "review_note": "",
                "dataset_id": row["dataset_id"],
                "split": row["split"],
                "source_annotation_sha256": row["source_annotation_sha256"],
                "annotation_id": row["annotation_id"],
                "annotated_label": row["annotated_label"],
                "proposed_label": row["predicted_label"],
                "confidence": row["confidence"],
                "margin": row["margin"],
                "image_file": row["image_file"],
                "bbox": json.dumps(row["bbox"], separators=(",", ":")),
            })
    return len(rows)


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _context_image(record: dict[str, Any]) -> tuple[Image.Image, Image.Image]:
    with Image.open(record["image_path"]) as opened:
        opened.load()
        source = ImageOps.exif_transpose(opened).convert("RGB")
    x, y, width, height = (float(v) for v in record["bbox"])
    pad = max(width, height) * 1.1
    left = max(0, math.floor(x - pad))
    top = max(0, math.floor(y - pad))
    right = min(source.width, math.ceil(x + width + pad))
    bottom = min(source.height, math.ceil(y + height + pad))
    context = source.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(context)
    draw.rectangle(
        (
            _round_half_up(x - left),
            _round_half_up(y - top),
            _round_half_up(x + width - left),
            _round_half_up(y + height - top),
        ),
        outline="red",
        width=max(2, context.width // 120),
    )
    crop = extract_coco_crop(source, record["bbox"]).image
    return context, crop


def write_contact_sheets(
    candidates: list[dict[str, Any]],
    output_dir: Path,
    *,
    limit_per_status: int,
    columns: int = 3,
    rows: int = 4,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_source_status: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        key = (
            str(record["dataset_id"]),
            str(record["split"]),
            str(record["review_status"]),
        )
        by_source_status[key].append(record)
    outputs: list[str] = []
    cell_w, cell_h = 520, 360
    page_size = columns * rows
    for (dataset_id, split, status), records in sorted(by_source_status.items()):
        records.sort(key=lambda row: (-float(row["confidence"]), -float(row["margin"])))
        if limit_per_status >= 0:
            records = records[:limit_per_status]
        for page_index in range(0, len(records), page_size):
            page_records = records[page_index: page_index + page_size]
            sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), "white")
            for index, record in enumerate(page_records):
                context, crop = _context_image(record)
                x0 = (index % columns) * cell_w
                y0 = (index // columns) * cell_h
                sheet.paste(_fit(context, (320, 230)), (x0 + 5, y0 + 5))
                sheet.paste(_fit(crop, (180, 180)), (x0 + 335, y0 + 5))
                draw = ImageDraw.Draw(sheet)
                lines = [
                    f"{record['dataset_id']} / {record['split']} ann={record['annotation_id']}",
                    f"GT {record['annotated_label']}  ->  pred {record['predicted_label']}",
                    f"conf={float(record['confidence']):.4f} margin={float(record['margin']):.4f}",
                    f"raw={record['raw_category_name']} cat={record['category_id']}",
                    f"bbox={record['bbox']}",
                ]
                draw.multiline_text((x0 + 8, y0 + 242), "\n".join(lines), fill="black", spacing=3)
            path = output_dir / (
                f"{dataset_id}_{split}_{status}_{page_index // page_size + 1:04d}.jpg"
            )
            sheet.save(path, quality=90)
            outputs.append(str(path))
    return outputs


def merge_source_summaries(source_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    status: Counter[str] = Counter()
    by_dataset_split: dict[str, Any] = {}
    confusions: Counter[tuple[str, str]] = Counter()
    checked = 0
    clipped = 0
    for item in source_summaries:
        checked += int(item["checked"])
        clipped += int(item["crop_clipped"])
        status.update(item["status"])
        key = f"{item['dataset_id']}:{item['split']}"
        by_dataset_split[key] = {
            "checked": item["checked"],
            "status": item["status"],
            "by_category": item["by_category"],
        }
        for row in item["confusions"]:
            confusions[(row["annotated"], row["predicted"])] += int(row["count"])
    return {
        "annotations_checked": checked,
        "status_counts": dict(status),
        "crop_clipped_count": clipped,
        "by_source_split": by_dataset_split,
        "common_confusions": [
            {"annotated": a, "predicted": p, "count": c}
            for (a, p), c in confusions.most_common(50)
        ],
    }


def run_audit(args: argparse.Namespace) -> None:
    repository_root = args.repository_root.resolve()
    production_root = args.production_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_production_models(production_root)
    backend = str(args.backend)
    if backend == "auto":
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except ImportError:
            cuda_available = False
        backend = "torch-cuda" if cuda_available else "onnxruntime"

    if backend == "torch-cuda":
        base_session, red_session, inference_provenance = make_torch_sessions(
            models,
            base_architecture_path=args.base_architecture.resolve(),
            base_weights_path=args.base_weights.resolve(),
            base_weights_sha256=str(args.base_weights_sha256),
            red_metadata_path=args.red_export_metadata.resolve(),
        )
    elif backend == "onnxruntime":
        base_session = make_session(models["base"]["path"], providers=("CPUExecutionProvider",))
        red_session = make_session(models["red"]["path"], providers=("CPUExecutionProvider",))
        inference_provenance = {
            "base_runner": {
                "backend": "onnxruntime",
                "providers": base_session.get_providers(),
            },
            "red_runner": {
                "backend": "onnxruntime",
                "providers": red_session.get_providers(),
            },
        }
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    thresholds = {
        "strong_confidence": args.strong_confidence,
        "strong_margin": args.strong_margin,
        "low_confidence": args.low_confidence,
        "low_margin": args.low_margin,
    }

    requested_datasets = set(args.datasets)
    requested_splits = set(args.splits or [])
    sources = [
        source for source in build_sources(repository_root)
        if source.dataset_id in requested_datasets
        and (not requested_splits or source.split in requested_splits)
    ]
    if not sources:
        raise ValueError("No source splits selected")

    flagged_csv_path = output_dir / "flagged_samples.csv"
    flagged_jsonl_path = output_dir / "flagged_samples.jsonl"
    all_jsonl_path = output_dir / "all_results.jsonl" if args.write_all_jsonl else None
    contact_candidates: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []

    with (
        flagged_csv_path.open("w", encoding="utf-8", newline="") as csv_output,
        flagged_jsonl_path.open("w", encoding="utf-8") as jsonl_output,
    ):
        csv_writer = csv.DictWriter(csv_output, fieldnames=FLAGGED_COLUMNS)
        csv_writer.writeheader()
        all_output = all_jsonl_path.open("w", encoding="utf-8") if all_jsonl_path is not None else None
        try:
            for source in sources:
                print(f"[audit] {source.dataset_id}/{source.split}: {source.annotations_path}", flush=True)
                source_summary = audit_source(
                    source,
                    base_session=base_session,
                    red_session=red_session,
                    batch_size=args.batch_size,
                    thresholds=thresholds,
                    flagged_csv=csv_writer,
                    flagged_jsonl=jsonl_output,
                    all_jsonl=all_output,
                    contact_candidates=contact_candidates,
                    preprocess_workers=int(args.preprocess_workers),
                )
                source_summaries.append(source_summary)
                print(
                    f"[audit] checked={source_summary['checked']} status={source_summary['status']}",
                    flush=True,
                )
        finally:
            if all_output is not None:
                all_output.close()

    template_count = write_correction_template(
        flagged_jsonl_path, output_dir / "correction_review_template.csv"
    )
    sheet_paths = write_contact_sheets(
        contact_candidates,
        output_dir / "contact_sheets",
        limit_per_status=int(args.contact_sheet_limit_per_status),
    )
    merged = merge_source_summaries(source_summaries)
    summary = {
        "schema_version": 1,
        "method": {
            "production_model_set": str(models["model_set_path"]),
            "production_model_set_version": models["model_set_version"],
            "inference_backend": inference_provenance,
            "base_classifier": {
                "path": str(models["base"]["path"]),
                "sha256": models["base"]["sha256"],
                "runtime_spec": models["base"]["runtimeSpec"],
                "labels": list(BASE_LABELS),
                "normalization": {"mean": BASE_MEAN, "std": BASE_STD},
            },
            "red_five_classifier": {
                "path": str(models["red"]["path"]),
                "sha256": models["red"]["sha256"],
                "runtime_spec": models["red"]["runtimeSpec"],
                "normalization": {"mean": RED_MEAN.tolist(), "std": RED_STD.tolist()},
            },
            "preprocessing": (
                "source COCO bbox -> floor/ceil clipped RGB crop -> aspect-preserving "
                "64x64 Lanczos resize -> border-median letterbox -> production normalization"
            ),
            "semantic_confidence_rule": (
                "non-five: base softmax confidence/margin; five: min(base, red-five specialist)"
            ),
            "thresholds": thresholds,
        },
        "sources": source_summaries,
        "totals": merged,
        "artifacts": {
            "flagged_csv": str(flagged_csv_path),
            "flagged_jsonl": str(flagged_jsonl_path),
            "correction_review_template": str(output_dir / "correction_review_template.csv"),
            "correction_template_candidates": template_count,
            "contact_sheets": sheet_paths,
            "all_results_jsonl": None if all_jsonl_path is None else str(all_jsonl_path),
        },
        "interpretation": (
            "Classifier disagreement is an audit signal only. No source annotation is "
            "modified by audit mode; corrections require an explicit approved manifest."
        ),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps({"totals": merged, "artifacts": summary["artifacts"]}, ensure_ascii=False, indent=2))


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "approve", "approved"}


def run_apply_corrections(args: argparse.Namespace) -> None:
    repository_root = args.repository_root.resolve()
    selected = [
        source for source in build_sources(repository_root)
        if source.dataset_id == args.dataset_id and source.split == args.split
    ]
    if len(selected) != 1:
        raise ValueError(f"Unknown dataset/split: {args.dataset_id}/{args.split}")
    source = selected[0]
    output_json = args.output_json.resolve()
    if output_json == source.annotations_path.resolve():
        raise ValueError("Refusing to overwrite the source annotation JSON")

    payload = load_json(source.annotations_path)
    entries, _ = build_category_map(payload, source.dataset_id)
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("COCO annotations must be a list")
    by_id = {int(annotation["id"]): annotation for annotation in annotations}
    changes: list[dict[str, Any]] = []
    seen: set[int] = set()
    source_sha256 = sha256_file(source.annotations_path)

    with args.approved_manifest.resolve().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "approve", "approved_label", "dataset_id", "split",
            "source_annotation_sha256", "annotation_id",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Approved manifest is missing columns: {sorted(required)}")
        for row in reader:
            if not _truthy(row.get("approve", "")):
                continue
            if row["dataset_id"] != source.dataset_id or row["split"] != source.split:
                continue
            if row["source_annotation_sha256"].strip() != source_sha256:
                raise ValueError(
                    "Approved manifest source SHA-256 does not match current annotation file"
                )
            annotation_id = int(row["annotation_id"])
            if annotation_id in seen:
                raise ValueError(f"Duplicate approved correction for annotation {annotation_id}")
            seen.add(annotation_id)
            annotation = by_id.get(annotation_id)
            if annotation is None:
                raise ValueError(f"Approved annotation id not found: {annotation_id}")
            approved_label = row["approved_label"].strip()
            if not approved_label:
                raise ValueError(f"Approved row {annotation_id} has an empty approved_label")
            old_category_id = int(annotation["category_id"])
            new_category_id = choose_correction_category_id(
                entries,
                approved_label=approved_label,
                original_category_id=old_category_id,
            )
            if new_category_id == old_category_id:
                continue
            annotation["category_id"] = new_category_id
            changes.append({
                "annotation_id": annotation_id,
                "old_category_id": old_category_id,
                "new_category_id": new_category_id,
                "approved_label": approved_label,
                "review_note": row.get("review_note", ""),
            })

    atomic_write_json(output_json, payload, compact=True)
    report = {
        "source_json": str(source.annotations_path),
        "source_sha256": source_sha256,
        "output_json": str(output_json),
        "output_sha256": sha256_file(output_json),
        "approved_change_count": len(changes),
        "changes": changes,
    }
    atomic_write_json(output_json.with_suffix(output_json.suffix + ".corrections.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Audit legacy COCO Mahjong tile annotations with the current bound tile classifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--repository-root", type=Path, default=repository_root)
    audit.add_argument(
        "--production-root",
        type=Path,
        default=repository_root,
        help="Source tree whose production-model-set.json defines the current classifier binding.",
    )
    audit.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / ".local/recognition/coco_annotation_audit/current-plain",
    )
    audit.add_argument(
        "--backend",
        choices=("auto", "torch-cuda", "onnxruntime"),
        default="auto",
        help="Use the provenance-linked production checkpoint on CUDA when available.",
    )
    audit.add_argument(
        "--base-architecture",
        type=Path,
        default=repository_root / "mldb_data/tile-classifier/architectures/tile-plain-gray35-w500-late256-late-dw3-pw1-v1.py",
    )
    audit.add_argument(
        "--base-weights",
        type=Path,
        default=repository_root / ".local/recognition/current_plain_audit/weights.pt",
    )
    audit.add_argument(
        "--base-weights-sha256",
        default="132e415167c1d8fb752e62c3f0de29fd568e5077589fffdcf61afb70b29c5cc5",
    )
    audit.add_argument(
        "--red-export-metadata",
        type=Path,
        default=repository_root
        / ".local/recognition/red_five_runs/c8_rgb_cr_ycr_warmaug_seed42"
        / "c8_rgb_warmaug_rot22p5_seed42"
        / "red-five-c8-rgb-warmaug.onnx.metadata.json",
    )
    audit.add_argument(
        "--datasets",
        nargs="+",
        choices=("coco_mahjong", "coco_mahjong_jp_v2"),
        default=["coco_mahjong", "coco_mahjong_jp_v2"],
    )
    audit.add_argument("--splits", nargs="*", default=None)
    audit.add_argument("--batch-size", type=int, default=2048)
    audit.add_argument("--preprocess-workers", type=int, default=min(8, os.cpu_count() or 1))
    audit.add_argument("--strong-confidence", type=float, default=0.90)
    audit.add_argument("--strong-margin", type=float, default=0.25)
    audit.add_argument("--low-confidence", type=float, default=0.60)
    audit.add_argument("--low-margin", type=float, default=0.10)
    audit.add_argument("--contact-sheet-limit-per-status", type=int, default=240)
    audit.add_argument("--write-all-jsonl", action="store_true")
    audit.set_defaults(func=run_audit)

    apply_parser = subparsers.add_parser("apply-corrections")
    apply_parser.add_argument("--repository-root", type=Path, default=repository_root)
    apply_parser.add_argument("--dataset-id", choices=("coco_mahjong", "coco_mahjong_jp_v2"), required=True)
    apply_parser.add_argument("--split", required=True)
    apply_parser.add_argument("--approved-manifest", type=Path, required=True)
    apply_parser.add_argument("--output-json", type=Path, required=True)
    apply_parser.set_defaults(func=run_apply_corrections)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
