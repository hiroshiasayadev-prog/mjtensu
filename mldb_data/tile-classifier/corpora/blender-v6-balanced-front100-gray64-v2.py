from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LABELS = tuple(
    [f"{n}{s}" for s in "mps" for n in range(1, 10)]
    + ["east", "south", "west", "north", "white", "green", "red"]
)
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
RED_TO_BASE = {"red5m": "5m", "red5p": "5p", "red5s": "5s"}
IMAGE_SIZE = 64


def _base_label(value: str) -> str:
    return RED_TO_BASE.get(value, value)


def _preprocess_gray_u8(image: Image.Image) -> tuple[np.ndarray, int, int]:
    gray = image.convert("L")
    width, height = gray.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid crop size: {gray.size}")
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = max(1, min(IMAGE_SIZE, int(math.floor(width * scale + 0.5))))
    resized_height = max(1, min(IMAGE_SIZE, int(math.floor(height * scale + 0.5))))
    resized = gray.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.uint8)
    if width == 1 or height == 1:
        fill = int(np.median(pixels))
    else:
        border = np.concatenate([pixels[0, :], pixels[-1, :], pixels[1:-1, 0], pixels[1:-1, -1]])
        fill = int(np.median(border))
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=fill)
    canvas.paste(resized, ((IMAGE_SIZE - resized_width) // 2, (IMAGE_SIZE - resized_height) // 2))
    return np.asarray(canvas, dtype=np.uint8), width, height


def _rank(seed: int, annotation: dict[str, Any]) -> str:
    payload = f"{seed}:{annotation['image_id']}:{annotation['id']}:{annotation.get('tile_identity','')}".encode()
    return hashlib.sha256(payload).hexdigest()


def _select_balanced(annotations: list[dict[str, Any]], *, per_class: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        if ann.get("face") != "front":
            continue
        identity = ann.get("tile_identity")
        if not isinstance(identity, str):
            continue
        label = _base_label(identity)
        if label not in LABEL_TO_INDEX:
            continue
        bbox = ann.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4 and float(bbox[2]) > 1 and float(bbox[3]) > 1):
            continue
        grouped[label].append(ann)

    missing = [label for label in LABELS if len(grouped[label]) < per_class]
    if missing:
        raise ValueError(f"insufficient source samples for {missing}")

    selected: dict[str, list[dict[str, Any]]] = {}
    for label in LABELS:
        ordered = sorted(grouped[label], key=lambda ann: (_rank(seed, ann), int(ann["image_id"]), int(ann["id"])))
        chosen: list[dict[str, Any]] = []
        chosen_ids: set[int] = set()
        image_counts: Counter[int] = Counter()
        cap = 1
        while len(chosen) < per_class:
            progressed = False
            for ann in ordered:
                aid = int(ann["id"])
                image_id = int(ann["image_id"])
                if aid in chosen_ids or image_counts[image_id] >= cap:
                    continue
                chosen.append(ann)
                chosen_ids.add(aid)
                image_counts[image_id] += 1
                progressed = True
                if len(chosen) == per_class:
                    break
            if len(chosen) == per_class:
                break
            if not progressed and cap > len(ordered):
                raise RuntimeError(f"could not select {per_class} samples for {label}")
            cap += 1
        selected[label] = chosen
    return selected


def build(source_tar: str | Path, output_root: str | Path, *, per_class: int = 100, seed: int = 42,
          normalization_mean: float = 0.6815832403977486,
          normalization_std: float = 0.27255536819738574) -> None:
    source_tar = Path(source_tar)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if per_class < 1:
        raise ValueError("per_class must be positive")

    with tarfile.open(source_tar, "r") as tf:
        coco = json.load(tf.extractfile("annotations/synthetic_val.coco.json"))
        images_by_id = {int(row["id"]): row for row in coco["images"]}
        selected = _select_balanced(coco["annotations"], per_class=per_class, seed=seed)

        images: list[np.ndarray] = []
        labels: list[int] = []
        widths: list[int] = []
        heights: list[int] = []
        metadata: list[dict[str, Any]] = []
        for label in LABELS:
            for ann in selected[label]:
                image_row = images_by_id[int(ann["image_id"])]
                file_name = str(image_row["file_name"])
                member = f"images/synthetic/{file_name}"
                with Image.open(io.BytesIO(tf.extractfile(member).read())) as full:
                    x, y, w, h = map(float, ann["bbox"])
                    left = max(0, int(math.floor(x)))
                    top = max(0, int(math.floor(y)))
                    right = min(full.width, int(math.ceil(x + w)))
                    bottom = min(full.height, int(math.ceil(y + h)))
                    if right <= left or bottom <= top:
                        raise ValueError(f"invalid clipped bbox for annotation {ann['id']}")
                    crop = full.crop((left, top, right, bottom))
                    processed, crop_w, crop_h = _preprocess_gray_u8(crop)
                images.append(processed)
                labels.append(LABEL_TO_INDEX[label])
                widths.append(crop_w)
                heights.append(crop_h)
                metadata.append({
                    "crop_id": f"blender-v6:{int(ann['image_id'])}:{int(ann['id'])}",
                    "source": "blender_v6",
                    "source_partition": "synthetic_val",
                    "source_annotation_id": str(ann["id"]),
                    "source_image_path": member,
                    "image_id": int(ann["image_id"]),
                    "tile_identity": str(ann["tile_identity"]),
                    "base_label": label,
                    "region": ann.get("region"),
                    "bbox": [float(v) for v in ann["bbox"]],
                    "face": ann.get("face"),
                })

    images_np = np.stack(images).astype(np.uint8, copy=False)
    labels_np = np.asarray(labels, dtype=np.int64)
    widths_np = np.asarray(widths, dtype=np.int64)
    heights_np = np.asarray(heights, dtype=np.int64)
    if images_np.shape != (len(LABELS) * per_class, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"unexpected image tensor shape {images_np.shape}")
    counts = Counter(labels_np.tolist())
    if any(counts[i] != per_class for i in range(len(LABELS))):
        raise RuntimeError(f"class balance failure: {counts}")

    np.savez_compressed(
        output_root / "shard-0000.npz",
        images=images_np,
        labels=labels_np,
        widths=widths_np,
        heights=heights_np,
    )
    metadata_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in metadata
    )
    (output_root / "shard-0000.jsonl.gz").write_bytes(
        gzip.compress(metadata_bytes, compresslevel=9, mtime=0)
    )

    index = {
        "schema": "mjtensu.recognition/blender-balanced-classifier-corpus-index/v1",
        "labels": list(LABELS) + ["invalid"],
        "true_labels": list(LABELS),
        "counts": {"included": len(images), "per_class": per_class, "classes": len(LABELS)},
        "normalization": {
            "mean": float(normalization_mean),
            "std": float(normalization_std),
            "source_corpus": "tile-classifier/gray35-jp500-seed42-v3-jp189-v1",
            "source_split": "train",
        },
        "preprocess": "grayscale_aspect_preserving_letterbox_border_median_lanczos_u8",
        "selection": {
            "source": "Blender v6 synthetic_val front-face annotations",
            "per_class": per_class,
            "seed": seed,
            "red_five_policy": "red5m->5m,red5p->5p,red5s->5s",
            "image_diversity": "deterministic hashed order with progressively relaxed per-image cap",
        },
    }
    (output_root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source_tar", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.source_tar, args.output_root, per_class=args.per_class, seed=args.seed)
