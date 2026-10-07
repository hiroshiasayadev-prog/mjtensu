from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--val-images", type=int, default=1000)
    p.add_argument("--seed", type=int, default=760701)
    return p.parse_args()

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path: Path, value: Any, *, compact: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")

def partition(coco: dict[str, Any], selected_ids: set[int], description: str) -> dict[str, Any]:
    payload = {
        "info": dict(coco["info"]),
        "licenses": list(coco.get("licenses", [])),
        "images": [im for im in coco["images"] if int(im["id"]) in selected_ids],
        "annotations": [ann for ann in coco["annotations"] if int(ann["image_id"]) in selected_ids],
        "categories": list(coco["categories"]),
    }
    payload["info"]["description"] = description
    return payload

def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    all_path = root / "annotations" / "instances_all.json"
    coco = json.loads(all_path.read_text(encoding="utf-8"))
    images = sorted(coco["images"], key=lambda x: int(x["id"]))
    if args.val_images <= 0 or args.val_images >= len(images):
        raise ValueError("val-images must be between 1 and image_count-1")

    ids = [int(im["id"]) for im in images]
    shuffled = list(ids)
    random.Random(args.seed).shuffle(shuffled)
    val_ids = set(shuffled[:args.val_images])
    train_ids = set(ids) - val_ids
    train = partition(coco, train_ids, "Blender synthetic NanoDet training split")
    val = partition(coco, val_ids, "Blender synthetic NanoDet validation split")

    annotations = root / "annotations"
    train_path = annotations / "instances_train.json"
    val_path = annotations / "instances_val.json"
    write_json(train_path, train)
    write_json(val_path, val)

    key_files = [
        all_path,
        train_path,
        val_path,
        root / "manifest.jsonl",
        root / "provenance.json",
        root / "qa" / "validation_report.json",
        root / "qa" / "bbox_stats.txt",
        root / "qa" / "contact_sheets.json",
    ]
    checksums = {}
    for path in key_files:
        if path.is_file():
            checksums[str(path.relative_to(root))] = sha256(path)

    package = {
        "schema": "mjtensu.blender-synthetic-nanodet-package/v1",
        "corpus_root": str(root),
        "split_seed": args.seed,
        "counts": {
            "all_images": len(coco["images"]),
            "all_annotations": len(coco["annotations"]),
            "train_images": len(train["images"]),
            "train_annotations": len(train["annotations"]),
            "val_images": len(val["images"]),
            "val_annotations": len(val["annotations"]),
        },
        "paths": {
            "images": "images/",
            "records": "records/",
            "all_coco": "annotations/instances_all.json",
            "train_coco": "annotations/instances_train.json",
            "val_coco": "annotations/instances_val.json",
            "manifest": "manifest.jsonl",
            "provenance": "provenance.json",
            "qa": "qa/",
        },
        "checksums_sha256": checksums,
    }
    write_json(root / "package_manifest.json", package, compact=False)

    readme = f"""# Blender synthetic NanoDet corpus

This directory is a generated Mahjong tile detector corpus for the fixed 320x320 product composite.

- Images: {len(coco["images"])}
- Annotations: {len(coco["annotations"])}
- Train split: {len(train["images"])} images
- Validation split: {len(val["images"])} images
- Category: mahjong_tile (single class)
- Per-scene sidecars: records/
- Generator/config provenance: provenance.json
- Image hashes: manifest.jsonl
- QA: qa/

Composite destinations are fixed:
- completed_hand: x=7, y=0, width=306, height=72
- dora_indicators: x=7, y=74, width=306, height=72
- melds: x=74, y=148, width=172, height=172
- all pixels outside these destinations are black.

The train/validation split is deterministic from seed {args.seed}. Images are not duplicated between split files.
No detector training is included in this artifact.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(package, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
