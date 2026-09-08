from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CORPUS_ID = "rotated-obb-human-review-v1"
DEFAULT_SOURCE = Path(".local/recognition/rotated_detector_corpus/annotations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the MLDB rotated OBB Corpus SQLite.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source_dir = (args.source_dir or (repo_root / DEFAULT_SOURCE)).resolve()
    output = (args.output or (repo_root / "mldb_data" / "corpora" / f"{CORPUS_ID}.sqlite")).resolve()
    rows = []
    for split, filename in (("train", "train.json"), ("val", "val.json")):
        payload = json.loads((source_dir / filename).read_text(encoding="utf-8"))
        category_names = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
        annotations_by_image: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(
                {
                    "label": category_names[int(annotation["category_id"])],
                    "obb": [float(value) for value in annotation["obb"]],
                }
            )
        for image in payload["images"]:
            image_id = int(image["id"])
            image_path = _resolve_repo_path(repo_root, str(image["file_name"]))
            with Image.open(image_path) as opened:
                rgb = opened.convert("RGB")
                if rgb.size != (320, 320):
                    raise ValueError(f"expected 320x320 image: {image_path}")
                array = np.asarray(rgb, dtype=np.uint8)
            chw = np.ascontiguousarray(array.transpose(2, 0, 1))
            sample_id = str(image.get("capture_id") or f"{split}-{image_id}")
            rows.append(
                (
                    sample_id,
                    split,
                    json.dumps(annotations_by_image[image_id], separators=(",", ":")),
                    chw.tobytes(order="C"),
                    str(image.get("file_name", "")),
                    str(image.get("layout_id", "")),
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with sqlite3.connect(output) as connection:
        connection.execute(
            """
            CREATE TABLE sample (
                sample_id TEXT PRIMARY KEY,
                split TEXT NOT NULL,
                annotations_json TEXT NOT NULL,
                image_rgb_u8 BLOB NOT NULL,
                source_image_path TEXT NOT NULL,
                layout_id TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO sample (
                sample_id, split, annotations_json, image_rgb_u8,
                source_image_path, layout_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.execute("CREATE INDEX sample_split_idx ON sample(split)")
        connection.commit()
    print(f"materialized {len(rows)} samples -> {output}")


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe repository-relative path: {value}")
    resolved = repo_root / path
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


if __name__ == "__main__":
    main()
