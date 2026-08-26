from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


TARGET_CATEGORY = {
    "id": 1,
    "name": "mahjong_tile",
    "supercategory": "mahjong_tile",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert every COCO mahjong annotation category into one mahjong_tile class "
            "without modifying the source file."
        )
    )
    parser.add_argument("input_annotations", type=Path)
    parser.add_argument("output_annotations", type=Path)
    parser.add_argument(
        "--provenance-output",
        type=Path,
        help=(
            "Optional sidecar path. Defaults to the output file name with "
            ".category-provenance.json appended."
        ),
    )
    return parser.parse_args()


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"COCO root must be an object: {path}")
    for required_key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(required_key), list):
            raise ValueError(f"COCO field must be a list: {required_key}")
    return payload


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def validate_bbox(annotation: dict[str, Any]) -> None:
    annotation_id = annotation.get("id")
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Annotation {annotation_id} has an invalid bbox shape")
    _, _, width, height = bbox
    if float(width) <= 0.0 or float(height) <= 0.0:
        raise ValueError(f"Annotation {annotation_id} has a non-positive bbox")


def main() -> None:
    args = parse_args()
    input_path = args.input_annotations.resolve()
    output_path = args.output_annotations.resolve()

    if input_path == output_path:
        raise ValueError("The output path must differ from the source annotation path")

    payload = load_coco(input_path)
    original_categories = payload["categories"]
    annotations = payload["annotations"]

    annotation_ids = [int(annotation["id"]) for annotation in annotations]
    duplicate_annotation_ids = sorted(
        annotation_id
        for annotation_id, count in Counter(annotation_ids).items()
        if count > 1
    )
    if duplicate_annotation_ids:
        raise ValueError(f"Duplicate annotation ids: {duplicate_annotation_ids}")

    original_category_counts: Counter[int] = Counter()
    converted_annotations: list[dict[str, Any]] = []
    for annotation in annotations:
        validate_bbox(annotation)
        original_category_counts[int(annotation["category_id"])] += 1
        converted_annotation = dict(annotation)
        converted_annotation["category_id"] = TARGET_CATEGORY["id"]
        converted_annotations.append(converted_annotation)

    converted_payload = dict(payload)
    converted_payload["annotations"] = converted_annotations
    converted_payload["categories"] = [TARGET_CATEGORY]

    provenance_path = args.provenance_output
    if provenance_path is None:
        provenance_path = output_path.with_name(
            output_path.name + ".category-provenance.json"
        )
    else:
        provenance_path = provenance_path.resolve()

    provenance = {
        "source_annotations": str(input_path),
        "converted_annotations": str(output_path),
        "target_category": TARGET_CATEGORY,
        "original_categories": original_categories,
        "original_annotation_counts_by_category_id": {
            str(category_id): count
            for category_id, count in sorted(original_category_counts.items())
        },
        "annotation_count": len(annotations),
        "image_count": len(payload["images"]),
    }

    atomic_write_json(output_path, converted_payload)
    atomic_write_json(provenance_path, provenance)

    print(
        json.dumps(
            {
                "converted_annotations": str(output_path),
                "provenance": str(provenance_path),
                "image_count": len(payload["images"]),
                "annotation_count": len(annotations),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
