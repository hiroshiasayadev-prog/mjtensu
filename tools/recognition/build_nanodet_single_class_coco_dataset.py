from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


TARGET_CATEGORY = {
    "id": 1,
    "name": "mahjong_tile",
    "supercategory": "mahjong_tile",
}


@dataclass(frozen=True)
class SourceSplit:
    dataset_id: str
    source_split: str
    generated_split: str
    annotations_path: Path
    image_root_relative_to_data: PurePosixPath


def parse_args() -> argparse.Namespace:
    repository_root_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Build source-preserving single-class COCO annotations for NanoDet from "
            "data/coco_mahjong and data/coco_mahjong_jp_v2. Source images and source "
            "annotations are not modified or copied."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=repository_root_default,
        help=f"Repository root. Defaults to {repository_root_default}",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Generated dataset directory. Defaults to "
            ".local/recognition/nanodet_single_class_dataset under the repository root."
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
            raise ValueError(f"COCO field must be a list: {path}: {required_key}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_for_report(path: Path, repository_root: Path) -> str:
    try:
        report_path = path.relative_to(repository_root)
    except ValueError:
        report_path = path
    return str(report_path).replace("\\", "/")


def atomic_write_json(path: Path, payload: Any, *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as output:
            if compact:
                json.dump(
                    payload,
                    output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            else:
                json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def validate_bbox(annotation: dict[str, Any], source_path: Path) -> None:
    annotation_id = annotation.get("id")
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(
            f"Annotation {annotation_id} has an invalid bbox shape in {source_path}"
        )
    _, _, width, height = bbox
    if float(width) <= 0.0 or float(height) <= 0.0:
        raise ValueError(
            f"Annotation {annotation_id} has a non-positive bbox in {source_path}"
        )


def require_unique_integer_ids(
    records: list[dict[str, Any]],
    *,
    record_kind: str,
    source_path: Path,
) -> dict[int, dict[str, Any]]:
    records_by_id: dict[int, dict[str, Any]] = {}
    for record in records:
        if "id" not in record:
            raise ValueError(f"{record_kind} without id in {source_path}")
        record_id = int(record["id"])
        if record_id in records_by_id:
            raise ValueError(
                f"Duplicate {record_kind} id {record_id} in {source_path}"
            )
        records_by_id[record_id] = record
    return records_by_id


def generated_file_name(
    image_root_relative_to_data: PurePosixPath,
    source_file_name: Any,
) -> str:
    if not isinstance(source_file_name, str) or not source_file_name:
        raise ValueError(f"Invalid source image file_name: {source_file_name!r}")

    normalized = PurePosixPath(source_file_name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Unsafe source image file_name: {source_file_name}")

    return str(image_root_relative_to_data / normalized)


def build_sources(repository_root: Path) -> list[SourceSplit]:
    data_root = repository_root / "data"
    return [
        SourceSplit(
            dataset_id="coco_mahjong",
            source_split="train2017",
            generated_split="train",
            annotations_path=(
                data_root
                / "coco_mahjong"
                / "annotations"
                / "instances_train2017.json"
            ),
            image_root_relative_to_data=PurePosixPath(
                "coco_mahjong/train2017"
            ),
        ),
        SourceSplit(
            dataset_id="coco_mahjong_jp_v2",
            source_split="train",
            generated_split="train",
            annotations_path=(
                data_root
                / "coco_mahjong_jp_v2"
                / "train"
                / "_annotations.coco.json"
            ),
            image_root_relative_to_data=PurePosixPath(
                "coco_mahjong_jp_v2/train"
            ),
        ),
        SourceSplit(
            dataset_id="coco_mahjong",
            source_split="val2017",
            generated_split="val",
            annotations_path=(
                data_root
                / "coco_mahjong"
                / "annotations"
                / "instances_val2017.json"
            ),
            image_root_relative_to_data=PurePosixPath(
                "coco_mahjong/val2017"
            ),
        ),
        SourceSplit(
            dataset_id="coco_mahjong_jp_v2",
            source_split="valid",
            generated_split="val",
            annotations_path=(
                data_root
                / "coco_mahjong_jp_v2"
                / "valid"
                / "_annotations.coco.json"
            ),
            image_root_relative_to_data=PurePosixPath(
                "coco_mahjong_jp_v2/valid"
            ),
        ),
        SourceSplit(
            dataset_id="coco_mahjong_jp_v2",
            source_split="test",
            generated_split="test",
            annotations_path=(
                data_root
                / "coco_mahjong_jp_v2"
                / "test"
                / "_annotations.coco.json"
            ),
            image_root_relative_to_data=PurePosixPath(
                "coco_mahjong_jp_v2/test"
            ),
        ),
    ]


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    output_directory = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else repository_root
        / ".local"
        / "recognition"
        / "nanodet_single_class_dataset"
    )
    annotations_directory = output_directory / "annotations"
    data_root = repository_root / "data"

    sources = build_sources(repository_root)
    for source in sources:
        if not source.annotations_path.is_file():
            raise FileNotFoundError(source.annotations_path)

    generated_by_split: dict[str, dict[str, Any]] = {
        split: {
            "info": {
                "description": "Single-class mahjong tile detector dataset for NanoDet",
                "source_repository": str(repository_root),
                "image_root": "data",
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [TARGET_CATEGORY],
        }
        for split in ("train", "val", "test")
    }
    next_image_id_by_split = {split: 1 for split in generated_by_split}
    next_annotation_id_by_split = {split: 1 for split in generated_by_split}
    provenance_sources: list[dict[str, Any]] = []

    for source in sources:
        payload = load_coco(source.annotations_path)
        source_images = payload["images"]
        source_annotations = payload["annotations"]
        source_categories = payload["categories"]

        source_images_by_id = require_unique_integer_ids(
            source_images,
            record_kind="image",
            source_path=source.annotations_path,
        )
        require_unique_integer_ids(
            source_annotations,
            record_kind="annotation",
            source_path=source.annotations_path,
        )

        generated_payload = generated_by_split[source.generated_split]
        generated_image_start = next_image_id_by_split[source.generated_split]
        generated_annotation_start = next_annotation_id_by_split[
            source.generated_split
        ]
        generated_image_id_by_source_id: dict[int, int] = {}

        for source_image in source_images:
            source_image_id = int(source_image["id"])
            generated_image_id = next_image_id_by_split[source.generated_split]
            next_image_id_by_split[source.generated_split] += 1
            generated_image_id_by_source_id[source_image_id] = generated_image_id

            generated_image = {
                key: value
                for key, value in source_image.items()
                if key not in {"id", "file_name", "license"}
            }
            generated_image["id"] = generated_image_id
            generated_image["file_name"] = generated_file_name(
                source.image_root_relative_to_data,
                source_image.get("file_name"),
            )
            generated_image_path = data_root / Path(generated_image["file_name"])
            if not generated_image_path.is_file():
                raise FileNotFoundError(
                    "COCO image record does not resolve beneath the data root: "
                    f"{generated_image_path}"
                )
            generated_payload["images"].append(generated_image)

        original_annotation_counts_by_category_id: dict[int, int] = {}
        for source_annotation in source_annotations:
            validate_bbox(source_annotation, source.annotations_path)
            source_image_id = int(source_annotation["image_id"])
            if source_image_id not in source_images_by_id:
                raise ValueError(
                    f"Annotation {source_annotation.get('id')} references undefined "
                    f"image id {source_image_id} in {source.annotations_path}"
                )

            original_category_id = int(source_annotation["category_id"])
            original_annotation_counts_by_category_id[original_category_id] = (
                original_annotation_counts_by_category_id.get(
                    original_category_id, 0
                )
                + 1
            )

            generated_annotation = {
                key: value
                for key, value in source_annotation.items()
                if key not in {"id", "image_id", "category_id"}
            }
            generated_annotation["id"] = next_annotation_id_by_split[
                source.generated_split
            ]
            next_annotation_id_by_split[source.generated_split] += 1
            generated_annotation["image_id"] = generated_image_id_by_source_id[
                source_image_id
            ]
            generated_annotation["category_id"] = TARGET_CATEGORY["id"]
            generated_payload["annotations"].append(generated_annotation)

        generated_image_end = next_image_id_by_split[source.generated_split] - 1
        generated_annotation_end = (
            next_annotation_id_by_split[source.generated_split] - 1
        )
        provenance_sources.append(
            {
                "dataset_id": source.dataset_id,
                "source_split": source.source_split,
                "generated_split": source.generated_split,
                "source_annotations": path_for_report(
                    source.annotations_path,
                    repository_root,
                ),
                "source_annotations_sha256": sha256_file(
                    source.annotations_path
                ),
                "source_image_root_relative_to_data": str(
                    source.image_root_relative_to_data
                ),
                "generated_image_id_range": {
                    "start": generated_image_start,
                    "end": generated_image_end,
                },
                "generated_annotation_id_range": {
                    "start": generated_annotation_start,
                    "end": generated_annotation_end,
                },
                "id_mapping": (
                    "Generated IDs are sequential in source JSON list order. "
                    "Reverse lookup uses the generated range offset and the "
                    "corresponding source images or annotations list index."
                ),
                "image_count": len(source_images),
                "annotation_count": len(source_annotations),
                "original_categories": source_categories,
                "original_annotation_counts_by_category_id": {
                    str(category_id): count
                    for category_id, count in sorted(
                        original_annotation_counts_by_category_id.items()
                    )
                },
                "tile_type_classifier_eligible": (
                    source.dataset_id == "coco_mahjong_jp_v2"
                ),
            }
        )

    output_paths = {
        "train": annotations_directory / "instances_train.json",
        "val": annotations_directory / "instances_val.json",
        "test": annotations_directory / "instances_test.json",
    }
    for split, output_path in output_paths.items():
        atomic_write_json(
            output_path,
            generated_by_split[split],
            compact=True,
        )

    provenance = {
        "artifact": "nanodet_single_class_coco_dataset",
        "image_root": "data",
        "target_category": TARGET_CATEGORY,
        "split_policy": {
            "train": [
                "coco_mahjong/train2017",
                "coco_mahjong_jp_v2/train",
            ],
            "val": [
                "coco_mahjong/val2017",
                "coco_mahjong_jp_v2/valid",
            ],
            "test": ["coco_mahjong_jp_v2/test"],
        },
        "source_images_copied": False,
        "source_annotations_modified": False,
        "sources": provenance_sources,
        "generated_outputs": {
            split: {
                "annotations": path_for_report(path, repository_root),
                "image_count": len(generated_by_split[split]["images"]),
                "annotation_count": len(
                    generated_by_split[split]["annotations"]
                ),
            }
            for split, path in output_paths.items()
        },
    }
    provenance_path = output_directory / "provenance.json"
    atomic_write_json(provenance_path, provenance, compact=False)

    print(
        json.dumps(
            {
                "output_directory": str(output_directory),
                "annotations": {
                    split: str(path) for split, path in output_paths.items()
                },
                "provenance": str(provenance_path),
                "counts": {
                    split: {
                        "images": len(generated_by_split[split]["images"]),
                        "annotations": len(
                            generated_by_split[split]["annotations"]
                        ),
                    }
                    for split in ("train", "val", "test")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
