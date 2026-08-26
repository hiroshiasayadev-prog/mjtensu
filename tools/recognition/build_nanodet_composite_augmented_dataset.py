from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


TARGET_CATEGORY = {
    "id": 1,
    "name": "mahjong_tile",
    "supercategory": "mahjong_tile",
}


@dataclass(frozen=True)
class CompositeSplit:
    train_image_ids: frozenset[int]
    val_image_ids: frozenset[int]
    train_group_keys: frozenset[tuple[str, int]]
    val_group_keys: frozenset[tuple[str, int]]


class CocoAccumulator:
    def __init__(
        self,
        *,
        repository_root: Path,
        description: str,
        check_images: bool,
    ) -> None:
        self.repository_root = repository_root
        self.check_images = check_images
        self.payload: dict[str, Any] = {
            "info": {
                "description": description,
                "image_root": str(repository_root),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [TARGET_CATEGORY],
        }
        self._next_image_id = 1
        self._next_annotation_id = 1

    def append(
        self,
        source_payload: dict[str, Any],
        *,
        source_image_root: Path,
        included_image_ids: set[int] | frozenset[int] | None,
        dataset_origin: str,
    ) -> dict[str, int]:
        source_images = source_payload["images"]
        source_annotations = source_payload["annotations"]
        source_images_by_id = _unique_records_by_id(
            source_images,
            record_kind="image",
            context=dataset_origin,
        )
        _unique_records_by_id(
            source_annotations,
            record_kind="annotation",
            context=dataset_origin,
        )

        image_id_map: dict[int, int] = {}
        appended_images = 0
        appended_annotations = 0

        for source_image in source_images:
            source_image_id = int(source_image["id"])
            if (
                included_image_ids is not None
                and source_image_id not in included_image_ids
            ):
                continue

            generated_file_name = _repository_relative_image_name(
                self.repository_root,
                source_image_root,
                source_image.get("file_name"),
            )
            if self.check_images:
                image_path = self.repository_root / Path(generated_file_name)
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image does not exist for {dataset_origin}: {image_path}"
                    )

            generated_image_id = self._next_image_id
            self._next_image_id += 1
            image_id_map[source_image_id] = generated_image_id

            generated_image = {
                key: value
                for key, value in source_image.items()
                if key not in {"id", "file_name"}
            }
            generated_image["id"] = generated_image_id
            generated_image["file_name"] = generated_file_name
            generated_image["dataset_origin"] = dataset_origin
            self.payload["images"].append(generated_image)
            appended_images += 1

        for source_annotation in source_annotations:
            source_image_id = int(source_annotation["image_id"])
            generated_image_id = image_id_map.get(source_image_id)
            if generated_image_id is None:
                if source_image_id not in source_images_by_id:
                    raise ValueError(
                        f"Annotation {source_annotation.get('id')} in {dataset_origin} "
                        f"references undefined image id {source_image_id}"
                    )
                continue

            _validate_bbox(source_annotation, dataset_origin)
            generated_annotation = {
                key: value
                for key, value in source_annotation.items()
                if key not in {"id", "image_id", "category_id"}
            }
            generated_annotation["id"] = self._next_annotation_id
            self._next_annotation_id += 1
            generated_annotation["image_id"] = generated_image_id
            generated_annotation["category_id"] = TARGET_CATEGORY["id"]
            self.payload["annotations"].append(generated_annotation)
            appended_annotations += 1

        return {
            "images": appended_images,
            "annotations": appended_annotations,
        }


def parse_args() -> argparse.Namespace:
    repository_root_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Build NanoDet train/val COCO annotations by adding a deterministic "
            "source-image-grouped 80/20 split of the manual composite dataset to "
            "the existing single-class train/val datasets. Images are referenced "
            "in place and are not copied."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=repository_root_default,
    )
    parser.add_argument("--base-train-annotations", type=Path)
    parser.add_argument("--base-val-annotations", type=Path)
    parser.add_argument("--composite-annotations", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of composite images assigned to train by source group.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic composite source-group split seed.",
    )
    parser.add_argument(
        "--skip-image-existence-check",
        action="store_true",
        help="Do not verify that every referenced image exists beneath the repository root.",
    )
    return parser.parse_args()


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"COCO root must be an object: {path}")
    for field in ("images", "annotations", "categories"):
        if not isinstance(payload.get(field), list):
            raise ValueError(f"COCO field must be a list: {path}: {field}")
    _validate_target_category(payload["categories"], path)
    return payload


def split_composite_images(
    composite_payload: dict[str, Any],
    *,
    train_fraction: float,
    seed: int,
) -> CompositeSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be strictly between zero and one: {train_fraction}"
        )

    grouped_image_ids: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    for image in composite_payload["images"]:
        image_id = int(image["id"])
        source_annotation_json = image.get("source_annotation_json")
        source_image_id = image.get("source_image_id")
        if not isinstance(source_annotation_json, str) or not source_annotation_json:
            raise ValueError(
                f"Composite image {image_id} has no source_annotation_json provenance"
            )
        try:
            normalized_source_image_id = int(source_image_id)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Composite image {image_id} has invalid source_image_id: "
                f"{source_image_id!r}"
            ) from error
        group_key = (
            source_annotation_json.replace("\\", "/").casefold(),
            normalized_source_image_id,
        )
        grouped_image_ids[group_key].append(image_id)

    if len(grouped_image_ids) < 2:
        raise ValueError(
            "At least two composite source-image groups are required for an 80/20 split"
        )

    grouped_items = sorted(grouped_image_ids.items(), key=lambda item: item[0])
    random.Random(seed).shuffle(grouped_items)
    target_train_images = round(len(composite_payload["images"]) * train_fraction)
    selected_group_indices = _closest_group_subset(
        [len(image_ids) for _key, image_ids in grouped_items],
        target=target_train_images,
    )

    train_group_keys = frozenset(
        grouped_items[index][0] for index in selected_group_indices
    )
    val_group_keys = frozenset(grouped_image_ids) - train_group_keys
    train_image_ids = frozenset(
        image_id
        for group_key in train_group_keys
        for image_id in grouped_image_ids[group_key]
    )
    val_image_ids = frozenset(
        image_id
        for group_key in val_group_keys
        for image_id in grouped_image_ids[group_key]
    )
    if not train_image_ids or not val_image_ids:
        raise ValueError("Composite split produced an empty train or validation partition")
    if train_image_ids & val_image_ids:
        raise AssertionError("Composite train and validation image IDs overlap")

    return CompositeSplit(
        train_image_ids=train_image_ids,
        val_image_ids=val_image_ids,
        train_group_keys=train_group_keys,
        val_group_keys=val_group_keys,
    )


def _closest_group_subset(group_sizes: list[int], *, target: int) -> frozenset[int]:
    total = sum(group_sizes)
    if total <= 1:
        raise ValueError("At least two composite images are required")

    parents: dict[int, tuple[int, int] | None] = {0: None}
    for group_index, group_size in enumerate(group_sizes):
        if group_size <= 0:
            raise ValueError(f"Composite group has invalid size: {group_size}")
        for previous_sum in sorted(tuple(parents), reverse=True):
            new_sum = previous_sum + group_size
            if new_sum not in parents:
                parents[new_sum] = (previous_sum, group_index)

    valid_sums = [value for value in parents if 0 < value < total]
    if not valid_sums:
        raise ValueError("Unable to form non-empty composite train and validation splits")
    selected_sum = min(
        valid_sums,
        key=lambda value: (
            abs(value - target),
            value > target,
            -value,
        ),
    )

    selected_indices: set[int] = set()
    current_sum = selected_sum
    while current_sum != 0:
        parent = parents[current_sum]
        if parent is None:
            raise AssertionError("Invalid subset reconstruction state")
        previous_sum, group_index = parent
        selected_indices.add(group_index)
        current_sum = previous_sum
    return frozenset(selected_indices)


def build_partition_payload(
    *,
    repository_root: Path,
    base_payload: dict[str, Any] | None,
    base_image_root: Path | None,
    composite_payload: dict[str, Any],
    composite_image_root: Path,
    composite_image_ids: frozenset[int],
    partition_name: str,
    check_images: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    accumulator = CocoAccumulator(
        repository_root=repository_root,
        description=f"NanoDet mahjong tile dataset: {partition_name}",
        check_images=check_images,
    )
    counts: dict[str, dict[str, int]] = {}
    if base_payload is not None:
        if base_image_root is None:
            raise ValueError("base_image_root is required when base_payload is provided")
        counts["base"] = accumulator.append(
            base_payload,
            source_image_root=base_image_root,
            included_image_ids=None,
            dataset_origin=f"base_{partition_name}",
        )
    counts["composite"] = accumulator.append(
        composite_payload,
        source_image_root=composite_image_root,
        included_image_ids=composite_image_ids,
        dataset_origin=f"composite_{partition_name}",
    )
    return accumulator.payload, counts


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    base_dataset_root = (
        repository_root / ".local" / "recognition" / "nanodet_single_class_dataset"
    )
    composite_dataset_root = (
        repository_root / ".local" / "recognition" / "composite_capture_test_dataset"
    )
    output_directory = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else repository_root
        / ".local"
        / "recognition"
        / "nanodet_composite_augmented_dataset"
    )
    base_train_annotations = (
        args.base_train_annotations.resolve()
        if args.base_train_annotations is not None
        else base_dataset_root / "annotations" / "instances_train.json"
    )
    base_val_annotations = (
        args.base_val_annotations.resolve()
        if args.base_val_annotations is not None
        else base_dataset_root / "annotations" / "instances_val.json"
    )
    composite_annotations = (
        args.composite_annotations.resolve()
        if args.composite_annotations is not None
        else composite_dataset_root / "annotations" / "instances.json"
    )

    for path in (
        base_train_annotations,
        base_val_annotations,
        composite_annotations,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    base_train_payload = load_coco(base_train_annotations)
    base_val_payload = load_coco(base_val_annotations)
    composite_payload = load_coco(composite_annotations)
    composite_split = split_composite_images(
        composite_payload,
        train_fraction=float(args.train_fraction),
        seed=int(args.seed),
    )
    check_images = not args.skip_image_existence_check

    merged_train, merged_train_counts = build_partition_payload(
        repository_root=repository_root,
        base_payload=base_train_payload,
        base_image_root=repository_root / "data",
        composite_payload=composite_payload,
        composite_image_root=composite_dataset_root,
        composite_image_ids=composite_split.train_image_ids,
        partition_name="train",
        check_images=check_images,
    )
    merged_val, merged_val_counts = build_partition_payload(
        repository_root=repository_root,
        base_payload=base_val_payload,
        base_image_root=repository_root / "data",
        composite_payload=composite_payload,
        composite_image_root=composite_dataset_root,
        composite_image_ids=composite_split.val_image_ids,
        partition_name="val",
        check_images=check_images,
    )
    composite_train, composite_train_counts = build_partition_payload(
        repository_root=repository_root,
        base_payload=None,
        base_image_root=None,
        composite_payload=composite_payload,
        composite_image_root=composite_dataset_root,
        composite_image_ids=composite_split.train_image_ids,
        partition_name="composite_train",
        check_images=check_images,
    )
    composite_val, composite_val_counts = build_partition_payload(
        repository_root=repository_root,
        base_payload=None,
        base_image_root=None,
        composite_payload=composite_payload,
        composite_image_root=composite_dataset_root,
        composite_image_ids=composite_split.val_image_ids,
        partition_name="composite_val",
        check_images=check_images,
    )

    annotations_directory = output_directory / "annotations"
    output_paths = {
        "train": annotations_directory / "instances_train.json",
        "val": annotations_directory / "instances_val.json",
        "composite_train": annotations_directory / "instances_composite_train.json",
        "composite_val": annotations_directory / "instances_composite_val.json",
    }
    payloads = {
        "train": merged_train,
        "val": merged_val,
        "composite_train": composite_train,
        "composite_val": composite_val,
    }
    for name, path in output_paths.items():
        _atomic_write_json(path, payloads[name], compact=True)

    provenance = {
        "artifact": "nanodet_composite_augmented_dataset",
        "repository_root": str(repository_root),
        "image_root_for_nanodet": str(repository_root),
        "seed": int(args.seed),
        "requested_train_fraction": float(args.train_fraction),
        "composite_split": {
            "total_images": len(composite_payload["images"]),
            "total_annotations": len(composite_payload["annotations"]),
            "total_source_groups": len(
                composite_split.train_group_keys | composite_split.val_group_keys
            ),
            "train_images": len(composite_split.train_image_ids),
            "val_images": len(composite_split.val_image_ids),
            "actual_train_fraction": (
                len(composite_split.train_image_ids)
                / len(composite_payload["images"])
            ),
            "train_source_groups": len(composite_split.train_group_keys),
            "val_source_groups": len(composite_split.val_group_keys),
            "group_identity": ["source_annotation_json", "source_image_id"],
        },
        "source_annotations": {
            "base_train": str(base_train_annotations),
            "base_val": str(base_val_annotations),
            "composite": str(composite_annotations),
        },
        "counts": {
            "train": merged_train_counts,
            "val": merged_val_counts,
            "composite_train": composite_train_counts,
            "composite_val": composite_val_counts,
        },
        "outputs": {
            name: {
                "annotations": str(path),
                "images": len(payloads[name]["images"]),
                "annotations_count": len(payloads[name]["annotations"]),
            }
            for name, path in output_paths.items()
        },
        "images_copied": False,
    }
    provenance_path = output_directory / "provenance.json"
    _atomic_write_json(provenance_path, provenance, compact=False)

    print(
        json.dumps(
            {
                "status": "completed",
                "output_directory": str(output_directory),
                "composite_split": provenance["composite_split"],
                "outputs": provenance["outputs"],
                "provenance": str(provenance_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _validate_target_category(categories: list[Any], path: Path) -> None:
    if len(categories) != 1 or not isinstance(categories[0], dict):
        raise ValueError(f"Expected exactly one COCO category in {path}")
    category = categories[0]
    if int(category.get("id", -1)) != TARGET_CATEGORY["id"]:
        raise ValueError(f"Unexpected category id in {path}: {category!r}")
    if category.get("name") != TARGET_CATEGORY["name"]:
        raise ValueError(f"Unexpected category name in {path}: {category!r}")


def _unique_records_by_id(
    records: Iterable[dict[str, Any]],
    *,
    record_kind: str,
    context: str,
) -> dict[int, dict[str, Any]]:
    records_by_id: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid {record_kind} in {context}: {record!r}")
        try:
            record_id = int(record["id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid {record_kind} id in {context}: {record.get('id')!r}"
            ) from error
        if record_id in records_by_id:
            raise ValueError(f"Duplicate {record_kind} id {record_id} in {context}")
        records_by_id[record_id] = record
    return records_by_id


def _validate_bbox(annotation: dict[str, Any], context: str) -> None:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(
            f"Annotation {annotation.get('id')} has invalid bbox in {context}: {bbox!r}"
        )
    try:
        _x, _y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Annotation {annotation.get('id')} has non-numeric bbox in {context}: "
            f"{bbox!r}"
        ) from error
    if width <= 0.0 or height <= 0.0:
        raise ValueError(
            f"Annotation {annotation.get('id')} has non-positive bbox in {context}: "
            f"{bbox!r}"
        )


def _repository_relative_image_name(
    repository_root: Path,
    source_image_root: Path,
    source_file_name: Any,
) -> str:
    if not isinstance(source_file_name, str) or not source_file_name:
        raise ValueError(f"Invalid source image file_name: {source_file_name!r}")
    normalized_file_name = PurePosixPath(source_file_name.replace("\\", "/"))
    if normalized_file_name.is_absolute() or ".." in normalized_file_name.parts:
        raise ValueError(f"Unsafe source image file_name: {source_file_name!r}")
    try:
        root_relative = source_image_root.resolve().relative_to(repository_root)
    except ValueError as error:
        raise ValueError(
            f"Source image root must be inside repository root: {source_image_root}"
        ) from error
    return str(PurePosixPath(root_relative.as_posix()) / normalized_file_name)


def _atomic_write_json(path: Path, payload: Any, *, compact: bool) -> None:
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
                json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
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


if __name__ == "__main__":
    main()
