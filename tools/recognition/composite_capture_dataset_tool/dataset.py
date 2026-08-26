from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .composer import (
    DEFAULT_MIN_RETAINED_AREA_RATIO,
    CompositeResult,
    transform_capture_annotations,
)
from .layout import COMPOSITE_SIZE, LAYOUT_ID, REGION_SPECS, TARGET_CATEGORY
from .models import Rect, RegionSelection, SavedComposite


class OutputDatasetManager:
    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory.resolve()
        self.images_directory = self.output_directory / "images"
        self.annotations_directory = self.output_directory / "annotations"
        self.annotations_path = self.annotations_directory / "instances.json"
        self.images_directory.mkdir(parents=True, exist_ok=True)
        self.annotations_directory.mkdir(parents=True, exist_ok=True)
        self.payload = self._load_or_create_payload()
        self._next_image_id = _next_id(self.payload["images"])
        self._next_annotation_id = _next_id(self.payload["annotations"])

    @property
    def image_count(self) -> int:
        return len(self.payload["images"])

    @property
    def annotation_count(self) -> int:
        return len(self.payload["annotations"])

    def save_composite(
        self,
        result: CompositeResult,
        *,
        source_annotation_path: Path,
        source_image: dict[str, Any],
        selections: Mapping[str, RegionSelection],
        annotation_selection_policy: str,
        min_retained_area_ratio: float = DEFAULT_MIN_RETAINED_AREA_RATIO,
    ) -> SavedComposite:
        _validate_min_retained_area_ratio(min_retained_area_ratio)
        image_id = self._next_image_id
        image_name = f"composite_{image_id:06d}.png"
        final_image_path = self.images_directory / image_name
        while final_image_path.exists():
            image_id += 1
            image_name = f"composite_{image_id:06d}.png"
            final_image_path = self.images_directory / image_name

        image_record = {
            "id": image_id,
            "file_name": f"images/{image_name}",
            "width": COMPOSITE_SIZE[0],
            "height": COMPOSITE_SIZE[1],
            "source_annotation_json": str(source_annotation_path.resolve()),
            "source_image_id": int(source_image["id"]),
            "source_file_name": str(source_image["file_name"]),
            "annotation_selection_policy": annotation_selection_policy,
            "min_retained_area_ratio": min_retained_area_ratio,
            "capture_regions": {
                region_key: {
                    "crop": selection.crop.to_coco_bbox(),
                    "rotation_clockwise": selection.rotation_clockwise,
                    "destination": REGION_SPECS[
                        region_key
                    ].destination.to_coco_bbox(),
                }
                for region_key, selection in selections.items()
            },
        }

        annotation_records: list[dict[str, Any]] = []
        next_annotation_id = self._next_annotation_id
        for transformed in result.annotations:
            bbox = transformed.bbox.to_coco_bbox()
            annotation_records.append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": TARGET_CATEGORY["id"],
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 6),
                    "iscrowd": transformed.iscrowd,
                    "source_annotation_id": transformed.source_annotation_id,
                    "capture_region": transformed.region_key,
                }
            )
            next_annotation_id += 1

        temporary_image_path = _temporary_path(
            self.images_directory,
            image_name,
            suffix=".tmp.png",
        )
        images_length = len(self.payload["images"])
        annotations_length = len(self.payload["annotations"])
        try:
            result.image.save(temporary_image_path, format="PNG")
            os.replace(temporary_image_path, final_image_path)
            self.payload["images"].append(image_record)
            self.payload["annotations"].extend(annotation_records)
            _atomic_write_json(self.annotations_path, self.payload)
        except Exception:
            del self.payload["images"][images_length:]
            del self.payload["annotations"][annotations_length:]
            temporary_image_path.unlink(missing_ok=True)
            final_image_path.unlink(missing_ok=True)
            raise

        self._next_image_id = image_id + 1
        self._next_annotation_id = next_annotation_id
        return SavedComposite(
            image_id=image_id,
            annotation_count=len(annotation_records),
            image_path=str(final_image_path),
            annotations_path=str(self.annotations_path),
        )

    def rebuild_existing_annotations(
        self,
        *,
        min_retained_area_ratio: float = DEFAULT_MIN_RETAINED_AREA_RATIO,
    ) -> dict[str, Any]:
        """Rebuild saved annotations from source COCO provenance and saved crops."""

        _validate_min_retained_area_ratio(min_retained_area_ratio)
        image_records = self.payload["images"]
        before_annotations = list(self.payload["annotations"])
        if not image_records:
            return {
                "status": "unchanged",
                "images": 0,
                "annotations_before": len(before_annotations),
                "annotations_after": len(before_annotations),
                "removed_annotations": 0,
                "added_annotations": 0,
                "min_retained_area_ratio": min_retained_area_ratio,
                "comparison": "retained ratio must be greater than the threshold",
                "backup": None,
            }

        source_cache: dict[
            Path,
            tuple[
                dict[int, dict[str, Any]],
                dict[int, list[dict[str, Any]]],
            ],
        ] = {}
        rebuilt_annotations: list[dict[str, Any]] = []
        next_annotation_id = 1
        metadata_changed = False

        for image_record in image_records:
            image_id = _required_int(image_record, "id", "output image")
            source_annotation_path = Path(
                _required_string(
                    image_record,
                    "source_annotation_json",
                    f"output image {image_id}",
                )
            ).resolve()
            if source_annotation_path not in source_cache:
                source_cache[source_annotation_path] = _load_source_coco(
                    source_annotation_path
                )
            source_images_by_id, source_annotations_by_image = source_cache[
                source_annotation_path
            ]
            source_image_id = _required_int(
                image_record,
                "source_image_id",
                f"output image {image_id}",
            )
            try:
                source_image = source_images_by_id[source_image_id]
            except KeyError as error:
                raise ValueError(
                    f"Source image id {source_image_id} is missing from "
                    f"{source_annotation_path} for output image {image_id}"
                ) from error

            selections = _selections_from_image_record(image_record)
            annotation_selection_policy = str(
                image_record.get("annotation_selection_policy", "center")
            )
            source_size = (
                _required_int(source_image, "width", f"source image {source_image_id}"),
                _required_int(source_image, "height", f"source image {source_image_id}"),
            )
            transformed_annotations, _stats = transform_capture_annotations(
                source_size,
                selections,
                source_annotations_by_image.get(source_image_id, []),
                annotation_selection_policy=annotation_selection_policy,
                min_retained_area_ratio=min_retained_area_ratio,
            )

            previous_ratio = image_record.get("min_retained_area_ratio")
            if previous_ratio != min_retained_area_ratio:
                metadata_changed = True
            image_record["min_retained_area_ratio"] = min_retained_area_ratio

            for transformed in transformed_annotations:
                bbox = transformed.bbox.to_coco_bbox()
                rebuilt_annotations.append(
                    {
                        "id": next_annotation_id,
                        "image_id": image_id,
                        "category_id": TARGET_CATEGORY["id"],
                        "bbox": bbox,
                        "area": round(bbox[2] * bbox[3], 6),
                        "iscrowd": transformed.iscrowd,
                        "source_annotation_id": transformed.source_annotation_id,
                        "capture_region": transformed.region_key,
                    }
                )
                next_annotation_id += 1

        before_signatures = Counter(
            _annotation_signature(annotation) for annotation in before_annotations
        )
        after_signatures = Counter(
            _annotation_signature(annotation) for annotation in rebuilt_annotations
        )
        removed_annotations = sum((before_signatures - after_signatures).values())
        added_annotations = sum((after_signatures - before_signatures).values())
        semantic_changed = before_signatures != after_signatures
        changed = semantic_changed or metadata_changed
        backup_path: Path | None = None

        if changed:
            backup_path = _create_backup(self.annotations_path)
            self.payload["annotations"] = rebuilt_annotations
            info = self.payload.setdefault("info", {})
            if not isinstance(info, dict):
                raise ValueError("Output COCO info field must be an object")
            info["annotation_retention"] = {
                "measure": "crop intersection area / original source bbox area",
                "retain_when": "ratio > min_retained_area_ratio",
                "min_retained_area_ratio": min_retained_area_ratio,
            }
            _atomic_write_json(self.annotations_path, self.payload)
            self._next_annotation_id = next_annotation_id

        return {
            "status": "updated" if changed else "unchanged",
            "images": len(image_records),
            "annotations_before": len(before_annotations),
            "annotations_after": len(rebuilt_annotations),
            "removed_annotations": removed_annotations,
            "added_annotations": added_annotations,
            "min_retained_area_ratio": min_retained_area_ratio,
            "comparison": "retained ratio must be greater than the threshold",
            "backup": str(backup_path) if backup_path is not None else None,
            "annotations": str(self.annotations_path),
        }

    def _load_or_create_payload(self) -> dict[str, Any]:
        if not self.annotations_path.exists():
            return {
                "info": {
                    "description": (
                        "Manual fixed-region composite capture test dataset"
                    ),
                    "layout": LAYOUT_ID,
                    "padding_rgb": [0, 0, 0],
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [TARGET_CATEGORY],
            }

        with self.annotations_path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Output COCO root must be an object: {self.annotations_path}"
            )
        for field in ("images", "annotations", "categories"):
            if not isinstance(payload.get(field), list):
                raise ValueError(
                    f"Output COCO field must be a list: "
                    f"{self.annotations_path}: {field}"
                )
        categories = payload["categories"]
        if len(categories) != 1:
            raise ValueError(
                f"Output dataset must have exactly one category: {categories!r}"
            )
        category = categories[0]
        if (
            not isinstance(category, dict)
            or int(category.get("id", -1)) != TARGET_CATEGORY["id"]
            or category.get("name") != TARGET_CATEGORY["name"]
        ):
            raise ValueError(
                f"Output category is incompatible with {TARGET_CATEGORY}: {category!r}"
            )
        return payload


def _load_source_coco(
    path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Source COCO annotations do not exist: {path}")
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Source COCO root must be an object: {path}")
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"Source COCO must contain image and annotation lists: {path}")

    images_by_id: dict[int, dict[str, Any]] = {}
    for image in images:
        if not isinstance(image, dict):
            raise ValueError(f"Invalid source image record in {path}: {image!r}")
        image_id = _required_int(image, "id", f"source COCO {path}")
        images_by_id[image_id] = image

    annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"Invalid source annotation in {path}: {annotation!r}")
        image_id = _required_int(annotation, "image_id", f"source COCO {path}")
        annotations_by_image[image_id].append(annotation)
    return images_by_id, dict(annotations_by_image)


def _selections_from_image_record(
    image_record: dict[str, Any],
) -> dict[str, RegionSelection]:
    image_id = _required_int(image_record, "id", "output image")
    capture_regions = image_record.get("capture_regions")
    if not isinstance(capture_regions, dict) or not capture_regions:
        raise ValueError(
            f"Output image {image_id} has no usable capture_regions provenance"
        )

    selections: dict[str, RegionSelection] = {}
    for region_key, region_record in capture_regions.items():
        if not isinstance(region_key, str) or not isinstance(region_record, dict):
            raise ValueError(
                f"Invalid capture region for output image {image_id}: "
                f"{region_key!r}: {region_record!r}"
            )
        crop = Rect.from_coco_bbox(
            region_record.get("crop"),
            context=f"output image {image_id} capture region {region_key}",
        )
        rotation = _required_int(
            region_record,
            "rotation_clockwise",
            f"output image {image_id} capture region {region_key}",
        )
        selections[region_key] = RegionSelection(
            region_key=region_key,
            crop=crop,
            rotation_clockwise=rotation,
        )
    return selections


def _annotation_signature(annotation: dict[str, Any]) -> tuple[Any, ...]:
    bbox = Rect.from_coco_bbox(
        annotation.get("bbox"),
        context=f"output annotation {annotation.get('id')}",
    )
    return (
        _required_int(annotation, "image_id", "output annotation"),
        _required_int(annotation, "source_annotation_id", "output annotation"),
        _required_string(annotation, "capture_region", "output annotation"),
        tuple(bbox.to_coco_bbox()),
        int(annotation.get("iscrowd", 0)),
    )


def _required_int(record: Mapping[str, Any], field: str, context: str) -> int:
    try:
        return int(record[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} in {context}: {record.get(field)!r}") from error


def _required_string(record: Mapping[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {field} in {context}: {value!r}")
    return value


def _validate_min_retained_area_ratio(value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(
            "min_retained_area_ratio must be at least zero and less than one: "
            f"{value}"
        )


def _create_backup(path: Path) -> Path:
    backup_directory = path.parent / "backups"
    backup_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S.%f")
    backup_path = backup_directory / f"{path.stem}.{timestamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    return backup_path


def _next_id(records: list[dict[str, Any]]) -> int:
    if not records:
        return 1
    return max(int(record["id"]) for record in records) + 1


def _temporary_path(directory: Path, name: str, *, suffix: str) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{name}.",
        suffix=suffix,
        dir=directory,
    )
    os.close(file_descriptor)
    return Path(temporary_name)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
