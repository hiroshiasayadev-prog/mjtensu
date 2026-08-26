from __future__ import annotations

import json
from collections import defaultdict
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

from .models import Rect


class CocoDataset:
    def __init__(
        self,
        *,
        annotation_path: Path,
        image_root: Path,
        images: list[dict[str, Any]],
        annotations_by_image_id: dict[int, list[dict[str, Any]]],
        categories: list[dict[str, Any]],
        annotation_count: int,
        source_image_count: int | None = None,
        source_annotation_count: int | None = None,
        image_path_prefix: str | None = None,
        image_name_pattern: str | None = None,
    ) -> None:
        self.annotation_path = annotation_path
        self.image_root = image_root
        self.images = images
        self.annotations_by_image_id = annotations_by_image_id
        self.categories = categories
        self.annotation_count = annotation_count
        self.source_image_count = (
            len(images) if source_image_count is None else source_image_count
        )
        self.source_annotation_count = (
            annotation_count
            if source_annotation_count is None
            else source_annotation_count
        )
        self.image_path_prefix = image_path_prefix
        self.image_name_pattern = image_name_pattern
        self._images_by_id = {int(image["id"]): image for image in images}

    @classmethod
    def load(
        cls,
        annotation_path: Path,
        image_root: Path,
        *,
        image_path_prefix: str | None = None,
        image_name_pattern: str | None = None,
    ) -> "CocoDataset":
        annotation_path = annotation_path.resolve()
        image_root = image_root.resolve()
        with annotation_path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        if not isinstance(payload, dict):
            raise ValueError(f"COCO root must be an object: {annotation_path}")
        for field in ("images", "annotations", "categories"):
            if not isinstance(payload.get(field), list):
                raise ValueError(
                    f"COCO field must be a list: {annotation_path}: {field}"
                )

        source_images = payload["images"]
        source_annotations = payload["annotations"]
        categories = payload["categories"]
        image_ids: set[int] = set()
        retained_images: list[dict[str, Any]] = []
        retained_image_ids: set[int] = set()
        normalized_prefix = _normalize_filter_prefix(image_path_prefix)
        for image in source_images:
            if not isinstance(image, dict):
                raise ValueError(f"COCO image record must be an object: {image!r}")
            image_id = _required_int(image, "id", "image")
            if image_id in image_ids:
                raise ValueError(f"Duplicate COCO image id: {image_id}")
            image_ids.add(image_id)
            file_name = image.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                raise ValueError(f"Invalid file_name for image {image_id}: {file_name!r}")
            _required_positive_int(image, "width", f"image {image_id}")
            _required_positive_int(image, "height", f"image {image_id}")
            if _image_matches_filter(
                file_name,
                normalized_prefix,
                image_name_pattern,
            ):
                retained_images.append(image)
                retained_image_ids.add(image_id)

        annotations_by_image_id: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        annotation_ids: set[int] = set()
        retained_annotation_count = 0
        for annotation in source_annotations:
            if not isinstance(annotation, dict):
                raise ValueError(
                    f"COCO annotation record must be an object: {annotation!r}"
                )
            annotation_id = _required_int(annotation, "id", "annotation")
            if annotation_id in annotation_ids:
                raise ValueError(f"Duplicate COCO annotation id: {annotation_id}")
            annotation_ids.add(annotation_id)
            image_id = _required_int(
                annotation,
                "image_id",
                f"annotation {annotation_id}",
            )
            if image_id not in image_ids:
                raise ValueError(
                    f"Annotation {annotation_id} references undefined image {image_id}"
                )
            Rect.from_coco_bbox(
                annotation.get("bbox"),
                context=f"annotation {annotation_id}",
            )
            if image_id in retained_image_ids:
                annotations_by_image_id[image_id].append(annotation)
                retained_annotation_count += 1

        return cls(
            annotation_path=annotation_path,
            image_root=image_root,
            images=retained_images,
            annotations_by_image_id=dict(annotations_by_image_id),
            categories=categories,
            annotation_count=retained_annotation_count,
            source_image_count=len(source_images),
            source_annotation_count=len(source_annotations),
            image_path_prefix=normalized_prefix,
            image_name_pattern=image_name_pattern,
        )

    def image_at(self, index: int) -> dict[str, Any]:
        return self.images[index]

    def image_by_id(self, image_id: int) -> dict[str, Any]:
        return self._images_by_id[image_id]

    def annotations_for_image(self, image_id: int) -> list[dict[str, Any]]:
        return self.annotations_by_image_id.get(image_id, [])

    def resolve_image_path(self, image: dict[str, Any]) -> Path:
        image_id = int(image["id"])
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"Invalid file_name for image {image_id}: {file_name!r}")
        normalized = PurePosixPath(file_name.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(f"Unsafe COCO file_name for image {image_id}: {file_name}")
        return self.image_root.joinpath(*normalized.parts)

    def summary(self) -> dict[str, Any]:
        return {
            "annotation_path": str(self.annotation_path),
            "image_root": str(self.image_root),
            "image_count": len(self.images),
            "annotation_count": self.annotation_count,
            "source_image_count": self.source_image_count,
            "source_annotation_count": self.source_annotation_count,
            "image_filter": {
                "path_prefix": self.image_path_prefix,
                "name_pattern": self.image_name_pattern,
            },
            "category_count": len(self.categories),
            "categories": self.categories,
        }


def _normalize_filter_prefix(prefix: str | None) -> str | None:
    if prefix is None:
        return None
    normalized = prefix.replace("\\", "/").strip("/")
    return f"{normalized}/" if normalized else None


def _image_matches_filter(
    file_name: str,
    path_prefix: str | None,
    name_pattern: str | None,
) -> bool:
    normalized = file_name.replace("\\", "/").lstrip("/")
    if path_prefix is not None and not normalized.startswith(path_prefix):
        return False
    if name_pattern is not None:
        basename = PurePosixPath(normalized).name
        if not fnmatchcase(basename, name_pattern):
            return False
    return True


def _required_int(record: dict[str, Any], field: str, context: str) -> int:
    if field not in record:
        raise ValueError(f"Missing {field} in {context}")
    try:
        return int(record[field])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer {field} in {context}: {record[field]!r}") from error


def _required_positive_int(
    record: dict[str, Any],
    field: str,
    context: str,
) -> int:
    value = _required_int(record, field, context)
    if value <= 0:
        raise ValueError(f"Non-positive {field} in {context}: {value}")
    return value
