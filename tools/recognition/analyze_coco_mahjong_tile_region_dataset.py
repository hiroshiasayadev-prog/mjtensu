from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class BoundingBox:
    annotation_id: int
    image_id: int
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    extra_keys: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the existing COCO mahjong dataset for single-class tile-region "
            "detection experiments."
        )
    )
    parser.add_argument(
        "--train-annotations",
        type=Path,
        default=Path("data/coco_mahjong/annotations/instances_train2017.json"),
    )
    parser.add_argument(
        "--train-images",
        type=Path,
        default=Path("data/coco_mahjong/train2017"),
    )
    parser.add_argument(
        "--validation-annotations",
        type=Path,
        default=Path("data/coco_mahjong/annotations/instances_val2017.json"),
    )
    parser.add_argument(
        "--validation-images",
        type=Path,
        default=Path("data/coco_mahjong/val2017"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(".local/recognition/coco_mahjong_dataset_analysis"),
    )
    parser.add_argument(
        "--tight-adjacency-gap-ratio",
        type=float,
        default=0.08,
        help="Maximum horizontal gap divided by the smaller adjacent-box width.",
    )
    parser.add_argument(
        "--minimum-vertical-overlap-ratio",
        type=float,
        default=0.50,
        help="Minimum vertical intersection divided by the smaller box height.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=50,
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"COCO root must be an object: {path}")
    return payload


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_weight = upper_index - position
    upper_weight = position - lower_index
    return ordered[lower_index] * lower_weight + ordered[upper_index] * upper_weight


def describe_numeric(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    intersection_width = max(0.0, min(first.right, second.right) - max(first.x, second.x))
    intersection_height = max(0.0, min(first.bottom, second.bottom) - max(first.y, second.y))
    intersection_area = intersection_width * intersection_height
    if intersection_area <= 0.0:
        return 0.0
    first_area = first.width * first.height
    second_area = second.width * second.height
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area if union_area > 0.0 else 0.0


def vertical_overlap_ratio(first: BoundingBox, second: BoundingBox) -> float:
    overlap = max(0.0, min(first.bottom, second.bottom) - max(first.y, second.y))
    denominator = min(first.height, second.height)
    return overlap / denominator if denominator > 0.0 else 0.0


def horizontal_gap_ratio(first: BoundingBox, second: BoundingBox) -> float:
    gap = second.x - first.right
    denominator = min(first.width, second.width)
    return gap / denominator if denominator > 0.0 else math.inf


def longest_tight_horizontal_run(
    boxes: Sequence[BoundingBox],
    maximum_gap_ratio: float,
    minimum_vertical_overlap_ratio: float,
) -> tuple[int, list[int]]:
    if not boxes:
        return 0, []

    ordered = sorted(boxes, key=lambda box: (box.center_x, box.center_y, box.annotation_id))
    best_ids = [ordered[0].annotation_id]
    current_ids = [ordered[0].annotation_id]

    for previous, current in zip(ordered, ordered[1:]):
        is_tight = (
            vertical_overlap_ratio(previous, current) >= minimum_vertical_overlap_ratio
            and horizontal_gap_ratio(previous, current) <= maximum_gap_ratio
        )
        if is_tight:
            current_ids.append(current.annotation_id)
        else:
            current_ids = [current.annotation_id]
        if len(current_ids) > len(best_ids):
            best_ids = list(current_ids)

    return len(best_ids), best_ids


def parse_images(raw_images: Iterable[dict[str, Any]]) -> dict[int, ImageRecord]:
    records: dict[int, ImageRecord] = {}
    canonical_keys = {"id", "file_name", "width", "height"}
    for raw_image in raw_images:
        image_id = int(raw_image["id"])
        if image_id in records:
            raise ValueError(f"Duplicate image id: {image_id}")
        records[image_id] = ImageRecord(
            image_id=image_id,
            file_name=str(raw_image["file_name"]),
            width=int(raw_image["width"]),
            height=int(raw_image["height"]),
            extra_keys=tuple(sorted(set(raw_image) - canonical_keys)),
        )
    return records


def parse_bounding_boxes(
    raw_annotations: Iterable[dict[str, Any]],
) -> tuple[list[BoundingBox], Counter[int], list[int], list[int]]:
    boxes: list[BoundingBox] = []
    category_counts: Counter[int] = Counter()
    annotation_ids: list[int] = []
    invalid_annotation_ids: list[int] = []

    for raw_annotation in raw_annotations:
        annotation_id = int(raw_annotation["id"])
        image_id = int(raw_annotation["image_id"])
        category_id = int(raw_annotation["category_id"])
        raw_bbox = raw_annotation.get("bbox")
        annotation_ids.append(annotation_id)
        category_counts[category_id] += 1

        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            invalid_annotation_ids.append(annotation_id)
            continue

        x, y, width, height = (float(value) for value in raw_bbox)
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            invalid_annotation_ids.append(annotation_id)
            continue

        boxes.append(
            BoundingBox(
                annotation_id=annotation_id,
                image_id=image_id,
                x=x,
                y=y,
                width=width,
                height=height,
            )
        )

    return boxes, category_counts, annotation_ids, invalid_annotation_ids


def duplicate_values(values: Iterable[Any]) -> list[Any]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def list_image_files(image_directory: Path) -> set[str]:
    if not image_directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_directory}")
    return {
        path.name
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }


def analyze_split(
    split_name: str,
    annotation_path: Path,
    image_directory: Path,
    maximum_gap_ratio: float,
    minimum_vertical_overlap_ratio: float,
    candidate_limit: int,
) -> dict[str, Any]:
    payload = load_json(annotation_path)
    raw_images = payload.get("images", [])
    raw_annotations = payload.get("annotations", [])
    raw_categories = payload.get("categories", [])

    if not isinstance(raw_images, list):
        raise ValueError(f"images must be a list: {annotation_path}")
    if not isinstance(raw_annotations, list):
        raise ValueError(f"annotations must be a list: {annotation_path}")
    if not isinstance(raw_categories, list):
        raise ValueError(f"categories must be a list: {annotation_path}")

    images = parse_images(raw_images)
    boxes, category_counts, annotation_ids, invalid_bbox_shape_ids = parse_bounding_boxes(
        raw_annotations
    )
    boxes_by_image: dict[int, list[BoundingBox]] = defaultdict(list)
    for box in boxes:
        boxes_by_image[box.image_id].append(box)

    image_file_names = list_image_files(image_directory)
    json_file_names = {image.file_name for image in images.values()}

    widths: list[float] = []
    heights: list[float] = []
    aspect_ratios: list[float] = []
    areas: list[float] = []
    normalized_widths: list[float] = []
    normalized_heights: list[float] = []
    normalized_areas: list[float] = []
    non_positive_bbox_ids: list[int] = []
    missing_image_reference_ids: list[int] = []
    clipped_annotation_ids: list[int] = []
    clipped_by_edge: dict[str, list[int]] = {
        "left": [],
        "top": [],
        "right": [],
        "bottom": [],
    }

    for box in boxes:
        image = images.get(box.image_id)
        if image is None:
            missing_image_reference_ids.append(box.annotation_id)
            continue
        if box.width <= 0.0 or box.height <= 0.0:
            non_positive_bbox_ids.append(box.annotation_id)
            continue

        widths.append(box.width)
        heights.append(box.height)
        aspect_ratios.append(box.width / box.height)
        areas.append(box.width * box.height)
        normalized_widths.append(box.width / image.width)
        normalized_heights.append(box.height / image.height)
        normalized_areas.append((box.width * box.height) / (image.width * image.height))

        clipped = False
        if box.x <= 0.0:
            clipped_by_edge["left"].append(box.annotation_id)
            clipped = True
        if box.y <= 0.0:
            clipped_by_edge["top"].append(box.annotation_id)
            clipped = True
        if box.right >= image.width:
            clipped_by_edge["right"].append(box.annotation_id)
            clipped = True
        if box.bottom >= image.height:
            clipped_by_edge["bottom"].append(box.annotation_id)
            clipped = True
        if clipped:
            clipped_annotation_ids.append(box.annotation_id)

    per_image_counts = {image_id: len(boxes_by_image.get(image_id, [])) for image_id in images}
    count_values = list(per_image_counts.values())

    image_overlap_summaries: list[dict[str, Any]] = []
    duplicate_like_pairs: list[dict[str, Any]] = []
    all_positive_pair_ious: list[float] = []
    candidate_rows: list[dict[str, Any]] = []

    for image_id, image_boxes in boxes_by_image.items():
        positive_overlap_pairs = 0
        overlap_pairs_at_least_010 = 0
        maximum_pair_iou = 0.0

        for first_index, first in enumerate(image_boxes):
            for second in image_boxes[first_index + 1 :]:
                pair_iou = intersection_over_union(first, second)
                maximum_pair_iou = max(maximum_pair_iou, pair_iou)
                if pair_iou > 0.0:
                    positive_overlap_pairs += 1
                    all_positive_pair_ious.append(pair_iou)
                if pair_iou >= 0.10:
                    overlap_pairs_at_least_010 += 1
                if pair_iou >= 0.90:
                    duplicate_like_pairs.append(
                        {
                            "image_id": image_id,
                            "first_annotation_id": first.annotation_id,
                            "second_annotation_id": second.annotation_id,
                            "iou": pair_iou,
                        }
                    )

        run_length, run_annotation_ids = longest_tight_horizontal_run(
            image_boxes,
            maximum_gap_ratio=maximum_gap_ratio,
            minimum_vertical_overlap_ratio=minimum_vertical_overlap_ratio,
        )

        image = images.get(image_id)
        image_overlap_summaries.append(
            {
                "image_id": image_id,
                "file_name": image.file_name if image else None,
                "tile_count": len(image_boxes),
                "positive_overlap_pair_count": positive_overlap_pairs,
                "overlap_pair_count_iou_at_least_0_10": overlap_pairs_at_least_010,
                "maximum_pair_iou": maximum_pair_iou,
                "longest_tight_horizontal_run": run_length,
            }
        )

        if len(image_boxes) >= 14 or run_length >= 14:
            candidate_rows.append(
                {
                    "image_id": image_id,
                    "file_name": image.file_name if image else None,
                    "tile_count": len(image_boxes),
                    "longest_tight_horizontal_run": run_length,
                    "tight_run_annotation_ids": run_annotation_ids,
                    "maximum_pair_iou": maximum_pair_iou,
                    "positive_overlap_pair_count": positive_overlap_pairs,
                    "contains_clipped_bbox": any(
                        box.annotation_id in set(clipped_annotation_ids) for box in image_boxes
                    ),
                }
            )

    candidate_rows.sort(
        key=lambda candidate: (
            candidate["longest_tight_horizontal_run"],
            candidate["tile_count"],
            candidate["positive_overlap_pair_count"],
        ),
        reverse=True,
    )

    category_ids = [int(category["id"]) for category in raw_categories]
    category_names = [str(category["name"]) for category in raw_categories]
    defined_category_ids = set(category_ids)

    extra_image_keys = Counter(
        extra_key for image in images.values() for extra_key in image.extra_keys
    )

    images_with_annotations = sum(1 for count in count_values if count > 0)
    exact_14_image_ids = sorted(image_id for image_id, count in per_image_counts.items() if count == 14)
    at_least_14_image_ids = sorted(image_id for image_id, count in per_image_counts.items() if count >= 14)
    tight_run_at_least_14 = [
        candidate for candidate in candidate_rows if candidate["longest_tight_horizontal_run"] >= 14
    ]

    return {
        "split": split_name,
        "source": {
            "annotation_path": str(annotation_path),
            "image_directory": str(image_directory),
            "has_info_object": isinstance(payload.get("info"), dict),
            "license_entry_count": len(payload.get("licenses", []))
            if isinstance(payload.get("licenses"), list)
            else None,
        },
        "inventory": {
            "json_image_count": len(images),
            "image_file_count": len(image_file_names),
            "annotation_count": len(raw_annotations),
            "parsed_bbox_count": len(boxes),
            "category_entry_count": len(raw_categories),
            "images_with_annotations": images_with_annotations,
            "images_without_annotations": len(images) - images_with_annotations,
            "missing_image_files": sorted(json_file_names - image_file_names),
            "unreferenced_image_files": sorted(image_file_names - json_file_names),
        },
        "category_integrity": {
            "categories": raw_categories,
            "duplicate_category_ids": duplicate_values(category_ids),
            "duplicate_category_names": duplicate_values(category_names),
            "annotation_counts_by_category_id": {
                str(category_id): count
                for category_id, count in sorted(category_counts.items())
            },
            "annotation_category_ids_without_definition": sorted(
                set(category_counts) - defined_category_ids
            ),
            "defined_category_ids_without_annotations": sorted(
                defined_category_ids - set(category_counts)
            ),
        },
        "identifier_integrity": {
            "duplicate_annotation_ids": duplicate_values(annotation_ids),
            "invalid_bbox_shape_annotation_ids": invalid_bbox_shape_ids,
            "missing_image_reference_annotation_ids": missing_image_reference_ids,
        },
        "image_metadata": {
            "extra_image_keys": dict(sorted(extra_image_keys.items())),
        },
        "bbox_distribution": {
            "width_pixels": describe_numeric(widths),
            "height_pixels": describe_numeric(heights),
            "aspect_ratio_width_over_height": describe_numeric(aspect_ratios),
            "area_pixels": describe_numeric(areas),
            "normalized_width": describe_numeric(normalized_widths),
            "normalized_height": describe_numeric(normalized_heights),
            "normalized_area": describe_numeric(normalized_areas),
        },
        "bbox_integrity": {
            "non_positive_bbox_annotation_ids": non_positive_bbox_ids,
            "clipped_bbox_count": len(set(clipped_annotation_ids)),
            "clipped_bbox_rate": (
                len(set(clipped_annotation_ids)) / len(boxes) if boxes else None
            ),
            "clipped_annotation_ids_by_edge": clipped_by_edge,
            "duplicate_like_pairs_iou_at_least_0_90": duplicate_like_pairs,
        },
        "tiles_per_image": {
            "distribution": describe_numeric([float(value) for value in count_values]),
            "histogram": {
                str(count): frequency
                for count, frequency in sorted(Counter(count_values).items())
            },
            "exact_14_image_count": len(exact_14_image_ids),
            "exact_14_image_ids": exact_14_image_ids,
            "at_least_14_image_count": len(at_least_14_image_ids),
            "at_least_14_image_ids": at_least_14_image_ids,
            "maximum_tile_count": max(count_values) if count_values else 0,
        },
        "close_alignment_and_overlap": {
            "tight_adjacency_gap_ratio": maximum_gap_ratio,
            "minimum_vertical_overlap_ratio": minimum_vertical_overlap_ratio,
            "images_with_tight_run_at_least_14": len(tight_run_at_least_14),
            "positive_pair_iou_distribution": describe_numeric(all_positive_pair_ious),
            "candidate_images": candidate_rows[:candidate_limit],
            "all_image_summaries": image_overlap_summaries,
        },
    }


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_distribution_row(name: str, distribution: dict[str, Any]) -> str:
    return "| " + " | ".join(
        [
            name,
            format_number(distribution["minimum"]),
            format_number(distribution["p05"]),
            format_number(distribution["p25"]),
            format_number(distribution["median"]),
            format_number(distribution["p75"]),
            format_number(distribution["p95"]),
            format_number(distribution["maximum"]),
            format_number(distribution["mean"]),
        ]
    ) + " |"


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# COCO mahjong tile-region dataset analysis",
        "",
        "Generated by `tools/recognition/analyze_coco_mahjong_tile_region_dataset.py`.",
        "",
    ]

    for split_name in ("train", "validation"):
        split = summary["splits"][split_name]
        inventory = split["inventory"]
        category_integrity = split["category_integrity"]
        bbox_distribution = split["bbox_distribution"]
        tiles_per_image = split["tiles_per_image"]
        close_alignment = split["close_alignment_and_overlap"]
        bbox_integrity = split["bbox_integrity"]

        lines.extend(
            [
                f"## {split_name.capitalize()} split",
                "",
                "| measure | value |",
                "|---|---:|",
                f"| JSON images | {inventory['json_image_count']} |",
                f"| Image files | {inventory['image_file_count']} |",
                f"| Annotations | {inventory['annotation_count']} |",
                f"| Category entries | {inventory['category_entry_count']} |",
                f"| Exact fourteen-tile images | {tiles_per_image['exact_14_image_count']} |",
                f"| Images with at least fourteen tiles | {tiles_per_image['at_least_14_image_count']} |",
                f"| Images with a heuristic tight run of at least fourteen | {close_alignment['images_with_tight_run_at_least_14']} |",
                f"| Maximum tile count | {tiles_per_image['maximum_tile_count']} |",
                f"| Clipped bounding boxes | {bbox_integrity['clipped_bbox_count']} |",
                "",
                "### Category integrity",
                "",
                f"- Duplicate category IDs: `{category_integrity['duplicate_category_ids']}`",
                f"- Duplicate category names: `{category_integrity['duplicate_category_names']}`",
                f"- Undefined referenced category IDs: `{category_integrity['annotation_category_ids_without_definition']}`",
                "",
                "### Bounding-box distributions",
                "",
                "| measure | min | p05 | p25 | median | p75 | p95 | max | mean |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                markdown_distribution_row("width pixels", bbox_distribution["width_pixels"]),
                markdown_distribution_row("height pixels", bbox_distribution["height_pixels"]),
                markdown_distribution_row(
                    "width / height", bbox_distribution["aspect_ratio_width_over_height"]
                ),
                markdown_distribution_row("normalized width", bbox_distribution["normalized_width"]),
                markdown_distribution_row("normalized height", bbox_distribution["normalized_height"]),
                markdown_distribution_row("normalized area", bbox_distribution["normalized_area"]),
                markdown_distribution_row(
                    "tiles per image", tiles_per_image["distribution"]
                ),
                "",
                "### Fourteen-tile and close-alignment candidates",
                "",
                "The tight-run result is a screening heuristic, not ground truth for a hand row.",
                "",
                "| file | tile count | tight run | max pair IoU | overlapping pairs | clipped bbox |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )

        candidates = close_alignment["candidate_images"]
        if candidates:
            for candidate in candidates:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(candidate["file_name"]),
                            str(candidate["tile_count"]),
                            str(candidate["longest_tight_horizontal_run"]),
                            format_number(candidate["maximum_pair_iou"]),
                            str(candidate["positive_overlap_pair_count"]),
                            str(candidate["contains_clipped_bbox"]),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("| N/A | 0 | 0 | N/A | 0 | False |")
        lines.append("")

    leakage = summary["cross_split"]
    lines.extend(
        [
            "## Cross-split checks",
            "",
            f"- Shared file names: `{leakage['shared_file_names']}`",
            f"- Shared source-group metadata keys: `{leakage['shared_noncanonical_image_metadata_keys']}`",
            "",
            "The dataset does not establish capture-session, tile-set, background, or environment groups unless those fields appear above.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    train = analyze_split(
        split_name="train",
        annotation_path=args.train_annotations,
        image_directory=args.train_images,
        maximum_gap_ratio=args.tight_adjacency_gap_ratio,
        minimum_vertical_overlap_ratio=args.minimum_vertical_overlap_ratio,
        candidate_limit=args.candidate_limit,
    )
    validation = analyze_split(
        split_name="validation",
        annotation_path=args.validation_annotations,
        image_directory=args.validation_images,
        maximum_gap_ratio=args.tight_adjacency_gap_ratio,
        minimum_vertical_overlap_ratio=args.minimum_vertical_overlap_ratio,
        candidate_limit=args.candidate_limit,
    )

    train_file_names = {
        image["file_name"] for image in load_json(args.train_annotations).get("images", [])
    }
    validation_file_names = {
        image["file_name"]
        for image in load_json(args.validation_annotations).get("images", [])
    }
    train_extra_keys = set(train["image_metadata"]["extra_image_keys"])
    validation_extra_keys = set(validation["image_metadata"]["extra_image_keys"])

    summary = {
        "analysis_parameters": {
            "tight_adjacency_gap_ratio": args.tight_adjacency_gap_ratio,
            "minimum_vertical_overlap_ratio": args.minimum_vertical_overlap_ratio,
            "candidate_limit": args.candidate_limit,
        },
        "splits": {
            "train": train,
            "validation": validation,
        },
        "cross_split": {
            "shared_file_names": sorted(train_file_names & validation_file_names),
            "shared_noncanonical_image_metadata_keys": sorted(
                train_extra_keys & validation_extra_keys
            ),
        },
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    json_output = args.output_directory / "coco_mahjong_tile_region_dataset_analysis.json"
    markdown_output = args.output_directory / "coco_mahjong_tile_region_dataset_analysis.md"

    json_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown_report(summary), encoding="utf-8")

    print(json.dumps({"json": str(json_output), "markdown": str(markdown_output)}, indent=2))


if __name__ == "__main__":
    main()
