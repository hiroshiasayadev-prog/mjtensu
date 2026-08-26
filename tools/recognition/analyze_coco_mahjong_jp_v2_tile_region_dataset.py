from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from analyze_coco_mahjong_tile_region_dataset import (
    analyze_split,
    format_number,
    load_json,
    markdown_distribution_row,
)


SPLIT_DIRECTORY_NAMES = ("train", "valid", "test")
ANNOTATION_FILE_NAME = "_annotations.coco.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the local Mahjong-jp v2 COCO export for single-class "
            "mahjong-tile region detection experiments."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/coco_mahjong_jp_v2"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            ".local/recognition/coco_mahjong_jp_v2_dataset_analysis"
        ),
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


def get_split_paths(dataset_root: Path, split_name: str) -> tuple[Path, Path]:
    split_directory = dataset_root / split_name
    return split_directory / ANNOTATION_FILE_NAME, split_directory


def collect_split_file_names(annotation_path: Path) -> set[str]:
    payload = load_json(annotation_path)
    return {
        str(image["file_name"])
        for image in payload.get("images", [])
        if isinstance(image, dict) and "file_name" in image
    }


def build_cross_split_summary(
    analyses: dict[str, dict[str, Any]],
    annotation_paths: dict[str, Path],
) -> dict[str, Any]:
    file_names_by_split = {
        split_name: collect_split_file_names(annotation_path)
        for split_name, annotation_path in annotation_paths.items()
    }

    shared_file_names_by_pair: dict[str, list[str]] = {}
    shared_metadata_keys_by_pair: dict[str, list[str]] = {}

    for first_split, second_split in combinations(SPLIT_DIRECTORY_NAMES, 2):
        pair_name = f"{first_split}__{second_split}"
        shared_file_names_by_pair[pair_name] = sorted(
            file_names_by_split[first_split] & file_names_by_split[second_split]
        )

        first_metadata_keys = set(
            analyses[first_split]["image_metadata"]["extra_image_keys"]
        )
        second_metadata_keys = set(
            analyses[second_split]["image_metadata"]["extra_image_keys"]
        )
        shared_metadata_keys_by_pair[pair_name] = sorted(
            first_metadata_keys & second_metadata_keys
        )

    return {
        "shared_file_names_by_split_pair": shared_file_names_by_pair,
        "shared_noncanonical_image_metadata_keys_by_split_pair": (
            shared_metadata_keys_by_pair
        ),
    }


def render_split_section(split_name: str, split: dict[str, Any]) -> list[str]:
    inventory = split["inventory"]
    category_integrity = split["category_integrity"]
    bbox_distribution = split["bbox_distribution"]
    tiles_per_image = split["tiles_per_image"]
    close_alignment = split["close_alignment_and_overlap"]
    bbox_integrity = split["bbox_integrity"]
    source = split["source"]

    lines = [
        f"## {split_name} split",
        "",
        "| measure | value |",
        "|---|---:|",
        f"| JSON images | {inventory['json_image_count']} |",
        f"| Image files | {inventory['image_file_count']} |",
        f"| Annotations | {inventory['annotation_count']} |",
        f"| Category entries | {inventory['category_entry_count']} |",
        f"| Images without annotations | {inventory['images_without_annotations']} |",
        f"| Exact fourteen-tile images | {tiles_per_image['exact_14_image_count']} |",
        f"| Images with at least fourteen tiles | {tiles_per_image['at_least_14_image_count']} |",
        (
            "| Images with a heuristic tight run of at least fourteen | "
            f"{close_alignment['images_with_tight_run_at_least_14']} |"
        ),
        f"| Maximum tile count | {tiles_per_image['maximum_tile_count']} |",
        f"| Clipped bounding boxes | {bbox_integrity['clipped_bbox_count']} |",
        f"| COCO license entries | {source['license_entry_count']} |",
        "",
        "### Category integrity",
        "",
        f"- Duplicate category IDs: `{category_integrity['duplicate_category_ids']}`",
        f"- Duplicate category names: `{category_integrity['duplicate_category_names']}`",
        (
            "- Undefined referenced category IDs: "
            f"`{category_integrity['annotation_category_ids_without_definition']}`"
        ),
        "",
        "### Bounding-box distributions",
        "",
        "| measure | min | p05 | p25 | median | p75 | p95 | max | mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        markdown_distribution_row("width pixels", bbox_distribution["width_pixels"]),
        markdown_distribution_row("height pixels", bbox_distribution["height_pixels"]),
        markdown_distribution_row(
            "width / height",
            bbox_distribution["aspect_ratio_width_over_height"],
        ),
        markdown_distribution_row(
            "normalized width", bbox_distribution["normalized_width"]
        ),
        markdown_distribution_row(
            "normalized height", bbox_distribution["normalized_height"]
        ),
        markdown_distribution_row(
            "normalized area", bbox_distribution["normalized_area"]
        ),
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

    candidates = close_alignment["candidate_images"]
    if not candidates:
        lines.append("| N/A | 0 | 0 | N/A | 0 | False |")
    else:
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
    lines.append("")
    return lines


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Mahjong-jp v2 tile-region dataset analysis",
        "",
        (
            "Generated by "
            "`tools/recognition/analyze_coco_mahjong_jp_v2_tile_region_dataset.py`."
        ),
        "",
        "This report evaluates the dataset only for tile-region detection.",
        "Tile-type classifier training remains outside this report and is restricted to this Japanese-source dataset.",
        "",
    ]

    for split_name in SPLIT_DIRECTORY_NAMES:
        lines.extend(render_split_section(split_name, summary["splits"][split_name]))

    lines.extend(
        [
            "## Cross-split checks",
            "",
        ]
    )

    cross_split = summary["cross_split"]
    for pair_name, shared_file_names in cross_split[
        "shared_file_names_by_split_pair"
    ].items():
        lines.append(f"- Shared file names for `{pair_name}`: `{shared_file_names}`")

    for pair_name, metadata_keys in cross_split[
        "shared_noncanonical_image_metadata_keys_by_split_pair"
    ].items():
        lines.append(
            f"- Shared noncanonical image metadata keys for `{pair_name}`: `{metadata_keys}`"
        )

    lines.extend(
        [
            "",
            (
                "Matching file names do not detect renamed copies or near-duplicate frames. "
                "If capture-session metadata is absent, image-similarity or manual grouping is "
                "required before treating the provided split as leakage-safe."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    analyses: dict[str, dict[str, Any]] = {}
    annotation_paths: dict[str, Path] = {}

    for split_name in SPLIT_DIRECTORY_NAMES:
        annotation_path, image_directory = get_split_paths(
            args.dataset_root, split_name
        )
        annotation_paths[split_name] = annotation_path
        analyses[split_name] = analyze_split(
            split_name=split_name,
            annotation_path=annotation_path,
            image_directory=image_directory,
            maximum_gap_ratio=args.tight_adjacency_gap_ratio,
            minimum_vertical_overlap_ratio=args.minimum_vertical_overlap_ratio,
            candidate_limit=args.candidate_limit,
        )

    summary = {
        "dataset": {
            "name": "mahjong-jp v2",
            "root": str(args.dataset_root),
            "detector_eligibility": True,
            "tile_classifier_eligibility": True,
            "source_identity": "coco_mahjong_jp_v2",
        },
        "analysis_parameters": {
            "tight_adjacency_gap_ratio": args.tight_adjacency_gap_ratio,
            "minimum_vertical_overlap_ratio": args.minimum_vertical_overlap_ratio,
            "candidate_limit": args.candidate_limit,
        },
        "splits": analyses,
        "cross_split": build_cross_split_summary(analyses, annotation_paths),
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    json_output = (
        args.output_directory
        / "coco_mahjong_jp_v2_tile_region_dataset_analysis.json"
    )
    markdown_output = (
        args.output_directory
        / "coco_mahjong_jp_v2_tile_region_dataset_analysis.md"
    )

    json_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown_report(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "json": str(json_output),
                "markdown": str(markdown_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
