from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .coco import CocoDataset
from .composer import DEFAULT_MIN_RETAINED_AREA_RATIO
from .image_io import load_coco_image
from .dataset import OutputDatasetManager
from .geometry import ANNOTATION_SELECTION_POLICIES


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Create manual 320x320 fixed-capture composites and COCO tile "
            "annotations from an existing COCO dataset."
        )
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=(
            root
            / ".local"
            / "recognition"
            / "nanodet_single_class_dataset"
            / "annotations"
            / "instances_train.json"
        ),
        help="Input COCO annotation JSON.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=root / "data",
        help="Root directory used to resolve COCO image file_name values.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            root
            / ".local"
            / "recognition"
            / "composite_capture_test_dataset"
        ),
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--image-path-prefix",
        default="coco_mahjong_jp_v2/train/",
        help=(
            "Only include COCO image file_name values beneath this normalized "
            "path prefix. Defaults to the Japanese v2 training source."
        ),
    )
    parser.add_argument(
        "--image-name-pattern",
        default="img_00*.jpg",
        help=(
            "Only include image basenames matching this glob pattern. "
            "Defaults to img_00*.jpg."
        ),
    )
    parser.add_argument(
        "--annotation-selection-policy",
        choices=ANNOTATION_SELECTION_POLICIES,
        default="center",
        help=(
            "How a source bbox is retained for a crop: center (default), "
            "contained, or intersect. Retained bboxes are clipped to the crop."
        ),
    )
    parser.add_argument(
        "--min-retained-area-ratio",
        type=float,
        default=DEFAULT_MIN_RETAINED_AREA_RATIO,
        help=(
            "Retain a source bbox only when more than this fraction of its "
            "original area is inside the selected crop. Defaults to 0.6, so "
            "boxes with 60%% or less retained area are excluded."
        ),
    )
    parser.add_argument(
        "--repair-existing-only",
        action="store_true",
        help=(
            "Rebuild annotations for already-saved composites from their source "
            "COCO provenance and capture regions, then exit without opening the GUI."
        ),
    )
    parser.add_argument(
        "--start-image",
        type=int,
        default=1,
        metavar="N",
        help="One-based source image index opened at startup (default: 1).",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate COCO loading and the first source image, print JSON, and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.start_image < 1:
        raise SystemExit("--start-image must be at least 1")
    if not 0.0 <= args.min_retained_area_ratio < 1.0:
        raise SystemExit(
            "--min-retained-area-ratio must be at least zero and less than one"
        )

    if args.repair_existing_only:
        output_manager = OutputDatasetManager(args.output_directory)
        report = output_manager.rebuild_existing_annotations(
            min_retained_area_ratio=args.min_retained_area_ratio,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    dataset = CocoDataset.load(
        args.annotations,
        args.image_root,
        image_path_prefix=args.image_path_prefix,
        image_name_pattern=args.image_name_pattern,
    )
    if not dataset.images:
        raise SystemExit(
            "No COCO images matched the configured filters: "
            f"path_prefix={args.image_path_prefix!r}, "
            f"name_pattern={args.image_name_pattern!r}"
        )
    if args.start_image > len(dataset.images):
        raise SystemExit(
            f"--start-image {args.start_image} exceeds image count "
            f"{len(dataset.images)}"
        )

    if args.check_inputs:
        first_image = dataset.image_at(0)
        first_image_path = dataset.resolve_image_path(first_image)
        loaded_image = load_coco_image(first_image_path)
        report = dataset.summary()
        report.update(
            {
                "first_image_path": str(first_image_path),
                "first_image_exists": first_image_path.is_file(),
                "first_image_declared_size": [
                    int(first_image["width"]),
                    int(first_image["height"]),
                ],
                "first_image_raw_size": list(loaded_image.raw_size),
                "first_image_exif_orientation": loaded_image.exif_orientation,
                "first_image_exif_transpose_applied": (
                    loaded_image.exif_transpose_applied
                ),
                "first_image_actual_size": list(loaded_image.oriented_size),
                "output_directory": str(args.output_directory.resolve()),
            }
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    from tkinter import Tk

    from .gui import CompositeCaptureApp

    output_manager = OutputDatasetManager(args.output_directory)
    repair_report = output_manager.rebuild_existing_annotations(
        min_retained_area_ratio=args.min_retained_area_ratio,
    )
    if repair_report["status"] == "updated":
        print(json.dumps({"existing_annotation_repair": repair_report}, ensure_ascii=False, indent=2))

    root = Tk()
    CompositeCaptureApp(
        root,
        dataset=dataset,
        output_manager=output_manager,
        annotation_selection_policy=args.annotation_selection_policy,
        min_retained_area_ratio=args.min_retained_area_ratio,
        start_index=args.start_image - 1,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
