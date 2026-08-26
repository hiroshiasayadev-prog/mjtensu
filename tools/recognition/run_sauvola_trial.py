from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tools.recognition.run_lab_threshold_trial import (
        DEFAULT_SEED,
        atomic_write_json,
        decode_png,
        label_sort_key,
        paste_fitted,
        rgb_to_lab,
        save_png,
        sqlite_readonly_uri,
    )
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from run_lab_threshold_trial import (
        DEFAULT_SEED,
        atomic_write_json,
        decode_png,
        label_sort_key,
        paste_fitted,
        rgb_to_lab,
        save_png,
        sqlite_readonly_uri,
    )


DEFAULT_WINDOW_SIZE = 15
DEFAULT_SAUVOLA_K = 0.20
DEFAULT_SAUVOLA_DYNAMIC_RANGE = 50.0
DEFAULT_RED_SHARE_MIN = 0.42
DEFAULT_RED_DOMINANCE_MIN = 0.08
DEFAULT_COLORFULNESS_MIN = 0.12
DEFAULT_MIN_CHANNEL_INTENSITY = 0.05
DEFAULT_NEUTRAL_LIGHTNESS_QUANTILE = 0.45
DEFAULT_NEUTRAL_CHROMA_QUANTILE = 0.40
DEFAULT_NEUTRAL_MIN_FRACTION = 0.05
DEFAULT_NEUTRAL_MAX_SPREAD = 10.0
DEFAULT_RELATIVE_A_RED_MIN = 8.0
DEFAULT_RELATIVE_RED_CHROMA_MIN = 10.0
DEFAULT_RELATIVE_RED_DOMINANCE_MIN = 0.025
DEFAULT_RED_FIELD_MAX_FRACTION = 0.40
DEFAULT_RED_COMPONENT_MAX_FRACTION = 0.30
DEFAULT_RED_COMPONENT_MAX_BORDER_SIDES = 1


@dataclass(frozen=True)
class SauvolaParameters:
    window_size: int = DEFAULT_WINDOW_SIZE
    k: float = DEFAULT_SAUVOLA_K
    dynamic_range: float = DEFAULT_SAUVOLA_DYNAMIC_RANGE
    red_share_min: float = DEFAULT_RED_SHARE_MIN
    red_dominance_min: float = DEFAULT_RED_DOMINANCE_MIN
    colorfulness_min: float = DEFAULT_COLORFULNESS_MIN
    min_channel_intensity: float = DEFAULT_MIN_CHANNEL_INTENSITY
    neutral_lightness_quantile: float = DEFAULT_NEUTRAL_LIGHTNESS_QUANTILE
    neutral_chroma_quantile: float = DEFAULT_NEUTRAL_CHROMA_QUANTILE
    neutral_min_fraction: float = DEFAULT_NEUTRAL_MIN_FRACTION
    neutral_max_spread: float = DEFAULT_NEUTRAL_MAX_SPREAD
    relative_a_red_min: float = DEFAULT_RELATIVE_A_RED_MIN
    relative_red_chroma_min: float = DEFAULT_RELATIVE_RED_CHROMA_MIN
    relative_red_dominance_min: float = DEFAULT_RELATIVE_RED_DOMINANCE_MIN
    red_field_max_fraction: float = DEFAULT_RED_FIELD_MAX_FRACTION
    red_component_max_fraction: float = DEFAULT_RED_COMPONENT_MAX_FRACTION
    red_component_max_border_sides: int = DEFAULT_RED_COMPONENT_MAX_BORDER_SIDES


@dataclass(frozen=True)
class ImageMetric:
    crop_id: str
    source: str
    source_partition: str
    tile_label: str
    image_width: int
    image_height: int
    pixel_count: int
    white_pixels: int
    black_pixels: int
    red_pixels: int
    raw_red_pixels: int
    rejected_red_pixels: int
    sauvola_dark_pixels: int
    chromatic_black_pixels: int
    neutral_reference_pixels: int
    neutral_reference_reliable: int
    white_ratio: float
    black_ratio: float
    red_ratio: float
    raw_red_ratio: float
    rejected_red_ratio: float
    largest_raw_red_component_ratio: float
    red_field_rejected: int
    rejected_red_component_count: int
    sauvola_dark_ratio: float
    chromatic_black_ratio: float
    neutral_reference_ratio: float
    neutral_reference_a: float
    neutral_reference_b: float
    neutral_reference_spread: float
    red_detection_mode: str
    source_image_path: str
    source_image_id: str | None
    brightness: str | None
    shadow: str | None


@dataclass(frozen=True)
class ThresholdResult:
    lightness_image: Image.Image
    sauvola_mask_image: Image.Image
    neutral_reference_mask_image: Image.Image
    raw_red_mask_image: Image.Image
    red_mask_image: Image.Image
    ternary_image: Image.Image
    metrics: dict[str, int | float | str]


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run an adaptive three-color baseline using Sauvola on CIELAB L*, "
            "crop-relative neutral-reference red detection, and non-red chromatic "
            "ink collapsed to black."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--sample-database",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/"
            "color_trials/sample_seed42.sqlite."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/"
            "color_trials/sauvola_seed42."
        ),
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--k", type=float, default=DEFAULT_SAUVOLA_K)
    parser.add_argument(
        "--dynamic-range",
        type=float,
        default=DEFAULT_SAUVOLA_DYNAMIC_RANGE,
        help="Expected local standard-deviation range in CIELAB L* units.",
    )
    parser.add_argument("--red-share-min", type=float, default=DEFAULT_RED_SHARE_MIN)
    parser.add_argument(
        "--red-dominance-min",
        type=float,
        default=DEFAULT_RED_DOMINANCE_MIN,
    )
    parser.add_argument(
        "--colorfulness-min",
        type=float,
        default=DEFAULT_COLORFULNESS_MIN,
    )
    parser.add_argument(
        "--min-channel-intensity",
        type=float,
        default=DEFAULT_MIN_CHANNEL_INTENSITY,
    )
    parser.add_argument(
        "--neutral-lightness-quantile",
        type=float,
        default=DEFAULT_NEUTRAL_LIGHTNESS_QUANTILE,
    )
    parser.add_argument(
        "--neutral-chroma-quantile",
        type=float,
        default=DEFAULT_NEUTRAL_CHROMA_QUANTILE,
    )
    parser.add_argument(
        "--neutral-min-fraction",
        type=float,
        default=DEFAULT_NEUTRAL_MIN_FRACTION,
    )
    parser.add_argument(
        "--neutral-max-spread",
        type=float,
        default=DEFAULT_NEUTRAL_MAX_SPREAD,
    )
    parser.add_argument(
        "--relative-a-red-min",
        type=float,
        default=DEFAULT_RELATIVE_A_RED_MIN,
    )
    parser.add_argument(
        "--relative-red-chroma-min",
        type=float,
        default=DEFAULT_RELATIVE_RED_CHROMA_MIN,
    )
    parser.add_argument(
        "--relative-red-dominance-min",
        type=float,
        default=DEFAULT_RELATIVE_RED_DOMINANCE_MIN,
    )
    parser.add_argument(
        "--red-field-max-fraction",
        type=float,
        default=DEFAULT_RED_FIELD_MAX_FRACTION,
        help="Reject the entire raw red mask when it occupies more than this fraction.",
    )
    parser.add_argument(
        "--red-component-max-fraction",
        type=float,
        default=DEFAULT_RED_COMPONENT_MAX_FRACTION,
        help="Reject a connected red component larger than this crop-area fraction.",
    )
    parser.add_argument(
        "--red-component-max-border-sides",
        type=int,
        default=DEFAULT_RED_COMPONENT_MAX_BORDER_SIDES,
        help="Reject a red component touching more crop-border sides than this.",
    )
    parser.add_argument(
        "--diagnostic-count",
        type=int,
        default=24,
        help="Number of samples in each failure-oriented diagnostic sheet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    sample_database = (
        args.sample_database.resolve()
        if args.sample_database is not None
        else repository_root
        / ".local"
        / "recognition"
        / "color_trials"
        / f"sample_seed{DEFAULT_SEED}.sqlite"
    )
    output_directory = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else repository_root
        / ".local"
        / "recognition"
        / "color_trials"
        / f"sauvola_seed{DEFAULT_SEED}"
    )
    parameters = SauvolaParameters(
        window_size=int(args.window_size),
        k=float(args.k),
        dynamic_range=float(args.dynamic_range),
        red_share_min=float(args.red_share_min),
        red_dominance_min=float(args.red_dominance_min),
        colorfulness_min=float(args.colorfulness_min),
        min_channel_intensity=float(args.min_channel_intensity),
        neutral_lightness_quantile=float(args.neutral_lightness_quantile),
        neutral_chroma_quantile=float(args.neutral_chroma_quantile),
        neutral_min_fraction=float(args.neutral_min_fraction),
        neutral_max_spread=float(args.neutral_max_spread),
        relative_a_red_min=float(args.relative_a_red_min),
        relative_red_chroma_min=float(args.relative_red_chroma_min),
        relative_red_dominance_min=float(args.relative_red_dominance_min),
        red_field_max_fraction=float(args.red_field_max_fraction),
        red_component_max_fraction=float(args.red_component_max_fraction),
        red_component_max_border_sides=int(args.red_component_max_border_sides),
    )
    summary = run_trial(
        sample_database=sample_database,
        output_directory=output_directory,
        parameters=parameters,
        diagnostic_count=int(args.diagnostic_count),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_trial(
    *,
    sample_database: Path,
    output_directory: Path,
    parameters: SauvolaParameters,
    diagnostic_count: int = 24,
) -> dict[str, Any]:
    if not sample_database.is_file():
        raise FileNotFoundError(sample_database)
    if diagnostic_count < 1:
        raise ValueError("diagnostic_count must be positive")
    validate_parameters(parameters)

    output_directory.mkdir(parents=True, exist_ok=True)
    metrics: list[ImageMetric] = []
    with closing(sqlite3.connect(sqlite_readonly_uri(sample_database), uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        total = int(connection.execute("SELECT COUNT(*) FROM tile_crop").fetchone()[0])
        cursor = connection.execute(
            """
            SELECT
                crop_id,
                source,
                source_partition,
                tile_label,
                image_width,
                image_height,
                image_png,
                source_image_path,
                source_image_id,
                brightness,
                shadow
            FROM tile_crop
            ORDER BY source, tile_label, crop_id
            """
        )
        for index, row in enumerate(cursor, start=1):
            image = decode_png(bytes(row["image_png"]))
            result = threshold_image(image, parameters)
            values = result.metrics
            metrics.append(
                ImageMetric(
                    crop_id=str(row["crop_id"]),
                    source=str(row["source"]),
                    source_partition=str(row["source_partition"]),
                    tile_label=str(row["tile_label"]),
                    image_width=image.width,
                    image_height=image.height,
                    pixel_count=int(values["pixel_count"]),
                    white_pixels=int(values["white_pixels"]),
                    black_pixels=int(values["black_pixels"]),
                    red_pixels=int(values["red_pixels"]),
                    raw_red_pixels=int(values["raw_red_pixels"]),
                    rejected_red_pixels=int(values["rejected_red_pixels"]),
                    sauvola_dark_pixels=int(values["sauvola_dark_pixels"]),
                    chromatic_black_pixels=int(values["chromatic_black_pixels"]),
                    neutral_reference_pixels=int(values["neutral_reference_pixels"]),
                    neutral_reference_reliable=int(values["neutral_reference_reliable"]),
                    white_ratio=float(values["white_ratio"]),
                    black_ratio=float(values["black_ratio"]),
                    red_ratio=float(values["red_ratio"]),
                    raw_red_ratio=float(values["raw_red_ratio"]),
                    rejected_red_ratio=float(values["rejected_red_ratio"]),
                    largest_raw_red_component_ratio=float(
                        values["largest_raw_red_component_ratio"]
                    ),
                    red_field_rejected=int(values["red_field_rejected"]),
                    rejected_red_component_count=int(
                        values["rejected_red_component_count"]
                    ),
                    sauvola_dark_ratio=float(values["sauvola_dark_ratio"]),
                    chromatic_black_ratio=float(values["chromatic_black_ratio"]),
                    neutral_reference_ratio=float(values["neutral_reference_ratio"]),
                    neutral_reference_a=float(values["neutral_reference_a"]),
                    neutral_reference_b=float(values["neutral_reference_b"]),
                    neutral_reference_spread=float(values["neutral_reference_spread"]),
                    red_detection_mode=str(values["red_detection_mode"]),
                    source_image_path=str(row["source_image_path"]),
                    source_image_id=(
                        None
                        if row["source_image_id"] is None
                        else str(row["source_image_id"])
                    ),
                    brightness=(
                        None if row["brightness"] is None else str(row["brightness"])
                    ),
                    shadow=None if row["shadow"] is None else str(row["shadow"]),
                )
            )
            if index % 500 == 0 or index == total:
                print(f"[sauvola] processed {index}/{total} crops")

        write_metrics_csv(output_directory / "metrics.csv", metrics)
        metric_by_crop_id = {metric.crop_id: metric for metric in metrics}
        representatives = choose_representatives(metrics)
        save_png(
            make_contact_sheet(
                connection,
                representatives=representatives,
                metric_by_crop_id=metric_by_crop_id,
                parameters=parameters,
            ),
            output_directory / "contact_sheet.png",
        )

        manual_high_black = sorted(
            (metric for metric in metrics if metric.source == "manual"),
            key=lambda metric: (metric.black_ratio, metric.red_ratio),
            reverse=True,
        )[:diagnostic_count]
        save_png(
            make_diagnostic_sheet(
                connection,
                metrics=manual_high_black,
                parameters=parameters,
                title="Manual crops with highest final black ratio",
            ),
            output_directory / "manual_highest_black_ratio.png",
        )

        red_labels = {"red", "red5m", "red5p", "red5s"}
        weakest_red = sorted(
            (metric for metric in metrics if metric.tile_label in red_labels),
            key=lambda metric: (metric.red_ratio, -metric.black_ratio, metric.crop_id),
        )[:diagnostic_count]
        save_png(
            make_diagnostic_sheet(
                connection,
                metrics=weakest_red,
                parameters=parameters,
                title="Red-labelled crops with lowest final red ratio",
            ),
            output_directory / "red_labels_lowest_red_ratio.png",
        )

        strongest_false_red = sorted(
            (metric for metric in metrics if metric.tile_label not in red_labels),
            key=lambda metric: (metric.red_ratio, metric.black_ratio, metric.crop_id),
            reverse=True,
        )[:diagnostic_count]
        save_png(
            make_diagnostic_sheet(
                connection,
                metrics=strongest_false_red,
                parameters=parameters,
                title="Non-red crops with highest final red ratio",
            ),
            output_directory / "non_red_labels_highest_red_ratio.png",
        )

        strongest_surface_rejections = sorted(
            metrics,
            key=lambda metric: (
                metric.rejected_red_ratio,
                metric.raw_red_ratio,
                metric.crop_id,
            ),
            reverse=True,
        )[:diagnostic_count]
        save_png(
            make_diagnostic_sheet(
                connection,
                metrics=strongest_surface_rejections,
                parameters=parameters,
                title="Crops with most red rejected as surface/background",
            ),
            output_directory / "red_surface_rejections.png",
        )

    aggregates = aggregate_metrics(metrics)
    summary = {
        "status": "completed",
        "sample_database": str(sample_database.resolve()),
        "output_directory": str(output_directory.resolve()),
        "parameters": asdict(parameters),
        "crop_count": len(metrics),
        "aggregates_by_source": aggregates["by_source"],
        "aggregates_by_source_and_label": aggregates["by_source_and_label"],
        "outputs": {
            "metrics_csv": str(output_directory / "metrics.csv"),
            "contact_sheet": str(output_directory / "contact_sheet.png"),
            "manual_highest_black_ratio": str(
                output_directory / "manual_highest_black_ratio.png"
            ),
            "red_labels_lowest_red_ratio": str(
                output_directory / "red_labels_lowest_red_ratio.png"
            ),
            "non_red_labels_highest_red_ratio": str(
                output_directory / "non_red_labels_highest_red_ratio.png"
            ),
            "red_surface_rejections": str(
                output_directory / "red_surface_rejections.png"
            ),
        },
    }
    atomic_write_json(output_directory / "summary.json", summary)
    return summary


def validate_parameters(parameters: SauvolaParameters) -> None:
    if parameters.window_size < 3 or parameters.window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    numeric_values = asdict(parameters)
    if not all(math.isfinite(float(value)) for value in numeric_values.values()):
        raise ValueError("All parameters must be finite")
    if parameters.dynamic_range <= 0.0:
        raise ValueError("dynamic_range must be positive")
    for name in (
        "red_share_min",
        "red_dominance_min",
        "colorfulness_min",
        "min_channel_intensity",
        "neutral_lightness_quantile",
        "neutral_chroma_quantile",
        "neutral_min_fraction",
        "relative_red_dominance_min",
        "red_field_max_fraction",
        "red_component_max_fraction",
    ):
        value = float(getattr(parameters, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within 0..1")
    if parameters.neutral_max_spread <= 0.0:
        raise ValueError("neutral_max_spread must be positive")
    if parameters.relative_a_red_min < 0.0:
        raise ValueError("relative_a_red_min must be non-negative")
    if parameters.relative_red_chroma_min < 0.0:
        raise ValueError("relative_red_chroma_min must be non-negative")
    if not 0 <= parameters.red_component_max_border_sides <= 4:
        raise ValueError("red_component_max_border_sides must be within 0..4")


def threshold_image(
    image: Image.Image,
    parameters: SauvolaParameters,
) -> ThresholdResult:
    rgb_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = rgb_to_lab(rgb_u8)
    lightness = lab[..., 0]
    sauvola_threshold = sauvola_threshold_map(
        lightness,
        window_size=parameters.window_size,
        k=parameters.k,
        dynamic_range=parameters.dynamic_range,
    )
    sauvola_dark_mask = lightness <= sauvola_threshold

    rgb = rgb_u8.astype(np.float32) / np.float32(255.0)
    channel_sum = np.sum(rgb, axis=-1)
    safe_sum = np.maximum(channel_sum, np.float32(1e-6))
    shares = rgb / safe_sum[..., None]
    red_share = shares[..., 0]
    other_share = np.maximum(shares[..., 1], shares[..., 2])
    red_dominance = red_share - other_share
    maximum_channel = np.max(rgb, axis=-1)
    colorfulness = np.max(shares, axis=-1) - np.min(shares, axis=-1)
    visible_color = maximum_channel >= parameters.min_channel_intensity

    absolute_red_mask = (
        visible_color
        & (red_share >= parameters.red_share_min)
        & (red_dominance >= parameters.red_dominance_min)
    )
    (
        neutral_reference_mask,
        neutral_reference_a,
        neutral_reference_b,
        neutral_reference_spread,
        neutral_reference_reliable,
    ) = estimate_neutral_reference(
        lab,
        sauvola_dark_mask=sauvola_dark_mask,
        lightness_quantile=parameters.neutral_lightness_quantile,
        chroma_quantile=parameters.neutral_chroma_quantile,
        minimum_fraction=parameters.neutral_min_fraction,
        maximum_spread=parameters.neutral_max_spread,
    )

    delta_a = lab[..., 1] - np.float32(neutral_reference_a)
    delta_b = lab[..., 2] - np.float32(neutral_reference_b)
    relative_chroma = np.hypot(delta_a, delta_b)
    relative_red_mask = (
        visible_color
        & (delta_a >= parameters.relative_a_red_min)
        & (relative_chroma >= parameters.relative_red_chroma_min)
        & (red_dominance >= parameters.relative_red_dominance_min)
    )
    if neutral_reference_reliable:
        raw_red_mask = relative_red_mask
        red_detection_mode = "relative_lab"
    else:
        raw_red_mask = absolute_red_mask
        red_detection_mode = "absolute_rgb_fallback"

    (
        red_mask,
        rejected_red_mask,
        largest_raw_red_component_ratio,
        rejected_red_component_count,
        red_field_rejected,
    ) = filter_red_ink_components(
        raw_red_mask,
        field_max_fraction=parameters.red_field_max_fraction,
        component_max_fraction=parameters.red_component_max_fraction,
        component_max_border_sides=parameters.red_component_max_border_sides,
    )

    chromatic_black_mask = rejected_red_mask | (
        visible_color
        & (~red_mask)
        & (colorfulness >= parameters.colorfulness_min)
    )
    black_mask = (~red_mask) & (sauvola_dark_mask | chromatic_black_mask)
    white_mask = ~(red_mask | black_mask)

    ternary = np.full(rgb_u8.shape, 255, dtype=np.uint8)
    ternary[black_mask] = (0, 0, 0)
    ternary[red_mask] = (255, 0, 0)

    lightness_u8 = np.clip(np.rint(lightness * 2.55), 0, 255).astype(np.uint8)
    sauvola_visual = np.full(lightness.shape, 255, dtype=np.uint8)
    sauvola_visual[sauvola_dark_mask] = 0
    neutral_visual = np.zeros(lightness.shape, dtype=np.uint8)
    neutral_visual[neutral_reference_mask] = 255
    raw_red_visual = np.full(rgb_u8.shape, 255, dtype=np.uint8)
    raw_red_visual[raw_red_mask] = (255, 0, 0)
    red_visual = np.full(rgb_u8.shape, 255, dtype=np.uint8)
    red_visual[red_mask] = (255, 0, 0)

    pixel_count = int(white_mask.size)
    white_pixels = int(np.count_nonzero(white_mask))
    black_pixels = int(np.count_nonzero(black_mask))
    red_pixels = int(np.count_nonzero(red_mask))
    raw_red_pixels = int(np.count_nonzero(raw_red_mask))
    rejected_red_pixels = int(np.count_nonzero(rejected_red_mask))
    sauvola_dark_pixels = int(np.count_nonzero(sauvola_dark_mask))
    chromatic_black_pixels = int(np.count_nonzero(chromatic_black_mask))
    neutral_reference_pixels = int(np.count_nonzero(neutral_reference_mask))
    metrics: dict[str, int | float | str] = {
        "pixel_count": pixel_count,
        "white_pixels": white_pixels,
        "black_pixels": black_pixels,
        "red_pixels": red_pixels,
        "raw_red_pixels": raw_red_pixels,
        "rejected_red_pixels": rejected_red_pixels,
        "sauvola_dark_pixels": sauvola_dark_pixels,
        "chromatic_black_pixels": chromatic_black_pixels,
        "neutral_reference_pixels": neutral_reference_pixels,
        "neutral_reference_reliable": int(neutral_reference_reliable),
        "white_ratio": white_pixels / pixel_count,
        "black_ratio": black_pixels / pixel_count,
        "red_ratio": red_pixels / pixel_count,
        "raw_red_ratio": raw_red_pixels / pixel_count,
        "rejected_red_ratio": rejected_red_pixels / pixel_count,
        "largest_raw_red_component_ratio": largest_raw_red_component_ratio,
        "red_field_rejected": int(red_field_rejected),
        "rejected_red_component_count": rejected_red_component_count,
        "sauvola_dark_ratio": sauvola_dark_pixels / pixel_count,
        "chromatic_black_ratio": chromatic_black_pixels / pixel_count,
        "neutral_reference_ratio": neutral_reference_pixels / pixel_count,
        "neutral_reference_a": neutral_reference_a,
        "neutral_reference_b": neutral_reference_b,
        "neutral_reference_spread": neutral_reference_spread,
        "red_detection_mode": red_detection_mode,
    }
    return ThresholdResult(
        lightness_image=Image.fromarray(lightness_u8),
        sauvola_mask_image=Image.fromarray(sauvola_visual),
        neutral_reference_mask_image=Image.fromarray(neutral_visual),
        raw_red_mask_image=Image.fromarray(raw_red_visual),
        red_mask_image=Image.fromarray(red_visual),
        ternary_image=Image.fromarray(ternary),
        metrics=metrics,
    )


def filter_red_ink_components(
    raw_red_mask: np.ndarray,
    *,
    field_max_fraction: float,
    component_max_fraction: float,
    component_max_border_sides: int,
) -> tuple[np.ndarray, np.ndarray, float, int, bool]:
    if raw_red_mask.ndim != 2:
        raise ValueError(f"Expected a 2D red mask, found {raw_red_mask.shape}")

    height, width = raw_red_mask.shape
    pixel_count = int(raw_red_mask.size)
    accepted = np.zeros(raw_red_mask.shape, dtype=bool)
    rejected = np.zeros(raw_red_mask.shape, dtype=bool)
    raw_red_pixels = int(np.count_nonzero(raw_red_mask))
    if raw_red_pixels == 0:
        return accepted, rejected, 0.0, 0, False

    visited = np.zeros(raw_red_mask.shape, dtype=bool)
    components: list[tuple[list[tuple[int, int]], int]] = []
    largest_component_ratio = 0.0

    for start_y, start_x in np.argwhere(raw_red_mask):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x]:
            continue

        stack = [(y, x)]
        visited[y, x] = True
        pixels: list[tuple[int, int]] = []
        touched_borders: set[str] = set()
        while stack:
            current_y, current_x = stack.pop()
            pixels.append((current_y, current_x))
            if current_y == 0:
                touched_borders.add("top")
            if current_y == height - 1:
                touched_borders.add("bottom")
            if current_x == 0:
                touched_borders.add("left")
            if current_x == width - 1:
                touched_borders.add("right")

            for next_y, next_x in (
                (current_y - 1, current_x),
                (current_y + 1, current_x),
                (current_y, current_x - 1),
                (current_y, current_x + 1),
            ):
                if not (0 <= next_y < height and 0 <= next_x < width):
                    continue
                if visited[next_y, next_x] or not raw_red_mask[next_y, next_x]:
                    continue
                visited[next_y, next_x] = True
                stack.append((next_y, next_x))

        component_ratio = len(pixels) / pixel_count
        largest_component_ratio = max(largest_component_ratio, component_ratio)
        components.append((pixels, len(touched_borders)))

    field_rejected = raw_red_pixels / pixel_count > field_max_fraction
    rejected_component_count = 0
    for pixels, border_side_count in components:
        component_ratio = len(pixels) / pixel_count
        reject_component = (
            field_rejected
            or component_ratio > component_max_fraction
            or border_side_count > component_max_border_sides
        )
        target = rejected if reject_component else accepted
        if reject_component:
            rejected_component_count += 1
        for y, x in pixels:
            target[y, x] = True

    return (
        accepted,
        rejected,
        largest_component_ratio,
        rejected_component_count,
        field_rejected,
    )


def estimate_neutral_reference(
    lab: np.ndarray,
    *,
    sauvola_dark_mask: np.ndarray,
    lightness_quantile: float,
    chroma_quantile: float,
    minimum_fraction: float,
    maximum_spread: float,
) -> tuple[np.ndarray, float, float, float, bool]:
    if lab.ndim != 3 or lab.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 Lab array, found {lab.shape}")
    if sauvola_dark_mask.shape != lab.shape[:2]:
        raise ValueError("sauvola_dark_mask must match the Lab image dimensions")

    lightness = lab[..., 0]
    a_axis = lab[..., 1]
    b_axis = lab[..., 2]
    chroma = np.hypot(a_axis, b_axis)
    base_mask = ~sauvola_dark_mask
    pixel_count = int(base_mask.size)
    minimum_count = max(8, int(math.ceil(pixel_count * minimum_fraction)))
    empty = np.zeros(base_mask.shape, dtype=bool)

    if int(np.count_nonzero(base_mask)) < minimum_count:
        return empty, 0.0, 0.0, 0.0, False

    lightness_cutoff = float(np.quantile(lightness[base_mask], lightness_quantile))
    bright_mask = base_mask & (lightness >= lightness_cutoff)
    if int(np.count_nonzero(bright_mask)) < minimum_count:
        return bright_mask, 0.0, 0.0, 0.0, False

    chroma_cutoff = float(np.quantile(chroma[bright_mask], chroma_quantile))
    candidate_mask = bright_mask & (chroma <= chroma_cutoff)
    if int(np.count_nonzero(candidate_mask)) < minimum_count:
        return candidate_mask, 0.0, 0.0, 0.0, False

    initial_a = float(np.median(a_axis[candidate_mask]))
    initial_b = float(np.median(b_axis[candidate_mask]))
    distance = np.hypot(a_axis - initial_a, b_axis - initial_b)
    candidate_distance = distance[candidate_mask]
    median_distance = float(np.median(candidate_distance))
    mad = float(np.median(np.abs(candidate_distance - median_distance)))
    robust_cutoff = max(3.0, median_distance + 3.0 * 1.4826 * mad)
    refined_mask = candidate_mask & (distance <= robust_cutoff)
    if int(np.count_nonzero(refined_mask)) < minimum_count:
        return refined_mask, initial_a, initial_b, 0.0, False

    reference_a = float(np.median(a_axis[refined_mask]))
    reference_b = float(np.median(b_axis[refined_mask]))
    refined_distance = np.hypot(
        a_axis[refined_mask] - reference_a,
        b_axis[refined_mask] - reference_b,
    )
    spread = float(np.quantile(refined_distance, 0.90))
    reliable = spread <= maximum_spread
    return refined_mask, reference_a, reference_b, spread, reliable


def sauvola_threshold_map(
    lightness: np.ndarray,
    *,
    window_size: int,
    k: float,
    dynamic_range: float,
) -> np.ndarray:
    if lightness.ndim != 2:
        raise ValueError(f"Expected a 2D lightness array, found {lightness.shape}")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    if dynamic_range <= 0.0:
        raise ValueError("dynamic_range must be positive")

    values = lightness.astype(np.float32, copy=False)
    radius = window_size // 2
    pad_mode = "reflect" if min(values.shape) > 1 else "edge"
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode=pad_mode)
    mean = box_mean(padded, window_size)
    mean_square = box_mean(padded * padded, window_size)
    variance = np.maximum(mean_square - mean * mean, np.float32(0.0))
    standard_deviation = np.sqrt(variance)
    return mean * (
        np.float32(1.0)
        + np.float32(k)
        * (standard_deviation / np.float32(dynamic_range) - np.float32(1.0))
    )


def box_mean(padded: np.ndarray, window_size: int) -> np.ndarray:
    integral = np.pad(
        padded.astype(np.float64, copy=False),
        ((1, 0), (1, 0)),
        mode="constant",
    ).cumsum(axis=0).cumsum(axis=1)
    window_sum = (
        integral[window_size:, window_size:]
        - integral[:-window_size, window_size:]
        - integral[window_size:, :-window_size]
        + integral[:-window_size, :-window_size]
    )
    return (window_sum / float(window_size * window_size)).astype(np.float32)


def choose_representatives(
    metrics: Sequence[ImageMetric],
) -> dict[tuple[str, str], ImageMetric]:
    groups: dict[tuple[str, str], list[ImageMetric]] = defaultdict(list)
    for metric in metrics:
        groups[(metric.source, metric.tile_label)].append(metric)

    representatives: dict[tuple[str, str], ImageMetric] = {}
    for key, group in groups.items():
        ordered = sorted(group, key=lambda item: (item.black_ratio, item.red_ratio, item.crop_id))
        representatives[key] = ordered[(len(ordered) - 1) // 2]
    return representatives


def make_contact_sheet(
    connection: sqlite3.Connection,
    *,
    representatives: dict[tuple[str, str], ImageMetric],
    metric_by_crop_id: dict[str, ImageMetric],
    parameters: SauvolaParameters,
    cell_size: int = 72,
) -> Image.Image:
    available_labels = {
        label for source, label in representatives if source in {"jp", "manual"}
    }
    labels = sorted(available_labels, key=label_sort_key)
    label_width = 76
    cell_gap = 2
    source_gap = 14
    cells_per_source = 7
    source_width = cells_per_source * cell_size + (cells_per_source - 1) * cell_gap
    row_height = cell_size + 22
    header_height = 42
    sheet_width = label_width + 2 * source_width + source_gap + 16
    sheet_height = header_height + len(labels) * row_height + 10
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    stage_header = "original | L* | Sauvola | neutral | raw red | ink red | final"
    jp_x = label_width
    manual_x = label_width + source_width + source_gap
    draw.text((jp_x, 5), "JP median-black representative", fill="black", font=font)
    draw.text((jp_x, 20), stage_header, fill="black", font=font)
    draw.text((manual_x, 5), "manual median-black representative", fill="black", font=font)
    draw.text((manual_x, 20), stage_header, fill="black", font=font)

    for row_index, label in enumerate(labels):
        y = header_height + row_index * row_height
        draw.text((8, y + 4), label, fill="black", font=font)
        for source, start_x in (("jp", jp_x), ("manual", manual_x)):
            metric = representatives.get((source, label))
            if metric is None:
                draw.rectangle(
                    (start_x, y, start_x + source_width - 1, y + cell_size - 1),
                    outline="gray",
                )
                draw.text((start_x + 4, y + 4), "NO SAMPLE", fill="black", font=font)
                continue
            row = connection.execute(
                "SELECT image_png FROM tile_crop WHERE crop_id = ?",
                (metric.crop_id,),
            ).fetchone()
            if row is None:
                continue
            original = decode_png(bytes(row[0]))
            result = threshold_image(original, parameters)
            stages = (
                original,
                result.lightness_image,
                result.sauvola_mask_image,
                result.neutral_reference_mask_image,
                result.raw_red_mask_image,
                result.red_mask_image,
                result.ternary_image,
            )
            for stage_index, stage in enumerate(stages):
                x = start_x + stage_index * (cell_size + cell_gap)
                paste_fitted(sheet, stage, (x, y, cell_size, cell_size))
            selected_metric = metric_by_crop_id[metric.crop_id]
            draw.text(
                (start_x, y + cell_size + 3),
                f"B{selected_metric.black_ratio:.2f} rawR{selected_metric.raw_red_ratio:.2f} "
                f"R{selected_metric.red_ratio:.2f} fieldReject={selected_metric.red_field_rejected}",
                fill="black",
                font=font,
            )
    return sheet


def make_diagnostic_sheet(
    connection: sqlite3.Connection,
    *,
    metrics: Sequence[ImageMetric],
    parameters: SauvolaParameters,
    title: str,
    columns: int = 2,
    cell_size: int = 92,
) -> Image.Image:
    if not metrics:
        image = Image.new("RGB", (640, 80), "white")
        ImageDraw.Draw(image).text((8, 8), f"{title}: no samples", fill="black")
        return image

    item_width = 7 * cell_size + 20
    item_height = cell_size + 40
    rows = math.ceil(len(metrics) / columns)
    sheet = Image.new(
        "RGB",
        (columns * item_width + 12, 30 + rows * item_height + 12),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill="black", font=font)

    for index, metric in enumerate(metrics):
        row_index, column_index = divmod(index, columns)
        x = 8 + column_index * item_width
        y = 30 + row_index * item_height
        row = connection.execute(
            "SELECT image_png FROM tile_crop WHERE crop_id = ?",
            (metric.crop_id,),
        ).fetchone()
        if row is None:
            continue
        original = decode_png(bytes(row[0]))
        result = threshold_image(original, parameters)
        stages = (
            original,
            result.lightness_image,
            result.sauvola_mask_image,
            result.neutral_reference_mask_image,
            result.raw_red_mask_image,
            result.red_mask_image,
            result.ternary_image,
        )
        for stage_index, stage in enumerate(stages):
            paste_fitted(
                sheet,
                stage,
                (x + stage_index * cell_size, y, cell_size, cell_size),
            )
        draw.text(
            (x, y + cell_size + 3),
            f"{metric.source}/{metric.tile_label} B{metric.black_ratio:.2f} "
            f"rawR{metric.raw_red_ratio:.2f} R{metric.red_ratio:.2f} "
            f"fieldReject={metric.red_field_rejected}",
            fill="black",
            font=font,
        )
        draw.text(
            (x, y + cell_size + 17),
            metric.crop_id[:62],
            fill="black",
            font=font,
        )
    return sheet


def aggregate_metrics(metrics: Sequence[ImageMetric]) -> dict[str, Any]:
    by_source: dict[str, list[ImageMetric]] = defaultdict(list)
    by_source_and_label: dict[str, dict[str, list[ImageMetric]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for metric in metrics:
        by_source[metric.source].append(metric)
        by_source_and_label[metric.source][metric.tile_label].append(metric)
    return {
        "by_source": {
            source: aggregate_group(group)
            for source, group in sorted(by_source.items())
        },
        "by_source_and_label": {
            source: {
                label: aggregate_group(group)
                for label, group in sorted(
                    labels.items(), key=lambda item: label_sort_key(item[0])
                )
            }
            for source, labels in sorted(by_source_and_label.items())
        },
    }


def aggregate_group(metrics: Sequence[ImageMetric]) -> dict[str, int | float]:
    pixels = sum(metric.pixel_count for metric in metrics)
    white = sum(metric.white_pixels for metric in metrics)
    black = sum(metric.black_pixels for metric in metrics)
    red = sum(metric.red_pixels for metric in metrics)
    local_dark = sum(metric.sauvola_dark_pixels for metric in metrics)
    chromatic_black = sum(metric.chromatic_black_pixels for metric in metrics)
    reliable_references = sum(metric.neutral_reference_reliable for metric in metrics)
    return {
        "image_count": len(metrics),
        "pixel_count": pixels,
        "white_ratio": white / pixels,
        "black_ratio": black / pixels,
        "red_ratio": red / pixels,
        "sauvola_dark_ratio": local_dark / pixels,
        "chromatic_black_ratio": chromatic_black / pixels,
        "neutral_reference_reliable_ratio": reliable_references / len(metrics),
        "mean_neutral_reference_ratio": sum(
            metric.neutral_reference_ratio for metric in metrics
        ) / len(metrics),
        "mean_neutral_reference_spread": sum(
            metric.neutral_reference_spread for metric in metrics
        ) / len(metrics),
        "mean_image_white_ratio": sum(metric.white_ratio for metric in metrics)
        / len(metrics),
        "mean_image_black_ratio": sum(metric.black_ratio for metric in metrics)
        / len(metrics),
        "mean_image_red_ratio": sum(metric.red_ratio for metric in metrics)
        / len(metrics),
    }


def write_metrics_csv(path: Path, metrics: Sequence[ImageMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(metrics[0]).keys()) if metrics else list(ImageMetric.__annotations__)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for metric in metrics:
                writer.writerow(asdict(metric))
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
