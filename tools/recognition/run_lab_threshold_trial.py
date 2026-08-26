from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "NumPy is required. Run this with the repository .venv used by the "
        "recognition pipeline."
    ) from error

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_L_DARK = 50.0
DEFAULT_A_RED = 18.0
DEFAULT_A_GREEN = -18.0
DEFAULT_CHROMA_MIN = 15.0
DEFAULT_SEED = 42

LABEL_ORDER = tuple(
    label
    for suit in "mps"
    for label in (
        f"1{suit}",
        f"2{suit}",
        f"3{suit}",
        f"4{suit}",
        f"5{suit}",
        f"red5{suit}",
        f"6{suit}",
        f"7{suit}",
        f"8{suit}",
        f"9{suit}",
    )
) + ("east", "south", "west", "north", "white", "green", "red")


@dataclass(frozen=True)
class Thresholds:
    l_dark: float = DEFAULT_L_DARK
    a_red: float = DEFAULT_A_RED
    a_green: float = DEFAULT_A_GREEN
    chroma_min: float = DEFAULT_CHROMA_MIN


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
    white_ratio: float
    black_ratio: float
    red_ratio: float
    mean_l: float
    mean_a: float
    mean_b: float
    source_image_path: str
    source_image_id: str | None
    brightness: str | None
    shadow: str | None


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed CIELAB white/black/red threshold baseline against the compact "
            "color-trial sample database."
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
            "color_trials/lab_fixed_seed42."
        ),
    )
    parser.add_argument("--l-dark", type=float, default=DEFAULT_L_DARK)
    parser.add_argument("--a-red", type=float, default=DEFAULT_A_RED)
    parser.add_argument("--a-green", type=float, default=DEFAULT_A_GREEN)
    parser.add_argument("--chroma-min", type=float, default=DEFAULT_CHROMA_MIN)
    parser.add_argument(
        "--diagnostic-count",
        type=int,
        default=24,
        help="Number of image pairs in each diagnostic contact sheet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.diagnostic_count < 1:
        raise ValueError("--diagnostic-count must be positive")

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
        / f"lab_fixed_seed{DEFAULT_SEED}"
    )
    thresholds = Thresholds(
        l_dark=float(args.l_dark),
        a_red=float(args.a_red),
        a_green=float(args.a_green),
        chroma_min=float(args.chroma_min),
    )

    summary = run_trial(
        sample_database=sample_database,
        output_directory=output_directory,
        thresholds=thresholds,
        diagnostic_count=int(args.diagnostic_count),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_trial(
    *,
    sample_database: Path,
    output_directory: Path,
    thresholds: Thresholds,
    diagnostic_count: int = 24,
) -> dict[str, Any]:
    if not sample_database.is_file():
        raise FileNotFoundError(sample_database)
    if diagnostic_count < 1:
        raise ValueError("diagnostic_count must be positive")
    validate_thresholds(thresholds)

    output_directory.mkdir(parents=True, exist_ok=True)
    metrics: list[ImageMetric] = []
    with closing(
        sqlite3.connect(sqlite_readonly_uri(sample_database), uri=True)
    ) as connection:
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
            _, metric_values = threshold_image(image, thresholds)
            metrics.append(
                ImageMetric(
                    crop_id=str(row["crop_id"]),
                    source=str(row["source"]),
                    source_partition=str(row["source_partition"]),
                    tile_label=str(row["tile_label"]),
                    image_width=image.width,
                    image_height=image.height,
                    pixel_count=metric_values["pixel_count"],
                    white_pixels=metric_values["white_pixels"],
                    black_pixels=metric_values["black_pixels"],
                    red_pixels=metric_values["red_pixels"],
                    white_ratio=metric_values["white_ratio"],
                    black_ratio=metric_values["black_ratio"],
                    red_ratio=metric_values["red_ratio"],
                    mean_l=metric_values["mean_l"],
                    mean_a=metric_values["mean_a"],
                    mean_b=metric_values["mean_b"],
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
                print(f"[lab] processed {index}/{total} crops")

        write_metrics_csv(output_directory / "metrics.csv", metrics)
        metric_by_crop_id = {metric.crop_id: metric for metric in metrics}

        contact_sheet = make_label_contact_sheet(
            connection,
            thresholds=thresholds,
            metric_by_crop_id=metric_by_crop_id,
        )
        save_png(contact_sheet, output_directory / "contact_sheet.png")

        highest_red = sorted(
            (
                metric
                for metric in metrics
                if metric.tile_label not in {"red", "red5m", "red5p", "red5s"}
            ),
            key=lambda metric: (metric.red_ratio, metric.black_ratio),
            reverse=True,
        )[:diagnostic_count]
        save_png(
            make_diagnostic_sheet(
                connection,
                highest_red,
                thresholds=thresholds,
                title="Highest red ratio (excluding red and red-five labels)",
            ),
            output_directory / "highest_red_ratio_non_red.png",
        )

        lowest_black = sorted(
            (metric for metric in metrics if metric.tile_label != "white"),
            key=lambda metric: (metric.black_ratio, -metric.white_ratio),
        )[:diagnostic_count]
        save_png(
            make_diagnostic_sheet(
                connection,
                lowest_black,
                thresholds=thresholds,
                title="Lowest black ratio (excluding white tiles)",
            ),
            output_directory / "lowest_black_ratio_non_white.png",
        )

        manual_most_black = sorted(
            (metric for metric in metrics if metric.source == "manual"),
            key=lambda metric: (metric.black_ratio, metric.red_ratio),
            reverse=True,
        )[:diagnostic_count]
        save_png(
            make_diagnostic_sheet(
                connection,
                manual_most_black,
                thresholds=thresholds,
                title="Manual crops with highest black ratio",
            ),
            output_directory / "manual_highest_black_ratio.png",
        )

    aggregates = aggregate_metrics(metrics)
    summary = {
        "status": "completed",
        "sample_database": str(sample_database.resolve()),
        "output_directory": str(output_directory.resolve()),
        "thresholds": asdict(thresholds),
        "crop_count": len(metrics),
        "aggregates_by_source": aggregates["by_source"],
        "aggregates_by_source_and_label": aggregates["by_source_and_label"],
        "outputs": {
            "metrics_csv": str(output_directory / "metrics.csv"),
            "contact_sheet": str(output_directory / "contact_sheet.png"),
            "highest_red_ratio_non_red": str(
                output_directory / "highest_red_ratio_non_red.png"
            ),
            "lowest_black_ratio_non_white": str(
                output_directory / "lowest_black_ratio_non_white.png"
            ),
            "manual_highest_black_ratio": str(
                output_directory / "manual_highest_black_ratio.png"
            ),
        },
    }
    atomic_write_json(output_directory / "summary.json", summary)
    return summary


def validate_thresholds(thresholds: Thresholds) -> None:
    values = asdict(thresholds)
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("All thresholds must be finite")
    if not 0.0 <= thresholds.l_dark <= 100.0:
        raise ValueError("l_dark must be within CIELAB L* range 0..100")
    if thresholds.chroma_min < 0.0:
        raise ValueError("chroma_min must be non-negative")


def decode_png(image_png: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_png)) as source:
        source.load()
        return ImageOps.exif_transpose(source).convert("RGB")


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim < 1 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB array with final dimension 3, found {rgb.shape}")

    srgb = rgb.astype(np.float32) / np.float32(255.0)
    linear = np.where(
        srgb <= np.float32(0.04045),
        srgb / np.float32(12.92),
        ((srgb + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4),
    )

    red = linear[..., 0]
    green = linear[..., 1]
    blue = linear[..., 2]
    x = red * 0.4124564 + green * 0.3575761 + blue * 0.1804375
    y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750
    z = red * 0.0193339 + green * 0.1191920 + blue * 0.9503041

    x = x / np.float32(0.95047)
    z = z / np.float32(1.08883)
    epsilon = np.float32(216.0 / 24389.0)
    kappa = np.float32(24389.0 / 27.0)

    def lab_f(value: np.ndarray) -> np.ndarray:
        return np.where(
            value > epsilon,
            np.cbrt(value),
            (kappa * value + np.float32(16.0)) / np.float32(116.0),
        )

    fx = lab_f(x)
    fy = lab_f(y)
    fz = lab_f(z)
    lab = np.empty(rgb.shape, dtype=np.float32)
    lab[..., 0] = np.float32(116.0) * fy - np.float32(16.0)
    lab[..., 1] = np.float32(500.0) * (fx - fy)
    lab[..., 2] = np.float32(200.0) * (fy - fz)
    return lab


def classify_lab(
    lab: np.ndarray,
    thresholds: Thresholds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lab.ndim < 1 or lab.shape[-1] != 3:
        raise ValueError(f"Expected Lab array with final dimension 3, found {lab.shape}")
    lightness = lab[..., 0]
    a_axis = lab[..., 1]
    b_axis = lab[..., 2]
    chroma = np.hypot(a_axis, b_axis)

    red_mask = (a_axis >= thresholds.a_red) & (chroma >= thresholds.chroma_min)
    green_mask = (a_axis <= thresholds.a_green) & (
        chroma >= thresholds.chroma_min
    )
    black_mask = (~red_mask) & ((lightness <= thresholds.l_dark) | green_mask)
    white_mask = ~(red_mask | black_mask)
    return white_mask, black_mask, red_mask


def threshold_image(
    image: Image.Image,
    thresholds: Thresholds,
) -> tuple[Image.Image, dict[str, int | float]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = rgb_to_lab(rgb)
    white_mask, black_mask, red_mask = classify_lab(lab, thresholds)

    result = np.full(rgb.shape, 255, dtype=np.uint8)
    result[black_mask] = (0, 0, 0)
    result[red_mask] = (255, 0, 0)

    pixel_count = int(white_mask.size)
    white_pixels = int(np.count_nonzero(white_mask))
    black_pixels = int(np.count_nonzero(black_mask))
    red_pixels = int(np.count_nonzero(red_mask))
    metrics: dict[str, int | float] = {
        "pixel_count": pixel_count,
        "white_pixels": white_pixels,
        "black_pixels": black_pixels,
        "red_pixels": red_pixels,
        "white_ratio": white_pixels / pixel_count,
        "black_ratio": black_pixels / pixel_count,
        "red_ratio": red_pixels / pixel_count,
        "mean_l": float(np.mean(lab[..., 0])),
        "mean_a": float(np.mean(lab[..., 1])),
        "mean_b": float(np.mean(lab[..., 2])),
    }
    return Image.fromarray(result, mode="RGB"), metrics


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
                for label, group in sorted(labels.items(), key=lambda item: label_sort_key(item[0]))
            }
            for source, labels in sorted(by_source_and_label.items())
        },
    }


def aggregate_group(metrics: Sequence[ImageMetric]) -> dict[str, int | float]:
    pixels = sum(metric.pixel_count for metric in metrics)
    white = sum(metric.white_pixels for metric in metrics)
    black = sum(metric.black_pixels for metric in metrics)
    red = sum(metric.red_pixels for metric in metrics)
    return {
        "image_count": len(metrics),
        "pixel_count": pixels,
        "white_ratio": white / pixels,
        "black_ratio": black / pixels,
        "red_ratio": red / pixels,
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


def make_label_contact_sheet(
    connection: sqlite3.Connection,
    *,
    thresholds: Thresholds,
    metric_by_crop_id: dict[str, ImageMetric],
    cell_size: int = 88,
) -> Image.Image:
    available_labels = {
        str(row[0])
        for row in connection.execute("SELECT DISTINCT tile_label FROM tile_crop")
    }
    labels = sorted(available_labels, key=label_sort_key)
    row_height = cell_size + 28
    label_width = 76
    column_gap = 8
    sheet_width = label_width + 4 * cell_size + 3 * column_gap + 20
    sheet_height = 34 + len(labels) * row_height + 10
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((label_width + 5, 8), "JP original / LAB       manual original / LAB", fill="black", font=font)

    for row_index, label in enumerate(labels):
        y = 34 + row_index * row_height
        draw.text((8, y + 4), label, fill="black", font=font)
        x = label_width
        for source in ("jp", "manual"):
            row = connection.execute(
                """
                SELECT crop_id, image_png
                FROM tile_crop
                WHERE source = ? AND tile_label = ?
                ORDER BY crop_id
                LIMIT 1
                """,
                (source, label),
            ).fetchone()
            if row is None:
                draw.rectangle((x, y, x + cell_size - 1, y + cell_size - 1), outline="gray")
                draw.text((x + 4, y + 4), "NO SAMPLE", fill="black", font=font)
                x += 2 * cell_size + column_gap
                continue
            original = decode_png(bytes(row["image_png"]))
            ternary, _ = threshold_image(original, thresholds)
            paste_fitted(sheet, original, (x, y, cell_size, cell_size))
            paste_fitted(sheet, ternary, (x + cell_size, y, cell_size, cell_size))
            metric = metric_by_crop_id[str(row["crop_id"])]
            draw.text(
                (x, y + cell_size + 3),
                f"W{metric.white_ratio:.2f} B{metric.black_ratio:.2f} R{metric.red_ratio:.2f}",
                fill="black",
                font=font,
            )
            x += 2 * cell_size + column_gap
    return sheet


def make_diagnostic_sheet(
    connection: sqlite3.Connection,
    metrics: Sequence[ImageMetric],
    *,
    thresholds: Thresholds,
    title: str,
    columns: int = 4,
    cell_size: int = 104,
) -> Image.Image:
    if not metrics:
        image = Image.new("RGB", (640, 80), "white")
        ImageDraw.Draw(image).text((8, 8), f"{title}: no samples", fill="black")
        return image

    item_width = 2 * cell_size + 14
    item_height = cell_size + 42
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
        ternary, _ = threshold_image(original, thresholds)
        paste_fitted(sheet, original, (x, y, cell_size, cell_size))
        paste_fitted(sheet, ternary, (x + cell_size, y, cell_size, cell_size))
        draw.text(
            (x, y + cell_size + 3),
            f"{metric.source}/{metric.tile_label} W{metric.white_ratio:.2f} "
            f"B{metric.black_ratio:.2f} R{metric.red_ratio:.2f}",
            fill="black",
            font=font,
        )
        draw.text(
            (x, y + cell_size + 16),
            metric.crop_id[:34],
            fill="black",
            font=font,
        )
    return sheet


def paste_fitted(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    x, y, width, height = box
    canvas.paste((232, 232, 232), (x, y, x + width, y + height))
    fitted = image.copy().convert("RGB")
    fitted.thumbnail((width - 4, height - 4), Image.Resampling.LANCZOS)
    paste_x = x + (width - fitted.width) // 2
    paste_y = y + (height - fitted.height) // 2
    canvas.paste(fitted, (paste_x, paste_y))


def label_sort_key(label: str) -> tuple[int, str]:
    try:
        return (LABEL_ORDER.index(label), label)
    except ValueError:
        return (len(LABEL_ORDER), label)


def sqlite_readonly_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=1)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
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
