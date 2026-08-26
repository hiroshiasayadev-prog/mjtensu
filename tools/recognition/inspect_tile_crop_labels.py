from __future__ import annotations

import argparse
import io
import sqlite3
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_LABELS = ("east", "south", "west", "north", "white", "green", "red")


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Export representative tile crops from the persistent SQLite crop dataset "
            "as a labeled contact sheet."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "tile_crop_dataset"
        / "dataset.sqlite",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "tile_crop_dataset"
        / "jp_honor_contact_sheet.png",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="Tile labels to inspect. Defaults to east, south, west, north, white, green, red.",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=6,
        help="Number of representative crops shown for each label.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=128,
        help="Square image area size in pixels.",
    )
    return parser.parse_args()


def load_samples(
    database: Path,
    labels: list[str],
    samples_per_label: int,
) -> dict[str, list[tuple[bytes, str]]]:
    if not database.is_file():
        raise FileNotFoundError(f"Crop database not found: {database}")
    if samples_per_label < 1:
        raise ValueError("--samples-per-label must be at least 1")

    samples: dict[str, list[tuple[bytes, str]]] = {}
    with sqlite3.connect(database) as connection:
        for label in labels:
            rows = connection.execute(
                """
                SELECT image_png, source_image_path
                FROM tile_crop
                WHERE source = 'jp' AND tile_label = ?
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (label, samples_per_label),
            ).fetchall()
            samples[label] = [(bytes(image_png), str(source_path)) for image_png, source_path in rows]
    return samples


def make_contact_sheet(
    labels: list[str],
    samples: dict[str, list[tuple[bytes, str]]],
    samples_per_label: int,
    cell_size: int,
) -> Image.Image:
    if cell_size < 32:
        raise ValueError("--cell-size must be at least 32")

    label_height = 28
    footer_height = 18
    column_width = cell_size + 12
    row_height = cell_size + footer_height + 8
    sheet_width = max(1, len(labels)) * column_width + 12
    sheet_height = label_height + samples_per_label * row_height + 12

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for column_index, label in enumerate(labels):
        x = 12 + column_index * column_width
        draw.text((x + 2, 8), label, fill="black", font=font)

        label_samples = samples.get(label, [])
        for row_index in range(samples_per_label):
            y = label_height + row_index * row_height
            image_box = (x, y, x + cell_size, y + cell_size)
            draw.rectangle(image_box, fill=(232, 232, 232), outline=(128, 128, 128), width=1)

            if row_index >= len(label_samples):
                draw.text((x + 6, y + 6), "NO SAMPLE", fill="black", font=font)
                continue

            image_png, source_path = label_samples[row_index]
            with Image.open(io.BytesIO(image_png)) as source_image:
                tile = ImageOps.exif_transpose(source_image).convert("RGB")
                tile.thumbnail((cell_size - 8, cell_size - 8), Image.Resampling.LANCZOS)
                paste_x = x + (cell_size - tile.width) // 2
                paste_y = y + (cell_size - tile.height) // 2
                sheet.paste(tile, (paste_x, paste_y))

            source_name = Path(source_path).name
            if len(source_name) > 18:
                source_name = source_name[:15] + "..."
            draw.text(
                (x + 2, y + cell_size + 3),
                source_name,
                fill="black",
                font=font,
            )

    return sheet


def main() -> None:
    args = parse_args()
    labels = list(dict.fromkeys(args.labels))
    samples = load_samples(args.database, labels, args.samples_per_label)

    missing = [label for label in labels if not samples[label]]
    for label in labels:
        print(f"{label}: {len(samples[label])} sample(s)")

    sheet = make_contact_sheet(
        labels,
        samples,
        args.samples_per_label,
        args.cell_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, format="PNG", compress_level=1)
    print(f"output: {args.output}")

    if missing:
        raise SystemExit(f"No crops found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
