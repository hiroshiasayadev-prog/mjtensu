from __future__ import annotations

"""Render real INV-013 classifier crops through the exact training augmentation path.

The gallery reads actual 64x64 grayscale samples from the frozen classifier SQLite
corpus, applies the same sample/epoch deterministic random360 + INV-013 shared geometry
stream used by training, and writes one self-contained HTML file with embedded PNGs.

No synthetic tile drawing is involved: every source image is image_gray_u8 from the
selected database row.
"""

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import sqlite3
import struct
from typing import Any, Sequence
import zlib

import numpy as np
import torch

try:
    from perspective_classifier_augmentation import (
        AUGMENTATION_SPECS,
        GEOMETRY_UNIT_COUNT,
        PerspectiveAugmentationSpec,
        apply_training_geometry,
        deterministic_geometry_units,
    )
    from run_rotation_classifier_experiment import deterministic_random360_angles
except ModuleNotFoundError:  # package-style import used by tests
    from tools.recognition.perspective_classifier_augmentation import (
        AUGMENTATION_SPECS,
        GEOMETRY_UNIT_COUNT,
        PerspectiveAugmentationSpec,
        apply_training_geometry,
        deterministic_geometry_units,
    )
    from tools.recognition.run_rotation_classifier_experiment import (
        deterministic_random360_angles,
    )


DEFAULT_SEED = 42
DEFAULT_EPOCH = 73
DEFAULT_SPLIT = "manual_val"
DEFAULT_LABELS = ("5m", "6m", "7m")
DEFAULT_SAMPLES_PER_LABEL = 6
GEOMETRY_STREAM = "inv013-shared-geometry"
DISPLAY_AUGMENTATIONS = (
    "a0-random360",
    "a1-anisotropic-affine",
    "a2-perspective",
    "a3-perspective-recrop",
)


class Sample:
    def __init__(
        self,
        sample_id: str,
        class_index: int,
        label: str,
        image_u8: np.ndarray,
        original_width: int,
        original_height: int,
    ):
        self.sample_id = sample_id
        self.class_index = class_index
        self.label = label
        self.image_u8 = image_u8
        self.original_width = original_width
        self.original_height = original_height


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Render actual classifier crops through INV-013 A0/A1/A2/A3 into HTML."
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--database",
        type=Path,
        default=(
            repository_root
            / ".local"
            / "recognition"
            / "tile_classifier_datasets"
            / "gray35_jp500_seed42_v3_jp189.sqlite"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository_root
            / ".local"
            / "recognition"
            / "perspective_classifier_experiment"
            / "augmentation-gallery.html"
        ),
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--samples-per-label", type=int, default=DEFAULT_SAMPLES_PER_LABEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--epoch",
        type=int,
        default=DEFAULT_EPOCH,
        help=(
            "Training epoch whose deterministic geometry stream should be visualized. "
            "Changing this gives another real set of random A1/A2/A3 examples."
        ),
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        help="Optional exact sample IDs. When supplied, --labels/--samples-per-label are ignored.",
    )
    parser.add_argument(
        "--augmentations",
        nargs="+",
        choices=list(DISPLAY_AUGMENTATIONS),
        default=list(DISPLAY_AUGMENTATIONS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_label < 1:
        raise ValueError("--samples-per-label must be positive")
    if args.epoch < 1:
        raise ValueError("--epoch must be >= 1")

    database = args.database.resolve()
    output = args.output.resolve()
    samples, class_labels, image_size = load_samples(
        database,
        split=str(args.split),
        labels=tuple(str(value) for value in args.labels),
        samples_per_label=int(args.samples_per_label),
        seed=int(args.seed),
        sample_ids=(tuple(str(value) for value in args.sample_ids) if args.sample_ids else None),
    )
    if image_size != 64:
        raise ValueError(f"INV-013 gallery expects gray64 source crops; found image_size={image_size}")

    sample_ids = [sample.sample_id for sample in samples]
    angles_np = deterministic_random360_angles(
        sample_ids,
        seed=int(args.seed),
        epoch=int(args.epoch),
    )
    units_np = deterministic_geometry_units(
        sample_ids,
        seed=int(args.seed),
        epoch=int(args.epoch),
        stream=GEOMETRY_STREAM,
    )

    source_u8 = np.stack([sample.image_u8 for sample in samples], axis=0)
    source = torch.from_numpy(source_u8).float().unsqueeze(1).mul_(1.0 / 255.0)
    angles = torch.from_numpy(angles_np).to(dtype=torch.float32)
    units = torch.from_numpy(units_np).to(dtype=torch.float32)
    content_extents = np.asarray(
        [
            preletterbox_content_extent(
                sample.original_width,
                sample.original_height,
                image_size=image_size,
            )
            for sample in samples
        ],
        dtype=np.float32,
    )
    content_extent_x = torch.from_numpy(content_extents[:, 0])
    content_extent_y = torch.from_numpy(content_extents[:, 1])

    rendered: dict[str, np.ndarray] = {"original": source_u8.copy()}
    for augmentation in args.augmentations:
        spec = AUGMENTATION_SPECS[augmentation]
        with torch.inference_mode():
            transformed = apply_training_geometry(
                source,
                spec=spec,
                angles_deg=angles,
                units=units,
                content_extent_x=(
                    content_extent_x if augmentation == "a3-perspective-recrop" else None
                ),
                content_extent_y=(
                    content_extent_y if augmentation == "a3-perspective-recrop" else None
                ),
            )
        rendered[augmentation] = tensor_to_u8(transformed)

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        parameter_rows = {
            augmentation: geometry_parameter_summary(
                AUGMENTATION_SPECS[augmentation],
                angle_deg=float(angles_np[index]),
                units=units_np[index],
            )
            for augmentation in args.augmentations
        }
        rows.append(
            {
                "sample": sample,
                "images": {
                    key: png_data_uri(value[index])
                    for key, value in rendered.items()
                },
                "parameters": parameter_rows,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_html(
            rows,
            database=database,
            split=str(args.split),
            seed=int(args.seed),
            epoch=int(args.epoch),
            class_labels=class_labels,
            augmentations=tuple(str(value) for value in args.augmentations),
        ),
        encoding="utf-8",
    )
    print(f"samples={len(samples)}")
    print(f"database={database}")
    print(f"output={output}")


def load_samples(
    database: Path,
    *,
    split: str,
    labels: Sequence[str],
    samples_per_label: int,
    seed: int,
    sample_ids: Sequence[str] | None,
) -> tuple[list[Sample], tuple[str, ...], int]:
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM experiment_metadata")
        }
        image_size = int(metadata["image_size"])
        class_labels = tuple(str(value) for value in json.loads(metadata["base_labels"]))
        label_to_index = {label: index for index, label in enumerate(class_labels)}

        if sample_ids is not None:
            placeholders = ",".join("?" for _ in sample_ids)
            rows = list(
                connection.execute(
                    f"""
                    SELECT sample_id, class_index, image_gray_u8, split,
                           original_width, original_height
                    FROM sample
                    WHERE sample_id IN ({placeholders})
                    ORDER BY sample_id
                    """,
                    tuple(sample_ids),
                )
            )
            found = {str(row["sample_id"]) for row in rows}
            missing = [sample_id for sample_id in sample_ids if sample_id not in found]
            if missing:
                raise ValueError(f"Unknown --sample-ids: {missing}")
            row_order = {sample_id: index for index, sample_id in enumerate(sample_ids)}
            rows.sort(key=lambda row: row_order[str(row["sample_id"])])
        else:
            unknown = [label for label in labels if label not in label_to_index]
            if unknown:
                raise ValueError(f"Unknown labels: {unknown}")
            rows = []
            for label in labels:
                class_index = label_to_index[label]
                candidates = list(
                    connection.execute(
                        """
                        SELECT sample_id, class_index, image_gray_u8, split,
                               original_width, original_height
                        FROM sample
                        WHERE split=? AND class_index=?
                        ORDER BY sample_id
                        """,
                        (split, class_index),
                    )
                )
                if not candidates:
                    raise ValueError(f"No samples for split={split!r} label={label!r}")
                ranked = sorted(
                    candidates,
                    key=lambda row: stable_rank(str(row["sample_id"]), seed=seed),
                )
                rows.extend(ranked[:samples_per_label])
    finally:
        connection.close()

    expected = image_size * image_size
    samples: list[Sample] = []
    for row in rows:
        raw = bytes(row["image_gray_u8"])
        if len(raw) != expected:
            raise ValueError(
                f"{row['sample_id']} has {len(raw)} bytes; expected {expected} for {image_size}x{image_size}"
            )
        class_index = int(row["class_index"])
        if class_index < 0 or class_index >= len(class_labels):
            raise ValueError(f"class_index out of range for {row['sample_id']}: {class_index}")
        samples.append(
            Sample(
                sample_id=str(row["sample_id"]),
                class_index=class_index,
                label=class_labels[class_index],
                image_u8=np.frombuffer(raw, dtype=np.uint8).reshape(image_size, image_size).copy(),
                original_width=int(row["original_width"]),
                original_height=int(row["original_height"]),
            )
        )
    return samples, class_labels, image_size


def preletterbox_content_extent(
    original_width: int,
    original_height: int,
    *,
    image_size: int,
) -> tuple[float, float]:
    if original_width <= 0 or original_height <= 0 or image_size <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(image_size / original_width, image_size / original_height)
    resized_width = max(
        1, min(image_size, int(np.floor(original_width * scale + 0.5)))
    )
    resized_height = max(
        1, min(image_size, int(np.floor(original_height * scale + 0.5)))
    )
    return resized_width / image_size, resized_height / image_size


def stable_rank(sample_id: str, *, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0gallery\0{sample_id}".encode("utf-8")).digest()


def tensor_to_u8(images: torch.Tensor) -> np.ndarray:
    return (
        images.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .squeeze(1)
        .cpu()
        .numpy()
    )


def geometry_parameter_summary(
    spec: PerspectiveAugmentationSpec,
    *,
    angle_deg: float,
    units: np.ndarray,
) -> dict[str, float]:
    if units.shape[0] < GEOMETRY_UNIT_COUNT:
        raise ValueError("geometry unit vector is too short")

    def lerp(column: int, minimum: float, maximum: float) -> float:
        return float(minimum + float(units[column]) * (maximum - minimum))

    def signed(column: int, maximum: float) -> float:
        return float((float(units[column]) * 2.0 - 1.0) * maximum)

    result = {"angle_deg": float(angle_deg)}
    if spec.name == "a0-random360":
        return result
    result.update(
        {
            "scale_x": lerp(0, spec.scale_x_min, spec.scale_x_max),
            "scale_y": lerp(1, spec.scale_y_min, spec.scale_y_max),
            "shear_x": signed(2, spec.shear_x_max),
            "shear_y": signed(3, spec.shear_y_max),
        }
    )
    if spec.use_perspective:
        result.update(
            {
                "perspective_yaw": signed(4, spec.perspective_yaw_max),
                "perspective_pitch": signed(5, spec.perspective_pitch_max),
                "keystone_x": signed(6, spec.keystone_x_max),
                "keystone_y": signed(7, spec.keystone_y_max),
            }
        )
    if spec.detector_style_recrop:
        delta = spec.bbox_scale_jitter_fraction
        result.update(
            {
                "bbox_center_x": signed(8, spec.bbox_center_jitter_fraction),
                "bbox_center_y": signed(9, spec.bbox_center_jitter_fraction),
                "bbox_scale_x": 1.0 + signed(10, delta),
                "bbox_scale_y": 1.0 + signed(11, delta),
            }
        )
    return result


def png_data_uri(image_u8: np.ndarray) -> str:
    height, width = image_u8.shape
    raw_rows = b"".join(
        b"\x00" + image_u8[row].tobytes()
        for row in range(height)
    )
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = signature + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", zlib.compress(raw_rows, 9)) + png_chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def format_parameters(parameters: dict[str, float]) -> str:
    chunks: list[str] = []
    for key, value in parameters.items():
        if key == "angle_deg":
            chunks.append(f"angle={value:+.1f}°")
        elif key.startswith("scale_") or key.startswith("bbox_scale_"):
            chunks.append(f"{key}={value:.3f}")
        else:
            chunks.append(f"{key}={value:+.3f}")
    return " · ".join(chunks)


def build_html(
    rows: Sequence[dict[str, Any]],
    *,
    database: Path,
    split: str,
    seed: int,
    epoch: int,
    class_labels: Sequence[str],
    augmentations: Sequence[str],
) -> str:
    columns = ("original",) + tuple(augmentations)
    label_names = {
        "original": "Original DB crop",
        "a0-random360": "A0 random360",
        "a1-anisotropic-affine": "A1 anisotropic affine",
        "a2-perspective": "A2 perspective",
        "a3-perspective-recrop": "A3 perspective + recrop",
    }
    sections: list[str] = []
    for row in rows:
        sample: Sample = row["sample"]
        cards: list[str] = []
        for column in columns:
            parameter_text = (
                "unaltered image_gray_u8"
                if column == "original"
                else format_parameters(row["parameters"][column])
            )
            cards.append(
                f"""
                <article class="card" data-augmentation="{html.escape(column)}">
                  <h3>{html.escape(label_names[column])}</h3>
                  <img src="{row['images'][column]}" alt="{html.escape(sample.sample_id)} {html.escape(column)}">
                  <div class="params">{html.escape(parameter_text)}</div>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="sample" data-label="{html.escape(sample.label)}">
              <header>
                <strong>{html.escape(sample.label)}</strong>
                <code>{html.escape(sample.sample_id)}</code>
              </header>
              <div class="cards">{''.join(cards)}</div>
            </section>
            """
        )

    labels_in_rows = sorted({row["sample"].label for row in rows})
    filter_buttons = "".join(
        f'<button type="button" data-filter="{html.escape(label)}">{html.escape(label)}</button>'
        for label in labels_in_rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>INV-013 real crop augmentation gallery</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0; padding: 24px; background: #111; color: #eee; }}
h1 {{ margin: 0 0 8px; }}
.meta {{ color: #aaa; margin-bottom: 16px; line-height: 1.5; }}
.controls {{ position: sticky; top: 0; z-index: 5; padding: 10px 0; background: #111e; backdrop-filter: blur(8px); }}
button {{ margin-right: 6px; margin-bottom: 4px; padding: 6px 10px; }}
.sample {{ border-top: 1px solid #333; padding: 20px 0; }}
.sample header {{ display: flex; gap: 12px; align-items: baseline; margin-bottom: 10px; }}
.sample header strong {{ font-size: 22px; }}
.sample header code {{ color: #999; word-break: break-all; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
.card {{ border: 1px solid #333; border-radius: 8px; padding: 10px; background: #181818; }}
.card h3 {{ font-size: 14px; margin: 0 0 8px; }}
.card img {{ width: 192px; height: 192px; max-width: 100%; object-fit: contain; image-rendering: pixelated; background: #777; border-radius: 4px; }}
.params {{ margin-top: 8px; color: #aaa; font: 12px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }}
.hidden {{ display: none; }}
body.zoom .card img {{ width: 320px; height: 320px; }}
</style>
</head>
<body>
<h1>INV-013 real crop augmentation gallery</h1>
<div class="meta">
  <div><strong>Database:</strong> <code>{html.escape(str(database))}</code></div>
  <div><strong>Split:</strong> {html.escape(split)} · <strong>seed:</strong> {seed} · <strong>epoch:</strong> {epoch}</div>
  <div>Images are actual <code>image_gray_u8</code> rows. A0-A3 use the exact INV-013 training transform path and <code>{GEOMETRY_STREAM}</code>.</div>
</div>
<div class="controls">
  <button type="button" data-filter="all">all</button>
  {filter_buttons}
  <button type="button" id="zoom">toggle zoom</button>
</div>
{''.join(sections)}
<script>
for (const button of document.querySelectorAll('[data-filter]')) {{
  button.addEventListener('click', () => {{
    const filter = button.dataset.filter;
    for (const section of document.querySelectorAll('.sample')) {{
      section.classList.toggle('hidden', filter !== 'all' && section.dataset.label !== filter);
    }}
  }});
}}
document.getElementById('zoom').addEventListener('click', () => document.body.classList.toggle('zoom'));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
