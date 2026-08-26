from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.recognition.build_red_five_dataset import build_red_five_dataset
from tools.recognition.build_tile_crop_dataset import SCHEMA as TILE_CROP_SCHEMA


class RedFiveDatasetBuildTests(unittest.TestCase):
    def test_collects_only_normal_and_red_fives_and_preserves_rgb_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_database = root / "source.sqlite"
            output_database = root / "red_five.sqlite"
            image_blob = b"fake-png-rgb-bytes"

            source = sqlite3.connect(source_database)
            try:
                source.executescript(TILE_CROP_SCHEMA)
                insert_crop(source, "jp:train:1", "jp", "train", "5m", image_blob)
                insert_crop(source, "jp:train:2", "jp", "train", "red5m", image_blob)
                insert_crop(source, "jp:valid:3", "jp", "valid", "5p", image_blob)
                insert_crop(source, "jp:test:4", "jp", "test", "red5p", image_blob)
                insert_crop(source, "manual:1", "manual", "capture", "5s", image_blob)
                insert_crop(source, "manual:2", "manual", "capture", "red5s", image_blob)
                insert_crop(source, "jp:train:7", "jp", "train", "1m", image_blob)
                source.commit()
            finally:
                source.close()

            summary = build_red_five_dataset(
                source_database=source_database,
                output_database=output_database,
            )

            target = sqlite3.connect(output_database)
            target.row_factory = sqlite3.Row
            try:
                rows = target.execute(
                    """
                    SELECT crop_id, suit, is_red, source_label, image_png
                    FROM sample
                    ORDER BY crop_id
                    """
                ).fetchall()
            finally:
                target.close()

            self.assertEqual(6, summary["sample_count"])
            self.assertEqual(
                {
                    "5m": 1,
                    "5p": 1,
                    "5s": 1,
                    "red5m": 1,
                    "red5p": 1,
                    "red5s": 1,
                },
                summary["counts_by_source_label"],
            )
            self.assertEqual({"normal": 3, "red": 3}, summary["counts_by_red_state"])
            self.assertEqual(
                [
                    ("jp:test:4", "p", 1, "red5p", image_blob),
                    ("jp:train:1", "m", 0, "5m", image_blob),
                    ("jp:train:2", "m", 1, "red5m", image_blob),
                    ("jp:valid:3", "p", 0, "5p", image_blob),
                    ("manual:1", "s", 0, "5s", image_blob),
                    ("manual:2", "s", 1, "red5s", image_blob),
                ],
                [
                    (
                        row["crop_id"],
                        row["suit"],
                        row["is_red"],
                        row["source_label"],
                        bytes(row["image_png"]),
                    )
                    for row in rows
                ],
            )

    def test_can_limit_collection_to_jp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_database = root / "source.sqlite"
            output_database = root / "red_five.sqlite"

            source = sqlite3.connect(source_database)
            try:
                source.executescript(TILE_CROP_SCHEMA)
                insert_crop(source, "jp:1", "jp", "train", "red5m", b"jp")
                insert_crop(source, "manual:1", "manual", "capture", "red5m", b"manual")
                source.commit()
            finally:
                source.close()

            summary = build_red_five_dataset(
                source_database=source_database,
                output_database=output_database,
                sources=("jp",),
            )

            self.assertEqual(1, summary["sample_count"])
            self.assertEqual(["jp"], summary["included_sources"])


def insert_crop(
    connection: sqlite3.Connection,
    crop_id: str,
    source: str,
    source_partition: str,
    tile_label: str,
    image_png: bytes,
) -> None:
    connection.execute(
        """
        INSERT INTO tile_crop(
            crop_id,
            source,
            source_partition,
            tile_label,
            raw_category_name,
            raw_category_id,
            image_format,
            image_width,
            image_height,
            image_png,
            source_image_path,
            source_image_id,
            source_annotation_id,
            bbox_json,
            annotation_angle_deg,
            expected_rotation_deg
        ) VALUES (?, ?, ?, ?, ?, ?, 'png', 10, 20, ?, ?, ?, ?, ?, 0, 0)
        """,
        (
            crop_id,
            source,
            source_partition,
            tile_label,
            tile_label,
            1 if source == "jp" else None,
            image_png,
            f"images/{crop_id}.png",
            crop_id,
            crop_id,
            '{"kind":"test"}',
        ),
    )


if __name__ == "__main__":
    unittest.main()
