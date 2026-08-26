from __future__ import annotations

import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.recognition.build_red_five_classifier_dataset import (
    build_red_five_classifier_dataset,
    preprocess_rgb_u8,
)
from tools.recognition.build_red_five_dataset import SCHEMA as RED_FIVE_SCHEMA


class RedFiveRgbPreprocessTests(unittest.TestCase):
    def test_preserves_rgb_and_aspect_ratio_with_border_median_letterbox(self) -> None:
        image = Image.new("RGB", (8, 16), (220, 220, 220))
        for y in range(4, 12):
            for x in range(2, 6):
                image.putpixel((x, y), (200, 20, 10))

        raw = preprocess_rgb_u8(_png(image), image_size=64)
        array = np.frombuffer(raw, dtype=np.uint8).reshape(64, 64, 3)

        self.assertEqual(64 * 64 * 3, len(raw))
        self.assertTrue(np.all(array[:, :16, :] == 220))
        self.assertTrue(np.all(array[:, 48:, :] == 220))
        self.assertGreater(int(array[32, 32, 0]), int(array[32, 32, 1]))


class RedFiveClassifierDatasetBuildTests(unittest.TestCase):
    def test_balances_jp_train_and_splits_manual_by_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_database = root / "red_five_all.sqlite"
            output_database = root / "rgb64.sqlite"

            source = sqlite3.connect(source_database)
            try:
                source.executescript(RED_FIVE_SCHEMA)
                index = 0
                for suit in "mps":
                    for is_red in (0, 1):
                        label = f"red5{suit}" if is_red else f"5{suit}"
                        for ordinal in range(3):
                            index += 1
                            _insert_source_sample(
                                source,
                                crop_id=f"jp:train:{label}:{ordinal}",
                                source_name="jp",
                                partition="train",
                                suit=suit,
                                is_red=is_red,
                                label=label,
                                annotation_id=index,
                            )
                        index += 1
                        _insert_source_sample(
                            source,
                            crop_id=f"jp:valid:{label}",
                            source_name="jp",
                            partition="valid",
                            suit=suit,
                            is_red=is_red,
                            label=label,
                            annotation_id=index,
                        )
                        index += 1
                        _insert_source_sample(
                            source,
                            crop_id=f"jp:test:{label}",
                            source_name="jp",
                            partition="test",
                            suit=suit,
                            is_red=is_red,
                            label=label,
                            annotation_id=index,
                        )

                for capture_index in range(4):
                    capture_id = f"capture-{capture_index}"
                    for is_red in (0, 1):
                        index += 1
                        label = "red5m" if is_red else "5m"
                        _insert_source_sample(
                            source,
                            crop_id=f"manual:{capture_id}:{label}",
                            source_name="manual",
                            partition="capture",
                            suit="m",
                            is_red=is_red,
                            label=label,
                            annotation_id=index,
                            capture_id=capture_id,
                            brightness="bright",
                            shadow="none",
                        )
                source.commit()
            finally:
                source.close()

            summary = build_red_five_classifier_dataset(
                source_database=source_database,
                output_database=output_database,
                seed=42,
                image_size=64,
                jp_train_per_group=2,
                manual_train_fraction=0.5,
                manual_train_repeat=7,
                workers=2,
            )

            target = sqlite3.connect(output_database)
            target.row_factory = sqlite3.Row
            try:
                jp_train_groups = target.execute(
                    """
                    SELECT suit, is_red, COUNT(*) AS count
                    FROM sample
                    WHERE split='train' AND source='jp'
                    GROUP BY suit, is_red
                    ORDER BY suit, is_red
                    """
                ).fetchall()
                manual_rows = target.execute(
                    """
                    SELECT capture_id, split, train_repeat
                    FROM sample
                    WHERE source='manual'
                    ORDER BY capture_id, crop_id
                    """
                ).fetchall()
            finally:
                target.close()

            self.assertEqual(
                [(suit, is_red, 2) for suit in "mps" for is_red in (0, 1)],
                [(row["suit"], row["is_red"], row["count"]) for row in jp_train_groups],
            )
            train_captures = {
                row["capture_id"] for row in manual_rows if row["split"] == "train"
            }
            val_captures = {
                row["capture_id"] for row in manual_rows if row["split"] == "manual_val"
            }
            self.assertTrue(train_captures)
            self.assertTrue(val_captures)
            self.assertFalse(train_captures & val_captures)
            self.assertTrue(
                all(
                    row["train_repeat"] == (7 if row["split"] == "train" else 1)
                    for row in manual_rows
                )
            )
            self.assertEqual(6, summary["counts_by_split"]["jp_val"])
            self.assertEqual(6, summary["counts_by_split"]["jp_test"])


def _insert_source_sample(
    connection: sqlite3.Connection,
    *,
    crop_id: str,
    source_name: str,
    partition: str,
    suit: str,
    is_red: int,
    label: str,
    annotation_id: int,
    capture_id: str | None = None,
    brightness: str | None = None,
    shadow: str | None = None,
) -> None:
    image = Image.new("RGB", (12, 18), (235, 235, 235))
    if is_red:
        for y in range(4, 14):
            for x in range(3, 9):
                image.putpixel((x, y), (210, 25, 20))
    connection.execute(
        """
        INSERT INTO sample(
            sample_id, crop_id, source, source_partition, suit, is_red, source_label,
            raw_category_name, raw_category_id, image_format, image_width, image_height,
            image_png, source_image_path, source_image_id, source_annotation_id, bbox_json,
            capture_id, layout_id, layout_ordinal, region, group_name, group_ordinal,
            tile_ordinal, brightness, shadow, annotation_angle_deg, expected_rotation_deg
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, 'png', 12, 18, ?, ?, ?, ?, ?,
            ?, 'layout', 0, 'completed_hand', 'hand', 0, 0, ?, ?, 0, 0
        )
        """,
        (
            crop_id,
            crop_id,
            source_name,
            partition,
            suit,
            is_red,
            label,
            label,
            annotation_id if source_name == "jp" else None,
            _png(image),
            f"images/{crop_id}.png",
            crop_id,
            str(annotation_id),
            '{"kind":"test"}',
            capture_id,
            brightness,
            shadow,
        ),
    )


def _png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


if __name__ == "__main__":
    unittest.main()
