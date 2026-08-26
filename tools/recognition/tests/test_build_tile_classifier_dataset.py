from __future__ import annotations

import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.recognition.build_tile_classifier_dataset import (
    BASE_LABELS,
    base_label,
    build_classifier_dataset,
    preprocess_gray_u8,
)


class BaseLabelTests(unittest.TestCase):
    def test_red_fives_merge_into_34_class_shape_labels(self) -> None:
        self.assertEqual("5m", base_label("red5m"))
        self.assertEqual("5p", base_label("red5p"))
        self.assertEqual("5s", base_label("red5s"))
        self.assertEqual("red", base_label("red"))
        self.assertEqual(34, len(BASE_LABELS))


class PreprocessTests(unittest.TestCase):
    def test_preprocess_is_lossless_u8_shape_contract_after_letterbox(self) -> None:
        source = Image.new("L", (8, 16), 220)
        for y in range(4, 12):
            for x in range(2, 6):
                source.putpixel((x, y), 20)
        raw = preprocess_gray_u8(_png(source), image_size=64)
        array = np.frombuffer(raw, dtype=np.uint8).reshape(64, 64)

        self.assertEqual(64 * 64, len(raw))
        # 1:2 source aspect ratio must remain 1:2, centered in the square canvas.
        self.assertTrue(np.all(array[:, :16] == 220))
        self.assertTrue(np.all(array[:, 48:] == 220))
        self.assertLess(int(array[32, 32]), 100)


class BuildClassifierDatasetTests(unittest.TestCase):
    def test_jp_is_capped_per_base_class_and_manual_is_split_by_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_database = root / "source.sqlite"
            output_database = root / "classifier.sqlite"
            source = sqlite3.connect(source_database)
            try:
                _create_source_schema(source)
                annotation = 1
                for partition in ("train", "valid"):
                    for label in BASE_LABELS:
                        _insert_crop(
                            source,
                            crop_id=f"jp:{partition}:{label}",
                            source_name="jp",
                            partition=partition,
                            label=label,
                            annotation_id=annotation,
                        )
                        annotation += 1
                for capture_index in range(2):
                    capture_id = f"capture-{capture_index}"
                    for label in BASE_LABELS:
                        _insert_crop(
                            source,
                            crop_id=f"manual:{capture_id}:{label}",
                            source_name="manual",
                            partition="capture",
                            label=label,
                            annotation_id=annotation,
                            capture_id=capture_id,
                            brightness="dark",
                            shadow="shadow",
                        )
                        annotation += 1
                source.commit()
            finally:
                source.close()

            summary = build_classifier_dataset(
                source_database=source_database,
                output_database=output_database,
                seed=42,
                image_size=64,
                jp_train_per_class=1,
                jp_valid_per_class=1,
                manual_train_fraction=0.5,
                workers=2,
            )

            self.assertEqual(34, summary["counts_by_split"]["jp_val"])
            self.assertEqual(34, summary["counts_by_split"]["manual_val"])
            self.assertEqual(68, summary["counts_by_split"]["train"])

            output = sqlite3.connect(output_database)
            try:
                captures_by_split = output.execute(
                    """
                    SELECT split, capture_id
                    FROM sample
                    WHERE source = 'manual'
                    GROUP BY split, capture_id
                    ORDER BY split, capture_id
                    """
                ).fetchall()
                image_lengths = {
                    int(row[0])
                    for row in output.execute("SELECT LENGTH(image_gray_u8) FROM sample")
                }
            finally:
                output.close()

            self.assertEqual(2, len(captures_by_split))
            self.assertNotEqual(captures_by_split[0][1], captures_by_split[1][1])
            self.assertEqual({4096}, image_lengths)

    def test_quality_audit_corrections_and_exclusions_are_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_database = root / "source.sqlite"
            quality_database = root / "quality.sqlite"
            output_database = root / "classifier.sqlite"

            source = sqlite3.connect(source_database)
            try:
                _create_source_schema(source)
                annotation = 1
                for partition in ("train", "valid"):
                    for label in BASE_LABELS:
                        _insert_crop(
                            source,
                            crop_id=f"jp:{partition}:{label}",
                            source_name="jp",
                            partition=partition,
                            label=label,
                            annotation_id=annotation,
                        )
                        annotation += 1
                for capture_index in range(2):
                    capture_id = f"capture-{capture_index}"
                    for label in BASE_LABELS:
                        _insert_crop(
                            source,
                            crop_id=f"manual:{capture_id}:{label}",
                            source_name="manual",
                            partition="capture",
                            label=label,
                            annotation_id=annotation,
                            capture_id=capture_id,
                            brightness="bright",
                            shadow="none",
                        )
                        annotation += 1
                source.commit()
            finally:
                source.close()

            quality = sqlite3.connect(quality_database)
            try:
                quality.execute(
                    """
                    CREATE TABLE review(
                        crop_id TEXT PRIMARY KEY,
                        decision TEXT NOT NULL,
                        corrected_label TEXT
                    )
                    """
                )
                quality.executemany(
                    "INSERT INTO review(crop_id, decision, corrected_label) VALUES (?, ?, ?)",
                    [
                        ("manual:capture-0:1m", "label_error", "2m"),
                        ("manual:capture-0:2m", "false_detection", None),
                        ("manual:capture-0:3m", "unusable_crop", None),
                        ("manual:capture-0:4m", "background", None),
                    ],
                )
                quality.commit()
            finally:
                quality.close()

            summary = build_classifier_dataset(
                source_database=source_database,
                output_database=output_database,
                seed=42,
                image_size=64,
                jp_train_per_class=1,
                jp_valid_per_class=1,
                manual_train_fraction=0.5,
                workers=2,
                quality_audit_database=quality_database,
            )

            self.assertEqual(4, summary["quality_audit"]["review_count"])
            self.assertEqual(1, summary["quality_audit"]["decision_counts"]["label_error"])
            self.assertEqual(1, summary["quality_audit"]["decision_counts"]["false_detection"])
            self.assertEqual(1, summary["quality_audit"]["decision_counts"]["unusable_crop"])
            self.assertEqual(1, summary["quality_audit"]["decision_counts"]["background"])

            output = sqlite3.connect(output_database)
            output.row_factory = sqlite3.Row
            try:
                corrected = output.execute(
                    """
                    SELECT source_label, original_label, base_label, quality_audit_decision
                    FROM sample WHERE crop_id = 'manual:capture-0:1m'
                    """
                ).fetchone()
                false_detection = output.execute(
                    """
                    SELECT source_label, original_label, base_label, quality_audit_decision
                    FROM sample WHERE crop_id = 'manual:capture-0:2m'
                    """
                ).fetchone()
                excluded_count = output.execute(
                    """
                    SELECT COUNT(*) FROM sample
                    WHERE crop_id IN ('manual:capture-0:3m', 'manual:capture-0:4m')
                    """
                ).fetchone()[0]
            finally:
                output.close()

            self.assertIsNotNone(corrected)
            assert corrected is not None
            self.assertEqual("1m", corrected["source_label"])
            self.assertEqual("2m", corrected["original_label"])
            self.assertEqual("2m", corrected["base_label"])
            self.assertEqual("label_error", corrected["quality_audit_decision"])

            self.assertIsNotNone(false_detection)
            assert false_detection is not None
            self.assertEqual("2m", false_detection["source_label"])
            self.assertEqual("2m", false_detection["original_label"])
            self.assertEqual("2m", false_detection["base_label"])
            self.assertEqual("false_detection", false_detection["quality_audit_decision"])
            self.assertEqual(0, excluded_count)


def _create_source_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE tile_crop(
            crop_id TEXT PRIMARY KEY,
            source TEXT,
            source_partition TEXT,
            tile_label TEXT,
            image_width INTEGER,
            image_height INTEGER,
            image_png BLOB,
            source_image_path TEXT,
            source_image_id TEXT,
            source_annotation_id TEXT,
            capture_id TEXT,
            layout_id TEXT,
            region TEXT,
            brightness TEXT,
            shadow TEXT,
            annotation_angle_deg REAL,
            expected_rotation_deg INTEGER
        );
        CREATE INDEX idx_tile_crop_source_label
        ON tile_crop(source, tile_label);
        """
    )


def _insert_crop(
    connection: sqlite3.Connection,
    *,
    crop_id: str,
    source_name: str,
    partition: str,
    label: str,
    annotation_id: int,
    capture_id: str | None = None,
    brightness: str | None = None,
    shadow: str | None = None,
) -> None:
    image = Image.new("RGB", (12, 18), (230, 230, 230))
    connection.execute(
        """
        INSERT INTO tile_crop(
            crop_id, source, source_partition, tile_label,
            image_width, image_height, image_png,
            source_image_path, source_image_id, source_annotation_id,
            capture_id, layout_id, region, brightness, shadow,
            annotation_angle_deg, expected_rotation_deg
        ) VALUES (?, ?, ?, ?, 12, 18, ?, ?, ?, ?, ?, 'layout', 'completed_hand', ?, ?, 0, 0)
        """,
        (
            crop_id,
            source_name,
            partition,
            label,
            _png(image),
            f"images/{crop_id}.png",
            crop_id,
            str(annotation_id),
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
