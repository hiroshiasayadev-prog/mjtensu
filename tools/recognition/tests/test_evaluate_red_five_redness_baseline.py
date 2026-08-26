from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.recognition.build_red_five_classifier_dataset import SCHEMA
from tools.recognition.evaluate_red_five_redness_baseline import (
    evaluate_database,
    fit_threshold,
    mean_red_dominance,
    red_pixel_fraction_margin20,
)


class RednessScoreTests(unittest.TestCase):
    def test_red_image_scores_above_neutral_image(self) -> None:
        neutral = np.full((64, 64, 3), 220, dtype=np.uint8)
        red = neutral.copy()
        red[16:48, 16:48, :] = np.asarray([220, 30, 20], dtype=np.uint8)

        self.assertGreater(mean_red_dominance(red), mean_red_dominance(neutral))
        self.assertGreater(
            red_pixel_fraction_margin20(red),
            red_pixel_fraction_margin20(neutral),
        )

    def test_threshold_fit_separates_simple_scores(self) -> None:
        scores = np.asarray([0.01, 0.02, 0.10, 0.12], dtype=np.float64)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)

        threshold, metrics = fit_threshold(scores, labels)

        self.assertGreater(threshold, 0.02)
        self.assertLess(threshold, 0.10)
        self.assertEqual(1.0, metrics["balanced_accuracy"])


class RednessBaselineDatabaseTests(unittest.TestCase):
    def test_fits_on_jp_val_and_evaluates_jp_test_and_manual_val(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "rgb64.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.executescript(SCHEMA)
                for split in ("jp_val", "jp_test", "manual_val"):
                    for suit in "mps":
                        _insert(connection, split, suit, 0, red_fraction=0.0)
                        _insert(connection, split, suit, 1, red_fraction=0.25)
                connection.commit()
            finally:
                connection.close()

            result = evaluate_database(database)

            self.assertEqual("completed", result["status"])
            for score_result in result["scores"].values():
                self.assertEqual("jp_val", score_result["threshold_fit_split"])
                self.assertEqual(
                    1.0,
                    score_result["splits"]["jp_test"]["overall"]["balanced_accuracy"],
                )
                self.assertEqual(
                    1.0,
                    score_result["splits"]["manual_val"]["overall"]["balanced_accuracy"],
                )


def _insert(
    connection: sqlite3.Connection,
    split: str,
    suit: str,
    is_red: int,
    *,
    red_fraction: float,
) -> None:
    image = np.full((64, 64, 3), 220, dtype=np.uint8)
    red_pixels = int(round(64 * 64 * red_fraction))
    if red_pixels:
        flat = image.reshape(-1, 3)
        flat[:red_pixels, :] = np.asarray([220, 25, 20], dtype=np.uint8)
    crop_id = f"{split}:{suit}:{is_red}"
    source = "manual" if split == "manual_val" else "jp"
    partition = "capture" if source == "manual" else ("valid" if split == "jp_val" else "test")
    connection.execute(
        """
        INSERT INTO sample(
            sample_id, split, source, source_partition, suit, is_red, source_label,
            crop_id, image_size, image_rgb_u8, train_repeat,
            original_width, original_height, source_image_path, source_image_id,
            source_annotation_id, capture_id, layout_id, region, brightness, shadow,
            annotation_angle_deg, expected_rotation_deg
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 64, ?, 1, 64, 64, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
        """,
        (
            crop_id,
            split,
            source,
            partition,
            suit,
            is_red,
            f"red5{suit}" if is_red else f"5{suit}",
            crop_id,
            image.tobytes(),
            f"images/{crop_id}.png",
            crop_id,
            crop_id,
            "capture" if source == "manual" else None,
            "layout" if source == "manual" else None,
            "completed_hand" if source == "manual" else None,
            "bright" if source == "manual" else None,
            "none" if source == "manual" else None,
        ),
    )


if __name__ == "__main__":
    unittest.main()
