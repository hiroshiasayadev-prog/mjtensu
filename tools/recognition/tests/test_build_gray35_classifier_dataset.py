from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from PIL import Image

from tools.recognition.build_gray35_classifier_dataset import (
    CLASS_LABELS,
    INVALID_LABEL,
    build_gray35_dataset,
)
from tools.recognition.build_tile_classifier_dataset import BASE_LABELS
from tools.recognition.review_detector_crop_audit import ReviewStore


class BuildGray35ClassifierDatasetTest(unittest.TestCase):
    def make_png(self, value: int) -> bytes:
        image = Image.new("L", (12, 18), color=value)
        output = io.BytesIO()
        image.save(output, format="PNG")
        image.close()
        return output.getvalue()

    def create_base_database(self, path: Path) -> None:
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(
                """
                CREATE TABLE experiment_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE sample(
                    sample_id TEXT PRIMARY KEY,
                    split TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_partition TEXT NOT NULL,
                    base_label TEXT NOT NULL,
                    class_index INTEGER NOT NULL,
                    original_label TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    quality_audit_decision TEXT,
                    crop_id TEXT NOT NULL,
                    image_size INTEGER NOT NULL,
                    image_gray_u8 BLOB NOT NULL,
                    original_width INTEGER NOT NULL,
                    original_height INTEGER NOT NULL,
                    source_image_path TEXT NOT NULL,
                    source_image_id TEXT,
                    source_annotation_id TEXT NOT NULL,
                    capture_id TEXT,
                    layout_id TEXT,
                    region TEXT,
                    brightness TEXT,
                    shadow TEXT,
                    annotation_angle_deg REAL NOT NULL,
                    expected_rotation_deg INTEGER NOT NULL
                );
                """
            )
            connection.executemany(
                "INSERT INTO experiment_metadata(key, value) VALUES (?, ?)",
                [
                    ("base_labels", json.dumps(BASE_LABELS)),
                    ("image_size", "16"),
                    ("schema_version", "2"),
                ],
            )
            raw = bytes([100]) * (16 * 16)
            rows = [
                (
                    "train:base-1", "train", "manual", "manual", "1m", 0,
                    "1m", "1m", None, "base-1", 16, raw, 12, 18,
                    "capture/train.png", "cap-train", "ann-1", "cap-train",
                    "layout-1", "completed_hand", "bright", "none", 0.0, 0,
                ),
                (
                    "manual_val:base-2", "manual_val", "manual", "manual", "2m", 1,
                    "2m", "2m", None, "base-2", 16, raw, 12, 18,
                    "capture/val.png", "cap-val", "ann-2", "cap-val",
                    "layout-2", "completed_hand", "dark", "partial", 0.0, 0,
                ),
            ]
            connection.executemany(
                "INSERT INTO sample VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            connection.commit()

    def create_detector_database(self, path: Path) -> None:
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(
                """
                CREATE TABLE dataset_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE candidate(
                    candidate_id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL,
                    layout_id TEXT NOT NULL,
                    brightness TEXT NOT NULL,
                    shadow TEXT NOT NULL,
                    region TEXT NOT NULL,
                    source_region_path TEXT NOT NULL,
                    detection_index INTEGER NOT NULL,
                    detection_confidence REAL NOT NULL,
                    bbox_x REAL NOT NULL,
                    bbox_y REAL NOT NULL,
                    bbox_width REAL NOT NULL,
                    bbox_height REAL NOT NULL,
                    crop_width INTEGER NOT NULL,
                    crop_height INTEGER NOT NULL,
                    image_png BLOB NOT NULL
                );
                CREATE TABLE postprocess_decision(
                    candidate_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    reason TEXT,
                    winner_candidate_id TEXT,
                    overlap_ratio REAL
                );
                """
            )
            connection.executemany(
                "INSERT INTO dataset_metadata(key, value) VALUES (?, ?)",
                [
                    ("detector_run_key", "run-test"),
                    ("duplicate_overlap_threshold", "0.8"),
                ],
            )
            connection.executemany(
                "INSERT INTO candidate VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        "candidate-valid", "cap-train", "layout-1", "bright", "none",
                        "completed_hand", "capture/train-region.png", 0, 0.90,
                        0.0, 0.0, 20.0, 30.0, 12, 18, self.make_png(120),
                    ),
                    (
                        "candidate-invalid", "cap-val", "layout-2", "dark", "partial",
                        "melds", "capture/val-region.png", 1, 0.90,
                        0.0, 0.0, 20.0, 30.0, 12, 18, self.make_png(80),
                    ),
                    (
                        "candidate-unreviewed", "cap-train", "layout-1", "bright", "none",
                        "completed_hand", "capture/train-region.png", 2, 0.80,
                        100.0, 0.0, 20.0, 30.0, 12, 18, self.make_png(200),
                    ),
                    (
                        "candidate-removed", "cap-train", "layout-1", "bright", "none",
                        "completed_hand", "capture/train-region.png", 3, 0.70,
                        201.0, 0.0, 20.0, 30.0, 12, 18, self.make_png(180),
                    ),
                    (
                        "candidate-winner", "cap-train", "layout-1", "bright", "none",
                        "completed_hand", "capture/train-region.png", 4, 0.95,
                        200.0, 0.0, 20.0, 30.0, 12, 18, self.make_png(170),
                    ),
                ],
            )
            connection.executemany(
                "INSERT INTO postprocess_decision VALUES (?,?,?,?,?)",
                [
                    ("candidate-valid", "keep", None, None, None),
                    ("candidate-invalid", "keep", None, None, None),
                    ("candidate-unreviewed", "keep", None, None, None),
                    # Deliberately wrong legacy postprocess rows. The rebuilt pipeline must ignore
                    # this table and recompute winner/loser directly from raw candidate geometry.
                    ("candidate-winner", "remove", "duplicate", "candidate-removed", 0.91),
                    ("candidate-removed", "keep", None, None, None),
                ],
            )
            connection.commit()

    def test_builder_adds_only_human_reviewed_detector_crops_and_preserves_capture_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = root / "base.sqlite"
            detector = root / "detector.sqlite"
            reviews = root / "reviews.sqlite"
            output = root / "gray35.sqlite"
            self.create_base_database(base)
            self.create_detector_database(detector)

            review_store = ReviewStore(reviews)
            review_store.bind_dataset(
                detector_run_key="run-test",
                source_dataset=str(detector),
            )
            review_store.save(
                "candidate-valid",
                decision="valid",
                label="3m",
                invalid_reason=None,
                note="",
            )
            review_store.save(
                "candidate-invalid",
                decision="invalid",
                label=None,
                invalid_reason="multi_tile",
                note="",
            )
            review_store.save(
                "candidate-removed",
                decision="invalid",
                label=None,
                invalid_reason="background",
                note="duplicate loser review must be ignored",
            )
            review_store.save(
                "candidate-winner",
                decision="valid",
                label="4m",
                invalid_reason=None,
                note="duplicate winner survives postprocess and remains a normal review sample",
            )

            summary = build_gray35_dataset(
                base_database=base,
                detector_database=detector,
                review_database=reviews,
                output_database=output,
                force=False,
            )

            self.assertEqual(5, summary["sample_count"])
            self.assertEqual(2, summary["copied_gray34_samples"])
            self.assertEqual(3, summary["reviewed_detector_samples"])
            self.assertEqual(1, summary["reviewed_detector_counts_by_label"][INVALID_LABEL])

            with closing(sqlite3.connect(output)) as connection:
                connection.row_factory = sqlite3.Row
                labels = tuple(
                    json.loads(
                        connection.execute(
                            "SELECT value FROM experiment_metadata WHERE key='base_labels'"
                        ).fetchone()[0]
                    )
                )
                self.assertEqual(CLASS_LABELS, labels)

                valid = connection.execute(
                    "SELECT * FROM sample WHERE detector_candidate_id='candidate-valid'"
                ).fetchone()
                self.assertIsNotNone(valid)
                self.assertEqual("train", valid["split"])
                self.assertEqual("3m", valid["base_label"])

                invalid = connection.execute(
                    "SELECT * FROM sample WHERE detector_candidate_id='candidate-invalid'"
                ).fetchone()
                self.assertIsNotNone(invalid)
                self.assertEqual("manual_val", invalid["split"])
                self.assertEqual("invalid", invalid["base_label"])
                self.assertEqual(34, invalid["class_index"])
                self.assertEqual("multi_tile", invalid["invalid_reason"])

                unreviewed = connection.execute(
                    "SELECT * FROM sample WHERE detector_candidate_id='candidate-unreviewed'"
                ).fetchone()
                self.assertIsNone(unreviewed)

                removed = connection.execute(
                    "SELECT * FROM sample WHERE detector_candidate_id='candidate-removed'"
                ).fetchone()
                self.assertIsNone(removed)

                winner = connection.execute(
                    "SELECT * FROM sample WHERE detector_candidate_id='candidate-winner'"
                ).fetchone()
                self.assertIsNotNone(winner)
                self.assertEqual("4m", winner["base_label"])


if __name__ == "__main__":
    unittest.main()
