from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from PIL import Image

from tools.recognition.build_gray35_classifier_dataset import CLASS_LABELS
from tools.recognition.extend_gray35_classifier_dataset import extend_gray35_dataset
from tools.recognition.review_detector_crop_audit import ReviewStore


class ExtendGray35ClassifierDatasetTest(unittest.TestCase):
    def png(self, value: int) -> bytes:
        image = Image.new("L", (12, 18), color=value)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def create_base(self, path: Path) -> None:
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(
                """
                CREATE TABLE experiment_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE sample(
                    sample_id TEXT PRIMARY KEY, split TEXT NOT NULL, source TEXT NOT NULL,
                    source_partition TEXT NOT NULL, base_label TEXT NOT NULL, class_index INTEGER NOT NULL,
                    original_label TEXT NOT NULL, source_label TEXT NOT NULL, quality_audit_decision TEXT,
                    crop_id TEXT NOT NULL, image_size INTEGER NOT NULL, image_gray_u8 BLOB NOT NULL,
                    original_width INTEGER NOT NULL, original_height INTEGER NOT NULL,
                    source_image_path TEXT NOT NULL, source_image_id TEXT, source_annotation_id TEXT NOT NULL,
                    capture_id TEXT, layout_id TEXT, region TEXT, brightness TEXT, shadow TEXT,
                    annotation_angle_deg REAL NOT NULL, expected_rotation_deg INTEGER NOT NULL,
                    detector_candidate_id TEXT, detector_review_decision TEXT, invalid_reason TEXT
                );
                CREATE UNIQUE INDEX idx_sample_detector_candidate ON sample(detector_candidate_id)
                WHERE detector_candidate_id IS NOT NULL;
                """
            )
            connection.executemany(
                "INSERT INTO experiment_metadata VALUES (?, ?)",
                [
                    ("base_labels", json.dumps(CLASS_LABELS)),
                    ("class_count", "35"),
                    ("image_size", "16"),
                    ("task", "gray35_base_tile_plus_invalid"),
                ],
            )
            raw = bytes([100]) * 256
            connection.execute(
                "INSERT INTO sample VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "manual_val:base", "manual_val", "manual", "capture", "1m", 0,
                    "1m", "1m", None, "base", 16, raw, 12, 18, "manual.png", "cap",
                    "ann", "cap", "layout", "completed_hand", "bright", "none", 0.0, 0,
                    None, None, None,
                ),
            )
            connection.commit()

    def create_detector(self, path: Path) -> None:
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(
                """
                CREATE TABLE dataset_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE candidate(
                    candidate_id TEXT PRIMARY KEY, capture_id TEXT NOT NULL, layout_id TEXT NOT NULL,
                    brightness TEXT NOT NULL, shadow TEXT NOT NULL, region TEXT NOT NULL,
                    source_region_path TEXT NOT NULL, detection_index INTEGER NOT NULL,
                    detection_confidence REAL NOT NULL, bbox_x REAL NOT NULL, bbox_y REAL NOT NULL,
                    bbox_width REAL NOT NULL, bbox_height REAL NOT NULL, crop_width INTEGER NOT NULL,
                    crop_height INTEGER NOT NULL, image_png BLOB NOT NULL
                );
                """
            )
            connection.executemany(
                "INSERT INTO dataset_metadata VALUES (?, ?)",
                [
                    ("detector_run_key", "jp-run"),
                    ("duplicate_overlap_threshold", "0.8"),
                    ("source", "jp"),
                    ("source_partition", "train"),
                ],
            )
            connection.executemany(
                "INSERT INTO candidate VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    ("jp-invalid", "jp:train:1", "full_image", "unknown", "unknown", "full_image",
                     "data/jp/1.jpg", 0, 0.8, 0.0, 0.0, 20.0, 30.0, 12, 18, self.png(80)),
                    ("jp-valid", "jp:train:2", "full_image", "unknown", "unknown", "full_image",
                     "data/jp/2.jpg", 0, 0.9, 0.0, 0.0, 20.0, 30.0, 12, 18, self.png(120)),
                ],
            )
            connection.commit()

    def test_appends_jp_train_without_changing_validation_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = root / "v2.sqlite"
            detector = root / "jp.sqlite"
            reviews = root / "reviews.sqlite"
            output = root / "v3.sqlite"
            self.create_base(base)
            self.create_detector(detector)
            store = ReviewStore(reviews)
            store.bind_dataset(detector_run_key="jp-run", source_dataset=str(detector))
            store.save("jp-invalid", decision="invalid", label=None, invalid_reason="background", note="")
            store.save("jp-valid", decision="valid", label="7p", invalid_reason=None, note="")

            summary = extend_gray35_dataset(
                base_database=base,
                detector_database=detector,
                review_database=reviews,
                output_database=output,
            )

            self.assertEqual(1, summary["base_sample_count"])
            self.assertEqual(2, summary["appended_reviewed_samples"])
            self.assertEqual({"manual_val": 1}, summary["counts_by_split_before"])
            self.assertEqual({"manual_val": 1, "train": 2}, summary["counts_by_split_after"])
            with closing(sqlite3.connect(output)) as connection:
                rows = connection.execute(
                    "SELECT split, source, base_label FROM sample WHERE detector_candidate_id IS NOT NULL ORDER BY detector_candidate_id"
                ).fetchall()
                self.assertEqual(
                    [("train", "detector_jp", "invalid"), ("train", "detector_jp", "7p")],
                    rows,
                )


if __name__ == "__main__":
    unittest.main()
