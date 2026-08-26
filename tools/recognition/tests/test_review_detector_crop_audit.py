from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from tools.recognition.detector_duplicate_groups import DetectorCandidate, build_duplicate_plan
from tools.recognition.review_detector_crop_audit import (
    PredictionStore,
    ReviewApplication,
    ReviewStore,
    classifier_matches_filter,
)


class ReviewDetectorCropAuditTest(unittest.TestCase):
    def test_valid_review_requires_base_label_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ReviewStore(Path(temporary_directory) / "reviews.sqlite")
            review = store.save(
                "candidate-1",
                decision="valid",
                label="3m",
                invalid_reason="background",
                note="usable detector crop",
            )
            self.assertEqual("valid", review["decision"])
            self.assertEqual("3m", review["label"])
            self.assertIsNone(review["invalid_reason"])

            with self.assertRaises(ValueError):
                store.save(
                    "candidate-2",
                    decision="valid",
                    label="invalid",
                    invalid_reason=None,
                    note="",
                )

    def test_invalid_review_clears_label_and_requires_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ReviewStore(Path(temporary_directory) / "reviews.sqlite")
            review = store.save(
                "candidate-1",
                decision="invalid",
                label="3m",
                invalid_reason="multi_tile",
                note="two tiles in one bbox",
            )
            self.assertEqual("invalid", review["decision"])
            self.assertIsNone(review["label"])
            self.assertEqual("multi_tile", review["invalid_reason"])

            with self.assertRaises(ValueError):
                store.save(
                    "candidate-2",
                    decision="invalid",
                    label=None,
                    invalid_reason="not-a-real-reason",
                    note="",
                )

    def test_delete_restores_unreviewed_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ReviewStore(Path(temporary_directory) / "reviews.sqlite")
            store.save(
                "candidate-1",
                decision="invalid",
                label=None,
                invalid_reason="background",
                note="",
            )
            self.assertTrue(store.delete("candidate-1"))
            self.assertIsNone(store.get("candidate-1"))

    def test_review_sidecar_is_bound_to_one_detector_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ReviewStore(Path(temporary_directory) / "reviews.sqlite")
            store.bind_dataset(detector_run_key="run-a", source_dataset="a.sqlite")
            store.bind_dataset(detector_run_key="run-a", source_dataset="moved-a.sqlite")
            with self.assertRaises(ValueError):
                store.bind_dataset(detector_run_key="run-b", source_dataset="b.sqlite")

    def test_duplicate_confirmation_is_separate_from_training_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ReviewStore(Path(temporary_directory) / "reviews.sqlite")
            store.confirm_duplicate("suppression-1")
            self.assertEqual({"suppression-1"}, store.duplicate_reviews())
            self.assertIsNone(store.get("suppression-1"))

    def test_classifier_disagreement_filters_use_human_review_as_truth(self) -> None:
        candidate = {"suggested_state": "single_gt", "suggested_label": "3m"}
        prediction = {"predicted_label": "4m", "confidence": 0.99}
        valid_review = {"decision": "valid", "label": "3m"}
        invalid_review = {"decision": "invalid", "label": None}

        self.assertTrue(
            classifier_matches_filter(
                "review_disagreement",
                candidate=candidate,
                review=valid_review,
                prediction=prediction,
                confidence_below=0.8,
            )
        )
        self.assertTrue(
            classifier_matches_filter(
                "review_disagreement",
                candidate=candidate,
                review=invalid_review,
                prediction=prediction,
                confidence_below=0.8,
            )
        )
        self.assertFalse(
            classifier_matches_filter(
                "strong_gt_disagreement",
                candidate=candidate,
                review=valid_review,
                prediction=prediction,
                confidence_below=0.8,
            )
        )

    def test_suspected_invalid_filter_targets_unreviewed_non_single_crops(self) -> None:
        self.assertTrue(
            classifier_matches_filter(
                "suspected_invalid_predicted_tile",
                candidate={"suggested_state": "multi_gt", "suggested_label": None},
                review=None,
                prediction={"predicted_label": "7s", "confidence": 0.95},
                confidence_below=0.8,
            )
        )
        self.assertFalse(
            classifier_matches_filter(
                "suspected_invalid_predicted_tile",
                candidate={"suggested_state": "multi_gt", "suggested_label": None},
                review=None,
                prediction={"predicted_label": "invalid", "confidence": 0.95},
                confidence_below=0.8,
            )
        )

    def test_review_application_contains_only_cluster_winner_and_singleton(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = root / "detector.sqlite"
            reviews = ReviewStore(root / "reviews.sqlite")
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """
                    CREATE TABLE candidate(
                        candidate_id TEXT PRIMARY KEY,
                        capture_id TEXT, campaign_id TEXT, layout_id TEXT, layout_ordinal INTEGER,
                        brightness TEXT, shadow TEXT, region TEXT,
                        source_region_path TEXT, source_composite_path TEXT,
                        detection_index INTEGER, detection_confidence REAL,
                        bbox_x REAL, bbox_y REAL, bbox_width REAL, bbox_height REAL,
                        crop_width INTEGER, crop_height INTEGER, image_png BLOB,
                        suggested_state TEXT, suggested_label TEXT,
                        best_gt_id TEXT, best_gt_label TEXT, best_iou REAL,
                        best_gt_coverage REAL, best_detection_coverage REAL,
                        substantial_gt_count INTEGER, gt_json TEXT
                    )
                    """
                )
                base = (
                    "campaign", "layout", 0, "bright", "none", "dora_indicators",
                    "region.png", "composite.png",
                )
                rows = [
                    ("winner", "cap", *base, 0, 0.95, 10.0, 10.0, 20.0, 30.0, 20, 30, b"w", "single_gt", "5m", None, "5m", 1.0, 1.0, 1.0, 1, "[]"),
                    ("loser", "cap", *base, 1, 0.70, 11.0, 10.0, 20.0, 30.0, 20, 30, b"l", "single_gt", "5m", None, "5m", 1.0, 1.0, 1.0, 1, "[]"),
                    ("singleton", "cap", *base, 2, 0.80, 100.0, 10.0, 20.0, 30.0, 20, 30, b"s", "background", None, None, None, 0.0, 0.0, 0.0, 0, "[]"),
                ]
                connection.executemany(
                    "INSERT INTO candidate VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                connection.commit()

            plan = build_duplicate_plan(
                [
                    DetectorCandidate("winner", "cap", "dora_indicators", 0, 0.95, 10.0, 10.0, 20.0, 30.0),
                    DetectorCandidate("loser", "cap", "dora_indicators", 1, 0.70, 11.0, 10.0, 20.0, 30.0),
                    DetectorCandidate("singleton", "cap", "dora_indicators", 2, 0.80, 100.0, 10.0, 20.0, 30.0),
                ],
                threshold=0.80,
            )
            application = ReviewApplication(
                repository_root=root,
                dataset_database=database,
                review_store=reviews,
                prediction_store=PredictionStore(None, None),
                duplicate_plan=plan,
            )
            self.assertEqual({"winner", "singleton"}, set(application.by_id))
            self.assertNotIn("loser", application.by_id)
            self.assertEqual(1, len(application.clusters))
            self.assertEqual("winner", application.clusters[0].winner.candidate_id)

            payload = application.candidate_payload(application.by_id["winner"].index)
            self.assertNotIn("image_png", payload)
            json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
