from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.recognition.review_tile_crop_label_audit import Candidate, ReviewStore


class ReviewTileCropLabelAuditTest(unittest.TestCase):
    def make_candidate(self) -> Candidate:
        return Candidate(
            rank=1,
            values={
                "crop_id": "jp:train:example",
                "tier": "1",
                "source": "jp",
                "source_partition": "train",
                "expected_label": "3s",
                "consensus_prediction": "3m",
            },
        )

    def test_label_error_requires_and_persists_corrected_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ReviewStore(Path(temporary_directory) / "quality.sqlite")
            candidate = self.make_candidate()
            review = store.save(
                candidate,
                decision="label_error",
                corrected_label="3m",
                note="wrong source label",
            )
            self.assertEqual("label_error", review["decision"])
            self.assertEqual("3m", review["corrected_label"])
            self.assertEqual("wrong source label", review["note"])

    def test_false_detection_keeps_original_label_and_clears_correction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ReviewStore(Path(temporary_directory) / "quality.sqlite")
            candidate = self.make_candidate()
            review = store.save(
                candidate,
                decision="false_detection",
                corrected_label="3m",
                note="classifier missed a valid crop",
            )
            self.assertEqual("false_detection", review["decision"])
            self.assertIsNone(review["corrected_label"])

    def test_unusable_and_background_are_persisted_as_exclusion_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ReviewStore(Path(temporary_directory) / "quality.sqlite")
            candidate = self.make_candidate()
            for decision in ("unusable_crop", "background"):
                review = store.save(
                    candidate,
                    decision=decision,
                    corrected_label=None,
                    note="",
                )
                self.assertEqual(decision, review["decision"])
                self.assertIsNone(review["corrected_label"])

    def test_delete_restores_unreviewed_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ReviewStore(Path(temporary_directory) / "quality.sqlite")
            candidate = self.make_candidate()
            store.save(
                candidate,
                decision="false_detection",
                corrected_label=None,
                note="",
            )
            self.assertTrue(store.delete(candidate.crop_id))
            self.assertIsNone(store.get(candidate.crop_id))


if __name__ == "__main__":
    unittest.main()
