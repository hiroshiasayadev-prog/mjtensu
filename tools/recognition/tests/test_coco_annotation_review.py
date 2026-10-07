from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.recognition.coco_annotation_review.server import ReviewStore, _sha256


class CocoAnnotationReviewStoreTests(unittest.TestCase):
    def make_store(self, root: Path) -> tuple[ReviewStore, Path]:
        dataset = root / "data/coco_mahjong_jp_v2"
        split = dataset / "train"
        split.mkdir(parents=True)
        Image.new("RGB", (20, 20), "white").save(split / "a.jpg")
        source = split / "_annotations.coco.json"
        source.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}],
                    "categories": [
                        {"id": 1, "name": "0", "supercategory": "mahjong-tiles"},
                        {"id": 2, "name": "1", "supercategory": "mahjong-tiles"},
                        {"id": 13, "name": "1m", "supercategory": "mahjong-tiles"},
                        {"id": 27, "name": "2m", "supercategory": "mahjong-tiles"},
                    ],
                    "annotations": [
                        {"id": 10, "image_id": 1, "category_id": 1, "bbox": [2, 3, 5, 7]}
                    ],
                }
            ),
            encoding="utf-8",
        )
        source_sha = _sha256(source)
        base = {
            "dataset_id": "coco_mahjong_jp_v2",
            "split": "train",
            "source_annotation_sha256": source_sha,
            "annotation_id": 10,
            "image_id": 1,
            "image_file": "a.jpg",
            "category_id": 1,
            "raw_category_name": "0",
            "annotated_label": "1m",
            "annotated_base_label": "1m",
            "bbox": [2.0, 3.0, 5.0, 7.0],
            "crop_pixel_box": [2, 3, 7, 10],
            "crop_width": 5,
            "crop_height": 7,
            "crop_clipped": False,
            "base_predicted_label": "2m",
            "base_confidence": 0.99,
            "base_margin": 0.90,
            "base_second_label": "1m",
            "base_second_confidence": 0.01,
            "red_five_predicted": None,
            "red_five_confidence": None,
            "red_five_margin": None,
            "predicted_label": "2m",
            "confidence": 0.99,
            "margin": 0.90,
            "review_status": "high_confidence_mismatch",
        }
        audit_path = root / "flagged.jsonl"
        records = [
            base,
            {**base, "dataset_id": "coco_mahjong", "annotation_id": 11},
            {**base, "annotation_id": 12, "review_status": "low_confidence_match"},
        ]
        audit_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        store = ReviewStore(
            audit_jsonl=audit_path,
            secondary_audit_jsonl=None,
            dataset_root=dataset,
            state_path=root / "review_state.json",
            export_root=root / "corrected",
        )
        return store, source

    def test_only_jp_mismatches_enter_review_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, _ = self.make_store(Path(directory))
            self.assertEqual([item.key for item in store.candidates], ["train:10"])
            self.assertEqual(store.bootstrap()["decision_counts"]["unreviewed"], 1)

    def test_review_persists_and_image_is_confined_to_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, _ = self.make_store(root)
            store.save_review("train:10", {"decision": "keep_annotation", "note": "classifier wrong"})
            reopened = ReviewStore(
                audit_jsonl=root / "flagged.jsonl",
                secondary_audit_jsonl=None,
                dataset_root=root / "data/coco_mahjong_jp_v2",
                state_path=root / "review_state.json",
                export_root=root / "corrected",
            )
            self.assertEqual(reopened.reviews["train:10"]["decision"], "keep_annotation")
            self.assertEqual(reopened.image_path("train:10"), root / "data/coco_mahjong_jp_v2/train/a.jpg")

    def test_export_changes_copy_only_and_keeps_original_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, source = self.make_store(root)
            before = source.read_bytes()
            store.save_review(
                "train:10",
                {"decision": "change", "approved_label": "2m", "note": "visual confirmation"},
            )
            report = store.export_corrected()
            self.assertEqual(report["total_changes"], 1)
            self.assertEqual(source.read_bytes(), before)
            corrected = json.loads((root / "corrected/train_annotations.corrected.coco.json").read_text())
            self.assertEqual(corrected["annotations"][0]["category_id"], 2)
            manifest = (root / "corrected/train_approved_corrections.csv").read_text()
            self.assertIn("visual confirmation", manifest)
            self.assertIn(",2m,", manifest)

    def test_secondary_audit_marks_dual_high_same_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, _ = self.make_store(root)
            primary = store.candidates[0].record
            secondary_path = root / "secondary.jsonl"
            secondary_path.write_text(
                json.dumps({
                    **primary,
                    "predicted_label": "2m",
                    "review_status": "high_confidence_mismatch",
                    "confidence": 0.98,
                    "margin": 0.88,
                }) + "\n",
                encoding="utf-8",
            )
            consensus = ReviewStore(
                audit_jsonl=root / "flagged.jsonl",
                secondary_audit_jsonl=secondary_path,
                dataset_root=root / "data/coco_mahjong_jp_v2",
                state_path=root / "review_state_consensus.json",
                export_root=root / "corrected_consensus",
            )
            candidate = consensus.candidates[0].record
            self.assertEqual(candidate["consensus_tier"], "dual_high_same")
            self.assertEqual(candidate["secondary_predicted_label"], "2m")
            self.assertEqual(consensus.bootstrap()["consensus_counts"]["dual_high_same"], 1)

    def test_remove_annotation_is_exported_without_touching_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, source = self.make_store(root)
            before = source.read_bytes()
            store.save_review(
                "train:10",
                {"decision": "remove_annotation", "note": "bbox is not a tile"},
            )
            report = store.export_corrected()
            self.assertEqual(report["total_removals"], 1)
            self.assertEqual(source.read_bytes(), before)
            corrected = json.loads((root / "corrected/train_annotations.corrected.coco.json").read_text())
            self.assertEqual(corrected["annotations"], [])
            manifest = (root / "corrected/train_approved_corrections.csv").read_text()
            self.assertIn("remove_annotation", manifest)
            self.assertIn("__REMOVE__", manifest)

    def test_change_rejects_non_jp_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, _ = self.make_store(Path(directory))
            with self.assertRaisesRegex(ValueError, "approved_label"):
                store.save_review(
                    "train:10",
                    {"decision": "change", "approved_label": "flower"},
                )


if __name__ == "__main__":
    unittest.main()
