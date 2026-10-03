from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from tools.recognition import audit_coco_mahjong_annotations as audit


class CategoryMappingTests(unittest.TestCase):
    def test_legacy_mapping_and_duplicate_id_are_preserved_semantically(self) -> None:
        payload = {
            "categories": [
                {"id": 1, "name": "circle_1"},
                {"id": 10, "name": "bamboo_1"},
                {"id": 19, "name": "character_1"},
                {"id": 34, "name": "white"},
                {"id": 34, "name": "white"},
                {"id": 37, "name": "circle_2"},
            ]
        }
        mapping, report = audit.build_category_map(payload, "coco_mahjong")
        self.assertEqual(mapping[1].semantic_label, "1p")
        self.assertEqual(mapping[10].semantic_label, "1s")
        self.assertEqual(mapping[19].semantic_label, "1m")
        self.assertEqual(mapping[34].semantic_label, "white")
        self.assertEqual(mapping[37].semantic_label, "2p")
        self.assertEqual(report["duplicate_category_ids"]["34"]["count"], 2)
        self.assertEqual(report["semantic_label_to_category_ids"]["2p"], [37])

    def test_duplicate_id_with_conflicting_semantics_is_rejected(self) -> None:
        payload = {
            "categories": [
                {"id": 34, "name": "white"},
                {"id": 34, "name": "red"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "Ambiguous duplicate category id 34"):
            audit.build_category_map(payload, "coco_mahjong")

    def test_jp_numeric_and_explicit_red_five_mapping(self) -> None:
        self.assertEqual(
            audit.normalize_category_label("coco_mahjong_jp_v2", "5"), "red5m"
        )
        self.assertEqual(
            audit.normalize_category_label("coco_mahjong_jp_v2", "15"), "red5p"
        )
        self.assertEqual(
            audit.normalize_category_label("coco_mahjong_jp_v2", "25"), "red5s"
        )
        self.assertEqual(
            audit.normalize_category_label("coco_mahjong_jp_v2", "5mr"), "red5m"
        )
        self.assertIsNone(
            audit.normalize_category_label("coco_mahjong_jp_v2", "mahjong-tiles")
        )


class CropExtractionTests(unittest.TestCase):
    def test_floor_ceil_bbox_crop_matches_detector_source_policy(self) -> None:
        pixels = np.zeros((5, 5, 3), dtype=np.uint8)
        for y in range(5):
            for x in range(5):
                pixels[y, x] = (x * 10, y * 20, x + y)
        image = Image.fromarray(pixels, mode="RGB")
        result = audit.extract_coco_crop(image, [1.2, 1.8, 2.1, 1.2])
        self.assertEqual(result.pixel_box, (1, 1, 4, 3))
        self.assertEqual(result.image.size, (3, 2))
        self.assertFalse(result.clipped)
        cropped = np.asarray(result.image)
        np.testing.assert_array_equal(cropped[0, 0], pixels[1, 1])
        np.testing.assert_array_equal(cropped[-1, -1], pixels[2, 3])

    def test_bbox_clipping_is_reported(self) -> None:
        image = Image.new("RGB", (4, 4), "white")
        result = audit.extract_coco_crop(image, [-0.2, 0.2, 2.0, 2.0])
        self.assertEqual(result.pixel_box, (0, 0, 2, 3))
        self.assertTrue(result.clipped)


class RuntimePreprocessingTests(unittest.TestCase):
    @staticmethod
    def literal_runtime_resample(source: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        source = np.asarray(source, dtype=np.uint8)
        source_height, source_width = source.shape
        output = np.zeros((target_height, target_width), dtype=np.uint8)
        scale_x = source_width / target_width
        scale_y = source_height / target_height
        for ty in range(target_height):
            sy = (ty + 0.5) * scale_y - 0.5
            min_y = math.ceil(sy - 3.0 + 1.0)
            max_y = math.floor(sy + 3.0)
            for tx in range(target_width):
                sx = (tx + 0.5) * scale_x - 0.5
                min_x = math.ceil(sx - 3.0 + 1.0)
                max_x = math.floor(sx + 3.0)
                weighted_sum = 0.0
                weight_sum = 0.0
                for y in range(min_y, max_y + 1):
                    cy = max(0, min(source_height - 1, y))
                    yw = audit._lanczos(sy - y)
                    for x in range(min_x, max_x + 1):
                        cx = max(0, min(source_width - 1, x))
                        weight = yw * audit._lanczos(sx - x)
                        weighted_sum += float(source[cy, cx]) * weight
                        weight_sum += weight
                value = 0.0 if weight_sum == 0.0 else weighted_sum / weight_sum
                output[ty, tx] = audit._clamp_byte(value)
        return output

    def test_optimized_resampler_matches_frontend_nested_kernel(self) -> None:
        source = np.asarray(
            [[0, 31, 255], [15, 127, 240], [2, 190, 17], [255, 64, 8]],
            dtype=np.uint8,
        )
        expected = self.literal_runtime_resample(source, 7, 9)
        actual = audit._runtime_resample_channel(source, 7, 9)
        np.testing.assert_array_equal(actual, expected)


class MismatchClassificationTests(unittest.TestCase):
    def test_high_confidence_mismatch_and_ambiguity_are_separate(self) -> None:
        classify = audit.classify_review_status
        kwargs = dict(
            strong_confidence=0.90,
            strong_margin=0.25,
            low_confidence=0.60,
            low_margin=0.10,
        )
        self.assertEqual(
            classify("3m", "4m", 0.96, 0.55, **kwargs),
            "high_confidence_mismatch",
        )
        self.assertEqual(
            classify("3m", "4m", 0.80, 0.30, **kwargs),
            "ambiguous_mismatch",
        )
        self.assertEqual(
            classify("3m", "3m", 0.55, 0.40, **kwargs),
            "low_confidence_match",
        )
        self.assertEqual(
            classify("3m", "3m", 0.95, 0.70, **kwargs),
            "match_confident",
        )


class CorrectionApplicationTests(unittest.TestCase):
    def test_only_explicitly_approved_rows_write_new_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_dir = root / "data/coco_mahjong/annotations"
            annotation_dir.mkdir(parents=True)
            source_path = annotation_dir / "instances_train2017.json"
            original = {
                "images": [{"id": 1, "file_name": "x.jpg", "width": 10, "height": 10}],
                "categories": [
                    {"id": 1, "name": "circle_1"},
                    {"id": 2, "name": "circle_2"},
                    {"id": 3, "name": "circle_3"},
                ],
                "annotations": [
                    {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]},
                    {"id": 11, "image_id": 1, "category_id": 1, "bbox": [5, 0, 5, 5]},
                ],
            }
            source_path.write_text(json.dumps(original), encoding="utf-8")
            source_sha256 = audit.sha256_file(source_path)
            manifest = root / "approved.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "approve", "approved_label", "review_note",
                        "dataset_id", "split", "source_annotation_sha256",
                        "annotation_id",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "approve": "yes", "approved_label": "2p", "review_note": "human checked",
                    "dataset_id": "coco_mahjong", "split": "train2017",
                    "source_annotation_sha256": source_sha256, "annotation_id": "10",
                })
                writer.writerow({
                    "approve": "", "approved_label": "3p", "review_note": "",
                    "dataset_id": "coco_mahjong", "split": "train2017", "annotation_id": "11",
                })

            output = root / "corrected.json"
            audit.run_apply_corrections(
                Namespace(
                    repository_root=root,
                    dataset_id="coco_mahjong",
                    split="train2017",
                    approved_manifest=manifest,
                    output_json=output,
                )
            )

            unchanged = json.loads(source_path.read_text(encoding="utf-8"))
            corrected = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(unchanged["annotations"][0]["category_id"], 1)
            self.assertEqual(corrected["annotations"][0]["category_id"], 2)
            self.assertEqual(corrected["annotations"][1]["category_id"], 1)
            sidecar = json.loads(
                output.with_suffix(".json.corrections.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["approved_change_count"], 1)

            manifest_text = manifest.read_text(encoding="utf-8")
            manifest.write_text(
                manifest_text.replace(source_sha256, "0" * 64),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source SHA-256"):
                audit.run_apply_corrections(
                    Namespace(
                        repository_root=root,
                        dataset_id="coco_mahjong",
                        split="train2017",
                        approved_manifest=manifest,
                        output_json=root / "stale-corrected.json",
                    )
                )


if __name__ == "__main__":
    unittest.main()
