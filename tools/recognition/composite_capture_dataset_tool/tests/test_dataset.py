from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.recognition.composite_capture_dataset_tool.composer import CompositeResult
from tools.recognition.composite_capture_dataset_tool.dataset import OutputDatasetManager
from tools.recognition.composite_capture_dataset_tool.models import (
    Rect,
    RegionCompositeStats,
    RegionSelection,
    TransformedAnnotation,
)


class OutputDatasetManagerTest(unittest.TestCase):
    def test_save_and_resume_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "dataset"
            manager = OutputDatasetManager(output_directory)
            result = CompositeResult(
                image=Image.new("RGB", (320, 320), "black"),
                annotations=[
                    TransformedAnnotation(
                        source_annotation_id=77,
                        region_key="melds",
                        bbox=Rect(80, 160, 20, 30),
                        iscrowd=0,
                    )
                ],
                stats_by_region={
                    "melds": RegionCompositeStats(1, 0),
                },
            )
            source_image = {
                "id": 12,
                "file_name": "source/train/image.jpg",
                "width": 960,
                "height": 960,
            }
            selections = {
                "melds": RegionSelection("melds", Rect(10, 20, 100, 100), 0)
            }

            first = manager.save_composite(
                result,
                source_annotation_path=Path(temporary_directory) / "source.json",
                source_image=source_image,
                selections=selections,
                annotation_selection_policy="center",
            )
            self.assertEqual(first.image_id, 1)
            self.assertEqual(first.annotation_count, 1)
            self.assertTrue(Path(first.image_path).is_file())

            payload = json.loads(Path(first.annotations_path).read_text("utf-8"))
            self.assertEqual(payload["categories"][0]["id"], 1)
            self.assertEqual(payload["categories"][0]["name"], "mahjong_tile")
            self.assertEqual(payload["images"][0]["source_image_id"], 12)
            annotation = payload["annotations"][0]
            self.assertEqual(annotation["id"], 1)
            self.assertEqual(annotation["image_id"], 1)
            self.assertEqual(annotation["category_id"], 1)
            self.assertEqual(annotation["source_annotation_id"], 77)
            self.assertEqual(annotation["capture_region"], "melds")
            self.assertNotIn("segmentation", annotation)

            resumed = OutputDatasetManager(output_directory)
            second = resumed.save_composite(
                result,
                source_annotation_path=Path(temporary_directory) / "source.json",
                source_image=source_image,
                selections=selections,
                annotation_selection_policy="center",
            )
            self.assertEqual(second.image_id, 2)
            payload = json.loads(Path(second.annotations_path).read_text("utf-8"))
            self.assertEqual([image["id"] for image in payload["images"]], [1, 2])
            self.assertEqual(
                [annotation["id"] for annotation in payload["annotations"]],
                [1, 2],
            )

    def test_rebuild_existing_annotations_filters_sixty_percent_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            output_directory = temporary_root / "dataset"
            source_annotation_path = temporary_root / "source.json"
            source_annotation_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": 12,
                                "file_name": "source/train/image.jpg",
                                "width": 200,
                                "height": 40,
                            }
                        ],
                        "annotations": [
                            {
                                "id": 77,
                                "image_id": 12,
                                "category_id": 1,
                                "bbox": [10, 5, 20, 10],
                            },
                            {
                                "id": 78,
                                "image_id": 12,
                                "category_id": 1,
                                "bbox": [146, 5, 40, 10],
                            },
                        ],
                        "categories": [{"id": 1, "name": "mahjong_tile"}],
                    }
                ),
                encoding="utf-8",
            )

            manager = OutputDatasetManager(output_directory)
            result = CompositeResult(
                image=Image.new("RGB", (320, 320), "black"),
                annotations=[
                    TransformedAnnotation(
                        source_annotation_id=77,
                        region_key="completed_hand",
                        bbox=Rect(25, 9, 36, 18),
                        iscrowd=0,
                    ),
                    TransformedAnnotation(
                        source_annotation_id=78,
                        region_key="completed_hand",
                        bbox=Rect(269.8, 9, 43.2, 18),
                        iscrowd=0,
                    ),
                ],
                stats_by_region={
                    "completed_hand": RegionCompositeStats(2, 1),
                },
            )
            source_image = {
                "id": 12,
                "file_name": "source/train/image.jpg",
                "width": 200,
                "height": 40,
            }
            selections = {
                "completed_hand": RegionSelection(
                    "completed_hand",
                    Rect(0, 0, 170, 40),
                    0,
                )
            }
            saved = manager.save_composite(
                result,
                source_annotation_path=source_annotation_path,
                source_image=source_image,
                selections=selections,
                annotation_selection_policy="center",
                min_retained_area_ratio=0.0,
            )

            report = manager.rebuild_existing_annotations(
                min_retained_area_ratio=0.6,
            )

            self.assertEqual(report["status"], "updated")
            self.assertEqual(report["annotations_before"], 2)
            self.assertEqual(report["annotations_after"], 1)
            self.assertEqual(report["removed_annotations"], 1)
            self.assertTrue(Path(str(report["backup"])).is_file())
            payload = json.loads(Path(saved.annotations_path).read_text("utf-8"))
            self.assertEqual(
                [annotation["source_annotation_id"] for annotation in payload["annotations"]],
                [77],
            )
            self.assertEqual(payload["images"][0]["min_retained_area_ratio"], 0.6)

    def test_incompatible_existing_category_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            annotations_directory = output_directory / "annotations"
            annotations_directory.mkdir()
            (annotations_directory / "instances.json").write_text(
                json.dumps(
                    {
                        "images": [],
                        "annotations": [],
                        "categories": [{"id": 1, "name": "wrong"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                OutputDatasetManager(output_directory)


if __name__ == "__main__":
    unittest.main()
