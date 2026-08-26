from __future__ import annotations

import unittest

from PIL import Image

from tools.recognition.composite_capture_dataset_tool.composer import (
    compose_capture_image,
)
from tools.recognition.composite_capture_dataset_tool.models import Rect, RegionSelection


class ComposeCaptureImageTest(unittest.TestCase):
    def test_composes_enabled_region_and_transforms_bbox(self) -> None:
        source = Image.new("RGB", (170, 40), (200, 10, 20))
        annotations = [
            {
                "id": 11,
                "image_id": 1,
                "category_id": 99,
                "bbox": [17, 4, 34, 8],
                "area": 272,
                "iscrowd": 0,
                "segmentation": [[17, 4, 51, 4, 51, 12, 17, 12]],
            }
        ]
        result = compose_capture_image(
            source,
            {
                "completed_hand": RegionSelection(
                    "completed_hand",
                    Rect(0, 0, 170, 40),
                    0,
                )
            },
            annotations,
        )

        self.assertEqual(result.image.size, (320, 320))
        self.assertEqual(result.image.getpixel((100, 30)), (200, 10, 20))
        self.assertEqual(result.image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.image.getpixel((100, 100)), (0, 0, 0))
        self.assertEqual(len(result.annotations), 1)
        transformed = result.annotations[0]
        self.assertEqual(transformed.source_annotation_id, 11)
        self.assertEqual(transformed.region_key, "completed_hand")
        self.assertAlmostEqual(transformed.bbox.x, 37.6)
        self.assertAlmostEqual(transformed.bbox.y, 7.2)
        self.assertAlmostEqual(transformed.bbox.width, 61.2)
        self.assertAlmostEqual(transformed.bbox.height, 14.4)
        self.assertEqual(
            result.stats_by_region["completed_hand"].retained_annotations,
            1,
        )

    def test_center_policy_clips_retained_bbox(self) -> None:
        source = Image.new("RGB", (170, 40), "white")
        annotations = [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [-10, 0, 40, 20],
            }
        ]
        result = compose_capture_image(
            source,
            {
                "completed_hand": RegionSelection(
                    "completed_hand",
                    Rect(0, 0, 170, 40),
                    0,
                )
            },
            annotations,
            annotation_selection_policy="center",
        )
        self.assertEqual(len(result.annotations), 1)
        self.assertEqual(result.annotations[0].bbox, Rect(7, 0, 54, 36))

    def test_excludes_bbox_when_crop_retains_sixty_percent_or_less(self) -> None:
        source = Image.new("RGB", (200, 40), "white")
        annotations = [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [146, 5, 40, 10],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [145, 20, 40, 10],
            },
        ]
        result = compose_capture_image(
            source,
            {
                "completed_hand": RegionSelection(
                    "completed_hand",
                    Rect(0, 0, 170, 40),
                    0,
                )
            },
            annotations,
            min_retained_area_ratio=0.6,
        )

        self.assertEqual(
            [annotation.source_annotation_id for annotation in result.annotations],
            [2],
        )

    def test_same_source_annotation_can_appear_in_multiple_regions(self) -> None:
        source = Image.new("RGB", (170, 170), "white")
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10]}
        ]
        result = compose_capture_image(
            source,
            {
                "completed_hand": RegionSelection(
                    "completed_hand", Rect(0, 0, 170, 40), 0
                ),
                "melds": RegionSelection("melds", Rect(0, 0, 170, 170), 0),
            },
            annotations,
        )
        self.assertEqual(len(result.annotations), 2)
        self.assertEqual(
            {annotation.region_key for annotation in result.annotations},
            {"completed_hand", "melds"},
        )

    def test_rotation_requires_swapped_source_aspect(self) -> None:
        source = Image.new("RGB", (40, 170), "white")
        result = compose_capture_image(
            source,
            {
                "completed_hand": RegionSelection(
                    "completed_hand", Rect(0, 0, 40, 170), 90
                )
            },
            [],
        )
        self.assertEqual(result.image.getpixel((100, 30)), (255, 255, 255))

    def test_empty_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            compose_capture_image(Image.new("RGB", (10, 10)), {}, [])


if __name__ == "__main__":
    unittest.main()
