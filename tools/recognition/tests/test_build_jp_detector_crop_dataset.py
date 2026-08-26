from __future__ import annotations

import unittest

from tools.recognition.build_jp_detector_crop_dataset import (
    build_ground_truths,
    scale_detection_rect,
    select_images,
)


class BuildJpDetectorCropDatasetTest(unittest.TestCase):
    def test_scale_detection_rect_maps_320_coordinates_back_to_source(self) -> None:
        self.assertEqual(
            (20.0, 10.0, 40.0, 20.0),
            scale_detection_rect(
                10.0,
                10.0,
                30.0,
                30.0,
                source_width=640,
                source_height=320,
            ),
        )

    def test_select_images_is_deterministic_and_limit_is_prefix_of_same_order(self) -> None:
        images = [
            {"id": index, "file_name": f"{index}.jpg"}
            for index in range(20)
        ]
        first = select_images(images, split="train", seed=42, image_limit=5)
        second = select_images(list(reversed(images)), split="train", seed=42, image_limit=10)
        self.assertEqual(
            [item["id"] for item in first],
            [item["id"] for item in second[:5]],
        )

    def test_ground_truth_numeric_red_five_is_folded_to_base_five(self) -> None:
        # Mahjong-jp numeric class 5 is red5m in the verified mapping.
        ground_truths = build_ground_truths(
            [
                {
                    "id": 123,
                    "category_id": 9,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                }
            ],
            category_names={9: "5"},
        )
        self.assertEqual(1, len(ground_truths))
        self.assertEqual("5m", ground_truths[0].label)
        self.assertEqual("red5m", ground_truths[0].source_label)
        self.assertEqual(25.0, ground_truths[0].center_x)
        self.assertEqual(40.0, ground_truths[0].center_y)


if __name__ == "__main__":
    unittest.main()
