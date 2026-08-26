from __future__ import annotations

import unittest

from tools.recognition.composite_capture_dataset_tool.geometry import (
    annotation_matches_crop,
    constrain_drag_rect,
    rotate_point_clockwise,
    transform_bbox,
)
from tools.recognition.composite_capture_dataset_tool.layout import REGION_SPECS
from tools.recognition.composite_capture_dataset_tool.models import Rect


class ConstrainDragRectTest(unittest.TestCase):
    def test_hand_selection_is_exact_17_by_4_multiple(self) -> None:
        rect = constrain_drag_rect((10, 20), (180, 60), (320, 240), (17, 4))
        self.assertEqual(rect, Rect(10, 20, 170, 40))

    def test_drag_up_and_left_stays_inside_image(self) -> None:
        rect = constrain_drag_rect((100, 100), (-20, -20), (120, 120), (17, 4))
        self.assertEqual(rect, Rect(15, 80, 85, 20))

    def test_returns_none_when_no_aspect_multiple_fits(self) -> None:
        rect = constrain_drag_rect((3, 3), (0, 0), (5, 5), (17, 4))
        self.assertIsNone(rect)


class RotationTest(unittest.TestCase):
    def test_clockwise_point_rotations(self) -> None:
        self.assertEqual(rotate_point_clockwise(2, 3, 10, 20, 0), (2, 3))
        self.assertEqual(rotate_point_clockwise(2, 3, 10, 20, 90), (17, 2))
        self.assertEqual(rotate_point_clockwise(2, 3, 10, 20, 180), (8, 17))
        self.assertEqual(rotate_point_clockwise(2, 3, 10, 20, 270), (3, 8))

    def test_unrotated_hand_bbox_transform(self) -> None:
        crop = Rect(100, 200, 170, 40)
        bbox = Rect(117, 204, 34, 8)
        destination = REGION_SPECS["completed_hand"].destination
        transformed = transform_bbox(bbox, crop, 0, destination)
        self.assertIsNotNone(transformed)
        assert transformed is not None
        self.assertAlmostEqual(transformed.x, 37.6)
        self.assertAlmostEqual(transformed.y, 7.2)
        self.assertAlmostEqual(transformed.width, 61.2)
        self.assertAlmostEqual(transformed.height, 14.4)

    def test_clockwise_90_degree_hand_bbox_transform(self) -> None:
        crop = Rect(10, 20, 40, 170)
        bbox = Rect(10, 20, 10, 20)
        destination = REGION_SPECS["completed_hand"].destination
        transformed = transform_bbox(bbox, crop, 90, destination)
        self.assertIsNotNone(transformed)
        assert transformed is not None
        self.assertAlmostEqual(transformed.x, 277.0)
        self.assertAlmostEqual(transformed.y, 0.0)
        self.assertAlmostEqual(transformed.width, 36.0)
        self.assertAlmostEqual(transformed.height, 18.0)

    def test_bbox_is_clipped_to_crop_before_transform(self) -> None:
        crop = Rect(10, 10, 10, 10)
        bbox = Rect(5, 5, 10, 10)
        destination = Rect(0, 0, 100, 100)
        transformed = transform_bbox(bbox, crop, 0, destination)
        self.assertEqual(transformed, Rect(0, 0, 50, 50))

    def test_non_uniform_resize_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "uniform resizing"):
            transform_bbox(
                Rect(0, 0, 1, 1),
                Rect(0, 0, 10, 10),
                0,
                Rect(0, 0, 20, 30),
            )


class AnnotationPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.crop = Rect(10, 10, 20, 20)

    def test_center_policy(self) -> None:
        self.assertTrue(
            annotation_matches_crop(Rect(5, 5, 20, 20), self.crop, "center")
        )
        self.assertFalse(
            annotation_matches_crop(Rect(0, 0, 10, 10), self.crop, "center")
        )

    def test_contained_policy(self) -> None:
        self.assertTrue(
            annotation_matches_crop(Rect(12, 12, 4, 4), self.crop, "contained")
        )
        self.assertFalse(
            annotation_matches_crop(Rect(5, 5, 20, 20), self.crop, "contained")
        )

    def test_intersect_policy(self) -> None:
        self.assertTrue(
            annotation_matches_crop(Rect(0, 0, 11, 11), self.crop, "intersect")
        )
        self.assertFalse(
            annotation_matches_crop(Rect(0, 0, 10, 10), self.crop, "intersect")
        )


if __name__ == "__main__":
    unittest.main()
