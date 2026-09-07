from __future__ import annotations

import math
import unittest

from PIL import Image

from tools.recognition.build_nanodet_region_rotation_augmented_dataset import (
    RegionRect,
    RegionTransform,
    angle_dependent_scale,
    annotation_polygon,
    group_annotations_by_region,
    minimal_translation_to_fit,
    plan_region_transform,
    point_bounds,
    rotate_point,
    scale_rotate_point,
    transform_region_image,
    transformed_annotation,
)


class NanoDetRegionRotationAugmentationTest(unittest.TestCase):
    def test_angle_dependent_scale_uses_quadratic_max_ten_percent_shrink(self) -> None:
        self.assertAlmostEqual(
            angle_dependent_scale(
                0.0, max_rotation_deg=12.0, max_shrink_fraction=0.10
            ),
            1.0,
        )
        self.assertAlmostEqual(
            angle_dependent_scale(
                3.0, max_rotation_deg=12.0, max_shrink_fraction=0.10
            ),
            0.99375,
        )
        self.assertAlmostEqual(
            angle_dependent_scale(
                6.0, max_rotation_deg=12.0, max_shrink_fraction=0.10
            ),
            0.975,
        )
        self.assertAlmostEqual(
            angle_dependent_scale(
                9.0, max_rotation_deg=12.0, max_shrink_fraction=0.10
            ),
            0.94375,
        )
        self.assertAlmostEqual(
            angle_dependent_scale(
                12.0, max_rotation_deg=12.0, max_shrink_fraction=0.10
            ),
            0.90,
        )
        self.assertAlmostEqual(
            angle_dependent_scale(
                -12.0, max_rotation_deg=12.0, max_shrink_fraction=0.10
            ),
            0.90,
        )

    def test_rotated_box_uses_enclosing_axis_aligned_bbox(self) -> None:
        source = {
            "id": 7,
            "image_id": 3,
            "category_id": 1,
            "bbox": [20.0, 20.0, 20.0, 10.0],
            "area": 200.0,
            "iscrowd": 0,
            "region": "completed_hand",
        }
        source_polygon = annotation_polygon(source)
        center_x = 30.0
        center_y = 25.0
        rotated = tuple(
            rotate_point(
                x,
                y,
                center_x=center_x,
                center_y=center_y,
                angle_deg=45.0,
            )
            for x, y in source_polygon
        )
        annotation = transformed_annotation(
            source,
            rotated,
            image_id=1,
            annotation_id=1,
            region_key="completed_hand",
            transform=RegionTransform(
                angle_deg=45.0,
                scale=1.0,
                center_x=center_x,
                center_y=center_y,
                translate_x=0.0,
                translate_y=0.0,
                resample_count=0,
            ),
        )

        min_x, min_y, max_x, max_y = point_bounds(rotated)
        self.assertAlmostEqual(annotation["bbox"][0], min_x)
        self.assertAlmostEqual(annotation["bbox"][1], min_y)
        self.assertAlmostEqual(annotation["bbox"][2], max_x - min_x)
        self.assertAlmostEqual(annotation["bbox"][3], max_y - min_y)
        self.assertGreater(annotation["bbox"][2], source["bbox"][2])
        self.assertGreater(annotation["bbox"][3], source["bbox"][3])

    def test_minimal_translation_moves_union_inside_without_clipping(self) -> None:
        points = ((-3.5, 5.0), (8.0, 5.0), (8.0, 18.0), (-3.5, 18.0))

        translation = minimal_translation_to_fit(points, width=20, height=20)

        self.assertIsNotNone(translation)
        dx, dy = translation or (0.0, 0.0)
        self.assertAlmostEqual(dx, 3.5)
        self.assertAlmostEqual(dy, 0.0)
        moved = [(x + dx, y + dy) for x, y in points]
        min_x, min_y, max_x, max_y = point_bounds(moved)
        self.assertGreaterEqual(min_x, 0.0)
        self.assertGreaterEqual(min_y, 0.0)
        self.assertLessEqual(max_x, 20.0)
        self.assertLessEqual(max_y, 20.0)

    def test_impossible_rotated_extent_is_rejected(self) -> None:
        points = ((-2.0, 0.0), (23.0, 0.0), (23.0, 10.0), (-2.0, 10.0))

        self.assertIsNone(minimal_translation_to_fit(points, width=20, height=20))

    def test_plan_uses_bbox_union_center_and_keeps_all_points_inside(self) -> None:
        polygons = (
            ((1.0, 2.0), (11.0, 2.0), (11.0, 12.0), (1.0, 12.0)),
            ((75.0, 8.0), (95.0, 8.0), (95.0, 28.0), (75.0, 28.0)),
        )

        plan = plan_region_transform(
            polygons,
            region_width=100,
            region_height=40,
            seed=42,
            sample_key=("image.png", 0, "completed_hand"),
            max_rotation_deg=12.0,
            max_shrink_fraction=0.10,
            max_resamples=32,
        )

        self.assertAlmostEqual(plan.transform.center_x, 48.0)
        self.assertAlmostEqual(plan.transform.center_y, 15.0)
        for polygon in plan.transformed_polygons:
            for x, y in polygon:
                self.assertGreaterEqual(x, -1e-6)
                self.assertGreaterEqual(y, -1e-6)
                self.assertLessEqual(x, 100.0 + 1e-6)
                self.assertLessEqual(y, 40.0 + 1e-6)

    def test_explicit_region_is_validated_against_fixed_destination(self) -> None:
        regions = {
            "completed_hand": RegionRect("completed_hand", 7, 0, 306, 72),
            "dora_indicators": RegionRect("dora_indicators", 7, 74, 306, 72),
            "melds": RegionRect("melds", 74, 148, 172, 172),
        }
        annotation = {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 5.0, 20.0, 30.0],
            "region": "completed_hand",
        }

        grouped = group_annotations_by_region([annotation], regions)

        self.assertEqual(grouped, {"completed_hand": [annotation]})

    def test_region_can_be_inferred_for_composite_replay_annotation(self) -> None:
        regions = {
            "completed_hand": RegionRect("completed_hand", 7, 0, 306, 72),
            "dora_indicators": RegionRect("dora_indicators", 7, 74, 306, 72),
            "melds": RegionRect("melds", 74, 148, 172, 172),
        }
        annotation = {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [90.0, 180.0, 20.0, 30.0],
        }

        grouped = group_annotations_by_region([annotation], regions)

        self.assertEqual(grouped, {"melds": [annotation]})

    def test_image_transform_uses_same_rotation_direction_as_point_transform(self) -> None:
        image = Image.new("RGB", (21, 21), (0, 0, 0))
        # A small bright block to the right of center.
        for y in range(9, 12):
            for x in range(15, 18):
                image.putpixel((x, y), (255, 255, 255))
        transform = RegionTransform(
            angle_deg=90.0,
            scale=1.0,
            center_x=10.0,
            center_y=10.0,
            translate_x=0.0,
            translate_y=0.0,
            resample_count=0,
        )

        result = transform_region_image(image, transform)
        expected_x, expected_y = rotate_point(
            16.0,
            10.0,
            center_x=10.0,
            center_y=10.0,
            angle_deg=90.0,
        )

        # Bilinear/image-center conventions can shift the peak by one pixel, but it
        # must land in the same transformed neighborhood as the polygon transform.
        bright = []
        for y in range(result.height):
            for x in range(result.width):
                if result.getpixel((x, y))[0] >= 200:
                    bright.append((x, y))
        self.assertTrue(bright)
        center_x = sum(x for x, _y in bright) / len(bright)
        center_y = sum(y for _x, y in bright) / len(bright)
        self.assertLessEqual(abs(center_x - expected_x), 1.0)
        self.assertLessEqual(abs(center_y - expected_y), 1.0)

    def test_image_transform_uses_same_scale_as_point_transform(self) -> None:
        image = Image.new("RGB", (21, 21), (0, 0, 0))
        for y in range(9, 12):
            for x in range(15, 18):
                image.putpixel((x, y), (255, 255, 255))
        transform = RegionTransform(
            angle_deg=0.0,
            scale=0.5,
            center_x=10.0,
            center_y=10.0,
            translate_x=0.0,
            translate_y=0.0,
            resample_count=0,
        )

        result = transform_region_image(image, transform)
        expected_x, expected_y = scale_rotate_point(
            16.0,
            10.0,
            center_x=10.0,
            center_y=10.0,
            angle_deg=0.0,
            scale=0.5,
        )
        bright = []
        for y in range(result.height):
            for x in range(result.width):
                if result.getpixel((x, y))[0] >= 150:
                    bright.append((x, y))
        self.assertTrue(bright)
        center_x = sum(x for x, _y in bright) / len(bright)
        center_y = sum(y for _x, y in bright) / len(bright)
        self.assertLessEqual(abs(center_x - expected_x), 1.0)
        self.assertLessEqual(abs(center_y - expected_y), 1.0)

    def test_image_transform_fills_out_of_bounds_background_black(self) -> None:
        image = Image.new("RGB", (21, 21), (255, 255, 255))
        transform = RegionTransform(
            angle_deg=45.0,
            scale=1.0,
            center_x=10.0,
            center_y=10.0,
            translate_x=0.0,
            translate_y=0.0,
            resample_count=0,
        )

        result = transform_region_image(image, transform)

        # A 45-degree rotation requires source pixels outside the original region
        # at the output corners. Those pixels must be black, never mirrored content.
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((20, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((0, 20)), (0, 0, 0))
        self.assertEqual(result.getpixel((20, 20)), (0, 0, 0))
        self.assertEqual(result.getpixel((10, 10)), (255, 255, 255))

    def test_deterministic_plan_is_reproducible(self) -> None:
        polygons = (((20.0, 10.0), (40.0, 10.0), (40.0, 30.0), (20.0, 30.0)),)
        kwargs = dict(
            region_width=100,
            region_height=50,
            seed=42,
            sample_key=("same.png", 0, "completed_hand"),
            max_rotation_deg=12.0,
            max_shrink_fraction=0.10,
            max_resamples=32,
        )

        first = plan_region_transform(polygons, **kwargs)
        second = plan_region_transform(polygons, **kwargs)

        self.assertEqual(first, second)
        self.assertTrue(math.isfinite(first.transform.angle_deg))


if __name__ == "__main__":
    unittest.main()
