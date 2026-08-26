from __future__ import annotations

import unittest

from tools.recognition.build_detector_crop_dataset import (
    GroundTruth,
    PreparedDetection,
    match_detection_to_ground_truth,
    polygon_area,
    rect_min_area_overlap,
    rotated_rectangle_polygon,
    suggest_state,
    suppress_near_duplicate_detections,
)


class BuildDetectorCropDatasetTest(unittest.TestCase):
    def gt(
        self,
        box_id: str,
        label: str,
        *,
        center_x: float,
        center_y: float,
        width: float = 20.0,
        height: float = 30.0,
        angle_deg: float = 0.0,
    ) -> GroundTruth:
        return GroundTruth(
            box_id=box_id,
            label=label,
            source_label=label,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            angle_deg=angle_deg,
        )

    def test_rotated_rectangle_preserves_area(self) -> None:
        ground_truth = self.gt(
            "gt-1",
            "3m",
            center_x=50.0,
            center_y=60.0,
            width=22.0,
            height=37.0,
            angle_deg=31.0,
        )
        self.assertAlmostEqual(
            22.0 * 37.0,
            polygon_area(rotated_rectangle_polygon(ground_truth)),
            places=6,
        )

    def test_single_gt_suggestion_for_one_well_contained_tile(self) -> None:
        ground_truth = self.gt("gt-1", "3m", center_x=50.0, center_y=60.0)
        matches = match_detection_to_ground_truth((39.0, 44.0, 22.0, 32.0), [ground_truth])
        self.assertEqual("single_gt", suggest_state(matches))
        self.assertGreater(matches[0].gt_coverage, 0.95)
        self.assertGreater(matches[0].detection_coverage, 0.80)

    def test_background_suggestion_when_detection_misses_all_tiles(self) -> None:
        ground_truth = self.gt("gt-1", "3m", center_x=50.0, center_y=60.0)
        matches = match_detection_to_ground_truth((2.0, 2.0, 10.0, 10.0), [ground_truth])
        self.assertEqual("background", suggest_state(matches))

    def test_multi_gt_suggestion_when_one_detection_contains_two_tiles(self) -> None:
        left = self.gt("left", "3m", center_x=40.0, center_y=50.0)
        right = self.gt("right", "4m", center_x=62.0, center_y=50.0)
        matches = match_detection_to_ground_truth((28.0, 33.0, 46.0, 34.0), [left, right])
        self.assertEqual("multi_gt", suggest_state(matches))
        self.assertEqual(2, sum(match.gt_coverage >= 0.30 for match in matches))

    def test_partial_suggestion_for_heavily_clipped_tile(self) -> None:
        ground_truth = self.gt("gt-1", "3m", center_x=50.0, center_y=60.0)
        matches = match_detection_to_ground_truth((48.0, 45.0, 12.0, 30.0), [ground_truth])
        self.assertEqual("partial", suggest_state(matches))

    def test_min_area_overlap_treats_contained_box_as_full_overlap(self) -> None:
        self.assertAlmostEqual(
            1.0,
            rect_min_area_overlap((10.0, 10.0, 20.0, 30.0), (8.0, 8.0, 25.0, 35.0)),
        )

    def test_duplicate_suppression_keeps_higher_detector_confidence(self) -> None:
        low = PreparedDetection(
            detection_index=3,
            confidence=0.61,
            region="completed_hand",
            local_rect=(10.0, 10.0, 20.0, 30.0),
        )
        high = PreparedDetection(
            detection_index=7,
            confidence=0.92,
            region="completed_hand",
            local_rect=(11.0, 11.0, 20.0, 30.0),
        )
        kept, suppressed = suppress_near_duplicate_detections([low, high], threshold=0.80)
        self.assertEqual([7], [item.detection_index for item in kept])
        self.assertEqual(1, len(suppressed))
        self.assertEqual(7, suppressed[0].winner.detection_index)
        self.assertEqual(3, suppressed[0].removed.detection_index)
        self.assertGreaterEqual(suppressed[0].overlap_ratio, 0.80)

    def test_duplicate_suppression_does_not_cross_regions(self) -> None:
        hand = PreparedDetection(1, 0.90, "completed_hand", (10.0, 10.0, 20.0, 30.0))
        dora = PreparedDetection(2, 0.80, "dora_indicators", (10.0, 10.0, 20.0, 30.0))
        kept, suppressed = suppress_near_duplicate_detections([hand, dora], threshold=0.80)
        self.assertEqual(2, len(kept))
        self.assertEqual([], suppressed)

    def test_duplicate_suppression_keeps_one_winner_per_connected_group(self) -> None:
        detections = [
            # duplicate group 1: two bboxes -> winner index 1
            PreparedDetection(0, 0.70, "dora_indicators", (10.0, 10.0, 20.0, 30.0)),
            PreparedDetection(1, 0.95, "dora_indicators", (11.0, 10.0, 20.0, 30.0)),
            # duplicate group 2: three bboxes -> winner index 4
            PreparedDetection(2, 0.80, "dora_indicators", (80.0, 10.0, 20.0, 30.0)),
            PreparedDetection(3, 0.85, "dora_indicators", (81.0, 10.0, 20.0, 30.0)),
            PreparedDetection(4, 0.97, "dora_indicators", (82.0, 10.0, 20.0, 30.0)),
        ]
        kept, suppressed = suppress_near_duplicate_detections(detections, threshold=0.80)
        self.assertEqual([1, 4], [item.detection_index for item in kept])
        self.assertEqual([0, 2, 3], [item.removed.detection_index for item in suppressed])
        self.assertEqual(
            [1, 4, 4],
            [item.winner.detection_index for item in suppressed],
        )

    def test_duplicate_suppression_uses_transitive_connected_component(self) -> None:
        # A overlaps B >= 0.8, B overlaps C >= 0.8, but A and C do not.
        # They are still one duplicate cluster and must yield only one winner.
        a = PreparedDetection(0, 0.70, "completed_hand", (0.0, 0.0, 20.0, 30.0))
        b = PreparedDetection(1, 0.99, "completed_hand", (4.0, 0.0, 20.0, 30.0))
        c = PreparedDetection(2, 0.80, "completed_hand", (8.0, 0.0, 20.0, 30.0))
        kept, suppressed = suppress_near_duplicate_detections([a, b, c], threshold=0.80)
        self.assertEqual([1], [item.detection_index for item in kept])
        self.assertEqual({0, 2}, {item.removed.detection_index for item in suppressed})
        self.assertTrue(all(item.winner.detection_index == 1 for item in suppressed))


if __name__ == "__main__":
    unittest.main()
