from __future__ import annotations

import unittest

from tools.recognition.composite_capture_dataset_tool.gui import (
    HAND_DORA_RANDOM_CROPS,
    MELDS_RANDOM_CROPS,
)
from tools.recognition.composite_capture_dataset_tool.layout import (
    COMPOSITE_SIZE,
    REGION_SPECS,
)
from tools.recognition.composite_capture_dataset_tool.models import Rect


class FixedLayoutTest(unittest.TestCase):
    def test_exact_adr_002_destinations(self) -> None:
        self.assertEqual(COMPOSITE_SIZE, (320, 320))
        self.assertEqual(
            REGION_SPECS["completed_hand"].destination,
            Rect(7, 0, 306, 72),
        )
        self.assertEqual(
            REGION_SPECS["dora_indicators"].destination,
            Rect(7, 74, 306, 72),
        )
        self.assertEqual(
            REGION_SPECS["melds"].destination,
            Rect(74, 148, 172, 172),
        )

    def test_regions_do_not_overlap(self) -> None:
        destinations = [spec.destination for spec in REGION_SPECS.values()]
        for index, first in enumerate(destinations):
            for second in destinations[index + 1 :]:
                self.assertIsNone(first.intersection(second))

    def test_rotation_swaps_non_square_source_aspect(self) -> None:
        hand = REGION_SPECS["completed_hand"]
        self.assertEqual(hand.source_aspect_for_rotation(0), (17, 4))
        self.assertEqual(hand.source_aspect_for_rotation(90), (4, 17))
        self.assertEqual(hand.source_aspect_for_rotation(180), (17, 4))
        self.assertEqual(hand.source_aspect_for_rotation(270), (4, 17))

    def test_random_crop_presets_match_expected_aspects_and_source_size(self) -> None:
        source_bounds = Rect(0, 0, 960, 960)
        self.assertGreaterEqual(len(HAND_DORA_RANDOM_CROPS), 2)
        for crop in HAND_DORA_RANDOM_CROPS:
            self.assertTrue(source_bounds.contains_rect(crop))
            self.assertEqual(crop.width * 4, crop.height * 17)
        for crop in MELDS_RANDOM_CROPS:
            self.assertTrue(source_bounds.contains_rect(crop))
            self.assertEqual(crop.width, crop.height)


if __name__ == "__main__":
    unittest.main()
