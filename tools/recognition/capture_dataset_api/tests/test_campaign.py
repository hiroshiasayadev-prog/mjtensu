from __future__ import annotations

import unittest
from collections import Counter

from tools.recognition.capture_dataset_api.campaign import (
    CAMPAIGN_ID,
    LAYOUT_COUNT,
    VISIBLE_TILE_CODES,
    generate_campaign,
)
from tools.recognition.capture_dataset_api.tile_catalog_campaign import (
    CAMPAIGN_ID as TILE_CATALOG_CAMPAIGN_ID,
    HONORS,
    MANZU,
    PINZU,
    SOUZU,
    generate_tile_catalog_campaign,
)


class CampaignTest(unittest.TestCase):
    def test_initial_campaign_contract(self) -> None:
        campaign = generate_campaign()
        self.assertEqual("initial-120", CAMPAIGN_ID)
        self.assertEqual(CAMPAIGN_ID, campaign["id"])
        self.assertEqual(LAYOUT_COUNT, len(campaign["layouts"]))
        self.assertEqual(120, len(campaign["tasks"]))
        self.assertNotIn("repetitions", campaign)
        self.assertEqual(64, len(campaign["definitionSha256"]))

        coverage = Counter(campaign["coverage"])
        for tile_code in VISIBLE_TILE_CODES:
            self.assertGreaterEqual(coverage[tile_code], 3, tile_code)
        self.assertGreaterEqual(coverage["back"], 3)

    def test_each_layout_obeys_physical_inventory(self) -> None:
        campaign = generate_campaign()
        for layout in campaign["layouts"]:
            inventory: Counter[str] = Counter()
            inventory.update(slot["tile"] for slot in layout["hand"])
            inventory.update(slot["tile"] for slot in layout["dora"]["visible"])
            inventory.update(slot["tile"] for slot in layout["dora"]["ura"])
            for meld in layout["melds"]:
                inventory.update(slot["tile"] for slot in meld["tiles"])

            for tile_code, count in inventory.items():
                if tile_code in {"5m", "5p", "5s"}:
                    capacity = 3
                elif tile_code in {"red5m", "red5p", "red5s"}:
                    capacity = 1
                else:
                    capacity = 4
                self.assertLessEqual(count, capacity, f"{layout['id']}: {tile_code}")

    def test_tile_catalog_campaign_contract(self) -> None:
        campaign = generate_tile_catalog_campaign()
        self.assertEqual("tile-catalog-warm-4-v2", TILE_CATALOG_CAMPAIGN_ID)
        self.assertEqual(TILE_CATALOG_CAMPAIGN_ID, campaign["id"])
        self.assertEqual(1, len(campaign["layouts"]))
        self.assertEqual(4, len(campaign["tasks"]))
        self.assertEqual(64, len(campaign["definitionSha256"]))

        expected_tiles = [*MANZU, *PINZU, *SOUZU, *HONORS]
        self.assertEqual(37, len(expected_tiles))
        self.assertEqual(37, len(set(expected_tiles)))
        for task_order, task in enumerate(campaign["tasks"]):
            self.assertEqual(task_order, task["taskOrder"])
            self.assertEqual(task_order, task["repetition"])
            self.assertEqual(0, task["expected"]["hand"])
            self.assertEqual(0, task["expected"]["dora"])
            self.assertEqual(37, task["expected"]["meld"])
            self.assertEqual(37, sum(task["expected"].values()))
            self.assertEqual([10, 10, 10, 7], [len(meld["tiles"]) for meld in task["melds"]])
            self.assertEqual("warm", task["environment"]["lighting"])
            self.assertEqual(4, len(task["catalogRows"]))

    def test_task_order_and_expected_counts(self) -> None:
        campaign = generate_campaign()
        for task_order, task in enumerate(campaign["tasks"]):
            self.assertEqual(task_order, task["taskOrder"])
            self.assertNotIn("repetition", task)
            self.assertEqual(len(task["hand"]), task["expected"]["hand"])
            self.assertEqual(
                len(task["dora"]["visible"]) + len(task["dora"]["ura"]),
                task["expected"]["dora"],
            )
            self.assertEqual(
                sum(len(meld["tiles"]) for meld in task["melds"]),
                task["expected"]["meld"],
            )


if __name__ == "__main__":
    unittest.main()
