from __future__ import annotations

import unittest

from tools.recognition.detector_duplicate_groups import (
    DetectorCandidate,
    build_duplicate_plan,
)


def candidate(
    candidate_id: str,
    *,
    capture: str = "cap-1",
    region: str = "dora_indicators",
    index: int,
    confidence: float,
    x: float,
) -> DetectorCandidate:
    return DetectorCandidate(
        candidate_id=candidate_id,
        capture_id=capture,
        region=region,
        detection_index=index,
        confidence=confidence,
        bbox_x=x,
        bbox_y=10.0,
        bbox_width=20.0,
        bbox_height=30.0,
    )


class DetectorDuplicateGroupsTest(unittest.TestCase):
    def test_two_duplicate_clusters_in_one_shot_leave_two_winners(self) -> None:
        plan = build_duplicate_plan(
            [
                candidate("a", index=0, confidence=0.70, x=10.0),
                candidate("b", index=1, confidence=0.95, x=11.0),
                candidate("c", index=2, confidence=0.80, x=80.0),
                candidate("d", index=3, confidence=0.85, x=81.0),
                candidate("e", index=4, confidence=0.97, x=82.0),
            ],
            threshold=0.80,
        )
        self.assertEqual({"b", "e"}, set(plan.winner_candidate_ids))
        self.assertEqual({"a", "c", "d"}, set(plan.loser_candidate_ids))
        self.assertEqual(2, len(plan.clusters))
        self.assertEqual([2, 3], sorted(len(cluster.members) for cluster in plan.clusters))

    def test_transitive_overlap_is_one_cluster(self) -> None:
        # A-B = 0.8, B-C = 0.8, A-C = 0.6. Connected component must still be one cluster.
        plan = build_duplicate_plan(
            [
                candidate("a", index=0, confidence=0.70, x=0.0),
                candidate("b", index=1, confidence=0.99, x=4.0),
                candidate("c", index=2, confidence=0.80, x=8.0),
            ],
            threshold=0.80,
        )
        self.assertEqual({"b"}, set(plan.winner_candidate_ids))
        self.assertEqual({"a", "c"}, set(plan.loser_candidate_ids))
        self.assertEqual(1, len(plan.clusters))
        self.assertEqual("b", plan.clusters[0].winner.candidate_id)

    def test_singleton_remains_classifier_candidate(self) -> None:
        plan = build_duplicate_plan(
            [candidate("a", index=0, confidence=0.50, x=0.0)],
            threshold=0.80,
        )
        self.assertEqual({"a"}, set(plan.winner_candidate_ids))
        self.assertEqual(set(), set(plan.loser_candidate_ids))
        self.assertEqual(0, len(plan.clusters))

    def test_different_regions_never_join(self) -> None:
        plan = build_duplicate_plan(
            [
                candidate("hand", region="completed_hand", index=0, confidence=0.50, x=0.0),
                candidate("dora", region="dora_indicators", index=1, confidence=0.90, x=0.0),
            ],
            threshold=0.80,
        )
        self.assertEqual({"hand", "dora"}, set(plan.winner_candidate_ids))
        self.assertEqual(0, len(plan.clusters))

    def test_equal_confidence_uses_lower_detection_index(self) -> None:
        plan = build_duplicate_plan(
            [
                candidate("later", index=9, confidence=0.90, x=0.0),
                candidate("earlier", index=3, confidence=0.90, x=1.0),
            ],
            threshold=0.80,
        )
        self.assertEqual({"earlier"}, set(plan.winner_candidate_ids))
        self.assertEqual({"later"}, set(plan.loser_candidate_ids))


if __name__ == "__main__":
    unittest.main()
