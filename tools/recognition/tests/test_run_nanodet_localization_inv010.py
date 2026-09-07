from __future__ import annotations

import unittest

from tools.recognition.build_nanodet_capture_finetune_dataset import empty_coco
from tools.recognition.run_nanodet_localization_inv010 import (
    build_source_variant_exposures,
    combine_frozen_real_train_with_new_layouts,
    prediction_gt_metrics,
    region_for_point,
    sample_images_near_annotation_target,
    source_mix_stats,
    split_extra_layouts,
)


class NanoDetLocalizationInv010Test(unittest.TestCase):
    def test_r1_real_train_keeps_frozen_d0_source_even_if_current_db_cannot_rebuild_it(self) -> None:
        frozen = empty_coco("frozen")
        frozen["images"] = [
            {"id": 7, "file_name": "frozen-layout-007.png", "layout_id": "layout-007"}
        ]
        frozen["annotations"] = [
            {"id": 9, "image_id": 7, "category_id": 1, "bbox": [1, 2, 3, 4]}
        ]

        result = combine_frozen_real_train_with_new_layouts(frozen, None)

        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(len(result["annotations"]), 1)
        self.assertEqual(result["images"][0]["layout_id"], "layout-007")
        self.assertEqual(result["images"][0]["fine_tune_source"], "frozen_d0_real_train")

    def test_r1_real_train_adds_only_new_completed_layout_payload_to_frozen_source(self) -> None:
        frozen = empty_coco("frozen")
        frozen["images"] = [{"id": 1, "file_name": "old.png", "layout_id": "layout-001"}]
        frozen["annotations"] = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]}
        ]
        new = empty_coco("new")
        new["images"] = [{"id": 1, "file_name": "new.png", "layout_id": "layout-011"}]
        new["annotations"] = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2]}
        ]

        result = combine_frozen_real_train_with_new_layouts(frozen, new)

        self.assertEqual([image["layout_id"] for image in result["images"]], ["layout-001", "layout-011"])
        self.assertEqual(
            [image["fine_tune_source"] for image in result["images"]],
            ["frozen_d0_real_train", "newly_completed_real"],
        )
        self.assertEqual([annotation["image_id"] for annotation in result["annotations"]], [1, 2])

    def test_single_extra_layout_is_reserved_as_final_holdout(self) -> None:
        train, holdout = split_extra_layouts(
            ["layout-011"], train_fraction=0.75, seed=42
        )

        self.assertEqual(train, frozenset())
        self.assertEqual(holdout, frozenset({"layout-011"}))

    def test_extra_layout_split_is_disjoint_and_keeps_a_holdout(self) -> None:
        layouts = [f"layout-{index:03d}" for index in range(11, 19)]

        train, holdout = split_extra_layouts(layouts, train_fraction=0.75, seed=42)

        self.assertFalse(train & holdout)
        self.assertEqual(train | holdout, frozenset(layouts))
        self.assertGreaterEqual(len(train), 1)
        self.assertGreaterEqual(len(holdout), 1)

    def test_source_variant_exposure_preserves_per_source_exposure_and_annotations(self) -> None:
        originals = empty_coco("originals")
        originals["images"] = [
            {"id": 1, "file_name": "a.png"},
            {"id": 2, "file_name": "b.png"},
        ]
        originals["annotations"] = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10]},
        ]
        augmented = empty_coco("augmented")
        augmented["images"] = [
            {
                "id": 10,
                "file_name": "a0.png",
                "source_image_id": 1,
                "region_rotation_copy_index": 0,
            },
            {
                "id": 11,
                "file_name": "a1.png",
                "source_image_id": 1,
                "region_rotation_copy_index": 1,
            },
            {
                "id": 20,
                "file_name": "b0.png",
                "source_image_id": 2,
                "region_rotation_copy_index": 0,
            },
            {
                "id": 21,
                "file_name": "b1.png",
                "source_image_id": 2,
                "region_rotation_copy_index": 1,
            },
        ]
        augmented["annotations"] = [
            {"id": 10, "image_id": 10, "category_id": 1, "bbox": [0, 0, 9, 9]},
            {"id": 11, "image_id": 11, "category_id": 1, "bbox": [0, 0, 9, 9]},
            {"id": 20, "image_id": 20, "category_id": 1, "bbox": [0, 0, 9, 9]},
            {"id": 21, "image_id": 21, "category_id": 1, "bbox": [0, 0, 9, 9]},
        ]

        result = build_source_variant_exposures(
            originals,
            augmented,
            exposures_per_source=5,
            seed=42,
        )

        self.assertEqual(len(result["images"]), 10)
        self.assertEqual(len(result["annotations"]), 10)
        by_source = {1: 0, 2: 0}
        kinds = set()
        for image in result["images"]:
            by_source[int(image["inv010_source_image_id"])] += 1
            kinds.add(str(image["inv010_variant_kind"]))
        self.assertEqual(by_source, {1: 5, 2: 5})
        self.assertEqual(kinds, {"original", "a1"})

    def test_base_sampler_gets_close_to_annotation_target_without_using_image_count_ratio(self) -> None:
        payload = empty_coco("base")
        annotation_id = 1
        for image_id, annotation_count in enumerate((80, 90, 100, 110, 120), start=1):
            payload["images"].append(
                {"id": image_id, "file_name": f"{image_id}.png"}
            )
            for _ in range(annotation_count):
                payload["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [0, 0, 1, 1],
                    }
                )
                annotation_id += 1

        sampled = sample_images_near_annotation_target(
            payload, target_annotations=205, seed=42
        )

        self.assertGreaterEqual(len(sampled["images"]), 1)
        self.assertLess(abs(len(sampled["annotations"]) - 205), 120)

    def test_crop_metrics_use_polygon_coverage_and_prediction_purity(self) -> None:
        annotation = {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "segmentation": [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]],
        }

        perfect = prediction_gt_metrics([10.0, 10.0, 20.0, 20.0], annotation)
        half = prediction_gt_metrics([10.0, 10.0, 10.0, 20.0], annotation)

        self.assertAlmostEqual(perfect.iou, 1.0)
        self.assertAlmostEqual(perfect.gt_coverage, 1.0)
        self.assertAlmostEqual(perfect.crop_purity, 1.0)
        self.assertAlmostEqual(half.gt_coverage, 0.5)
        self.assertAlmostEqual(half.crop_purity, 1.0)
        self.assertAlmostEqual(half.iou, 0.5)

    def test_region_assignment_uses_fixed_semantic_destinations(self) -> None:
        destinations = {
            "completed_hand": {"x": 7, "y": 0, "width": 306, "height": 72},
            "dora_indicators": {"x": 7, "y": 74, "width": 306, "height": 72},
            "melds": {"x": 74, "y": 148, "width": 172, "height": 172},
        }

        self.assertEqual(region_for_point(100, 30, destinations), "completed_hand")
        self.assertEqual(region_for_point(100, 100, destinations), "dora_indicators")
        self.assertEqual(region_for_point(100, 200, destinations), "melds")
        self.assertEqual(region_for_point(10, 200, destinations), "outside")

    def test_source_mix_stats_reports_annotation_share_not_only_image_share(self) -> None:
        payload = empty_coco("mix")
        payload["images"] = [
            {"id": 1, "file_name": "real.png", "fine_tune_source": "real"},
            {"id": 2, "file_name": "base.png", "fine_tune_source": "base"},
        ]
        payload["annotations"] = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]},
            *[
                {
                    "id": index + 2,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [0, 0, 1, 1],
                }
                for index in range(9)
            ],
        ]

        stats = source_mix_stats(payload)

        self.assertAlmostEqual(stats["real"]["image_entry_share"], 0.5)
        self.assertAlmostEqual(stats["real"]["annotation_target_share"], 0.1)
        self.assertAlmostEqual(stats["base"]["annotation_target_share"], 0.9)


if __name__ == "__main__":
    unittest.main()
