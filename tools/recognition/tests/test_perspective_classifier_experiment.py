from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

RECOGNITION_TOOLS = Path(__file__).resolve().parents[1]
if str(RECOGNITION_TOOLS) not in sys.path:
    sys.path.insert(0, str(RECOGNITION_TOOLS))

from perspective_classifier_augmentation import (
    AUGMENTATION_SPECS,
    GEOMETRY_UNIT_COUNT,
    PERSPECTIVE_EVALUATION_CASES,
    GeometryCase,
    apply_evaluation_case,
    apply_training_geometry,
    deterministic_geometry_units,
    solve_homography,
    transform_points,
    warp_homography,
)
from run_perspective_classifier_experiment import (
    AUGMENTATIONS,
    ARCHITECTURES,
    CONDITIONS,
    EXPERIMENT_IMPLEMENTATION_VERSION,
    build_condition_model,
    prior_result_is_reusable,
    summarize_confusion,
)
from run_rotation_classifier_experiment import rotate_batch


class PerspectiveClassifierExperimentTest(unittest.TestCase):
    def test_condition_matrix_is_complete_plain_f8_by_a0_to_a3_factorial(self) -> None:
        self.assertEqual(ARCHITECTURES, ("plain", "f8-r1"))
        self.assertEqual(
            AUGMENTATIONS,
            (
                "a0-random360",
                "a1-anisotropic-affine",
                "a2-perspective",
                "a3-perspective-recrop",
            ),
        )
        self.assertEqual(len(CONDITIONS), 8)
        self.assertEqual(
            {(condition.architecture, condition.augmentation) for condition in CONDITIONS},
            {
                (architecture, augmentation)
                for architecture in ARCHITECTURES
                for augmentation in AUGMENTATIONS
            },
        )

    def test_a0_matches_existing_random360_implementation(self) -> None:
        generator = torch.Generator().manual_seed(20260903)
        images = torch.randn((3, 1, 64, 64), generator=generator)
        angles = torch.tensor([-123.5, 0.0, 47.25], dtype=torch.float32)
        units = torch.zeros((3, GEOMETRY_UNIT_COUNT), dtype=torch.float32)
        observed = apply_training_geometry(
            images,
            spec=AUGMENTATION_SPECS["a0-random360"],
            angles_deg=angles,
            units=units,
        )
        expected = rotate_batch(images, angles)
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

    def test_geometry_units_are_sample_epoch_and_stream_deterministic(self) -> None:
        sample_ids = ["sample-a", "sample-b", "sample-c"]
        first = deterministic_geometry_units(
            sample_ids, seed=42, epoch=17, stream="inv013-shared-geometry"
        )
        repeated = deterministic_geometry_units(
            sample_ids, seed=42, epoch=17, stream="inv013-shared-geometry"
        )
        next_epoch = deterministic_geometry_units(
            sample_ids, seed=42, epoch=18, stream="inv013-shared-geometry"
        )
        reordered = deterministic_geometry_units(
            list(reversed(sample_ids)),
            seed=42,
            epoch=17,
            stream="inv013-shared-geometry",
        )
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_allclose(first, reordered[::-1], rtol=0.0, atol=0.0)
        self.assertEqual(first.shape, (3, GEOMETRY_UNIT_COUNT))
        self.assertTrue(np.all(first >= 0.0))
        self.assertTrue(np.all(first < 1.0))
        self.assertFalse(np.array_equal(first, next_epoch))

    def test_identity_homography_maps_points_and_image_to_themselves(self) -> None:
        source = torch.tensor(
            [[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]],
            dtype=torch.float32,
        )
        homography = solve_homography(source, source.clone())
        mapped = transform_points(homography, source)
        torch.testing.assert_close(mapped, source, rtol=1.0e-6, atol=1.0e-6)

        image = torch.linspace(0.0, 1.0, steps=32 * 32).reshape(1, 1, 32, 32)
        warped = warp_homography(
            image,
            homography,
            output_height=32,
            output_width=32,
        )
        torch.testing.assert_close(warped, image, rtol=1.0e-5, atol=1.0e-5)

    def test_a1_a2_a3_preserve_shape_and_apply_nontrivial_geometry(self) -> None:
        image = torch.zeros((2, 1, 64, 64), dtype=torch.float32)
        image[:, :, 14:50, 24:40] = 1.0
        angles = torch.tensor([18.0, -27.0], dtype=torch.float32)
        units = torch.tensor(
            [
                [0.0, 0.7, 0.9, 0.1, 0.9, 0.2, 0.8, 0.3, 0.8, 0.2, 0.9, 0.3],
                [1.0, 0.2, 0.1, 0.9, 0.1, 0.8, 0.2, 0.7, 0.2, 0.8, 0.2, 0.9],
            ],
            dtype=torch.float32,
        )
        for name in (
            "a1-anisotropic-affine",
            "a2-perspective",
            "a3-perspective-recrop",
        ):
            observed = apply_training_geometry(
                image,
                spec=AUGMENTATION_SPECS[name],
                angles_deg=angles,
                units=units,
            )
            self.assertEqual(tuple(observed.shape), tuple(image.shape), msg=name)
            self.assertTrue(torch.isfinite(observed).all(), msg=name)
            self.assertFalse(torch.equal(observed, image), msg=name)

    def test_evaluation_grid_covers_front_affine_projective_and_recrop_surfaces(self) -> None:
        names = {case.name for case in PERSPECTIVE_EVALUATION_CASES}
        augmentations = {case.augmentation for case in PERSPECTIVE_EVALUATION_CASES}
        self.assertIn("front-facing", names)
        self.assertIn("a1-anisotropic-affine", augmentations)
        self.assertIn("a2-perspective", augmentations)
        self.assertIn("a3-perspective-recrop", augmentations)
        self.assertTrue(any("left" in name for name in names))
        self.assertTrue(any("right" in name for name in names))
        self.assertTrue(any("0p15" in name for name in names))

    def test_evaluation_case_is_deterministic_and_preserves_contract(self) -> None:
        image = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        image[:, :, 12:52, 18:46] = 1.0
        case = GeometryCase(
            "fixture",
            "a3-perspective-recrop",
            angle_deg=15.0,
            perspective_yaw=0.12,
            bbox_center_x=0.03,
            bbox_scale_y=0.95,
        )
        first = apply_evaluation_case(image, case)
        second = apply_evaluation_case(image, case)
        self.assertEqual(tuple(first.shape), (1, 1, 64, 64))
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(first, image))

    def test_plain_and_f8_r1_keep_same_gray64_35_logit_contract(self) -> None:
        image = torch.randn((2, 1, 64, 64), dtype=torch.float32)
        for architecture in ARCHITECTURES:
            model = build_condition_model(
                architecture, class_count=35, image_size=64
            ).eval()
            with torch.inference_mode():
                logits = model(image)
            self.assertEqual(tuple(logits.shape), (2, 35), msg=architecture)

    def test_confusion_summary_explicitly_surfaces_six_m_to_five_or_seven(self) -> None:
        labels = ("5m", "6m", "7m", "1p")
        confusion = np.asarray(
            [
                [9, 1, 0, 0],
                [2, 6, 2, 0],
                [0, 1, 9, 0],
                [0, 0, 0, 10],
            ],
            dtype=np.int64,
        )
        summary = summarize_confusion(confusion, labels)
        self.assertEqual(summary["six_m_to_5m_or_7m_count"], 4)
        self.assertAlmostEqual(summary["six_m_to_5m_or_7m_rate"], 0.4)
        focus = {
            (row["true"], row["predicted"]): row
            for row in summary["focus_5m_6m_7m"]
        }
        self.assertEqual(focus[("6m", "5m")]["count"], 2)
        self.assertEqual(focus[("6m", "7m")]["count"], 2)

    def test_resume_requires_exact_inv013_version_and_condition(self) -> None:
        condition = CONDITIONS[0]
        current = {
            "status": "completed",
            "condition": condition.__dict__,
            "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        }
        old_version = {
            **current,
            "implementation_version": "inv013-perspective-augmentation-v0",
        }
        wrong_condition = {
            **current,
            "condition": CONDITIONS[1].__dict__,
        }
        self.assertTrue(prior_result_is_reusable(condition, current))
        self.assertFalse(prior_result_is_reusable(condition, old_version))
        self.assertFalse(prior_result_is_reusable(condition, wrong_condition))


if __name__ == "__main__":
    unittest.main()
