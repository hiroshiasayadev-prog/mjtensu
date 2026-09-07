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
    apply_training_geometry,
)
from run_perspective_classifier_mixture_experiment import (
    A3_CONTENT_POLICY,
    ARCHITECTURES,
    BRANCHES,
    BRANCH_TO_INDEX,
    CONDITIONS,
    RECIPES,
    RECIPE_BY_NAME,
    apply_mixed_training_geometry,
    deterministic_recipe_choices,
    preletterbox_content_extent,
    validate_recipe,
)


class PerspectiveClassifierMixtureExperimentTest(unittest.TestCase):
    def test_matrix_is_plain_and_f8_r1_by_three_recipes(self) -> None:
        self.assertEqual(ARCHITECTURES, ("plain", "f8-r1"))
        self.assertEqual(tuple(recipe.name for recipe in RECIPES), ("mix-light", "mix-mid", "mix-heavy"))
        self.assertEqual(len(CONDITIONS), 6)
        self.assertEqual(
            {(condition.architecture, condition.recipe) for condition in CONDITIONS},
            {
                (architecture, recipe.name)
                for architecture in ARCHITECTURES
                for recipe in RECIPES
            },
        )

    def test_recipes_cover_original_a0_a1_a2_a3_and_sum_to_one(self) -> None:
        self.assertEqual(
            BRANCHES,
            (
                "original",
                "a0-random360",
                "a1-anisotropic-affine",
                "a2-perspective",
                "a3-perspective-recrop",
            ),
        )
        expected = {
            "mix-light": (0.30, 0.25, 0.15, 0.20, 0.10),
            "mix-mid": (0.20, 0.20, 0.15, 0.30, 0.15),
            "mix-heavy": (0.10, 0.15, 0.15, 0.35, 0.25),
        }
        for recipe in RECIPES:
            validate_recipe(recipe)
            self.assertEqual(recipe.weights(), expected[recipe.name])
            self.assertAlmostEqual(sum(recipe.weights()), 1.0)

    def test_branch_choice_is_deterministic_and_recipe_thresholded(self) -> None:
        sample_ids = [f"sample-{index:05d}" for index in range(20_000)]
        for recipe in RECIPES:
            first = deterministic_recipe_choices(
                sample_ids,
                recipe=recipe,
                seed=42,
                epoch=73,
            )
            repeated = deterministic_recipe_choices(
                sample_ids,
                recipe=recipe,
                seed=42,
                epoch=73,
            )
            np.testing.assert_array_equal(first, repeated)
            self.assertTrue(np.all(first >= 0))
            self.assertTrue(np.all(first < len(BRANCHES)))
            observed = np.bincount(first, minlength=len(BRANCHES)) / len(first)
            np.testing.assert_allclose(
                observed,
                np.asarray(recipe.weights()),
                rtol=0.0,
                atol=0.012,
            )

    def test_same_uniform_choice_stream_makes_recipe_changes_nested(self) -> None:
        sample_ids = [f"sample-{index:04d}" for index in range(5000)]
        light = deterministic_recipe_choices(
            sample_ids,
            recipe=RECIPE_BY_NAME["mix-light"],
            seed=42,
            epoch=17,
        )
        heavy = deterministic_recipe_choices(
            sample_ids,
            recipe=RECIPE_BY_NAME["mix-heavy"],
            seed=42,
            epoch=17,
        )
        # Heavy deliberately shrinks Original from 30% to 10%. With the same random
        # choice stream, every heavy-Original sample must also be light-Original.
        heavy_original = heavy == BRANCH_TO_INDEX["original"]
        self.assertTrue(np.all(light[heavy_original] == BRANCH_TO_INDEX["original"]))

    def test_preletterbox_extent_reconstructs_dataset_builder_resize(self) -> None:
        self.assertEqual(preletterbox_content_extent(64, 64, image_size=64), (1.0, 1.0))
        self.assertEqual(preletterbox_content_extent(48, 64, image_size=64), (0.75, 1.0))
        self.assertEqual(preletterbox_content_extent(64, 48, image_size=64), (1.0, 0.75))
        self.assertEqual(preletterbox_content_extent(100, 50, image_size=64), (1.0, 0.5))
        self.assertEqual(A3_CONTENT_POLICY, "pre-letterbox-content-from-original-width-height")

    def test_corrected_a3_differs_from_legacy_when_cached_crop_has_letterbox(self) -> None:
        image = torch.full((1, 1, 64, 64), 0.2, dtype=torch.float32)
        # 48x64 pre-letterbox content: 8 px side padding in the 64x64 cached image.
        image[:, :, :, 8:56] = 0.8
        angles = torch.tensor([35.0], dtype=torch.float32)
        units = torch.tensor(
            [[0.65, 0.35, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.8, 0.2]],
            dtype=torch.float32,
        )
        spec = AUGMENTATION_SPECS["a3-perspective-recrop"]
        legacy = apply_training_geometry(
            image,
            spec=spec,
            angles_deg=angles,
            units=units,
        )
        corrected = apply_training_geometry(
            image,
            spec=spec,
            angles_deg=angles,
            units=units,
            content_extent_x=torch.tensor([0.75]),
            content_extent_y=torch.tensor([1.0]),
        )
        self.assertEqual(tuple(corrected.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(corrected).all())
        self.assertFalse(torch.equal(legacy, corrected))

    def test_mixture_keeps_original_branch_bit_exact_and_transforms_other_branches(self) -> None:
        generator = torch.Generator().manual_seed(20260903)
        images = torch.rand((5, 1, 64, 64), generator=generator)
        choices = torch.arange(5, dtype=torch.long)
        angles = torch.tensor([91.0, 47.0, -33.0, 26.0, 19.0], dtype=torch.float32)
        units = torch.tensor(
            [
                [0.4] * GEOMETRY_UNIT_COUNT,
                [0.3] * GEOMETRY_UNIT_COUNT,
                [0.2] * GEOMETRY_UNIT_COUNT,
                [0.8] * GEOMETRY_UNIT_COUNT,
                [0.7] * GEOMETRY_UNIT_COUNT,
            ],
            dtype=torch.float32,
        )
        observed = apply_mixed_training_geometry(
            images,
            choices=choices,
            angles_deg=angles,
            units=units,
            content_extent_x=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.75]),
            content_extent_y=torch.ones(5),
        )
        torch.testing.assert_close(observed[0], images[0], rtol=0.0, atol=0.0)
        for index in range(1, 5):
            self.assertFalse(torch.equal(observed[index], images[index]), msg=BRANCHES[index])


if __name__ == "__main__":
    unittest.main()
