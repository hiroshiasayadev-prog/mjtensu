from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.recognition.red_five_classifier import INPUT_CHANNELS, normalize_input_mode
from tools.recognition.train_red_five_classifier import (
    SplitCache,
    binary_metrics,
    compute_input_statistics,
    rgb_u8_to_input_numpy,
    rgb_u8_to_input_torch,
    robust_jp_validation_score,
)


class ColorTransformTests(unittest.TestCase):
    def test_input_channel_contracts(self) -> None:
        self.assertEqual(3, INPUT_CHANNELS[normalize_input_mode("RGB")])
        self.assertEqual(1, INPUT_CHANNELS[normalize_input_mode("cr")])
        self.assertEqual(2, INPUT_CHANNELS[normalize_input_mode("YCr")])

    def test_numpy_and_torch_transforms_match(self) -> None:
        images = np.asarray(
            [
                [
                    [[255, 0, 0], [0, 255, 0]],
                    [[0, 0, 255], [128, 128, 128]],
                ]
            ],
            dtype=np.uint8,
        )
        tensor = torch.from_numpy(images.copy())
        for mode in ("rgb", "cr", "ycr"):
            expected = rgb_u8_to_input_numpy(images, input_mode=mode)
            actual = (
                rgb_u8_to_input_torch(tensor, input_mode=mode)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_cr_is_high_for_red_and_neutral_for_gray(self) -> None:
        images = np.asarray([[[[255, 0, 0], [128, 128, 128]]]], dtype=np.uint8)
        cr = rgb_u8_to_input_numpy(images, input_mode="cr")
        self.assertGreater(float(cr[0, 0, 0, 0]), 0.95)
        self.assertAlmostEqual(0.5, float(cr[0, 0, 1, 0]), places=5)

    def test_statistics_respect_train_repeat(self) -> None:
        images = np.zeros((2, 1, 1, 3), dtype=np.uint8)
        images[1, 0, 0] = (255, 0, 0)
        split = SplitCache(
            name="train",
            images_u8=torch.from_numpy(images),
            labels=torch.tensor([0, 1], dtype=torch.int64),
            train_repeat=np.asarray([1, 3], dtype=np.int64),
            suit=np.asarray(["m", "m"], dtype=object),
            source=np.asarray(["jp", "manual"], dtype=object),
            brightness=np.asarray(["", "dark"], dtype=object),
            shadow=np.asarray(["", "partial"], dtype=object),
            sample_ids=["a", "b"],
        )
        mean, _ = compute_input_statistics(split, input_mode="rgb", block_size=2)
        self.assertAlmostEqual(0.75, mean[0], places=6)
        self.assertAlmostEqual(0.0, mean[1], places=6)
        self.assertAlmostEqual(0.0, mean[2], places=6)


class MetricTests(unittest.TestCase):
    def test_binary_metrics(self) -> None:
        metrics = binary_metrics(
            np.asarray([1, 0, 1, 0]),
            np.asarray([1, 1, 0, 0]),
        )
        self.assertEqual(1, metrics["tp"])
        self.assertEqual(1, metrics["tn"])
        self.assertEqual(1, metrics["fp"])
        self.assertEqual(1, metrics["fn"])
        self.assertAlmostEqual(0.5, metrics["balanced_accuracy"])

    def test_checkpoint_score_requires_complete_jp_angle_sweep(self) -> None:
        validation = {
            "jp_val": {
                "angles": {
                    "0deg": {"overall": {"balanced_accuracy": 1.0}},
                    "15deg": {"overall": {"balanced_accuracy": 0.9}},
                }
            }
        }
        self.assertIsNone(
            robust_jp_validation_score(validation, required_angles=(0.0, 15.0, 30.0))
        )
        self.assertAlmostEqual(
            0.95,
            robust_jp_validation_score(validation, required_angles=(0.0, 15.0)),
        )


if __name__ == "__main__":
    unittest.main()
