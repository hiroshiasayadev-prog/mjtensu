from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

RECOGNITION_TOOLS = Path(__file__).resolve().parents[1]
if str(RECOGNITION_TOOLS) not in sys.path:
    sys.path.insert(0, str(RECOGNITION_TOOLS))

from render_classifier_activation_comparison import (
    SelectedSamples,
    compute_stage_statistics,
    f8_stage_specs,
    plain_stage_specs,
    simple_inferno,
)
from resolution_preserving_mobile_models import build_resolution_preserving_mobile_classifier
from tile_shape_classifier import PlainTileShapeClassifier


class ClassifierActivationComparisonTest(unittest.TestCase):
    def test_plain_stage_specs_capture_three_pools_and_final_conv(self) -> None:
        model = PlainTileShapeClassifier(class_count=35).eval()
        stages = plain_stage_specs(model)
        self.assertEqual([stage.name for stage in stages], ["pool1", "pool2", "pool3", "conv4"])

        observed: dict[str, tuple[int, ...]] = {}
        handles = [
            stage.module.register_forward_hook(
                lambda _module, _inputs, output, name=stage.name: observed.__setitem__(
                    name, tuple(int(value) for value in output.shape)
                )
            )
            for stage in stages
        ]
        try:
            with torch.inference_mode():
                model(torch.zeros((2, 1, 64, 64), dtype=torch.float32))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(observed["pool1"], (2, 32, 32, 32))
        self.assertEqual(observed["pool2"], (2, 64, 16, 16))
        self.assertEqual(observed["pool3"], (2, 128, 8, 8))
        self.assertEqual(observed["conv4"], (2, 192, 8, 8))

    def test_f8_stage_specs_cover_spatial_schedule_and_final_feature_map(self) -> None:
        model = build_resolution_preserving_mobile_classifier(
            "mobile-tile-f8-r1", class_count=35
        ).eval()
        stages = f8_stage_specs(model, image_size=64, device=torch.device("cpu"))
        self.assertEqual(
            [stage.name for stage in stages],
            ["stem32", "stage16", "stage8-early", "stage8-late", "stage8-final"],
        )
        observed: dict[str, tuple[int, ...]] = {}
        handles = [
            stage.module.register_forward_hook(
                lambda _module, _inputs, output, name=stage.name: observed.__setitem__(
                    name, tuple(int(value) for value in output.shape)
                )
            )
            for stage in stages
        ]
        try:
            with torch.inference_mode():
                model(torch.zeros((2, 1, 64, 64), dtype=torch.float32))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(observed["stem32"], (2, 16, 32, 32))
        self.assertEqual(observed["stage16"], (2, 16, 16, 16))
        self.assertEqual(observed["stage8-early"], (2, 24, 8, 8))
        self.assertEqual(observed["stage8-late"], (2, 96, 8, 8))
        self.assertEqual(observed["stage8-final"], (2, 576, 8, 8))

    def test_channel_ranking_surfaces_reliable_six_m_specific_channel(self) -> None:
        labels = ("5m", "6m", "7m")
        positions = {
            "5m": np.asarray([0, 1], dtype=np.int64),
            "6m": np.asarray([2, 3], dtype=np.int64),
            "7m": np.asarray([4, 5], dtype=np.int64),
        }
        samples = SelectedSamples(
            split="fixture",
            labels=labels,
            images_u8=np.zeros((6, 64, 64), dtype=np.uint8),
            class_indices=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
            sample_ids=tuple(f"s{index}" for index in range(6)),
            label_positions=positions,
            content_extent_x=np.ones((6,), dtype=np.float32),
            content_extent_y=np.ones((6,), dtype=np.float32),
        )
        activation = np.zeros((6, 3, 4, 4), dtype=np.float32)
        activation[positions["5m"], 0] = 0.1
        activation[positions["6m"], 0] = 3.0
        activation[positions["7m"], 0] = 0.2
        activation[:, 1] = np.linspace(0.0, 0.01, 6, dtype=np.float32)[:, None, None]
        activation[:, 2] = 1.0
        stats = compute_stage_statistics(activation, samples=samples, top_channels=3)
        self.assertEqual(stats["top_channels"][0]["channel"], 0)
        self.assertGreater(stats["top_channels"][0]["signed_6m_vs_neighbors"], 0.0)

    def test_colormap_returns_rgb_u8(self) -> None:
        source = np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32)
        observed = simple_inferno(source)
        self.assertEqual(observed.shape, (1, 3, 3))
        self.assertEqual(observed.dtype, np.uint8)
        self.assertFalse(np.array_equal(observed[0, 0], observed[0, 2]))


if __name__ == "__main__":
    unittest.main()
