from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

RECOGNITION_TOOLS = Path(__file__).resolve().parents[1]
if str(RECOGNITION_TOOLS) not in sys.path:
    sys.path.insert(0, str(RECOGNITION_TOOLS))

from resolution_preserving_mobile_models import (
    ResolutionPreservingMobileNetV3TileClassifier,
    build_resolution_preserving_mobile_classifier,
    describe_resolution_preserving_mobile_classifier,
    parse_resolution_preserving_condition,
)
from run_resolution_preserving_mobile_experiment import (
    CONDITIONS,
    EXPERIMENT_IMPLEMENTATION_VERSION,
    ROBUSTNESS_PERTURBATIONS,
    RobustnessPerturbation,
    apply_robustness_perturbation,
    prior_result_is_reusable,
    summarize_confusion,
)
from run_mobile_classifier_experiment import (
    analyze_mobile_onnx_graph,
    smoke_onnx_dynamic_batch,
)


class ResolutionPreservingMobileExperimentTest(unittest.TestCase):
    def test_condition_matrix_is_the_controlled_six_condition_search(self) -> None:
        self.assertEqual(
            [condition.name for condition in CONDITIONS],
            [
                "mobile-tile-f8-r1",
                "mobile-tile-f8-r2",
                "mobile-tile-f8-r3",
                "mobile-tile-f4-r1",
                "mobile-tile-f4-r2",
                "mobile-tile-f4-r3",
            ],
        )
        self.assertEqual(
            {(condition.final_feature_resolution, condition.late_repeats) for condition in CONDITIONS},
            {(resolution, repeats) for resolution in (4, 8) for repeats in (1, 2, 3)},
        )

    def test_condition_names_parse_to_resolution_and_repeat_count(self) -> None:
        for condition in CONDITIONS:
            self.assertEqual(
                parse_resolution_preserving_condition(condition.name),
                (condition.final_feature_resolution, condition.late_repeats),
            )
        with self.assertRaises(ValueError):
            parse_resolution_preserving_condition("mobile-tile-f2-r3")
        with self.assertRaises(ValueError):
            parse_resolution_preserving_condition("mobile-tile-f8-r4")

    def test_every_candidate_preserves_declared_feature_resolution_and_classifier_contract(self) -> None:
        image = torch.randn((2, 1, 64, 64), dtype=torch.float32)
        for condition in CONDITIONS:
            model = build_resolution_preserving_mobile_classifier(
                condition.name,
                class_count=35,
            ).eval()
            with torch.inference_mode():
                features = model.forward_features(image)
                logits = model(image)
            expected = condition.final_feature_resolution
            self.assertEqual(tuple(features.shape[-2:]), (expected, expected), msg=condition.name)
            self.assertEqual(tuple(logits.shape), (2, 35), msg=condition.name)

    def test_endpoint_is_never_downsampled_again(self) -> None:
        for condition in CONDITIONS:
            model = build_resolution_preserving_mobile_classifier(condition.name)
            strides = [config.stride for config in model.block_configs]
            if condition.final_feature_resolution == 8:
                self.assertEqual(strides[:4], [2, 2, 1, 1], msg=condition.name)
            else:
                self.assertEqual(strides[:4], [2, 2, 1, 2], msg=condition.name)
            self.assertTrue(all(stride == 1 for stride in strides[4:]), msg=condition.name)

    def test_late_repeat_factor_adds_independent_96_channel_blocks(self) -> None:
        for resolution in (4, 8):
            models = [
                build_resolution_preserving_mobile_classifier(
                    f"mobile-tile-f{resolution}-r{repeats}"
                )
                for repeats in (1, 2, 3)
            ]
            parameter_counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
            self.assertLess(parameter_counts[0], parameter_counts[1])
            self.assertLess(parameter_counts[1], parameter_counts[2])
            for repeats, model in zip((1, 2, 3), models):
                terminal = model.block_configs[-repeats:]
                self.assertEqual(len(terminal), repeats)
                self.assertTrue(all(config.out_channels == 96 for config in terminal))
                self.assertTrue(all(config.stride == 1 for config in terminal))

    def test_candidates_keep_mobilenet_v3_depthwise_se_and_activation_operators(self) -> None:
        model = build_resolution_preserving_mobile_classifier("mobile-tile-f8-r2")
        depthwise = [
            module
            for module in model.modules()
            if isinstance(module, nn.Conv2d)
            and module.groups > 1
            and module.groups == module.in_channels
            and module.in_channels == module.out_channels
        ]
        self.assertGreater(len(depthwise), 0)
        self.assertTrue(any(isinstance(module, nn.Hardswish) for module in model.modules()))
        self.assertTrue(any(isinstance(module, nn.Hardsigmoid) for module in model.modules()))
        batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertTrue(batch_norms)
        self.assertTrue(all(abs(module.eps - 0.001) < 1.0e-12 for module in batch_norms))
        self.assertTrue(all(abs(module.momentum - 0.01) < 1.0e-12 for module in batch_norms))

    def test_model_description_records_the_controlled_factors(self) -> None:
        model = build_resolution_preserving_mobile_classifier("mobile-tile-f4-r3")
        description = describe_resolution_preserving_mobile_classifier(
            model,
            "mobile-tile-f4-r3",
        )
        self.assertEqual(description.family, "mobilenet-v3-small-tile")
        self.assertEqual(description.final_feature_resolution, 4)
        self.assertEqual(description.late_repeats, 3)
        self.assertGreater(description.parameter_count, 0)
        self.assertEqual(description.parameter_count, description.trainable_parameter_count)
        self.assertEqual(description.details["input_size"], 64)
        self.assertEqual(description.details["class_count"], 35)
        self.assertEqual(description.details["final_feature_resolution"], 4)
        self.assertEqual(description.details["late_repeats"], 3)

    def test_robustness_matrix_is_bounded_and_contains_geometry_plus_resampling(self) -> None:
        self.assertEqual(ROBUSTNESS_PERTURBATIONS[0].name, "identity")
        names = {value.name for value in ROBUSTNESS_PERTURBATIONS}
        self.assertIn("shift-x-minus-2px", names)
        self.assertIn("shift-y-plus-2px", names)
        self.assertIn("scale-0p94", names)
        self.assertIn("scale-1p06", names)
        self.assertIn("blur-3x3", names)
        self.assertLessEqual(len(ROBUSTNESS_PERTURBATIONS), 10)

    def test_robustness_perturbations_preserve_tensor_shape_and_change_non_identity_input(self) -> None:
        image = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        image[:, :, 28:36, 30:34] = 1.0
        identity = apply_robustness_perturbation(image, RobustnessPerturbation("identity"))
        shifted = apply_robustness_perturbation(
            image,
            RobustnessPerturbation("shift", shift_x_px=2.0),
        )
        scaled = apply_robustness_perturbation(
            image,
            RobustnessPerturbation("scale", content_scale=1.06),
        )
        blurred = apply_robustness_perturbation(
            image,
            RobustnessPerturbation("blur", blur_kernel=3),
        )
        self.assertTrue(torch.equal(identity, image))
        for observed in (shifted, scaled, blurred):
            self.assertEqual(tuple(observed.shape), tuple(image.shape))
            self.assertFalse(torch.equal(observed, image))

    def test_confusion_summary_surfaces_observed_manzu_failure_pairs(self) -> None:
        labels = ("2m", "6m", "7m", "invalid")
        confusion = np.asarray(
            [
                [8, 0, 2, 0],
                [0, 7, 3, 0],
                [1, 0, 9, 0],
                [0, 1, 0, 9],
            ],
            dtype=np.int64,
        )
        summary = summarize_confusion(confusion, labels)
        focus = {
            (row["true"], row["predicted"]): row
            for row in summary["focus_2m_6m_7m"]
        }
        self.assertEqual(focus[("2m", "7m")]["count"], 2)
        self.assertAlmostEqual(focus[("2m", "7m")]["rate_given_true"], 0.2)
        self.assertEqual(focus[("6m", "7m")]["count"], 3)
        self.assertEqual(summary["worst_within_suit_pairs"][0]["true"], "6m")
        self.assertEqual(summary["worst_within_suit_pairs"][0]["predicted"], "7m")
        self.assertEqual(summary["invalid_background"]["invalid_to_tile_count"], 1)

    def test_resume_requires_exact_inv012_implementation_version(self) -> None:
        condition = CONDITIONS[0]
        current = {
            "status": "completed",
            "condition": condition.__dict__,
            "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        }
        old = {
            "status": "completed",
            "condition": condition.__dict__,
            "implementation_version": "inv012-resolution-preserving-mobile-v0",
        }
        self.assertTrue(prior_result_is_reusable(condition, current))
        self.assertFalse(prior_result_is_reusable(condition, old))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for INV-012 backward preflight")
    def test_every_candidate_completes_cuda_backward_step(self) -> None:
        image = torch.randn((2, 1, 64, 64), device="cuda", dtype=torch.float32)
        target = torch.tensor([3, 17], device="cuda", dtype=torch.int64)
        for condition in CONDITIONS:
            model = build_resolution_preserving_mobile_classifier(condition.name).cuda().train()
            logits = model(image)
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(gradients, msg=condition.name)
            self.assertTrue(all(gradient is not None for gradient in gradients), msg=condition.name)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients),
                msg=condition.name,
            )
            del model, logits, loss, gradients
            torch.cuda.empty_cache()

    def test_representative_candidates_export_dynamic_opset16_and_match_ort(self) -> None:
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnx and onnxruntime are required for deployment preflight")

        generator = torch.Generator().manual_seed(20260902)
        parity_input = torch.randn(
            (2, 1, 64, 64), generator=generator, dtype=torch.float32
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("mobile-tile-f8-r1", "mobile-tile-f4-r3"):
                model = build_resolution_preserving_mobile_classifier(name).eval()
                output_path = root / f"{name}.onnx"
                torch.onnx.export(
                    model,
                    parity_input[:1],
                    str(output_path),
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=["images"],
                    output_names=["logits"],
                    dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                )
                onnx.checker.check_model(onnx.load(str(output_path)))
                smoke = smoke_onnx_dynamic_batch(
                    output_path,
                    image_size=64,
                    batch_sizes=(1, 16, 24),
                    seed=42,
                )
                self.assertEqual(smoke["batches"]["24"]["output_shape"], [24, 35])

                session = ort.InferenceSession(
                    str(output_path), providers=["CPUExecutionProvider"]
                )
                with torch.inference_mode():
                    expected = model(parity_input).numpy().astype(np.float32)
                observed = np.asarray(
                    session.run(
                        [session.get_outputs()[0].name],
                        {session.get_inputs()[0].name: parity_input.numpy()},
                    )[0],
                    dtype=np.float32,
                )
                difference = np.abs(expected - observed)
                self.assertEqual(
                    int(np.count_nonzero(expected.argmax(axis=1) != observed.argmax(axis=1))),
                    0,
                    msg=f"{name}: argmax parity failed",
                )
                self.assertTrue(
                    np.allclose(expected, observed, rtol=1.0e-4, atol=1.0e-4),
                    msg=(
                        f"{name}: logit parity failed "
                        f"max_abs={float(difference.max())} "
                        f"mean_abs={float(difference.mean())}"
                    ),
                )
                graph = analyze_mobile_onnx_graph(output_path, batch_size=16)
                self.assertGreater(graph["depthwise_conv_count"], 0, msg=name)
                self.assertGreater(graph["pointwise_conv_count"], 0, msg=name)
                self.assertGreater(graph["known_macs_per_sample_estimate"], 0, msg=name)


if __name__ == "__main__":
    unittest.main()
