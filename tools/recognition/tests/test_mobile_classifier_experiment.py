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

from mobile_classifier_experiment_models import (
    MobileNetV3SmallTileClassifier,
    ShuffleNetV2TileClassifier,
    build_mobile_classifier,
    channel_shuffle,
    describe_mobile_classifier,
)
from run_mobile_classifier_experiment import (
    CONDITIONS,
    EXPERIMENT_IMPLEMENTATION_VERSION,
    analyze_mobile_onnx_graph,
    prior_result_is_reusable,
    smoke_onnx_dynamic_batch,
)
from run_rotation_classifier_experiment import deterministic_random360_angles


class MobileClassifierExperimentTest(unittest.TestCase):
    def test_condition_matrix_contains_four_mobile_candidates(self) -> None:
        self.assertEqual(
            [condition.name for condition in CONDITIONS],
            [
                "shufflenet-v2-0.5x",
                "shufflenet-v2-1.0x",
                "mobilenet-v3-small-0.5x",
                "mobilenet-v3-small-1.0x",
            ],
        )
        self.assertEqual(
            {(condition.family, condition.width_mult) for condition in CONDITIONS},
            {
                ("shufflenet-v2", 0.5),
                ("shufflenet-v2", 1.0),
                ("mobilenet-v3-small", 0.5),
                ("mobilenet-v3-small", 1.0),
            },
        )

    def test_every_candidate_preserves_dynamic_batch_classifier_contract(self) -> None:
        image = torch.randn((3, 1, 64, 64), dtype=torch.float32)
        for condition in CONDITIONS:
            model = build_mobile_classifier(condition.name, class_count=35).eval()
            with torch.inference_mode():
                logits = model(image)
            self.assertEqual(tuple(logits.shape), (3, 35), msg=condition.name)

    def test_channel_shuffle_permutates_two_groups_as_standard_shufflenet(self) -> None:
        source = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)
        observed = channel_shuffle(source, 2).flatten().tolist()
        self.assertEqual(observed, [0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0])

    def test_shufflenet_v2_uses_standard_stage_schedules(self) -> None:
        half = ShuffleNetV2TileClassifier(class_count=35, width_mult=0.5)
        full = ShuffleNetV2TileClassifier(class_count=35, width_mult=1.0)
        self.assertEqual(half.stage_channels, (24, 48, 96, 192, 1024))
        self.assertEqual(full.stage_channels, (24, 116, 232, 464, 1024))
        self.assertEqual(half.stage_repeats, (4, 8, 4))
        self.assertEqual(full.stage_repeats, (4, 8, 4))
        self.assertEqual(half.conv1[0].in_channels, 1)
        self.assertEqual(full.conv1[0].in_channels, 1)

    def test_shufflenet_v2_contains_true_depthwise_convolutions(self) -> None:
        model = ShuffleNetV2TileClassifier(class_count=35, width_mult=1.0)
        depthwise = [
            module
            for module in model.modules()
            if isinstance(module, nn.Conv2d)
            and module.groups > 1
            and module.groups == module.in_channels
            and module.in_channels == module.out_channels
        ]
        self.assertGreater(len(depthwise), 0)
        self.assertTrue(all(module.kernel_size == (3, 3) for module in depthwise))

    def test_mobilenet_v3_small_preserves_standard_block_features(self) -> None:
        model = MobileNetV3SmallTileClassifier(class_count=35, width_mult=1.0)
        self.assertEqual(len(model.block_configs), 11)
        self.assertEqual(model.block_configs[0].kernel, 3)
        self.assertEqual(model.block_configs[0].stride, 2)
        self.assertTrue(model.block_configs[0].use_se)
        self.assertEqual(model.block_configs[-1].out_channels, 96)
        self.assertEqual(model.last_conv_channels, 576)
        self.assertEqual(model.last_channel, 1024)
        self.assertTrue(any(isinstance(module, nn.Hardswish) for module in model.modules()))
        self.assertTrue(any(isinstance(module, nn.Hardsigmoid) for module in model.modules()))
        batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertTrue(batch_norms)
        self.assertTrue(all(abs(module.eps - 0.001) < 1.0e-12 for module in batch_norms))
        self.assertTrue(all(abs(module.momentum - 0.01) < 1.0e-12 for module in batch_norms))

    def test_mobilenet_v3_small_contains_true_depthwise_convolutions(self) -> None:
        model = MobileNetV3SmallTileClassifier(class_count=35, width_mult=0.5)
        depthwise = [
            module
            for module in model.modules()
            if isinstance(module, nn.Conv2d)
            and module.groups > 1
            and module.groups == module.in_channels
            and module.in_channels == module.out_channels
        ]
        self.assertEqual(len(depthwise), 11)
        self.assertTrue(all(module.kernel_size in {(3, 3), (5, 5)} for module in depthwise))

    def test_model_descriptions_record_family_width_and_topology(self) -> None:
        for condition in CONDITIONS:
            model = build_mobile_classifier(condition.name, class_count=35)
            description = describe_mobile_classifier(model, condition.name)
            self.assertEqual(description.name, condition.name)
            self.assertEqual(description.family, condition.family)
            self.assertEqual(description.width_mult, condition.width_mult)
            self.assertGreater(description.parameter_count, 0)
            self.assertEqual(
                description.parameter_count, description.trainable_parameter_count
            )
            self.assertEqual(description.details["input_channels"], 1)
            self.assertEqual(description.details["input_size"], 64)
            self.assertEqual(description.details["class_count"], 35)

    def test_random360_assignment_is_architecture_independent(self) -> None:
        sample_ids = ["sample-a", "sample-b", "sample-c", "sample-d"]
        expected = deterministic_random360_angles(sample_ids, seed=42, epoch=73)
        for _condition in CONDITIONS:
            observed = deterministic_random360_angles(sample_ids, seed=42, epoch=73)
            np.testing.assert_array_equal(observed, expected)

    def test_resume_requires_exact_inv011_implementation_version(self) -> None:
        condition = CONDITIONS[0]
        current = {
            "status": "completed",
            "condition": condition.__dict__,
            "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        }
        old = {
            "status": "completed",
            "condition": condition.__dict__,
            "implementation_version": "inv011-mobile-v0",
        }
        self.assertTrue(prior_result_is_reusable(condition, current))
        self.assertFalse(prior_result_is_reusable(condition, old))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for mobile classifier backward preflight")
    def test_every_candidate_completes_cuda_backward_step(self) -> None:
        image = torch.randn((2, 1, 64, 64), device="cuda", dtype=torch.float32)
        target = torch.tensor([3, 17], device="cuda", dtype=torch.int64)
        for condition in CONDITIONS:
            model = build_mobile_classifier(condition.name, class_count=35).cuda().train()
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

    def test_every_candidate_exports_dynamic_opset16_and_matches_ort(self) -> None:
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
            for condition in CONDITIONS:
                model = build_mobile_classifier(condition.name, class_count=35).eval()
                output_path = root / f"{condition.name}.onnx"
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
                    batch_sizes=(1, 16),
                    seed=42,
                )
                self.assertEqual(smoke["batches"]["16"]["output_shape"], [16, 35])

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
                    msg=f"{condition.name}: argmax parity failed",
                )
                self.assertTrue(
                    np.allclose(expected, observed, rtol=1.0e-4, atol=1.0e-4),
                    msg=(
                        f"{condition.name}: logit parity failed "
                        f"max_abs={float(difference.max())} "
                        f"mean_abs={float(difference.mean())}"
                    ),
                )

                graph = analyze_mobile_onnx_graph(output_path, batch_size=16)
                self.assertGreater(graph["depthwise_conv_count"], 0, msg=condition.name)
                self.assertGreater(graph["pointwise_conv_count"], 0, msg=condition.name)
                self.assertGreater(
                    graph["known_macs_per_sample_estimate"], 0, msg=condition.name
                )


if __name__ == "__main__":
    unittest.main()
