from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

RECOGNITION_TOOLS = Path(__file__).resolve().parents[1]
if str(RECOGNITION_TOOLS) not in sys.path:
    sys.path.insert(0, str(RECOGNITION_TOOLS))

from rotation_classifier_experiment_models import (
    RICCNNTileClassifier,
    RICConv2d,
    RotEqConv2d,
    RotEqNetTileClassifier,
    SConv2d,
    SConvTileClassifier,
    build_experiment_model,
    build_polar_ring_grids,
    build_ric_sampling_grid,
    build_square_ring_row_major_gather,
)
from run_rotation_classifier_experiment import (
    CONDITIONS,
    EXPERIMENT_IMPLEMENTATION_VERSION,
    deterministic_random360_angles,
    load_checkpoint_model,
    prior_result_is_reusable,
)


class RotationClassifierExperimentTest(unittest.TestCase):
    def test_condition_matrix_contains_five_families_and_ten_accuracy_conditions(self) -> None:
        self.assertEqual(len(CONDITIONS), 10)
        self.assertEqual(
            {condition.architecture for condition in CONDITIONS},
            {"c8", "plain", "roteqnet", "riccnn", "sconv"},
        )
        for architecture in ("plain", "roteqnet", "riccnn", "sconv"):
            augmentations = {
                condition.augmentation
                for condition in CONDITIONS
                if condition.architecture == architecture
            }
            self.assertEqual(augmentations, {"none", "random360"})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for escnn device regression")
    def test_c8_checkpoint_reload_enters_eval_only_after_device_move(self) -> None:
        try:
            import escnn  # noqa: F401
        except ImportError:
            self.skipTest("escnn is required for C8 checkpoint reload regression")

        # Reproduce the runner state that triggered the overnight failure: an escnn C8
        # model has already expanded/evaluated its basis on CUDA, then a second instance
        # is reconstructed from a checkpoint in the same Python process.
        source = build_experiment_model("c8", class_count=35, image_size=64).cuda()
        source.eval()
        with torch.inference_mode():
            source(torch.zeros((1, 1, 64, 64), device="cuda"))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "c8.pt"
            torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)

            loaded_cuda = load_checkpoint_model(
                checkpoint_path,
                architecture="c8",
                class_count=35,
                image_size=64,
                device="cuda",
            )
            self.assertEqual(next(loaded_cuda.parameters()).device.type, "cuda")
            with torch.inference_mode():
                cuda_logits = loaded_cuda(
                    torch.zeros((1, 1, 64, 64), device="cuda")
                )
            self.assertEqual(tuple(cuda_logits.shape), (1, 35))

            loaded_cpu = load_checkpoint_model(
                checkpoint_path,
                architecture="c8",
                class_count=35,
                image_size=64,
                device="cpu",
            )
            self.assertEqual(next(loaded_cpu.parameters()).device.type, "cpu")
            with torch.inference_mode():
                cpu_logits = loaded_cpu(torch.zeros((1, 1, 64, 64)))
            self.assertEqual(tuple(cpu_logits.shape), (1, 35))

        del source, loaded_cuda, loaded_cpu
        torch.cuda.empty_cache()

    def test_resume_does_not_reuse_old_research_architecture_results(self) -> None:
        plain = next(condition for condition in CONDITIONS if condition.name == "plain-noaug")
        roteq = next(condition for condition in CONDITIONS if condition.name == "roteqnet-noaug")
        old_plain = {"status": "completed", "condition": plain.__dict__}
        old_roteq = {"status": "completed", "condition": roteq.__dict__}
        current_roteq = {
            "status": "completed",
            "condition": roteq.__dict__,
            "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        }
        self.assertTrue(prior_result_is_reusable(plain, old_plain))
        self.assertFalse(prior_result_is_reusable(roteq, old_roteq))
        self.assertTrue(prior_result_is_reusable(roteq, current_roteq))

    def test_random360_angles_are_sample_and_epoch_deterministic(self) -> None:
        sample_ids = ["sample-a", "sample-b", "sample-c"]
        first = deterministic_random360_angles(sample_ids, seed=42, epoch=7)
        repeated = deterministic_random360_angles(sample_ids, seed=42, epoch=7)
        next_epoch = deterministic_random360_angles(sample_ids, seed=42, epoch=8)
        reordered = deterministic_random360_angles(list(reversed(sample_ids)), seed=42, epoch=7)

        np.testing.assert_array_equal(first, repeated)
        self.assertTrue(np.all(first >= -180.0))
        self.assertTrue(np.all(first < 180.0))
        self.assertFalse(np.array_equal(first, next_epoch))
        np.testing.assert_allclose(first, reordered[::-1], rtol=0.0, atol=0.0)

    def test_roteqnet_public_topology_is_preserved_for_64x64_classifier(self) -> None:
        model = RotEqNetTileClassifier(class_count=35)
        self.assertEqual(model.channels, (6, 16, 32))
        self.assertEqual(model.n_angles, 17)
        self.assertEqual(model.rot1.kernel_size, 9)
        self.assertEqual(model.rot2.kernel_size, 9)
        self.assertEqual(model.rot3.kernel_size, 9)
        self.assertEqual((model.rot1.padding, model.rot2.padding, model.rot3.padding), (4, 4, 1))
        self.assertEqual(model.norm1.momentum, 0.5)
        self.assertEqual(model.norm2.momentum, 0.5)
        self.assertIsInstance(model.head[0], torch.nn.Conv2d)
        self.assertEqual(model.head[0].in_channels, 32)
        self.assertEqual(model.head[0].out_channels, 128)
        self.assertIsInstance(model.head[3], torch.nn.Dropout2d)
        self.assertAlmostEqual(float(model.head[3].p), 0.7)
        self.assertIsInstance(model.head[4], torch.nn.Conv2d)
        self.assertEqual(model.head[4].out_channels, 35)

    def test_roteqnet_logits_are_invariant_to_quarter_turn_for_four_orientations(self) -> None:
        torch.manual_seed(1234)
        model = RotEqNetTileClassifier(
            class_count=5, channels=(2, 3, 4), n_angles=4
        )
        model.eval()
        image = torch.randn((2, 1, 64, 64), dtype=torch.float32)
        with torch.inference_mode():
            original = model(image)
            rotated = model(torch.rot90(image, 1, dims=(-2, -1)))
        torch.testing.assert_close(original, rotated, rtol=1.0e-4, atol=1.0e-4)

    def test_roteqnet_operator_emits_vector_field_after_orientation_pooling(self) -> None:
        layer = RotEqConv2d(1, 2, 3, n_angles=4, mode=1)
        image = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
        image[:, :, 4, 2:7] = 1.0
        u, v = layer(image)
        self.assertEqual(tuple(u.shape), (1, 2, 9, 9))
        self.assertEqual(tuple(v.shape), (1, 2, 9, 9))
        self.assertEqual(layer.rotation_grids.shape[0], 4)
        magnitude = torch.sqrt(u.square() + v.square())
        self.assertTrue(torch.isfinite(magnitude).all())

    def test_ric_grid_uses_position_relative_to_feature_map_center(self) -> None:
        grid = build_ric_sampling_grid(9, 9)
        self.assertEqual(tuple(grid.shape), (9, 9, 9, 2))
        # The center sample is the fifth point and exactly samples the current pixel.
        center_sample = grid[4]
        self.assertAlmostEqual(float(center_sample[4, 4, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(center_sample[4, 4, 1]), 0.0, places=6)
        # Radial direction changes with spatial position: the first neighbor above the
        # center and to the right of the center cannot share the same sampling offset.
        above = grid[0, 1, 4]
        right = grid[0, 4, 7]
        self.assertFalse(torch.allclose(above, right))

    def test_ric_operator_has_fixed_nonlearned_sampling_coordinates(self) -> None:
        layer = RICConv2d(1, 2, height=9, width=9)
        parameter_names = {name for name, _ in layer.named_parameters()}
        self.assertEqual(parameter_names, {"weight"})
        image = torch.randn((2, 1, 9, 9), dtype=torch.float32)
        output = layer(image)
        self.assertEqual(tuple(output.shape), (2, 2, 9, 9))

    def test_ric_grid_sample_form_matches_author_deform_conv_offsets(self) -> None:
        try:
            from torchvision.ops import deform_conv2d
        except (ImportError, RuntimeError, OSError):
            self.skipTest("torchvision deform_conv2d is unavailable")

        height = width = 9
        layer = RICConv2d(1, 2, height=height, width=width, bias=False)
        image = torch.randn((2, 1, height, width), dtype=torch.float32)
        row, col = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        center_row = (height - 1.0) / 2.0
        center_col = (width - 1.0) / 2.0
        # The author code names the row axis x and the column axis y, then uses
        # atan2(delta_y, delta_x), rounded to four decimals.
        theta = torch.atan2(col - center_col, row - center_row) % (2.0 * np.pi)
        theta = torch.round(theta * 10000.0) / 10000.0
        kernel_positions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
        offsets = torch.zeros((2, 18, height, width), dtype=torch.float32)
        ring_index = 0
        for kernel_index, (base_row, base_col) in enumerate(kernel_positions):
            if kernel_index == 4:
                continue
            angle = theta + ring_index * (np.pi / 4.0)
            desired_row = torch.cos(angle)
            desired_col = torch.sin(angle)
            offsets[:, 2 * kernel_index] = desired_row - float(base_row)
            offsets[:, 2 * kernel_index + 1] = desired_col - float(base_col)
            ring_index += 1

        expected = deform_conv2d(
            input=image,
            offset=offsets,
            weight=layer.weight,
            padding=(1, 1),
        )
        observed = layer(image)
        torch.testing.assert_close(observed, expected, rtol=3.0e-4, atol=3.0e-4)

    def test_sconv_polar_sampling_uses_8r_points_per_ring(self) -> None:
        rings = build_polar_ring_grids(9, 9, 5)
        self.assertEqual([int(ring.shape[0]) for ring in rings], [8, 16])
        self.assertEqual(1 + sum(int(ring.shape[0]) for ring in rings), 25)

    def test_sconv_ring_sorted_values_are_restored_to_square_ring_row_major_positions(self) -> None:
        self.assertEqual(
            build_square_ring_row_major_gather(3).tolist(),
            [1, 2, 3, 4, 0, 5, 6, 7, 8],
        )
        gather5 = build_square_ring_row_major_gather(5).tolist()
        self.assertEqual(gather5[12], 0)  # center
        inner_positions = [6, 7, 8, 11, 13, 16, 17, 18]
        self.assertEqual([gather5[index] for index in inner_positions], list(range(1, 9)))
        outer_positions = [
            index
            for index in range(25)
            if max(abs(index // 5 - 2), abs(index % 5 - 2)) == 2
        ]
        self.assertEqual([gather5[index] for index in outer_positions], list(range(9, 25)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for full research-model preflight")
    def test_full_research_backbones_complete_one_cuda_backward_step(self) -> None:
        factories = (
            lambda: RotEqNetTileClassifier(class_count=35),
            lambda: RICCNNTileClassifier(class_count=35, image_size=64),
            lambda: SConvTileClassifier(class_count=35, image_size=64),
        )
        image = torch.randn((1, 1, 64, 64), device="cuda", dtype=torch.float32)
        target = torch.tensor([3], device="cuda", dtype=torch.int64)
        for factory in factories:
            model = factory().cuda().train()
            logits = model(image)
            self.assertEqual(tuple(logits.shape), (1, 35))
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            del model, logits, loss, gradients
            torch.cuda.empty_cache()

    def test_small_specialized_backbones_preserve_35_class_output_contract(self) -> None:
        models = (
            RotEqNetTileClassifier(class_count=35, channels=(2, 4, 8), n_angles=4),
            RICCNNTileClassifier(class_count=35, channels=(2, 4, 8, 8), image_size=64),
            SConvTileClassifier(class_count=35, channels=(2, 4, 8, 8), image_size=64),
        )
        image = torch.randn((1, 1, 64, 64), dtype=torch.float32)
        for model in models:
            model.eval()
            with torch.inference_mode():
                output = model(image)
            self.assertEqual(tuple(output.shape), (1, 35))

    def test_small_specialized_backbones_export_to_opset16_and_run_in_ort(self) -> None:
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnx and onnxruntime are required for deployment preflight")

        models = (
            RotEqNetTileClassifier(class_count=35, channels=(2, 4, 8), n_angles=4),
            RICCNNTileClassifier(class_count=35, channels=(2, 4, 8, 8), image_size=64),
            SConvTileClassifier(class_count=35, channels=(2, 4, 8, 8), image_size=64),
        )
        example = torch.randn((1, 1, 64, 64), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, model in enumerate(models):
                model.eval()
                output_path = root / f"model-{index}.onnx"
                torch.onnx.export(
                    model,
                    example,
                    str(output_path),
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=["images"],
                    output_names=["logits"],
                    dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                )
                exported = onnx.load(str(output_path))
                onnx.checker.check_model(exported)
                session = ort.InferenceSession(
                    str(output_path), providers=["CPUExecutionProvider"]
                )
                observed = session.run(
                    [session.get_outputs()[0].name],
                    {session.get_inputs()[0].name: example.numpy()},
                )[0]
                with torch.inference_mode():
                    expected = model(example).numpy()
                np.testing.assert_allclose(expected, observed, rtol=1.0e-4, atol=1.0e-4)

    def test_full_research_backbones_export_to_opset16_and_match_ort(self) -> None:
        """Exercise the exact full-size deployment graphs used by the overnight run.

        The first INV-007 run only smoke-tested reduced research models, while the
        trained full RotEqNet graph later failed the PyTorch/ORT parity gate. This test
        intentionally exports the production-size experiment architectures so graph
        size/orientation count/operator composition cannot hide behind the small smoke
        models.
        """
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnx and onnxruntime are required for deployment preflight")

        factories = (
            ("roteqnet", lambda: RotEqNetTileClassifier(class_count=35)),
            ("riccnn", lambda: RICCNNTileClassifier(class_count=35, image_size=64)),
            ("sconv", lambda: SConvTileClassifier(class_count=35, image_size=64)),
        )
        generator = torch.Generator().manual_seed(20260830)
        example = torch.randn((2, 1, 64, 64), generator=generator, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, factory in factories:
                model = factory().eval()
                output_path = root / f"full-{name}.onnx"
                torch.onnx.export(
                    model,
                    example[:1],
                    str(output_path),
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=["images"],
                    output_names=["logits"],
                    dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                )
                exported = onnx.load(str(output_path))
                onnx.checker.check_model(exported)
                session = ort.InferenceSession(
                    str(output_path), providers=["CPUExecutionProvider"]
                )
                with torch.inference_mode():
                    expected = model(example).numpy().astype(np.float32)
                observed = np.asarray(
                    session.run(
                        [session.get_outputs()[0].name],
                        {session.get_inputs()[0].name: example.numpy()},
                    )[0],
                    dtype=np.float32,
                )
                difference = np.abs(expected - observed)
                mismatches = int(
                    np.count_nonzero(expected.argmax(axis=1) != observed.argmax(axis=1))
                )
                self.assertEqual(mismatches, 0, msg=f"{name} argmax parity failed")
                self.assertTrue(
                    np.allclose(expected, observed, rtol=1.0e-4, atol=1.0e-4),
                    msg=(
                        f"{name} logit parity failed: "
                        f"max_abs={float(difference.max())} "
                        f"mean_abs={float(difference.mean())}"
                    ),
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for trained-state ONNX parity preflight")
    def test_full_research_backbones_match_ort_after_cuda_training_steps(self) -> None:
        """Mimic the exact failure class seen only after training in the first run."""
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnx and onnxruntime are required for deployment preflight")

        factories = (
            ("roteqnet", lambda: RotEqNetTileClassifier(class_count=35)),
            ("riccnn", lambda: RICCNNTileClassifier(class_count=35, image_size=64)),
            ("sconv", lambda: SConvTileClassifier(class_count=35, image_size=64)),
        )
        cuda_generator = torch.Generator(device="cuda").manual_seed(20260831)
        train_images = torch.randn(
            (2, 1, 64, 64), generator=cuda_generator, device="cuda", dtype=torch.float32
        )
        targets = torch.tensor([3, 17], device="cuda", dtype=torch.int64)
        parity_input = torch.linspace(
            -2.0, 2.0, steps=2 * 64 * 64, dtype=torch.float32
        ).reshape(2, 1, 64, 64)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, factory in factories:
                model = factory().cuda().train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(train_images)
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                    loss.backward()
                    optimizer.step()
                model = model.cpu().eval()

                output_path = root / f"trained-{name}.onnx"
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
                exported = onnx.load(str(output_path))
                onnx.checker.check_model(exported)
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
                    msg=f"{name} trained-state argmax parity failed",
                )
                self.assertTrue(
                    np.allclose(expected, observed, rtol=1.0e-4, atol=1.0e-4),
                    msg=(
                        f"{name} trained-state logit parity failed: "
                        f"max_abs={float(difference.max())} "
                        f"mean_abs={float(difference.mean())}"
                    ),
                )
                del model, optimizer, logits, loss
                torch.cuda.empty_cache()

    def test_sconv_center_response_is_invariant_to_quarter_turn_ring_permutation(self) -> None:
        layer = SConv2d(1, 1, 3, height=9, width=9, bias=False)
        with torch.no_grad():
            # Distinct weights make this test meaningful: invariance comes from sorting,
            # not from an accidentally symmetric convolution kernel.
            layer.weight.copy_(torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3))
        image = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
        image[0, 0, 3:6, 3:6] = torch.tensor(
            [[1.0, 7.0, 3.0], [8.0, 5.0, 2.0], [6.0, 4.0, 9.0]]
        )
        rotated = torch.rot90(image, 1, dims=(-2, -1))
        response = layer(image)[0, 0, 4, 4]
        rotated_response = layer(rotated)[0, 0, 4, 4]
        torch.testing.assert_close(response, rotated_response, rtol=1.0e-6, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
