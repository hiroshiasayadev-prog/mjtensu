from __future__ import annotations

import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    StudyRunId,
    TaskId,
    TrainingRunId,
    TrainProtocolId,
)
from mldb.src.common.parameters import PublicParameterDeclaration
from mldb.src.training.protocol import (
    TrainProtocol,
    TrainProtocolImplementation,
    TrainProtocolStatus,
    validate_train_protocol_metadata,
)
from mldb.src.training.run import (
    TrainingRun,
    TrainingRunExecution,
    TrainingRunFailure,
    TrainingRunResult,
    TrainingRunStatus,
    TrainingRunStudyLineage,
    validate_training_run,
    validate_training_run_transition,
)
from mldb.src.training.weights import (
    CanonicalWeightsArtifact,
    accept_trained_state,
    load_canonical_weights,
    serialize_canonical_weights,
)


class _TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


class _TinyModuleWithExtraState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.extra = torch.nn.Parameter(torch.zeros(1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


class _RequiresGradStateModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def state_dict(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        return {"weight": self.weight}


class _NonStringKeyModule(torch.nn.Module):
    def state_dict(self, *args: object, **kwargs: object) -> dict[object, torch.Tensor]:
        return {1: torch.tensor([1.0])}


class _NonTensorStateModule(torch.nn.Module):
    def state_dict(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {"weight": "not-a-tensor"}


class CanonicalWeightsTests(unittest.TestCase):
    def test_accepts_valid_state_and_normalizes_to_detached_cpu_tensors(self) -> None:
        module = _RequiresGradStateModule()
        self.assertTrue(module.state_dict()["weight"].requires_grad)

        state = accept_trained_state(module, _RequiresGradStateModule)

        self.assertTrue(state)
        self.assertTrue(all(type(key) is str for key in state))
        for tensor in state.values():
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual("cpu", tensor.device.type)
            self.assertFalse(tensor.requires_grad)

    def test_rejects_non_string_state_key(self) -> None:
        with self.assertRaises(TypeError):
            accept_trained_state(_NonStringKeyModule(), _TinyModule)

    def test_rejects_non_tensor_state_value(self) -> None:
        with self.assertRaises(TypeError):
            accept_trained_state(_NonTensorStateModule(), _TinyModule)

    def test_rejects_checkpoint_wrapper_instead_of_plain_tensor_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "weights.pt"
            wrapper = {"state_dict": {"weight": torch.tensor([1.0])}}

            with self.assertRaises(TypeError):
                serialize_canonical_weights(wrapper, path)  # type: ignore[arg-type]

            torch.save(wrapper, path)
            with self.assertRaises(TypeError):
                load_canonical_weights(path, _TinyModule)

    def test_load_rejects_non_string_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "weights.pt"
            torch.save({1: torch.tensor([1.0])}, path)

            with self.assertRaises(TypeError):
                load_canonical_weights(path, _TinyModule)

    def test_round_trip_serialization_records_exact_bytes_and_loads_on_cpu(self) -> None:
        torch.manual_seed(123)
        trained = _TinyModule()
        state = accept_trained_state(trained, _TinyModule)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "weights.pt"
            artifact = serialize_canonical_weights(state, path)

            persisted = path.read_bytes()
            self.assertEqual("pytorch-state-dict", artifact.format)
            self.assertEqual("artifacts/weights.pt", artifact.path)
            self.assertEqual(len(persisted), artifact.bytes)
            self.assertEqual(hashlib.sha256(persisted).hexdigest(), artifact.sha256)

            with patch(
                "mldb.src.training.weights.torch.load",
                wraps=torch.load,
            ) as mocked_load:
                loaded = load_canonical_weights(path, _TinyModule)

            mocked_load.assert_called_once_with(
                path,
                map_location="cpu",
                weights_only=True,
            )

        loaded_state = loaded.state_dict()
        self.assertEqual(set(state), set(loaded_state))
        for key, expected in state.items():
            self.assertTrue(torch.equal(expected, loaded_state[key]))
            self.assertEqual("cpu", loaded_state[key].device.type)

    def test_strict_topology_mismatch_fails_acceptance_and_loading(self) -> None:
        trained = _TinyModule()
        state = accept_trained_state(trained, _TinyModule)

        with self.assertRaises(RuntimeError):
            accept_trained_state(trained, _TinyModuleWithExtraState)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "weights.pt"
            serialize_canonical_weights(state, path)

            with self.assertRaises(RuntimeError):
                load_canonical_weights(path, _TinyModuleWithExtraState)

    def test_current_torch_exposes_restricted_weights_load_api(self) -> None:
        signature = inspect.signature(torch.load)
        self.assertIn("weights_only", signature.parameters)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "probe.pt"
            torch.save({"x": torch.tensor([1.0])}, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual([1.0], loaded["x"].tolist())
        self.assertEqual("cpu", loaded["x"].device.type)


class TrainProtocolMetadataTests(unittest.TestCase):
    def test_valid_sealed_protocol_metadata(self) -> None:
        protocol = TrainProtocol(
            schema="mjtensu.mldb/train-protocol/v1",
            id=TrainProtocolId("tile-classifier-standard-v1"),
            status=TrainProtocolStatus.SEALED,
            task=TaskId("tile-shape-classification-35-v1"),
            name="Standard training",
            description="Train a compatible classifier.",
            implementation=TrainProtocolImplementation(
                entrypoint="train",
                sha256="a" * 64,
            ),
            parameters={
                "epochs": PublicParameterDeclaration(default=10),
                "options": PublicParameterDeclaration(
                    default={"amp": True, "tags": ["baseline", None]}
                ),
            },
        )

        self.assertTrue(validate_train_protocol_metadata(protocol).valid)

    def test_draft_protocol_may_omit_hash(self) -> None:
        protocol = TrainProtocol(
            schema="mjtensu.mldb/train-protocol/v1",
            id=TrainProtocolId("tile-classifier-standard-v2"),
            status=TrainProtocolStatus.DRAFT,
            task=TaskId("tile-shape-classification-35-v1"),
            name="Draft training",
            description="Draft.",
            implementation=TrainProtocolImplementation(entrypoint="train"),
            parameters={},
        )

        self.assertTrue(validate_train_protocol_metadata(protocol).valid)

    def test_invalid_metadata_and_non_json_parameter_default_are_rejected(self) -> None:
        protocol = TrainProtocol(
            schema="wrong",
            id=TrainProtocolId("no-revision"),
            status=TrainProtocolStatus.SEALED,
            task=TaskId("tile-shape-classification-35-v1"),
            name="",
            description="Invalid fixture.",
            implementation=TrainProtocolImplementation(
                entrypoint="wrong",  # type: ignore[arg-type]
                sha256=None,
            ),
            parameters={
                "learning_rate": PublicParameterDeclaration(default=float("nan")),
            },
        )

        report = validate_train_protocol_metadata(protocol)

        self.assertFalse(report.valid)
        codes = {issue.code for issue in report.issues}
        self.assertIn("train_protocol.schema.unsupported", codes)
        self.assertIn("train_protocol.id.invalid", codes)
        self.assertIn("train_protocol.name.invalid", codes)
        self.assertIn("train_protocol.implementation.entrypoint.invalid", codes)
        self.assertIn("train_protocol.implementation.sha256.required", codes)
        self.assertIn("train_protocol.parameter.default.invalid", codes)


class TrainingRunDomainTests(unittest.TestCase):
    def _running_run(self, **changes: object) -> TrainingRun:
        values: dict[str, object] = {
            "schema": "mjtensu.mldb/training-run/v1",
            "id": TrainingRunId("tr-20260908-001"),
            "status": TrainingRunStatus.RUNNING,
            "corpus": CorpusId("gray35-train-v1"),
            "architecture": ArchitectureId("plain-cnn-v1"),
            "train_protocol": TrainProtocolId("tile-classifier-standard-v1"),
            "parameters": {"epochs": 10, "learning_rate": 0.001},
            "execution": TrainingRunExecution(
                seed=42,
                started_at="2026-09-08T01:00:00+09:00",
            ),
        }
        values.update(changes)
        return TrainingRun(**values)  # type: ignore[arg-type]

    def test_initial_running_run_contains_resolved_parameters_and_exact_seed(self) -> None:
        run = self._running_run()

        self.assertTrue(validate_training_run(run).valid)
        self.assertEqual(
            {"epochs": 10, "learning_rate": 0.001},
            run.parameters,
        )
        self.assertEqual(42, run.execution.seed)
        self.assertIs(type(run.execution.seed), int)

    def test_boolean_seed_is_rejected(self) -> None:
        run = self._running_run(
            execution=TrainingRunExecution(
                seed=True,
                started_at="2026-09-08T01:00:00+09:00",
            )
        )

        self.assertFalse(validate_training_run(run).valid)

    def test_terminal_facts_and_result_conditionals(self) -> None:
        artifact = CanonicalWeightsArtifact(
            format="pytorch-state-dict",
            path="artifacts/weights.pt",
            sha256="b" * 64,
            bytes=123,
        )
        completed = self._running_run(
            status=TrainingRunStatus.COMPLETED,
            execution=TrainingRunExecution(
                seed=42,
                started_at="start",
                finished_at="finish",
            ),
            result=TrainingRunResult(weights=artifact),
        )
        self.assertTrue(validate_training_run(completed).valid)

        completed_without_result = self._running_run(
            status=TrainingRunStatus.COMPLETED,
            execution=TrainingRunExecution(
                seed=42,
                started_at="start",
                finished_at="finish",
            ),
        )
        self.assertFalse(validate_training_run(completed_without_result).valid)

        running_with_result = self._running_run(
            result=TrainingRunResult(weights=artifact),
        )
        self.assertFalse(validate_training_run(running_with_result).valid)

        failed_without_finished_at = self._running_run(
            status=TrainingRunStatus.FAILED,
            failure=TrainingRunFailure(type="RuntimeError", message="failed"),
        )
        self.assertFalse(validate_training_run(failed_without_finished_at).valid)

        running_with_finished_at = self._running_run(
            execution=TrainingRunExecution(
                seed=42,
                started_at="start",
                finished_at="finish",
            )
        )
        self.assertFalse(validate_training_run(running_with_finished_at).valid)

    def test_invalid_terminal_result_and_failure_metadata_are_rejected(self) -> None:
        invalid_artifact = CanonicalWeightsArtifact(
            format="wrong",  # type: ignore[arg-type]
            path="wrong.pt",  # type: ignore[arg-type]
            sha256="not-a-digest",
            bytes=True,
        )
        run = self._running_run(
            status=TrainingRunStatus.COMPLETED,
            execution=TrainingRunExecution(
                seed=42,
                started_at="start",
                finished_at="finish",
            ),
            result=TrainingRunResult(weights=invalid_artifact),
            failure=TrainingRunFailure(type="", message=""),
        )

        report = validate_training_run(run)

        self.assertFalse(report.valid)
        codes = {issue.code for issue in report.issues}
        self.assertIn("training_run.result.weights.format.invalid", codes)
        self.assertIn("training_run.result.weights.path.invalid", codes)
        self.assertIn("training_run.result.weights.sha256.invalid", codes)
        self.assertIn("training_run.result.weights.bytes.invalid", codes)
        self.assertIn("training_run.failure.type.invalid", codes)
        self.assertIn("training_run.failure.message.invalid", codes)

    def test_study_lineage_shape(self) -> None:
        valid = self._running_run(
            study=TrainingRunStudyLineage(
                run=StudyRunId("sr-20260908-001"),
                trial="trial-0001",
            )
        )
        self.assertTrue(validate_training_run(valid).valid)

        invalid = self._running_run(
            study=TrainingRunStudyLineage(
                run=StudyRunId("sr-20260908-000"),
                trial="trial-0000",
            )
        )
        self.assertFalse(validate_training_run(invalid).valid)

    def test_only_running_to_terminal_transitions_are_valid(self) -> None:
        for target in (
            TrainingRunStatus.COMPLETED,
            TrainingRunStatus.FAILED,
            TrainingRunStatus.CANCELLED,
        ):
            with self.subTest(target=target):
                self.assertTrue(
                    validate_training_run_transition(
                        TrainingRunStatus.RUNNING,
                        target,
                    ).valid
                )

        self.assertFalse(
            validate_training_run_transition(
                TrainingRunStatus.RUNNING,
                TrainingRunStatus.RUNNING,
            ).valid
        )
        self.assertFalse(
            validate_training_run_transition(
                TrainingRunStatus.COMPLETED,
                TrainingRunStatus.RUNNING,
            ).valid
        )
        self.assertFalse(
            validate_training_run_transition(
                TrainingRunStatus.FAILED,
                TrainingRunStatus.COMPLETED,
            ).valid
        )


if __name__ == "__main__":
    unittest.main()
