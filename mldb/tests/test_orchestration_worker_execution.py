from __future__ import annotations

import ast
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from mldb.src.evaluation.interface import EvaluationContext, EvaluationResult
from mldb.src.model.loading import ModelHandle
from mldb.src.orchestration import worker_execution as execution
from mldb.src.orchestration.worker_api import EvaluationAssignment, TrainingAssignment
from mldb.src.orchestration.worker_execution import (
    EvaluationExecutionFiles,
    TrainingExecutionCandidate,
    TrainingExecutionFiles,
    execute_evaluation_attempt,
    execute_training_attempt,
)
from mldb.src.runtime.catalog_handles import ArchitectureHandle, CorpusHandle, TaskHandle
from mldb.src.runtime.definition_handles import EvaluationProtocolHandle, TrainProtocolHandle
from mldb.src.training.protocol import TrainContext
from mldb.src.training.weights import accept_trained_state, serialize_canonical_weights
from mldb.tests.signature_guard import compare_module_signatures


def _training_assignment() -> TrainingAssignment:
    return TrainingAssignment(
        attempt_id=11,
        run_id="tr-20260908-011",
        lease_token="lease-training",
        lease_until="later",
        kind="training",
        task=object(),
        corpus=object(),
        architecture=object(),
        protocol=object(),
        seed=123,
        parameters={"epochs": 7, "amp": False},
        corpus_artifact=object(),
        architecture_implementation=object(),
        train_protocol_implementation=object(),
    )


def _evaluation_assignment() -> EvaluationAssignment:
    return EvaluationAssignment(
        attempt_id=12,
        run_id="ev-20260908-012",
        lease_token="lease-evaluation",
        lease_until="later",
        kind="evaluation",
        task=object(),
        corpus=object(),
        model=object(),
        training_run=object(),
        model_architecture=object(),
        protocol=object(),
        parameters={"threshold": 0.25},
        corpus_artifact=object(),
        model_architecture_implementation=object(),
        model_weights=object(),
        evaluation_protocol_implementation=object(),
    )


def _training_files(root: Path) -> TrainingExecutionFiles:
    return TrainingExecutionFiles(
        corpus_artifact=root / "corpus.sqlite",
        architecture_implementation=root / "architecture.py",
        train_protocol_implementation=root / "train.py",
        work_dir=root / "work",
        weights_candidate_path=root / "weights-candidate.pt",
    )


def _evaluation_files(root: Path) -> EvaluationExecutionFiles:
    return EvaluationExecutionFiles(
        corpus_artifact=root / "corpus.sqlite",
        model_architecture_implementation=root / "architecture.py",
        model_weights=root / "weights.pt",
        evaluation_protocol_implementation=root / "evaluate.py",
        work_dir=root / "work",
    )


class TrainingExecutionTests(unittest.TestCase):
    def test_success_constructs_exact_worker_handles_context_and_candidate(self) -> None:
        assignment = _training_assignment()
        with tempfile.TemporaryDirectory() as temporary:
            files = _training_files(Path(temporary))
            files.work_dir.mkdir()
            trained = torch.nn.Linear(2, 2)
            train = Mock(return_value=trained)
            architecture_build = Mock(return_value=torch.nn.Linear(2, 2))

            with (
                patch.object(execution, "load_train_entrypoint", return_value=train) as load_train,
                patch.object(
                    execution,
                    "load_architecture_build",
                    return_value=architecture_build,
                ) as load_architecture,
                patch.object(
                    execution,
                    "accept_trained_state",
                    wraps=accept_trained_state,
                ) as accept,
                patch.object(
                    execution,
                    "serialize_canonical_weights",
                    wraps=serialize_canonical_weights,
                ) as serialize,
            ):
                candidate = execute_training_attempt(assignment, files)

            self.assertIsInstance(candidate, TrainingExecutionCandidate)
            self.assertEqual(files.weights_candidate_path, candidate.local_weights_path)
            self.assertEqual("pytorch-state-dict", candidate.artifact.format)
            self.assertEqual("artifacts/weights.pt", candidate.artifact.path)
            payload = files.weights_candidate_path.read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), candidate.artifact.sha256)
            self.assertEqual(len(payload), candidate.artifact.bytes)

            load_train.assert_called_once()
            load_architecture.assert_called_once()
            train.assert_called_once()
            accept.assert_called_once()
            serialize.assert_called_once()
            self.assertEqual(files.weights_candidate_path, serialize.call_args.args[1])

            protocol_handle = load_train.call_args.args[0]
            architecture_handle = load_architecture.call_args.args[0]
            context = train.call_args.args[0]
            self.assertIsInstance(protocol_handle, TrainProtocolHandle)
            self.assertIs(protocol_handle.metadata, assignment.protocol)
            self.assertIsNone(protocol_handle.metadata_path)
            self.assertEqual(files.train_protocol_implementation, protocol_handle.implementation_path)
            self.assertIsInstance(architecture_handle, ArchitectureHandle)
            self.assertIs(architecture_handle.metadata, assignment.architecture)
            self.assertIsNone(architecture_handle.metadata_path)
            self.assertEqual(files.architecture_implementation, architecture_handle.implementation_path)
            self.assertIsInstance(context, TrainContext)
            self.assertIsInstance(context.task, TaskHandle)
            self.assertIs(context.task.metadata, assignment.task)
            self.assertIsNone(context.task.metadata_path)
            self.assertIsInstance(context.corpus, CorpusHandle)
            self.assertIs(context.corpus.metadata, assignment.corpus)
            self.assertIsNone(context.corpus.metadata_path)
            self.assertEqual(files.corpus_artifact, context.corpus.artifact_path)
            self.assertIsNone(context.corpus.builder_path)
            self.assertIs(context.architecture, architecture_handle)
            self.assertEqual(assignment.seed, context.seed)
            self.assertIs(context.parameters, assignment.parameters)
            self.assertEqual(files.work_dir, context.work_dir)
            accept.assert_called_once_with(trained, architecture_build)

    def test_train_loader_failure_does_not_run_domain(self) -> None:
        assignment = _training_assignment()
        files = _training_files(Path("worker"))
        train = Mock()
        with (
            patch.object(
                execution,
                "load_train_entrypoint",
                side_effect=RuntimeError("train loader failed"),
            ),
            patch.object(execution, "load_architecture_build") as load_architecture,
            self.assertRaisesRegex(RuntimeError, "train loader failed"),
        ):
            execute_training_attempt(assignment, files)
        train.assert_not_called()
        load_architecture.assert_not_called()
    def test_architecture_loader_failure_does_not_run_domain(self) -> None:
        assignment = _training_assignment()
        files = _training_files(Path("worker"))
        train = Mock()
        with (
            patch.object(execution, "load_train_entrypoint", return_value=train),
            patch.object(
                execution,
                "load_architecture_build",
                side_effect=RuntimeError("architecture loader failed"),
            ),
            patch.object(execution, "accept_trained_state") as accept,
            patch.object(execution, "serialize_canonical_weights") as serialize,
            self.assertRaisesRegex(RuntimeError, "architecture loader failed"),
        ):
            execute_training_attempt(assignment, files)
        train.assert_not_called()
        accept.assert_not_called()
        serialize.assert_not_called()

    def test_domain_failure_escapes_once_without_acceptance_or_serialization(self) -> None:
        assignment = _training_assignment()
        files = _training_files(Path("worker"))
        train = Mock(side_effect=RuntimeError("train failed"))
        with (
            patch.object(execution, "load_train_entrypoint", return_value=train),
            patch.object(execution, "load_architecture_build", return_value=Mock()),
            patch.object(execution, "accept_trained_state") as accept,
            patch.object(execution, "serialize_canonical_weights") as serialize,
            self.assertRaisesRegex(RuntimeError, "train failed"),
        ):
            execute_training_attempt(assignment, files)
        train.assert_called_once()
        accept.assert_not_called()
        serialize.assert_not_called()

    def test_acceptance_failure_escapes_after_one_domain_call(self) -> None:
        assignment = _training_assignment()
        files = _training_files(Path("worker"))
        train = Mock(return_value=object())
        with (
            patch.object(execution, "load_train_entrypoint", return_value=train),
            patch.object(execution, "load_architecture_build", return_value=Mock()),
            patch.object(
                execution,
                "accept_trained_state",
                side_effect=RuntimeError("state rejected"),
            ) as accept,
            patch.object(execution, "serialize_canonical_weights") as serialize,
            self.assertRaisesRegex(RuntimeError, "state rejected"),
        ):
            execute_training_attempt(assignment, files)
        train.assert_called_once()
        accept.assert_called_once()
        serialize.assert_not_called()

    def test_serialization_failure_escapes_without_rerunning_domain(self) -> None:
        assignment = _training_assignment()
        files = _training_files(Path("worker"))
        train = Mock(return_value=object())
        accepted = object()
        with (
            patch.object(execution, "load_train_entrypoint", return_value=train),
            patch.object(execution, "load_architecture_build", return_value=Mock()),
            patch.object(execution, "accept_trained_state", return_value=accepted) as accept,
            patch.object(
                execution,
                "serialize_canonical_weights",
                side_effect=OSError("serialize failed"),
            ) as serialize,
            self.assertRaisesRegex(OSError, "serialize failed"),
        ):
            execute_training_attempt(assignment, files)
        train.assert_called_once()
        accept.assert_called_once()
        serialize.assert_called_once_with(accepted, files.weights_candidate_path)


class EvaluationExecutionTests(unittest.TestCase):
    def test_success_constructs_exact_worker_handles_and_returns_same_result(self) -> None:
        assignment = _evaluation_assignment()
        with tempfile.TemporaryDirectory() as temporary:
            files = _evaluation_files(Path(temporary))
            files.work_dir.mkdir()
            result = EvaluationResult(
                metrics={"accuracy": 0.9},
                artifacts={"report": files.work_dir / "report.json"},
                unavailable_outputs=(),
            )
            evaluate = Mock(return_value=result)
            with patch.object(
                execution,
                "load_evaluation_entrypoint",
                return_value=evaluate,
            ) as load_evaluation:
                returned = execute_evaluation_attempt(assignment, files)

            self.assertIs(result, returned)
            load_evaluation.assert_called_once()
            evaluate.assert_called_once()

            protocol_handle = load_evaluation.call_args.args[0]
            context = evaluate.call_args.args[0]
            self.assertIsInstance(protocol_handle, EvaluationProtocolHandle)
            self.assertIs(protocol_handle.metadata, assignment.protocol)
            self.assertIsNone(protocol_handle.metadata_path)
            self.assertEqual(
                files.evaluation_protocol_implementation,
                protocol_handle.implementation_path,
            )
            self.assertIsInstance(context, EvaluationContext)
            self.assertIsInstance(context.task, TaskHandle)
            self.assertIs(context.task.metadata, assignment.task)
            self.assertIsNone(context.task.metadata_path)
            self.assertIsInstance(context.corpus, CorpusHandle)
            self.assertIs(context.corpus.metadata, assignment.corpus)
            self.assertIsNone(context.corpus.metadata_path)
            self.assertEqual(files.corpus_artifact, context.corpus.artifact_path)
            self.assertIsNone(context.corpus.builder_path)
            self.assertIsInstance(context.model, ModelHandle)
            self.assertIs(context.model.metadata, assignment.model)
            self.assertIsNone(context.model.metadata_path)
            self.assertIs(context.model.training_run, assignment.training_run)
            self.assertIsNone(context.model.training_run_metadata_path)
            self.assertEqual(files.model_weights, context.model.weights_path)
            self.assertIsInstance(context.model.architecture, ArchitectureHandle)
            self.assertIs(context.model.architecture.metadata, assignment.model_architecture)
            self.assertIsNone(context.model.architecture.metadata_path)
            self.assertEqual(
                files.model_architecture_implementation,
                context.model.architecture.implementation_path,
            )
            self.assertIs(context.parameters, assignment.parameters)
            self.assertEqual(files.work_dir, context.work_dir)

    def test_evaluation_loader_failure_does_not_run_domain(self) -> None:
        assignment = _evaluation_assignment()
        files = _evaluation_files(Path("worker"))
        evaluate = Mock()
        with (
            patch.object(
                execution,
                "load_evaluation_entrypoint",
                side_effect=RuntimeError("evaluation loader failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "evaluation loader failed"),
        ):
            execute_evaluation_attempt(assignment, files)
        evaluate.assert_not_called()
    def test_domain_failure_escapes_after_exactly_one_evaluate_call(self) -> None:
        assignment = _evaluation_assignment()
        files = _evaluation_files(Path("worker"))
        evaluate = Mock(side_effect=RuntimeError("evaluate failed"))
        with (
            patch.object(execution, "load_evaluation_entrypoint", return_value=evaluate),
            self.assertRaisesRegex(RuntimeError, "evaluate failed"),
        ):
            execute_evaluation_attempt(assignment, files)
        evaluate.assert_called_once()

    def test_protocol_can_use_existing_model_loader_boundary(self) -> None:
        assignment = _evaluation_assignment()
        with tempfile.TemporaryDirectory() as temporary:
            files = _evaluation_files(Path(temporary))
            files.work_dir.mkdir()
            result = EvaluationResult(metrics={}, artifacts={}, unavailable_outputs=())

            def evaluate(context: EvaluationContext) -> EvaluationResult:
                from mldb.src.model.loading import load_model

                load_model(context.model)
                return result

            with (
                patch.object(execution, "load_evaluation_entrypoint", return_value=evaluate),
                patch("mldb.src.model.loading.load_model", return_value=object()) as load_model,
            ):
                returned = execute_evaluation_attempt(assignment, files)

            self.assertIs(result, returned)
            load_model.assert_called_once()
            model_handle = load_model.call_args.args[0]
            self.assertIsInstance(model_handle, ModelHandle)
            self.assertIs(model_handle.training_run, assignment.training_run)
            self.assertEqual(files.model_weights, model_handle.weights_path)
    def test_invalid_result_type_fails_after_one_domain_call(self) -> None:
        assignment = _evaluation_assignment()
        files = _evaluation_files(Path("worker"))
        evaluate = Mock(return_value=object())
        with (
            patch.object(execution, "load_evaluation_entrypoint", return_value=evaluate),
            self.assertRaisesRegex(TypeError, "must return EvaluationResult"),
        ):
            execute_evaluation_attempt(assignment, files)
        evaluate.assert_called_once()

    def test_artifact_outside_work_dir_is_rejected_without_domain_retry(self) -> None:
        assignment = _evaluation_assignment()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = _evaluation_files(root)
            files.work_dir.mkdir()
            result = EvaluationResult(
                metrics={},
                artifacts={"report": root / "outside.json"},
                unavailable_outputs=(),
            )
            evaluate = Mock(return_value=result)
            with (
                patch.object(execution, "load_evaluation_entrypoint", return_value=evaluate),
                self.assertRaisesRegex(ValueError, "beneath work_dir"),
            ):
                execute_evaluation_attempt(assignment, files)
            evaluate.assert_called_once()

    def test_non_path_artifact_is_execution_contract_failure(self) -> None:
        assignment = _evaluation_assignment()
        files = _evaluation_files(Path("worker"))
        result = EvaluationResult(metrics={}, artifacts={"report": "report.json"}, unavailable_outputs=())
        evaluate = Mock(return_value=result)
        with (
            patch.object(execution, "load_evaluation_entrypoint", return_value=evaluate),
            self.assertRaisesRegex(TypeError, "pathlib.Path"),
        ):
            execute_evaluation_attempt(assignment, files)
        evaluate.assert_called_once()

    def test_missing_file_beneath_work_dir_is_not_revalidated_here(self) -> None:
        assignment = _evaluation_assignment()
        with tempfile.TemporaryDirectory() as temporary:
            files = _evaluation_files(Path(temporary))
            files.work_dir.mkdir()
            candidate = files.work_dir / "not-yet-uploaded.json"
            self.assertFalse(candidate.exists())
            result = EvaluationResult(
                metrics={},
                artifacts={"report": candidate},
                unavailable_outputs=(),
            )
            evaluate = Mock(return_value=result)
            with patch.object(
                execution,
                "load_evaluation_entrypoint",
                return_value=evaluate,
            ):
                returned = execute_evaluation_attempt(assignment, files)
            self.assertIs(result, returned)
            evaluate.assert_called_once()


class BoundaryAndSignatureTests(unittest.TestCase):
    def test_worker_execution_has_no_repository_queue_or_worker_api_calls(self) -> None:
        source_path = Path(execution.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules: list[str] = []
        worker_api_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_modules.append(module)
                if module == "worker_api" and node.level == 1:
                    worker_api_names.update(alias.name for alias in node.names)

        self.assertFalse(
            any(module.startswith("repository") or "queue" in module for module in imported_modules)
        )
        self.assertEqual(
            {"EvaluationAssignment", "TrainingAssignment"},
            worker_api_names,
        )
        referenced_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        self.assertTrue(
            {
                "CandidateUploadRequest",
                "TrainingSuccessCandidate",
                "EvaluationSuccessCandidate",
                "AttemptFailed",
                "WorkerApi",
            }.isdisjoint(referenced_names)
        )

    def test_public_signature_matches_frozen_worker_execution_skeleton(self) -> None:
        implementation = Path(execution.__file__)
        skeleton = implementation.parents[2] / "skeleton" / "orchestration" / "worker_execution.py"
        self.assertEqual((), compare_module_signatures(skeleton, implementation))


if __name__ == "__main__":
    unittest.main()
