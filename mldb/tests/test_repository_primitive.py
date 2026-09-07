from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EntityKind,
    EvaluationProtocolId,
    EvaluationRunId,
    ModelId,
    StudyId,
    StudyRunId,
    TaskId,
    TrainingRunId,
    TrainProtocolId,
)
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.tests.signature_guard import compare_module_signatures


MLDB_ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.layout = RepositoryLayout(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_canonical_roots(self) -> None:
        self.assertEqual(self.root / "mldb_data", self.layout.data_root)
        self.assertEqual(self.root / "mldb_tests", self.layout.tests_root)

    def test_all_definition_and_asset_paths(self) -> None:
        self.assertEqual(
            self.root / "mldb_data" / "tasks" / "task-v1.yaml",
            self.layout.task_metadata_path(TaskId("task-v1")),
        )
        self.assertEqual(
            self.root / "mldb_data" / "corpora" / "corpus-v1.yaml",
            self.layout.corpus_metadata_path(CorpusId("corpus-v1")),
        )
        self.assertEqual(
            self.root / "mldb_data" / "corpora" / "corpus-v1.sqlite",
            self.layout.corpus_artifact_path(CorpusId("corpus-v1")),
        )
        self.assertEqual(
            self.root / "mldb_data" / "corpora" / "corpus-v1.py",
            self.layout.corpus_builder_path(CorpusId("corpus-v1")),
        )
        self.assertEqual(
            self.root / "mldb_data" / "architectures" / "architecture-v1.yaml",
            self.layout.architecture_metadata_path(ArchitectureId("architecture-v1")),
        )
        self.assertEqual(
            self.root / "mldb_data" / "architectures" / "architecture-v1.py",
            self.layout.architecture_implementation_path(ArchitectureId("architecture-v1")),
        )
        self.assertEqual(
            self.root / "mldb_data" / "train_protocols" / "train-v1.yaml",
            self.layout.train_protocol_metadata_path(TrainProtocolId("train-v1")),
        )
        self.assertEqual(
            self.root / "mldb_data" / "train_protocols" / "train-v1.py",
            self.layout.train_protocol_implementation_path(TrainProtocolId("train-v1")),
        )
        self.assertEqual(
            self.root / "mldb_data" / "models" / "model-v1.yaml",
            self.layout.model_metadata_path(ModelId("model-v1")),
        )
        self.assertEqual(
            self.root / "mldb_data" / "evaluation_protocols" / "eval-v1.yaml",
            self.layout.evaluation_protocol_metadata_path(EvaluationProtocolId("eval-v1")),
        )
        self.assertEqual(
            self.root / "mldb_data" / "evaluation_protocols" / "eval-v1.py",
            self.layout.evaluation_protocol_implementation_path(EvaluationProtocolId("eval-v1")),
        )
        self.assertEqual(
            self.root / "mldb_data" / "studies" / "study-v1.yaml",
            self.layout.study_metadata_path(StudyId("study-v1")),
        )

    def test_all_run_paths(self) -> None:
        training_directory = self.root / "mldb_data" / "training_runs" / "training-run-v1"
        training = self.layout.training_run_paths(TrainingRunId("training-run-v1"))
        self.assertEqual(training_directory, training.directory)
        self.assertEqual(training_directory / "run.yaml", training.metadata_path)
        self.assertEqual(training_directory / "work", training.work_dir)
        self.assertEqual(training_directory / "artifacts", training.artifacts_dir)
        self.assertEqual(training_directory / "artifacts" / "weights.pt", training.weights_path)

        evaluation_directory = self.root / "mldb_data" / "evaluation_runs" / "evaluation-run-v1"
        evaluation = self.layout.evaluation_run_paths(EvaluationRunId("evaluation-run-v1"))
        self.assertEqual(evaluation_directory, evaluation.directory)
        self.assertEqual(evaluation_directory / "run.yaml", evaluation.metadata_path)
        self.assertEqual(evaluation_directory / "work", evaluation.work_dir)
        self.assertEqual(evaluation_directory / "artifacts", evaluation.artifacts_dir)

        study_directory = self.root / "mldb_data" / "study_runs" / "study-run-v1"
        study = self.layout.study_run_paths(StudyRunId("study-run-v1"))
        self.assertEqual(study_directory, study.directory)
        self.assertEqual(study_directory / "run.yaml", study.metadata_path)
        self.assertEqual(study_directory / "plan.jsonl", study.plan_path)

    def test_all_executable_asset_test_directories(self) -> None:
        self.assertEqual(
            self.root / "mldb_tests" / "architectures" / "architecture-v1",
            self.layout.architecture_test_dir(ArchitectureId("architecture-v1")),
        )
        self.assertEqual(
            self.root / "mldb_tests" / "train_protocols" / "train-v1",
            self.layout.train_protocol_test_dir(TrainProtocolId("train-v1")),
        )
        self.assertEqual(
            self.root / "mldb_tests" / "evaluation_protocols" / "eval-v1",
            self.layout.evaluation_protocol_test_dir(EvaluationProtocolId("eval-v1")),
        )

    def test_entity_directory_mapping(self) -> None:
        expected = {
            EntityKind.TASK: "tasks",
            EntityKind.CORPUS: "corpora",
            EntityKind.ARCHITECTURE: "architectures",
            EntityKind.TRAIN_PROTOCOL: "train_protocols",
            EntityKind.TRAINING_RUN: "training_runs",
            EntityKind.MODEL: "models",
            EntityKind.EVALUATION_PROTOCOL: "evaluation_protocols",
            EntityKind.EVALUATION_RUN: "evaluation_runs",
            EntityKind.STUDY: "studies",
            EntityKind.STUDY_RUN: "study_runs",
        }
        for kind, directory_name in expected.items():
            with self.subTest(kind=kind):
                self.assertEqual(
                    self.root / "mldb_data" / directory_name,
                    self.layout.entity_directory(kind),
                )


class RepositorySignatureTests(unittest.TestCase):
    def test_frozen_repository_signatures_match(self) -> None:
        for filename in ("layout.py", "ports.py"):
            with self.subTest(filename=filename):
                mismatches = compare_module_signatures(
                    MLDB_ROOT / "skeleton" / "repository" / filename,
                    MLDB_ROOT / "src" / "repository" / filename,
                )
                self.assertEqual((), mismatches, "\n".join(mismatches))


class LocalFilesystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.filesystem = LocalFilesystem()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_file_and_directory_existence(self) -> None:
        directory = self.root / "directory"
        file_path = self.root / "file.bin"
        directory.mkdir()
        file_path.write_bytes(b"data")

        self.assertTrue(self.filesystem.directory_exists(directory))
        self.assertFalse(self.filesystem.file_exists(directory))
        self.assertTrue(self.filesystem.file_exists(file_path))
        self.assertFalse(self.filesystem.directory_exists(file_path))
        self.assertFalse(self.filesystem.file_exists(self.root / "missing"))
        self.assertFalse(self.filesystem.directory_exists(self.root / "missing"))

    def test_list_directory_uses_python_path_name_order(self) -> None:
        for name in ("zeta", "Alpha", "beta", "aardvark"):
            (self.root / name).write_text(name, encoding="utf-8")

        entries = self.filesystem.list_directory(self.root)

        self.assertEqual(
            tuple(sorted(entries, key=lambda child: child.name)),
            entries,
        )
        self.assertEqual(
            ("Alpha", "aardvark", "beta", "zeta"),
            tuple(entry.name for entry in entries),
        )

    def test_ensure_directory_creates_required_parents(self) -> None:
        directory = self.root / "one" / "two" / "three"

        self.filesystem.ensure_directory(directory)
        self.filesystem.ensure_directory(directory)

        self.assertTrue(directory.is_dir())

    def test_replace_and_read_text(self) -> None:
        path = self.root / "value.txt"

        self.filesystem.replace_text(path, "first\n日本語", encoding="utf-8")
        self.assertEqual("first\n日本語", self.filesystem.read_text(path, encoding="utf-8"))

        self.filesystem.replace_text(path, "replacement", encoding="utf-8")
        self.assertEqual("replacement", self.filesystem.read_text(path, encoding="utf-8"))
        self.assertEqual(b"replacement", path.read_bytes())

    def test_replace_and_read_bytes(self) -> None:
        path = self.root / "value.bin"

        self.filesystem.replace_bytes(path, b"\x00\x01initial")
        self.assertEqual(b"\x00\x01initial", self.filesystem.read_bytes(path))

        self.filesystem.replace_bytes(path, b"\xffreplacement\x00")
        self.assertEqual(b"\xffreplacement\x00", self.filesystem.read_bytes(path))

    def test_successful_replace_leaves_only_complete_destination(self) -> None:
        path = self.root / "value.bin"
        path.write_bytes(b"old")
        replacement = b"new-content" * 1024

        self.filesystem.replace_bytes(path, replacement)

        self.assertEqual(replacement, path.read_bytes())
        self.assertEqual((path,), self.filesystem.list_directory(self.root))

    def test_replace_does_not_create_parent_directory(self) -> None:
        missing_parent = self.root / "missing"
        path = missing_parent / "value.bin"

        with self.assertRaises(FileNotFoundError):
            self.filesystem.replace_bytes(path, b"value")

        self.assertFalse(missing_parent.exists())


if __name__ == "__main__":
    unittest.main()
