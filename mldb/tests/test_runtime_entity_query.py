from __future__ import annotations

from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import mldb.src.runtime.entity_query as query
from mldb.src.common.errors import NotFoundError, ValidationFailedError
from mldb.src.common.ids import EntityKind
from mldb.src.repository.layout import RepositoryLayout


class _Status(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"


class _StrictFilesystem:
    def __init__(self) -> None:
        self.entries: dict[Path, tuple[Path, ...]] = {}
        self.texts: dict[Path, str] = {}
        self.bytes: dict[Path, bytes] = {}
        self.list_calls: list[Path] = []
        self.read_text_calls: list[Path] = []
        self.read_bytes_calls: list[Path] = []
        self.mutations: list[tuple[str, Path]] = []
        self.list_error: BaseException | None = None

    def set_entries(self, directory: Path, names: tuple[str, ...]) -> None:
        self.entries[directory] = tuple(directory / name for name in names)

    def file_exists(self, path: Path) -> bool:
        return path in self.texts or path in self.bytes

    def directory_exists(self, path: Path) -> bool:
        return path in self.entries

    def read_text(self, path: Path, *, encoding: str) -> str:
        self.read_text_calls.append(path)
        return self.texts[path]

    def read_bytes(self, path: Path) -> bytes:
        self.read_bytes_calls.append(path)
        return self.bytes[path]

    def list_directory(self, path: Path) -> tuple[Path, ...]:
        self.list_calls.append(path)
        if self.list_error is not None:
            raise self.list_error
        if path not in self.entries:
            raise FileNotFoundError(path)
        return self.entries[path]

    def ensure_directory(self, path: Path) -> None:
        self.mutations.append(("ensure_directory", path))
        raise AssertionError("entity query must be read-only")

    def replace_text(self, path: Path, text: str, *, encoding: str) -> None:
        self.mutations.append(("replace_text", path))
        raise AssertionError("entity query must be read-only")

    def replace_bytes(self, path: Path, data: bytes) -> None:
        self.mutations.append(("replace_bytes", path))
        raise AssertionError("entity query must be read-only")


def _task_yaml(entity_id: str) -> str:
    return f"""schema: mjtensu.mldb/task/v1
id: {entity_id}
name: tiles
problem_type: classification
description: tile classification
input:
  semantic_unit: single-tile-image
target:
  type: categorical
  labels: [a, b]
semantics: {{}}
scope:
  includes: [tiles]
  excludes: []
"""


class EntityQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.layout = RepositoryLayout(self.root)
        self.fs = _StrictFilesystem()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_get_entity_exact_dispatch_all_ten_kinds(self) -> None:
        cases = (
            (EntityKind.TASK, "_TaskId", "_resolve_task", True),
            (EntityKind.CORPUS, "_CorpusId", "_resolve_corpus", True),
            (EntityKind.ARCHITECTURE, "_ArchitectureId", "_resolve_architecture", True),
            (EntityKind.TRAIN_PROTOCOL, "_TrainProtocolId", "_resolve_train_protocol", True),
            (EntityKind.TRAINING_RUN, "_TrainingRunId", "_read_training_run", False),
            (EntityKind.MODEL, "_ModelId", "_resolve_model", True),
            (EntityKind.EVALUATION_PROTOCOL, "_EvaluationProtocolId", "_resolve_evaluation_protocol", True),
            (EntityKind.EVALUATION_RUN, "_EvaluationRunId", "_read_evaluation_run", False),
            (EntityKind.STUDY, "_StudyId", "_resolve_study", True),
            (EntityKind.STUDY_RUN, "_StudyRunId", "_read_study_run", False),
        )
        for kind, constructor_name, target_name, returns_handle in cases:
            with self.subTest(kind=kind):
                typed_id = object()
                metadata = object()
                target_result = SimpleNamespace(metadata=metadata) if returns_handle else metadata
                with (
                    patch.object(query, constructor_name, return_value=typed_id) as constructor,
                    patch.object(query, target_name, return_value=target_result) as target,
                ):
                    result = query.get_entity(kind, "requested-id", self.layout, self.fs)

                self.assertIs(metadata, result)
                constructor.assert_called_once_with("requested-id")
                target.assert_called_once_with(typed_id, self.layout, self.fs)

    def test_definition_handle_projects_metadata_only(self) -> None:
        metadata = object()
        handle = SimpleNamespace(metadata=metadata, implementation_path=object())
        with patch.object(query, "_resolve_architecture", return_value=handle):
            result = query.get_entity(
                EntityKind.ARCHITECTURE,
                "plain-v1",
                self.layout,
                self.fs,
            )
        self.assertIs(metadata, result)
        self.assertIsNot(handle, result)

    def test_study_run_read_does_not_read_plan_jsonl(self) -> None:
        run_id = "sr-20260908-001"
        paths = self.layout.study_run_paths(run_id)
        self.fs.texts[paths.metadata_path] = f"""schema: mjtensu.mldb/study-run/v1
id: {run_id}
status: running
study: study-v1
execution:
  started_at: 2026-09-08T00:00:00+09:00
plan:
  path: plan.jsonl
  sha256: {'a' * 64}
  bytes: 7
  trials: 1
  evaluation_jobs: 1
"""
        self.fs.bytes[paths.plan_path] = b"ignored\n"

        run = query.get_entity(EntityKind.STUDY_RUN, run_id, self.layout, self.fs)

        self.assertEqual(run_id, run.id)
        self.assertEqual([paths.metadata_path], self.fs.read_text_calls)
        self.assertEqual([], self.fs.read_bytes_calls)
        self.assertEqual([], self.fs.mutations)

    def test_canonical_suffix_rules_and_sibling_deduplication(self) -> None:
        cases = {
            EntityKind.TASK: (("alpha-v1.yaml", "ignored.py", "ignored.yaml.bak"), ("alpha-v1",)),
            EntityKind.CORPUS: (("alpha-v1.yaml", "alpha-v1.sqlite", "alpha-v1.py", "beta-v1.sqlite", "gamma-v1.py", "ignored.json"), ("alpha-v1", "beta-v1", "gamma-v1")),
            EntityKind.ARCHITECTURE: (("alpha-v1.yaml", "alpha-v1.py", "beta-v1.py", "ignored.sqlite"), ("alpha-v1", "beta-v1")),
            EntityKind.TRAIN_PROTOCOL: (("alpha-v1.yaml", "alpha-v1.py", "beta-v1.py", "ignored.txt"), ("alpha-v1", "beta-v1")),
            EntityKind.MODEL: (("alpha-v1.yaml", "ignored.py", "ignored.YAML"), ("alpha-v1",)),
            EntityKind.EVALUATION_PROTOCOL: (("alpha-v1.yaml", "alpha-v1.py", "beta-v1.py", "ignored.sqlite"), ("alpha-v1", "beta-v1")),
            EntityKind.STUDY: (("alpha-v1.yaml", "ignored.py", "ignored.json"), ("alpha-v1",)),
        }
        for kind, (names, expected_ids) in cases.items():
            with self.subTest(kind=kind):
                filesystem = _StrictFilesystem()
                directory = self.layout.entity_directory(kind)
                filesystem.set_entries(directory, names)

                def resolved(_kind: EntityKind, entity_id: str, _layout: RepositoryLayout, _filesystem: _StrictFilesystem):
                    if kind in {EntityKind.TASK, EntityKind.CORPUS, EntityKind.MODEL}:
                        return SimpleNamespace(id=entity_id)
                    return SimpleNamespace(id=entity_id, status=_Status.DRAFT)

                with patch.object(query, "get_entity", side_effect=resolved) as get_entity:
                    summaries = query.list_entities(kind, self.layout, filesystem)

                self.assertEqual(expected_ids, tuple(summary.id for summary in summaries))
                self.assertEqual([directory], filesystem.list_calls)
                self.assertEqual(expected_ids, tuple(call.args[1] for call in get_entity.call_args_list))

    def test_orphan_canonical_sibling_is_not_skipped(self) -> None:
        cases = (
            (EntityKind.ARCHITECTURE, "orphan-v1.py"),
            (EntityKind.CORPUS, "orphan-v1.sqlite"),
        )
        for kind, name in cases:
            with self.subTest(kind=kind):
                filesystem = _StrictFilesystem()
                directory = self.layout.entity_directory(kind)
                filesystem.set_entries(directory, (name,))
                with self.assertRaises(NotFoundError):
                    query.list_entities(kind, self.layout, filesystem)
                self.assertEqual([directory], filesystem.list_calls)

    def test_malformed_candidate_aborts_complete_listing(self) -> None:
        directory = self.layout.entity_directory(EntityKind.TASK)
        bad_path = directory / "bad-v1.yaml"
        self.fs.set_entries(directory, ("bad-v1.yaml", "good-v1.yaml"))
        self.fs.texts[bad_path] = "schema: [unterminated\n"
        self.fs.texts[directory / "good-v1.yaml"] = _task_yaml("good-v1")

        with self.assertRaises(ValidationFailedError):
            query.list_entities(EntityKind.TASK, self.layout, self.fs)

        self.assertEqual([bad_path], self.fs.read_text_calls)

    def test_run_listing_uses_typed_list_operations_and_sorts_resolved_ids(self) -> None:
        cases = (
            (EntityKind.TRAINING_RUN, "_list_training_runs", "tr-20260908"),
            (EntityKind.EVALUATION_RUN, "_list_evaluation_runs", "ev-20260908"),
            (EntityKind.STUDY_RUN, "_list_study_runs", "sr-20260908"),
        )
        for kind, target_name, prefix in cases:
            with self.subTest(kind=kind):
                filesystem = _StrictFilesystem()
                runs = (
                    SimpleNamespace(id=f"{prefix}-002", status=_Status.RUNNING),
                    SimpleNamespace(id=f"{prefix}-001", status=_Status.RUNNING),
                )
                with patch.object(query, target_name, return_value=runs) as target:
                    summaries = query.list_entities(kind, self.layout, filesystem)

                self.assertEqual(
                    (f"{prefix}-001", f"{prefix}-002"),
                    tuple(summary.id for summary in summaries),
                )
                self.assertEqual(("running", "running"), tuple(summary.status for summary in summaries))
                target.assert_called_once_with(self.layout, filesystem)
                self.assertEqual([], filesystem.list_calls)

    def test_status_projection_matrix(self) -> None:
        no_status = {EntityKind.TASK, EntityKind.CORPUS, EntityKind.MODEL}
        for kind in EntityKind:
            with self.subTest(kind=kind):
                filesystem = _StrictFilesystem()
                entity = SimpleNamespace(id=f"{kind.name.lower()}-id")
                if kind not in no_status:
                    entity.status = _Status.DRAFT

                if kind in {EntityKind.TRAINING_RUN, EntityKind.EVALUATION_RUN, EntityKind.STUDY_RUN}:
                    target_name = {
                        EntityKind.TRAINING_RUN: "_list_training_runs",
                        EntityKind.EVALUATION_RUN: "_list_evaluation_runs",
                        EntityKind.STUDY_RUN: "_list_study_runs",
                    }[kind]
                    with patch.object(query, target_name, return_value=(entity,)):
                        summaries = query.list_entities(kind, self.layout, filesystem)
                else:
                    directory = self.layout.entity_directory(kind)
                    filesystem.set_entries(directory, ("candidate.yaml",))
                    with patch.object(query, "get_entity", return_value=entity):
                        summaries = query.list_entities(kind, self.layout, filesystem)

                expected = None if kind in no_status else "draft"
                self.assertEqual(expected, summaries[0].status)

    def test_filesystem_failure_propagates(self) -> None:
        error = OSError("listing failed")
        self.fs.list_error = error
        with self.assertRaises(OSError) as caught:
            query.list_entities(EntityKind.TASK, self.layout, self.fs)
        self.assertIs(error, caught.exception)

    def test_resolver_failure_propagates_without_translation(self) -> None:
        error = RuntimeError("resolver failed")
        with patch.object(query, "_resolve_task", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                query.get_entity(EntityKind.TASK, "task-v1", self.layout, self.fs)
        self.assertIs(error, caught.exception)

    def test_exact_read_is_read_only_and_does_not_scan(self) -> None:
        task_id = "tile-task-v1"
        path = self.layout.task_metadata_path(task_id)
        self.fs.texts[path] = _task_yaml(task_id)

        task = query.get_entity(EntityKind.TASK, task_id, self.layout, self.fs)

        self.assertEqual(task_id, task.id)
        self.assertEqual([], self.fs.list_calls)
        self.assertEqual([], self.fs.read_bytes_calls)
        self.assertEqual([], self.fs.mutations)


if __name__ == "__main__":
    unittest.main()
