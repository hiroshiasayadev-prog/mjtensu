from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from mldb.src.common.errors import LifecycleConflictError, ValidationFailedError
from mldb.src.common.ids import ArchitectureId, CorpusId, EntityKind, EvaluationProtocolId, ModelId, StudyId, TrainingRunId, TrainProtocolId
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.run_finalization import (
    cancel_evaluation_run,
    cancel_training_run,
    fail_evaluation_run,
    fail_training_run,
)
from mldb.src.runtime._run_serialization import dump_yaml_document, load_yaml_document
from mldb.src.runtime.run_persistence import (
    allocate_evaluation_run,
    allocate_study_run,
    allocate_training_run,
    finalize_study_plan,
    list_evaluation_runs,
    list_study_runs,
    list_training_runs,
    read_evaluation_run,
    read_study_plan,
    read_study_run,
    read_training_run,
    persist_study_run_transition,
)
from mldb.src.study.plan import StudyPlanEvaluation, StudyPlanRow, StudyPlanTraining
from mldb.src.study.run import StudyRun, StudyRunExecution, StudyRunStatus
from mldb.src.training.run import TrainingRunFailure, TrainingRunStatus
from mldb.src.evaluation.run import EvaluationRunFailure, EvaluationRunStatus


class CanonicalRunPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.layout = RepositoryLayout(Path(self.temp.name))
        self.fs = LocalFilesystem()
        self.day = date(2026, 9, 8)
        self.started = datetime(2026, 9, 8, 1, 30, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def training_preflight(self):
        return SimpleNamespace(
            corpus=SimpleNamespace(metadata=SimpleNamespace(id=CorpusId("corpus-v1"))),
            architecture=SimpleNamespace(metadata=SimpleNamespace(id=ArchitectureId("plain-v1"))),
            protocol=SimpleNamespace(metadata=SimpleNamespace(id=TrainProtocolId("train-v1"))),
            parameters={"epochs": 10, "lr": 0.001},
            seed=42,
        )

    def evaluation_preflight(self):
        return SimpleNamespace(
            model=SimpleNamespace(metadata=SimpleNamespace(id=ModelId("mdl-20260908-001"))),
            corpus=SimpleNamespace(metadata=SimpleNamespace(id=CorpusId("holdout-v1"))),
            protocol=SimpleNamespace(metadata=SimpleNamespace(id=EvaluationProtocolId("eval-v1"))),
            parameters={"batch_size": 128},
        )

    def rich_parameters(self):
        return {
            "nested": [{"a": {"b": 1}, "c": [2, 3]}],
            "mapping": {"items": [None, True, False, -7, 2.5]},
            "unicode": "麻雀牌: 東,南 [test] #1 / ✓",
            "punct:key": {"quoted value": "a:b,c{d}[e]#f"},
        }

    def study_prepared(self):
        return SimpleNamespace(study=SimpleNamespace(metadata=SimpleNamespace(id=StudyId("study-v1"))))

    def plan(self):
        return (
            StudyPlanRow(
                trial="trial-0001",
                training=StudyPlanTraining(
                    architecture=ArchitectureId("plain-v1"),
                    corpus=CorpusId("corpus-v1"),
                    protocol=TrainProtocolId("train-v1"),
                    seed=42,
                    parameters={"epochs": 10, "lr": 0.001},
                ),
                evaluations=(
                    StudyPlanEvaluation(
                        stage="holdout",
                        corpus=CorpusId("holdout-v1"),
                        protocol=EvaluationProtocolId("eval-v1"),
                        parameters={"batch_size": 128},
                    ),
                ),
            ),
            StudyPlanRow(
                trial="trial-0002",
                model=ModelId("mdl-20260908-001"),
                evaluations=(
                    StudyPlanEvaluation(
                        stage="holdout",
                        corpus=CorpusId("holdout-v1"),
                        protocol=EvaluationProtocolId("eval-v1"),
                        parameters={"batch_size": 128},
                    ),
                    StudyPlanEvaluation(
                        stage="robustness",
                        corpus=CorpusId("holdout-v1"),
                        protocol=EvaluationProtocolId("eval-v1"),
                        parameters={"batch_size": 64},
                    ),
                ),
            ),
        )

    def test_sequential_typed_id_allocation_and_unrelated_kind_ignored(self) -> None:
        unrelated = self.layout.entity_directory(EntityKind.EVALUATION_RUN)
        unrelated.mkdir(parents=True)
        (unrelated / "ev-20260908-099").mkdir()
        training_dir = self.layout.entity_directory(EntityKind.TRAINING_RUN)
        training_dir.mkdir(parents=True)
        (training_dir / "README.txt").write_text("not a run", encoding="utf-8")

        first = allocate_training_run(self.training_preflight(), self.day, self.started, self.layout, self.fs)
        second = allocate_training_run(self.training_preflight(), self.day, self.started, self.layout, self.fs)
        evaluation = allocate_evaluation_run(self.evaluation_preflight(), self.day, self.started, self.layout, self.fs)
        study = allocate_study_run(self.study_prepared(), self.day, self.started, self.layout, self.fs)

        self.assertEqual("tr-20260908-001", first.id)
        self.assertEqual("tr-20260908-002", second.id)
        self.assertEqual("ev-20260908-100", evaluation.id)
        self.assertEqual("sr-20260908-001", study.id)

    def test_partial_existing_training_identity_is_not_reused(self) -> None:
        path = self.layout.training_run_paths(TrainingRunId("tr-20260908-001"))
        path.directory.mkdir(parents=True)
        run = allocate_training_run(self.training_preflight(), self.day, self.started, self.layout, self.fs)
        self.assertEqual("tr-20260908-002", run.id)

    def test_initial_running_records_and_required_paths(self) -> None:
        training = allocate_training_run(self.training_preflight(), self.day, self.started, self.layout, self.fs)
        evaluation = allocate_evaluation_run(self.evaluation_preflight(), self.day, self.started, self.layout, self.fs)
        study = allocate_study_run(self.study_prepared(), self.day, self.started, self.layout, self.fs)

        tp = self.layout.training_run_paths(training.id)
        ep = self.layout.evaluation_run_paths(evaluation.id)
        sp = self.layout.study_run_paths(study.id)
        self.assertTrue(tp.metadata_path.is_file())
        self.assertTrue(tp.work_dir.is_dir())
        self.assertTrue(tp.artifacts_dir.is_dir())
        self.assertTrue(ep.metadata_path.is_file())
        self.assertTrue(ep.work_dir.is_dir())
        self.assertTrue(ep.artifacts_dir.is_dir())
        self.assertTrue(sp.metadata_path.is_file())
        self.assertFalse(sp.plan_path.exists())
        self.assertEqual(TrainingRunStatus.RUNNING, list_training_runs(self.layout, self.fs)[0].status)
        self.assertEqual(EvaluationRunStatus.RUNNING, list_evaluation_runs(self.layout, self.fs)[0].status)
        self.assertIsNone(read_study_run(study.id, self.layout, self.fs).plan)
        yaml_text = tp.metadata_path.read_text(encoding="utf-8")
        self.assertIn("status: \"running\"", yaml_text)
        self.assertIn("seed: 42", yaml_text)
        self.assertNotIn("finished_at", yaml_text)

    def test_yaml_round_trip_preserves_complete_public_parameter_domain(self) -> None:
        document = {"parameters": self.rich_parameters()}
        dumped = dump_yaml_document(document)

        self.assertEqual(document, load_yaml_document(dumped))
        self.assertIn("麻雀牌", dumped)

    def test_nested_parameters_survive_allocation_and_terminal_persistence(self) -> None:
        training_preflight = self.training_preflight()
        training_preflight.parameters = self.rich_parameters()
        evaluation_preflight = self.evaluation_preflight()
        evaluation_preflight.parameters = self.rich_parameters()

        training = allocate_training_run(
            training_preflight, self.day, self.started, self.layout, self.fs
        )
        evaluation = allocate_evaluation_run(
            evaluation_preflight, self.day, self.started, self.layout, self.fs
        )
        self.assertEqual(training, read_training_run(training.id, self.layout, self.fs))
        self.assertEqual(evaluation, read_evaluation_run(evaluation.id, self.layout, self.fs))

        finished = datetime(2026, 9, 8, 2, 0, tzinfo=timezone.utc)
        failed_training = fail_training_run(training, finished, self.layout, self.fs)
        failed_evaluation = fail_evaluation_run(evaluation, finished, self.layout, self.fs)
        self.assertEqual(failed_training, read_training_run(training.id, self.layout, self.fs))
        self.assertEqual(failed_evaluation, read_evaluation_run(evaluation.id, self.layout, self.fs))
        self.assertEqual((failed_training,), list_training_runs(self.layout, self.fs))
        self.assertEqual((failed_evaluation,), list_evaluation_runs(self.layout, self.fs))

    def test_study_plan_finalize_hash_counts_and_read(self) -> None:
        study = allocate_study_run(self.study_prepared(), self.day, self.started, self.layout, self.fs)
        finalized = finalize_study_plan(study.id, self.plan(), self.layout, self.fs)
        data = self.layout.study_run_paths(study.id).plan_path.read_bytes()

        self.assertEqual(hashlib.sha256(data).hexdigest(), finalized.plan.sha256)
        self.assertEqual(len(data), finalized.plan.bytes)
        self.assertEqual(2, finalized.plan.trials)
        self.assertEqual(3, finalized.plan.evaluation_jobs)
        self.assertEqual(self.plan(), read_study_plan(study.id, self.layout, self.fs))

    def test_unfinalized_plan_bytes_are_replaceable(self) -> None:
        study = allocate_study_run(self.study_prepared(), self.day, self.started, self.layout, self.fs)
        path = self.layout.study_run_paths(study.id).plan_path
        path.write_bytes(b"stale-unfinalized-bytes\n")

        finalized = finalize_study_plan(study.id, self.plan(), self.layout, self.fs)

        self.assertNotEqual(b"stale-unfinalized-bytes\n", path.read_bytes())
        self.assertEqual(finalized.plan.sha256, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_finalized_plan_is_immutable(self) -> None:
        study = allocate_study_run(self.study_prepared(), self.day, self.started, self.layout, self.fs)
        finalize_study_plan(study.id, self.plan(), self.layout, self.fs)
        original = self.layout.study_run_paths(study.id).plan_path.read_bytes()

        with self.assertRaises(LifecycleConflictError):
            finalize_study_plan(study.id, self.plan(), self.layout, self.fs)

        self.assertEqual(original, self.layout.study_run_paths(study.id).plan_path.read_bytes())

    def test_failure_and_cancel_transitions_and_terminal_listing(self) -> None:
        training1 = allocate_training_run(self.training_preflight(), self.day, self.started, self.layout, self.fs)
        training2 = allocate_training_run(self.training_preflight(), self.day, self.started, self.layout, self.fs)
        evaluation1 = allocate_evaluation_run(self.evaluation_preflight(), self.day, self.started, self.layout, self.fs)
        evaluation2 = allocate_evaluation_run(self.evaluation_preflight(), self.day, self.started, self.layout, self.fs)
        finished = datetime(2026, 9, 8, 2, 0, tzinfo=timezone.utc)

        failed_t = fail_training_run(training1, finished, self.layout, self.fs, failure=TrainingRunFailure("worker-error", "boom"))
        cancelled_t = cancel_training_run(training2, finished, self.layout, self.fs)
        failed_e = fail_evaluation_run(evaluation1, finished, self.layout, self.fs, failure=EvaluationRunFailure("worker-error", "boom"))
        cancelled_e = cancel_evaluation_run(evaluation2, finished, self.layout, self.fs)

        self.assertEqual(TrainingRunStatus.FAILED, failed_t.status)
        self.assertEqual(TrainingRunStatus.CANCELLED, cancelled_t.status)
        self.assertEqual(EvaluationRunStatus.FAILED, failed_e.status)
        self.assertEqual(EvaluationRunStatus.CANCELLED, cancelled_e.status)
        self.assertEqual((failed_t, cancelled_t), list_training_runs(self.layout, self.fs))
        self.assertEqual((failed_e, cancelled_e), list_evaluation_runs(self.layout, self.fs))

    def test_failure_none_is_legal_and_terminal_rewrite_rejected(self) -> None:
        training = allocate_training_run(self.training_preflight(), self.day, self.started, self.layout, self.fs)
        finished = datetime(2026, 9, 8, 2, 0, tzinfo=timezone.utc)
        failed = fail_training_run(training, finished, self.layout, self.fs)
        self.assertIsNone(failed.failure)

        with self.assertRaises(LifecycleConflictError):
            cancel_training_run(training, finished, self.layout, self.fs)

    def test_study_terminal_read_list_and_rewrite_rejection(self) -> None:
        study = allocate_study_run(self.study_prepared(), self.day, self.started, self.layout, self.fs)
        finished = datetime(2026, 9, 8, 2, 0, tzinfo=timezone.utc)
        failed = StudyRun(
            schema=study.schema,
            id=study.id,
            status=StudyRunStatus.FAILED,
            study=study.study,
            execution=StudyRunExecution(started_at=study.execution.started_at, finished_at=finished),
            plan=None,
            summary=None,
        )
        persist_study_run_transition(failed, self.layout, self.fs)
        self.assertEqual((failed,), list_study_runs(self.layout, self.fs))

        cancelled = StudyRun(
            schema=failed.schema,
            id=failed.id,
            status=StudyRunStatus.CANCELLED,
            study=failed.study,
            execution=StudyRunExecution(started_at=failed.execution.started_at, finished_at=finished),
            plan=None,
            summary=None,
        )
        with self.assertRaises(ValidationFailedError):
            persist_study_run_transition(cancelled, self.layout, self.fs)

    def test_stale_running_equality_is_rejected(self) -> None:
        training = allocate_training_run(self.training_preflight(), self.day, self.started, self.layout, self.fs)
        finished = datetime(2026, 9, 8, 2, 0, tzinfo=timezone.utc)
        fail_training_run(training, finished, self.layout, self.fs)
        with self.assertRaises(LifecycleConflictError):
            fail_training_run(training, finished, self.layout, self.fs)


if __name__ == "__main__":
    unittest.main()
