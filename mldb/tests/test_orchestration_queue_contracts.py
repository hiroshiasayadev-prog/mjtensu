from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
import inspect
from pathlib import Path
from typing import get_type_hints
import unittest

from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    EvaluationRunId,
    ModelId,
    StudyRunId,
    TrainingRunId,
    TrainProtocolId,
)
from mldb.src.orchestration.jobs import (
    EvaluationJob,
    EvaluationJobCoordinate,
    TrainingJob,
    TrainingJobCoordinate,
    derive_study_jobs,
)
from mldb.src.orchestration.queue import (
    QueueAttempt,
    QueueJob,
    QueueJobStatus,
    queue_database_path,
)
from mldb.src.orchestration.queue_ports import QueuePort
from mldb.src.orchestration.retry_policy import RetryDecision
from mldb.src.study.plan import (
    StudyPlanEvaluation,
    StudyPlanRow,
    StudyPlanTraining,
)


class LogicalJobDerivationTests(unittest.TestCase):
    def test_training_source_plan_derives_complete_job_set(self) -> None:
        study_run = StudyRunId("sr-20260908-101")
        plan = [
            _training_row("trial-0001", ("quality", "latency")),
            _training_row("trial-0002", ("quality",)),
        ]

        jobs = derive_study_jobs(study_run, plan)

        training_1 = TrainingJobCoordinate(study_run, "trial-0001")
        training_2 = TrainingJobCoordinate(study_run, "trial-0002")
        self.assertEqual(
            frozenset(
                {
                    TrainingJob(training_1),
                    EvaluationJob(
                        EvaluationJobCoordinate(study_run, "trial-0001", "quality"),
                        training_1,
                    ),
                    EvaluationJob(
                        EvaluationJobCoordinate(study_run, "trial-0001", "latency"),
                        training_1,
                    ),
                    TrainingJob(training_2),
                    EvaluationJob(
                        EvaluationJobCoordinate(study_run, "trial-0002", "quality"),
                        training_2,
                    ),
                }
            ),
            jobs,
        )
        self.assertIsInstance(jobs, frozenset)

    def test_existing_model_plan_derives_only_dependency_free_evaluations(self) -> None:
        study_run = StudyRunId("sr-20260908-102")
        plan = [
            _existing_model_row("trial-0001", "mdl-20260908-011", ("a", "b")),
            _existing_model_row("trial-0002", "mdl-20260908-012", ("a",)),
        ]

        jobs = derive_study_jobs(study_run, plan)

        expected = frozenset(
            EvaluationJob(
                EvaluationJobCoordinate(study_run, trial, stage),
                None,
            )
            for trial, stages in (
                ("trial-0001", ("a", "b")),
                ("trial-0002", ("a",)),
            )
            for stage in stages
        )
        self.assertEqual(expected, jobs)
        self.assertFalse(any(isinstance(job, TrainingJob) for job in jobs))

    def test_multi_stage_dependencies_are_same_row_only(self) -> None:
        study_run = StudyRunId("sr-20260908-103")
        jobs = derive_study_jobs(
            study_run,
            [
                _training_row("trial-0001", ("first", "second", "third")),
                _training_row("trial-0002", ("first", "second")),
            ],
        )

        evaluations = [job for job in jobs if isinstance(job, EvaluationJob)]
        for job in evaluations:
            self.assertEqual(job.coordinate.study_run, job.training_dependency.study_run)
            self.assertEqual(job.coordinate.trial, job.training_dependency.trial)

    def test_frozenset_identity_collapses_duplicate_derivations(self) -> None:
        study_run = StudyRunId("sr-20260908-104")
        row = _training_row("trial-0001", ("eval",))

        jobs = derive_study_jobs(study_run, [row, row])

        self.assertEqual(2, len(jobs))
        self.assertEqual(jobs, derive_study_jobs(study_run, [row]))

    def test_plan_execution_payload_is_not_copied_into_jobs(self) -> None:
        study_run = StudyRunId("sr-20260908-105")
        row = _training_row("trial-0001", ("eval",))

        jobs = derive_study_jobs(study_run, [row])

        self.assertEqual(("study_run", "trial"), tuple(field.name for field in fields(TrainingJobCoordinate)))
        self.assertEqual(
            ("study_run", "trial", "stage"),
            tuple(field.name for field in fields(EvaluationJobCoordinate)),
        )
        self.assertEqual(("coordinate",), tuple(field.name for field in fields(TrainingJob)))
        self.assertEqual(
            ("coordinate", "training_dependency"),
            tuple(field.name for field in fields(EvaluationJob)),
        )
        rendered = repr(jobs)
        for payload in (
            "arch-private-v1",
            "train-corpus-private-v1",
            "train-protocol-private-v1",
            "eval-corpus-private-v1",
            "eval-protocol-private-v1",
            "secret-parameter-value",
        ):
            self.assertNotIn(payload, rendered)

    def test_derivation_is_pure_and_does_not_mutate_plan(self) -> None:
        study_run = StudyRunId("sr-20260908-106")
        plan = [
            _training_row("trial-0001", ("a", "b")),
            _existing_model_row("trial-0002", "mdl-20260908-099", ("a",)),
        ]
        before = deepcopy(plan)

        first = derive_study_jobs(study_run, plan)
        second = derive_study_jobs(study_run, plan)

        self.assertEqual(before, plan)
        self.assertEqual(first, second)


class QueueValueContractTests(unittest.TestCase):
    def test_queue_status_has_exact_frozen_seven_values(self) -> None:
        self.assertEqual(
            [
                "blocked",
                "ready",
                "active",
                "retry_wait",
                "satisfied",
                "failed",
                "cancelled",
            ],
            [status.value for status in QueueJobStatus],
        )
        self.assertIs(QueueJobStatus("ready"), QueueJobStatus.READY)

    def test_queue_job_and_attempt_values_construct_without_payload_reinterpretation(self) -> None:
        logical = TrainingJob(
            TrainingJobCoordinate(StudyRunId("sr-20260908-107"), "trial-0001")
        )
        queued = QueueJob(
            job_id=7,
            logical=logical,
            status=QueueJobStatus.ACTIVE,
            dependency_job_id=None,
            retry_not_before=None,
            created_at="2026-09-08T01:02:03.000004Z",
            updated_at="2026-09-08T01:02:04.000005Z",
        )
        training_run = TrainingRunId("tr-20260908-107")
        attempt = _attempt(training_run)

        self.assertIs(queued.logical, logical)
        self.assertEqual(training_run, attempt.run_id)

    def test_queue_attempt_run_id_annotation_preserves_typed_union(self) -> None:
        hints = get_type_hints(QueueAttempt)
        self.assertEqual(TrainingRunId | EvaluationRunId, hints["run_id"])

        training = _attempt(TrainingRunId("tr-20260908-201"))
        evaluation = _attempt(EvaluationRunId("er-20260908-202"))
        self.assertEqual("tr-20260908-201", training.run_id)
        self.assertEqual("er-20260908-202", evaluation.run_id)

    def test_queue_database_path_is_exact_repository_local_path(self) -> None:
        root = Path("C:/repository-root")
        self.assertEqual(
            root / ".local" / "mldb" / "queue.sqlite",
            queue_database_path(root),
        )

    def test_queue_port_exposes_semantic_consumer_operations_not_crud(self) -> None:
        methods = {
            name
            for name, value in QueuePort.__dict__.items()
            if inspect.isfunction(value) and not name.startswith("__")
        }
        self.assertEqual(
            {
                "admit_study_jobs",
                "jobs_for_study_run",
                "job_by_id",
                "attempts_for_job",
                "attempt_by_id",
                "open_attempt_for_job",
                "attempt_by_acquire_token",
                "open_attempt_for_worker",
                "authorized_open_attempt",
                "select_ready_job",
                "defer_ready_job",
                "fail_ready_job",
                "activate_attempt",
                "heartbeat_attempt",
                "expired_open_attempts",
                "close_attempt",
                "cancel_job",
                "repair_job_state",
            },
            methods,
        )
        self.assertTrue(methods.isdisjoint({"create", "read", "update", "delete"}))


class RetryDecisionContractTests(unittest.TestCase):
    def test_retry_decision_has_only_retry_and_no_retry_forms(self) -> None:
        retry_at = "2026-09-08T02:03:04.000005Z"
        retry = RetryDecision(retry_not_before=retry_at)
        stop = RetryDecision(retry_not_before=None)

        self.assertEqual(retry_at, retry.retry_not_before)
        self.assertIsNone(stop.retry_not_before)
        self.assertEqual(("retry_not_before",), tuple(field.name for field in fields(RetryDecision)))
        with self.assertRaises(FrozenInstanceError):
            retry.retry_not_before = None  # type: ignore[misc]


def _training_row(trial: str, stages: tuple[str, ...]) -> StudyPlanRow:
    return StudyPlanRow(
        trial=trial,
        training=StudyPlanTraining(
            architecture=ArchitectureId("arch-private-v1"),
            corpus=CorpusId("train-corpus-private-v1"),
            protocol=TrainProtocolId("train-protocol-private-v1"),
            seed=42,
            parameters={
                "private": "secret-parameter-value",
                "nested": [{"x": [1, True, None]}],
            },
        ),
        evaluations=[_evaluation(stage) for stage in stages],
    )


def _existing_model_row(
    trial: str,
    model: str,
    stages: tuple[str, ...],
) -> StudyPlanRow:
    return StudyPlanRow(
        trial=trial,
        model=ModelId(model),
        evaluations=[_evaluation(stage) for stage in stages],
    )


def _evaluation(stage: str) -> StudyPlanEvaluation:
    return StudyPlanEvaluation(
        stage=stage,
        corpus=CorpusId("eval-corpus-private-v1"),
        protocol=EvaluationProtocolId("eval-protocol-private-v1"),
        parameters={"threshold": 0.5, "private": "secret-parameter-value"},
    )


def _attempt(run_id: TrainingRunId | EvaluationRunId) -> QueueAttempt:
    return QueueAttempt(
        attempt_id=1,
        job_id=1,
        attempt_no=1,
        run_id=run_id,
        worker_id="worker-1",
        acquire_token="acquire-1",
        lease_token="lease-1",
        lease_until="2026-09-08T03:00:00.000000Z",
        started_at="2026-09-08T02:00:00.000000Z",
    )


if __name__ == "__main__":
    unittest.main()
