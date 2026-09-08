from __future__ import annotations

import hashlib
import inspect
from threading import Event, Thread
import time
import unittest
from unittest.mock import MagicMock, patch

from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    EvaluationRunId,
    ModelId,
    StudyId,
    StudyRunId,
    TrainingRunId,
    TrainProtocolId,
)
from mldb.src.evaluation.interface import UnavailableOutput
from mldb.src.evaluation.run import (
    EvaluationRun,
    EvaluationRunExecution,
    EvaluationRunFailure,
    EvaluationRunResult,
    EvaluationRunStatus,
    EvaluationRunStudyLineage,
)
from mldb.src.orchestration.jobs import (
    EvaluationJob,
    EvaluationJobCoordinate,
    TrainingJob,
    TrainingJobCoordinate,
)
from mldb.src.orchestration.queue import QueueAttempt, QueueJob, QueueJobStatus
from mldb.src.orchestration.retry_policy import RetryDecision
from mldb.src.orchestration.worker_api import (
    AttemptCancelled,
    AttemptFailed,
    CandidateArtifactRef,
    EvaluationSuccessCandidate,
    EvaluationSucceeded,
    OutcomeAcknowledgement,
    TrainingSuccessCandidate,
    TrainingSucceeded,
)
from mldb.src.study.run import StudyRun, StudyRunExecution, StudyRunStatus
from mldb.src.training.run import (
    TrainingRun,
    TrainingRunExecution,
    TrainingRunFailure,
    TrainingRunResult,
    TrainingRunStatus,
    TrainingRunStudyLineage,
)
from mldb.src.training.weights import CanonicalWeightsArtifact
import mldb.src.orchestration.outcome as subject


T0 = "2026-09-08T10:00:00.000000Z"
T1 = "2026-09-08T10:30:00.000000Z"
T2 = "2026-09-08T11:30:00.000000Z"
STUDY = StudyRunId("sr-20260908-001")


class CandidateStore:
    def __init__(self, payload: bytes = b"weights") -> None:
        self.payload = payload

    def read(self, attempt_id: int, key: str, content_identity: str) -> bytes:
        return self.payload


def study(status: StudyRunStatus) -> StudyRun:
    return StudyRun(
        schema="mjtensu.mldb/study-run/v1",
        id=STUDY,
        status=status,
        study=StudyId("study.test"),
        execution=StudyRunExecution(
            started_at="started",
            finished_at=None if status is StudyRunStatus.RUNNING else "finished",
        ),
    )


def queue_job(kind: str, *, status: QueueJobStatus = QueueJobStatus.ACTIVE) -> QueueJob:
    if kind == "training":
        logical = TrainingJob(TrainingJobCoordinate(STUDY, "trial-0001"))
        job_id = 1
    else:
        logical = EvaluationJob(
            EvaluationJobCoordinate(STUDY, "trial-0001", "eval"),
            None,
        )
        job_id = 2
    return QueueJob(job_id, logical, status, None, None, T0, T0)


def attempt(job: QueueJob, run_id: str, *, finished_at: str | None = None) -> QueueAttempt:
    return QueueAttempt(
        10,
        job.job_id,
        1,
        run_id,
        "worker",
        "acquire",
        "lease",
        T2,
        T0,
        finished_at,
    )


def training_run(
    status: TrainingRunStatus,
    *,
    failure: TrainingRunFailure | None = None,
    payload: bytes = b"weights",
) -> TrainingRun:
    result = None
    if status is TrainingRunStatus.COMPLETED:
        result = TrainingRunResult(
            CanonicalWeightsArtifact(
                "pytorch-state-dict",
                "artifacts/weights.pt",
                hashlib.sha256(payload).hexdigest(),
                len(payload),
            )
        )
    return TrainingRun(
        schema="mjtensu.mldb/training-run/v1",
        id=TrainingRunId("tr-20260908-001"),
        status=status,
        corpus=CorpusId("corpus.test"),
        architecture=ArchitectureId("arch.test"),
        train_protocol=TrainProtocolId("train.test"),
        parameters={},
        execution=TrainingRunExecution(
            seed=1,
            started_at="started",
            finished_at=None if status is TrainingRunStatus.RUNNING else "finished",
        ),
        result=result,
        failure=failure,
        study=TrainingRunStudyLineage(STUDY, "trial-0001"),
    )


def evaluation_run(
    status: EvaluationRunStatus,
    *,
    failure: EvaluationRunFailure | None = None,
) -> EvaluationRun:
    partial = status is EvaluationRunStatus.COMPLETED_PARTIAL
    result = None
    if status in {EvaluationRunStatus.COMPLETED, EvaluationRunStatus.COMPLETED_PARTIAL}:
        result = EvaluationRunResult(metrics={"score": 1.0}, artifacts={})
    unavailable = (
        (UnavailableOutput("metrics.other", "missing", "not produced"),)
        if partial
        else ()
    )
    return EvaluationRun(
        schema="mjtensu.mldb/evaluation-run/v1",
        id=EvaluationRunId("ev-20260908-001"),
        status=status,
        model=ModelId("model.test"),
        corpus=CorpusId("corpus.test"),
        evaluation_protocol=EvaluationProtocolId("eval.test"),
        parameters={},
        execution=EvaluationRunExecution(
            started_at="started",
            finished_at=None if status is EvaluationRunStatus.RUNNING else "finished",
        ),
        result=result,
        unavailable_outputs=unavailable,
        failure=failure,
        study=EvaluationRunStudyLineage(STUDY, "trial-0001", "eval"),
    )


def training_success(payload: bytes = b"weights", *, bytes_override: int | None = None):
    digest = hashlib.sha256(payload).hexdigest()
    return TrainingSucceeded(
        10,
        "lease",
        TrainingSuccessCandidate(
            CandidateArtifactRef(
                "weights",
                digest,
                len(payload) if bytes_override is None else bytes_override,
            )
        ),
    )


def evaluation_success(*, partial: bool = False):
    unavailable = (
        (UnavailableOutput("metrics.other", "missing", "not produced"),)
        if partial
        else ()
    )
    return EvaluationSucceeded(
        10,
        "lease",
        EvaluationSuccessCandidate(
            metrics={"score": 1.0},
            artifacts={},
            unavailable_outputs=unavailable,
        ),
    )


class OutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = object()
        self.filesystem = object()
        self.queue = MagicMock()
        self.policy = MagicMock()
        self.policy.after_unsatisfied_attempt.return_value = RetryDecision(None)
        self.candidates = CandidateStore()

    def authorize(self, job: QueueJob, run_id: str) -> QueueAttempt:
        current = attempt(job, run_id)
        self.queue.authorized_open_attempt.return_value = current
        self.queue.job_by_id.return_value = job
        self.queue.attempts_for_job.return_value = (current,)
        return current

    def call(self, outcome):
        return subject.handle_attempt_outcome(
            outcome,
            "run-finished",
            T1,
            self.layout,
            self.filesystem,
            self.queue,
            self.candidates,
            self.policy,
        )

    def test_training_success_accepts_before_queue_satisfied(self) -> None:
        job = queue_job("training")
        self.authorize(job, "tr-20260908-001")
        running = training_run(TrainingRunStatus.RUNNING)
        completed = training_run(TrainingRunStatus.COMPLETED)
        events: list[str] = []

        def accept(*args, **kwargs):
            events.append("canonical")
            return completed

        self.queue.close_attempt.side_effect = lambda *a, **k: events.append("queue")
        with patch.object(subject, "read_training_run", return_value=running), patch.object(
            subject, "accept_training_success", side_effect=accept
        ):
            result = self.call(training_success())

        self.assertIs(OutcomeAcknowledgement.ACCEPTED, result)
        self.assertEqual(["canonical", "queue"], events)
        self.assertEqual(
            QueueJobStatus.SATISFIED,
            self.queue.close_attempt.call_args.kwargs["target_status"],
        )

    def test_training_model_gap_recovery_and_terminal_before_queue_close(self) -> None:
        job = queue_job("training")
        self.authorize(job, "tr-20260908-001")
        completed = training_run(TrainingRunStatus.COMPLETED)
        with patch.object(subject, "read_training_run", return_value=completed), patch.object(
            subject, "ensure_model_for_completed_training_run"
        ) as ensure:
            result = self.call(training_success())
        self.assertIs(OutcomeAcknowledgement.ACCEPTED, result)
        ensure.assert_called_once_with(completed, self.layout, self.filesystem)
        self.assertEqual(
            QueueJobStatus.SATISFIED,
            self.queue.close_attempt.call_args.kwargs["target_status"],
        )

    def test_evaluation_success_and_partial_policy(self) -> None:
        job = queue_job("evaluation")
        self.authorize(job, "ev-20260908-001")
        running = evaluation_run(EvaluationRunStatus.RUNNING)
        completed = evaluation_run(EvaluationRunStatus.COMPLETED)
        with patch.object(subject, "read_evaluation_run", return_value=running), patch.object(
            subject, "accept_evaluation_success", return_value=completed
        ):
            self.assertIs(OutcomeAcknowledgement.ACCEPTED, self.call(evaluation_success()))
        self.assertEqual(
            QueueJobStatus.SATISFIED,
            self.queue.close_attempt.call_args.kwargs["target_status"],
        )

        self.queue.reset_mock()
        self.policy.reset_mock()
        self.policy.after_unsatisfied_attempt.return_value = RetryDecision(T2)
        self.authorize(job, "ev-20260908-001")
        partial = evaluation_run(EvaluationRunStatus.COMPLETED_PARTIAL)
        with patch.object(subject, "read_evaluation_run", return_value=running), patch.object(
            subject, "accept_evaluation_success", return_value=partial
        ), patch.object(subject, "read_study_run", return_value=study(StudyRunStatus.RUNNING)):
            self.assertIs(
                OutcomeAcknowledgement.ACCEPTED,
                self.call(evaluation_success(partial=True)),
            )
        self.policy.after_unsatisfied_attempt.assert_called_once()
        self.assertEqual(
            QueueJobStatus.RETRY_WAIT,
            self.queue.close_attempt.call_args.kwargs["target_status"],
        )
        self.assertEqual(T2, self.queue.close_attempt.call_args.kwargs["retry_not_before"])

    def test_attempt_failed_preserves_exact_worker_metadata(self) -> None:
        job = queue_job("training")
        self.authorize(job, "tr-20260908-001")
        running = training_run(TrainingRunStatus.RUNNING)
        captured = {}

        def fail(run, finished_at, layout, filesystem, *, failure):
            captured["failure"] = failure
            return training_run(TrainingRunStatus.FAILED, failure=failure)

        with patch.object(subject, "read_training_run", return_value=running), patch.object(
            subject, "fail_training_run", side_effect=fail
        ), patch.object(subject, "read_study_run", return_value=study(StudyRunStatus.RUNNING)):
            result = self.call(AttemptFailed(10, "lease", "worker-error", "exact message"))

        self.assertIs(OutcomeAcknowledgement.ACCEPTED, result)
        self.assertEqual("worker-error", captured["failure"].type)
        self.assertEqual("exact message", captured["failure"].message)
        self.assertEqual(
            QueueJobStatus.FAILED,
            self.queue.close_attempt.call_args.kwargs["target_status"],
        )

    def test_attempt_cancelled_uses_policy_only_while_study_running(self) -> None:
        job = queue_job("evaluation")
        self.authorize(job, "ev-20260908-001")
        running = evaluation_run(EvaluationRunStatus.RUNNING)
        with patch.object(subject, "read_evaluation_run", return_value=running), patch.object(
            subject, "cancel_evaluation_run"
        ), patch.object(subject, "read_study_run", return_value=study(StudyRunStatus.RUNNING)):
            result = self.call(AttemptCancelled(10, "lease"))
        self.assertIs(OutcomeAcknowledgement.ACCEPTED, result)
        self.policy.after_unsatisfied_attempt.assert_called_once()
        self.assertEqual("cancelled", self.policy.after_unsatisfied_attempt.call_args.args[2])

    def test_cancelled_study_suppresses_policy_and_cancels_job(self) -> None:
        job = queue_job("training")
        self.authorize(job, "tr-20260908-001")
        failed = training_run(
            TrainingRunStatus.FAILED,
            failure=TrainingRunFailure("x", "y"),
        )
        with patch.object(subject, "read_training_run", return_value=failed), patch.object(
            subject, "read_study_run", return_value=study(StudyRunStatus.CANCELLED)
        ):
            result = self.call(AttemptFailed(10, "lease", "x", "y"))
        self.assertIs(OutcomeAcknowledgement.ACCEPTED, result)
        self.policy.after_unsatisfied_attempt.assert_not_called()
        self.queue.cancel_job.assert_called_once_with(job.job_id, at=T1)

    def test_other_terminal_study_is_inconsistency(self) -> None:
        job = queue_job("training")
        self.authorize(job, "tr-20260908-001")
        failed = training_run(
            TrainingRunStatus.FAILED,
            failure=TrainingRunFailure("x", "y"),
        )
        with patch.object(subject, "read_training_run", return_value=failed), patch.object(
            subject, "read_study_run", return_value=study(StudyRunStatus.COMPLETED)
        ):
            with self.assertRaises(RuntimeError):
                self.call(AttemptFailed(10, "lease", "x", "y"))
        self.policy.after_unsatisfied_attempt.assert_not_called()

    def test_success_candidate_acceptance_failure_fails_only_fresh_running(self) -> None:
        job = queue_job("training")
        self.authorize(job, "tr-20260908-001")
        running = training_run(TrainingRunStatus.RUNNING)
        self.queue.attempts_for_job.return_value = ()
        with patch.object(
            subject, "read_training_run", side_effect=[running, running]
        ), patch.object(
            subject, "accept_training_success", side_effect=ValueError("bad candidate")
        ), patch.object(subject, "fail_training_run") as fail, patch.object(
            subject, "read_study_run", return_value=study(StudyRunStatus.RUNNING)
        ):
            result = self.call(training_success())
        self.assertIs(OutcomeAcknowledgement.REJECTED, result)
        self.assertIsNone(fail.call_args.kwargs["failure"])
        self.assertEqual(
            QueueJobStatus.FAILED,
            self.queue.close_attempt.call_args.kwargs["target_status"],
        )

    def test_closed_exact_replay_and_conflicting_replay(self) -> None:
        job = queue_job("training", status=QueueJobStatus.SATISFIED)
        closed = attempt(job, "tr-20260908-001", finished_at=T1)
        self.queue.authorized_open_attempt.return_value = None
        self.queue.attempt_by_id.return_value = closed
        self.queue.job_by_id.return_value = job
        completed = training_run(TrainingRunStatus.COMPLETED)

        with patch.object(subject, "read_training_run", return_value=completed), patch.object(
            subject, "ensure_model_for_completed_training_run"
        ):
            self.assertIs(
                OutcomeAcknowledgement.ALREADY_FINALIZED,
                self.call(training_success()),
            )
            self.assertIs(
                OutcomeAcknowledgement.REJECTED,
                self.call(training_success(bytes_override=999)),
            )
        self.queue.close_attempt.assert_not_called()

    def test_failure_none_is_not_worker_failure_replay_evidence(self) -> None:
        job = queue_job("training", status=QueueJobStatus.FAILED)
        self.queue.authorized_open_attempt.return_value = None
        self.queue.attempt_by_id.return_value = attempt(
            job, "tr-20260908-001", finished_at=T1
        )
        self.queue.job_by_id.return_value = job
        failed = training_run(TrainingRunStatus.FAILED, failure=None)
        with patch.object(subject, "read_training_run", return_value=failed):
            result = self.call(AttemptFailed(10, "lease", "worker", "boom"))
        self.assertIs(OutcomeAcknowledgement.REJECTED, result)

    def test_stale_open_lease_is_rejected_without_mutation(self) -> None:
        job = queue_job("training")
        self.queue.authorized_open_attempt.return_value = None
        self.queue.attempt_by_id.return_value = attempt(job, "tr-20260908-001")
        result = self.call(AttemptCancelled(10, "lease"))
        self.assertIs(OutcomeAcknowledgement.REJECTED, result)
        self.queue.job_by_id.assert_not_called()
        self.queue.close_attempt.assert_not_called()

    def test_job_kind_prevents_cross_success_type(self) -> None:
        job = queue_job("training")
        self.authorize(job, "tr-20260908-001")
        with patch.object(subject, "read_training_run") as read:
            result = self.call(evaluation_success())
        self.assertIs(OutcomeAcknowledgement.REJECTED, result)
        read.assert_not_called()

    def test_candidate_store_failure_is_not_silently_rejected_on_replay(self) -> None:
        job = queue_job("training", status=QueueJobStatus.SATISFIED)
        self.queue.authorized_open_attempt.return_value = None
        self.queue.attempt_by_id.return_value = attempt(
            job, "tr-20260908-001", finished_at=T1
        )
        self.queue.job_by_id.return_value = job
        completed = training_run(TrainingRunStatus.COMPLETED)
        broken = MagicMock()
        broken.read.side_effect = OSError("store unavailable")
        self.candidates = broken
        with patch.object(subject, "read_training_run", return_value=completed):
            with self.assertRaises(OSError):
                self.call(training_success())

    def test_same_attempt_outcome_waits_for_recovery_exclusion(self) -> None:
        import mldb.src.orchestration.lease_recovery as recovery

        job = queue_job("training")
        closed = attempt(job, "tr-20260908-001", finished_at=T1)
        entered = Event()
        release = Event()
        outcome_authorized = Event()

        class SharedQueue:
            def expired_open_attempts(self, *, as_of):
                return (attempt(job, "tr-20260908-001"),)

            def attempt_by_id(self, attempt_id):
                if not entered.is_set():
                    entered.set()
                    release.wait(2)
                return closed

            def authorized_open_attempt(self, attempt_id, lease_token, *, as_of):
                outcome_authorized.set()
                return None

            def job_by_id(self, job_id):
                return None

        queue = SharedQueue()
        recovery_thread = Thread(
            target=lambda: recovery.recover_expired_attempts(
                as_of=T1,
                run_finished_at="done",
                layout=self.layout,
                filesystem=self.filesystem,
                queue=queue,
                retry_policy=self.policy,
            ),
            daemon=True,
        )
        recovery_thread.start()
        self.assertTrue(entered.wait(1))

        outcome_thread = Thread(
            target=lambda: subject.handle_attempt_outcome(
                AttemptCancelled(10, "lease"),
                "done",
                T1,
                self.layout,
                self.filesystem,
                queue,
                self.candidates,
                self.policy,
            ),
            daemon=True,
        )
        outcome_thread.start()
        time.sleep(0.05)
        self.assertFalse(outcome_authorized.is_set())
        release.set()
        recovery_thread.join(2)
        outcome_thread.join(2)
        self.assertTrue(outcome_authorized.is_set())

    def test_signature_is_frozen(self) -> None:
        self.assertEqual(
            [
                "outcome",
                "run_finished_at",
                "queue_at",
                "layout",
                "filesystem",
                "queue",
                "candidates",
                "retry_policy",
            ],
            list(inspect.signature(subject.handle_attempt_outcome).parameters),
        )


if __name__ == "__main__":
    unittest.main()
