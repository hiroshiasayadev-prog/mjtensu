from __future__ import annotations

import ast
import dataclasses
import hashlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mldb.src.evaluation.interface import EvaluationResult, UnavailableOutput
from mldb.src.orchestration import worker_loop as subject
from mldb.src.orchestration.worker_api import (
    AssetRejection,
    CandidateUploadAcknowledgement,
    EvaluationAssignment,
    EvaluationSucceeded,
    HeartbeatAccepted,
    HeartbeatRejection,
    OutcomeAcknowledgement,
    RetrievedAsset,
    TrainingAssignment,
    TrainingSucceeded,
    AttemptCancelled,
    AttemptFailed,
    ImmutableAssetDescriptor,
)
from mldb.src.orchestration.worker_execution import (
    EvaluationExecutionFiles,
    TrainingExecutionCandidate,
    TrainingExecutionFiles,
)
from mldb.src.training.weights import CanonicalWeightsArtifact
from mldb.tests.signature_guard import compare_module_signatures


def _descriptor(key: str, content: bytes) -> ImmutableAssetDescriptor:
    return ImmutableAssetDescriptor(
        key=key,
        retrieval_id=f"retrieve-{key}",
        sha256=hashlib.sha256(content).hexdigest(),
        bytes=len(content),
    )


def _training_assignment(contents: dict[str, bytes]) -> TrainingAssignment:
    return TrainingAssignment(
        attempt_id=11,
        run_id="tr-20260908-011",
        lease_token="lease-t",
        lease_until="2026-09-08T19:00:00.000000Z",
        kind="training",
        task=object(), corpus=object(), architecture=object(), protocol=object(),
        seed=42, parameters={"epochs": 3},
        corpus_artifact=_descriptor("corpus", contents["corpus"]),
        architecture_implementation=_descriptor("architecture", contents["architecture"]),
        train_protocol_implementation=_descriptor("train", contents["train"]),
    )


def _evaluation_assignment(contents: dict[str, bytes]) -> EvaluationAssignment:
    return EvaluationAssignment(
        attempt_id=12,
        run_id="ev-20260908-012",
        lease_token="lease-e",
        lease_until="2026-09-08T19:00:00.000000Z",
        kind="evaluation",
        task=object(), corpus=object(), model=object(), training_run=object(),
        model_architecture=object(), protocol=object(), parameters={"batch": 8},
        corpus_artifact=_descriptor("corpus", contents["corpus"]),
        model_architecture_implementation=_descriptor("architecture", contents["architecture"]),
        model_weights=_descriptor("weights", contents["weights"]),
        evaluation_protocol_implementation=_descriptor("evaluation", contents["evaluation"]),
    )


class _Api:
    def __init__(self, assets: dict[str, bytes]) -> None:
        self.assets = dict(assets)
        self.retrieve_requests = []
        self.upload_requests = []
        self.outcomes = []
        self.heartbeat_requests = []
        self.heartbeat_response = HeartbeatAccepted(
            lease_until="2026-09-08T20:00:00.000000Z",
            cancel_requested=False,
        )
        self.upload_ack = CandidateUploadAcknowledgement.ACCEPTED
        self.outcome_ack = OutcomeAcknowledgement.ACCEPTED

    def heartbeat(self, request):
        self.heartbeat_requests.append(request)
        return self.heartbeat_response

    def retrieve_asset(self, request):
        self.retrieve_requests.append(request)
        return RetrievedAsset(self.assets[request.asset.key])

    def upload_candidate(self, request):
        self.upload_requests.append(request)
        return self.upload_ack

    def report_outcome(self, outcome):
        self.outcomes.append(outcome)
        return self.outcome_ack


def _training_materializer(root: Path):
    def materialize(assignment, assets):
        corpus = root / "corpus.bin"; corpus.write_bytes(assets["corpus"])
        architecture = root / "architecture.py"; architecture.write_bytes(assets["architecture"])
        train = root / "train.py"; train.write_bytes(assets["train"])
        return TrainingExecutionFiles(
            corpus_artifact=corpus,
            architecture_implementation=architecture,
            train_protocol_implementation=train,
            work_dir=root / "work",
            weights_candidate_path=root / "candidate.pt",
        )
    return materialize


def _evaluation_materializer(root: Path):
    def materialize(assignment, assets):
        corpus = root / "corpus.bin"; corpus.write_bytes(assets["corpus"])
        architecture = root / "architecture.py"; architecture.write_bytes(assets["architecture"])
        weights = root / "weights.pt"; weights.write_bytes(assets["weights"])
        evaluation = root / "evaluate.py"; evaluation.write_bytes(assets["evaluation"])
        work = root / "work"; work.mkdir(exist_ok=True)
        return EvaluationExecutionFiles(
            corpus_artifact=corpus,
            model_architecture_implementation=architecture,
            model_weights=weights,
            evaluation_protocol_implementation=evaluation,
            work_dir=work,
        )
    return materialize


class WorkerLoopTests(unittest.TestCase):
    def setUp(self):
        self.training_assets = {"corpus": b"c", "architecture": b"a", "train": b"t"}
        self.evaluation_assets = {"corpus": b"c", "architecture": b"a", "weights": b"w", "evaluation": b"e"}

    def test_training_success_exact_descriptor_order_candidate_and_outcome(self):
        assignment = _training_assignment(self.training_assets)
        api = _Api(self.training_assets)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = b"trained-weights"
            path = root / "candidate.pt"; path.write_bytes(weights)
            candidate = TrainingExecutionCandidate(
                local_weights_path=path,
                artifact=CanonicalWeightsArtifact(
                    format="pytorch-state-dict", path="artifacts/weights.pt",
                    sha256=hashlib.sha256(weights).hexdigest(), bytes=len(weights),
                ),
            )
            with patch.object(subject, "execute_training_attempt", return_value=candidate) as execute:
                subject.run_worker_attempt(assignment, api, _training_materializer(root), lambda exc: False)
        execute.assert_called_once()
        self.assertEqual(["corpus", "architecture", "train"], [r.asset.key for r in api.retrieve_requests])
        self.assertEqual(1, len(api.upload_requests))
        upload = api.upload_requests[0]
        self.assertEqual("artifacts/weights.pt", upload.key)
        self.assertEqual(weights, upload.content)
        self.assertIsInstance(api.outcomes[0], TrainingSucceeded)
        self.assertEqual("artifacts/weights.pt", api.outcomes[0].candidate.weights.key)

    def test_evaluation_success_preserves_mapping_keys_metrics_and_unavailable(self):
        assignment = _evaluation_assignment(self.evaluation_assets)
        api = _Api(self.evaluation_assets)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize = _evaluation_materializer(root)
            work = root / "work"; work.mkdir()
            first = work / "a.json"; first.write_bytes(b"one")
            second = work / "b.csv"; second.write_bytes(b"two")
            metrics = {"accuracy": 0.9}
            unavailable = (UnavailableOutput("metrics.loss", "Missing", "not emitted"),)
            result = EvaluationResult(metrics=metrics, artifacts={"report": first, "rows": second}, unavailable_outputs=unavailable)
            with patch.object(subject, "execute_evaluation_attempt", return_value=result) as execute:
                subject.run_worker_attempt(assignment, api, materialize, lambda exc: False)
        execute.assert_called_once()
        self.assertEqual(["corpus", "architecture", "weights", "evaluation"], [r.asset.key for r in api.retrieve_requests])
        self.assertEqual(["report", "rows"], [r.key for r in api.upload_requests])
        outcome = api.outcomes[0]
        self.assertIsInstance(outcome, EvaluationSucceeded)
        self.assertIs(outcome.candidate.metrics, metrics)
        self.assertIs(outcome.candidate.unavailable_outputs, unavailable)
        self.assertEqual(["report", "rows"], list(outcome.candidate.artifacts))

    def test_retrieval_communication_retry_uses_same_request_and_exact_ten_seconds(self):
        assignment = _training_assignment(self.training_assets)
        api = _Api(self.training_assets)
        error = OSError("lost")
        calls = 0
        original = api.retrieve_asset
        def retrieve(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                api.retrieve_requests.append(request)
                raise error
            return original(request)
        api.retrieve_asset = retrieve
        with tempfile.TemporaryDirectory() as temporary, patch.object(subject.time, "sleep") as sleep, patch.object(subject, "execute_training_attempt", side_effect=RuntimeError("domain")):
            subject.run_worker_attempt(assignment, api, _training_materializer(Path(temporary)), lambda exc: exc is error)
        sleep.assert_called_once_with(10.0)
        self.assertIs(api.retrieve_requests[0], api.retrieve_requests[1])

    def test_nonretryable_api_failure_propagates_unchanged(self):
        assignment = _training_assignment(self.training_assets)
        api = _Api(self.training_assets)
        error = RuntimeError("protocol down")
        api.retrieve_asset = Mock(side_effect=error)
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(RuntimeError) as raised:
            subject.run_worker_attempt(assignment, api, _training_materializer(Path(temporary)), lambda exc: False)
        self.assertIs(error, raised.exception)
        self.assertEqual([], api.outcomes)

    def test_heartbeat_communication_retry_waits_exact_ten_seconds(self):
        assignment = _training_assignment(self.training_assets)
        api = _Api(self.training_assets)
        error = OSError("heartbeat lost")
        api.heartbeat = Mock(side_effect=[error, HeartbeatRejection("Stale", "done")])
        control = subject._HeartbeatControl()
        control.stop.wait = Mock(return_value=False)
        subject._heartbeat_loop(assignment, api, lambda exc: exc is error, control)
        self.assertEqual(2, api.heartbeat.call_count)
        self.assertEqual(10.0, control.stop.wait.call_args_list[0].args[0])
        self.assertTrue(control.lease_lost)

    def test_optional_descriptor_byte_count_mismatch_is_attempt_failure(self):
        assignment = _training_assignment(self.training_assets)
        descriptor = dataclasses.replace(assignment.corpus_artifact, bytes=999)
        assignment = dataclasses.replace(assignment, corpus_artifact=descriptor)
        api = _Api(self.training_assets)
        with tempfile.TemporaryDirectory() as temporary, patch.object(subject, "execute_training_attempt") as execute:
            subject.run_worker_attempt(assignment, api, _training_materializer(Path(temporary)), lambda exc: False)
        execute.assert_not_called()
        self.assertIsInstance(api.outcomes[0], AttemptFailed)
        self.assertEqual("ValueError", api.outcomes[0].type)

    def test_asset_rejection_reports_failure_without_domain(self):
        assignment = _training_assignment(self.training_assets)
        api = _Api(self.training_assets)
        api.retrieve_asset = Mock(return_value=AssetRejection("Unavailable", "gone"))
        with tempfile.TemporaryDirectory() as temporary, patch.object(subject, "execute_training_attempt") as execute:
            subject.run_worker_attempt(assignment, api, _training_materializer(Path(temporary)), lambda exc: False)
        execute.assert_not_called()
        self.assertIsInstance(api.outcomes[0], AttemptFailed)
        self.assertEqual(("Unavailable", "gone"), (api.outcomes[0].type, api.outcomes[0].message))

    def test_sha_mismatch_and_materialization_failure_are_attempt_failures(self):
        assignment = _training_assignment(self.training_assets)
        api = _Api(self.training_assets)
        api.assets["corpus"] = b"tampered"
        with tempfile.TemporaryDirectory() as temporary, patch.object(subject, "execute_training_attempt") as execute:
            subject.run_worker_attempt(assignment, api, _training_materializer(Path(temporary)), lambda exc: False)
        execute.assert_not_called()
        self.assertEqual("ValueError", api.outcomes[0].type)

        api = _Api(self.training_assets)
        failure = OSError("materialize failed")
        subject.run_worker_attempt(assignment, api, Mock(side_effect=failure), lambda exc: False)
        self.assertEqual("OSError", api.outcomes[0].type)

    def test_post_materialization_tamper_prevents_domain(self):
        assignment = _training_assignment(self.training_assets)
        api = _Api(self.training_assets)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = _training_materializer(root)
            def materialize(a, assets):
                files = base(a, assets)
                files.corpus_artifact.write_bytes(b"changed")
                return files
            with patch.object(subject, "execute_training_attempt") as execute:
                subject.run_worker_attempt(assignment, api, materialize, lambda exc: False)
        execute.assert_not_called()
        self.assertEqual("ValueError", api.outcomes[0].type)

    def test_cancellation_before_domain_reports_cancelled(self):
        assignment = _training_assignment(self.training_assets)
        api = _Api(self.training_assets)
        api.heartbeat_response = HeartbeatAccepted("later", True)
        with tempfile.TemporaryDirectory() as temporary, patch.object(subject, "execute_training_attempt") as execute:
            subject.run_worker_attempt(assignment, api, _training_materializer(Path(temporary)), lambda exc: False)
        execute.assert_not_called()
        self.assertTrue(api.outcomes)
        self.assertIsInstance(api.outcomes[0], AttemptCancelled)

    def test_lease_rejection_before_domain_starts_no_domain_candidate_or_outcome(self):
        assignment = _training_assignment(self.training_assets)
        api = _Api(self.training_assets)
        api.heartbeat_response = HeartbeatRejection("Stale", "lost")
        with tempfile.TemporaryDirectory() as temporary, patch.object(subject, "execute_training_attempt") as execute:
            subject.run_worker_attempt(assignment, api, _training_materializer(Path(temporary)), lambda exc: False)
        execute.assert_not_called()
        self.assertEqual([], api.upload_requests)
        self.assertEqual([], api.outcomes)

    def test_cancellation_during_successful_domain_keeps_truthful_success(self):
        assignment = _training_assignment(self.training_assets)
        api = _Api(self.training_assets)
        heartbeat_calls = 0
        domain_entered = threading.Event()
        def heartbeat(request):
            nonlocal heartbeat_calls
            heartbeat_calls += 1
            return HeartbeatAccepted("later", domain_entered.is_set())
        api.heartbeat = heartbeat
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = b"trained"
            path = root / "candidate.pt"; path.write_bytes(weights)
            candidate = TrainingExecutionCandidate(path, CanonicalWeightsArtifact("pytorch-state-dict", "artifacts/weights.pt", hashlib.sha256(weights).hexdigest(), len(weights)))
            def execute(a, files):
                domain_entered.set()
                deadline = time.monotonic() + 1
                while heartbeat_calls < 2 and time.monotonic() < deadline:
                    time.sleep(0.001)
                return candidate
            with patch.object(subject, "_HEALTHY_HEARTBEAT_INTERVAL_SECONDS", 0.001), patch.object(subject, "execute_training_attempt", side_effect=execute) as domain:
                subject.run_worker_attempt(assignment, api, _training_materializer(root), lambda exc: False)
        domain.assert_called_once()
        self.assertIsInstance(api.outcomes[0], TrainingSucceeded)

    def test_domain_failure_is_reported_once_and_never_classified_as_communication(self):
        assignment = _training_assignment(self.training_assets)
        api = _Api(self.training_assets)
        classifier = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as temporary, patch.object(subject, "execute_training_attempt", side_effect=RuntimeError("train failed")) as domain:
            subject.run_worker_attempt(assignment, api, _training_materializer(Path(temporary)), classifier)
        domain.assert_called_once()
        self.assertIsInstance(api.outcomes[0], AttemptFailed)
        self.assertEqual("RuntimeError", api.outcomes[0].type)
        classifier.assert_not_called()

    def test_training_upload_already_present_and_rejected_are_definitive_without_rerun(self):
        assignment = _training_assignment(self.training_assets)
        for ack, outcome_type in ((CandidateUploadAcknowledgement.ALREADY_PRESENT, TrainingSucceeded), (CandidateUploadAcknowledgement.REJECTED, AttemptFailed)):
            api = _Api(self.training_assets); api.upload_ack = ack
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); weights = b"trained"; path = root / "candidate.pt"; path.write_bytes(weights)
                candidate = TrainingExecutionCandidate(path, CanonicalWeightsArtifact("pytorch-state-dict", "artifacts/weights.pt", hashlib.sha256(weights).hexdigest(), len(weights)))
                with patch.object(subject, "execute_training_attempt", return_value=candidate) as domain:
                    subject.run_worker_attempt(assignment, api, _training_materializer(root), lambda exc: False)
            domain.assert_called_once()
            self.assertIsInstance(api.outcomes[0], outcome_type)

    def test_evaluation_upload_retry_preserves_prior_success_and_same_request(self):
        assignment = _evaluation_assignment(self.evaluation_assets)
        api = _Api(self.evaluation_assets)
        attempts = []
        def upload(request):
            attempts.append(request)
            if request.key == "b" and sum(r.key == "b" for r in attempts) == 1:
                raise OSError("lost ack")
            return CandidateUploadAcknowledgement.ACCEPTED
        api.upload_candidate = upload
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); work = root / "work"; work.mkdir()
            a = work / "a"; a.write_bytes(b"a")
            b = work / "b"; b.write_bytes(b"b")
            result = EvaluationResult({}, {"a": a, "b": b}, ())
            with patch.object(subject.time, "sleep") as sleep, patch.object(subject, "execute_evaluation_attempt", return_value=result) as domain:
                subject.run_worker_attempt(assignment, api, _evaluation_materializer(root), lambda exc: isinstance(exc, OSError))
        domain.assert_called_once()
        sleep.assert_called_once_with(10.0)
        self.assertEqual(["a", "b", "b"], [r.key for r in attempts])
        self.assertIs(attempts[1], attempts[2])
        self.assertEqual({"a", "b"}, set(api.outcomes[0].candidate.artifacts))

    def test_candidate_file_read_failure_does_not_rerun_evaluation(self):
        assignment = _evaluation_assignment(self.evaluation_assets)
        api = _Api(self.evaluation_assets)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); work = root / "work"; work.mkdir()
            missing = work / "missing.json"
            result = EvaluationResult({}, {"report": missing}, ())
            with patch.object(subject, "execute_evaluation_attempt", return_value=result) as domain:
                subject.run_worker_attempt(assignment, api, _evaluation_materializer(root), lambda exc: False)
        domain.assert_called_once()
        self.assertIsInstance(api.outcomes[0], AttemptFailed)

    def test_outcome_retry_reuses_same_object_and_all_three_acknowledgements_end(self):
        assignment = _training_assignment(self.training_assets)
        for final_ack in OutcomeAcknowledgement:
            api = _Api(self.training_assets); api.outcome_ack = final_ack
            seen = []
            first = True
            def report(outcome):
                nonlocal first
                seen.append(outcome)
                if first:
                    first = False
                    raise OSError("lost response")
                return final_ack
            api.report_outcome = report
            with tempfile.TemporaryDirectory() as temporary, patch.object(subject.time, "sleep") as sleep, patch.object(subject, "execute_training_attempt", side_effect=RuntimeError("failed")) as domain:
                subject.run_worker_attempt(assignment, api, _training_materializer(Path(temporary)), lambda exc: isinstance(exc, OSError))
            domain.assert_called_once()
            sleep.assert_called_once_with(10.0)
            self.assertEqual(2, len(seen)); self.assertIs(seen[0], seen[1])

    def test_heartbeat_runs_while_domain_is_blocked_and_thread_is_cleaned_up(self):
        assignment = _training_assignment(self.training_assets)
        api = _Api(self.training_assets)
        entered = threading.Event(); release = threading.Event()
        before = {t.ident for t in threading.enumerate()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def domain(a, files):
                entered.set(); release.wait(1); raise RuntimeError("done")
            with patch.object(subject, "_HEALTHY_HEARTBEAT_INTERVAL_SECONDS", 0.005), patch.object(subject, "execute_training_attempt", side_effect=domain):
                worker = threading.Thread(target=lambda: subject.run_worker_attempt(assignment, api, _training_materializer(root), lambda exc: False))
                worker.start(); self.assertTrue(entered.wait(1))
                deadline = time.monotonic() + 1
                while len(api.heartbeat_requests) < 2 and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertGreaterEqual(len(api.heartbeat_requests), 2)
                release.set(); worker.join(1); self.assertFalse(worker.is_alive())
        leaked = [t for t in threading.enumerate() if t.ident not in before and t.name.startswith("mldb-heartbeat-")]
        self.assertEqual([], leaked)

    def test_public_signature_and_no_queue_repository_dependency(self):
        implementation = Path(subject.__file__)
        skeleton = implementation.parents[2] / "skeleton" / "orchestration" / "worker_loop.py"
        self.assertEqual((), compare_module_signatures(skeleton, implementation))
        tree = ast.parse(implementation.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom): imports.append(node.module or "")
            elif isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
        self.assertFalse(any("queue" in name or "repository" in name for name in imports))
        self.assertNotIn("acquire_work", implementation.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

