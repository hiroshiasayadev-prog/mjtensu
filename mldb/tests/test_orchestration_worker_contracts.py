from __future__ import annotations

import ast
from dataclasses import fields
import hashlib
import inspect
from pathlib import Path
from typing import get_args, get_type_hints
import unittest

import mldb.src.orchestration.asset_source as asset_source_module
import mldb.src.orchestration.candidates as candidates_module
import mldb.src.orchestration.worker_api as worker_api_module
from mldb.src.orchestration.asset_source import ImmutableAssetSource
from mldb.src.orchestration.candidates import (
    read_verified_candidate_bytes,
    stage_candidate_upload,
)
from mldb.src.orchestration.worker_api import (
    AcquireRejection,
    AcquireWorkRequest,
    AcquireWorkResponse,
    AssetRejection,
    AssetRetrievalResponse,
    AttemptCancelled,
    AttemptFailed,
    AttemptOutcome,
    CandidateArtifactRef,
    CandidateUploadAcknowledgement,
    CandidateUploadRequest,
    EvaluationAssignment,
    EvaluationSuccessCandidate,
    EvaluationSucceeded,
    HeartbeatAccepted,
    HeartbeatRejection,
    HeartbeatRequest,
    HeartbeatResponse,
    ImmutableAssetDescriptor,
    NoWork,
    OutcomeAcknowledgement,
    RetrievedAsset,
    RetrieveAssetRequest,
    TrainingAssignment,
    TrainingSuccessCandidate,
    TrainingSucceeded,
    WorkerApi,
    WorkerJobKind,
)


class WorkerApiValueContractTests(unittest.TestCase):
    def test_public_dto_fields_are_exact_and_frozen_slotted(self) -> None:
        expected = {
            AcquireWorkRequest: ("worker_id", "acquire_token", "accepts"),
            ImmutableAssetDescriptor: ("key", "retrieval_id", "sha256", "bytes"),
            TrainingAssignment: (
                "attempt_id", "run_id", "lease_token", "lease_until", "kind",
                "task", "corpus", "architecture", "protocol", "seed", "parameters",
                "corpus_artifact", "architecture_implementation",
                "train_protocol_implementation",
            ),
            EvaluationAssignment: (
                "attempt_id", "run_id", "lease_token", "lease_until", "kind",
                "task", "corpus", "model", "training_run", "model_architecture",
                "protocol", "parameters", "corpus_artifact",
                "model_architecture_implementation", "model_weights",
                "evaluation_protocol_implementation",
            ),
            NoWork: (),
            AcquireRejection: ("type", "message"),
            RetrieveAssetRequest: ("asset",),
            RetrievedAsset: ("content",),
            AssetRejection: ("type", "message"),
            HeartbeatRequest: ("attempt_id", "lease_token"),
            HeartbeatAccepted: ("lease_until", "cancel_requested"),
            HeartbeatRejection: ("type", "message"),
            CandidateUploadRequest: (
                "attempt_id", "lease_token", "key", "content_identity", "content",
            ),
            CandidateArtifactRef: ("key", "content_identity", "bytes"),
            TrainingSuccessCandidate: ("weights",),
            EvaluationSuccessCandidate: ("metrics", "artifacts", "unavailable_outputs"),
            TrainingSucceeded: ("attempt_id", "lease_token", "candidate"),
            EvaluationSucceeded: ("attempt_id", "lease_token", "candidate"),
            AttemptFailed: ("attempt_id", "lease_token", "type", "message"),
            AttemptCancelled: ("attempt_id", "lease_token"),
        }

        for dto, names in expected.items():
            with self.subTest(dto=dto.__name__):
                self.assertEqual(names, tuple(field.name for field in fields(dto)))
                self.assertTrue(dto.__dataclass_params__.frozen)
                self.assertTrue(hasattr(dto, "__slots__"))

    def test_public_enum_and_kind_values_are_exact(self) -> None:
        self.assertEqual(("training", "evaluation"), get_args(WorkerJobKind))
        self.assertEqual(
            ["accepted", "already_present", "rejected"],
            [value.value for value in CandidateUploadAcknowledgement],
        )
        self.assertEqual(
            ["accepted", "already_finalized", "rejected"],
            [value.value for value in OutcomeAcknowledgement],
        )

    def test_response_unions_represent_only_definitive_responses(self) -> None:
        self.assertEqual(
            {TrainingAssignment, EvaluationAssignment, NoWork, AcquireRejection},
            set(get_args(AcquireWorkResponse)),
        )
        self.assertEqual(
            {RetrievedAsset, AssetRejection},
            set(get_args(AssetRetrievalResponse)),
        )
        self.assertEqual(
            {HeartbeatAccepted, HeartbeatRejection},
            set(get_args(HeartbeatResponse)),
        )
        self.assertEqual(
            {TrainingSucceeded, EvaluationSucceeded, AttemptFailed, AttemptCancelled},
            set(get_args(AttemptOutcome)),
        )
        for alias in (AcquireWorkResponse, AssetRetrievalResponse, HeartbeatResponse):
            self.assertFalse(any(_is_exception_type(item) for item in get_args(alias)))

    def test_acquire_request_rejects_empty_worker_and_token(self) -> None:
        valid = frozenset({"training"})
        with self.assertRaises(ValueError):
            AcquireWorkRequest("", "token", valid)
        with self.assertRaises(ValueError):
            AcquireWorkRequest("worker", "", valid)

    def test_acquire_request_rejects_empty_or_invalid_accepts(self) -> None:
        with self.assertRaises(ValueError):
            AcquireWorkRequest("worker", "token", frozenset())
        with self.assertRaises(ValueError):
            AcquireWorkRequest("worker", "token", frozenset({"training", "other"}))
        with self.assertRaises(ValueError):
            AcquireWorkRequest("worker", "token", {"training"})  # type: ignore[arg-type]

        request = AcquireWorkRequest(
            "worker",
            "token",
            frozenset({"training", "evaluation"}),
        )
        self.assertEqual(frozenset({"training", "evaluation"}), request.accepts)

    def test_descriptor_validates_identity_and_integrity_surface(self) -> None:
        digest = "a" * 64
        descriptor = ImmutableAssetDescriptor("corpus", "opaque-1", digest, 0)
        self.assertEqual(("corpus", "opaque-1", digest, 0), (
            descriptor.key,
            descriptor.retrieval_id,
            descriptor.sha256,
            descriptor.bytes,
        ))
        self.assertEqual(
            ("key", "retrieval_id", "sha256", "bytes"),
            tuple(field.name for field in fields(ImmutableAssetDescriptor)),
        )
        self.assertTrue(
            {"path", "url", "auth", "token"}.isdisjoint(
                {field.name.lower() for field in fields(ImmutableAssetDescriptor)}
            )
        )

        invalid = (
            {"key": ""},
            {"retrieval_id": ""},
            {"sha256": "a" * 63},
            {"sha256": "g" * 64},
            {"bytes": -1},
            {"bytes": True},
        )
        base = {"key": "asset", "retrieval_id": "opaque", "sha256": digest, "bytes": None}
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                ImmutableAssetDescriptor(**(base | override))

    def test_assignment_surface_carries_values_not_remote_paths(self) -> None:
        for assignment in (TrainingAssignment, EvaluationAssignment):
            annotations = assignment.__annotations__
            self.assertFalse(
                any("Path" in str(annotation) for annotation in annotations.values()),
                annotations,
            )
        descriptor_hints = get_type_hints(ImmutableAssetDescriptor)
        self.assertNotIn(Path, descriptor_hints.values())

    def test_worker_api_has_no_queue_import_dependency(self) -> None:
        imports = _imported_modules(Path(worker_api_module.__file__))
        self.assertFalse(
            any(
                module.endswith(".queue")
                or module.endswith(".queue_ports")
                or module.endswith("orchestration.queue")
                or module.endswith("orchestration.queue_ports")
                for module in imports
            ),
            imports,
        )
        methods = {
            name
            for name, value in WorkerApi.__dict__.items()
            if inspect.isfunction(value) and not name.startswith("__")
        }
        self.assertEqual(
            {"acquire_work", "retrieve_asset", "heartbeat", "upload_candidate", "report_outcome"},
            methods,
        )


class CandidateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = _MemoryCandidateStore()
        self.content = b"candidate-payload"
        self.digest = hashlib.sha256(self.content).hexdigest()

    def test_first_candidate_is_accepted_and_identical_replay_is_already_present(self) -> None:
        first = stage_candidate_upload(7, "weights", self.digest, self.content, self.store)
        replay = stage_candidate_upload(7, "weights", self.digest, self.content, self.store)

        self.assertIs(first, CandidateUploadAcknowledgement.ACCEPTED)
        self.assertIs(replay, CandidateUploadAcknowledgement.ALREADY_PRESENT)
        self.assertEqual(2, self.store.put_calls)

    def test_conflicting_replay_is_rejected_without_overwrite(self) -> None:
        self.assertIs(
            stage_candidate_upload(7, "weights", self.digest, self.content, self.store),
            CandidateUploadAcknowledgement.ACCEPTED,
        )
        conflict = b"different-candidate"
        conflict_digest = hashlib.sha256(conflict).hexdigest()

        acknowledgement = stage_candidate_upload(
            7,
            "weights",
            conflict_digest,
            conflict,
            self.store,
        )

        self.assertIs(acknowledgement, CandidateUploadAcknowledgement.REJECTED)
        self.assertEqual((self.digest, self.content), self.store.entries[(7, "weights")])

    def test_upload_sha_mismatch_is_rejected_without_store_put(self) -> None:
        acknowledgement = stage_candidate_upload(
            7,
            "weights",
            hashlib.sha256(b"other").hexdigest(),
            self.content,
            self.store,
        )

        self.assertIs(acknowledgement, CandidateUploadAcknowledgement.REJECTED)
        self.assertEqual(0, self.store.put_calls)
        self.assertEqual({}, self.store.entries)

    def test_verified_read_success(self) -> None:
        stage_candidate_upload(7, "weights", self.digest, self.content, self.store)
        ref = CandidateArtifactRef("weights", self.digest, len(self.content))

        content = read_verified_candidate_bytes(7, ref, self.store)

        self.assertEqual(self.content, content)
        self.assertEqual(1, self.store.read_calls)

    def test_verified_read_missing_propagates_store_failure(self) -> None:
        ref = CandidateArtifactRef("weights", self.digest, len(self.content))
        with self.assertRaises(KeyError):
            read_verified_candidate_bytes(7, ref, self.store)

    def test_verified_read_rejects_corrupt_non_bytes_store_payload(self) -> None:
        store = _ReadOverrideStore(bytearray(self.content))
        ref = CandidateArtifactRef("weights", self.digest, len(self.content))
        with self.assertRaisesRegex(ValueError, "non-bytes"):
            read_verified_candidate_bytes(7, ref, store)  # type: ignore[arg-type]
        self.assertEqual(1, store.read_calls)

    def test_verified_read_revalidates_identity_after_store_read(self) -> None:
        store = _ReadOverrideStore(self.content)
        ref = CandidateArtifactRef("weights", "A" * 64, len(self.content))
        with self.assertRaisesRegex(ValueError, "content identity"):
            read_verified_candidate_bytes(7, ref, store)
        self.assertEqual(1, store.read_calls)

    def test_verified_read_rejects_hash_mismatch(self) -> None:
        store = _ReadOverrideStore(b"tampered")
        ref = CandidateArtifactRef("weights", self.digest, len(b"tampered"))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            read_verified_candidate_bytes(7, ref, store)
        self.assertEqual(1, store.read_calls)

    def test_verified_read_rejects_byte_count_mismatch(self) -> None:
        store = _ReadOverrideStore(self.content)
        ref = CandidateArtifactRef("weights", self.digest, len(self.content) + 1)
        with self.assertRaisesRegex(ValueError, "byte count"):
            read_verified_candidate_bytes(7, ref, store)
        self.assertEqual(1, store.read_calls)

    def test_physical_store_exceptions_propagate(self) -> None:
        failing = _FailingStore()
        with self.assertRaises(OSError):
            stage_candidate_upload(7, "weights", self.digest, self.content, failing)

        ref = CandidateArtifactRef("weights", self.digest, len(self.content))
        with self.assertRaises(OSError):
            read_verified_candidate_bytes(7, ref, failing)

    def test_candidate_helpers_have_no_canonical_state_write_dependency(self) -> None:
        imports = _imported_modules(Path(candidates_module.__file__))
        forbidden_suffixes = (
            ".queue",
            ".queue_ports",
            ".run_persistence",
            ".result_acceptance",
            ".model.persistence",
            ".repository.layout",
            ".repository.ports",
        )
        self.assertFalse(
            any(module.endswith(forbidden_suffixes) for module in imports),
            imports,
        )
        self.assertEqual(
            ("attempt_id", "key", "content_identity", "content", "store"),
            tuple(inspect.signature(stage_candidate_upload).parameters),
        )
        self.assertEqual(
            ("attempt_id", "ref", "store"),
            tuple(inspect.signature(read_verified_candidate_bytes).parameters),
        )


class AssetSourceContractTests(unittest.TestCase):
    def test_asset_source_is_protocol_only_with_frozen_operations(self) -> None:
        self.assertTrue(getattr(ImmutableAssetSource, "_is_protocol", False))
        methods = {
            name
            for name, value in ImmutableAssetSource.__dict__.items()
            if inspect.isfunction(value) and not name.startswith("__")
        }
        self.assertEqual({"describe", "retrieve"}, methods)

        public_classes = _public_classes(Path(asset_source_module.__file__))
        self.assertEqual({"ImmutableAssetSource"}, public_classes)
        describe_hints = get_type_hints(ImmutableAssetSource.describe)
        self.assertIs(Path, describe_hints["source_path"])
        self.assertIs(ImmutableAssetDescriptor, describe_hints["return"])


class _MemoryCandidateStore:
    def __init__(self) -> None:
        self.entries: dict[tuple[int, str], tuple[str, bytes]] = {}
        self.put_calls = 0
        self.read_calls = 0

    def put(self, attempt_id: int, key: str, content_identity: str, content: bytes) -> CandidateUploadAcknowledgement:
        self.put_calls += 1
        storage_key = (attempt_id, key)
        existing = self.entries.get(storage_key)
        if existing is None:
            self.entries[storage_key] = (content_identity, content)
            return CandidateUploadAcknowledgement.ACCEPTED
        if existing == (content_identity, content):
            return CandidateUploadAcknowledgement.ALREADY_PRESENT
        return CandidateUploadAcknowledgement.REJECTED

    def read(self, attempt_id: int, key: str, content_identity: str) -> bytes:
        self.read_calls += 1
        existing = self.entries.get((attempt_id, key))
        if existing is None or existing[0] != content_identity:
            raise KeyError((attempt_id, key, content_identity))
        return existing[1]


class _ReadOverrideStore:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.read_calls = 0

    def put(self, attempt_id: int, key: str, content_identity: str, content: bytes) -> CandidateUploadAcknowledgement:
        raise AssertionError("put is not used by read verification tests")

    def read(self, attempt_id: int, key: str, content_identity: str) -> bytes:
        self.read_calls += 1
        return self.payload  # type: ignore[return-value]


class _FailingStore:
    def put(self, attempt_id: int, key: str, content_identity: str, content: bytes) -> CandidateUploadAcknowledgement:
        raise OSError("physical put failed")

    def read(self, attempt_id: int, key: str, content_identity: str) -> bytes:
        raise OSError("physical read failed")


def _is_exception_type(value: object) -> bool:
    return isinstance(value, type) and issubclass(value, BaseException)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            modules.add(prefix + (node.module or ""))
    return modules


def _public_classes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    }


if __name__ == "__main__":
    unittest.main()
