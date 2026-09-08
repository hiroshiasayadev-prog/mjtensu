from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import unittest

import mldb.src.orchestration.worker_data as worker_data_module
from mldb.src.orchestration.worker_api import (
    CandidateUploadAcknowledgement,
    CandidateUploadRequest,
    ImmutableAssetDescriptor,
    RetrievedAsset,
    RetrieveAssetRequest,
)
from mldb.src.orchestration.worker_data import (
    handle_asset_retrieval,
    handle_candidate_upload,
)


AS_OF = "2026-09-08T07:54:00.000000Z"


class WorkerAssetRetrievalTests(unittest.TestCase):
    def test_success_retrieves_once_and_wraps_exact_bytes(self) -> None:
        descriptor = _descriptor()
        assets = _AssetSource(b"asset-bytes")

        result = handle_asset_retrieval(RetrieveAssetRequest(descriptor), assets)

        self.assertEqual(RetrievedAsset(b"asset-bytes"), result)
        self.assertEqual([descriptor], assets.retrieve_calls)

    def test_infrastructure_failure_propagates(self) -> None:
        assets = _AssetSource(error=OSError("asset backend unavailable"))

        with self.assertRaisesRegex(OSError, "asset backend unavailable"):
            handle_asset_retrieval(RetrieveAssetRequest(_descriptor()), assets)

        self.assertEqual(1, len(assets.retrieve_calls))


class WorkerCandidateUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.content = b"candidate-bytes"
        self.digest = hashlib.sha256(self.content).hexdigest()

    def test_malformed_shape_rejects_before_authorization_or_store(self) -> None:
        for key, identity in (("", self.digest), ("weights", ""), (None, self.digest)):
            with self.subTest(key=key, identity=identity):
                queue = _QueueAuthority(authorized=True)
                store = _MemoryCandidateStore()
                request = CandidateUploadRequest(
                    7, "lease-7", key, identity, self.content  # type: ignore[arg-type]
                )

                result = handle_candidate_upload(request, AS_OF, queue, store)

                self.assertIs(result, CandidateUploadAcknowledgement.REJECTED)
                self.assertEqual([], queue.authorization_calls)
                self.assertEqual([], store.put_calls)

    def test_authorized_upload_forwards_exact_authority_and_staging_arguments(self) -> None:
        queue = _QueueAuthority(authorized=True)
        store = _MemoryCandidateStore()
        request = CandidateUploadRequest(7, "lease-7", "weights", self.digest, self.content)

        result = handle_candidate_upload(request, AS_OF, queue, store)

        self.assertIs(result, CandidateUploadAcknowledgement.ACCEPTED)
        self.assertEqual([(7, "lease-7", AS_OF)], queue.authorization_calls)
        self.assertEqual([(7, "weights", self.digest, self.content)], store.put_calls)

    def test_unauthorized_upload_rejects_without_touching_store(self) -> None:
        queue = _QueueAuthority(authorized=False)
        store = _MemoryCandidateStore()
        request = CandidateUploadRequest(7, "stale", "weights", self.digest, self.content)

        result = handle_candidate_upload(request, AS_OF, queue, store)

        self.assertIs(result, CandidateUploadAcknowledgement.REJECTED)
        self.assertEqual([(7, "stale", AS_OF)], queue.authorization_calls)
        self.assertEqual([], store.put_calls)

    def test_nonempty_invalid_digest_is_authorized_then_rejected_by_staging(self) -> None:
        queue = _QueueAuthority(authorized=True)
        store = _MemoryCandidateStore()
        request = CandidateUploadRequest(7, "lease-7", "weights", "not-a-digest", self.content)

        result = handle_candidate_upload(request, AS_OF, queue, store)

        self.assertIs(result, CandidateUploadAcknowledgement.REJECTED)
        self.assertEqual([(7, "lease-7", AS_OF)], queue.authorization_calls)
        self.assertEqual([], store.put_calls)

    def test_first_replay_and_conflict_preserve_store_semantics(self) -> None:
        queue = _QueueAuthority(authorized=True)
        store = _MemoryCandidateStore()
        request = CandidateUploadRequest(7, "lease-7", "weights", self.digest, self.content)

        first = handle_candidate_upload(request, AS_OF, queue, store)
        replay = handle_candidate_upload(request, AS_OF, queue, store)
        conflict_content = b"conflicting-bytes"
        conflict_digest = hashlib.sha256(conflict_content).hexdigest()
        conflict = handle_candidate_upload(
            CandidateUploadRequest(7, "lease-7", "weights", conflict_digest, conflict_content),
            AS_OF,
            queue,
            store,
        )

        self.assertIs(first, CandidateUploadAcknowledgement.ACCEPTED)
        self.assertIs(replay, CandidateUploadAcknowledgement.ALREADY_PRESENT)
        self.assertIs(conflict, CandidateUploadAcknowledgement.REJECTED)
        self.assertEqual((self.digest, self.content), store.entries[(7, "weights")])
        self.assertEqual(3, len(queue.authorization_calls))

    def test_identical_replay_after_lease_loss_is_rejected_before_store(self) -> None:
        queue = _QueueAuthority(authorized=True)
        store = _MemoryCandidateStore()
        request = CandidateUploadRequest(7, "lease-7", "weights", self.digest, self.content)

        self.assertIs(
            handle_candidate_upload(request, AS_OF, queue, store),
            CandidateUploadAcknowledgement.ACCEPTED,
        )
        queue.authorized = False
        replay = handle_candidate_upload(request, AS_OF, queue, store)

        self.assertIs(replay, CandidateUploadAcknowledgement.REJECTED)
        self.assertEqual(1, len(store.put_calls))
        self.assertEqual(2, len(queue.authorization_calls))

    def test_store_infrastructure_failure_propagates(self) -> None:
        queue = _QueueAuthority(authorized=True)
        store = _FailingCandidateStore()
        request = CandidateUploadRequest(7, "lease-7", "weights", self.digest, self.content)

        with self.assertRaisesRegex(OSError, "candidate store unavailable"):
            handle_candidate_upload(request, AS_OF, queue, store)

        self.assertEqual([(7, "lease-7", AS_OF)], queue.authorization_calls)

    def test_handler_surface_has_no_queue_transition_or_canonical_dependency(self) -> None:
        source = Path(worker_data_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            ("." * node.level) + (node.module or "")
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        }
        forbidden = (
            "run_persistence",
            "run_finalization",
            "result_acceptance",
            "model.persistence",
            "repository",
            "_coordination",
        )
        self.assertFalse(any(any(part in module for part in forbidden) for module in imported))
        self.assertNotIn("heartbeat_attempt", source)
        self.assertNotIn("close_attempt", source)
        self.assertNotIn("activate_attempt", source)

    def test_public_handler_signatures_are_frozen(self) -> None:
        self.assertEqual(
            ("request", "assets"),
            tuple(inspect.signature(handle_asset_retrieval).parameters),
        )
        self.assertEqual(
            ("request", "as_of", "queue", "candidates"),
            tuple(inspect.signature(handle_candidate_upload).parameters),
        )


class _AssetSource:
    def __init__(self, content: bytes = b"", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.retrieve_calls: list[ImmutableAssetDescriptor] = []

    def retrieve(self, asset: ImmutableAssetDescriptor) -> bytes:
        self.retrieve_calls.append(asset)
        if self.error is not None:
            raise self.error
        return self.content


class _QueueAuthority:
    def __init__(self, *, authorized: bool) -> None:
        self.authorized = authorized
        self.authorization_calls: list[tuple[int, str, str]] = []

    def authorized_open_attempt(self, attempt_id: int, lease_token: str, *, as_of: str) -> object | None:
        self.authorization_calls.append((attempt_id, lease_token, as_of))
        return object() if self.authorized else None


class _MemoryCandidateStore:
    def __init__(self) -> None:
        self.entries: dict[tuple[int, str], tuple[str, bytes]] = {}
        self.put_calls: list[tuple[int, str, str, bytes]] = []

    def put(
        self,
        attempt_id: int,
        key: str,
        content_identity: str,
        content: bytes,
    ) -> CandidateUploadAcknowledgement:
        self.put_calls.append((attempt_id, key, content_identity, content))
        storage_key = (attempt_id, key)
        existing = self.entries.get(storage_key)
        if existing is None:
            self.entries[storage_key] = (content_identity, content)
            return CandidateUploadAcknowledgement.ACCEPTED
        if existing == (content_identity, content):
            return CandidateUploadAcknowledgement.ALREADY_PRESENT
        return CandidateUploadAcknowledgement.REJECTED

    def read(self, attempt_id: int, key: str, content_identity: str) -> bytes:
        raise AssertionError("read is not used by candidate upload")


class _FailingCandidateStore:
    def put(
        self,
        attempt_id: int,
        key: str,
        content_identity: str,
        content: bytes,
    ) -> CandidateUploadAcknowledgement:
        raise OSError("candidate store unavailable")

    def read(self, attempt_id: int, key: str, content_identity: str) -> bytes:
        raise AssertionError("read is not used by candidate upload")


def _descriptor() -> ImmutableAssetDescriptor:
    return ImmutableAssetDescriptor(
        key="corpus",
        retrieval_id="opaque-corpus",
        sha256="a" * 64,
        bytes=None,
    )


if __name__ == "__main__":
    unittest.main()
