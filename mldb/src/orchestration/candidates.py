"""Controller-local staging boundary for Worker-uploaded result candidates.

Worker candidate upload is operational handoff state, not canonical MLDB artifact
persistence. This module fixes only the small attempt-local byte boundary needed between
the frozen Worker API and later Controller result acceptance: immutable put/replay
semantics for one ``attempt_id``/``key`` and exact verified byte retrieval from a
:class:`~mldb.skeleton.orchestration.worker_api.CandidateArtifactRef`.

No candidate directory grammar, object-store API, generic artifact repository, Queue
lease policy, Worker acknowledgement handler, canonical Run artifact placement, retry
policy, or cleanup lifecycle is defined here. A concrete store may use files, SQLite, or
another Controller-local mechanism provided the semantics below remain observable.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

from .worker_api import CandidateArtifactRef, CandidateUploadAcknowledgement


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CandidateStore(Protocol):
    """Minimal persistence port for immutable attempt-local candidate bytes.

    The tuple ``(attempt_id, key)`` is the storage identity. ``content_identity`` is the
    SHA-256 identity supplied for the exact bytes. ``put`` must be atomic with respect to
    that identity: an absent key may be established once, a replay whose already-stored
    identity and exact bytes agree with the supplied candidate returns
    ``ALREADY_PRESENT``, and any conflicting existing candidate returns ``REJECTED``
    without overwriting either metadata or bytes.

    A successful ``put`` is durable operational staging: the candidate must remain
    readable across ordinary Controller restart for result-delivery replay until a later
    owner makes it safe to discard. This module deliberately defines no cleanup API or
    retention schedule.

    ``read`` returns the exact bytes currently stored for the attempt/key only when the
    stored upload identity also equals the supplied ``content_identity``; missing or
    conflicting identity fails the operation. The port deliberately does not accept a
    canonical MLDB path and does not expose candidate placement, temporary-file naming,
    cleanup, enumeration, mutation, or generic blob operations.

    Lease validity and attempt authorization are caller responsibilities. This store is
    invoked only after the later Worker API handler has established that the upload or
    outcome belongs to the currently relevant attempt.
    """

    def put(
        self,
        attempt_id: int,
        key: str,
        content_identity: str,
        content: bytes,
    ) -> CandidateUploadAcknowledgement:
        """Establish one immutable candidate or report same-identity replay/conflict."""

        ...

    def read(
        self,
        attempt_id: int,
        key: str,
        content_identity: str,
    ) -> bytes:
        """Return exact bytes only for the matching stored upload identity."""

        ...


def stage_candidate_upload(
    attempt_id: int,
    key: str,
    content_identity: str,
    content: bytes,
    store: CandidateStore,
) -> CandidateUploadAcknowledgement:
    """Validate and persist one Worker-uploaded candidate without canonical commit.

    ``content_identity`` must equal the SHA-256 identity of the exact supplied
    ``content``. A digest mismatch is definitively ``REJECTED`` and the bytes
    must not be stored. After local integrity succeeds, the operation delegates the
    immutable ``attempt_id``/``key`` establish-or-replay decision to ``store.put``.

    A first successful establish returns ``ACCEPTED``. A same-key replay is
    ``ALREADY_PRESENT`` only when the store already contains the same content identity
    and exact bytes; conflicting content is ``REJECTED`` and never overwrites the
    existing candidate.
    Physical persistence failure is an operation failure rather than being disguised as
    a definitive Worker-domain rejection.

    This function deliberately receives no lease token. Lease/attempt authorization is
    owned by the later Worker API handler before it calls this staging boundary. Success
    here creates no canonical Training/Evaluation artifact, mutates no Run, creates no
    Model, and chooses no Queue state.
    """

    if not isinstance(key, str) or not key:
        return CandidateUploadAcknowledgement.REJECTED
    if (
        not isinstance(content_identity, str)
        or _SHA256_RE.fullmatch(content_identity) is None
        or hashlib.sha256(content).hexdigest() != content_identity
    ):
        return CandidateUploadAcknowledgement.REJECTED

    return store.put(attempt_id, key, content_identity, content)


def read_verified_candidate_bytes(
    attempt_id: int,
    ref: CandidateArtifactRef,
    store: CandidateStore,
) -> bytes:
    """Retrieve and independently verify the exact bytes named by one candidate ref.

    The operation reads only
    ``store.read(attempt_id, ref.key, ref.content_identity)`` and never trusts the
    reference metadata as proof of stored content. Successful return requires all of the
    following to hold:

    - the attempt/key exists;
    - the stored upload identity equals ``ref.content_identity``;
    - SHA-256 of the exact retrieved bytes equals ``ref.content_identity`` exactly;
    - ``len(content) == ref.bytes``.

    Missing bytes, corrupted store state, stale/conflicting metadata, hash mismatch, or
    byte-count mismatch is an operation failure. The returned ``bytes`` value is the
    exact verified payload later Training/Evaluation acceptance may stage or commit.

    This helper does not decide whether the attempt is still authorized, whether the
    candidate is semantically valid, or whether any canonical Run is running/terminal.
    Those responsibilities remain with the later outcome and result-acceptance layers.
    """

    content = store.read(attempt_id, ref.key, ref.content_identity)

    if not isinstance(content, bytes):
        raise ValueError("Candidate store returned a non-bytes payload.")
    if not isinstance(ref.key, str) or not ref.key:
        raise ValueError("Candidate key must be a non-empty string.")
    if (
        not isinstance(ref.content_identity, str)
        or _SHA256_RE.fullmatch(ref.content_identity) is None
    ):
        raise ValueError("Candidate content identity must be a lowercase SHA-256 digest.")
    if hashlib.sha256(content).hexdigest() != ref.content_identity:
        raise ValueError("Candidate bytes do not match the referenced SHA-256 identity.")
    if (
        not isinstance(ref.bytes, int)
        or isinstance(ref.bytes, bool)
        or ref.bytes < 0
        or len(content) != ref.bytes
    ):
        raise ValueError("Candidate byte count does not match the reference.")

    return content
