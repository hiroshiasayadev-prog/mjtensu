"""Controller-side Worker immutable-asset and candidate-upload boundaries.

This module connects the frozen Worker data-plane request/response values to the
existing immutable-asset source, Queue lease authority, and attempt-local candidate
staging boundary. It owns only transport-independent handling for immutable asset
retrieval and candidate upload.

The module deliberately introduces no Worker data service, blob/artifact manager,
transfer session, streaming or URL/authentication abstraction, cleanup policy, Worker
loop, heartbeat, outcome handling, canonical artifact commit, Run terminalization,
Model creation, Study mutation, or Queue lifecycle transition.
"""

from __future__ import annotations

from .asset_source import ImmutableAssetSource
from .candidates import CandidateStore, stage_candidate_upload
from .queue_ports import QueuePort
from .worker_api import (
    AssetRejection,
    AssetRetrievalResponse,
    CandidateUploadAcknowledgement,
    CandidateUploadRequest,
    RetrieveAssetRequest,
    RetrievedAsset,
)


def handle_asset_retrieval(
    request: RetrieveAssetRequest,
    assets: ImmutableAssetSource,
) -> AssetRetrievalResponse:
    """Return the exact immutable bytes named by a frozen Worker asset request.

    ``request`` contains exactly the :class:`ImmutableAssetDescriptor` previously
    supplied by Controller. This operation adds no ``attempt_id`` or ``lease_token``;
    retrieval authorization/authentication outside the descriptor contract remains a
    transport concern under the frozen Worker API.

    The successful path is exactly::

        content = assets.retrieve(request.asset)
        RetrievedAsset(content=content)

    :class:`ImmutableAssetSource` already owns recovery of the selected bytes and must
    return only bytes whose required SHA-256 and optional byte count agree with the
    supplied descriptor. This handler must therefore not recalculate or duplicate the
    descriptor hash/size verification owned by that port before wrapping the successful
    bytes in :class:`RetrievedAsset`.

    A malformed, tampered, or unsupported descriptor may be represented by
    :class:`AssetRejection` only when the Controller can establish that rejection
    definitively. The frozen :class:`ImmutableAssetSource` port intentionally exposes no
    public exception taxonomy, so this boundary must not catch an undifferentiated
    retrieval failure and automatically convert it into ``AssetRejection``. Missing
    storage, transient filesystem/service failure, or any other infrastructure failure
    for which no definitive Worker response can safely be established remains an
    operation failure so the transport layer can apply the Worker API communication
    retry semantics.

    This operation performs no canonical or Queue mutation.
    """

    ...


def handle_candidate_upload(
    request: CandidateUploadRequest,
    as_of: str,
    queue: QueuePort,
    candidates: CandidateStore,
) -> CandidateUploadAcknowledgement:
    """Authorize and stage one Worker candidate without canonical result acceptance.

    ``as_of`` is passed unchanged to the frozen Queue lease-authority read. Before any
    candidate bytes are stored, the handler requires current authorization exactly
    through::

        queue.authorized_open_attempt(
            request.attempt_id,
            request.lease_token,
            as_of=as_of,
        )

    ``None`` is a definitive ``CandidateUploadAcknowledgement.REJECTED``. No historical
    attempt lookup or closed-attempt replay is permitted for candidate upload: a closed,
    expired, invalidated, stale, or superseded lease never regains upload authority.
    Candidate replay remains available only while this exact attempt/lease is currently
    authorized.

    Request-local candidate shape additionally requires ``request.key`` and
    ``request.content_identity`` to be non-empty. Invalid shape is definitively
    ``REJECTED`` and must not call the candidate store. This boundary invents no further
    key grammar, content-identity vocabulary, normalization, or generic request
    validator.

    Once request shape and current lease authorization are established, persistence is
    delegated exactly to the frozen staging helper::

        stage_candidate_upload(
            request.attempt_id,
            request.key,
            request.content_identity,
            request.content,
            candidates,
        )

    The handler must not reimplement SHA-256 validation or immutable
    ``(attempt_id, key)`` establish/replay semantics. The staging boundary therefore
    determines the successful replay results while the lease remains authorized:

    - first valid establish -> ``ACCEPTED``;
    - same attempt/key and exact bytes -> ``ALREADY_PRESENT``;
    - digest mismatch or conflicting existing candidate -> ``REJECTED`` without
      overwrite.

    Once the attempt is no longer currently authorized, even a byte-identical upload
    replay is ``REJECTED``. Later outcome replay may still read already-durable
    CandidateStore bytes; it does not reopen candidate-upload authorization.

    CandidateStore or other infrastructure failure for which no definitive
    acknowledgement is established remains an operation failure rather than being
    silently converted to ``REJECTED``.

    Successful candidate staging is operational handoff state only. This operation does
    not commit canonical artifacts, terminalize a Training/Evaluation Run, create a
    Model, mutate a Study, satisfy/fail/cancel a Queue job, close an attempt, extend a
    lease, or otherwise change Queue lifecycle state.
    """

    ...
