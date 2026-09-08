"""Controller-side Worker immutable-asset and candidate-upload boundaries."""

from __future__ import annotations

from .asset_source import ImmutableAssetSource
from .candidates import CandidateStore, stage_candidate_upload
from .queue_ports import QueuePort
from .worker_api import (
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
    """Retrieve one immutable asset without duplicating source integrity checks."""

    content = assets.retrieve(request.asset)
    return RetrievedAsset(content=content)


def handle_candidate_upload(
    request: CandidateUploadRequest,
    as_of: str,
    queue: QueuePort,
    candidates: CandidateStore,
) -> CandidateUploadAcknowledgement:
    """Authorize one upload, then delegate candidate integrity/replay semantics."""

    if not isinstance(request.key, str) or not request.key:
        return CandidateUploadAcknowledgement.REJECTED
    if not isinstance(request.content_identity, str) or not request.content_identity:
        return CandidateUploadAcknowledgement.REJECTED

    attempt = queue.authorized_open_attempt(
        request.attempt_id,
        request.lease_token,
        as_of=as_of,
    )
    if attempt is None:
        return CandidateUploadAcknowledgement.REJECTED

    return stage_candidate_upload(
        request.attempt_id,
        request.key,
        request.content_identity,
        request.content,
        candidates,
    )
