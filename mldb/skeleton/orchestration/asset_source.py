"""Minimal Controller-side source for Worker immutable-asset descriptors and bytes.

This port connects assignment projection to the frozen Worker API asset-retrieval
boundary without exposing canonical repository paths as remote locators. It owns only
opaque replay-stable descriptor exposure for one already-selected immutable file and
later recovery of the exact bytes named by that descriptor.

The port defines no cache, registry, URL grammar, storage-key grammar, authentication,
streaming protocol, filesystem hierarchy, repository resolution, asset discovery, or
Worker transport. Canonical immutable object selection and integrity authority are
established by earlier runtime/preflight contracts before this boundary is called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .worker_api import ImmutableAssetDescriptor


class ImmutableAssetSource(Protocol):
    """Consumer-oriented descriptor/byte source for assigned immutable objects.

    ``describe()`` receives one Controller-selected canonical source path together with
    the assignment-local logical key and authoritative integrity identity already fixed
    by the applicable frozen asset contract. It does not resolve an MLDB ID, discover a
    path, recalculate semantic metadata, or decide which object belongs in an
    assignment.

    The returned descriptor must copy ``key``, ``sha256``, and ``bytes`` exactly.
    ``retrieval_id`` is opaque to every caller. For the same ``source_path`` plus the
    same logical key and integrity identity, repeated calls must reproduce the same
    ``retrieval_id`` so a lost acquire response can be projected again without storing
    the prior Worker API response or generating a response-local token. The concrete
    derivation may remain entirely private and must not make an absolute path, URL,
    storage key, authentication token, or response-local UUID part of the public
    contract.

    Replay stability does not require a retrieval identity to be unique per Queue
    attempt. Assignment authorization remains a Controller/Queue/Worker-API concern;
    this port only guarantees stable naming of the already-selected immutable byte
    object under its assignment-local role.

    ``retrieve()`` is the Controller-side byte connection used by a later
    ``WorkerApi.retrieve_asset()`` implementation. Given a descriptor previously
    derivable by this source, it returns only the exact selected bytes whose SHA-256 and
    optional byte count agree with that descriptor. Missing source bytes, descriptor
    tampering, or integrity disagreement are operation failures; this port does not
    invent a recovery source or a second immutable-object authority.
    """

    def describe(
        self,
        source_path: Path,
        *,
        key: str,
        sha256: str,
        bytes: int | None,
    ) -> ImmutableAssetDescriptor:
        """Expose one replay-stable opaque descriptor for selected immutable bytes."""

        ...

    def retrieve(self, asset: ImmutableAssetDescriptor) -> bytes:
        """Return the exact immutable bytes identified by ``asset`` or fail."""

        ...
