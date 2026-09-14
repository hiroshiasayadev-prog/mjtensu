"""Internal immutable object-byte boundary for later MLDB v2 runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mldb_v2.src.storage.artifact_reference import (
    ArtifactRef,
    _artifact_ref_for_bytes,
    _validate_artifact_ref,
    _verify_artifact_bytes,
)


class _ObjectByteTransport(Protocol):
    """Configured runtime transport; canonical values never contain its credentials/config."""

    def read_bytes(self, uri: str) -> bytes: ...

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        """Atomically create immutable bytes, failing rather than replacing different bytes."""
        ...


class _ObjectByteAccess:
    def __init__(self, transport: _ObjectByteTransport) -> None:
        self._transport = transport

    def read_verified(self, ref: ArtifactRef) -> bytes:
        validated = _validate_artifact_ref(ref)
        data = self._transport.read_bytes(validated["uri"])
        if type(data) is not bytes:
            raise ValueError("object transport must return exact bytes")
        return _verify_artifact_bytes(ref=validated, data=data)

    def publish(self, *, uri: str, data: bytes) -> ArtifactRef:
        ref = _artifact_ref_for_bytes(uri=uri, data=data)
        self._transport.publish_bytes_immutable(uri, data)
        return ref

    def materialize_verified(self, ref: ArtifactRef, destination: str | Path) -> Path:
        data = self.read_verified(ref)
        target = Path(destination)
        created = False
        try:
            with target.open("xb") as stream:
                created = True
                stream.write(data)
        except Exception:
            if created:
                try:
                    target.unlink()
                except OSError:
                    pass
            raise
        return target
