"""Generic candidate-artifact publication for MLDB v2 runtimes."""

from __future__ import annotations

from pathlib import Path

from mldb_v2.src.storage.artifact_reference import ArtifactRef, _artifact_ref_for_bytes
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess


def publish_candidate_artifact_bytes(
    *,
    object_bytes: _ObjectByteAccess,
    uri: str,
    data: bytes,
) -> ArtifactRef:
    """Publish exact bytes immutably and return only the base ArtifactRef fields."""
    if type(data) is not bytes:
        raise ValueError("candidate artifact data must be exact bytes")
    ref = _artifact_ref_for_bytes(uri=uri, data=data)
    try:
        return object_bytes.publish(uri=uri, data=data)
    except FileExistsError:
        object_bytes.read_verified(ref)
        return ref


def publish_candidate_artifact_file(
    *,
    object_bytes: _ObjectByteAccess,
    uri: str,
    path: str | Path,
) -> ArtifactRef:
    """Publish one execution-local regular file without leaking its local path."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("candidate artifact path must be a regular file")
    data = source.read_bytes()
    return publish_candidate_artifact_bytes(
        object_bytes=object_bytes,
        uri=uri,
        data=data,
    )
