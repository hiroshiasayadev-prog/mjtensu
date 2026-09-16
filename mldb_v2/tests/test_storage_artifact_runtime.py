from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mldb_v2.src.storage.artifact_runtime import (
    publish_candidate_artifact_bytes,
    publish_candidate_artifact_file,
)
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess


class CreateOnlyTransport:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def read_bytes(self, uri: str) -> bytes:
        return self.objects[uri]

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        if uri in self.objects:
            raise FileExistsError(uri)
        self.objects[uri] = data


def _expected_ref(uri: str, data: bytes) -> dict[str, object]:
    return {
        "uri": uri,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_publish_bytes_returns_exact_base_artifact_ref() -> None:
    uri = "s3://bucket/results/predictions.jsonl"
    data = b"exact artifact\n"
    transport = CreateOnlyTransport()
    ref = publish_candidate_artifact_bytes(
        object_bytes=_ObjectByteAccess(transport), uri=uri, data=data
    )
    assert ref == _expected_ref(uri, data)
    assert set(ref) == {"uri", "bytes", "sha256"}
    assert "format" not in ref and "schema" not in ref


def test_same_bytes_at_same_uri_are_idempotent() -> None:
    uri = "s3://bucket/results/object.bin"
    data = b"same"
    transport = CreateOnlyTransport()
    access = _ObjectByteAccess(transport)
    first = publish_candidate_artifact_bytes(object_bytes=access, uri=uri, data=data)
    second = publish_candidate_artifact_bytes(object_bytes=access, uri=uri, data=data)
    assert first == second == _expected_ref(uri, data)
    assert transport.objects[uri] == data


def test_different_bytes_at_existing_uri_fail_without_overwrite() -> None:
    uri = "s3://bucket/results/object.bin"
    transport = CreateOnlyTransport()
    transport.objects[uri] = b"first"
    with pytest.raises(ValueError):
        publish_candidate_artifact_bytes(
            object_bytes=_ObjectByteAccess(transport), uri=uri, data=b"second"
        )
    assert transport.objects[uri] == b"first"


def test_file_publication_requires_regular_file_and_does_not_leak_local_path(tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"payload")
    uri = "s3://bucket/results/artifact.bin"
    ref = publish_candidate_artifact_file(
        object_bytes=_ObjectByteAccess(CreateOnlyTransport()),
        uri=uri,
        path=source,
    )
    assert ref == _expected_ref(uri, b"payload")
    assert str(tmp_path) not in repr(ref)

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        publish_candidate_artifact_file(
            object_bytes=_ObjectByteAccess(CreateOnlyTransport()),
            uri="s3://bucket/results/directory",
            path=directory,
        )


def test_file_publication_rejects_symlink_even_when_target_is_regular(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    link = tmp_path / "link.bin"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlink creation unavailable on this host")
    with pytest.raises(ValueError, match="regular file"):
        publish_candidate_artifact_file(
            object_bytes=_ObjectByteAccess(CreateOnlyTransport()),
            uri="s3://bucket/results/link.bin",
            path=link,
        )
