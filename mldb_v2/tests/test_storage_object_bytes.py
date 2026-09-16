import hashlib
from pathlib import Path

import pytest

from mldb_v2.src.storage.object_bytes import _ObjectByteAccess


class FakeImmutableTransport:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.read_error: Exception | None = None
        self.publish_error: Exception | None = None

    def read_bytes(self, uri: str) -> bytes:
        if self.read_error is not None:
            raise self.read_error
        return self.objects[uri]

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        if self.publish_error is not None:
            raise self.publish_error
        if uri in self.objects:
            raise FileExistsError(uri)
        self.objects[uri] = data


def _ref(uri: str, data: bytes) -> dict[str, object]:
    return {"uri": uri, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_object_access_reads_and_verifies_expected_bytes() -> None:
    uri = "s3://bucket/object.bin"
    data = b"payload"
    transport = FakeImmutableTransport()
    transport.objects[uri] = data
    access = _ObjectByteAccess(transport)
    assert access.read_verified(_ref(uri, data)) == data


def test_object_access_rejects_size_and_digest_mismatch() -> None:
    uri = "s3://bucket/object.bin"
    data = b"payload"
    transport = FakeImmutableTransport()
    transport.objects[uri] = data
    access = _ObjectByteAccess(transport)

    wrong_size = _ref(uri, data)
    wrong_size["bytes"] = len(data) + 1
    with pytest.raises(ValueError, match="byte length"):
        access.read_verified(wrong_size)

    wrong_digest = _ref(uri, data)
    wrong_digest["sha256"] = hashlib.sha256(b"different").hexdigest()
    with pytest.raises(ValueError, match="sha256"):
        access.read_verified(wrong_digest)


def test_object_publish_is_create_only_and_never_overwrites_existing_uri() -> None:
    uri = "s3://bucket/object.bin"
    transport = FakeImmutableTransport()
    access = _ObjectByteAccess(transport)

    first = access.publish(uri=uri, data=b"first")
    assert transport.objects[uri] == b"first"
    assert first == _ref(uri, b"first")

    with pytest.raises(FileExistsError):
        access.publish(uri=uri, data=b"second")
    assert transport.objects[uri] == b"first"


def test_configured_transport_failures_propagate() -> None:
    transport = FakeImmutableTransport()
    access = _ObjectByteAccess(transport)
    transport.read_error = RuntimeError("configured read failed")
    with pytest.raises(RuntimeError, match="configured read failed"):
        access.read_verified(_ref("s3://bucket/object.bin", b"x"))

    transport.read_error = None
    transport.publish_error = RuntimeError("configured publish failed")
    with pytest.raises(RuntimeError, match="configured publish failed"):
        access.publish(uri="s3://bucket/new.bin", data=b"x")


def test_materialize_verified_creates_exact_file_without_replacing_existing(tmp_path: Path) -> None:
    uri = "s3://bucket/object.bin"
    data = b"exact bytes\x00"
    transport = FakeImmutableTransport()
    transport.objects[uri] = data
    access = _ObjectByteAccess(transport)

    target = tmp_path / "object.bin"
    assert access.materialize_verified(_ref(uri, data), target) == target
    assert target.read_bytes() == data

    with pytest.raises(FileExistsError):
        access.materialize_verified(_ref(uri, data), target)
    assert target.read_bytes() == data
