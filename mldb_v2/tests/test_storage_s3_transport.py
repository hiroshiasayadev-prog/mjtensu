from __future__ import annotations

import builtins
import io
import sys
from types import SimpleNamespace

import pytest

from mldb_v2.src.storage.artifact_runtime import publish_candidate_artifact_bytes
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.storage.s3_transport import (
    S3ObjectByteTransport,
    _S3TransportConfig,
    _create_s3_transport,
    _parse_s3_uri,
)


class PreconditionFailed(Exception):
    def __init__(self) -> None:
        self.response = {
            "Error": {"Code": "PreconditionFailed"},
            "ResponseMetadata": {"HTTPStatusCode": 412},
        }
        super().__init__("conditional create failed")


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.gets: list[dict[str, object]] = []
        self.puts: list[dict[str, object]] = []

    def get_object(self, **kwargs: object) -> dict[str, object]:
        self.gets.append(dict(kwargs))
        key = (str(kwargs["Bucket"]), str(kwargs["Key"]))
        return {"Body": io.BytesIO(self.objects[key])}

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.puts.append(dict(kwargs))
        key = (str(kwargs["Bucket"]), str(kwargs["Key"]))
        if key in self.objects:
            raise PreconditionFailed()
        body = kwargs["Body"]
        assert type(body) is bytes
        self.objects[key] = body
        return {"ETag": "fake"}


def test_s3_uri_maps_to_exact_bucket_and_key_and_reads_exact_bytes() -> None:
    assert _parse_s3_uri("s3://bucket/a/b.bin") == ("bucket", "a/b.bin")
    client = FakeS3Client()
    client.objects[("bucket", "a/b.bin")] = b"payload"
    transport = S3ObjectByteTransport(client)
    assert transport.read_bytes("s3://bucket/a/b.bin") == b"payload"
    assert client.gets == [{"Bucket": "bucket", "Key": "a/b.bin"}]


def test_conditional_create_uses_if_none_match_and_succeeds() -> None:
    client = FakeS3Client()
    transport = S3ObjectByteTransport(client)
    transport.publish_bytes_immutable("s3://bucket/new.bin", b"new")
    assert client.objects[("bucket", "new.bin")] == b"new"
    assert client.puts == [{
        "Bucket": "bucket", "Key": "new.bin", "Body": b"new", "IfNoneMatch": "*"
    }]


def test_existing_identical_object_is_idempotent_after_conditional_failure() -> None:
    client = FakeS3Client()
    client.objects[("bucket", "same.bin")] = b"same"
    transport = S3ObjectByteTransport(client)
    transport.publish_bytes_immutable("s3://bucket/same.bin", b"same")
    assert len(client.puts) == 1
    assert client.puts[0]["IfNoneMatch"] == "*"
    assert client.gets == [{"Bucket": "bucket", "Key": "same.bin"}]


def test_existing_different_object_fails_without_unconditional_overwrite() -> None:
    client = FakeS3Client()
    client.objects[("bucket", "same.bin")] = b"first"
    transport = S3ObjectByteTransport(client)
    with pytest.raises(FileExistsError, match="different bytes"):
        transport.publish_bytes_immutable("s3://bucket/same.bin", b"second")
    assert client.objects[("bucket", "same.bin")] == b"first"
    assert len(client.puts) == 1
    assert client.puts[0]["IfNoneMatch"] == "*"


def test_config_is_only_client_configuration_and_not_returned_canonical_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3Client()
    captured: dict[str, object] = {}

    def client_factory(service: str, **kwargs: object) -> FakeS3Client:
        captured["service"] = service
        captured["kwargs"] = dict(kwargs)
        return client

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=client_factory))
    config = _S3TransportConfig(
        endpoint_url="http://minio.internal:9000",
        region_name="us-east-1",
        access_key_id="access",
        secret_access_key="secret",
        session_token="token",
    )
    transport = _create_s3_transport(config)
    ref = publish_candidate_artifact_bytes(
        object_bytes=_ObjectByteAccess(transport),
        uri="s3://bucket/object.bin",
        data=b"payload",
    )
    assert set(ref) == {"uri", "bytes", "sha256"}
    assert "minio.internal" not in repr(ref)
    assert "secret" not in repr(ref)
    assert "secret" not in repr(config)
    assert captured == {
        "service": "s3",
        "kwargs": {
            "endpoint_url": "http://minio.internal:9000",
            "region_name": "us-east-1",
            "aws_access_key_id": "access",
            "aws_secret_access_key": "secret",
            "aws_session_token": "token",
        },
    }


def test_missing_boto3_reports_explicit_production_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "boto3", raising=False)
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "boto3":
            raise ModuleNotFoundError("No module named 'boto3'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="boto3 is required"):
        _create_s3_transport()


def test_invalid_logical_s3_uri_is_rejected_before_client_call() -> None:
    client = FakeS3Client()
    transport = S3ObjectByteTransport(client)
    with pytest.raises(ValueError):
        transport.read_bytes("s3://user:secret@bucket/object.bin")
    assert client.gets == []
    assert client.puts == []
