"""Production S3-compatible immutable object-byte transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from mldb_v2.src.storage.artifact_reference import _validate_logical_object_uri


@dataclass(frozen=True)
class _S3TransportConfig:
    endpoint_url: str | None = None
    region_name: str | None = None
    access_key_id: str | None = field(default=None, repr=False)
    secret_access_key: str | None = field(default=None, repr=False)
    session_token: str | None = field(default=None, repr=False)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    validated = _validate_logical_object_uri(uri)
    parsed = urlsplit(validated)
    return parsed.netloc, parsed.path[1:]


def _is_conditional_create_conflict(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    if type(response) is not dict:
        return False
    error_data = response.get("Error")
    if type(error_data) is dict:
        code = str(error_data.get("Code", ""))
        if code in {"PreconditionFailed", "ConditionalRequestConflict", "409", "412"}:
            return True
    metadata = response.get("ResponseMetadata")
    if type(metadata) is dict and metadata.get("HTTPStatusCode") in {409, 412}:
        return True
    return False


class S3ObjectByteTransport:
    """Configured boto3-style S3 client adapter for logical ``s3://`` identities."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def read_bytes(self, uri: str) -> bytes:
        bucket, key = _parse_s3_uri(uri)
        response = self._client.get_object(Bucket=bucket, Key=key)
        body = response.get("Body")
        if body is None or not hasattr(body, "read"):
            raise ValueError("S3 get_object response is missing a readable Body")
        try:
            data = body.read()
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if type(data) is not bytes:
            raise ValueError("S3 object body must produce exact bytes")
        return data

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        if type(data) is not bytes:
            raise ValueError("S3 object data must be exact bytes")
        bucket, key = _parse_s3_uri(uri)
        try:
            self._client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                IfNoneMatch="*",
            )
            return
        except Exception as exc:
            if not _is_conditional_create_conflict(exc):
                raise
            existing = self.read_bytes(uri)
            if existing == data:
                return
            raise FileExistsError(
                f"immutable S3 object already exists with different bytes: {uri}"
            ) from exc


def _create_s3_transport(
    config: _S3TransportConfig | None = None,
) -> S3ObjectByteTransport:
    """Create the production adapter without importing boto3 at module import time."""
    effective = config or _S3TransportConfig()
    try:
        import boto3  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "boto3 is required to activate the production S3-compatible transport"
        ) from exc

    kwargs: dict[str, object] = {}
    if effective.endpoint_url is not None:
        kwargs["endpoint_url"] = effective.endpoint_url
    if effective.region_name is not None:
        kwargs["region_name"] = effective.region_name
    if effective.access_key_id is not None:
        kwargs["aws_access_key_id"] = effective.access_key_id
    if effective.secret_access_key is not None:
        kwargs["aws_secret_access_key"] = effective.secret_access_key
    if effective.session_token is not None:
        kwargs["aws_session_token"] = effective.session_token

    return S3ObjectByteTransport(boto3.client("s3", **kwargs))
