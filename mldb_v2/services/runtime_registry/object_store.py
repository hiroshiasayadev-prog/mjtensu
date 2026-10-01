"""Standalone S3-compatible immutable object store for the runtime registry."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.startswith("/"):
        raise ValueError(f"invalid s3 uri: {uri!r}")
    key = parsed.path[1:]
    if not key:
        raise ValueError(f"invalid s3 uri without object key: {uri!r}")
    return parsed.netloc, key


def _is_create_conflict(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return False
    error_data = response.get("Error")
    if isinstance(error_data, dict):
        code = str(error_data.get("Code", ""))
        if code in {"PreconditionFailed", "ConditionalRequestConflict", "409", "412"}:
            return True
    metadata = response.get("ResponseMetadata")
    return isinstance(metadata, dict) and metadata.get("HTTPStatusCode") in {409, 412}


class S3ObjectStore:
    def __init__(self, client: Any) -> None:
        self._client = client

    def read_bytes(self, uri: str) -> bytes:
        bucket, key = _parse_s3_uri(uri)
        response = self._client.get_object(Bucket=bucket, Key=key)
        body = response.get("Body")
        if body is None or not hasattr(body, "read"):
            raise ValueError("S3 get_object response is missing readable Body")
        try:
            data = body.read()
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if not isinstance(data, bytes):
            raise ValueError("S3 object body must return bytes")
        return data

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("S3 object data must be bytes")
        bucket, key = _parse_s3_uri(uri)
        try:
            self._client.put_object(Bucket=bucket, Key=key, Body=data, IfNoneMatch="*")
            return
        except Exception as exc:
            if not _is_create_conflict(exc):
                raise
            if self.read_bytes(uri) == data:
                return
            raise FileExistsError(f"immutable S3 object already exists with different bytes: {uri}") from exc
