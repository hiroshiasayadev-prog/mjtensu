"""Backend-neutral immutable artifact reference runtime validation."""

from __future__ import annotations

import hashlib as _hashlib
import re as _re
from typing import TypedDict

from mldb_v2.src.common.ids import _validate_canonical_json_value
from urllib.parse import parse_qsl as _parse_qsl, urlsplit as _urlsplit


class ArtifactRef(TypedDict):
    uri: str
    bytes: int
    sha256: str


_SHA256_RE = _re.compile(r"[0-9a-f]{64}", _re.ASCII)
_REQUIRED_FIELDS = {"uri", "bytes", "sha256"}
_FORBIDDEN_METADATA_FIELD_TOKENS = frozenset(
    {
        "credential",
        "credentials",
        "accesskey",
        "accesskeys",
        "accesskeyid",
        "accesskeyids",
        "secretaccesskey",
        "secretaccesskeys",
        "awsaccesskeyid",
        "awssecretaccesskey",
        "sessiontoken",
        "awssessiontoken",
        "password",
        "passwords",
        "presignedurl",
        "presignedurls",
        "cachepath",
        "cachepaths",
        "cachedir",
        "cachedirs",
        "cachedirectory",
        "cachedirectories",
        "localcachepath",
        "localcachedir",
        "backendcachepath",
        "backendcachedir",
        "localpath",
        "localpaths",
        "backendlocalpath",
        "backendlocalpaths",
    }
)
_PRESIGNED_QUERY_FIELD_TOKENS = frozenset(
    {
        "xamzalgorithm",
        "xamzcredential",
        "xamzdate",
        "xamzexpires",
        "xamzsecuritytoken",
        "xamzsignature",
        "xamzsignedheaders",
        "awsaccesskeyid",
    }
)
_WINDOWS_ABSOLUTE_PATH_RE = _re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_OPERATIONAL_PATH_CONTEXT_FIELD_TOKENS = frozenset(
    {
        "backend",
        "runtime",
        "cache",
        "backendruntime",
        "runtimebackend",
        "backendcache",
        "cachebackend",
        "runtimecache",
        "cacheruntime",
        "localbackend",
        "localruntime",
        "localcache",
    }
)


def _validate_sha256(value: object, *, label: str = "sha256") -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")
    return value


def _validate_byte_count(value: object, *, label: str = "bytes") -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _metadata_field_token(key: str) -> str:
    return "".join(character for character in key.casefold() if character.isalnum())


def _looks_like_presigned_url(value: str) -> bool:
    parsed = _urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.query:
        return False
    query_tokens = {
        _metadata_field_token(key)
        for key, _ in _parse_qsl(parsed.query, keep_blank_values=True)
    }
    return bool(query_tokens & _PRESIGNED_QUERY_FIELD_TOKENS)


def _looks_like_local_absolute_path(value: str) -> bool:
    return (
        value.startswith("/")
        or _WINDOWS_ABSOLUTE_PATH_RE.match(value) is not None
        or value.casefold().startswith("file://")
    )


def _validate_no_forbidden_artifact_metadata(
    value: object,
    *,
    key_path: tuple[str, ...] = (),
) -> None:
    if type(value) is dict:
        for key, item in value.items():
            token = _metadata_field_token(key)
            if token in _FORBIDDEN_METADATA_FIELD_TOKENS:
                raise ValueError(f"forbidden operational ArtifactRef metadata field: {key}")
            _validate_no_forbidden_artifact_metadata(item, key_path=(*key_path, token))
        return
    if type(value) is list:
        for item in value:
            _validate_no_forbidden_artifact_metadata(item, key_path=key_path)
        return
    if type(value) is str:
        if _looks_like_presigned_url(value):
            raise ValueError("forbidden presigned URL in ArtifactRef metadata")
        if (
            any(token in _OPERATIONAL_PATH_CONTEXT_FIELD_TOKENS for token in key_path)
            and _looks_like_local_absolute_path(value)
        ):
            raise ValueError("forbidden backend-local operational path in ArtifactRef metadata")


def _validate_logical_object_uri(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("artifact uri must be a non-empty string")
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("artifact uri contains whitespace or control characters")
    if "\\" in value:
        raise ValueError("artifact uri must use URI separators")

    parsed = _urlsplit(value)
    if parsed.scheme != "s3":
        raise ValueError("artifact uri must be an s3 logical object URI")
    if not parsed.netloc or parsed.hostname is None:
        raise ValueError("artifact uri requires a bucket")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ValueError("artifact uri must not contain credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid artifact uri authority") from exc
    if port is not None or ":" in parsed.netloc:
        raise ValueError("artifact uri must not contain endpoint configuration")
    if parsed.query or parsed.fragment:
        raise ValueError("artifact uri must not contain query or fragment data")
    if not parsed.path or parsed.path == "/":
        raise ValueError("artifact uri requires an object key")
    return value


def _validate_artifact_ref(value: object) -> ArtifactRef:
    if type(value) is not dict or not _REQUIRED_FIELDS <= set(value):
        raise ValueError("artifact reference requires uri, bytes, and sha256")
    _validate_canonical_json_value(value)
    _validate_no_forbidden_artifact_metadata(value)
    _validate_logical_object_uri(value["uri"])
    _validate_byte_count(value["bytes"])
    _validate_sha256(value["sha256"])
    return value


def _artifact_ref_for_bytes(*, uri: str, data: bytes) -> ArtifactRef:
    _validate_logical_object_uri(uri)
    if type(data) is not bytes:
        raise ValueError("artifact data must be exact bytes")
    return {
        "uri": uri,
        "bytes": len(data),
        "sha256": _hashlib.sha256(data).hexdigest(),
    }


def _verify_artifact_bytes(*, ref: ArtifactRef, data: bytes) -> bytes:
    _validate_artifact_ref(ref)
    if type(data) is not bytes:
        raise ValueError("artifact data must be exact bytes")
    if len(data) != ref["bytes"]:
        raise ValueError("artifact byte length mismatch")
    if _hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("artifact sha256 mismatch")
    return data
