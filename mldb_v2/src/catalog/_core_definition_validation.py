"""Private shared validation helpers for core Catalog definitions."""

from __future__ import annotations

import re as _re
from urllib.parse import urlsplit as _urlsplit

from mldb_v2.src.common.ids import (
    _validate_canonical_json_value,
    _validate_typed_reference,
)
from mldb_v2.src.common.parameters import _validate_public_parameter_value

_VERSIONED_LOCAL_ID_RE = _re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*-v[1-9][0-9]*", _re.ASCII)
_LIFECYCLE_STATES = frozenset({"draft", "sealed"})


def _require_exact_mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a mapping")
    _validate_canonical_json_value(value)
    return value


def _require_required_fields(document: dict[str, object], required: set[str], *, label: str) -> None:
    missing = required - set(document)
    if missing:
        raise ValueError(f"{label} missing required field")


def _require_exact_fields(document: dict[str, object], allowed: set[str], *, label: str) -> None:
    if set(document) != allowed:
        raise ValueError(f"invalid {label} top-level fields")


def _require_string(value: object, *, label: str, non_empty: bool = False) -> str:
    if type(value) is not str or (non_empty and not value):
        qualifier = "non-empty " if non_empty else ""
        raise ValueError(f"{label} must be a {qualifier}string")
    return value


def _validate_lifecycle(value: object) -> str:
    if type(value) is not str or value not in _LIFECYCLE_STATES:
        raise ValueError("status must be draft or sealed")
    return value


def _validate_versioned_entity_id(value: object, *, expected_id: str | None = None) -> str:
    typed_id = _validate_typed_reference(value)
    _, local_id = typed_id.split("/", 1)
    if _VERSIONED_LOCAL_ID_RE.fullmatch(local_id) is None:
        raise ValueError("entity local id must end in canonical -v<positive-integer>")
    if expected_id is not None and typed_id != expected_id:
        raise ValueError("definition id does not match canonical path identity")
    return typed_id


def _validate_json_mapping(value: object, *, label: str) -> dict[str, object]:
    mapping = _require_exact_mapping(value, label=label)
    _validate_public_parameter_value(mapping)
    return mapping


def _validate_s3_root_uri(value: object) -> str:
    uri = _require_string(value, label="storage.root_uri", non_empty=True)
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in uri):
        raise ValueError("storage.root_uri contains whitespace or control characters")
    if "\\" in uri:
        raise ValueError("storage.root_uri must use URI separators")

    parsed = _urlsplit(uri)
    if parsed.scheme != "s3":
        raise ValueError("storage.root_uri must be an s3 logical root URI")
    if not parsed.netloc or parsed.hostname is None:
        raise ValueError("storage.root_uri requires a bucket")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ValueError("storage.root_uri must not contain credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid storage.root_uri authority") from exc
    if port is not None or ":" in parsed.netloc:
        raise ValueError("storage.root_uri must not contain endpoint configuration")
    if parsed.query or parsed.fragment:
        raise ValueError("storage.root_uri must not contain query or fragment data")
    return uri
