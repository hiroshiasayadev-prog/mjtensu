"""Canonical MLDB v2 Corpus manifest parsing and integrity validation."""

from __future__ import annotations

import hashlib as _hashlib
import json as _json
from typing import NotRequired, TypedDict

from mldb_v2.src.storage._paths import _validate_safe_relative_path
from mldb_v2.src.storage.artifact_reference import _validate_byte_count, _validate_sha256


class CorpusManifestEntry(TypedDict):
    path: str
    bytes: int
    sha256: str
    split: NotRequired[str]


_REQUIRED_FIELDS = {"path", "bytes", "sha256"}
_ALLOWED_FIELDS = _REQUIRED_FIELDS | {"split"}


class _StrictJsonError(ValueError):
    pass


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _StrictJsonError(f"duplicate JSON member: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(value: str) -> object:
    raise _StrictJsonError(f"non-finite JSON number: {value}")


def _validate_corpus_manifest_entry(value: object) -> CorpusManifestEntry:
    if type(value) is not dict:
        raise ValueError("corpus manifest entry must be a JSON object")
    fields = set(value)
    if not _REQUIRED_FIELDS <= fields or not fields <= _ALLOWED_FIELDS:
        raise ValueError("invalid corpus manifest entry fields")

    _validate_safe_relative_path(value["path"], label="manifest path")
    _validate_byte_count(value["bytes"], label="manifest bytes")
    _validate_sha256(value["sha256"], label="manifest entry sha256")
    if "split" in value and type(value["split"]) is not str:
        raise ValueError("manifest split must be a string when present")
    return value


def _validate_manifest_entries(entries: list[CorpusManifestEntry]) -> list[CorpusManifestEntry]:
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("corpus manifest paths must be unique")
    if paths != sorted(paths):
        raise ValueError("corpus manifest entries must be in lexical path order")
    return entries


def _parse_corpus_manifest(data: bytes) -> list[CorpusManifestEntry]:
    if type(data) is not bytes:
        raise ValueError("corpus manifest must be exact bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("corpus manifest must be UTF-8") from exc

    raw_lines = text.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()
    entries: list[CorpusManifestEntry] = []
    for line_number, line in enumerate(raw_lines, start=1):
        if not line:
            raise ValueError(f"blank corpus manifest line at {line_number}")
        try:
            decoded = _json.loads(
                line,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (_json.JSONDecodeError, _StrictJsonError) as exc:
            raise ValueError(f"malformed corpus manifest JSON at line {line_number}") from exc
        entries.append(_validate_corpus_manifest_entry(decoded))
    return _validate_manifest_entries(entries)


def _corpus_manifest_sha256(data: bytes) -> str:
    if type(data) is not bytes:
        raise ValueError("corpus manifest must be exact bytes")
    return _hashlib.sha256(data).hexdigest()


def _verify_corpus_manifest(
    data: bytes,
    *,
    expected_sha256: str,
    expected_count: int | None = None,
) -> list[CorpusManifestEntry]:
    _validate_sha256(expected_sha256, label="manifest sha256")
    entries = _parse_corpus_manifest(data)
    if _corpus_manifest_sha256(data) != expected_sha256:
        raise ValueError("corpus manifest sha256 mismatch")
    if expected_count is not None:
        if type(expected_count) is not int or expected_count < 0:
            raise ValueError("expected manifest entry count must be a non-negative integer")
        if len(entries) != expected_count:
            raise ValueError("corpus manifest entry count mismatch")
    return entries
