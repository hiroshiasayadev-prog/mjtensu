import hashlib

import pytest

from mldb_v2.src.storage.artifact_reference import (
    _artifact_ref_for_bytes,
    _validate_artifact_ref,
    _verify_artifact_bytes,
)


def _ref(data: bytes = b"payload") -> dict[str, object]:
    return {
        "uri": "s3://mldb-artifacts/ns/object.bin",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_artifact_ref_accepts_exact_frozen_shape() -> None:
    value = _ref()
    assert _validate_artifact_ref(value) is value


@pytest.mark.parametrize("value", [-1, True, 1.5, "7"])
def test_artifact_ref_rejects_invalid_byte_count(value: object) -> None:
    ref = _ref()
    ref["bytes"] = value
    with pytest.raises(ValueError):
        _validate_artifact_ref(ref)


@pytest.mark.parametrize(
    "digest",
    ["", "0" * 63, "g" * 64, "A" * 64, "0" * 65],
)
def test_artifact_ref_rejects_malformed_or_uppercase_digest(digest: str) -> None:
    ref = _ref()
    ref["sha256"] = digest
    with pytest.raises(ValueError):
        _validate_artifact_ref(ref)


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "https://bucket.example/object?X-Amz-Signature=secret",
        "s3://access:secret@bucket/object",
        "s3://bucket/object?X-Amz-Credential=secret",
        "s3://bucket/object#fragment",
        "s3://bucket:9000/object",
        "s3:///object",
        "s3://bucket/white space",
        "file:///tmp/cache/object",
        r"C:\cache\object.bin",
        "s3://bucket",
    ],
)
def test_artifact_ref_rejects_nonlogical_or_credential_style_uri(uri: str) -> None:
    ref = _ref()
    ref["uri"] = uri
    with pytest.raises(ValueError):
        _validate_artifact_ref(ref)


def test_artifact_ref_accepts_domain_specific_immutable_metadata_extensions() -> None:
    ref = _ref()
    ref["format"] = "pytorch-state-dict/v1"
    ref["schema"] = "mjtensu.example/artifact/v1"
    assert _validate_artifact_ref(ref) is ref


def test_artifact_ref_accepts_nested_benign_immutable_metadata() -> None:
    ref = _ref()
    ref["metadata"] = {
        "format": "jsonl",
        "schema": "mjtensu.example/predictions/v1",
        "media_type": "application/jsonl",
        "compression": "none",
        "logical_name": "predictions",
        "key": "descriptive-key",
        "path": "logical/path",
        "url": "https://example.invalid/schema-doc",
    }
    assert _validate_artifact_ref(ref) is ref


@pytest.mark.parametrize(
    "field",
    ["password", "credentials", "access_key", "accessKeys", "presigned_url", "cache_path"],
)
def test_artifact_ref_rejects_forbidden_operational_metadata(field: str) -> None:
    ref = _ref()
    ref[field] = "runtime-only"
    with pytest.raises(ValueError, match="forbidden operational"):
        _validate_artifact_ref(ref)


def test_artifact_ref_rejects_nested_forbidden_operational_metadata() -> None:
    ref = _ref()
    ref["metadata"] = {"descriptive": {"credentials": {"access_key": "secret"}}}
    with pytest.raises(ValueError, match="forbidden operational"):
        _validate_artifact_ref(ref)


def test_artifact_ref_rejects_presigned_url_hidden_under_generic_metadata_key() -> None:
    ref = _ref()
    ref["metadata"] = {
        "url": "https://bucket.example/object?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=secret"
    }
    with pytest.raises(ValueError, match="presigned URL"):
        _validate_artifact_ref(ref)


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({"backend": {"path": "C:/runtime/object.bin"}}, id="backend-windows"),
        pytest.param({"backend": {"path": "/srv/runtime/object.bin"}}, id="backend-unix"),
        pytest.param({"runtime": {"path": r"C:\\runtime\\object.bin"}}, id="runtime-windows"),
        pytest.param({"runtime": {"path": "/srv/runtime/object.bin"}}, id="runtime-unix"),
        pytest.param({"cache": {"path": "C:/runtime/cache/object.bin"}}, id="cache-regression"),
        pytest.param(
            {"execution": {"backend": {"output": {"path": "C:/tmp/x"}}}},
            id="deep-backend",
        ),
        pytest.param(
            {"execution": {"runtime": {"output": {"path": "/srv/tmp/x"}}}},
            id="deep-runtime",
        ),
        pytest.param({"backend": {"path": r"\\server\share\x"}}, id="backend-unc"),
        pytest.param({"runtime": {"path": "file:///tmp/x"}}, id="runtime-file-uri"),
        pytest.param({"backend_runtime": {"path": "/srv/runtime/x"}}, id="backend-runtime-variant"),
        pytest.param({"local-cache": {"path": "C:/runtime/x"}}, id="local-cache-variant"),
    ],
)
def test_artifact_ref_rejects_absolute_local_paths_in_operational_context(
    metadata: dict[str, object],
) -> None:
    ref = _ref()
    ref["metadata"] = metadata
    with pytest.raises(ValueError, match="backend-local operational path"):
        _validate_artifact_ref(ref)


def test_artifact_ref_allows_relative_paths_inside_operational_context() -> None:
    ref = _ref()
    ref["metadata"] = {
        "backend": {"path": "runtime/output.bin"},
        "runtime": {"path": "./relative/output.bin"},
    }
    assert _validate_artifact_ref(ref) is ref


def test_artifact_ref_does_not_apply_operational_path_rule_to_descriptive_context() -> None:
    ref = _ref()
    ref["metadata"] = {
        "descriptive": {"path": "/some/domain-defined/value"},
        "backend_notes": {"path": "C:/domain-defined/value"},
        "runtime_format": {"path": "/domain-defined/value"},
    }
    assert _validate_artifact_ref(ref) is ref


def test_artifact_ref_allows_benign_generic_url_and_logical_path_metadata() -> None:
    ref = _ref()
    ref["metadata"] = {
        "url": "https://example.invalid/schema?version=1",
        "path": "logical/name",
    }
    assert _validate_artifact_ref(ref) is ref


def test_artifact_ref_rejects_non_json_extension_value() -> None:
    ref = _ref()
    ref["metadata"] = object()
    with pytest.raises(ValueError):
        _validate_artifact_ref(ref)


def test_artifact_ref_for_bytes_and_verification_are_exact() -> None:
    data = b"\x00exact\xffbytes\n"
    ref = _artifact_ref_for_bytes(uri="s3://bucket/path/object", data=data)
    assert ref["bytes"] == len(data)
    assert ref["sha256"] == hashlib.sha256(data).hexdigest()
    assert _verify_artifact_bytes(ref=ref, data=data) == data


def test_artifact_verification_rejects_size_and_digest_mismatch() -> None:
    data = b"payload"
    ref = _artifact_ref_for_bytes(uri="s3://bucket/path/object", data=data)
    with pytest.raises(ValueError, match="byte length"):
        _verify_artifact_bytes(ref=ref, data=data + b"x")
    wrong = dict(ref)
    wrong["sha256"] = hashlib.sha256(b"different").hexdigest()
    with pytest.raises(ValueError, match="sha256"):
        _verify_artifact_bytes(ref=wrong, data=data)
