import hashlib
import json

import pytest

from mldb_v2.src.storage.corpus_manifest import (
    _corpus_manifest_sha256,
    _parse_corpus_manifest,
    _verify_corpus_manifest,
)


DIGEST_A = hashlib.sha256(b"a").hexdigest()
DIGEST_B = hashlib.sha256(b"b").hexdigest()


def _line(path: str, *, size: int = 1, digest: str = DIGEST_A, split: str | None = None) -> bytes:
    value: dict[str, object] = {"path": path, "bytes": size, "sha256": digest}
    if split is not None:
        value["split"] = split
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def test_manifest_parses_valid_lexically_ordered_jsonl() -> None:
    data = _line("images/a.jpg", split="train") + b"\n" + _line("images/b.jpg", digest=DIGEST_B) + b"\n"
    entries = _parse_corpus_manifest(data)
    assert [entry["path"] for entry in entries] == ["images/a.jpg", "images/b.jpg"]
    assert entries[0]["split"] == "train"


@pytest.mark.parametrize("field", ["path", "bytes", "sha256", "split"])
def test_manifest_rejects_duplicate_json_members(field: str) -> None:
    values = {
        "path": ('"images/a.jpg"', '"images/b.jpg"'),
        "bytes": ("1", "2"),
        "sha256": (f'"{DIGEST_A}"', f'"{DIGEST_B}"'),
        "split": ('"train"', '"val"'),
    }
    first, second = values[field]
    members = [
        '"path":"images/a.jpg"',
        '"bytes":1',
        f'"sha256":"{DIGEST_A}"',
        '"split":"train"',
    ]
    index = next(i for i, member in enumerate(members) if member.startswith(f'"{field}":'))
    members[index:index + 1] = [f'"{field}":{first}', f'"{field}":{second}']
    with pytest.raises(ValueError, match="malformed corpus manifest JSON"):
        _parse_corpus_manifest(("{" + ",".join(members) + "}\n").encode("utf-8"))


def test_manifest_duplicate_member_cannot_change_lexical_order_outcome() -> None:
    first = f'{{"path":"images/z.jpg","path":"images/a.jpg","bytes":1,"sha256":"{DIGEST_A}"}}\n'.encode()
    second = _line("images/b.jpg") + b"\n"
    with pytest.raises(ValueError, match="malformed corpus manifest JSON"):
        _parse_corpus_manifest(first + second)


def test_manifest_rejects_nonfinite_json_numbers() -> None:
    data = f'{{"path":"images/a.jpg","bytes":NaN,"sha256":"{DIGEST_A}"}}\n'.encode()
    with pytest.raises(ValueError, match="malformed corpus manifest JSON"):
        _parse_corpus_manifest(data)


@pytest.mark.parametrize(
    "path",
    [
        "../secret",
        "images/../../secret",
        "/absolute/path",
        r"C:\absolute\path",
        "C:/absolute/path",
        "",
        ".",
        "images/./a.jpg",
        "images//a.jpg",
        r"images\a.jpg",
    ],
)
def test_manifest_rejects_unsafe_cross_platform_paths(path: str) -> None:
    with pytest.raises(ValueError):
        _parse_corpus_manifest(_line(path))


def test_manifest_rejects_nonlexical_order_and_duplicate_path() -> None:
    with pytest.raises(ValueError, match="lexical"):
        _parse_corpus_manifest(_line("images/b.jpg") + b"\n" + _line("images/a.jpg"))
    with pytest.raises(ValueError, match="unique"):
        _parse_corpus_manifest(_line("images/a.jpg") + b"\n" + _line("images/a.jpg"))


@pytest.mark.parametrize(
    "data",
    [
        b"{not-json}\n",
        b"\n",
        json.dumps(["not", "an", "object"]).encode("utf-8"),
        json.dumps({"path": "images/a.jpg", "bytes": 1}).encode("utf-8"),
        json.dumps({"path": "images/a.jpg", "bytes": True, "sha256": DIGEST_A}).encode("utf-8"),
        json.dumps({"path": "images/a.jpg", "bytes": 1.0, "sha256": DIGEST_A}).encode("utf-8"),
        json.dumps({"path": "images/a.jpg", "bytes": 1, "sha256": "A" * 64}).encode("utf-8"),
        json.dumps({"path": "images/a.jpg", "bytes": 1, "sha256": DIGEST_A, "split": 1}).encode("utf-8"),
        json.dumps({"path": "images/a.jpg", "bytes": 1, "sha256": DIGEST_A, "extra": "x"}).encode("utf-8"),
    ],
)
def test_manifest_rejects_malformed_or_nonconforming_rows(data: bytes) -> None:
    with pytest.raises(ValueError):
        _parse_corpus_manifest(data)


def test_manifest_digest_is_over_exact_file_bytes_and_count_is_verified() -> None:
    data = _line("images/a.jpg") + b"\n" + _line("images/b.jpg", digest=DIGEST_B) + b"\n"
    digest = hashlib.sha256(data).hexdigest()
    assert _corpus_manifest_sha256(data) == digest
    assert len(_verify_corpus_manifest(data, expected_sha256=digest, expected_count=2)) == 2

    without_final_lf = data[:-1]
    assert _corpus_manifest_sha256(without_final_lf) != digest
    with pytest.raises(ValueError, match="sha256"):
        _verify_corpus_manifest(without_final_lf, expected_sha256=digest, expected_count=2)
    with pytest.raises(ValueError, match="count"):
        _verify_corpus_manifest(data, expected_sha256=digest, expected_count=1)
