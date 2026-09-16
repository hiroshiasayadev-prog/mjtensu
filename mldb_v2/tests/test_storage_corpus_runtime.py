from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mldb_v2.src.common.ids import CorpusId
from mldb_v2.src.storage.corpus_runtime import (
    _target_for_manifest_path,
    materialize_sealed_corpus,
)
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess


class DictTransport:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.reads: list[str] = []

    def read_bytes(self, uri: str) -> bytes:
        self.reads.append(uri)
        return self.objects[uri]

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        raise AssertionError("Corpus materialization must not publish objects")


def _entry(path: str, data: bytes, *, size: int | None = None, digest: str | None = None) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(data) if size is None else size,
        "sha256": hashlib.sha256(data).hexdigest() if digest is None else digest,
    }


def _write_repo(
    tmp_path: Path,
    entries: list[dict[str, object]],
    *,
    manifest_sha256: str | None = None,
    manifest_count: int | None = None,
    with_builder: bool = False,
) -> tuple[Path, CorpusId, bytes]:
    root = tmp_path / "mldb_data"
    namespace = root / "demo"
    corpus_dir = namespace / "corpora"
    corpus_dir.mkdir(parents=True)
    (namespace / "namespace.yaml").write_text(json.dumps({
        "schema": "mjtensu.mldb-v2/namespace/v1",
        "id": "demo", "name": "Demo", "description": "test",
    }), encoding="utf-8")
    manifest_bytes = b"".join(
        json.dumps(entry, separators=(",", ":")).encode() + b"\n" for entry in entries
    )
    manifest_path = corpus_dir / "corpus-v1.manifest.jsonl"
    manifest_path.write_bytes(manifest_bytes)
    manifest = {
        "file": "corpus-v1.manifest.jsonl",
        "sha256": manifest_sha256 or hashlib.sha256(manifest_bytes).hexdigest(),
        "entries": len(entries) if manifest_count is None else manifest_count,
    }
    corpus: dict[str, object] = {
        "schema": "mjtensu.mldb-v2/corpus/v1", "id": "demo/corpus-v1",
        "status": "sealed", "task": "demo/task-v1", "description": "test",
        "storage": {"root_uri": "s3://bucket/corpus-v1"}, "manifest": manifest,
        "representation": {"kind": "test"}, "splits": {},
    }
    if with_builder:
        corpus["builder"] = {"entrypoint": "build", "sha256": "0" * 64}
        (corpus_dir / "corpus-v1.py").write_text(
            "raise RuntimeError('builder must never be imported')\n", encoding="utf-8"
        )
    (corpus_dir / "corpus-v1.yaml").write_text(json.dumps(corpus), encoding="utf-8")
    return root, CorpusId("demo/corpus-v1"), manifest_bytes


def _materialize(
    root: Path, corpus_id: CorpusId, transport: DictTransport, destination: Path
):
    return materialize_sealed_corpus(
        mldb_data_root=root,
        corpus_id=corpus_id,
        object_bytes=_ObjectByteAccess(transport),
        destination_root=destination,
    )


def test_materializes_valid_sealed_manifest_with_safe_nested_paths(tmp_path: Path) -> None:
    a = b"alpha"
    b = b"beta"
    entries = [_entry("images/a.bin", a), _entry("labels/nested/b.bin", b)]
    root, corpus_id, _ = _write_repo(tmp_path, entries, with_builder=True)
    transport = DictTransport({
        "s3://bucket/corpus-v1/images/a.bin": a,
        "s3://bucket/corpus-v1/labels/nested/b.bin": b,
        "s3://bucket/corpus-v1/undeclared.bin": b"extra",
    })
    destination = tmp_path / "materialized"

    corpus, materialized_root = _materialize(root, corpus_id, transport, destination)

    assert corpus["id"] == "demo/corpus-v1"
    assert materialized_root == destination
    assert (destination / "images" / "a.bin").read_bytes() == a
    assert (destination / "labels" / "nested" / "b.bin").read_bytes() == b
    assert not (destination / "undeclared.bin").exists()
    assert transport.reads == [
        "s3://bucket/corpus-v1/images/a.bin",
        "s3://bucket/corpus-v1/labels/nested/b.bin",
    ]


def test_manifest_digest_and_entry_count_mismatch_fail_before_object_reads(tmp_path: Path) -> None:
    entry = _entry("a.bin", b"a")
    bad_digest_root, corpus_id, _ = _write_repo(
        tmp_path / "digest", [entry], manifest_sha256="0" * 64
    )
    transport = DictTransport({"s3://bucket/corpus-v1/a.bin": b"a"})
    with pytest.raises(ValueError, match="manifest sha256 mismatch"):
        _materialize(bad_digest_root, corpus_id, transport, tmp_path / "out-digest")
    assert transport.reads == []

    bad_count_root, corpus_id, _ = _write_repo(
        tmp_path / "count", [entry], manifest_count=2
    )
    transport = DictTransport({"s3://bucket/corpus-v1/a.bin": b"a"})
    with pytest.raises(ValueError, match="entry count mismatch"):
        _materialize(bad_count_root, corpus_id, transport, tmp_path / "out-count")
    assert transport.reads == []


@pytest.mark.parametrize(
    ("entry", "objects", "error"),
    [
        (_entry("a.bin", b"expected"), {}, KeyError),
        (_entry("a.bin", b"expected", size=9), {"s3://bucket/corpus-v1/a.bin": b"expected"}, ValueError),
        (_entry("a.bin", b"expected", digest=hashlib.sha256(b"other").hexdigest()), {"s3://bucket/corpus-v1/a.bin": b"expected"}, ValueError),
    ],
    ids=["missing", "size-mismatch", "sha-mismatch"],
)
def test_object_integrity_failures_are_rejected(
    tmp_path: Path,
    entry: dict[str, object],
    objects: dict[str, bytes],
    error: type[BaseException],
) -> None:
    root, corpus_id, _ = _write_repo(tmp_path, [entry])
    with pytest.raises(error):
        _materialize(root, corpus_id, DictTransport(objects), tmp_path / "out")


def test_manifest_traversal_is_rejected_before_materialization(tmp_path: Path) -> None:
    root, corpus_id, _ = _write_repo(tmp_path, [_entry("../escape.bin", b"x")])
    transport = DictTransport({})
    with pytest.raises(ValueError, match="unsafe path segment"):
        _materialize(root, corpus_id, transport, tmp_path / "out")
    assert transport.reads == []


def test_destination_escape_is_rejected_by_resolved_root_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    original_resolve = Path.resolve

    def fake_resolve(path: Path, strict: bool = False) -> Path:
        if path == root / "nested" / "object.bin":
            return outside / "object.bin"
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    with pytest.raises(ValueError, match="escapes materialization root"):
        _target_for_manifest_path(root, "nested/object.bin")
