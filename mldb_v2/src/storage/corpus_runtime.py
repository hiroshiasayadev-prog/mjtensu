"""Sealed Corpus materialization for MLDB v2 execution runtimes."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TypeAlias

from mldb_v2.src.catalog.corpus import Corpus, _load_corpus
from mldb_v2.src.common.ids import CorpusId
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.storage._paths import _validate_safe_relative_path
from mldb_v2.src.storage.artifact_reference import ArtifactRef
from mldb_v2.src.storage.corpus_manifest import _verify_corpus_manifest
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess

CorpusMaterialization: TypeAlias = tuple[Corpus, Path]


def _logical_object_uri(root_uri: str, relative_path: str) -> str:
    validated_path = _validate_safe_relative_path(relative_path, label="manifest path")
    return f"{root_uri.rstrip('/')}/{validated_path}"


def _target_for_manifest_path(root: Path, relative_path: str) -> Path:
    validated_path = _validate_safe_relative_path(relative_path, label="manifest path")
    root_resolved = root.resolve(strict=True)
    target = root.joinpath(*PurePosixPath(validated_path).parts)
    target_resolved = target.resolve(strict=False)
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("manifest destination escapes materialization root") from exc
    return target


def _manifest_file_path(mldb_data_root: Path, corpus: Corpus) -> Path:
    namespace, _ = corpus["id"].split("/", 1)
    return mldb_data_root / namespace / "corpora" / corpus["manifest"]["file"]


def materialize_sealed_corpus(
    *,
    mldb_data_root: str | Path,
    corpus_id: CorpusId,
    object_bytes: _ObjectByteAccess,
    destination_root: str | Path,
) -> CorpusMaterialization:
    """Resolve and materialize exactly one sealed Corpus manifest object set."""
    repository_root = Path(mldb_data_root)
    resolver = CanonicalRepositoryResolver(repository_root)
    corpus = _load_corpus(resolver, corpus_id)
    if corpus["status"] != "sealed":
        raise ValueError("Corpus must be sealed before materialization")

    manifest = corpus["manifest"]
    manifest_bytes = _manifest_file_path(repository_root, corpus).read_bytes()
    entries = _verify_corpus_manifest(
        manifest_bytes,
        expected_sha256=manifest["sha256"],
        expected_count=manifest["entries"],
    )

    destination = Path(destination_root)
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise ValueError("Corpus materialization root must be a directory")

    for entry in entries:
        target = _target_for_manifest_path(destination, entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        _target_for_manifest_path(destination, entry["path"])
        ref: ArtifactRef = {
            "uri": _logical_object_uri(corpus["storage"]["root_uri"], entry["path"]),
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
        object_bytes.materialize_verified(ref, target)

    return corpus, destination
