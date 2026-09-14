"""MLDB v2 Corpus public shapes and private validation."""

from typing import Literal, NotRequired, TypeAlias, TypedDict, cast

from mldb_v2.src.common.ids import CorpusId, EntityKind, TaskId
from mldb_v2.src.common.parameters import PublicParameterValue
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.storage.artifact_reference import _validate_byte_count, _validate_sha256

from ._core_definition_validation import (
    _require_exact_fields,
    _require_exact_mapping,
    _require_required_fields,
    _require_string,
    _validate_json_mapping,
    _validate_lifecycle,
    _validate_s3_root_uri,
    _validate_versioned_entity_id,
)

CorpusRepresentation: TypeAlias = dict[str, PublicParameterValue]
CorpusSplits: TypeAlias = dict[str, int]
CorpusBuilderParameters: TypeAlias = dict[str, PublicParameterValue]


class CorpusStorage(TypedDict):
    root_uri: str


class CorpusManifest(TypedDict):
    file: str
    sha256: NotRequired[str]
    entries: NotRequired[int]


class CorpusBuilder(TypedDict):
    entrypoint: Literal["build"]
    sha256: NotRequired[str]
    parameters: NotRequired[CorpusBuilderParameters]


class Corpus(TypedDict):
    schema: Literal["mjtensu.mldb-v2/corpus/v1"]
    id: CorpusId
    status: Literal["draft", "sealed"]
    task: TaskId
    description: str
    storage: CorpusStorage
    manifest: CorpusManifest
    representation: CorpusRepresentation
    splits: CorpusSplits
    builder: NotRequired[CorpusBuilder]


_SCHEMA = "mjtensu.mldb-v2/corpus/v1"
_REQUIRED_FIELDS = {
    "schema",
    "id",
    "status",
    "task",
    "description",
    "storage",
    "manifest",
    "representation",
    "splits",
}
_ALLOWED_FIELDS = _REQUIRED_FIELDS | {"builder"}


def _validate_storage(value: object) -> None:
    storage = _require_exact_mapping(value, label="Corpus storage")
    _require_required_fields(storage, {"root_uri"}, label="Corpus storage")
    _validate_s3_root_uri(storage["root_uri"])


def _validate_manifest(value: object, *, local_id: str, sealed: bool) -> None:
    manifest = _require_exact_mapping(value, label="Corpus manifest")
    _require_required_fields(manifest, {"file"}, label="Corpus manifest")
    expected_file = f"{local_id}.manifest.jsonl"
    if manifest["file"] != expected_file:
        raise ValueError("manifest.file must name the same-basename manifest")
    if "sha256" in manifest:
        _validate_sha256(manifest["sha256"], label="manifest sha256")
    if "entries" in manifest:
        _validate_byte_count(manifest["entries"], label="manifest entries")
    if sealed and ("sha256" not in manifest or "entries" not in manifest):
        raise ValueError("sealed Corpus requires manifest sha256 and entries")


def _validate_representation(value: object) -> None:
    representation = _validate_json_mapping(value, label="Corpus representation")
    _require_string(representation.get("kind"), label="Corpus representation.kind", non_empty=True)


def _validate_splits(value: object) -> None:
    splits = _require_exact_mapping(value, label="Corpus splits")
    for split_name, count in splits.items():
        _require_string(split_name, label="Corpus split name", non_empty=True)
        _validate_byte_count(count, label=f"Corpus split {split_name!r} count")


def _validate_builder(value: object, *, sealed: bool) -> None:
    builder = _require_exact_mapping(value, label="Corpus builder")
    _require_required_fields(builder, {"entrypoint"}, label="Corpus builder")
    if builder["entrypoint"] != "build":
        raise ValueError("Corpus builder entrypoint must be build")
    if "sha256" in builder:
        _validate_sha256(builder["sha256"], label="builder sha256")
    if sealed and "sha256" not in builder:
        raise ValueError("sealed Corpus builder requires sha256")
    if "parameters" in builder:
        _validate_json_mapping(builder["parameters"], label="Corpus builder parameters")


def _validate_corpus(value: object, *, expected_id: str | None = None) -> Corpus:
    document = _require_exact_mapping(value, label="Corpus")
    if not _REQUIRED_FIELDS <= set(document) or not set(document) <= _ALLOWED_FIELDS:
        raise ValueError("invalid Corpus top-level fields")
    if document["schema"] != _SCHEMA:
        raise ValueError("unsupported Corpus schema")
    corpus_id = _validate_versioned_entity_id(document["id"], expected_id=expected_id)
    status = _validate_lifecycle(document["status"])
    _validate_versioned_entity_id(document["task"])
    _require_string(document["description"], label="Corpus description")

    _, local_id = corpus_id.split("/", 1)
    _validate_storage(document["storage"])
    _validate_manifest(document["manifest"], local_id=local_id, sealed=status == "sealed")
    _validate_representation(document["representation"])
    _validate_splits(document["splits"])
    if "builder" in document:
        _validate_builder(document["builder"], sealed=status == "sealed")
    return cast(Corpus, document)


def _load_corpus(resolver: CanonicalRepositoryResolver, corpus_id: CorpusId) -> Corpus:
    expected_id = _validate_versioned_entity_id(corpus_id)
    document = resolver.resolve(kind=EntityKind.CORPUS, entity_id=corpus_id)
    return _validate_corpus(document, expected_id=expected_id)
