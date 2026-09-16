from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from mldb_v2.src.catalog.corpus import _load_corpus, _validate_corpus
from mldb_v2.src.catalog.namespace import _load_namespace, _validate_namespace
from mldb_v2.src.catalog.task import _load_task, _validate_task
from mldb_v2.src.common.ids import CorpusId, NamespaceId, TaskId
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver


def _task(*, entity_id: str = "demo/task-v1", status: str = "draft") -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/task/v1",
        "id": entity_id,
        "status": status,
        "name": "Task",
        "problem_type": "classification",
        "description": "description",
        "input": {"semantic_unit": "sample"},
        "target": {"type": "categorical", "labels": ["a", "b"]},
        "semantics": {"flag": True},
        "scope": {"count": 1},
    }


def _corpus(*, entity_id: str = "demo/corpus-v1", status: str = "draft") -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/corpus/v1",
        "id": entity_id,
        "status": status,
        "task": "demo/task-v1",
        "description": "description",
        "storage": {"root_uri": "s3://bucket/corpus-v1"},
        "manifest": {"file": "corpus-v1.manifest.jsonl"},
        "representation": {"kind": "custom", "shape": [1, 2, 3]},
        "splits": {"train": 3, "val": 0},
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_current_namespace_task_corpus_examples_validate() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    resolver = CanonicalRepositoryResolver(repo_root / "mldb_data")
    for namespace, task, corpus in [
        ("tile-classifier", "tile-shape-classification-35-v1", "gray35-jp500-seed42-v3-jp189-v1"),
        ("rotated-fcos", "mahjong-tile-rotated-detection-v1", "mahjong-rotated-detector-320-v1"),
    ]:
        assert _load_namespace(resolver, NamespaceId(namespace))["id"] == namespace
        assert _load_task(resolver, TaskId(f"{namespace}/{task}"))["id"] == f"{namespace}/{task}"
        assert _load_corpus(resolver, CorpusId(f"{namespace}/{corpus}"))["id"] == f"{namespace}/{corpus}"


def test_namespace_contract_and_unknown_field_policy() -> None:
    value = {
        "schema": "mjtensu.mldb-v2/namespace/v1",
        "id": "demo",
        "name": "Demo",
        "description": "description",
        "owner": {"team": "vision"},
    }
    assert _validate_namespace(value, expected_id="demo") is value
    assert value["owner"] == {"team": "vision"}
    assert "status" not in value


@pytest.mark.parametrize(
    "update",
    [
        {"schema": "wrong"},
        {"id": "Bad_Id"},
        {"name": ""},
        {"backend": "clearml"},
        {"execution_id": "opaque"},
    ],
)
def test_namespace_rejects_invalid_core_or_backend_metadata(update: dict[str, object]) -> None:
    value = {
        "schema": "mjtensu.mldb-v2/namespace/v1",
        "id": "demo",
        "name": "Demo",
        "description": "description",
    }
    value.update(update)
    with pytest.raises(ValueError):
        _validate_namespace(value, expected_id="demo")


def test_namespace_path_identity_mismatch_is_rejected() -> None:
    value = {"schema": "mjtensu.mldb-v2/namespace/v1", "id": "demo", "name": "Demo", "description": ""}
    with pytest.raises(ValueError, match="directory identity"):
        _validate_namespace(value, expected_id="other")


@pytest.mark.parametrize("entity_id", ["demo/task", "demo/task-v0", "demo/task-v01", "Demo/task-v1", "demo/task_v1"])
def test_task_requires_canonical_versioned_id(entity_id: str) -> None:
    with pytest.raises(ValueError):
        _validate_task(_task(entity_id=entity_id))


def test_task_rejects_path_id_mismatch_and_unknown_top_level() -> None:
    with pytest.raises(ValueError, match="path identity"):
        _validate_task(_task(), expected_id="demo/other-v1")
    value = _task()
    value["unexpected"] = 1
    with pytest.raises(ValueError, match="top-level"):
        _validate_task(value)


@pytest.mark.parametrize("status", ["", "complete", True])
def test_task_status_is_exact_lifecycle(status: object) -> None:
    value = _task()
    value["status"] = status
    with pytest.raises(ValueError):
        _validate_task(value)


@pytest.mark.parametrize("field", ["name", "problem_type"])
def test_task_requires_non_empty_name_and_problem_type(field: str) -> None:
    value = _task()
    value[field] = ""
    with pytest.raises(ValueError):
        _validate_task(value)


@pytest.mark.parametrize("field", ["input", "target", "semantics", "scope"])
def test_task_semantic_sections_are_mappings(field: str) -> None:
    value = _task()
    value[field] = []
    with pytest.raises(ValueError):
        _validate_task(value)


def test_task_target_type_and_categorical_labels() -> None:
    value = _task()
    labels = value["target"]["labels"]  # type: ignore[index]
    assert _validate_task(value) is value
    assert value["target"]["labels"] is labels  # type: ignore[index]
    for bad_labels in [[], ["a", "a"], ["a", ""], ["a", 1]]:
        candidate = _task()
        candidate["target"]["labels"] = bad_labels  # type: ignore[index]
        with pytest.raises(ValueError):
            _validate_task(candidate)


def test_task_requires_non_empty_target_type() -> None:
    for target in [{}, {"type": ""}, {"type": 1}]:
        value = _task()
        value["target"] = target
        with pytest.raises(ValueError):
            _validate_task(value)


def test_noncategorical_target_remains_open_and_values_preserve_types() -> None:
    value = _task()
    value["target"] = {
        "type": "rotated-object-detection",
        "labels": ["optional-here"],
        "flag": True,
        "count": 1,
        "ratio": 1.0,
        "text": "1",
        "geometry": {"period": 180},
    }
    result = _validate_task(value)
    assert result is value
    target = result["target"]
    assert type(target["flag"]) is bool
    assert type(target["count"]) is int
    assert type(target["ratio"]) is float
    assert type(target["text"]) is str
    assert target["geometry"] == {"period": 180}


def test_task_rejects_non_json_metadata_without_coercion() -> None:
    value = _task()
    value["scope"] = {"bad": math.nan}
    with pytest.raises(ValueError):
        _validate_task(value)


@pytest.mark.parametrize("entity_id", ["demo/corpus", "demo/corpus-v0", "demo/corpus-v01", "demo/corpus_v1"])
def test_corpus_requires_canonical_versioned_id(entity_id: str) -> None:
    with pytest.raises(ValueError):
        _validate_corpus(_corpus(entity_id=entity_id))


def test_corpus_rejects_unknown_top_level_and_path_mismatch() -> None:
    value = _corpus()
    value["unexpected"] = 1
    with pytest.raises(ValueError, match="top-level"):
        _validate_corpus(value)
    with pytest.raises(ValueError, match="path identity"):
        _validate_corpus(_corpus(), expected_id="demo/other-v1")


@pytest.mark.parametrize("status", ["", "complete", 1])
def test_corpus_status_is_exact_lifecycle(status: object) -> None:
    value = _corpus()
    value["status"] = status
    with pytest.raises(ValueError):
        _validate_corpus(value)


@pytest.mark.parametrize("task_ref", ["task-v1", "Demo/task-v1", "demo/task", "demo/task-v0", "demo/task-v01"])
def test_corpus_task_reference_uses_task_identity_grammar(task_ref: str) -> None:
    value = _corpus()
    value["task"] = task_ref
    with pytest.raises(ValueError):
        _validate_corpus(value)


@pytest.mark.parametrize(
    "root_uri",
    [
        "",
        "https://bucket/prefix",
        "s3://user:pass@bucket/prefix",
        "s3://bucket:9000/prefix",
        "s3://bucket/prefix?token=x",
        "s3://bucket/prefix#fragment",
        "s3://bucket\\prefix",
    ],
)
def test_corpus_storage_root_uri_rejects_noncanonical_backend_or_credential_forms(root_uri: str) -> None:
    value = _corpus()
    value["storage"] = {"root_uri": root_uri}
    with pytest.raises(ValueError):
        _validate_corpus(value)


@pytest.mark.parametrize("root_uri", ["s3://bucket", "s3://bucket/prefix", "s3://bucket/a/b"])
def test_corpus_storage_root_uri_accepts_logical_s3_roots_without_normalizing(root_uri: str) -> None:
    value = _corpus()
    value["storage"] = {"root_uri": root_uri}
    assert _validate_corpus(value)["storage"]["root_uri"] == root_uri


def test_corpus_manifest_filename_and_draft_shape() -> None:
    value = _corpus()
    assert _validate_corpus(value) is value
    value["manifest"] = {"file": "wrong.manifest.jsonl"}
    with pytest.raises(ValueError, match="same-basename"):
        _validate_corpus(value)


def test_sealed_corpus_requires_manifest_digest_and_count() -> None:
    value = _corpus(status="sealed")
    with pytest.raises(ValueError, match="sha256 and entries"):
        _validate_corpus(value)
    value["manifest"] = {"file": "corpus-v1.manifest.jsonl", "sha256": "0" * 64, "entries": 0}
    assert _validate_corpus(value) is value


@pytest.mark.parametrize("entries", [True, -1, 1.0, "1"])
def test_corpus_manifest_entries_requires_non_negative_exact_int(entries: object) -> None:
    value = _corpus()
    value["manifest"] = {"file": "corpus-v1.manifest.jsonl", "entries": entries}
    with pytest.raises(ValueError):
        _validate_corpus(value)


def test_corpus_representation_requires_kind_and_preserves_domain_metadata() -> None:
    for representation in [{}, {"kind": ""}, {"kind": 1}]:
        value = _corpus()
        value["representation"] = representation
        with pytest.raises(ValueError):
            _validate_corpus(value)
    value = _corpus()
    metadata = {"kind": "detector-format", "geometry": {"angle_period": 180}, "shape": [3, 320, 320]}
    value["representation"] = metadata
    assert _validate_corpus(value)["representation"] is metadata


@pytest.mark.parametrize("splits", [{"train": True}, {"train": -1}, {"train": 1.0}, {"": 1}])
def test_corpus_splits_validate_names_and_exact_counts(splits: dict[str, object]) -> None:
    value = _corpus()
    value["splits"] = splits
    with pytest.raises(ValueError):
        _validate_corpus(value)


def test_corpus_builder_absent_and_draft_builder_are_valid() -> None:
    value = _corpus()
    assert "builder" not in _validate_corpus(value)
    builder = {"entrypoint": "build", "parameters": {"seed": 42, "flag": True}}
    value["builder"] = builder
    assert _validate_corpus(value)["builder"] is builder


def test_sealed_corpus_builder_requires_valid_hash() -> None:
    value = _corpus(status="sealed")
    value["manifest"] = {"file": "corpus-v1.manifest.jsonl", "sha256": "0" * 64, "entries": 1}
    value["builder"] = {"entrypoint": "build"}
    with pytest.raises(ValueError, match="builder requires sha256"):
        _validate_corpus(value)
    value["builder"] = {"entrypoint": "build", "sha256": "1" * 64}
    assert _validate_corpus(value) is value


def test_corpus_builder_rejects_invalid_entrypoint_hash_and_parameters() -> None:
    for builder in [
        {"entrypoint": "run"},
        {"entrypoint": "build", "sha256": "ABC"},
        {"entrypoint": "build", "parameters": []},
        {"entrypoint": "build", "parameters": {"bad": math.nan}},
    ]:
        value = _corpus()
        value["builder"] = builder
        with pytest.raises(ValueError):
            _validate_corpus(value)


def test_catalog_loaders_use_exact_resolver_without_legacy_or_cross_namespace_fallback(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _write_json(root / "alpha" / "namespace.yaml", {
        "schema": "mjtensu.mldb-v2/namespace/v1", "id": "alpha", "name": "Alpha", "description": ""
    })
    _write_json(root / "beta" / "namespace.yaml", {
        "schema": "mjtensu.mldb-v2/namespace/v1", "id": "beta", "name": "Beta", "description": ""
    })
    _write_json(root / "beta" / "tasks" / "task-v1.yaml", _task(entity_id="beta/task-v1"))
    _write_json(root / "tasks" / "task-v1.yaml", _task(entity_id="alpha/task-v1"))

    resolver = CanonicalRepositoryResolver(root)
    assert _load_task(resolver, TaskId("beta/task-v1"))["id"] == "beta/task-v1"
    with pytest.raises(FileNotFoundError):
        _load_task(resolver, TaskId("alpha/task-v1"))
