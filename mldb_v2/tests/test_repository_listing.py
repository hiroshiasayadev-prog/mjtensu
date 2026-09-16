from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mldb_v2.src.common.ids import EntityKind
from mldb_v2.src.repository.listing import CanonicalRepositoryListing


_SCHEMA = {
    EntityKind.TASK: "mjtensu.mldb-v2/task/v1",
    EntityKind.CORPUS: "mjtensu.mldb-v2/corpus/v1",
    EntityKind.ARCHITECTURE: "mjtensu.mldb-v2/architecture/v1",
    EntityKind.STUDY: "mjtensu.mldb-v2/study/v1",
    EntityKind.STUDY_RESULT: "mjtensu.mldb-v2/study-result/v1",
}
_DOMAIN = {
    EntityKind.TASK: "tasks",
    EntityKind.CORPUS: "corpora",
    EntityKind.ARCHITECTURE: "architectures",
    EntityKind.STUDY: "studies",
    EntityKind.STUDY_RESULT: "study_results",
}


def _namespace(root: Path, namespace: str, *, raw: str | None = None) -> None:
    path = root / namespace / "namespace.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is None:
        raw = json.dumps(
            {
                "schema": "mjtensu.mldb-v2/namespace/v1",
                "id": namespace,
                "name": namespace,
                "description": "test namespace",
            }
        )
    path.write_text(raw, encoding="utf-8")


def _entity(
    root: Path,
    namespace: str,
    kind: EntityKind,
    local_id: str,
    *,
    document: dict[str, object] | None = None,
    raw: str | None = None,
) -> Path:
    path = root / namespace / _DOMAIN[kind] / f"{local_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is None:
        value = {"schema": _SCHEMA[kind], "id": f"{namespace}/{local_id}"}
        if document:
            value.update(document)
        raw = json.dumps(value)
    path.write_text(raw, encoding="utf-8")
    return path


def _study_result(
    root: Path,
    namespace: str,
    key_char: str,
    *,
    study: str,
    status: str,
    created_at: str,
) -> str:
    local_id = "run-" + key_char * 32
    _entity(
        root,
        namespace,
        EntityKind.STUDY_RESULT,
        local_id,
        document={"study": study, "status": status, "created_at": created_at},
    )
    return f"{namespace}/{local_id}"


def _codes(listing: dict[str, object]) -> list[str]:
    return [issue["code"] for issue in listing["issues"]]  # type: ignore[index]


def test_two_namespaces_kind_and_lexical_order_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    for namespace in ("beta", "alpha"):
        _namespace(root, namespace)
    _entity(root, "alpha", EntityKind.TASK, "z-task-v1")
    _entity(root, "alpha", EntityKind.TASK, "a-task-v1")
    _entity(root, "alpha", EntityKind.STUDY, "study-v1")
    _entity(root, "beta", EntityKind.TASK, "task-v1")

    listing = CanonicalRepositoryListing(root).list_entities()
    entity_ids = [str(item["id"]) for item in listing["items"] if "/" in str(item["id"])]
    assert entity_ids == [
        "alpha/a-task-v1",
        "alpha/z-task-v1",
        "alpha/study-v1",
        "beta/task-v1",
    ]


def test_absent_empty_domains_are_legal(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.TASK)
    assert listing == {"items": (), "issues": ()}


def test_malformed_namespace_is_issue_and_not_item(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha", raw="schema: [")
    listing = CanonicalRepositoryListing(root).list_entities()
    assert listing["items"] == ()
    assert "repository_yaml_malformed" in _codes(listing)


def test_namespace_path_id_mismatch_is_issue(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(
        root,
        "alpha",
        raw=json.dumps(
            {
                "schema": "mjtensu.mldb-v2/namespace/v1",
                "id": "beta",
                "name": "Alpha",
                "description": "mismatch",
            }
        ),
    )
    listing = CanonicalRepositoryListing(root).list_entities()
    assert listing["items"] == ()
    assert "repository_namespace_id_mismatch" in _codes(listing)


def test_malformed_entity_is_issue_and_excluded(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    _entity(root, "alpha", EntityKind.TASK, "bad-v1", raw="schema: [")
    _entity(root, "alpha", EntityKind.TASK, "good-v1")

    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.TASK)
    assert [item["id"] for item in listing["items"]] == ["alpha/good-v1"]
    assert "repository_yaml_malformed" in _codes(listing)


def test_block_scalar_entity_is_reported_as_malformed_and_excluded(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    _entity(
        root,
        "alpha",
        EntityKind.TASK,
        "bad-v1",
        raw="""schema: mjtensu.mldb-v2/task/v1
id: alpha/bad-v1
description: |
  unsupported
""",
    )

    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.TASK)
    assert listing["items"] == ()
    assert "repository_yaml_malformed" in _codes(listing)


def test_orphan_executable_is_issue(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    domain = root / "alpha" / "architectures"
    domain.mkdir()
    (domain / "orphan-v1.py").write_text("def build(): pass\n", encoding="utf-8")

    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.ARCHITECTURE)
    assert listing["items"] == ()
    assert "repository_orphan_executable" in _codes(listing)


def test_unknown_direct_domain_is_issue(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    (root / "alpha" / "metrics").mkdir()

    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.TASK)
    assert "repository_unknown_domain" in _codes(listing)


def test_namespace_lib_accepts_python_helpers_and_rejects_other_files(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    lib = root / "alpha" / "lib" / "pkg"
    lib.mkdir(parents=True)
    (lib / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.TASK)
    assert "repository_unknown_domain" not in _codes(listing)
    assert "repository_unexpected_source_file" not in _codes(listing)
    (lib / "notes.txt").write_text("not source\n", encoding="utf-8")
    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.TASK)
    assert "repository_unexpected_source_file" in _codes(listing)


def test_invalid_basename_schema_and_id_candidates_are_excluded(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    _entity(root, "alpha", EntityKind.TASK, "good-v1")
    _entity(root, "alpha", EntityKind.TASK, "wrong-schema-v1", document={"schema": "mjtensu.mldb-v2/corpus/v1"})
    _entity(root, "alpha", EntityKind.TASK, "wrong-id-v1", document={"id": "alpha/other-v1"})
    _entity(root, "alpha", EntityKind.TASK, "Bad_name")

    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.TASK)
    assert [item["id"] for item in listing["items"]] == ["alpha/good-v1"]
    assert {
        "repository_invalid_yaml_basename",
        "repository_schema_kind_mismatch",
        "repository_document_id_mismatch",
    }.issubset(set(_codes(listing)))


def test_no_recursive_arbitrary_discovery(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    nested = root / "alpha" / "tasks" / "nested"
    nested.mkdir(parents=True)
    (nested / "hidden-v1.yaml").write_text(
        json.dumps({"schema": "mjtensu.mldb-v2/task/v1", "id": "alpha/hidden-v1"}),
        encoding="utf-8",
    )

    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.TASK)
    assert listing["items"] == ()
    assert "repository_disallowed_subdirectory" in _codes(listing)


def test_disallowed_python_is_issue_without_hiding_yaml(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    task = _entity(root, "alpha", EntityKind.TASK, "task-v1")
    task.with_suffix(".py").write_text("x = 1\n", encoding="utf-8")

    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.TASK)
    assert [item["id"] for item in listing["items"]] == ["alpha/task-v1"]
    assert "repository_disallowed_python" in _codes(listing)


def test_malformed_manifest_companion_placement_is_issue(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    _entity(
        root,
        "alpha",
        EntityKind.CORPUS,
        "corpus-v1",
        document={"manifest": {"file": "wrong.manifest.jsonl"}},
    )
    manifest = root / "alpha" / "corpora" / "wrong.manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")

    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.CORPUS)
    assert [item["id"] for item in listing["items"]] == ["alpha/corpus-v1"]
    assert "repository_manifest_placement_invalid" in _codes(listing)


@pytest.mark.parametrize("filename", ["unexpected.txt", "unexpected.yaml", "unexpected.py", "readme.md"])
def test_unexpected_namespace_root_files_are_structural_issues(tmp_path: Path, filename: str) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    (root / "alpha" / filename).write_text("unexpected\n", encoding="utf-8")

    listing = CanonicalRepositoryListing(root).list_entities()
    assert "repository_unexpected_namespace_entry" in _codes(listing)
    assert any(filename in issue["message"] for issue in listing["issues"])


@pytest.mark.parametrize("filename", ["garbage.txt", "notes.md", "random.json", "random.bin"])
def test_unexpected_domain_files_are_structural_issues(tmp_path: Path, filename: str) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    domain = root / "alpha" / "tasks"
    domain.mkdir()
    (domain / filename).write_text("unexpected\n", encoding="utf-8")

    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.TASK)
    assert "repository_unexpected_domain_file" in _codes(listing)
    assert any(filename in issue["message"] for issue in listing["issues"])


def test_valid_architecture_same_basename_python_remains_valid(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    yaml_path = _entity(root, "alpha", EntityKind.ARCHITECTURE, "arch-v1")
    yaml_path.with_suffix(".py").write_text("def build(): pass\n", encoding="utf-8")

    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.ARCHITECTURE)
    assert [item["id"] for item in listing["items"]] == ["alpha/arch-v1"]
    assert listing["issues"] == ()


def test_valid_corpus_manifest_remains_recognized(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    _entity(root, "alpha", EntityKind.CORPUS, "corpus-v1", document={"manifest": {"file": "corpus-v1.manifest.jsonl"}})
    manifest = root / "alpha" / "corpora" / "corpus-v1.manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")

    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.CORPUS)
    assert [item["id"] for item in listing["items"]] == ["alpha/corpus-v1"]
    assert "repository_unexpected_domain_file" not in _codes(listing)


def test_specific_companion_diagnostics_are_not_duplicated_as_unknown_files(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    tasks = root / "alpha" / "tasks"
    tasks.mkdir()
    (tasks / "orphan.py").write_text("x = 1\n", encoding="utf-8")
    (tasks / "wrong.manifest.jsonl").write_text("{}\n", encoding="utf-8")

    listing = CanonicalRepositoryListing(root).list_entities(kind=EntityKind.TASK)
    codes = _codes(listing)
    assert "repository_disallowed_python" in codes
    assert "repository_disallowed_manifest" in codes
    assert "repository_unexpected_domain_file" not in codes


def test_recognized_domain_and_namespace_yaml_remain_valid_entries(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    (root / "alpha" / "tasks").mkdir()

    listing = CanonicalRepositoryListing(root).list_entities()
    assert [item["id"] for item in listing["items"]] == ["alpha"]
    assert listing["issues"] == ()


def test_legacy_flat_v1_root_remains_excluded_without_v2_issue(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    legacy = root / "tasks"
    legacy.mkdir(parents=True)
    (legacy / "legacy-v1.yaml").write_text("legacy: true\n", encoding="utf-8")

    listing = CanonicalRepositoryListing(root).list_entities()
    assert listing == {"items": (), "issues": ()}


def test_recognized_domain_name_as_file_keeps_specific_issue(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _namespace(root, "alpha")
    (root / "alpha" / "tasks").write_text("not a directory\n", encoding="utf-8")

    listing = CanonicalRepositoryListing(root).list_entities()
    assert "repository_domain_not_directory" in _codes(listing)
    assert "repository_unexpected_namespace_entry" not in _codes(listing)
