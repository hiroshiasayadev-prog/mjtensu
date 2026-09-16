from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from mldb_v2.src.study._planning_preflight import (
    _PlanningPreflightError,
    _StudyPlanningPreflight,
)
from mldb_v2.src.study._source_pinning import (
    _SourcePinningError,
    _StudySourcePinCollector,
    _same_record,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    return result.stdout.decode("ascii").strip()


def _init_git(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "MLDB Test")
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    _write_bytes(repo / "mldb_v2" / "src" / "core.py", b"CORE = 1\n")


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


class _Verifier:
    def verify(self, *, request):
        return {"valid": True, "diagnostics": []}


def _namespace(root: Path, namespace: str) -> None:
    _write_json(
        root / namespace / "namespace.yaml",
        {
            "schema": "mjtensu.mldb-v2/namespace/v1",
            "id": namespace,
            "name": namespace,
            "description": "",
        },
    )


def _task(entity_id: str) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/task/v1",
        "id": entity_id,
        "status": "sealed",
        "name": "Task",
        "problem_type": "classification",
        "description": "",
        "input": {},
        "target": {"type": "categorical", "labels": ["a"]},
        "semantics": {},
        "scope": {},
    }


def _corpus(
    entity_id: str,
    *,
    task: str,
    manifest: bytes,
    builder: bytes | None = None,
) -> dict[str, object]:
    local = entity_id.split("/", 1)[1]
    value: dict[str, object] = {
        "schema": "mjtensu.mldb-v2/corpus/v1",
        "id": entity_id,
        "status": "sealed",
        "task": task,
        "description": "",
        "storage": {"root_uri": f"s3://bucket/{local}"},
        "manifest": {"file": f"{local}.manifest.jsonl", "sha256": _sha(manifest), "entries": len(manifest.splitlines())},
        "representation": {"kind": "bytes"},
        "splits": {"train": 1},
    }
    if builder is not None:
        value["builder"] = {"entrypoint": "build", "sha256": _sha(builder), "parameters": {}}
    return value


def _implementation(entrypoint: str, companion: bytes, sources: list[tuple[str, bytes]]) -> dict[str, object]:
    return {
        "entrypoint": entrypoint,
        "sha256": _sha(companion),
        "sources": [{"path": path, "sha256": _sha(data)} for path, data in sorted(sources)],
    }


def _architecture(entity_id: str, *, task: str, companion: bytes, sources: list[tuple[str, bytes]]) -> dict[str, object]:
    implementation = _implementation("build", companion, sources)
    implementation["framework"] = "pytorch"
    return {
        "schema": "mjtensu.mldb-v2/architecture/v1",
        "id": entity_id,
        "status": "sealed",
        "task": task,
        "name": entity_id,
        "family": "demo",
        "description": "",
        "implementation": implementation,
        "interface": {"input": {"kind": "x"}, "output": {"kind": "y"}},
        "structure": {"summary": "demo"},
    }


def _train_protocol(entity_id: str, *, task: str, companion: bytes, sources: list[tuple[str, bytes]]) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/train-protocol/v1",
        "id": entity_id,
        "status": "sealed",
        "task": task,
        "name": entity_id,
        "description": "",
        "implementation": _implementation("train", companion, sources),
        "parameters": {},
    }


def _evaluation_protocol(entity_id: str, *, task: str, companion: bytes, sources: list[tuple[str, bytes]]) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/evaluation-protocol/v1",
        "id": entity_id,
        "status": "sealed",
        "task": task,
        "name": entity_id,
        "description": "",
        "implementation": _implementation("evaluate", companion, sources),
        "parameters": {},
        "metrics": {"score": {"type": "number", "required": True}},
        "artifacts": {},
    }


def _study_training(*, entity_id: str, corpus: str, protocol: str, architectures: list[str], eval_corpus: str, eval_protocol: str) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/study/v1",
        "id": entity_id,
        "status": "sealed",
        "name": "Study",
        "description": "",
        "model": {"train": {"corpus": corpus, "protocol": protocol, "architectures": architectures, "parameters": {}, "seeds": [7]}},
        "evaluations": [
            {"stage": "eval-a", "corpus": eval_corpus, "protocol": eval_protocol, "parameters": {}},
            {"stage": "eval-b", "corpus": eval_corpus, "protocol": eval_protocol, "parameters": {}},
        ],
    }


def _install_definition(root: Path, kind: str, entity_id: str, document: object, companion: bytes | None = None) -> None:
    namespace, local = entity_id.split("/", 1)
    domain = {
        "task": "tasks",
        "corpus": "corpora",
        "architecture": "architectures",
        "train_protocol": "train_protocols",
        "evaluation_protocol": "evaluation_protocols",
        "study": "studies",
        "model": "models",
        "training_result": "training_results",
    }[kind]
    path = root / namespace / domain / f"{local}.yaml"
    _write_json(path, document)
    if companion is not None:
        _write_bytes(path.with_suffix(".py"), companion)


def _planning(repo: Path, study_id: str):
    return _StudyPlanningPreflight(
        mldb_data_root=repo / "mldb_data", verifier=_Verifier()
    ).prepare(study_id)


def _training_repo(tmp_path: Path) -> tuple[Path, str, str, dict[str, bytes]]:
    repo = tmp_path / "repo"
    _init_git(repo)
    root = repo / "mldb_data"
    ids = {
        "study": "study-ns/study-v1",
        "task": "task-ns/task-v1",
        "train_corpus": "data-ns/train-v1",
        "eval_corpus": "data-ns/eval-v1",
        "arch_z": "arch-ns/arch-z-v1",
        "arch_a": "arch-ns/arch-a-v1",
        "train_protocol": "proto-ns/train-v1",
        "eval_protocol": "proto-ns/eval-v1",
    }
    for namespace in {value.split("/", 1)[0] for value in ids.values()}:
        _namespace(root, namespace)

    source_a = b"VALUE_A = 1\n"
    source_b = b"VALUE_B = 2\n"
    source_common = b"COMMON = 3\n"
    sources = {
        "mldb_data/arch-ns/lib/a.py": source_a,
        "mldb_data/arch-ns/lib/b.py": source_b,
        "mldb_data/proto-ns/lib/common.py": source_common,
    }
    for relative, data in sources.items():
        _write_bytes(repo / relative, data)

    train_manifest = b'{"path":"train.bin","bytes":1,"sha256":"' + b"a" * 64 + b'","split":"train"}\n'
    eval_manifest = b'{"path":"eval.bin","bytes":1,"sha256":"' + b"b" * 64 + b'","split":"train"}\n'
    builder = b"raise RuntimeError('builder executed')\n"
    arch_a_py = b"raise RuntimeError('arch-a executed')\n"
    arch_z_py = b"raise RuntimeError('arch-z executed')\n"
    train_py = b"raise RuntimeError('train executed')\n"
    eval_py = b"raise RuntimeError('eval executed')\n"

    _install_definition(root, "task", ids["task"], _task(ids["task"]))
    _install_definition(root, "corpus", ids["train_corpus"], _corpus(ids["train_corpus"], task=ids["task"], manifest=train_manifest, builder=builder), builder)
    _install_definition(root, "corpus", ids["eval_corpus"], _corpus(ids["eval_corpus"], task=ids["task"], manifest=eval_manifest))
    train_local = ids["train_corpus"].split("/", 1)[1]
    eval_local = ids["eval_corpus"].split("/", 1)[1]
    _write_bytes(root / "data-ns" / "corpora" / f"{train_local}.manifest.jsonl", train_manifest)
    _write_bytes(root / "data-ns" / "corpora" / f"{eval_local}.manifest.jsonl", eval_manifest)

    _install_definition(root, "architecture", ids["arch_z"], _architecture(ids["arch_z"], task=ids["task"], companion=arch_z_py, sources=[("mldb_data/arch-ns/lib/b.py", source_b)]), arch_z_py)
    _install_definition(root, "architecture", ids["arch_a"], _architecture(ids["arch_a"], task=ids["task"], companion=arch_a_py, sources=[("mldb_data/arch-ns/lib/a.py", source_a)]), arch_a_py)

    _install_definition(root, "train_protocol", ids["train_protocol"], _train_protocol(ids["train_protocol"], task=ids["task"], companion=train_py, sources=[("mldb_data/proto-ns/lib/common.py", source_common)]), train_py)
    _install_definition(root, "evaluation_protocol", ids["eval_protocol"], _evaluation_protocol(ids["eval_protocol"], task=ids["task"], companion=eval_py, sources=[("mldb_data/proto-ns/lib/common.py", source_common)]), eval_py)
    _install_definition(
        root,
        "study",
        ids["study"],
        _study_training(
            entity_id=ids["study"],
            corpus=ids["train_corpus"],
            protocol=ids["train_protocol"],
            architectures=[ids["arch_z"], ids["arch_a"]],
            eval_corpus=ids["eval_corpus"],
            eval_protocol=ids["eval_protocol"],
        ),
    )
    (repo / "README.md").write_text("clean\n", encoding="utf-8")
    _write_bytes(repo / "mldb_v2" / "records" / "note.md", b"record clean\n")
    _write_bytes(repo / "mldb_v2" / "tests" / "placeholder.py", b"TEST = 1\n")
    _namespace(root, "unrelated-ns")
    commit = _commit_all(repo, "fixture")
    return repo, ids["study"], commit, {**sources, "builder": builder, "train_manifest": train_manifest, "eval_manifest": eval_manifest}


def _study_existing(*, entity_id: str, models: list[str], eval_corpus: str, eval_protocol: str) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/study/v1",
        "id": entity_id,
        "status": "sealed",
        "name": "Existing Study",
        "description": "",
        "model": {"existing": models},
        "evaluations": [
            {"stage": "eval", "corpus": eval_corpus, "protocol": eval_protocol, "parameters": {}}
        ],
    }


def _existing_repo(tmp_path: Path) -> tuple[Path, str, str, dict[str, str]]:
    repo, _old_study, _old_commit, _ = _training_repo(tmp_path)
    root = repo / "mldb_data"
    ids = {
        "study": "study-ns/existing-v1",
        "model": "model-ns/model-v1",
        "result": "result-ns/result-v1",
        "architecture": "arch-ns/arch-a-v1",
        "historical_corpus": "history-ns/old-corpus-v1",
        "historical_protocol": "history-ns/old-train-v1",
    }
    _namespace(root, "model-ns")
    _namespace(root, "result-ns")
    _namespace(root, "history-ns")

    _install_definition(
        root,
        "model",
        ids["model"],
        {"schema": "mjtensu.mldb-v2/model/v1", "id": ids["model"], "training_result": ids["result"]},
    )
    _install_definition(
        root,
        "training_result",
        ids["result"],
        {
            "schema": "mjtensu.mldb-v2/training-result/v1",
            "id": ids["result"],
            "task": "task-ns/task-v1",
            "architecture": ids["architecture"],
            "corpus": ids["historical_corpus"],
            "train_protocol": ids["historical_protocol"],
            "status": "completed",
        },
    )
    _install_definition(
        root,
        "study",
        ids["study"],
        _study_existing(entity_id=ids["study"], models=[ids["model"]], eval_corpus="data-ns/eval-v1", eval_protocol="proto-ns/eval-v1"),
    )
    commit = _commit_all(repo, "existing fixture")
    return repo, ids["study"], commit, ids


def _collect(repo: Path, study_id: str, commit: str):
    planning = _planning(repo, study_id)
    return _StudySourcePinCollector(
        repository_root=repo, mldb_data_root=repo / "mldb_data"
    ).collect(selected_commit=commit, planning=planning)


def test_training_preflight_to_source_pins_exact_hashes_order_and_dedup(tmp_path: Path) -> None:
    repo, study_id, commit, blobs = _training_repo(tmp_path)
    result = _collect(repo, study_id, commit)

    assert result.source_commit == commit
    keys = [(pin.kind, pin.id) for pin in result.pins]
    assert keys == sorted(keys, key=lambda item: (("namespace", "task", "corpus", "architecture", "train_protocol", "evaluation_protocol", "study", "model", "training_result").index(item[0]), item[1]))
    assert len(keys) == len(set(keys))
    assert ("architecture", "arch-ns/arch-a-v1") in keys
    assert keys.index(("architecture", "arch-ns/arch-a-v1")) < keys.index(("architecture", "arch-ns/arch-z-v1"))
    assert [pin.id for pin in result.pins if pin.kind == "namespace"] == ["arch-ns", "data-ns", "proto-ns", "study-ns", "task-ns"]

    train_corpus = next(pin for pin in result.pins if pin.id == "data-ns/train-v1")
    eval_corpus = next(pin for pin in result.pins if pin.id == "data-ns/eval-v1")
    assert train_corpus.companion_sha256 == _sha(blobs["builder"])
    assert train_corpus.manifest_sha256 == _sha(blobs["train_manifest"])
    assert train_corpus.manifest_entries == 1
    assert eval_corpus.companion_sha256 is None
    assert eval_corpus.manifest_sha256 == _sha(blobs["eval_manifest"])

    arch_a = next(pin for pin in result.pins if pin.id == "arch-ns/arch-a-v1")
    assert [(source.path, source.sha256) for source in arch_a.sources] == [
        ("mldb_data/arch-ns/lib/a.py", _sha(blobs["mldb_data/arch-ns/lib/a.py"]))
    ]
    assert arch_a.manifest_sha256 is None
    study_pin = next(pin for pin in result.pins if pin.id == study_id)
    study_path = repo / "mldb_data" / "study-ns" / "studies" / "study-v1.yaml"
    assert study_pin.yaml_sha256 == _sha(study_path.read_bytes())
    namespace_pin = next(pin for pin in result.pins if pin.id == "study-ns")
    assert namespace_pin.companion_sha256 is None
    assert namespace_pin.sources == ()
    assert namespace_pin.manifest_sha256 is None


def test_collection_is_deterministic_and_pure(tmp_path: Path) -> None:
    repo, study_id, commit, _ = _training_repo(tmp_path)
    before = _git(repo, "status", "--porcelain=v1")
    first = _collect(repo, study_id, commit)
    second = _collect(repo, study_id, commit)
    after = _git(repo, "status", "--porcelain=v1")
    assert first == second
    assert before == after == ""
    assert not list((repo / "mldb_data").rglob("__pycache__"))


@pytest.mark.parametrize(
    "relative",
    [
        "mldb_v2/src/core.py",
        "mldb_data/study-ns/namespace.yaml",
        "mldb_data/study-ns/studies/study-v1.yaml",
        "mldb_data/arch-ns/architectures/arch-a-v1.py",
        "mldb_data/arch-ns/lib/a.py",
        "mldb_data/data-ns/corpora/train-v1.manifest.jsonl",
        "mldb_data/data-ns/corpora/train-v1.py",
    ],
)
def test_dirty_required_inputs_rejected(tmp_path: Path, relative: str) -> None:
    repo, study_id, commit, _ = _training_repo(tmp_path)
    planning = _planning(repo, study_id)
    target = repo / relative
    target.write_bytes(target.read_bytes() + b"dirty\n")
    with pytest.raises(_SourcePinningError) as error:
        _StudySourcePinCollector(repository_root=repo, mldb_data_root=repo / "mldb_data").collect(
            selected_commit=commit, planning=planning
        )
    expected = "mldb_core_source_dirty" if relative.startswith("mldb_v2/src/") else "required_source_dirty"
    assert error.value.code == expected


def test_deleted_core_and_untracked_core_rejected_but_ignored_cache_allowed(tmp_path: Path) -> None:
    repo, study_id, commit, _ = _training_repo(tmp_path)
    planning = _planning(repo, study_id)
    core = repo / "mldb_v2" / "src" / "core.py"
    core.unlink()
    with pytest.raises(_SourcePinningError, match="mldb_core_source_dirty"):
        _StudySourcePinCollector(repository_root=repo, mldb_data_root=repo / "mldb_data").collect(selected_commit=commit, planning=planning)
    core.write_bytes(b"CORE = 1\n")
    _write_bytes(repo / "mldb_v2" / "src" / "new_source.py", b"NEW = 1\n")
    with pytest.raises(_SourcePinningError, match="mldb_core_source_dirty"):
        _StudySourcePinCollector(repository_root=repo, mldb_data_root=repo / "mldb_data").collect(selected_commit=commit, planning=planning)
    (repo / "mldb_v2" / "src" / "new_source.py").unlink()
    _write_bytes(repo / "mldb_v2" / "src" / "__pycache__" / "ignored.pyc", b"cache")
    _StudySourcePinCollector(repository_root=repo, mldb_data_root=repo / "mldb_data").collect(selected_commit=commit, planning=planning)


def test_unrelated_dirty_files_are_allowed(tmp_path: Path) -> None:
    repo, study_id, commit, _ = _training_repo(tmp_path)
    planning = _planning(repo, study_id)
    (repo / "README.md").write_text("dirty readme\n", encoding="utf-8")
    (repo / "mldb_v2" / "records" / "note.md").write_text("dirty record\n", encoding="utf-8")
    (repo / "mldb_v2" / "tests" / "placeholder.py").write_text("TEST = 2\n", encoding="utf-8")
    ns = repo / "mldb_data" / "unrelated-ns" / "namespace.yaml"
    ns.write_bytes(ns.read_bytes() + b"\n")
    _write_bytes(repo / "app" / "scratch.py", b"untracked = True\n")
    result = _StudySourcePinCollector(repository_root=repo, mldb_data_root=repo / "mldb_data").collect(
        selected_commit=commit, planning=planning
    )
    assert result.source_commit == commit


def test_existing_model_pins_model_result_and_lineage_architecture_only(tmp_path: Path) -> None:
    repo, study_id, commit, ids = _existing_repo(tmp_path)
    result = _collect(repo, study_id, commit)
    keys = {(pin.kind, pin.id) for pin in result.pins}
    assert ("model", ids["model"]) in keys
    assert ("training_result", ids["result"]) in keys
    assert ("architecture", ids["architecture"]) in keys
    assert all(pin.id != ids["historical_corpus"] for pin in result.pins)
    assert all(pin.id != ids["historical_protocol"] for pin in result.pins)
    assert "history-ns" not in {pin.id for pin in result.pins if pin.kind == "namespace"}


@pytest.mark.parametrize("kind", ["model", "result"])
def test_dirty_existing_model_records_rejected(tmp_path: Path, kind: str) -> None:
    repo, study_id, commit, ids = _existing_repo(tmp_path)
    planning = _planning(repo, study_id)
    entity_id = ids["model"] if kind == "model" else ids["result"]
    namespace, local = entity_id.split("/", 1)
    domain = "models" if kind == "model" else "training_results"
    path = repo / "mldb_data" / namespace / domain / f"{local}.yaml"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(_SourcePinningError, match="required_source_dirty"):
        _StudySourcePinCollector(repository_root=repo, mldb_data_root=repo / "mldb_data").collect(selected_commit=commit, planning=planning)


def test_invalid_abbreviated_and_missing_commits_rejected(tmp_path: Path) -> None:
    repo, study_id, commit, _ = _training_repo(tmp_path)
    planning = _planning(repo, study_id)
    collector = _StudySourcePinCollector(repository_root=repo, mldb_data_root=repo / "mldb_data")
    for selected in (commit[:12], "0" * 40):
        with pytest.raises(_SourcePinningError, match="invalid_commit"):
            collector.collect(selected_commit=selected, planning=planning)


def test_selected_non_head_commit_succeeds_when_required_set_is_unchanged(tmp_path: Path) -> None:
    repo, study_id, commit_a, _ = _training_repo(tmp_path)
    (repo / "README.md").write_text("later unrelated commit\n", encoding="utf-8")
    commit_b = _commit_all(repo, "unrelated later commit")
    assert commit_b != commit_a
    result = _collect(repo, study_id, commit_a)
    assert result.source_commit == commit_a


def test_selected_non_head_commit_rejects_required_working_tree_difference(tmp_path: Path) -> None:
    repo, study_id, commit_a, _ = _training_repo(tmp_path)
    study_path = repo / "mldb_data" / "study-ns" / "studies" / "study-v1.yaml"
    study_path.write_bytes(study_path.read_bytes() + b"\n")
    _commit_all(repo, "required later commit")
    planning = _planning(repo, study_id)
    with pytest.raises(_SourcePinningError, match="required_source_dirty"):
        _StudySourcePinCollector(repository_root=repo, mldb_data_root=repo / "mldb_data").collect(selected_commit=commit_a, planning=planning)


def test_required_file_absent_at_selected_commit_rejected(tmp_path: Path) -> None:
    repo, study_id, commit_a, blobs = _training_repo(tmp_path)
    root = repo / "mldb_data"
    new_id = "arch-ns/arch-new-v1"
    companion = b"raise RuntimeError('new arch executed')\n"
    _install_definition(
        root,
        "architecture",
        new_id,
        _architecture(new_id, task="task-ns/task-v1", companion=companion, sources=[]),
        companion,
    )
    study_path = root / "study-ns" / "studies" / "study-v1.yaml"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["model"]["train"]["architectures"] = [new_id]
    _write_json(study_path, study)
    _commit_all(repo, "add required architecture")
    planning = _planning(repo, study_id)
    with pytest.raises(_SourcePinningError, match="required_source_missing"):
        _StudySourcePinCollector(repository_root=repo, mldb_data_root=root).collect(
            selected_commit=commit_a, planning=planning
        )


def test_committed_companion_hash_must_match_sealed_digest(tmp_path: Path) -> None:
    repo, study_id, _commit, _ = _training_repo(tmp_path)
    path = repo / "mldb_data" / "arch-ns" / "architectures" / "arch-a-v1.yaml"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["implementation"]["sha256"] = "f" * 64
    _write_json(path, value)
    commit = _commit_all(repo, "bad companion seal")
    with pytest.raises(_SourcePinningError, match="committed_hash_mismatch"):
        _collect(repo, study_id, commit)


def test_committed_declared_source_hash_must_match_definition(tmp_path: Path) -> None:
    repo, study_id, _commit, _ = _training_repo(tmp_path)
    path = repo / "mldb_data" / "arch-ns" / "architectures" / "arch-a-v1.yaml"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["implementation"]["sources"][0]["sha256"] = "e" * 64
    _write_json(path, value)
    commit = _commit_all(repo, "bad source seal")
    with pytest.raises(_SourcePinningError, match="committed_hash_mismatch"):
        _collect(repo, study_id, commit)


def test_manifest_digest_and_count_must_match_sealed_corpus(tmp_path: Path) -> None:
    repo, study_id, _commit, _ = _training_repo(tmp_path)
    path = repo / "mldb_data" / "data-ns" / "corpora" / "train-v1.yaml"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["manifest"]["entries"] = 2
    _write_json(path, value)
    commit = _commit_all(repo, "bad manifest count")
    with pytest.raises(_SourcePinningError, match="corpus_manifest_mismatch"):
        _collect(repo, study_id, commit)


def test_builder_digest_must_match_sealed_corpus(tmp_path: Path) -> None:
    repo, study_id, _commit, _ = _training_repo(tmp_path)
    path = repo / "mldb_data" / "data-ns" / "corpora" / "train-v1.yaml"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["builder"]["sha256"] = "d" * 64
    _write_json(path, value)
    commit = _commit_all(repo, "bad builder seal")
    with pytest.raises(_SourcePinningError, match="committed_hash_mismatch"):
        _collect(repo, study_id, commit)


def test_stale_study_planning_input_rejected_against_selected_commit(tmp_path: Path) -> None:
    repo, study_id, _commit_a, _ = _training_repo(tmp_path)
    planning_a = _planning(repo, study_id)
    study_path = repo / "mldb_data" / "study-ns" / "studies" / "study-v1.yaml"
    study_b = json.loads(study_path.read_text(encoding="utf-8"))
    study_b["model"]["train"]["architectures"] = ["arch-ns/arch-a-v1"]
    _write_json(study_path, study_b)
    commit_b = _commit_all(repo, "study graph B")

    with pytest.raises(_SourcePinningError) as error:
        _StudySourcePinCollector(
            repository_root=repo, mldb_data_root=repo / "mldb_data"
        ).collect(selected_commit=commit_b, planning=planning_a)
    assert error.value.code == "planning_input_stale"


def test_stale_train_protocol_default_rejected_against_selected_commit(tmp_path: Path) -> None:
    repo, study_id, _initial, _ = _training_repo(tmp_path)
    protocol_path = repo / "mldb_data" / "proto-ns" / "train_protocols" / "train-v1.yaml"
    protocol_a = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_a["parameters"] = {"batch_size": {"default": 32, "type": "integer"}}
    _write_json(protocol_path, protocol_a)
    _commit_all(repo, "protocol A")
    planning_a = _planning(repo, study_id)

    protocol_b = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_b["parameters"]["batch_size"]["default"] = 64
    _write_json(protocol_path, protocol_b)
    commit_b = _commit_all(repo, "protocol B")

    with pytest.raises(_SourcePinningError) as error:
        _StudySourcePinCollector(
            repository_root=repo, mldb_data_root=repo / "mldb_data"
        ).collect(selected_commit=commit_b, planning=planning_a)
    assert error.value.code == "planning_input_stale"


def test_stale_existing_training_result_rejected_against_selected_commit(tmp_path: Path) -> None:
    repo, study_id, _commit_a, ids = _existing_repo(tmp_path)
    planning_a = _planning(repo, study_id)
    result_path = repo / "mldb_data" / "result-ns" / "training_results" / "result-v1.yaml"
    result_b = json.loads(result_path.read_text(encoding="utf-8"))
    result_b["status"] = "failed"
    _write_json(result_path, result_b)
    commit_b = _commit_all(repo, "training result B")

    with pytest.raises(_SourcePinningError) as error:
        _StudySourcePinCollector(
            repository_root=repo, mldb_data_root=repo / "mldb_data"
        ).collect(selected_commit=commit_b, planning=planning_a)
    assert error.value.code == "planning_input_stale"


def test_record_binding_comparison_is_type_sensitive_and_key_order_independent() -> None:
    assert _same_record({"a": 1, "b": [2]}, {"b": [2], "a": 1})
    assert not _same_record({"value": True}, {"value": 1})
    assert not _same_record({"value": 1}, {"value": 1.0})
