from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from mldb_v2.src.common.ids import DefinitionKind
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.verification._definition_lifecycle import (
    _RepositoryDefinitionValidator,
    _RepositoryDefinitionVerifier,
)
from mldb_v2.src.verification._definition_sealing import _RepositoryDefinitionSealer
from mldb_v2.src.verification._executable_integrity import (
    _RepositoryExecutableIntegrityVerifier,
)
from mldb_v2.src.verification.definition_lifecycle import DefinitionSealer


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _namespace(root: Path, namespace: str = "demo") -> None:
    _write(root / namespace / "namespace.yaml", {
        "schema": "mjtensu.mldb-v2/namespace/v1", "id": namespace,
        "name": namespace, "description": "",
    })

def _task(entity_id: str = "demo/task-v1", *, status: str = "draft") -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/task/v1", "id": entity_id, "status": status,
        "name": "Task", "problem_type": "classification", "description": "",
        "input": {"unit": "tile"},
        "target": {"type": "categorical", "labels": ["a", "b"]},
        "semantics": {"order": ["a", "b"]}, "scope": {"mode": "demo"},
    }


def _corpus(
    *, status: str = "draft", builder: bool = False, manifest_sha: str | None = None,
    manifest_entries: int | None = None, builder_sha: str | None = None,
    task: str = "demo/task-v1",
) -> dict[str, object]:
    manifest: dict[str, object] = {"file": "corpus-v1.manifest.jsonl"}
    if manifest_sha is not None: manifest["sha256"] = manifest_sha
    if manifest_entries is not None: manifest["entries"] = manifest_entries
    value: dict[str, object] = {
        "schema": "mjtensu.mldb-v2/corpus/v1", "id": "demo/corpus-v1",
        "status": status, "task": task, "description": "Corpus",
        "storage": {"root_uri": "s3://bucket/corpus-v1"}, "manifest": manifest,
        "representation": {"kind": "bytes", "channels": ["x", "y"]},
        "splits": {"train": 1},
    }
    if builder:
        b: dict[str, object] = {"entrypoint": "build", "parameters": {"seed": 42}}
        if builder_sha is not None: b["sha256"] = builder_sha
        value["builder"] = b
    return value

def _architecture(*, status: str = "draft", task: str = "demo/task-v1", source_sha: str | None = None) -> dict[str, object]:
    implementation: dict[str, object] = {"framework": "pytorch", "entrypoint": "build"}
    if source_sha is not None:
        implementation["sources"] = [{"path": "product/demo_source.py", "sha256": source_sha}]
    return {
        "schema": "mjtensu.mldb-v2/architecture/v1", "id": "demo/arch-v1",
        "status": status, "task": task, "name": "Arch", "family": "demo",
        "description": "Architecture", "implementation": implementation,
        "interface": {"input": {"kind": "x"}, "output": {"kind": "y"}},
        "structure": {"summary": "demo", "traits": ["small", "fast"]},
        "parameters": {"note": "authored"},
    }


def _train(*, status: str = "draft", task: str = "demo/task-v1") -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/train-protocol/v1", "id": "demo/train-v1",
        "status": status, "task": task, "name": "Train", "description": "Train",
        "implementation": {"entrypoint": "train"},
        "parameters": {"epochs": {"default": 1, "type": "integer", "minimum": 1}},
    }


def _evaluation(*, status: str = "draft", task: str = "demo/task-v1") -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/evaluation-protocol/v1", "id": "demo/eval-v1",
        "status": status, "task": task, "name": "Eval", "description": "Eval",
        "implementation": {"entrypoint": "evaluate"}, "parameters": {},
        "metrics": {"score": {"type": "number", "required": True}}, "artifacts": {},
    }

def _study(*, status: str = "draft", architecture: str = "demo/arch-v1") -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/study/v1", "id": "demo/study-v1", "status": status,
        "name": "Study", "description": "Study",
        "model": {"train": {
            "corpus": "demo/corpus-v1", "protocol": "demo/train-v1",
            "architectures": [architecture], "parameters": {"epochs": {"values": [1]}},
            "seeds": [42],
        }},
        "evaluations": [{
            "stage": "holdout", "corpus": "demo/corpus-v1", "protocol": "demo/eval-v1",
            "parameters": {},
        }],
    }


def _manifest(object_bytes: bytes = b"abc") -> bytes:
    row = {"path": "data.bin", "bytes": len(object_bytes), "sha256": _sha(object_bytes)}
    return (json.dumps(row, separators=(",", ":")) + "\n").encode()


class _Transport:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
    def read_bytes(self, uri: str) -> bytes:
        if uri not in self.objects: raise FileNotFoundError(uri)
        return self.objects[uri]
    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        raise AssertionError("sealing verification must not publish")


class _Runner:
    def __init__(self, result: dict[str, int] | None = None) -> None:
        self.result = result or {"collected": 1, "passed": 1, "failed": 0, "errors": 0}
    def run(self, *, test_directory: Path):
        return dict(self.result)

def _install_exec(root: Path, domain: str, local_id: str, document: dict[str, object], companion: bytes) -> None:
    _write(root / "demo" / domain / f"{local_id}.yaml", document)
    path = root / "demo" / domain / f"{local_id}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(companion)


def _repo(tmp_path: Path, *, builder: bool = False, runner: _Runner | None = None):
    repo = tmp_path / "repo"; root = repo / "mldb_data"; tests = repo / "mldb_tests"
    tests.mkdir(parents=True); _namespace(root)
    source_bytes = b"VALUE = 1\n"; source_path = repo / "product/demo_source.py"
    source_path.parent.mkdir(parents=True); source_path.write_bytes(source_bytes)
    _write(root / "demo/tasks/task-v1.yaml", _task())
    manifest = _manifest(); _write(root / "demo/corpora/corpus-v1.yaml", _corpus(builder=builder))
    (root / "demo/corpora/corpus-v1.manifest.jsonl").write_bytes(manifest)
    if builder:
        (root / "demo/corpora/corpus-v1.py").write_bytes(b"def build():\n    return None\n")
    companions = {
        "architectures": ("arch-v1", _architecture(source_sha=_sha(source_bytes)), b"def build():\n    return None\n"),
        "train_protocols": ("train-v1", _train(), b"def train(context):\n    return context\n"),
        "evaluation_protocols": ("eval-v1", _evaluation(), b"def evaluate(context):\n    return context\n"),
    }
    for domain, (local_id, document, companion) in companions.items():
        _install_exec(root, domain, local_id, document, companion)
        (tests / "demo" / domain / local_id).mkdir(parents=True)
    _write(root / "demo/studies/study-v1.yaml", _study())
    transport = _Transport({"s3://bucket/corpus-v1/data.bin": b"abc"})
    integrity = _RepositoryExecutableIntegrityVerifier(repo, root)
    verifier = _RepositoryDefinitionVerifier(
        repository_root=repo, mldb_data_root=root, mldb_tests_root=tests,
        object_access=_ObjectByteAccess(transport), integrity_verifier=integrity,
        pytest_runner=runner or _Runner(),
        architecture_interface_loader=lambda _r, _i: object(),
        train_interface_loader=lambda _r, _i: object(),
        evaluation_interface_loader=lambda _r, _i: object(),
    )
    sealer = _RepositoryDefinitionSealer(
        repository_root=repo, mldb_data_root=root, mldb_tests_root=tests,
        definition_verifier=verifier, integrity_verifier=integrity,
    )
    return repo, root, tests, transport, integrity, verifier, sealer


def _path(root: Path, kind: DefinitionKind) -> Path:
    domain = {
        DefinitionKind.TASK: "tasks", DefinitionKind.CORPUS: "corpora",
        DefinitionKind.ARCHITECTURE: "architectures", DefinitionKind.TRAIN_PROTOCOL: "train_protocols",
        DefinitionKind.EVALUATION_PROTOCOL: "evaluation_protocols", DefinitionKind.STUDY: "studies",
    }[kind]
    local = {
        DefinitionKind.TASK: "task-v1", DefinitionKind.CORPUS: "corpus-v1",
        DefinitionKind.ARCHITECTURE: "arch-v1", DefinitionKind.TRAIN_PROTOCOL: "train-v1",
        DefinitionKind.EVALUATION_PROTOCOL: "eval-v1", DefinitionKind.STUDY: "study-v1",
    }[kind]
    return root / "demo" / domain / f"{local}.yaml"


def _id(kind: DefinitionKind) -> str:
    return {
        DefinitionKind.TASK: "demo/task-v1", DefinitionKind.CORPUS: "demo/corpus-v1",
        DefinitionKind.ARCHITECTURE: "demo/arch-v1", DefinitionKind.TRAIN_PROTOCOL: "demo/train-v1",
        DefinitionKind.EVALUATION_PROTOCOL: "demo/eval-v1", DefinitionKind.STUDY: "demo/study-v1",
    }[kind]


def _seal(sealer: _RepositoryDefinitionSealer, kind: DefinitionKind):
    return sealer.seal(request={"kind": kind, "id": _id(kind)})["definition"]


def _seal_dependencies(sealer: _RepositoryDefinitionSealer) -> None:
    for kind in [DefinitionKind.TASK, DefinitionKind.CORPUS, DefinitionKind.ARCHITECTURE,
                 DefinitionKind.TRAIN_PROTOCOL, DefinitionKind.EVALUATION_PROTOCOL]:
        _seal(sealer, kind)

def test_private_sealer_matches_frozen_public_call_shape() -> None:
    assert list(inspect.signature(DefinitionSealer.seal).parameters) == ["self", "request"]
    assert list(inspect.signature(_RepositoryDefinitionSealer.seal).parameters) == ["self", "request"]


@pytest.mark.parametrize("sealing_request", [
    {}, {"kind": "namespace", "id": "demo"}, {"kind": "model", "id": "demo/model-v1"},
    {"kind": "task", "id": "bad"}, {"kind": "task", "id": "demo/task"},
])
def test_malformed_or_non_definition_request_rejected_without_write(tmp_path: Path, sealing_request: dict[str, object]) -> None:
    _repo_path, root, *_rest, sealer = _repo(tmp_path)
    before = _path(root, DefinitionKind.TASK).read_bytes()
    with pytest.raises(ValueError): sealer.seal(request=sealing_request)  # type: ignore[arg-type]
    assert _path(root, DefinitionKind.TASK).read_bytes() == before


def test_missing_and_invalid_definition_leave_bytes_unchanged(tmp_path: Path) -> None:
    _repo_path, root, _tests, _transport, _integrity, _verifier, sealer = _repo(tmp_path)
    with pytest.raises(FileNotFoundError):
        sealer.seal(request={"kind": DefinitionKind.TASK, "id": "demo/missing-v1"})
    path = _path(root, DefinitionKind.TASK); doc = json.loads(path.read_text())
    doc["extra"] = 1; _write(path, doc); before = path.read_bytes()
    with pytest.raises(ValueError, match="verification failed"):
        _seal(sealer, DefinitionKind.TASK)
    assert path.read_bytes() == before


def test_task_seal_changes_only_status_and_reresolves_exact_content(tmp_path: Path) -> None:
    _repo_path, root, *_rest, sealer = _repo(tmp_path)
    path = _path(root, DefinitionKind.TASK); before = json.loads(path.read_text())
    sealed = _seal(sealer, DefinitionKind.TASK)
    after = json.loads(path.read_text())
    expected = copy.deepcopy(before); expected["status"] = "sealed"
    assert after == expected == sealed
    assert _RepositoryDefinitionValidator(root).validate(
        request={"kind": DefinitionKind.TASK, "id": "demo/task-v1"})["valid"] is True


def test_already_sealed_replay_is_lifecycle_conflict_and_never_mutates(tmp_path: Path) -> None:
    _repo_path, root, *_rest, sealer = _repo(tmp_path)
    _seal(sealer, DefinitionKind.TASK)
    path = _path(root, DefinitionKind.TASK); before = path.read_bytes()
    with pytest.raises(ValueError, match="not a draft"):
        _seal(sealer, DefinitionKind.TASK)
    assert path.read_bytes() == before


def test_corpus_without_builder_seals_exact_manifest_evidence_and_invents_nothing(tmp_path: Path) -> None:
    _repo_path, root, _tests, _transport, _integrity, verifier, sealer = _repo(tmp_path)
    _seal(sealer, DefinitionKind.TASK)
    manifest = (root / "demo/corpora/corpus-v1.manifest.jsonl").read_bytes()
    sealed = _seal(sealer, DefinitionKind.CORPUS)
    assert sealed["manifest"]["sha256"] == _sha(manifest)
    assert sealed["manifest"]["entries"] == 1
    assert "builder" not in sealed
    assert verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"})["valid"] is True


def test_corpus_builder_seals_exact_builder_digest_and_preserves_authored_mapping(tmp_path: Path) -> None:
    _repo_path, root, _tests, _transport, _integrity, verifier, sealer = _repo(tmp_path, builder=True)
    _seal(sealer, DefinitionKind.TASK)
    path = root / "demo/corpora/corpus-v1.py"; expected = _sha(path.read_bytes())
    before = json.loads(_path(root, DefinitionKind.CORPUS).read_text())
    sealed = _seal(sealer, DefinitionKind.CORPUS)
    assert sealed["builder"]["sha256"] == expected
    assert sealed["builder"]["parameters"] == before["builder"]["parameters"]
    assert verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"})["valid"] is True


def test_matching_draft_corpus_integrity_is_accepted_and_mismatch_is_rejected(tmp_path: Path) -> None:
    _repo_path, root, _tests, _transport, _integrity, _verifier, sealer = _repo(tmp_path, builder=True)
    _seal(sealer, DefinitionKind.TASK)
    manifest = (root / "demo/corpora/corpus-v1.manifest.jsonl").read_bytes()
    builder = (root / "demo/corpora/corpus-v1.py").read_bytes()
    _write(_path(root, DefinitionKind.CORPUS), _corpus(
        builder=True, manifest_sha=_sha(manifest), manifest_entries=1, builder_sha=_sha(builder)))
    assert _seal(sealer, DefinitionKind.CORPUS)["status"] == "sealed"


def test_mismatching_corpus_recorded_evidence_or_object_failure_preserves_draft(tmp_path: Path) -> None:
    _repo_path, root, _tests, transport, _integrity, _verifier, sealer = _repo(tmp_path, builder=True)
    _seal(sealer, DefinitionKind.TASK)
    path = _path(root, DefinitionKind.CORPUS)
    bad = _corpus(builder=True, manifest_sha="f" * 64, manifest_entries=1)
    _write(path, bad); before = path.read_bytes()
    with pytest.raises(ValueError, match="verification failed"):
        _seal(sealer, DefinitionKind.CORPUS)
    assert path.read_bytes() == before

    _write(path, _corpus(builder=True)); transport.objects.clear(); before = path.read_bytes()
    with pytest.raises(ValueError, match="verification failed"):
        _seal(sealer, DefinitionKind.CORPUS)
    assert path.read_bytes() == before


def test_mismatching_builder_hash_preserves_draft(tmp_path: Path) -> None:
    _repo_path, root, *_rest, sealer = _repo(tmp_path, builder=True)
    _seal(sealer, DefinitionKind.TASK)
    path = _path(root, DefinitionKind.CORPUS)
    _write(path, _corpus(builder=True, builder_sha="0" * 64)); before = path.read_bytes()
    with pytest.raises(ValueError, match="verification failed"):
        _seal(sealer, DefinitionKind.CORPUS)
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", [
    DefinitionKind.ARCHITECTURE, DefinitionKind.TRAIN_PROTOCOL,
    DefinitionKind.EVALUATION_PROTOCOL,
])
def test_executable_seal_records_exact_companion_digest_and_preserves_semantics(tmp_path: Path, kind: DefinitionKind) -> None:
    _repo_path, root, _tests, _transport, _integrity, verifier, sealer = _repo(tmp_path)
    _seal(sealer, DefinitionKind.TASK)
    path = _path(root, kind); before = json.loads(path.read_text())
    companion = path.with_suffix(".py"); expected_sha = _sha(companion.read_bytes())
    sealed = _seal(sealer, kind); after = json.loads(path.read_text())
    expected = copy.deepcopy(before); expected["status"] = "sealed"
    expected["implementation"]["sha256"] = expected_sha
    assert after == expected == sealed
    assert verifier.verify(request={"kind": kind, "id": _id(kind)})["valid"] is True


def test_executable_sources_preserved_exactly(tmp_path: Path) -> None:
    _repo_path, root, *_rest, sealer = _repo(tmp_path)
    _seal(sealer, DefinitionKind.TASK)
    before = json.loads(_path(root, DefinitionKind.ARCHITECTURE).read_text())
    sealed = _seal(sealer, DefinitionKind.ARCHITECTURE)
    assert sealed["implementation"]["sources"] == before["implementation"]["sources"]


def test_referenced_task_not_sealed_blocks_executable_without_write(tmp_path: Path) -> None:
    _repo_path, root, *_rest, sealer = _repo(tmp_path)
    path = _path(root, DefinitionKind.ARCHITECTURE); before = path.read_bytes()
    with pytest.raises(ValueError, match="verification failed"):
        _seal(sealer, DefinitionKind.ARCHITECTURE)
    assert path.read_bytes() == before


def test_executable_integrity_failure_preserves_draft(tmp_path: Path) -> None:
    _repo_path, root, *_rest, sealer = _repo(tmp_path)
    _seal(sealer, DefinitionKind.TASK)
    path = _path(root, DefinitionKind.TRAIN_PROTOCOL); doc = json.loads(path.read_text())
    doc["implementation"]["sha256"] = "0" * 64; _write(path, doc); before = path.read_bytes()
    with pytest.raises(ValueError, match="verification failed"):
        _seal(sealer, DefinitionKind.TRAIN_PROTOCOL)
    assert path.read_bytes() == before


def test_executable_asset_failure_preserves_draft(tmp_path: Path) -> None:
    failed = _Runner({"collected": 1, "passed": 0, "failed": 1, "errors": 0})
    _repo_path, root, *_rest, sealer = _repo(tmp_path, runner=failed)
    _seal(sealer, DefinitionKind.TASK)
    path = _path(root, DefinitionKind.EVALUATION_PROTOCOL); before = path.read_bytes()
    with pytest.raises(ValueError, match="verification failed"):
        _seal(sealer, DefinitionKind.EVALUATION_PROTOCOL)
    assert path.read_bytes() == before


def test_study_seal_changes_only_status_after_complete_dependency_lifecycle(tmp_path: Path) -> None:
    _repo_path, root, _tests, _transport, _integrity, verifier, sealer = _repo(tmp_path)
    _seal_dependencies(sealer)
    path = _path(root, DefinitionKind.STUDY); before = json.loads(path.read_text())
    sealed = _seal(sealer, DefinitionKind.STUDY)
    expected = copy.deepcopy(before); expected["status"] = "sealed"
    assert json.loads(path.read_text()) == expected == sealed
    assert set(sealed) == set(before)
    assert verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"})["valid"] is True


def test_study_task_incompatibility_preserves_draft(tmp_path: Path) -> None:
    _repo_path, root, _tests, _transport, _integrity, _verifier, sealer = _repo(tmp_path)
    _seal(sealer, DefinitionKind.TASK)
    _write(root / "other/namespace.yaml", {
        "schema": "mjtensu.mldb-v2/namespace/v1", "id": "other", "name": "other", "description": ""})
    _write(root / "other/tasks/task-v1.yaml", _task("other/task-v1", status="sealed"))
    arch_path = _path(root, DefinitionKind.ARCHITECTURE)
    arch = json.loads(arch_path.read_text()); arch["task"] = "other/task-v1"; _write(arch_path, arch)
    _seal(sealer, DefinitionKind.CORPUS)
    _seal(sealer, DefinitionKind.ARCHITECTURE)
    _seal(sealer, DefinitionKind.TRAIN_PROTOCOL)
    _seal(sealer, DefinitionKind.EVALUATION_PROTOCOL)
    path = _path(root, DefinitionKind.STUDY); before = path.read_bytes()
    with pytest.raises(ValueError, match="verification failed"):
        _seal(sealer, DefinitionKind.STUDY)
    assert path.read_bytes() == before


def test_authorized_draft_byte_change_is_detected_and_newer_bytes_survive(tmp_path: Path) -> None:
    repo, root, tests, _transport, integrity, verifier, _sealer = _repo(tmp_path)
    path = _path(root, DefinitionKind.TASK); original = json.loads(path.read_text())
    newer = copy.deepcopy(original); newer["description"] = "newer draft"

    class MutatingVerifier:
        _integrity = integrity
        def verify(self, *, request):
            result = verifier.verify(request=request)
            _write(path, newer)
            return result

    sealer = _RepositoryDefinitionSealer(
        repository_root=repo, mldb_data_root=root, mldb_tests_root=tests,
        definition_verifier=MutatingVerifier(), integrity_verifier=integrity)
    with pytest.raises(ValueError, match="became stale"):
        _seal(sealer, DefinitionKind.TASK)
    assert json.loads(path.read_text()) == newer


def test_atomic_replace_failure_preserves_exact_draft_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _repo_path, root, *_rest, sealer = _repo(tmp_path)
    path = _path(root, DefinitionKind.TASK); before = path.read_bytes()
    def fail(_path_value, _record): raise OSError("injected")
    monkeypatch.setattr("mldb_v2.src.verification._definition_sealing._atomic_replace_record", fail)
    with pytest.raises(OSError, match="injected"):
        _seal(sealer, DefinitionKind.TASK)
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".*.tmp"))


def test_proposed_sealed_validation_happens_before_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _repo_path, root, *_rest, sealer = _repo(tmp_path)
    _seal(sealer, DefinitionKind.TASK)
    path = _path(root, DefinitionKind.TRAIN_PROTOCOL); before = path.read_bytes()
    monkeypatch.setattr(sealer, "_executable_sha256", lambda **_kwargs: "bad")
    with pytest.raises(ValueError):
        _seal(sealer, DefinitionKind.TRAIN_PROTOCOL)
    assert path.read_bytes() == before


def test_sealed_definition_cannot_be_overwritten_by_later_stale_transition(tmp_path: Path) -> None:
    _repo_path, root, *_rest, sealer = _repo(tmp_path)
    path = _path(root, DefinitionKind.TASK); old_draft = path.read_bytes()
    first = _seal(sealer, DefinitionKind.TASK)
    sealed_bytes = path.read_bytes()
    assert first["status"] == "sealed" and sealed_bytes != old_draft
    with pytest.raises(ValueError, match="not a draft"):
        _seal(sealer, DefinitionKind.TASK)
    assert path.read_bytes() == sealed_bytes


def test_all_six_kind_end_to_end_seal_validate_verify_and_filesystem_hygiene(tmp_path: Path) -> None:
    repo, root, _tests, _transport, _integrity, verifier, sealer = _repo(tmp_path, builder=True)
    kinds = [
        DefinitionKind.TASK, DefinitionKind.CORPUS, DefinitionKind.ARCHITECTURE,
        DefinitionKind.TRAIN_PROTOCOL, DefinitionKind.EVALUATION_PROTOCOL, DefinitionKind.STUDY,
    ]
    validator = _RepositoryDefinitionValidator(root)
    for kind in kinds:
        result = _seal(sealer, kind)
        assert result["status"] == "sealed"
        assert validator.validate(request={"kind": kind, "id": _id(kind)}) == {
            "valid": True, "diagnostics": []}
        assert verifier.verify(request={"kind": kind, "id": _id(kind)}) == {
            "valid": True, "diagnostics": []}

    assert not list(root.rglob("__pycache__"))
    assert not list(root.rglob("*.tmp"))
    assert not list(root.rglob("*.lock"))
    assert not list(root.rglob("study_plans"))
    lock_root = repo / ".local/mldb_v2/definition_seal_locks"
    assert lock_root.is_dir() and list(lock_root.glob("*.lock"))
