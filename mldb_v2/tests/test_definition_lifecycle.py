from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from mldb_v2.src.common.ids import DefinitionKind
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.verification._definition_lifecycle import (
    _RepositoryDefinitionValidator,
    _RepositoryDefinitionVerifier,
)
from mldb_v2.src.verification.definition_lifecycle import (
    DefinitionSealer,
    DefinitionSealingRequest,
    DefinitionSealingResult,
    DefinitionValidationRequest,
    DefinitionValidationResult,
    DefinitionValidator,
    DefinitionVerificationRequest,
    DefinitionVerificationResult,
    DefinitionVerifier,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _namespace(root: Path, namespace: str) -> None:
    _write(root / namespace / "namespace.yaml", {
        "schema": "mjtensu.mldb-v2/namespace/v1",
        "id": namespace,
        "name": namespace,
        "description": "",
    })


def _task(entity_id: str = "demo/task-v1", *, status: str = "sealed") -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/task/v1", "id": entity_id, "status": status,
        "name": "Task", "problem_type": "classification", "description": "",
        "input": {}, "target": {"type": "categorical", "labels": ["a"]},
        "semantics": {}, "scope": {},
    }


def _corpus(entity_id: str = "demo/corpus-v1", *, task: str = "demo/task-v1", status: str = "sealed", digest: str | None = None, entries: int | None = None) -> dict[str, object]:
    local = entity_id.split("/", 1)[1]
    manifest: dict[str, object] = {"file": f"{local}.manifest.jsonl"}
    if status == "sealed":
        manifest["sha256"] = digest if digest is not None else "0" * 64
        manifest["entries"] = 1 if entries is None else entries
    return {
        "schema": "mjtensu.mldb-v2/corpus/v1", "id": entity_id, "status": status,
        "task": task, "description": "", "storage": {"root_uri": f"s3://bucket/{local}"},
        "manifest": manifest, "representation": {"kind": "bytes"}, "splits": {"train": 1},
    }


def _architecture(entity_id: str = "demo/arch-v1", *, task: str = "demo/task-v1", status: str = "sealed") -> dict[str, object]:
    implementation: dict[str, object] = {"framework": "pytorch", "entrypoint": "build"}
    if status == "sealed": implementation["sha256"] = "0" * 64
    return {
        "schema": "mjtensu.mldb-v2/architecture/v1", "id": entity_id, "status": status,
        "task": task, "name": "Arch", "family": "demo", "description": "",
        "implementation": implementation,
        "interface": {"input": {"kind": "x"}, "output": {"kind": "y"}},
        "structure": {"summary": "demo"},
    }


def _train(entity_id: str = "demo/train-v1", *, task: str = "demo/task-v1", status: str = "sealed") -> dict[str, object]:
    implementation: dict[str, object] = {"entrypoint": "train"}
    if status == "sealed": implementation["sha256"] = "0" * 64
    return {
        "schema": "mjtensu.mldb-v2/train-protocol/v1", "id": entity_id, "status": status,
        "task": task, "name": "Train", "description": "", "implementation": implementation,
        "parameters": {
            "epochs": {"default": 2, "type": "integer", "minimum": 1, "maximum": 4},
            "mode": {"default": "a", "type": "string", "enum": ["a", "b"]},
        },
    }


def _evaluation(entity_id: str = "demo/eval-v1", *, task: str = "demo/task-v1", status: str = "sealed") -> dict[str, object]:
    implementation: dict[str, object] = {"entrypoint": "evaluate"}
    if status == "sealed": implementation["sha256"] = "0" * 64
    return {
        "schema": "mjtensu.mldb-v2/evaluation-protocol/v1", "id": entity_id, "status": status,
        "task": task, "name": "Eval", "description": "", "implementation": implementation,
        "parameters": {"threshold": {"default": 0.5, "type": "number", "minimum": 0.0, "maximum": 1.0}},
        "metrics": {"score": {"type": "number", "required": True}}, "artifacts": {},
    }


def _study(*, status: str = "draft", corpus: str = "demo/corpus-v1", protocol: str = "demo/train-v1", architecture: str = "demo/arch-v1", eval_corpus: str = "demo/corpus-v1", eval_protocol: str = "demo/eval-v1") -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/study/v1", "id": "demo/study-v1", "status": status,
        "name": "Study", "description": "",
        "model": {"train": {
            "corpus": corpus, "protocol": protocol, "architectures": [architecture],
            "parameters": {"epochs": {"values": [1, 4]}}, "seeds": [42],
        }},
        "evaluations": [{
            "stage": "holdout", "corpus": eval_corpus, "protocol": eval_protocol,
            "parameters": {"threshold": {"values": [0.25]}},
        }],
    }


class _Transport:
    def __init__(self, objects: dict[str, bytes], *, error: Exception | None = None) -> None:
        self.objects = objects; self.error = error; self.reads: list[str] = []
    def read_bytes(self, uri: str) -> bytes:
        self.reads.append(uri)
        if self.error is not None: raise self.error
        if uri not in self.objects: raise FileNotFoundError(uri)
        return self.objects[uri]
    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        raise AssertionError("verification must not publish")


class _Integrity:
    def __init__(self) -> None:
        self.executable_result: object = {"valid": True, "diagnostics": []}
        self.builder_result: object = {"valid": True, "diagnostics": []}
        self.executable_calls: list[dict[str, object]] = []
        self.builder_calls: list[dict[str, object]] = []
    def verify_executable_definition(self, *, request):
        self.executable_calls.append(dict(request)); return self.executable_result
    def verify_corpus_builder(self, *, request):
        self.builder_calls.append(dict(request)); return self.builder_result


class _Asset:
    def __init__(self) -> None:
        self.result: object = {"valid": True, "run": {"collected": 1, "passed": 1, "failed": 0, "errors": 0}, "diagnostics": []}
        self.calls: list[dict[str, object]] = []
    def verify(self, *, request, runner):
        self.calls.append(dict(request)); return self.result


class _Runner:
    pass


def _manifest(data: bytes = b"abc") -> bytes:
    return (json.dumps({"path": "data.bin", "bytes": len(data), "sha256": _sha(data)}, separators=(",", ":")) + "\n").encode()


def _install_definition(root: Path, kind: str, entity_id: str, document: object, *, companion: bool = False) -> None:
    namespace, local = entity_id.split("/", 1)
    domain = {"task": "tasks", "corpus": "corpora", "architecture": "architectures", "train_protocol": "train_protocols", "evaluation_protocol": "evaluation_protocols", "study": "studies"}[kind]
    _write(root / namespace / domain / f"{local}.yaml", document)
    if companion:
        (root / namespace / domain / f"{local}.py").write_text("# exact companion\n", encoding="utf-8")


def _repo(tmp_path: Path, *, task_status: str = "sealed", definition_status: str = "sealed") -> tuple[Path, Path, Path, _Transport, _Integrity, _Asset]:
    repo = tmp_path / "repo"; root = repo / "mldb_data"; tests = repo / "mldb_tests"; tests.mkdir(parents=True)
    _namespace(root, "demo")
    _install_definition(root, "task", "demo/task-v1", _task(status=task_status))
    data = b"abc"; manifest = _manifest(data); digest = _sha(manifest)
    corpus = _corpus(status=definition_status, digest=digest, entries=1)
    _install_definition(root, "corpus", "demo/corpus-v1", corpus)
    (root / "demo/corpora/corpus-v1.manifest.jsonl").write_bytes(manifest)
    _install_definition(root, "architecture", "demo/arch-v1", _architecture(status=definition_status), companion=True)
    _install_definition(root, "train_protocol", "demo/train-v1", _train(status=definition_status), companion=True)
    _install_definition(root, "evaluation_protocol", "demo/eval-v1", _evaluation(status=definition_status), companion=True)
    _install_definition(root, "study", "demo/study-v1", _study())
    transport = _Transport({"s3://bucket/corpus-v1/data.bin": data})
    return repo, root, tests, transport, _Integrity(), _Asset()


def _verifier(tmp_path: Path, **repo_kwargs):
    repo, root, tests, transport, integrity, asset = _repo(tmp_path, **repo_kwargs)
    interface_calls: list[str] = []
    def arch_loader(root_path, entity_id): interface_calls.append("architecture"); return lambda: None
    def train_loader(root_path, entity_id): interface_calls.append("train_protocol"); return lambda context: (_ for _ in ()).throw(AssertionError("train executed"))
    def eval_loader(root_path, entity_id): interface_calls.append("evaluation_protocol"); return lambda context: (_ for _ in ()).throw(AssertionError("evaluate executed"))
    verifier = _RepositoryDefinitionVerifier(
        repository_root=repo, mldb_data_root=root, mldb_tests_root=tests,
        object_access=_ObjectByteAccess(transport), integrity_verifier=integrity,
        asset_test_verifier=asset, pytest_runner=_Runner(),
        architecture_interface_loader=arch_loader,
        train_interface_loader=train_loader,
        evaluation_interface_loader=eval_loader,
    )
    return verifier, repo, root, tests, transport, integrity, asset, interface_calls


def _codes(result: dict[str, object]) -> list[str]:
    return [item["code"] for item in result["diagnostics"]]


def test_definition_validator_accepts_all_six_draft_kinds_without_external_gates(tmp_path: Path) -> None:
    repo, root, _tests, _transport, _integrity, _asset = _repo(tmp_path, task_status="draft", definition_status="draft")
    validator = _RepositoryDefinitionValidator(root)
    cases = [
        (DefinitionKind.TASK, "demo/task-v1"), (DefinitionKind.CORPUS, "demo/corpus-v1"),
        (DefinitionKind.ARCHITECTURE, "demo/arch-v1"), (DefinitionKind.TRAIN_PROTOCOL, "demo/train-v1"),
        (DefinitionKind.EVALUATION_PROTOCOL, "demo/eval-v1"), (DefinitionKind.STUDY, "demo/study-v1"),
    ]
    for kind, entity_id in cases:
        assert validator.validate(request={"kind": kind, "id": entity_id}) == {"valid": True, "diagnostics": []}


def test_definition_validator_missing_malformed_and_unsupported_fail_cleanly(tmp_path: Path) -> None:
    _repo_path, root, *_ = _repo(tmp_path)
    validator = _RepositoryDefinitionValidator(root)
    assert _codes(validator.validate(request={"kind": DefinitionKind.TASK, "id": "demo/missing-v1"})) == ["definition_not_found"]
    (root / "demo/tasks/task-v1.yaml").write_text("{not valid", encoding="utf-8")
    assert _codes(validator.validate(request={"kind": DefinitionKind.TASK, "id": "demo/task-v1"})) == ["definition_invalid"]
    assert _codes(validator.validate(request={"kind": "namespace", "id": "demo"})) == ["definition_kind_invalid"]


def test_draft_architecture_with_sealed_task_verifies_but_draft_task_blocks(tmp_path: Path) -> None:
    verifier, *_ = _verifier(tmp_path, definition_status="draft")
    assert verifier.verify(request={"kind": DefinitionKind.ARCHITECTURE, "id": "demo/arch-v1"})["valid"] is True
    verifier2, *_rest = _verifier(tmp_path / "b", task_status="draft", definition_status="draft")
    result = verifier2.verify(request={"kind": DefinitionKind.ARCHITECTURE, "id": "demo/arch-v1"})
    assert _codes(result) == ["referenced_definition_not_sealed"]


def test_missing_referenced_task_fails_before_executable_gates(tmp_path: Path) -> None:
    verifier, _repo_path, root, _tests, _transport, integrity, asset, calls = _verifier(tmp_path, definition_status="draft")
    (root / "demo/tasks/task-v1.yaml").unlink()
    result = verifier.verify(request={"kind": DefinitionKind.TRAIN_PROTOCOL, "id": "demo/train-v1"})
    assert _codes(result) == ["referenced_definition_missing"]
    assert integrity.executable_calls == [] and asset.calls == [] and calls == []


def test_training_study_complete_compatibility_and_parameter_declarations(tmp_path: Path) -> None:
    verifier, *_ = _verifier(tmp_path)
    result = verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"})
    assert result == {"valid": True, "diagnostics": []}


@pytest.mark.parametrize(("key", "values", "code"), [
    ("unknown", [1], "parameter_key_unknown"),
    ("epochs", [True], "parameter_value_invalid"),
    ("epochs", [5], "parameter_value_invalid"),
    ("mode", ["c"], "parameter_value_invalid"),
])
def test_training_parameter_axis_declaration_failures(tmp_path: Path, key: str, values: list[object], code: str) -> None:
    verifier, _repo_path, root, *_ = _verifier(tmp_path)
    study = _study(); study["model"]["train"]["parameters"] = {key: {"values": values}}
    _write(root / "demo/studies/study-v1.yaml", study)
    assert code in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))


def test_omitted_protocol_parameter_key_is_valid_without_default_materialization(tmp_path: Path) -> None:
    verifier, _repo_path, root, *_ = _verifier(tmp_path)
    study = _study(); study["model"]["train"]["parameters"] = {}
    _write(root / "demo/studies/study-v1.yaml", study)
    before = json.loads((root / "demo/studies/study-v1.yaml").read_text())
    assert verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"})["valid"] is True
    after = json.loads((root / "demo/studies/study-v1.yaml").read_text())
    assert after == before and after["model"]["train"]["parameters"] == {}


def test_evaluation_parameter_declaration_failures(tmp_path: Path) -> None:
    verifier, _repo_path, root, *_ = _verifier(tmp_path)
    study = _study(); study["evaluations"][0]["parameters"] = {"unknown": {"values": [1]}}
    _write(root / "demo/studies/study-v1.yaml", study)
    assert "parameter_key_unknown" in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))
    study["evaluations"][0]["parameters"] = {"threshold": {"values": [2.0]}}
    _write(root / "demo/studies/study-v1.yaml", study)
    assert "parameter_value_invalid" in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))


def test_training_and_evaluation_task_mismatch_reject(tmp_path: Path) -> None:
    verifier, _repo_path, root, *_ = _verifier(tmp_path)
    _namespace(root, "other"); _install_definition(root, "task", "other/task-v1", _task("other/task-v1"))
    arch = _architecture(task="other/task-v1"); _write(root / "demo/architectures/arch-v1.yaml", arch)
    assert "definition_task_mismatch" in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))


def test_cross_namespace_references_use_task_identity_not_namespace_identity(tmp_path: Path) -> None:
    verifier, repo, root, tests, transport, integrity, asset, calls = _verifier(tmp_path)
    for ns in ["data", "models", "protocols"]: _namespace(root, ns)
    manifest = _manifest(); digest = _sha(manifest)
    _install_definition(root, "corpus", "data/corpus-v1", _corpus("data/corpus-v1", digest=digest))
    (root / "data/corpora/corpus-v1.manifest.jsonl").write_bytes(manifest)
    transport.objects["s3://bucket/corpus-v1/data.bin"] = b"abc"
    _install_definition(root, "architecture", "models/arch-v1", _architecture("models/arch-v1"), companion=True)
    _install_definition(root, "train_protocol", "protocols/train-v1", _train("protocols/train-v1"), companion=True)
    study = _study(corpus="data/corpus-v1", protocol="protocols/train-v1", architecture="models/arch-v1")
    _write(root / "demo/studies/study-v1.yaml", study)
    assert verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"})["valid"] is True


def test_one_referenced_definition_draft_and_missing_reference_reject(tmp_path: Path) -> None:
    verifier, _repo_path, root, *_ = _verifier(tmp_path)
    doc = _evaluation(status="draft"); _write(root / "demo/evaluation_protocols/eval-v1.yaml", doc)
    assert "referenced_definition_not_sealed" in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))
    (root / "demo/evaluation_protocols/eval-v1.yaml").unlink()
    assert "referenced_definition_missing" in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))


def _install_existing_lineage(root: Path, model_id: str, result_id: str, *, task: str = "demo/task-v1", architecture: str = "demo/arch-v1", status: str = "completed", result_model: str | None = None, result: object = "default") -> None:
    _write(root / "demo/models" / f"{model_id.split('/',1)[1]}.yaml", {
        "schema": "mjtensu.mldb-v2/model/v1", "id": model_id, "training_result": result_id,
    })
    payload = {"model": result_model or model_id} if result == "default" else result
    _write(root / "demo/training_results" / f"{result_id.split('/',1)[1]}.yaml", {
        "schema": "mjtensu.mldb-v2/training-result/v1", "id": result_id,
        "status": status, "task": task, "architecture": architecture, "result": payload,
    })


def _existing_study(models: list[str]) -> dict[str, object]:
    value = _study(); value["model"] = {"existing": models}; return value


def test_existing_model_completed_lineage_verifies_without_weight_reads(tmp_path: Path) -> None:
    verifier, _repo_path, root, _tests, transport, *_ = _verifier(tmp_path)
    _install_existing_lineage(root, "demo/model-a", "demo/result-a")
    _install_existing_lineage(root, "demo/model-b", "demo/result-b")
    _write(root / "demo/studies/study-v1.yaml", _existing_study(["demo/model-a", "demo/model-b"]))
    assert verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"})["valid"] is True
    assert set(transport.reads) == {"s3://bucket/corpus-v1/data.bin"}


@pytest.mark.parametrize(("status", "result", "code"), [
    ("failed", None, "training_result_not_completed"),
    ("cancelled", None, "training_result_not_completed"),
    ("completed", None, "model_lineage_invalid"),
])
def test_existing_model_training_result_terminal_lineage_failures(tmp_path: Path, status: str, result: object, code: str) -> None:
    verifier, _repo_path, root, *_ = _verifier(tmp_path)
    _install_existing_lineage(root, "demo/model-a", "demo/result-a", status=status, result=result)
    _write(root / "demo/studies/study-v1.yaml", _existing_study(["demo/model-a"]))
    assert code in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))


def test_existing_model_missing_wrong_shape_result_mismatch_and_task_mismatch(tmp_path: Path) -> None:
    verifier, _repo_path, root, *_ = _verifier(tmp_path)
    _write(root / "demo/studies/study-v1.yaml", _existing_study(["demo/missing-model"]))
    assert "model_lineage_invalid" in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))
    _write(root / "demo/models/model-a.yaml", {"schema": "mjtensu.mldb-v2/model/v1", "id": "demo/model-a", "training_result": "demo/result-a", "extra": 1})
    _write(root / "demo/studies/study-v1.yaml", _existing_study(["demo/model-a"]))
    assert "model_lineage_invalid" in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))
    _install_existing_lineage(root, "demo/model-a", "demo/result-a", result_model="demo/other-model")
    assert "model_lineage_invalid" in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))
    _install_existing_lineage(root, "demo/model-a", "demo/result-a")
    _install_existing_lineage(root, "demo/model-b", "demo/result-b", task="demo/other-task-v1")
    _write(root / "demo/studies/study-v1.yaml", _existing_study(["demo/model-a", "demo/model-b"]))
    assert "definition_task_mismatch" in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))


def test_existing_model_lineage_architecture_draft_and_task_mismatch_reject(tmp_path: Path) -> None:
    verifier, _repo_path, root, *_ = _verifier(tmp_path)
    _install_existing_lineage(root, "demo/model-a", "demo/result-a")
    _write(root / "demo/studies/study-v1.yaml", _existing_study(["demo/model-a"]))
    _write(root / "demo/architectures/arch-v1.yaml", _architecture(status="draft"))
    assert "referenced_definition_not_sealed" in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))
    _namespace(root, "other"); _install_definition(root, "task", "other/task-v1", _task("other/task-v1"))
    _write(root / "demo/architectures/arch-v1.yaml", _architecture(task="other/task-v1"))
    assert "definition_task_mismatch" in _codes(verifier.verify(request={"kind": DefinitionKind.STUDY, "id": "demo/study-v1"}))


def test_corpus_draft_can_verify_and_private_evidence_uses_exact_manifest_bytes(tmp_path: Path) -> None:
    verifier, _repo_path, root, _tests, transport, integrity, *_ = _verifier(tmp_path, definition_status="draft")
    yaml_path = root / "demo/corpora/corpus-v1.yaml"; before = yaml_path.read_bytes()
    result = verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"})
    assert result == {"valid": True, "diagnostics": []}
    evidence = verifier._corpus_sealing_evidence(corpus_id="demo/corpus-v1")
    manifest_bytes = (root / "demo/corpora/corpus-v1.manifest.jsonl").read_bytes()
    assert evidence.manifest_sha256 == _sha(manifest_bytes) and evidence.manifest_entries == 1
    assert yaml_path.read_bytes() == before


def test_corpus_manifest_missing_malformed_digest_and_count_failures(tmp_path: Path) -> None:
    verifier, _repo_path, root, *_ = _verifier(tmp_path)
    path = root / "demo/corpora/corpus-v1.manifest.jsonl"; path.unlink()
    assert "corpus_manifest_invalid" in _codes(verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"}))
    manifest = _manifest(); path.write_bytes(b"not-json\n")
    assert "corpus_manifest_invalid" in _codes(verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"}))
    path.write_bytes(manifest); doc = _corpus(digest="f" * 64, entries=1); _write(root / "demo/corpora/corpus-v1.yaml", doc)
    assert "corpus_manifest_invalid" in _codes(verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"}))
    doc = _corpus(digest=_sha(manifest), entries=2); _write(root / "demo/corpora/corpus-v1.yaml", doc)
    assert "corpus_manifest_invalid" in _codes(verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"}))


def test_corpus_manifest_duplicate_and_unsorted_paths_fail(tmp_path: Path) -> None:
    verifier, _repo_path, root, *_ = _verifier(tmp_path, definition_status="draft")
    for rows in [
        [{"path":"a","bytes":0,"sha256":_sha(b"")},{"path":"a","bytes":0,"sha256":_sha(b"")}],
        [{"path":"b","bytes":0,"sha256":_sha(b"")},{"path":"a","bytes":0,"sha256":_sha(b"")}],
    ]:
        data = b"".join((json.dumps(row,separators=(",",":"))+"\n").encode() for row in rows)
        (root / "demo/corpora/corpus-v1.manifest.jsonl").write_bytes(data)
        assert "corpus_manifest_invalid" in _codes(verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"}))


def test_corpus_object_missing_wrong_bytes_and_transport_error_are_bounded(tmp_path: Path) -> None:
    verifier, _repo_path, _root, _tests, transport, *_ = _verifier(tmp_path)
    transport.objects.clear()
    assert _codes(verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"})) == ["corpus_object_invalid"]
    transport.objects["s3://bucket/corpus-v1/data.bin"] = b"wrong"
    assert _codes(verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"})) == ["corpus_object_invalid"]
    transport.error = RuntimeError("secret backend failure")
    result = verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"})
    assert _codes(result) == ["corpus_object_invalid"] and "secret" not in result["diagnostics"][0]["message"]


def test_corpus_builder_failure_is_delegated_and_bounded(tmp_path: Path) -> None:
    verifier, _repo_path, _root, _tests, _transport, integrity, *_ = _verifier(tmp_path)
    integrity.builder_result = {"valid": False, "diagnostics": [{"code": "builder_hash_mismatch", "message": "mismatch"}]}
    result = verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"})
    assert _codes(result) == ["corpus_builder_invalid", "builder_hash_mismatch"]


def test_executable_gate_order_integrity_interface_asset_and_success(tmp_path: Path) -> None:
    verifier, _repo_path, _root, _tests, _transport, integrity, asset, calls = _verifier(tmp_path, definition_status="draft")
    integrity.executable_result = {"valid": False, "diagnostics": [{"code":"companion_hash_mismatch","message":"bad"}]}
    result = verifier.verify(request={"kind": DefinitionKind.ARCHITECTURE, "id": "demo/arch-v1"})
    assert _codes(result) == ["executable_integrity_failed", "companion_hash_mismatch"] and calls == [] and asset.calls == []
    integrity.executable_result = {"valid": True, "diagnostics": []}
    def bad_loader(root_path, entity_id): raise ValueError("bad")
    verifier._architecture_interface_loader = bad_loader
    assert _codes(verifier.verify(request={"kind": DefinitionKind.ARCHITECTURE, "id": "demo/arch-v1"})) == ["executable_interface_failed"]
    verifier._architecture_interface_loader = lambda root_path, entity_id: object()
    asset.result = {"valid": False, "run": None, "diagnostics": [{"code":"asset_test_failed","message":"bad"}]}
    assert _codes(verifier.verify(request={"kind": DefinitionKind.ARCHITECTURE, "id": "demo/arch-v1"})) == ["executable_asset_tests_failed", "asset_test_failed"]
    asset.result = {"valid": True, "run": {"collected":1,"passed":1,"failed":0,"errors":0}, "diagnostics": []}
    assert verifier.verify(request={"kind": DefinitionKind.ARCHITECTURE, "id": "demo/arch-v1"})["valid"] is True


def test_train_and_evaluate_callables_are_loaded_not_executed(tmp_path: Path) -> None:
    verifier, *_rest, calls = _verifier(tmp_path, definition_status="draft")
    assert verifier.verify(request={"kind": DefinitionKind.TRAIN_PROTOCOL, "id": "demo/train-v1"})["valid"] is True
    assert verifier.verify(request={"kind": DefinitionKind.EVALUATION_PROTOCOL, "id": "demo/eval-v1"})["valid"] is True
    assert "train_protocol" in calls and "evaluation_protocol" in calls


def test_repository_structure_orphan_corpus_python_and_missing_companion_fail(tmp_path: Path) -> None:
    verifier, _repo_path, root, *_ = _verifier(tmp_path, definition_status="draft")
    (root / "demo/architectures/orphan-v1.py").write_text("x=1", encoding="utf-8")
    assert "repository_structure_invalid" in _codes(verifier.verify(request={"kind": DefinitionKind.ARCHITECTURE, "id": "demo/arch-v1"}))
    (root / "demo/architectures/orphan-v1.py").unlink(); (root / "demo/corpora/corpus-v1.py").write_text("x=1", encoding="utf-8")
    assert "repository_structure_invalid" in _codes(verifier.verify(request={"kind": DefinitionKind.CORPUS, "id": "demo/corpus-v1"}))
    (root / "demo/corpora/corpus-v1.py").unlink(); (root / "demo/train_protocols/train-v1.py").unlink()
    assert "repository_structure_invalid" in _codes(verifier.verify(request={"kind": DefinitionKind.TRAIN_PROTOCOL, "id": "demo/train-v1"}))


def test_unrelated_namespace_structure_issue_does_not_poison_requested_namespace(tmp_path: Path) -> None:
    verifier, _repo_path, root, *_ = _verifier(tmp_path, definition_status="draft")
    _namespace(root, "other"); p = root / "other/architectures/orphan-v1.py"; p.parent.mkdir(parents=True); p.write_text("x=1", encoding="utf-8")
    assert verifier.verify(request={"kind": DefinitionKind.ARCHITECTURE, "id": "demo/arch-v1"})["valid"] is True


def test_current_studies_reflect_sealed_dependency_state_without_mutation(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]; root = repo / "mldb_data"; tests = repo / "mldb_tests"
    validator = _RepositoryDefinitionValidator(root)
    class _VerifiedObjects:
        def read_verified(self, ref):
            return b""
    verifier = _RepositoryDefinitionVerifier(
        repository_root=repo, mldb_data_root=root, mldb_tests_root=tests,
        object_access=_VerifiedObjects(),
    )
    classifier_id = "tile-classifier/rotation-robustness-example-v1"
    assert validator.validate(request={"kind": DefinitionKind.STUDY, "id": classifier_id})["valid"] is True
    assert verifier.verify(request={"kind": DefinitionKind.STUDY, "id": classifier_id}) == {"valid": True, "diagnostics": []}

    detector_id = "rotated-fcos/spatial-screen-example-v1"
    assert validator.validate(request={"kind": DefinitionKind.STUDY, "id": detector_id})["valid"] is True
    result = verifier.verify(request={"kind": DefinitionKind.STUDY, "id": detector_id})
    assert result["valid"] is False
    assert "referenced_definition_not_sealed" in _codes(result)


def test_public_definition_lifecycle_shape_is_exact_skeleton_mirror() -> None:
    repo = Path(__file__).resolve().parents[2]
    runtime = (repo / "mldb_v2/src/verification/definition_lifecycle.py").read_text(encoding="utf-8")
    skeleton = (repo / "mldb_v2/skeleton/verification/definition_lifecycle.py").read_text(encoding="utf-8")
    assert runtime.replace("\r\n", "\n") == skeleton.replace("mldb_v2.skeleton.", "mldb_v2.src.").replace("\r\n", "\n")
    assert DefinitionValidationRequest.__required_keys__ == frozenset({"kind", "id"})
    assert DefinitionValidationResult.__required_keys__ == frozenset({"valid", "diagnostics"})
    assert DefinitionVerificationRequest.__required_keys__ == frozenset({"kind", "id"})
    assert DefinitionVerificationResult.__required_keys__ == frozenset({"valid", "diagnostics"})
    assert DefinitionSealingRequest.__required_keys__ == frozenset({"kind", "id"})
    assert DefinitionSealingResult.__required_keys__ == frozenset({"definition"})
    assert list(inspect.signature(DefinitionValidator.validate).parameters) == ["self", "request"]
    assert list(inspect.signature(DefinitionVerifier.verify).parameters) == ["self", "request"]
    assert list(inspect.signature(DefinitionSealer.seal).parameters) == ["self", "request"]
