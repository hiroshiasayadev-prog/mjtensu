from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
from pathlib import Path

import pytest

from mldb_v2.src.common.diagnostic import _validate_diagnostic
from mldb_v2.src.verification import _executable_integrity as integrity_impl
from mldb_v2.src.verification._executable_integrity import (
    _RepositoryExecutableIntegrityVerifier,
)
from mldb_v2.src.verification.executable_integrity import (
    CorpusBuilderIntegrityRequest,
    ExecutableDefinitionIntegrityRequest,
    ExecutableIntegrityResult,
    ExecutableIntegrityVerifier,
    ExecutableSource,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    root = repo / "mldb_data"
    _write_json(
        root / "demo" / "namespace.yaml",
        {
            "schema": "mjtensu.mldb-v2/namespace/v1",
            "id": "demo",
            "name": "Demo",
            "description": "",
        },
    )
    return repo, root


def _implementation(
    kind: str, *, sha256: str | None = None, sources: list[dict[str, str]] | None = None
) -> dict[str, object]:
    if kind == "architecture":
        value: dict[str, object] = {"framework": "pytorch", "entrypoint": "build"}
    elif kind == "train_protocol":
        value = {"entrypoint": "train"}
    else:
        value = {"entrypoint": "evaluate"}
    if sha256 is not None:
        value["sha256"] = sha256
    if sources is not None:
        value["sources"] = sources
    return value
def _executable_document(
    kind: str,
    *,
    status: str = "draft",
    sha256: str | None = None,
    sources: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    implementation = _implementation(kind, sha256=sha256, sources=sources)
    common = {
        "id": f"demo/{'model' if kind == 'architecture' else 'train' if kind == 'train_protocol' else 'eval'}-v1",
        "status": status,
        "task": "demo/task-v1",
        "name": "Definition",
        "description": "description",
        "implementation": implementation,
    }
    if kind == "architecture":
        return {
            "schema": "mjtensu.mldb-v2/architecture/v1",
            **common,
            "family": "demo",
            "interface": {"input": {"kind": "image"}, "output": {"kind": "logits"}},
            "structure": {"summary": "summary"},
        }
    if kind == "train_protocol":
        return {"schema": "mjtensu.mldb-v2/train-protocol/v1", **common, "parameters": {}}
    return {
        "schema": "mjtensu.mldb-v2/evaluation-protocol/v1",
        **common,
        "parameters": {},
        "metrics": {"score": {"type": "number", "required": True}},
        "artifacts": {},
    }
_DOMAIN = {
    "architecture": "architectures",
    "train_protocol": "train_protocols",
    "evaluation_protocol": "evaluation_protocols",
}
_LOCAL = {"architecture": "model-v1", "train_protocol": "train-v1", "evaluation_protocol": "eval-v1"}


def _install_executable(
    root: Path,
    kind: str,
    document: dict[str, object],
    *,
    companion: bytes | None = b"VALUE = 1\n",
) -> Path:
    directory = root / "demo" / _DOMAIN[kind]
    _write_json(directory / f"{_LOCAL[kind]}.yaml", document)
    path = directory / f"{_LOCAL[kind]}.py"
    if companion is not None:
        path.write_bytes(companion)
    return path


def _request(kind: str) -> ExecutableDefinitionIntegrityRequest:
    return {"kind": kind, "id": f"demo/{_LOCAL[kind]}"}  # type: ignore[typeddict-item]


def _corpus(*, status: str = "draft", builder: dict[str, object] | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "mjtensu.mldb-v2/corpus/v1",
        "id": "demo/corpus-v1",
        "status": status,
        "task": "demo/task-v1",
        "description": "description",
        "storage": {"root_uri": "s3://bucket/corpus-v1"},
        "manifest": {"file": "corpus-v1.manifest.jsonl"},
        "representation": {"kind": "custom"},
        "splits": {"train": 1},
    }
    if status == "sealed":
        value["manifest"] = {
            "file": "corpus-v1.manifest.jsonl",
            "sha256": "0" * 64,
            "entries": 1,
        }
    if builder is not None:
        value["builder"] = builder
    return value


def _install_corpus(
    root: Path,
    document: dict[str, object],
    *,
    builder_bytes: bytes | None = None,
) -> Path:
    directory = root / "demo" / "corpora"
    _write_json(directory / "corpus-v1.yaml", document)
    path = directory / "corpus-v1.py"
    if builder_bytes is not None:
        path.write_bytes(builder_bytes)
    return path


def _verifier(repo: Path, root: Path) -> _RepositoryExecutableIntegrityVerifier:
    return _RepositoryExecutableIntegrityVerifier(repo, root)


@pytest.mark.parametrize("kind", ["architecture", "train_protocol", "evaluation_protocol"])
@pytest.mark.parametrize(
    ("status", "recorded", "expected_valid", "expected_code"),
    [
        ("draft", "omit", True, None),
        ("draft", "correct", True, None),
        ("draft", "wrong", False, "companion_hash_mismatch"),
        ("sealed", "correct", True, None),
        ("sealed", "wrong", False, "companion_hash_mismatch"),
    ],
)
def test_executable_draft_and_sealed_hash_semantics(
    tmp_path: Path, kind: str, status: str, recorded: str, expected_valid: bool, expected_code: str | None
) -> None:
    repo, root = _make_repo(tmp_path)
    companion = b"VALUE = 'exact companion bytes'\n"
    if recorded == "omit":
        digest = None
    elif recorded == "correct":
        digest = _sha(companion)
    else:
        digest = "f" * 64
    document = _executable_document(kind, status=status, sha256=digest)
    _install_executable(root, kind, document, companion=companion)

    result = _verifier(repo, root).verify_executable_definition(request=_request(kind))

    assert result["valid"] is expected_valid
    assert [diagnostic["code"] for diagnostic in result["diagnostics"]] == (
        [] if expected_code is None else [expected_code]
    )


@pytest.mark.parametrize(("mode", "code"), [("missing", "companion_missing"), ("directory", "companion_invalid")])
def test_executable_companion_missing_or_nonregular(tmp_path: Path, mode: str, code: str) -> None:
    repo, root = _make_repo(tmp_path)
    path = _install_executable(root, "architecture", _executable_document("architecture"), companion=None)
    if mode == "directory":
        path.mkdir(parents=True)
    result = _verifier(repo, root).verify_executable_definition(request=_request("architecture"))
    assert result == {"valid": False, "diagnostics": [{"code": code, "message": result["diagnostics"][0]["message"]}]}


def test_executable_definition_missing_malformed_and_wrong_dispatch_fail_cleanly(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    verifier = _verifier(repo, root)
    missing = verifier.verify_executable_definition(request=_request("architecture"))
    assert [item["code"] for item in missing["diagnostics"]] == ["definition_not_found"]

    _install_executable(root, "architecture", _executable_document("architecture"))
    yaml_path = root / "demo" / "architectures" / "model-v1.yaml"
    yaml_path.write_text("{not valid", encoding="utf-8")
    malformed = verifier.verify_executable_definition(request=_request("architecture"))
    assert [item["code"] for item in malformed["diagnostics"]] == ["definition_invalid"]
    _write_json(yaml_path, _executable_document("architecture"))
    wrong_kind_request = {"kind": "train_protocol", "id": "demo/model-v1"}
    wrong_kind = verifier.verify_executable_definition(request=wrong_kind_request)  # type: ignore[arg-type]
    assert [item["code"] for item in wrong_kind["diagnostics"]] == ["definition_not_found"]

    bad_kind_request = {"kind": "bogus", "id": "demo/model-v1"}
    bad_kind = verifier.verify_executable_definition(request=bad_kind_request)  # type: ignore[arg-type]
    assert [item["code"] for item in bad_kind["diagnostics"]] == ["definition_invalid"]


def test_declared_same_namespace_lib_source_verifies_exact_bytes(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    source_bytes = b"VALUE = 7\n"
    source = root / "demo" / "lib" / "behavior.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(source_bytes)
    sources = [{"path": "mldb_data/demo/lib/behavior.py", "sha256": _sha(source_bytes)}]
    _install_executable(
        root, "architecture", _executable_document("architecture", sources=sources),
        companion=b"from ..lib.behavior import VALUE\nRESULT = VALUE\n",
    )
    verifier = _verifier(repo, root)
    assert verifier.verify_executable_definition(request=_request("architecture")) == {"valid": True, "diagnostics": []}
    evidence = verifier._executable_sealing_evidence(request=_request("architecture"))
    assert evidence.sources == (("mldb_data/demo/lib/behavior.py", _sha(source_bytes)),)
    source.write_bytes(b"VALUE = 8\n")
    result = verifier.verify_executable_definition(request=_request("architecture"))
    assert [item["code"] for item in result["diagnostics"]] == ["source_hash_mismatch"]


def test_empty_sources_remain_valid_and_sealing_evidence_is_empty(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    companion = b"VALUE = 'companion evidence'\n"
    _install_executable(root, "architecture", _executable_document("architecture", sources=[]), companion=companion)
    verifier = _verifier(repo, root)
    assert verifier.verify_executable_definition(request=_request("architecture")) == {"valid": True, "diagnostics": []}
    assert verifier._executable_sealing_evidence(request=_request("architecture")).sources == ()


def test_corpus_without_builder_requires_no_same_basename_python(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    path = _install_corpus(root, _corpus())
    verifier = _verifier(repo, root)
    result = verifier.verify_corpus_builder(request={"corpus": "demo/corpus-v1"})
    assert result == {"valid": True, "diagnostics": []}
    assert verifier._corpus_builder_sealing_evidence(
        request={"corpus": "demo/corpus-v1"}
    ).builder_sha256 is None

    path.write_bytes(b"unowned\n")
    result = verifier.verify_corpus_builder(request={"corpus": "demo/corpus-v1"})
    assert [item["code"] for item in result["diagnostics"]] == ["builder_unowned"]


@pytest.mark.parametrize(
    ("status", "recorded", "expected_valid", "expected_code"),
    [
        ("draft", "omit", True, None),
        ("draft", "correct", True, None),
        ("draft", "wrong", False, "builder_hash_mismatch"),
        ("sealed", "correct", True, None),
        ("sealed", "wrong", False, "builder_hash_mismatch"),
    ],
)
def test_corpus_builder_draft_and_sealed_hash_semantics(
    tmp_path: Path, status: str, recorded: str, expected_valid: bool, expected_code: str | None
) -> None:
    repo, root = _make_repo(tmp_path)
    builder_bytes = b"raise RuntimeError('must never execute')\n"
    if recorded == "omit":
        digest = None
    elif recorded == "correct":
        digest = _sha(builder_bytes)
    else:
        digest = "f" * 64
    builder: dict[str, object] = {"entrypoint": "build"}
    if digest is not None:
        builder["sha256"] = digest
    yaml_path = root / "demo" / "corpora" / "corpus-v1.yaml"
    _install_corpus(root, _corpus(status=status, builder=builder), builder_bytes=builder_bytes)
    before = yaml_path.read_bytes()
    verifier = _verifier(repo, root)
    result = verifier.verify_corpus_builder(request={"corpus": "demo/corpus-v1"})
    assert result["valid"] is expected_valid
    assert [item["code"] for item in result["diagnostics"]] == (
        [] if expected_code is None else [expected_code]
    )
    if expected_valid:
        evidence = verifier._corpus_builder_sealing_evidence(
            request={"corpus": "demo/corpus-v1"}
        )
        assert evidence.builder_sha256 == _sha(builder_bytes)
    assert yaml_path.read_bytes() == before


@pytest.mark.parametrize(("mode", "code"), [("missing", "builder_missing"), ("directory", "builder_invalid")])
def test_corpus_builder_missing_or_nonregular(tmp_path: Path, mode: str, code: str) -> None:
    repo, root = _make_repo(tmp_path)
    path = _install_corpus(root, _corpus(builder={"entrypoint": "build"}))
    if mode == "directory":
        path.mkdir(parents=True)
    result = _verifier(repo, root).verify_corpus_builder(request={"corpus": "demo/corpus-v1"})
    assert [item["code"] for item in result["diagnostics"]] == [code]


def test_corpus_definition_missing_and_invalid_fail_cleanly(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    verifier = _verifier(repo, root)
    missing = verifier.verify_corpus_builder(request={"corpus": "demo/corpus-v1"})
    assert [item["code"] for item in missing["diagnostics"]] == ["definition_not_found"]
    _install_corpus(root, _corpus())
    yaml_path = root / "demo" / "corpora" / "corpus-v1.yaml"
    yaml_path.write_text("not: [valid", encoding="utf-8")
    invalid = verifier.verify_corpus_builder(request={"corpus": "demo/corpus-v1"})
    assert [item["code"] for item in invalid["diagnostics"]] == ["definition_invalid"]
@pytest.mark.parametrize(
    "sources",
    [
        [
            {"path": "mldb_data/demo/lib/b.py", "sha256": "0" * 64},
            {"path": "mldb_data/demo/lib/a.py", "sha256": "1" * 64},
        ],
        [
            {"path": "mldb_data/demo/lib/a.py", "sha256": "0" * 64},
            {"path": "mldb_data/demo/lib/a.py", "sha256": "1" * 64},
        ],
    ],
)
def test_unsorted_or_duplicate_sources_fail_through_typed_parser(
    tmp_path: Path, sources: list[dict[str, str]]
) -> None:
    repo, root = _make_repo(tmp_path)
    _install_executable(
        root,
        "architecture",
        _executable_document("architecture", sources=sources),
    )
    result = _verifier(repo, root).verify_executable_definition(request=_request("architecture"))
    assert [item["code"] for item in result["diagnostics"]] == ["definition_invalid"]


def test_unreadable_companion_and_builder_are_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, root = _make_repo(tmp_path)
    _install_executable(root, "architecture", _executable_document("architecture"))
    real_hash = integrity_impl._sha256_file

    def fail_companion(path: Path) -> str:
        if path.name == "model-v1.py":
            raise PermissionError("denied")
        return real_hash(path)

    monkeypatch.setattr(integrity_impl, "_sha256_file", fail_companion)
    result = _verifier(repo, root).verify_executable_definition(request=_request("architecture"))
    assert [item["code"] for item in result["diagnostics"]] == ["companion_unreadable"]

    builder_bytes = b"builder\n"
    _install_corpus(root, _corpus(builder={"entrypoint": "build"}), builder_bytes=builder_bytes)

    def fail_builder(path: Path) -> str:
        if path.name == "corpus-v1.py":
            raise PermissionError("denied")
        return real_hash(path)

    monkeypatch.setattr(integrity_impl, "_sha256_file", fail_builder)
    result = _verifier(repo, root).verify_corpus_builder(request={"corpus": "demo/corpus-v1"})
    assert [item["code"] for item in result["diagnostics"]] == ["builder_unreadable"]


def test_diagnostics_are_common_valid_and_codes_are_stable_lowercase(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    _install_executable(root, "architecture", _executable_document("architecture"), companion=None)
    result = _verifier(repo, root).verify_executable_definition(request=_request("architecture"))
    assert result["diagnostics"]
    for diagnostic in result["diagnostics"]:
        assert _validate_diagnostic(diagnostic) == diagnostic
        assert re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", diagnostic["code"])
def test_current_draft_examples_verify_and_establish_companion_digest_without_git_mutation() -> None:
    repo = Path(__file__).resolve().parents[2]
    root = repo / "mldb_data"
    verifier = _verifier(repo, root)
    before = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True
    ).stdout
    cases = [
        ("architecture", "tile-classifier/tile-plain-gray35-v2"),
        ("architecture", "rotated-fcos/rotated-fcos-s05-rgb-baseline-v1"),
        ("architecture", "rotated-fcos/rotated-fcos-s05-rgb-nopool-v1"),
        ("architecture", "rotated-fcos/rotated-fcos-s05-rgb-p2-v1"),
        ("architecture", "rotated-fcos/rotated-fcos-s05-rgb-stem-s1-v1"),
        ("train_protocol", "tile-classifier/tile-shape-train-gpu-v3"),
        ("train_protocol", "rotated-fcos/rotated-fcos-train-gpu-v2"),
        ("evaluation_protocol", "tile-classifier/tile-shape-angle-robustness-v1"),
        ("evaluation_protocol", "rotated-fcos/detector-crop-quality-v1"),
        ("evaluation_protocol", "rotated-fcos/rotated-fcos-standard-v1"),
    ]
    for kind, entity_id in cases:
        request = {"kind": kind, "id": entity_id}
        result = verifier.verify_executable_definition(request=request)  # type: ignore[arg-type]
        assert result == {"valid": True, "diagnostics": []}
        evidence = verifier._executable_sealing_evidence(request=request)  # type: ignore[arg-type]
        namespace, local_id = entity_id.split("/", 1)
        companion = root / namespace / _DOMAIN[kind] / f"{local_id}.py"
        assert evidence.companion_sha256 == _sha(companion.read_bytes())
        assert evidence.sources == ()
    for corpus_id in [
        "tile-classifier/gray35-jp500-seed42-v3-jp189-v1",
        "rotated-fcos/mahjong-rotated-detector-320-v1",
    ]:
        result = verifier.verify_corpus_builder(request={"corpus": corpus_id})  # type: ignore[typeddict-item]
        assert result == {"valid": True, "diagnostics": []}
        assert verifier._corpus_builder_sealing_evidence(
            request={"corpus": corpus_id}  # type: ignore[typeddict-item]
        ).builder_sha256 is None
    after = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True
    ).stdout
    assert after == before


def test_public_executable_integrity_shape_remains_exact_skeleton_mirror() -> None:
    repo = Path(__file__).resolve().parents[2]
    runtime = (repo / "mldb_v2" / "src" / "verification" / "executable_integrity.py").read_text(
        encoding="utf-8"
    )
    skeleton = (repo / "mldb_v2" / "skeleton" / "verification" / "executable_integrity.py").read_text(
        encoding="utf-8"
    )
    normalized_skeleton = skeleton.replace("mldb_v2.skeleton.", "mldb_v2.src.")
    assert runtime.replace("\r\n", "\n") == normalized_skeleton.replace("\r\n", "\n")

    assert ExecutableSource.__required_keys__ == frozenset({"path", "sha256"})
    assert ExecutableDefinitionIntegrityRequest.__required_keys__ == frozenset({"kind", "id"})
    assert CorpusBuilderIntegrityRequest.__required_keys__ == frozenset({"corpus"})
    assert ExecutableIntegrityResult.__required_keys__ == frozenset({"valid", "diagnostics"})
    assert list(inspect.signature(ExecutableIntegrityVerifier.verify_executable_definition).parameters) == [
        "self", "request"
    ]
    assert list(inspect.signature(ExecutableIntegrityVerifier.verify_corpus_builder).parameters) == [
        "self", "request"
    ]


def test_undeclared_direct_project_owned_import_is_rejected(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    source = repo / "product" / "behavior.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 7\n")
    companion = b"from product.behavior import VALUE\nRESULT = VALUE\n"
    _install_executable(
        root,
        "architecture",
        _executable_document("architecture"),
        companion=companion,
    )
    result = _verifier(repo, root).verify_executable_definition(
        request=_request("architecture")
    )
    assert result["valid"] is False
    assert [item["code"] for item in result["diagnostics"]] == [
        "source_import_forbidden"
    ]
    assert "product/behavior.py" in result["diagnostics"][0]["message"]


def test_declaring_project_owned_source_cannot_bypass_self_contained_contract(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    source_bytes = b"VALUE = 7\n"
    source = repo / "product" / "behavior.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(source_bytes)
    _install_executable(
        root,
        "train_protocol",
        _executable_document(
            "train_protocol",
            sources=[{"path": "product/behavior.py", "sha256": _sha(source_bytes)}],
        ),
        companion=b"from product.behavior import VALUE\nRESULT = VALUE\n",
    )
    result = _verifier(repo, root).verify_executable_definition(request=_request("train_protocol"))
    assert result["valid"] is False
    assert [item["code"] for item in result["diagnostics"]] == ["definition_invalid"]


def test_same_namespace_lib_import_must_be_declared(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    helper = root / "demo" / "lib" / "helper.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_bytes(b"VALUE = 7\n")
    _install_executable(
        root, "evaluation_protocol", _executable_document("evaluation_protocol"),
        companion=b"from ..lib.helper import VALUE\nRESULT = VALUE\n",
    )
    result = _verifier(repo, root).verify_executable_definition(request=_request("evaluation_protocol"))
    assert [item["code"] for item in result["diagnostics"]] == ["source_declaration_missing"]
    assert "mldb_data/demo/lib/helper.py" in result["diagnostics"][0]["message"]


def test_transitive_same_namespace_lib_import_must_also_be_declared(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    behavior = b"from .helper import OFFSET\nVALUE = OFFSET + 1\n"
    helper = b"OFFSET = 6\n"
    lib = root / "demo" / "lib"
    lib.mkdir(parents=True)
    (lib / "behavior.py").write_bytes(behavior)
    (lib / "helper.py").write_bytes(helper)
    declared = [{"path": "mldb_data/demo/lib/behavior.py", "sha256": _sha(behavior)}]
    _install_executable(
        root, "train_protocol", _executable_document("train_protocol", sources=declared),
        companion=b"from ..lib.behavior import VALUE\nRESULT = VALUE\n",
    )
    verifier = _verifier(repo, root)
    missing = verifier.verify_executable_definition(request=_request("train_protocol"))
    assert [item["code"] for item in missing["diagnostics"]] == ["source_declaration_missing"]
    assert "mldb_data/demo/lib/helper.py" in missing["diagnostics"][0]["message"]
    all_sources = [
        {"path": "mldb_data/demo/lib/behavior.py", "sha256": _sha(behavior)},
        {"path": "mldb_data/demo/lib/helper.py", "sha256": _sha(helper)},
    ]
    _install_executable(
        root, "train_protocol", _executable_document("train_protocol", sources=all_sources),
        companion=b"from ..lib.behavior import VALUE\nRESULT = VALUE\n",
    )
    assert verifier.verify_executable_definition(request=_request("train_protocol")) == {"valid": True, "diagnostics": []}


def test_package_init_under_namespace_lib_is_part_of_required_source_graph(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    package_init = b"OFFSET = 2\n"
    helper = b"from . import OFFSET\nVALUE = OFFSET + 5\n"
    package = root / "demo" / "lib" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(package_init)
    (package / "helper.py").write_bytes(helper)
    declared = [{"path": "mldb_data/demo/lib/pkg/helper.py", "sha256": _sha(helper)}]
    _install_executable(
        root, "architecture", _executable_document("architecture", sources=declared),
        companion=b"from ..lib.pkg.helper import VALUE\nRESULT = VALUE\n",
    )
    verifier = _verifier(repo, root)
    missing = verifier.verify_executable_definition(request=_request("architecture"))
    assert [item["code"] for item in missing["diagnostics"]] == ["source_declaration_missing"]
    assert "mldb_data/demo/lib/pkg/__init__.py" in missing["diagnostics"][0]["message"]

    all_sources = [
        {"path": "mldb_data/demo/lib/pkg/__init__.py", "sha256": _sha(package_init)},
        {"path": "mldb_data/demo/lib/pkg/helper.py", "sha256": _sha(helper)},
    ]
    _install_executable(
        root, "architecture", _executable_document("architecture", sources=all_sources),
        companion=b"from ..lib.pkg.helper import VALUE\nRESULT = VALUE\n",
    )
    assert verifier.verify_executable_definition(request=_request("architecture")) == {"valid": True, "diagnostics": []}


def test_stdlib_third_party_and_mldb_infrastructure_imports_do_not_require_sources(
    tmp_path: Path,
) -> None:
    repo, root = _make_repo(tmp_path)
    mldb_ids = repo / "mldb_v2" / "src" / "common" / "ids.py"
    mldb_ids.parent.mkdir(parents=True)
    mldb_ids.write_text("VALUE = 1\n", encoding="utf-8")
    companion = (
        b"import json\n"
        b"import torch\n"
        b"from mldb_v2.src.common.ids import VALUE\n"
        b"RESULT = (json.__name__, VALUE)\n"
    )
    _install_executable(
        root,
        "architecture",
        _executable_document("architecture"),
        companion=companion,
    )
    assert _verifier(repo, root).verify_executable_definition(
        request=_request("architecture")
    ) == {"valid": True, "diagnostics": []}


def test_type_checking_only_project_import_is_not_treated_as_runtime_source(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    typing_source = repo / "product" / "types.py"
    typing_source.parent.mkdir(parents=True)
    typing_source.write_text("class Shape: pass\n", encoding="utf-8")
    companion = (
        b"from typing import TYPE_CHECKING\n"
        b"if TYPE_CHECKING:\n"
        b"    from product.types import Shape\n"
        b"VALUE = 1\n"
    )
    _install_executable(
        root,
        "train_protocol",
        _executable_document("train_protocol"),
        companion=companion,
    )
    assert _verifier(repo, root).verify_executable_definition(
        request=_request("train_protocol")
    ) == {"valid": True, "diagnostics": []}


def test_project_owned_tools_import_is_rejected_as_forbidden_source(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    tool = repo / "tools" / "behavior.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("VALUE = 1\n", encoding="utf-8")
    _install_executable(
        root,
        "architecture",
        _executable_document("architecture"),
        companion=b"from tools.behavior import VALUE\nRESULT = VALUE\n",
    )
    result = _verifier(repo, root).verify_executable_definition(
        request=_request("architecture")
    )
    assert [item["code"] for item in result["diagnostics"]] == [
        "source_import_forbidden"
    ]


def test_from_namespace_package_import_discovers_project_submodule(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    behavior_bytes = b"VALUE = 3\n"
    source = repo / "product" / "behavior.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(behavior_bytes)
    _install_executable(
        root,
        "architecture",
        _executable_document("architecture"),
        companion=b"from product import behavior\nRESULT = behavior.VALUE\n",
    )
    result = _verifier(repo, root).verify_executable_definition(
        request=_request("architecture")
    )
    assert [item["code"] for item in result["diagnostics"]] == [
        "source_import_forbidden"
    ]
    assert "product/behavior.py" in result["diagnostics"][0]["message"]


def test_package_import_outside_companion_is_forbidden(tmp_path: Path) -> None:
    repo, root = _make_repo(tmp_path)
    package = repo / "product" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"VALUE = 11\n")
    _install_executable(
        root,
        "train_protocol",
        _executable_document("train_protocol"),
        companion=b"from product.pkg import VALUE\nRESULT = VALUE\n",
    )
    result = _verifier(repo, root).verify_executable_definition(request=_request("train_protocol"))
    assert [item["code"] for item in result["diagnostics"]] == ["source_import_forbidden"]
    assert "product/pkg/__init__.py" in result["diagnostics"][0]["message"]
