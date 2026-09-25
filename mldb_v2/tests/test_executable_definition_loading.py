from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import get_type_hints

import pytest

from mldb_v2.src.catalog.architecture import (
    Architecture,
    ArchitectureImplementation,
    _load_architecture_definition,
    _parse_architecture_document,
)
from mldb_v2.src.catalog.architecture_build import ArchitectureBuild, _load_architecture_build
from mldb_v2.src.catalog._executable_definition_loading import _load_companion_module
from mldb_v2.src.common.ids import EntityKind
from mldb_v2.src.evaluation.evaluate_interface import (
    EvaluationCandidate,
    EvaluationCallable,
    EvaluationContext,
    LoadedModel,
    MaterializedCorpus as EvaluationMaterializedCorpus,
    _load_evaluation_callable,
)
from mldb_v2.src.evaluation.evaluation_protocol import (
    EvaluationArtifactDeclaration,
    EvaluationMetricDeclaration,
    EvaluationProtocol,
    EvaluationProtocolImplementation,
    _load_evaluation_protocol_definition,
    _parse_evaluation_protocol_document,
)
from mldb_v2.src.training.train_interface import (
    MaterializedCorpus as TrainMaterializedCorpus,
    TrainCallable,
    TrainContext,
    _load_train_callable,
)
from mldb_v2.src.training.train_protocol import (
    TrainProtocol,
    TrainProtocolImplementation,
    _load_train_protocol_definition,
    _parse_train_protocol_document,
)
from mldb_v2.src.verification.executable_integrity import (
    CorpusBuilderIntegrityRequest,
    ExecutableDefinitionIntegrityRequest,
    ExecutableIntegrityResult,
    ExecutableIntegrityVerifier,
    ExecutableSource,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "mldb_data"
    _write_json(root / "demo" / "namespace.yaml", {
        "schema": "mjtensu.mldb-v2/namespace/v1",
        "id": "demo", "name": "Demo", "description": "",
    })
    return root


def _architecture(*, entity_id: str = "demo/model-v1", status: str = "draft") -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "mjtensu.mldb-v2/architecture/v1",
        "id": entity_id,
        "status": status,
        "task": "demo/task-v1",
        "name": "Architecture",
        "family": "demo-family",
        "description": "description",
        "implementation": {"framework": "pytorch", "entrypoint": "build"},
        "interface": {"input": {"kind": "image"}, "output": {"kind": "logits"}},
        "structure": {"summary": "summary", "traits": ["a", "b"]},
        "parameters": {"channels": 8, "flag": True},
    }
    if status == "sealed":
        value["implementation"] = {
            "framework": "pytorch", "entrypoint": "build", "sha256": "a" * 64
        }
    return value


def _train(*, entity_id: str = "demo/train-v1", status: str = "draft") -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "mjtensu.mldb-v2/train-protocol/v1",
        "id": entity_id, "status": status, "task": "demo/task-v1",
        "name": "Train", "description": "description",
        "implementation": {"entrypoint": "train"},
        "parameters": {"epochs": {"default": 1, "type": "integer", "minimum": 1}},
    }
    if status == "sealed":
        value["implementation"] = {"entrypoint": "train", "sha256": "b" * 64}
    return value


def _evaluation(*, entity_id: str = "demo/eval-v1", status: str = "draft") -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "mjtensu.mldb-v2/evaluation-protocol/v1",
        "id": entity_id, "status": status, "task": "demo/task-v1",
        "name": "Evaluation", "description": "description",
        "implementation": {"entrypoint": "evaluate"},
        "parameters": {},
        "metrics": {"score": {"type": "number", "required": True}},
        "artifacts": {},
    }
    if status == "sealed":
        value["implementation"] = {"entrypoint": "evaluate", "sha256": "c" * 64}
    return value


def _write_executable(root: Path, domain: str, local_id: str, document: object, source: str) -> None:
    directory = root / "demo" / domain
    _write_json(directory / f"{local_id}.yaml", document)
    (directory / f"{local_id}.py").write_text(source, encoding="utf-8")


def test_current_executable_examples_parse() -> None:
    root = Path(__file__).resolve().parents[2] / "mldb_data"
    architecture_ids = [
        "tile-classifier/tile-plain-gray35-v2",
        "rotated-fcos/rotated-fcos-s05-rgb-baseline-v1",
        "rotated-fcos/rotated-fcos-s05-rgb-nopool-v1",
        "rotated-fcos/rotated-fcos-s05-rgb-p2-v1",
        "rotated-fcos/rotated-fcos-s05-rgb-stem-s1-v1",
    ]
    for entity_id in architecture_ids:
        assert _load_architecture_definition(root, entity_id)["id"] == entity_id
    for entity_id in [
        "tile-classifier/tile-shape-train-gpu-v3",
        "rotated-fcos/rotated-fcos-train-gpu-v2",
    ]:
        assert _load_train_protocol_definition(root, entity_id)["id"] == entity_id
    for entity_id in [
        "tile-classifier/tile-shape-angle-robustness-v1",
        "rotated-fcos/detector-crop-quality-v1",
        "rotated-fcos/rotated-fcos-standard-v1",
    ]:
        assert _load_evaluation_protocol_definition(root, entity_id)["id"] == entity_id


def test_architecture_parser_accepts_descriptive_fields() -> None:
    value = _architecture()
    parsed = _parse_architecture_document(value, expected_id="demo/model-v1")
    assert parsed["interface"]["input"]["kind"] == "image"
    assert parsed["structure"]["traits"] == ["a", "b"]
    assert parsed["parameters"] == {"channels": 8, "flag": True}


@pytest.mark.parametrize("mutation", [
    {"schema": "wrong"}, {"id": "demo/model"}, {"id": "demo/model-v0"},
    {"id": "demo/model-v01"}, {"id": "Demo/model-v1"}, {"status": "complete"},
    {"task": "demo/task"}, {"task": "Demo/task-v1"}, {"name": ""}, {"family": ""},
])
def test_architecture_rejects_invalid_core_fields(mutation: dict[str, object]) -> None:
    value = _architecture()
    value.update(mutation)
    with pytest.raises(ValueError):
        _parse_architecture_document(value, expected_id="demo/model-v1")


def test_architecture_rejects_unknown_top_level_and_path_mismatch() -> None:
    value = _architecture()
    value["unexpected"] = 1
    with pytest.raises(ValueError):
        _parse_architecture_document(value, expected_id="demo/model-v1")
    with pytest.raises(ValueError, match="path identity"):
        _parse_architecture_document(_architecture(), expected_id="demo/other-v1")


@pytest.mark.parametrize("implementation", [
    {"framework": "tensorflow", "entrypoint": "build"},
    {"framework": "pytorch", "entrypoint": "run"},
    {"framework": "pytorch", "entrypoint": "build", "extra": 1},
    {"framework": "pytorch", "entrypoint": "build", "sha256": "A" * 64},
])
def test_architecture_rejects_invalid_implementation(implementation: dict[str, object]) -> None:
    value = _architecture()
    value["implementation"] = implementation
    with pytest.raises(ValueError):
        _parse_architecture_document(value, expected_id="demo/model-v1")


def test_sealed_architecture_requires_sha256() -> None:
    value = _architecture(status="sealed")
    value["implementation"] = {"framework": "pytorch", "entrypoint": "build"}
    with pytest.raises(ValueError, match="sha256"):
        _parse_architecture_document(value, expected_id="demo/model-v1")


@pytest.mark.parametrize("bad", [
    [], {"input": {"kind": "image"}},
    {"input": {"kind": ""}, "output": {"kind": "logits"}},
    {"input": {"kind": "image"}, "output": {"kind": ""}},
])
def test_architecture_interface_contract(bad: object) -> None:
    value = _architecture(); value["interface"] = bad
    with pytest.raises(ValueError):
        _parse_architecture_document(value, expected_id="demo/model-v1")


@pytest.mark.parametrize("structure", [
    {}, {"summary": ""}, {"summary": "ok", "traits": ["a", "a"]},
    {"summary": "ok", "traits": ["a", ""]},
])
def test_architecture_structure_contract(structure: object) -> None:
    value = _architecture(); value["structure"] = structure
    with pytest.raises(ValueError):
        _parse_architecture_document(value, expected_id="demo/model-v1")


@pytest.mark.parametrize("sources", [
    [{"path": "tools/x.py", "sha256": "0" * 64}],
    [{"path": "/abs/x.py", "sha256": "0" * 64}],
    [{"path": "a/../x.py", "sha256": "0" * 64}],
    [{"path": "a/*.py", "sha256": "0" * 64}],
    [{"path": "b.py", "sha256": "0" * 64}, {"path": "a.py", "sha256": "1" * 64}],
    [{"path": "a.py", "sha256": "0" * 64}, {"path": "a.py", "sha256": "1" * 64}],
    [{"path": "a.py", "sha256": "A" * 64}],
    [{"path": "product/x.py", "sha256": "0" * 64}],
    [{"path": "mldb_data/other/lib/x.py", "sha256": "0" * 64}],
])
def test_executable_sources_reject_unsafe_unsorted_duplicate_or_bad_hash(sources: object) -> None:
    value = _train()
    value["implementation"] = {"entrypoint": "train", "sources": sources}
    with pytest.raises(ValueError):
        _parse_train_protocol_document(value, expected_id="demo/train-v1")


def test_executable_sources_accept_same_namespace_lib_entries() -> None:
    value = _train()
    value["implementation"] = {"entrypoint": "train", "sources": [
        {"path": "mldb_data/demo/lib/a.py", "sha256": "0" * 64},
        {"path": "mldb_data/demo/lib/pkg/b.py", "sha256": "1" * 64},
    ]}
    parsed = _parse_train_protocol_document(value, expected_id="demo/train-v1")
    assert parsed["implementation"]["sources"] == value["implementation"]["sources"]


def test_companion_relative_imports_namespace_private_lib_with_hyphen_namespace(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    namespace = root / "rotated-fcos"
    (namespace / "architectures").mkdir(parents=True)
    (namespace / "lib").mkdir()
    (namespace / "lib" / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")
    (namespace / "architectures" / "model-v1.py").write_text(
        "from ..lib.helper import VALUE\ndef build():\n    return VALUE\n", encoding="utf-8"
    )
    module = _load_companion_module(root, kind=EntityKind.ARCHITECTURE, entity_id="rotated-fcos/model-v1")
    assert module.build() == 7


@pytest.mark.parametrize("mutation", [
    {"schema": "wrong"}, {"id": "demo/train"}, {"status": "done"},
    {"task": "demo/task"}, {"name": ""},
])
def test_train_protocol_rejects_invalid_core_fields(mutation: dict[str, object]) -> None:
    value = _train(); value.update(mutation)
    with pytest.raises(ValueError):
        _parse_train_protocol_document(value, expected_id="demo/train-v1")


def test_train_protocol_entrypoint_sha_and_empty_parameters() -> None:
    value = _train(); value["parameters"] = {}
    assert _parse_train_protocol_document(value, expected_id="demo/train-v1")["parameters"] == {}
    value = _train(); value["implementation"] = {"entrypoint": "run"}
    with pytest.raises(ValueError):
        _parse_train_protocol_document(value, expected_id="demo/train-v1")
    value = _train(status="sealed"); value["implementation"] = {"entrypoint": "train"}
    with pytest.raises(ValueError, match="sha256"):
        _parse_train_protocol_document(value, expected_id="demo/train-v1")


@pytest.mark.parametrize("declaration", [
    {}, {"default": 1, "unknown": 2}, {"default": True, "type": "integer"},
    {"default": 1, "type": "integer", "minimum": 2},
    {"default": 1, "enum": []}, {"default": float("inf")},
])
def test_train_protocol_parameter_declaration_semantics(declaration: object) -> None:
    value = _train(); value["parameters"] = {"p": declaration}
    with pytest.raises(ValueError):
        _parse_train_protocol_document(value, expected_id="demo/train-v1")


@pytest.mark.parametrize("mutation", [
    {"schema": "wrong"}, {"id": "demo/eval"}, {"status": "done"},
    {"task": "demo/task"}, {"name": ""},
])
def test_evaluation_protocol_rejects_invalid_core_fields(mutation: dict[str, object]) -> None:
    value = _evaluation(); value.update(mutation)
    with pytest.raises(ValueError):
        _parse_evaluation_protocol_document(value, expected_id="demo/eval-v1")


def test_evaluation_protocol_rejects_unknown_top_level_and_empty_outputs() -> None:
    value = _evaluation(); value["unexpected"] = 1
    with pytest.raises(ValueError):
        _parse_evaluation_protocol_document(value, expected_id="demo/eval-v1")
    value = _evaluation(); value["metrics"] = {}; value["artifacts"] = {}
    with pytest.raises(ValueError, match="at least one"):
        _parse_evaluation_protocol_document(value, expected_id="demo/eval-v1")


@pytest.mark.parametrize("declaration", [
    {"type": "string", "required": True},
    {"type": "number", "required": 1},
    {"type": "integer", "required": False, "unknown": 1},
    {"type": "number", "required": True, "preference": "largest"},
    {"type": "number", "required": True, "preference": 1},
    {"type": "number"},
])
def test_evaluation_metric_exact_contract(declaration: object) -> None:
    value = _evaluation(); value["metrics"] = {"score": declaration}
    with pytest.raises(ValueError):
        _parse_evaluation_protocol_document(value, expected_id="demo/eval-v1")


def test_evaluation_metric_preference_contract() -> None:
    for preference in ("higher", "lower", "neutral"):
        value = _evaluation()
        value["metrics"] = {
            "score": {"type": "number", "required": True, "preference": preference}
        }
        parsed = _parse_evaluation_protocol_document(value, expected_id="demo/eval-v1")
        assert parsed["metrics"]["score"]["preference"] == preference

    value = _evaluation()
    parsed = _parse_evaluation_protocol_document(value, expected_id="demo/eval-v1")
    assert "preference" not in parsed["metrics"]["score"]


@pytest.mark.parametrize("declaration", [
    {"format": "", "schema": "demo/artifact/v1", "required": True},
    {"format": "json", "schema": "demo/artifact", "required": True},
    {"format": "json", "schema": "demo/artifact/v0", "required": True},
    {"format": "json", "schema": "demo/artifact/v01", "required": True},
    {"format": "json", "schema": "demo/artifact/v1", "required": 1},
    {"format": "json", "schema": "demo/artifact/v1", "required": False, "extra": 1},
    {"format": "json", "schema": "demo/artifact/v1", "required": False, "study_view": "dropdown"},
    {"format": "json", "schema": "demo/artifact/v1", "required": False, "study_view": 1},
])
def test_evaluation_artifact_exact_contract(declaration: object) -> None:
    value = _evaluation(); value["metrics"] = {}; value["artifacts"] = {"artifact": declaration}
    with pytest.raises(ValueError):
        _parse_evaluation_protocol_document(value, expected_id="demo/eval-v1")


def test_evaluation_artifact_and_parameter_valid_cases() -> None:
    value = _evaluation()
    value["parameters"] = {"threshold": {"default": 0.5, "type": "number", "minimum": 0.0}}
    value["artifacts"] = {"details": {
        "format": "jsonl", "schema": "mjtensu.mldb-v2/demo-artifact/v2",
        "required": False, "description": "details", "study_view": "select",
    }}
    parsed = _parse_evaluation_protocol_document(value, expected_id="demo/eval-v1")
    assert parsed["artifacts"]["details"]["format"] == "jsonl"
    assert parsed["artifacts"]["details"]["study_view"] == "select"

    for study_view in ("hidden", "select", "all"):
        value = _evaluation()
        value["artifacts"] = {"details": {
            "format": "jsonl", "schema": "mjtensu.mldb-v2/demo-artifact/v2",
            "required": False, "study_view": study_view,
        }}
        parsed = _parse_evaluation_protocol_document(value, expected_id="demo/eval-v1")
        assert parsed["artifacts"]["details"]["study_view"] == study_view


def test_evaluation_entrypoint_and_sealed_sha() -> None:
    value = _evaluation(); value["implementation"] = {"entrypoint": "run"}
    with pytest.raises(ValueError):
        _parse_evaluation_protocol_document(value, expected_id="demo/eval-v1")
    value = _evaluation(status="sealed"); value["implementation"] = {"entrypoint": "evaluate"}
    with pytest.raises(ValueError, match="sha256"):
        _parse_evaluation_protocol_document(value, expected_id="demo/eval-v1")


def test_architecture_loader_requires_canonical_yaml_before_companion(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    directory = root / "demo" / "architectures"; directory.mkdir(parents=True)
    marker = tmp_path / "imported.txt"
    (directory / "model-v1.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\ndef build():\n import torch.nn as nn\n return nn.Linear(1,1)\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        _load_architecture_build(root, "demo/model-v1")
    assert not marker.exists()


def test_architecture_loader_rejects_invalid_definition_before_import(tmp_path: Path) -> None:
    root = _make_root(tmp_path); directory = root / "demo" / "architectures"
    marker = tmp_path / "imported.txt"
    value = _architecture(); value["implementation"] = {"framework": "pytorch", "entrypoint": "wrong"}
    _write_executable(root, "architectures", "model-v1", value,
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\ndef build():\n return None\n")
    with pytest.raises(ValueError):
        _load_architecture_build(root, "demo/model-v1")
    assert not marker.exists()


def test_architecture_loader_exact_same_basename_success(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _write_executable(root, "architectures", "model-v1", _architecture(),
        "import torch.nn as nn\ndef build():\n return nn.Linear(1, 1)\n")
    assert callable(_load_architecture_build(root, "demo/model-v1"))


@pytest.mark.parametrize("source", [
    "x = 1\n",
    "build = 1\n",
    "def build(x):\n return x\n",
    "def build():\n raise RuntimeError('boom')\n",
    "def build():\n return object()\n",
    "import torch.nn as nn\nMODEL=nn.Linear(1,1)\ndef build():\n return MODEL\n",
])
def test_architecture_loader_rejects_bad_build_contract(tmp_path: Path, source: str) -> None:
    root = _make_root(tmp_path)
    _write_executable(root, "architectures", "model-v1", _architecture(), source)
    with pytest.raises(ValueError):
        _load_architecture_build(root, "demo/model-v1")


def test_architecture_loader_wrong_basename_does_not_count(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _write_json(root / "demo" / "architectures" / "model-v1.yaml", _architecture())
    (root / "demo" / "architectures" / "other.py").write_text(
        "import torch.nn as nn\ndef build():\n return nn.Linear(1,1)\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        _load_architecture_build(root, "demo/model-v1")


@pytest.mark.parametrize("domain,document,loader,entrypoint", [
    ("train_protocols", _train(), _load_train_callable, "train"),
    ("evaluation_protocols", _evaluation(), _load_evaluation_callable, "evaluate"),
])
def test_context_callable_loader_requires_canonical_definition_first(
    tmp_path: Path, domain: str, document: dict[str, object], loader, entrypoint: str
) -> None:
    root = _make_root(tmp_path); directory = root / "demo" / domain; directory.mkdir(parents=True)
    local_id = "train-v1" if domain == "train_protocols" else "eval-v1"
    marker = tmp_path / f"{domain}.txt"
    (directory / f"{local_id}.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\ndef {entrypoint}(context):\n raise AssertionError('not called')\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        loader(root, f"demo/{local_id}")
    assert not marker.exists()


@pytest.mark.parametrize("domain,document,loader,entrypoint", [
    ("train_protocols", _train(), _load_train_callable, "train"),
    ("evaluation_protocols", _evaluation(), _load_evaluation_callable, "evaluate"),
])
def test_context_callable_loader_valid_signature_not_invoked(
    tmp_path: Path, domain: str, document: dict[str, object], loader, entrypoint: str
) -> None:
    root = _make_root(tmp_path); local_id = "train-v1" if domain == "train_protocols" else "eval-v1"
    _write_executable(root, domain, local_id, document,
        f"def {entrypoint}(context):\n raise AssertionError('loader must not execute')\n")
    assert callable(loader(root, f"demo/{local_id}"))


@pytest.mark.parametrize("domain,document,loader,entrypoint", [
    ("train_protocols", _train(), _load_train_callable, "train"),
    ("evaluation_protocols", _evaluation(), _load_evaluation_callable, "evaluate"),
])
@pytest.mark.parametrize("source", [
    "x = 1\n",
    "{entrypoint} = 1\n",
    "def {entrypoint}():\n return None\n",
    "def {entrypoint}(context, extra):\n return None\n",
    "def {entrypoint}(*, context):\n return None\n",
])
def test_context_callable_loader_rejects_missing_noncallable_or_bad_signature(
    tmp_path: Path, domain: str, document: dict[str, object], loader, entrypoint: str, source: str
) -> None:
    root = _make_root(tmp_path); local_id = "train-v1" if domain == "train_protocols" else "eval-v1"
    _write_executable(root, domain, local_id, document, source.format(entrypoint=entrypoint))
    with pytest.raises(ValueError):
        loader(root, f"demo/{local_id}")


@pytest.mark.parametrize("domain,document,loader", [
    ("train_protocols", _train(), _load_train_callable),
    ("evaluation_protocols", _evaluation(), _load_evaluation_callable),
])
def test_context_callable_loader_wrong_basename_does_not_count(
    tmp_path: Path, domain: str, document: dict[str, object], loader
) -> None:
    root = _make_root(tmp_path); local_id = "train-v1" if domain == "train_protocols" else "eval-v1"
    _write_json(root / "demo" / domain / f"{local_id}.yaml", document)
    (root / "demo" / domain / "other.py").write_text("def train(context): pass\ndef evaluate(context): pass\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        loader(root, f"demo/{local_id}")


def test_public_typed_dict_shapes_match_frozen_contract() -> None:
    assert ArchitectureImplementation.__required_keys__ == frozenset({"framework", "entrypoint"})
    assert ArchitectureImplementation.__optional_keys__ == frozenset({"sha256", "sources"})
    assert Architecture.__required_keys__ == frozenset({
        "schema", "id", "status", "task", "name", "family", "description",
        "implementation", "interface", "structure",
    })
    assert Architecture.__optional_keys__ == frozenset({"parameters"})
    assert TrainProtocolImplementation.__required_keys__ == frozenset({"entrypoint"})
    assert TrainProtocolImplementation.__optional_keys__ == frozenset({"sha256", "sources"})
    assert TrainProtocol.__required_keys__ == frozenset({
        "schema", "id", "status", "task", "name", "description", "implementation", "parameters",
    })
    assert EvaluationProtocolImplementation.__required_keys__ == frozenset({"entrypoint"})
    assert EvaluationProtocolImplementation.__optional_keys__ == frozenset({"sha256", "sources"})
    assert EvaluationMetricDeclaration.__required_keys__ == frozenset({"type", "required"})
    assert EvaluationMetricDeclaration.__optional_keys__ == frozenset({"description", "preference"})
    assert EvaluationArtifactDeclaration.__required_keys__ == frozenset({"format", "schema", "required"})
    assert EvaluationArtifactDeclaration.__optional_keys__ == frozenset({"description", "study_view"})


def test_executable_integrity_public_shapes_match_frozen_contract() -> None:
    assert ExecutableSource.__required_keys__ == frozenset({"path", "sha256"})
    assert ExecutableDefinitionIntegrityRequest.__required_keys__ == frozenset({"kind", "id"})
    assert CorpusBuilderIntegrityRequest.__required_keys__ == frozenset({"corpus"})
    assert ExecutableIntegrityResult.__required_keys__ == frozenset({"valid", "diagnostics"})
    assert set(ExecutableIntegrityVerifier.__dict__) >= {"verify_executable_definition", "verify_corpus_builder"}


def _field_names(cls: type) -> tuple[str, ...]:
    return tuple(field.name for field in dataclasses.fields(cls))


def test_train_and_evaluation_dataclass_shapes_are_frozen_and_exact() -> None:
    assert _field_names(TrainMaterializedCorpus) == ("definition", "root")
    assert _field_names(TrainContext) == (
        "task", "corpus", "architecture", "model", "seed", "parameters", "telemetry", "work_dir"
    )
    assert _field_names(EvaluationMaterializedCorpus) == ("definition", "root")
    assert _field_names(LoadedModel) == (
        "definition", "training_result", "architecture", "module"
    )
    assert _field_names(EvaluationContext) == (
        "task", "corpus", "model", "parameters", "telemetry", "work_dir"
    )
    assert _field_names(EvaluationCandidate) == ("metrics", "artifacts")
    for cls in (
        TrainMaterializedCorpus, TrainContext, EvaluationMaterializedCorpus,
        LoadedModel, EvaluationContext, EvaluationCandidate,
    ):
        assert cls.__dataclass_params__.frozen is True
    assert "__call__" in TrainCallable.__dict__
    assert "__call__" in EvaluationCallable.__dict__


@pytest.mark.parametrize("domain,local_id,loader,entrypoint", [
    ("architectures", "model-v1", _load_architecture_build, "build"),
    ("train_protocols", "train-v1", _load_train_callable, "train"),
    ("evaluation_protocols", "eval-v1", _load_evaluation_callable, "evaluate"),
])
def test_malformed_canonical_yaml_never_imports_existing_companion(
    tmp_path: Path, domain: str, local_id: str, loader, entrypoint: str
) -> None:
    root = _make_root(tmp_path); directory = root / "demo" / domain; directory.mkdir(parents=True)
    marker = tmp_path / f"malformed-{domain}.txt"
    (directory / f"{local_id}.yaml").write_text("{not: [valid", encoding="utf-8")
    (directory / f"{local_id}.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\ndef {entrypoint}(context=None):\n return None\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        loader(root, f"demo/{local_id}")
    assert not marker.exists()


def test_companion_loading_does_not_create_bytecode_cache_in_canonical_domain(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _write_executable(
        root,
        "train_protocols",
        "train-v1",
        _train(),
        "def train(context):\n    return context.model\n",
    )
    domain = root / "demo" / "train_protocols"
    assert not (domain / "__pycache__").exists()
    assert callable(_load_train_callable(root, "demo/train-v1"))
    assert not (domain / "__pycache__").exists()
