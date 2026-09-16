from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.training import runtime as training_runtime
from mldb_v2.src.training.canonical_weights import _build_fresh_architecture_module
from mldb_v2.src.training.runtime import _execute_training_stage


class DictTransport:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.reads: list[str] = []
        self.publications: list[tuple[str, bytes]] = []

    def read_bytes(self, uri: str) -> bytes:
        self.reads.append(uri)
        return self.objects[uri]

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        if uri in self.objects:
            raise FileExistsError(uri)
        self.objects[uri] = data
        self.publications.append((uri, data))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _architecture_document(
    *, task: str = "demo/task-v1", status: str = "sealed", family: str = "toy"
) -> dict[str, object]:
    implementation: dict[str, object] = {"framework": "pytorch", "entrypoint": "build"}
    if status == "sealed":
        implementation["sha256"] = "1" * 64
    return {
        "schema": "mjtensu.mldb-v2/architecture/v1",
        "id": "demo/arch-v1",
        "status": status,
        "task": task,
        "name": "Architecture",
        "family": family,
        "description": "",
        "implementation": implementation,
        "interface": {"input": {"kind": "tensor"}, "output": {"kind": "tensor"}},
        "structure": {"summary": family},
    }


def _protocol_document(
    *, task: str = "demo/task-v1", status: str = "sealed"
) -> dict[str, object]:
    implementation: dict[str, object] = {"entrypoint": "train"}
    if status == "sealed":
        implementation["sha256"] = "2" * 64
    return {
        "schema": "mjtensu.mldb-v2/train-protocol/v1",
        "id": "demo/train-v1",
        "status": status,
        "task": task,
        "name": "Train",
        "description": "",
        "implementation": implementation,
        "parameters": {
            "epochs": {"default": 9, "type": "integer", "minimum": 1, "maximum": 20},
            "learning_rate": {
                "default": 0.5,
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
    }


def _install_repo(
    tmp_path: Path,
    *,
    task_status: str = "sealed",
    corpus_status: str = "sealed",
    architecture_status: str = "sealed",
    protocol_status: str = "sealed",
    corpus_task: str = "demo/task-v1",
    architecture_task: str = "demo/task-v1",
    protocol_task: str = "demo/task-v1",
    architecture_source: str | None = None,
    train_source: str | None = None,
    family: str = "toy",
) -> tuple[Path, bytes]:
    root = tmp_path / "mldb_data"
    _write_json(
        root / "demo" / "namespace.yaml",
        {
            "schema": "mjtensu.mldb-v2/namespace/v1",
            "id": "demo",
            "name": "Demo",
            "description": "",
        },
    )
    _write_json(
        root / "demo" / "tasks" / "task-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/task/v1",
            "id": "demo/task-v1",
            "status": task_status,
            "name": "Task",
            "problem_type": "regression",
            "description": "",
            "input": {},
            "target": {"type": "continuous"},
            "semantics": {},
            "scope": {},
        },
    )

    corpus_bytes = b"sealed corpus object\n"
    manifest_entry = {
        "path": "data.bin",
        "bytes": len(corpus_bytes),
        "sha256": _sha(corpus_bytes),
    }
    manifest_bytes = json.dumps(manifest_entry, separators=(",", ":")).encode() + b"\n"
    corpus_dir = root / "demo" / "corpora"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "corpus-v1.manifest.jsonl").write_bytes(manifest_bytes)
    _write_json(
        corpus_dir / "corpus-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/corpus/v1",
            "id": "demo/corpus-v1",
            "status": corpus_status,
            "task": corpus_task,
            "description": "",
            "storage": {"root_uri": "s3://bucket/corpus-v1"},
            "manifest": {
                "file": "corpus-v1.manifest.jsonl",
                "sha256": _sha(manifest_bytes),
                "entries": 1,
            },
            "representation": {"kind": "binary"},
            "splits": {"train": 1},
        },
    )

    architecture_dir = root / "demo" / "architectures"
    _write_json(
        architecture_dir / "arch-v1.yaml",
        _architecture_document(
            task=architecture_task, status=architecture_status, family=family
        ),
    )
    architecture_dir.mkdir(parents=True, exist_ok=True)
    (architecture_dir / "arch-v1.py").write_text(
        architecture_source
        or "import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n",
        encoding="utf-8",
    )

    protocol_dir = root / "demo" / "train_protocols"
    _write_json(
        protocol_dir / "train-v1.yaml",
        _protocol_document(task=protocol_task, status=protocol_status),
    )
    protocol_dir.mkdir(parents=True, exist_ok=True)
    (protocol_dir / "train-v1.py").write_text(
        train_source
        or (
            "import torch\n"
            "def train(context):\n"
            "    with torch.no_grad():\n"
            "        for parameter in context.model.parameters():\n"
            "            parameter.add_(1.0)\n"
            "    context.telemetry.report_scalar(group='fixture', series='signal', value=1.0, step=0)\n"
            "    return context.model\n"
        ),
        encoding="utf-8",
    )
    return root, corpus_bytes


def _stage_input(**stage_overrides: object) -> dict[str, object]:
    stage: dict[str, object] = {
        "task": "demo/task-v1",
        "corpus": "demo/corpus-v1",
        "architecture": "demo/arch-v1",
        "train_protocol": "demo/train-v1",
        "parameters": {"epochs": 2, "learning_rate": 0.25},
        "seed": 42,
    }
    stage.update(stage_overrides)
    return {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": "demo/run-v1",
        "plan": "demo/plan-v1",
        "plan_sha256": "0" * 64,
        "trial": "trial-0001",
        "kind": "training",
        "coordinate": None,
        "source_commit": "a" * 40,
        "pins": [],
        "stage": stage,
        "runtime_model": None,
    }


def _access(corpus_bytes: bytes) -> tuple[_ObjectByteAccess, DictTransport]:
    transport = DictTransport({"s3://bucket/corpus-v1/data.bin": corpus_bytes})
    return _ObjectByteAccess(transport), transport


def _run(
    tmp_path: Path,
    root: Path,
    corpus_bytes: bytes,
    stage_input: dict[str, object] | None = None,
    *,
    weights_uri: str = "s3://bucket/candidates/weights.pt",
    telemetry_sink=None,
):
    access, transport = _access(corpus_bytes)
    result = _execute_training_stage(
        stage_input or _stage_input(),  # type: ignore[arg-type]
        mldb_data_root=root,
        object_bytes=access,
        corpus_destination_root=tmp_path / "corpus-runtime",
        work_dir=tmp_path / "work",
        weights_uri=weights_uri,
        telemetry_sink=telemetry_sink,
    )
    return result, transport


def test_valid_training_stage_returns_exact_canonical_candidate(tmp_path: Path) -> None:
    root, corpus_bytes = _install_repo(tmp_path)
    result, transport = _run(tmp_path, root, corpus_bytes)

    assert set(result) == {"weights"}
    weights = result["weights"]
    assert set(weights) == {"uri", "bytes", "sha256", "format"}
    assert weights["uri"] == "s3://bucket/candidates/weights.pt"
    assert weights["format"] == "pytorch-state-dict/v1"
    published = transport.objects[weights["uri"]]
    assert weights["bytes"] == len(published)
    assert weights["sha256"] == _sha(published)

    loaded = torch.load(BytesIO(published), map_location="cpu", weights_only=True)
    assert type(loaded) is dict
    assert set(loaded) == {"weight", "bias"}
    assert all(isinstance(value, torch.Tensor) and value.device.type == "cpu" for value in loaded.values())
    assert "state_dict" not in loaded


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "evaluation", "kind"),
        ("coordinate", "eval-0001", "coordinate"),
        ("runtime_model", {}, "runtime_model"),
    ],
)
def test_training_variant_rejects_wrong_kind_coordinate_or_runtime_model(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    root, corpus_bytes = _install_repo(tmp_path)
    stage_input = _stage_input()
    stage_input[field] = value
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, root, corpus_bytes, stage_input)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_stage_input_and_training_stage_require_exact_shapes(tmp_path: Path, mutation: str) -> None:
    root, corpus_bytes = _install_repo(tmp_path)
    stage_input = _stage_input()
    if mutation == "missing":
        stage_input["stage"].pop("architecture")  # type: ignore[union-attr]
    else:
        stage_input["stage"]["extra"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="TrainingStage fields"):
        _run(tmp_path, root, corpus_bytes, stage_input)

    stage_input = _stage_input()
    stage_input["unexpected"] = 1
    with pytest.raises(ValueError, match="TrainingStageInput fields"):
        _run(tmp_path, root, corpus_bytes, stage_input)


def test_invalid_typed_reference_and_trial_id_are_rejected(tmp_path: Path) -> None:
    root, corpus_bytes = _install_repo(tmp_path)
    with pytest.raises(ValueError):
        _run(tmp_path, root, corpus_bytes, _stage_input(task="not-a-typed-ref"))
    stage_input = _stage_input()
    stage_input["trial"] = "trial-0000"
    with pytest.raises(ValueError, match="trial"):
        _run(tmp_path, root, corpus_bytes, stage_input)


def test_bool_seed_is_rejected(tmp_path: Path) -> None:
    root, corpus_bytes = _install_repo(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        _run(tmp_path, root, corpus_bytes, _stage_input(seed=True))


def test_unresolved_definition_is_rejected(tmp_path: Path) -> None:
    root, corpus_bytes = _install_repo(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing-v1"):
        _run(tmp_path, root, corpus_bytes, _stage_input(architecture="demo/missing-v1"))


@pytest.mark.parametrize(
    "draft_target", ["task", "corpus", "architecture", "protocol"]
)
def test_execution_definitions_must_be_sealed(tmp_path: Path, draft_target: str) -> None:
    kwargs = {
        "task_status": "sealed",
        "corpus_status": "sealed",
        "architecture_status": "sealed",
        "protocol_status": "sealed",
    }
    kwargs[f"{draft_target}_status"] = "draft"
    root, corpus_bytes = _install_repo(tmp_path, **kwargs)
    with pytest.raises(ValueError, match="sealed"):
        _run(tmp_path, root, corpus_bytes)


@pytest.mark.parametrize(
    ("repo_override", "message"),
    [
        ({"corpus_task": "demo/other-task-v1"}, "Corpus task"),
        ({"architecture_task": "demo/other-task-v1"}, "Architecture task"),
        ({"protocol_task": "demo/other-task-v1"}, "TrainProtocol task"),
    ],
)
def test_task_compatibility_is_exact(
    tmp_path: Path, repo_override: dict[str, object], message: str
) -> None:
    root, corpus_bytes = _install_repo(tmp_path, **repo_override)
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, root, corpus_bytes)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"epochs": 2}, "incomplete"),
        ({"epochs": 2, "learning_rate": 0.25, "unknown": 1}, "unknown"),
        ({"epochs": "2", "learning_rate": 0.25}, "declared type"),
        ({"epochs": 2, "learning_rate": 1.5}, "maximum"),
    ],
)
def test_complete_declared_parameter_mapping_is_required(
    tmp_path: Path, parameters: dict[str, object], message: str
) -> None:
    root, corpus_bytes = _install_repo(tmp_path)
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, root, corpus_bytes, _stage_input(parameters=parameters))


def test_materialization_delegation_and_exact_train_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, corpus_bytes = _install_repo(tmp_path)
    access, transport = _access(corpus_bytes)
    destination = tmp_path / "exact-corpus-destination"
    work_dir = tmp_path / "exact-work-dir"
    captured: dict[str, object] = {}

    original_materialize = training_runtime.materialize_sealed_corpus

    def materialize_spy(**kwargs):
        captured["materialize"] = kwargs.copy()
        return original_materialize(**kwargs)

    def train_spy(context):
        captured["context"] = context
        context.telemetry.report_scalar(
            group="fixture", series="signal", value=1.0, step=0
        )
        return context.model

    monkeypatch.setattr(training_runtime, "materialize_sealed_corpus", materialize_spy)
    monkeypatch.setattr(training_runtime, "_load_train_callable", lambda *_args, **_kwargs: train_spy)

    _execute_training_stage(
        _stage_input(),  # type: ignore[arg-type]
        mldb_data_root=root,
        object_bytes=access,
        corpus_destination_root=destination,
        work_dir=work_dir,
        weights_uri="s3://bucket/candidates/context.pt",
    )

    materialize = captured["materialize"]
    assert materialize == {
        "mldb_data_root": root,
        "corpus_id": "demo/corpus-v1",
        "object_bytes": access,
        "destination_root": destination,
    }
    context = captured["context"]
    assert set(context.__dict__) == {
        "task", "corpus", "architecture", "model", "seed", "parameters", "telemetry", "work_dir"
    }
    assert context.task["id"] == "demo/task-v1"
    assert context.corpus.definition["id"] == "demo/corpus-v1"
    assert context.corpus.root == destination
    assert context.architecture["id"] == "demo/arch-v1"
    assert isinstance(context.model, nn.Linear)
    assert context.seed == 42
    assert context.parameters == {"epochs": 2, "learning_rate": 0.25}
    assert context.work_dir == work_dir
    assert not any(
        token in key
        for key in context.__dict__
        for token in ("backend", "credential", "storage", "s3", "study", "clearml", "uri")
    )
    another = _build_fresh_architecture_module(root, "demo/arch-v1")
    assert context.model is not another
    assert transport.reads == ["s3://bucket/corpus-v1/data.bin"]


def test_protocol_may_return_different_compatible_module(tmp_path: Path) -> None:
    root, corpus_bytes = _install_repo(
        tmp_path,
        train_source=(
            "import torch.nn as nn\n"
            "def train(context):\n"
            "    context.telemetry.report_scalar(group='fixture', series='signal', value=1, step=0)\n"
            "    return nn.Linear(3, 2)\n"
        ),
    )
    result, _ = _run(tmp_path, root, corpus_bytes)
    assert result["weights"]["format"] == "pytorch-state-dict/v1"


def test_protocol_non_module_and_incompatible_state_are_rejected(tmp_path: Path) -> None:
    non_module_root, corpus_bytes = _install_repo(
        tmp_path / "non-module",
        train_source=(
            "def train(context):\n"
            "    context.telemetry.report_scalar(group='fixture', series='signal', value=1, step=0)\n"
            "    return object()\n"
        ),
    )
    with pytest.raises(ValueError, match="torch.nn.Module"):
        _run(tmp_path / "non-module", non_module_root, corpus_bytes)

    incompatible_root, corpus_bytes = _install_repo(
        tmp_path / "incompatible",
        train_source=(
            "import torch.nn as nn\n"
            "def train(context):\n"
            "    context.telemetry.report_scalar(group='fixture', series='signal', value=1, step=0)\n"
            "    return nn.Linear(4, 2)\n"
        ),
    )
    with pytest.raises(ValueError, match="strictly compatible"):
        _run(tmp_path / "incompatible", incompatible_root, corpus_bytes)


def test_multiple_valid_telemetry_reports_allow_training_success(tmp_path: Path) -> None:
    root, corpus_bytes = _install_repo(
        tmp_path,
        train_source=(
            "def train(context):\n"
            "    context.telemetry.report_scalar(group='first', series='a', value=1, step=0)\n"
            "    context.telemetry.report_scalar(group='second', series='b', value=2.5, step=3)\n"
            "    return context.model\n"
        ),
    )
    result, transport = _run(tmp_path, root, corpus_bytes)
    assert set(result) == {"weights"}
    assert len(transport.publications) == 1


def test_telemetry_sink_failure_does_not_change_training_result_or_weights(tmp_path: Path) -> None:
    root, corpus_bytes = _install_repo(tmp_path)
    delivered = []

    torch.manual_seed(1234)
    success_result, success_transport = _run(
        tmp_path / "success", root, corpus_bytes, telemetry_sink=delivered.append
    )

    def failing_sink(_event) -> None:
        raise RuntimeError("clearml delivery unavailable")

    torch.manual_seed(1234)
    failed_delivery_result, failed_delivery_transport = _run(
        tmp_path / "failed-delivery", root, corpus_bytes, telemetry_sink=failing_sink
    )

    assert len(delivered) == 1
    assert failed_delivery_result == success_result
    assert failed_delivery_transport.publications == success_transport.publications


def test_zero_telemetry_fails_before_weights_publication(tmp_path: Path) -> None:
    root, corpus_bytes = _install_repo(
        tmp_path,
        train_source="def train(context):\n    return context.model\n",
    )
    access, transport = _access(corpus_bytes)
    uri = "s3://bucket/candidates/silent.pt"
    with pytest.raises(ValueError, match="at least one valid scalar telemetry event"):
        _execute_training_stage(
            _stage_input(),  # type: ignore[arg-type]
            mldb_data_root=root,
            object_bytes=access,
            corpus_destination_root=tmp_path / "corpus-runtime",
            work_dir=tmp_path / "work",
            weights_uri=uri,
        )
    assert transport.publications == []
    assert uri not in transport.objects


def test_malformed_telemetry_fails_protocol_invocation_before_publication(tmp_path: Path) -> None:
    root, corpus_bytes = _install_repo(
        tmp_path,
        train_source=(
            "def train(context):\n"
            "    context.telemetry.report_scalar(group='', series='signal', value=1, step=0)\n"
            "    return context.model\n"
        ),
    )
    access, transport = _access(corpus_bytes)
    with pytest.raises(ValueError, match="TrainProtocol raised") as excinfo:
        _execute_training_stage(
            _stage_input(),  # type: ignore[arg-type]
            mldb_data_root=root,
            object_bytes=access,
            corpus_destination_root=tmp_path / "corpus-runtime",
            work_dir=tmp_path / "work",
            weights_uri="s3://bucket/candidates/malformed.pt",
        )
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "telemetry" in str(excinfo.value.__cause__)
    assert transport.publications == []


def test_protocol_exception_is_bounded_by_runtime(tmp_path: Path) -> None:
    root, corpus_bytes = _install_repo(
        tmp_path,
        train_source="def train(context):\n    raise RuntimeError('backend detail')\n",
    )
    with pytest.raises(ValueError, match="TrainProtocol raised") as excinfo:
        _run(tmp_path, root, corpus_bytes)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_publication_is_immutable_and_does_not_persist_formal_results(tmp_path: Path) -> None:
    root, corpus_bytes = _install_repo(tmp_path)
    access, transport = _access(corpus_bytes)
    uri = "s3://bucket/candidates/immutable.pt"
    transport.objects[uri] = b"existing different bytes"

    with pytest.raises(ValueError, match="sha256|byte length"):
        _execute_training_stage(
            _stage_input(),  # type: ignore[arg-type]
            mldb_data_root=root,
            object_bytes=access,
            corpus_destination_root=tmp_path / "corpus-runtime",
            work_dir=tmp_path / "work",
            weights_uri=uri,
        )
    assert transport.objects[uri] == b"existing different bytes"
    assert not (root / "demo" / "models").exists()
    assert not (root / "demo" / "training_results").exists()


def test_two_structurally_different_architectures_use_same_training_runtime(tmp_path: Path) -> None:
    linear_root, linear_corpus = _install_repo(tmp_path / "linear", family="linear")
    linear_result, _ = _run(tmp_path / "linear", linear_root, linear_corpus)
    assert linear_result["weights"]["format"] == "pytorch-state-dict/v1"

    stack_source = (
        "import torch.nn as nn\n"
        "def build():\n"
        "    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))\n"
    )
    stack_root, stack_corpus = _install_repo(
        tmp_path / "stack", architecture_source=stack_source, family="stack"
    )
    stack_result, stack_transport = _run(
        tmp_path / "stack", stack_root, stack_corpus,
        weights_uri="s3://bucket/candidates/stack.pt",
    )
    state = torch.load(
        BytesIO(stack_transport.objects[stack_result["weights"]["uri"]]),
        map_location="cpu",
        weights_only=True,
    )
    assert set(state) == {"0.weight", "0.bias", "2.weight", "2.bias"}


def test_runtime_source_contains_no_specialization_or_persistence_dependencies() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "training" / "runtime.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "rotated-fcos",
        "tile-classifier",
        "classifier",
        "detector",
        "TrainingResult(",
        "Model(",
        "AttemptSummary",
        "TerminalCandidate",
        "clearml",
        "boto3",
        "importlib",
    ):
        assert forbidden not in source
