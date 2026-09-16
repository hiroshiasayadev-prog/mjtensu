from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mldb_v2.src.common.telemetry import _AcceptedScalarEvent, _RecordingTelemetryReporter
from mldb_v2.src.evaluation import runtime as evaluation_runtime
from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate
from mldb_v2.src.evaluation.runtime import _execute_evaluation_stage
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.training.canonical_weights import _serialize_canonical_state_dict

MODEL_ID = "demo/run-v1-trial-0001-model"
TRAINING_RESULT_ID = "demo/run-v1-trial-0001-train"
WEIGHTS_URI = "s3://bucket/models/model.pt"
CORPUS_URI = "s3://bucket/eval-corpus/data.bin"


class DictTransport:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
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


def _write_namespace(root: Path) -> None:
    _write_json(
        root / "demo" / "namespace.yaml",
        {
            "schema": "mjtensu.mldb-v2/namespace/v1",
            "id": "demo",
            "name": "Demo",
            "description": "",
        },
    )


def _architecture_document(
    *,
    task: str = "demo/task-v1",
    status: str = "sealed",
    family: str = "toy",
) -> dict[str, object]:
    implementation: dict[str, object] = {
        "framework": "pytorch",
        "entrypoint": "build",
    }
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
        "interface": {
            "input": {"kind": "tensor"},
            "output": {"kind": "tensor"},
        },
        "structure": {"summary": family},
    }


def _protocol_document(
    *,
    task: str = "demo/task-v1",
    status: str = "sealed",
    parameters: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
    artifacts: dict[str, object] | None = None,
) -> dict[str, object]:
    implementation: dict[str, object] = {"entrypoint": "evaluate"}
    if status == "sealed":
        implementation["sha256"] = "2" * 64
    return {
        "schema": "mjtensu.mldb-v2/evaluation-protocol/v1",
        "id": "demo/eval-v1",
        "status": status,
        "task": task,
        "name": "Evaluation",
        "description": "",
        "implementation": implementation,
        "parameters": parameters
        if parameters is not None
        else {"threshold": {"default": 0.5, "type": "number", "minimum": 0.0, "maximum": 1.0}},
        "metrics": metrics
        if metrics is not None
        else {"score": {"type": "number", "required": True}},
        "artifacts": artifacts if artifacts is not None else {},
    }


def _install_pinned_root(
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
    evaluate_source: str | None = None,
    family: str = "toy",
    parameters: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
    artifacts: dict[str, object] | None = None,
) -> tuple[Path, bytes]:
    root = tmp_path / "pinned"
    _write_namespace(root)
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

    corpus_bytes = b"evaluation corpus\n"
    manifest_entry = {
        "path": "data.bin",
        "bytes": len(corpus_bytes),
        "sha256": _sha(corpus_bytes),
    }
    manifest_bytes = json.dumps(
        manifest_entry, separators=(",", ":")
    ).encode() + b"\n"
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
            "storage": {"root_uri": "s3://bucket/eval-corpus"},
            "manifest": {
                "file": "corpus-v1.manifest.jsonl",
                "sha256": _sha(manifest_bytes),
                "entries": 1,
            },
            "representation": {"kind": "binary"},
            "splits": {"eval": 1},
        },
    )

    architecture_dir = root / "demo" / "architectures"
    _write_json(
        architecture_dir / "arch-v1.yaml",
        _architecture_document(
            task=architecture_task,
            status=architecture_status,
            family=family,
        ),
    )
    architecture_dir.mkdir(parents=True, exist_ok=True)
    (architecture_dir / "arch-v1.py").write_text(
        architecture_source
        or "import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n",
        encoding="utf-8",
    )

    protocol_dir = root / "demo" / "evaluation_protocols"
    _write_json(
        protocol_dir / "eval-v1.yaml",
        _protocol_document(
            task=protocol_task,
            status=protocol_status,
            parameters=parameters,
            metrics=metrics,
            artifacts=artifacts,
        ),
    )
    protocol_dir.mkdir(parents=True, exist_ok=True)
    (protocol_dir / "eval-v1.py").write_text(
        evaluate_source
        or (
            "from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate\n"
            "def evaluate(context):\n"
            "    return EvaluationCandidate(metrics={'score': 0.75}, artifacts={})\n"
        ),
        encoding="utf-8",
    )
    return root, corpus_bytes


def _weight_ref(data: bytes) -> dict[str, object]:
    return {
        "uri": WEIGHTS_URI,
        "bytes": len(data),
        "sha256": _sha(data),
        "format": "pytorch-state-dict/v1",
    }


def _training_result(
    weight_data: bytes,
    **overrides: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "mjtensu.mldb-v2/training-result/v1",
        "id": TRAINING_RESULT_ID,
        "study_result": "demo/source-run-v1",
        "plan": "demo/source-plan-v1",
        "trial": "trial-0001",
        "task": "demo/task-v1",
        "architecture": "demo/arch-v1",
        "corpus": "demo/train-corpus-v1",
        "train_protocol": "demo/train-v1",
        "parameters": {"epochs": 1},
        "seed": 42,
        "source_commit": "b" * 40,
        "attempts": [
            {
                "backend": "fake",
                "execution_id": "attempt-1",
                "status": "completed",
                "started_at": "2026-09-13T00:00:00Z",
                "ended_at": "2026-09-13T00:00:01Z",
                "diagnostic": None,
            }
        ],
        "status": "completed",
        "diagnostic": None,
        "result": {"weights": _weight_ref(weight_data), "model": MODEL_ID},
    }
    value.update(overrides)
    return value


def _install_runtime_root(
    tmp_path: Path,
    weight_data: bytes,
    *,
    model: dict[str, object] | None = None,
    training_result: dict[str, object] | None = None,
) -> Path:
    root = tmp_path / "runtime"
    _write_namespace(root)
    _write_json(
        root / "demo" / "models" / "run-v1-trial-0001-model.yaml",
        model
        or {
            "schema": "mjtensu.mldb-v2/model/v1",
            "id": MODEL_ID,
            "training_result": TRAINING_RESULT_ID,
        },
    )
    _write_json(
        root / "demo" / "training_results" / "run-v1-trial-0001-train.yaml",
        training_result or _training_result(weight_data),
    )
    return root


def _runtime_model(weight_data: bytes, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "model": MODEL_ID,
        "training_result": TRAINING_RESULT_ID,
        "task": "demo/task-v1",
        "architecture": "demo/arch-v1",
        "weights": _weight_ref(weight_data),
    }
    value.update(overrides)
    return value


def _stage_input(
    weight_data: bytes,
    *,
    runtime_model: object | None = None,
    **stage_overrides: object,
) -> dict[str, object]:
    stage: dict[str, object] = {
        "name": "holdout",
        "task": "demo/task-v1",
        "corpus": "demo/corpus-v1",
        "evaluation_protocol": "demo/eval-v1",
        "parameters": {"threshold": 0.25},
    }
    stage.update(stage_overrides)
    return {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": "demo/run-v2",
        "plan": "demo/plan-v1",
        "plan_sha256": "0" * 64,
        "trial": "trial-0001",
        "kind": "evaluation",
        "coordinate": "eval-0001",
        "source_commit": "a" * 40,
        "pins": [],
        "stage": stage,
        "runtime_model": _runtime_model(weight_data)
        if runtime_model is None
        else runtime_model,
    }


def _access(
    corpus_bytes: bytes,
    weight_data: bytes,
) -> tuple[_ObjectByteAccess, DictTransport]:
    transport = DictTransport(
        {CORPUS_URI: corpus_bytes, WEIGHTS_URI: weight_data}
    )
    return _ObjectByteAccess(transport), transport


def _run(
    tmp_path: Path,
    pinned_root: Path,
    runtime_root: Path,
    corpus_bytes: bytes,
    weight_data: bytes,
    stage_input: dict[str, object] | None = None,
    *,
    artifact_uris: dict[str, str] | None = None,
    telemetry_sink=None,
):
    access, transport = _access(corpus_bytes, weight_data)
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    result = _execute_evaluation_stage(
        stage_input or _stage_input(weight_data),  # type: ignore[arg-type]
        pinned_mldb_data_root=pinned_root,
        runtime_mldb_data_root=runtime_root,
        object_bytes=access,
        corpus_destination_root=tmp_path / "corpus-runtime",
        work_dir=work_dir,
        artifact_uris=artifact_uris or {},
        telemetry_sink=telemetry_sink,
    )
    return result, transport, work_dir


def _default_roots(tmp_path: Path):
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path)
    module = nn.Linear(3, 2)
    weight_data = _serialize_canonical_state_dict(module.state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    return pinned_root, runtime_root, corpus_bytes, weight_data


def _capture_telemetry(monkeypatch: pytest.MonkeyPatch) -> dict[str, _RecordingTelemetryReporter]:
    captured: dict[str, _RecordingTelemetryReporter] = {}

    def factory(*, sink=None) -> _RecordingTelemetryReporter:
        reporter = _RecordingTelemetryReporter(sink=sink)
        captured["reporter"] = reporter
        return reporter

    monkeypatch.setattr(evaluation_runtime, "_RecordingTelemetryReporter", factory)
    return captured


def test_valid_evaluation_stage_uses_separate_pinned_and_runtime_roots(tmp_path: Path) -> None:
    pinned_root, runtime_root, corpus_bytes, weight_data = _default_roots(tmp_path)
    assert not (pinned_root / "demo" / "models").exists()
    assert not (pinned_root / "demo" / "training_results").exists()
    assert not (runtime_root / "demo" / "architectures").exists()
    assert not (runtime_root / "demo" / "evaluation_protocols").exists()

    result, transport, _ = _run(
        tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data
    )
    assert result == {"metrics": {"score": 0.75}, "artifacts": {}}
    assert transport.reads == [CORPUS_URI, WEIGHTS_URI]


def test_validated_metric_is_automatically_projected_with_stage_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned_root, runtime_root, corpus_bytes, weight_data = _default_roots(tmp_path)
    captured = _capture_telemetry(monkeypatch)
    result, _, _ = _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data)
    assert result["metrics"] == {"score": 0.75}
    assert captured["reporter"].accepted_events == (
        _AcceptedScalarEvent(group="holdout", series="score", value=0.75, step=0),
    )


def test_validated_metric_is_delivered_to_sink_with_exact_stage_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics = {"strange_scalar": {"type": "number", "required": True}}
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path, metrics=metrics)
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    monkeypatch.setattr(
        evaluation_runtime,
        "_load_evaluation_callable",
        lambda *_a, **_k: lambda _ctx: EvaluationCandidate(
            metrics={"strange_scalar": 7.25}, artifacts={}
        ),
    )
    delivered: list[_AcceptedScalarEvent] = []
    result, _, _ = _run(
        tmp_path,
        pinned_root,
        runtime_root,
        corpus_bytes,
        weight_data,
        _stage_input(weight_data, name="final-check"),
        telemetry_sink=delivered.append,
    )
    assert result["metrics"] == {"strange_scalar": 7.25}
    assert delivered == [
        _AcceptedScalarEvent(
            group="final-check", series="strange_scalar", value=7.25, step=0
        )
    ]


def test_sink_failure_does_not_change_evaluation_result_or_artifact_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = {
        "custom": {
            "format": "custom-bin",
            "schema": "demo/custom-output/v1",
            "required": True,
        }
    }
    metrics = {"score": {"type": "number", "required": True}}
    pinned_root, corpus_bytes = _install_pinned_root(
        tmp_path, metrics=metrics, artifacts=artifacts
    )
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)

    def evaluate(context):
        path = context.work_dir / "custom.bin"
        path.write_bytes(b"custom payload")
        return EvaluationCandidate(metrics={"score": 0.75}, artifacts={"custom": path})

    monkeypatch.setattr(evaluation_runtime, "_load_evaluation_callable", lambda *_a, **_k: evaluate)
    uri = "s3://bucket/candidates/custom.bin"
    delivered: list[_AcceptedScalarEvent] = []
    success, success_transport, _ = _run(
        tmp_path / "success",
        pinned_root,
        runtime_root,
        corpus_bytes,
        weight_data,
        artifact_uris={"custom": uri},
        telemetry_sink=delivered.append,
    )

    def failing_sink(_event: _AcceptedScalarEvent) -> None:
        raise RuntimeError("clearml delivery unavailable")

    failed_delivery, failed_transport, _ = _run(
        tmp_path / "failed-delivery",
        pinned_root,
        runtime_root,
        corpus_bytes,
        weight_data,
        artifact_uris={"custom": uri},
        telemetry_sink=failing_sink,
    )
    assert len(delivered) == 1
    assert failed_delivery == success
    assert failed_transport.objects[uri] == success_transport.objects[uri]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "training", "kind"),
        ("coordinate", None, "coordinate"),
        ("coordinate", "eval-0000", "coordinate"),
        ("trial", "trial-0000", "trial"),
    ],
)
def test_invalid_variant_coordinate_or_trial_is_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    pinned_root, runtime_root, corpus_bytes, weight_data = _default_roots(tmp_path)
    stage_input = _stage_input(weight_data)
    stage_input[field] = value
    with pytest.raises(ValueError, match=message):
        _run(
            tmp_path,
            pinned_root,
            runtime_root,
            corpus_bytes,
            weight_data,
            stage_input,
        )


def test_missing_or_malformed_runtime_model_is_rejected(tmp_path: Path) -> None:
    pinned_root, runtime_root, corpus_bytes, weight_data = _default_roots(tmp_path)
    for runtime_model in (None, {}, {"model": MODEL_ID}):
        stage_input = _stage_input(weight_data)
        stage_input["runtime_model"] = runtime_model
        with pytest.raises(ValueError, match="RuntimeModel"):
            _run(
                tmp_path,
                pinned_root,
                runtime_root,
                corpus_bytes,
                weight_data,
                stage_input,
            )


def test_stage_and_top_level_shapes_are_exact(tmp_path: Path) -> None:
    pinned_root, runtime_root, corpus_bytes, weight_data = _default_roots(tmp_path)
    stage_input = _stage_input(weight_data)
    stage_input["stage"]["extra"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="EvaluationStage fields"):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data, stage_input)

    stage_input = _stage_input(weight_data)
    stage_input["unexpected"] = 1
    with pytest.raises(ValueError, match="EvaluationStageInput fields"):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data, stage_input)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "bad"),
        ("training_result", "bad"),
        ("task", "bad"),
        ("architecture", "bad"),
    ],
)
def test_runtime_model_requires_valid_typed_ids(
    tmp_path: Path, field: str, value: object
) -> None:
    pinned_root, runtime_root, corpus_bytes, weight_data = _default_roots(tmp_path)
    stage_input = _stage_input(weight_data)
    stage_input["runtime_model"][field] = value  # type: ignore[index]
    with pytest.raises(ValueError):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data, stage_input)


def test_runtime_model_snapshot_must_match_accepted_lineage(tmp_path: Path) -> None:
    pinned_root, runtime_root, corpus_bytes, weight_data = _default_roots(tmp_path)
    mutations = [
        {"model": "demo/other-model-v1"},
        {"training_result": "demo/other-train-v1"},
        {"task": "demo/other-task-v1"},
        {"architecture": "demo/other-arch-v1"},
        {"weights": {**_weight_ref(weight_data), "uri": "s3://bucket/other.pt"}},
    ]
    for mutation in mutations:
        stage_input = _stage_input(weight_data)
        stage_input["runtime_model"].update(mutation)  # type: ignore[union-attr]
        with pytest.raises((ValueError, FileNotFoundError)):
            _run(
                tmp_path,
                pinned_root,
                runtime_root,
                corpus_bytes,
                weight_data,
                stage_input,
            )


def test_training_result_identity_and_completed_payload_are_enforced(tmp_path: Path) -> None:
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path)
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    bad = _training_result(weight_data, status="failed", result=None,
                           diagnostic={"code": "failed", "message": "x"})
    runtime_root = _install_runtime_root(tmp_path, weight_data, training_result=bad)
    with pytest.raises(ValueError, match="completed TrainingResult"):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data)

    bad = _training_result(weight_data)
    bad["result"] = {"weights": _weight_ref(weight_data), "model": "demo/other-model-v1"}
    runtime_root = _install_runtime_root(tmp_path / "model-mismatch", weight_data,
                                         training_result=bad)
    with pytest.raises(ValueError, match="result.model"):
        _run(tmp_path / "model-mismatch", pinned_root, runtime_root,
             corpus_bytes, weight_data)


@pytest.mark.parametrize(
    ("pinned_overrides", "runtime_override", "message"),
    [
        ({"corpus_task": "demo/other-task-v1"}, None, "Corpus task"),
        ({"protocol_task": "demo/other-task-v1"}, None, "EvaluationProtocol task"),
        ({"architecture_task": "demo/other-task-v1"}, None, "Architecture task"),
        ({}, {"task": "demo/other-task-v1"}, "RuntimeModel task"),
    ],
)
def test_task_compatibility_is_exact(
    tmp_path: Path,
    pinned_overrides: dict[str, object],
    runtime_override: dict[str, object] | None,
    message: str,
) -> None:
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path, **pinned_overrides)
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    stage_input = _stage_input(weight_data)
    if runtime_override:
        stage_input["runtime_model"].update(runtime_override)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data, stage_input)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({}, "incomplete"),
        ({"threshold": 0.25, "unknown": 1}, "unknown"),
        ({"threshold": "bad"}, "declared type"),
        ({"threshold": 2.0}, "maximum"),
    ],
)
def test_complete_resolved_parameters_are_required(
    tmp_path: Path, parameters: dict[str, object], message: str
) -> None:
    pinned_root, runtime_root, corpus_bytes, weight_data = _default_roots(tmp_path)
    with pytest.raises(ValueError, match=message):
        _run(
            tmp_path,
            pinned_root,
            runtime_root,
            corpus_bytes,
            weight_data,
            _stage_input(weight_data, parameters=parameters),
        )


@pytest.mark.parametrize("target", ["task", "corpus", "architecture", "protocol"])
def test_execution_definitions_must_be_sealed(tmp_path: Path, target: str) -> None:
    kwargs = {
        "task_status": "sealed",
        "corpus_status": "sealed",
        "architecture_status": "sealed",
        "protocol_status": "sealed",
    }
    kwargs[f"{target}_status"] = "draft"
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path, **kwargs)
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    with pytest.raises(ValueError, match="sealed"):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data)


@pytest.mark.parametrize(
    "weights_mutation",
    [
        {"uri": "s3://bucket/models/other.pt"},
        {"bytes": 1},
        {"sha256": "0" * 64},
        {"format": "checkpoint/v1"},
    ],
)
def test_runtime_weight_snapshot_must_match_exact_ref(
    tmp_path: Path, weights_mutation: dict[str, object]
) -> None:
    pinned_root, runtime_root, corpus_bytes, weight_data = _default_roots(tmp_path)
    stage_input = _stage_input(weight_data)
    weights = dict(stage_input["runtime_model"]["weights"])  # type: ignore[index]
    weights.update(weights_mutation)
    stage_input["runtime_model"]["weights"] = weights  # type: ignore[index]
    with pytest.raises(ValueError):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data, stage_input)


def test_incompatible_canonical_state_is_rejected_strictly(tmp_path: Path) -> None:
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path)
    weight_data = _serialize_canonical_state_dict(nn.Linear(4, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    with pytest.raises(ValueError, match="strictly compatible"):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data)


def test_exact_loaded_model_and_evaluation_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path)
    source = nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(4.0)
    weight_data = _serialize_canonical_state_dict(source.state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    captured: dict[str, object] = {}

    def evaluate_spy(context):
        captured["context"] = context
        return EvaluationCandidate(metrics={"score": 1.0}, artifacts={})

    monkeypatch.setattr(evaluation_runtime, "_load_evaluation_callable",
                        lambda *_args, **_kwargs: evaluate_spy)
    result, transport, work_dir = _run(
        tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data
    )
    assert result == {"metrics": {"score": 1.0}, "artifacts": {}}
    context = captured["context"]
    assert set(context.__dict__) == {"task", "corpus", "model", "parameters", "telemetry", "work_dir"}
    assert context.task["id"] == "demo/task-v1"
    assert context.corpus.definition["id"] == "demo/corpus-v1"
    assert context.model.definition["id"] == MODEL_ID
    assert context.model.training_result["id"] == TRAINING_RESULT_ID
    assert context.model.architecture["id"] == "demo/arch-v1"
    assert context.parameters == {"threshold": 0.25}
    assert context.work_dir == work_dir
    assert torch.equal(context.model.module.weight, source.weight)
    assert torch.equal(context.model.module.bias, source.bias)
    assert transport.reads == [CORPUS_URI, WEIGHTS_URI]


def test_metric_only_candidate_and_optional_metric_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics = {
        "count": {"type": "integer", "required": True},
        "optional_score": {"type": "number", "required": False},
    }
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path, metrics=metrics)
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    monkeypatch.setattr(
        evaluation_runtime,
        "_load_evaluation_callable",
        lambda *_a, **_k: lambda _ctx: EvaluationCandidate(metrics={"count": 3}, artifacts={}),
    )
    result, _, _ = _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data)
    assert result == {"metrics": {"count": 3}, "artifacts": {}}


@pytest.mark.parametrize(
    ("declarations", "candidate_metrics", "message"),
    [
        ({"required": {"type": "number", "required": True}}, {}, "missing required"),
        ({"score": {"type": "number", "required": True}}, {"score": 1.0, "extra": 2.0}, "undeclared"),
        ({"count": {"type": "integer", "required": True}}, {"count": True}, "exact integer"),
        ({"count": {"type": "integer", "required": True}}, {"count": 1.5}, "exact integer"),
        ({"score": {"type": "number", "required": True}}, {"score": True}, "exact number"),
        ({"score": {"type": "number", "required": True}}, {"score": float("nan")}, "finite"),
        ({"score": {"type": "number", "required": True}}, {"score": float("inf")}, "finite"),
    ],
)
def test_metric_candidate_validation_rejects_invalid_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declarations: dict[str, object],
    candidate_metrics: dict[str, object],
    message: str,
) -> None:
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path, metrics=declarations)
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    monkeypatch.setattr(evaluation_runtime, "_load_evaluation_callable",
                        lambda *_a, **_k: lambda _ctx: EvaluationCandidate(metrics=candidate_metrics, artifacts={}))
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data)


@pytest.mark.parametrize("invalid_value", [True, float("nan"), float("inf")])
def test_invalid_candidate_metric_is_not_projected_before_formal_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_value: object
) -> None:
    pinned_root, corpus_bytes = _install_pinned_root(
        tmp_path, metrics={"score": {"type": "number", "required": True}}
    )
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    captured = _capture_telemetry(monkeypatch)
    monkeypatch.setattr(
        evaluation_runtime,
        "_load_evaluation_callable",
        lambda *_a, **_k: lambda _ctx: EvaluationCandidate(
            metrics={"score": invalid_value}, artifacts={}
        ),
    )
    with pytest.raises(ValueError):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data)
    assert captured["reporter"].accepted_events == ()


def test_explicit_and_automatic_evaluation_telemetry_preserve_duplicates_and_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned_root, runtime_root, corpus_bytes, weight_data = _default_roots(tmp_path)
    captured = _capture_telemetry(monkeypatch)

    def evaluate(context):
        context.telemetry.report_scalar(
            group="holdout", series="score", value=0.5, step=0
        )
        return EvaluationCandidate(metrics={"score": 0.75}, artifacts={})

    monkeypatch.setattr(evaluation_runtime, "_load_evaluation_callable", lambda *_a, **_k: evaluate)
    result, _, _ = _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data)
    assert result["metrics"] == {"score": 0.75}
    assert captured["reporter"].accepted_events == (
        _AcceptedScalarEvent(group="holdout", series="score", value=0.5, step=0),
        _AcceptedScalarEvent(group="holdout", series="score", value=0.75, step=0),
    )


def test_artifact_only_custom_candidate_publication_and_optional_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = {
        "custom": {"format": "custom-bin", "schema": "demo/custom-output/v3", "required": True},
        "optional": {"format": "text", "schema": "demo/optional-output/v1", "required": False},
    }
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path, metrics={}, artifacts=artifacts)
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)

    def evaluate(context):
        context.telemetry.report_scalar(
            group="fixture", series="artifact_ready", value=1, step=0
        )
        path = context.work_dir / "custom.bin"
        path.write_bytes(b"custom payload")
        return EvaluationCandidate(metrics={}, artifacts={"custom": path})

    monkeypatch.setattr(evaluation_runtime, "_load_evaluation_callable", lambda *_a, **_k: evaluate)
    uris = {"custom": "s3://bucket/candidates/custom.bin",
            "optional": "s3://bucket/candidates/optional.txt"}
    result, transport, _ = _run(tmp_path, pinned_root, runtime_root, corpus_bytes,
                                 weight_data, artifact_uris=uris)
    assert result["metrics"] == {}
    assert set(result["artifacts"]) == {"custom"}
    ref = result["artifacts"]["custom"]
    assert ref["uri"] == uris["custom"]
    assert ref["format"] == "custom-bin"
    assert ref["schema"] == "demo/custom-output/v3"
    assert ref["bytes"] == len(b"custom payload")
    assert ref["sha256"] == _sha(b"custom payload")
    assert transport.objects[uris["custom"]] == b"custom payload"
    assert uris["optional"] not in transport.objects


def test_artifact_only_candidate_without_telemetry_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = {
        "custom": {"format": "custom-bin", "schema": "demo/custom-output/v3", "required": True}
    }
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path, metrics={}, artifacts=artifacts)
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    access, transport = _access(corpus_bytes, weight_data)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    def evaluate(context):
        path = context.work_dir / "custom.bin"
        path.write_bytes(b"custom payload")
        return EvaluationCandidate(metrics={}, artifacts={"custom": path})

    monkeypatch.setattr(evaluation_runtime, "_load_evaluation_callable", lambda *_a, **_k: evaluate)
    uri = "s3://bucket/candidates/custom.bin"
    with pytest.raises(ValueError, match="at least one valid scalar telemetry event"):
        _execute_evaluation_stage(
            _stage_input(weight_data),  # type: ignore[arg-type]
            pinned_mldb_data_root=pinned_root,
            runtime_mldb_data_root=runtime_root,
            object_bytes=access,
            corpus_destination_root=tmp_path / "corpus-runtime",
            work_dir=work_dir,
            artifact_uris={"custom": uri},
        )
    assert transport.publications == []
    assert uri not in transport.objects


@pytest.mark.parametrize(
    ("candidate_artifacts", "message"),
    [
        ({}, "missing required"),
        ({"required": Path("missing.bin"), "extra": Path("extra.bin")}, "undeclared"),
    ],
)
def test_artifact_candidate_validation_rejects_missing_or_undeclared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_artifacts: dict[str, Path],
    message: str,
) -> None:
    declarations = {
        "required": {"format": "bin", "schema": "demo/required/v1", "required": True}
    }
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path, metrics={}, artifacts=declarations)
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)

    def evaluate(context):
        context.telemetry.report_scalar(group="fixture", series="probe", value=1, step=0)
        return EvaluationCandidate(metrics={}, artifacts=candidate_artifacts)

    monkeypatch.setattr(evaluation_runtime, "_load_evaluation_callable", lambda *_a, **_k: evaluate)
    with pytest.raises(ValueError, match=message):
        _run(
            tmp_path,
            pinned_root,
            runtime_root,
            corpus_bytes,
            weight_data,
            artifact_uris={"required": "s3://bucket/candidates/required.bin"},
        )


def test_artifact_path_must_be_regular_file_beneath_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declarations = {
        "artifact": {"format": "bin", "schema": "demo/artifact/v1", "required": True}
    }
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path, metrics={}, artifacts=declarations)
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")

    def evaluate(context):
        context.telemetry.report_scalar(group="fixture", series="probe", value=1, step=0)
        return EvaluationCandidate(metrics={}, artifacts={"artifact": outside})

    monkeypatch.setattr(evaluation_runtime, "_load_evaluation_callable", lambda *_a, **_k: evaluate)
    with pytest.raises(ValueError, match="escapes work_dir"):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data,
             artifact_uris={"artifact": "s3://bucket/candidates/artifact.bin"})


def test_artifact_directory_and_symlink_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declarations = {
        "artifact": {"format": "bin", "schema": "demo/artifact/v1", "required": True}
    }
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path, metrics={}, artifacts=declarations)
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    uri_map = {"artifact": "s3://bucket/candidates/artifact.bin"}

    def evaluate_directory(context):
        context.telemetry.report_scalar(group="fixture", series="probe", value=1, step=0)
        directory = context.work_dir / "artifact-dir"
        directory.mkdir()
        return EvaluationCandidate(metrics={}, artifacts={"artifact": directory})

    monkeypatch.setattr(evaluation_runtime, "_load_evaluation_callable",
                        lambda *_a, **_k: evaluate_directory)
    with pytest.raises(ValueError, match="regular file"):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data,
             artifact_uris=uri_map)


def test_artifact_symlink_is_rejected_when_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declarations = {
        "artifact": {"format": "bin", "schema": "demo/artifact/v1", "required": True}
    }
    pinned_root, corpus_bytes = _install_pinned_root(tmp_path, metrics={}, artifacts=declarations)
    weight_data = _serialize_canonical_state_dict(nn.Linear(3, 2).state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)

    def evaluate(context):
        context.telemetry.report_scalar(group="fixture", series="probe", value=1, step=0)
        target = context.work_dir / "target.bin"
        target.write_bytes(b"payload")
        link = context.work_dir / "link.bin"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation unavailable on this host")
        return EvaluationCandidate(metrics={}, artifacts={"artifact": link})

    monkeypatch.setattr(evaluation_runtime, "_load_evaluation_callable", lambda *_a, **_k: evaluate)
    with pytest.raises(ValueError, match="symlink"):
        _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data,
             artifact_uris={"artifact": "s3://bucket/candidates/artifact.bin"})


def test_runtime_does_not_persist_formal_result_or_fabricate_backend_provenance(
    tmp_path: Path
) -> None:
    pinned_root, runtime_root, corpus_bytes, weight_data = _default_roots(tmp_path)
    result, _, _ = _run(
        tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data
    )
    assert set(result) == {"metrics", "artifacts"}
    assert not any(
        key in result
        for key in ("attempts", "execution_id", "execution_ids", "backend", "status", "diagnostic")
    )
    assert not (runtime_root / "demo" / "evaluation_results").exists()
    assert not (pinned_root / "demo" / "evaluation_results").exists()


@pytest.mark.parametrize(
    ("family", "architecture_source", "module"),
    [
        ("classifier-like", "import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n", nn.Linear(3, 2)),
        (
            "detector-like",
            "import torch.nn as nn\ndef build():\n    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))\n",
            nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)),
        ),
    ],
)
def test_classifier_and_detector_like_architectures_share_same_runtime(
    tmp_path: Path, family: str, architecture_source: str, module: nn.Module
) -> None:
    pinned_root, corpus_bytes = _install_pinned_root(
        tmp_path, family=family, architecture_source=architecture_source
    )
    weight_data = _serialize_canonical_state_dict(module.state_dict())
    runtime_root = _install_runtime_root(tmp_path, weight_data)
    result, _, _ = _run(tmp_path, pinned_root, runtime_root, corpus_bytes, weight_data)
    assert result == {"metrics": {"score": 0.75}, "artifacts": {}}


def test_runtime_source_contains_no_family_or_persistence_specialization() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src" / "evaluation" / "runtime.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "rotated-fcos",
        "tile-classifier",
        "classifier",
        "detector",
        "AttemptSummary",
        "TerminalCandidate",
        "EvaluationResult(",
        "clearml",
        "boto3",
        "importlib",
    ):
        assert forbidden not in source
