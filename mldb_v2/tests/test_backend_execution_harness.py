from __future__ import annotations

import copy
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from mldb_v2.src.backend import execution_harness as harness_module
from mldb_v2.src.backend.execution_harness import CommonExecutionHarness, ExecutionHarness
from mldb_v2.src.common.telemetry import _AcceptedScalarEvent
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study._plan_build import _content_digest, _plan_id, _validate_study_plan


class DictTransport:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.publications: list[tuple[str, bytes]] = []

    def read_bytes(self, uri: str) -> bytes:
        return self.objects[uri]

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        if uri in self.objects and self.objects[uri] != data:
            raise FileExistsError(uri)
        self.objects[uri] = data
        self.publications.append((uri, data))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _pin(kind: str, entity_id: str, yaml_bytes: bytes, **extra: object) -> dict[str, object]:
    value: dict[str, object] = {
        "kind": kind,
        "id": entity_id,
        "yaml_sha256": _sha(yaml_bytes),
        "companion_sha256": None,
        "sources": [],
        "manifest_sha256": None,
        "manifest_entries": None,
    }
    value.update(extra)
    return value


def _finalize_plan(root: Path, plan: dict[str, object]) -> dict[str, object]:
    plan = copy.deepcopy(plan)
    digest = _content_digest(plan)
    plan["content_sha256"] = digest
    plan["id"] = _plan_id(str(plan["study"]), digest)
    validated = dict(_validate_study_plan(plan))
    _, local_id = str(validated["id"]).split("/", 1)
    _write_json(root / "demo" / "study_plans" / f"{local_id}.yaml", validated)
    return validated


def _install_fixture(
    tmp_path: Path,
    *,
    architecture_source: str,
    train_source: bytes | None = None,
) -> dict[str, object]:
    repo = tmp_path / "repo"
    root = repo / "mldb_data"
    (repo / "mldb_v2" / "src").mkdir(parents=True)
    (repo / "mldb_v2" / "src" / "marker.py").write_text("VALUE = 1\n", encoding="utf-8")
    shared = repo / "ml_impl" / "shared.py"
    shared.parent.mkdir(parents=True)
    shared.write_text("TOKEN = 7\n", encoding="utf-8")

    namespace = _write_json(
        root / "demo" / "namespace.yaml",
        {"schema": "mjtensu.mldb-v2/namespace/v1", "id": "demo", "name": "Demo", "description": ""},
    )
    task = _write_json(
        root / "demo" / "tasks" / "task-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/task/v1",
            "id": "demo/task-v1",
            "status": "sealed",
            "name": "Task",
            "problem_type": "regression",
            "description": "",
            "input": {},
            "target": {"type": "continuous"},
            "semantics": {},
            "scope": {},
        },
    )
    corpus_payload = b"corpus-object\n"
    manifest_bytes = (
        json.dumps(
            {"path": "data.bin", "bytes": len(corpus_payload), "sha256": _sha(corpus_payload)},
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    corpus_dir = root / "demo" / "corpora"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "corpus-v1.manifest.jsonl").write_bytes(manifest_bytes)
    builder_bytes = b"raise RuntimeError('Corpus builder must never execute')\n"
    (corpus_dir / "corpus-v1.py").write_bytes(builder_bytes)
    corpus = _write_json(
        corpus_dir / "corpus-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/corpus/v1",
            "id": "demo/corpus-v1",
            "status": "sealed",
            "task": "demo/task-v1",
            "description": "",
            "storage": {"root_uri": "s3://bucket/corpus-v1"},
            "manifest": {"file": "corpus-v1.manifest.jsonl", "sha256": _sha(manifest_bytes), "entries": 1},
            "representation": {"kind": "binary"},
            "splits": {"train": 1},
            "builder": {"entrypoint": "build", "sha256": _sha(builder_bytes)},
        },
    )
    arch_dir = root / "demo" / "architectures"
    arch_dir.mkdir(parents=True, exist_ok=True)
    arch_bytes = architecture_source.encode()
    (arch_dir / "arch-v1.py").write_bytes(arch_bytes)
    architecture = _write_json(
        arch_dir / "arch-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/architecture/v1",
            "id": "demo/arch-v1",
            "status": "sealed",
            "task": "demo/task-v1",
            "name": "Architecture",
            "family": "generic-fixture",
            "description": "",
            "implementation": {
                "framework": "pytorch",
                "entrypoint": "build",
                "sha256": _sha(arch_bytes),
                "sources": [{"path": "ml_impl/shared.py", "sha256": _sha(shared.read_bytes())}],
            },
            "interface": {"input": {"kind": "tensor"}, "output": {"kind": "tensor"}},
            "structure": {"summary": "fixture"},
        },
    )
    train_dir = root / "demo" / "train_protocols"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_bytes = train_source or (
        b"import torch\n"
        b"def train(context):\n"
        b"    with torch.no_grad():\n"
        b"        for parameter in context.model.parameters(): parameter.add_(0.25)\n"
        b"    context.telemetry.report_scalar(group='fixture', series='signal', value=1.0, step=0)\n"
        b"    return context.model\n"
    )
    (train_dir / "train-v1.py").write_bytes(train_bytes)
    train_protocol = _write_json(
        train_dir / "train-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/train-protocol/v1",
            "id": "demo/train-v1",
            "status": "sealed",
            "task": "demo/task-v1",
            "name": "Train",
            "description": "",
            "implementation": {"entrypoint": "train", "sha256": _sha(train_bytes)},
            "parameters": {"epochs": {"default": 1, "type": "integer", "minimum": 1, "maximum": 3}},
        },
    )
    eval_dir = root / "demo" / "evaluation_protocols"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_bytes = (
        b"from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate\n"
        b"def evaluate(context):\n"
        b"    target = context.work_dir / 'opaque.bin'\n"
        b"    target.write_bytes(b'opaque-output')\n"
        b"    return EvaluationCandidate(metrics={'strange_scalar': 7.25}, artifacts={'opaque_bundle': target})\n"
    )
    (eval_dir / "eval-v1.py").write_bytes(eval_bytes)
    eval_protocol = _write_json(
        eval_dir / "eval-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/evaluation-protocol/v1",
            "id": "demo/eval-v1",
            "status": "sealed",
            "task": "demo/task-v1",
            "name": "Evaluation",
            "description": "",
            "implementation": {"entrypoint": "evaluate", "sha256": _sha(eval_bytes)},
            "parameters": {"limit": {"default": 2, "type": "integer", "minimum": 1, "maximum": 5}},
            "metrics": {"strange_scalar": {"type": "number", "required": True}},
            "artifacts": {
                "opaque_bundle": {
                    "format": "opaque-bin",
                    "schema": "demo/opaque-output/v7",
                    "required": True,
                }
            },
        },
    )
    study = _write_json(
        root / "demo" / "studies" / "study-v1.yaml",
        {"schema": "mjtensu.mldb-v2/study/v1", "id": "demo/study-v1"},
    )

    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.com")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", "mldb_v2/src", "mldb_data", "ml_impl/shared.py")
    _git(repo, "commit", "-m", "fixture source")
    commit = _git(repo, "rev-parse", "HEAD")

    pins = [
        _pin("namespace", "demo", namespace),
        _pin("task", "demo/task-v1", task),
        _pin(
            "corpus",
            "demo/corpus-v1",
            corpus,
            companion_sha256=_sha(builder_bytes),
            manifest_sha256=_sha(manifest_bytes),
            manifest_entries=1,
        ),
        _pin(
            "architecture",
            "demo/arch-v1",
            architecture,
            companion_sha256=_sha(arch_bytes),
            sources=[{"path": "ml_impl/shared.py", "sha256": _sha(shared.read_bytes())}],
        ),
        _pin("train_protocol", "demo/train-v1", train_protocol, companion_sha256=_sha(train_bytes)),
        _pin("evaluation_protocol", "demo/eval-v1", eval_protocol, companion_sha256=_sha(eval_bytes)),
        _pin("study", "demo/study-v1", study),
    ]
    plan = _finalize_plan(
        root,
        {
            "schema": "mjtensu.mldb-v2/study-plan/v1",
            "id": "demo/placeholder",
            "content_sha256": "0" * 64,
            "study": "demo/study-v1",
            "source_commit": commit,
            "pins": pins,
            "trials": [
                {
                    "trial": "trial-0001",
                    "source": {
                        "kind": "training",
                        "task": "demo/task-v1",
                        "corpus": "demo/corpus-v1",
                        "architecture": "demo/arch-v1",
                        "train_protocol": "demo/train-v1",
                        "parameters": {"epochs": 1},
                        "seed": 42,
                    },
                    "evaluations": [
                        {
                            "coordinate": "eval-0001",
                            "stage": "final-check",
                            "task": "demo/task-v1",
                            "corpus": "demo/corpus-v1",
                            "evaluation_protocol": "demo/eval-v1",
                            "parameters": {"limit": 2},
                        }
                    ],
                }
            ],
        },
    )
    return {
        "repo": repo,
        "root": root,
        "commit": commit,
        "plan": plan,
        "corpus_payload": corpus_payload,
    }


def _training_input(fx: dict[str, object]) -> dict[str, object]:
    plan = fx["plan"]
    assert type(plan) is dict
    trial = plan["trials"][0]
    return {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": "demo/study-result-v1",
        "plan": plan["id"],
        "plan_sha256": plan["content_sha256"],
        "trial": "trial-0001",
        "kind": "training",
        "coordinate": None,
        "source_commit": plan["source_commit"],
        "pins": copy.deepcopy(plan["pins"]),
        "stage": {key: value for key, value in trial["source"].items() if key != "kind"},
        "runtime_model": None,
    }


def _harness(
    fx: dict[str, object],
    access: _ObjectByteAccess,
    tmp_path: Path,
    *,
    weights_uri: str | None = "s3://bucket/candidates/weights.pt",
    artifact_uris: dict[str, str] | None = None,
    suffix: str = "attempt",
    telemetry_sink=None,
) -> CommonExecutionHarness:
    return CommonExecutionHarness(
        repository_root=fx["repo"],
        pinned_mldb_data_root=fx["root"],
        runtime_mldb_data_root=fx["root"],
        object_bytes=access,
        corpus_destination_root=tmp_path / f"corpus-{suffix}",
        work_dir=tmp_path / f"work-{suffix}",
        training_weights_uri=weights_uri,
        evaluation_artifact_uris=artifact_uris,
        backend="fixture-backend",
        execution_id=f"exec-{suffix}",
        started_at="2026-09-13T00:00:00Z",
        telemetry_sink=telemetry_sink,
    )


def _install_runtime_lineage(
    fx: dict[str, object], training_candidate: dict[str, object]
) -> dict[str, object]:
    root = fx["root"]
    plan = fx["plan"]
    assert isinstance(root, Path) and type(plan) is dict
    weights = training_candidate["result"]["weights"]
    model_id = "demo/model-v1"
    training_result_id = "demo/training-result-v1"
    _write_json(
        root / "demo" / "models" / "model-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/model/v1",
            "id": model_id,
            "training_result": training_result_id,
        },
    )
    _write_json(
        root / "demo" / "training_results" / "training-result-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/training-result/v1",
            "id": training_result_id,
            "study_result": "demo/study-result-v1",
            "plan": plan["id"],
            "trial": "trial-0001",
            "task": "demo/task-v1",
            "architecture": "demo/arch-v1",
            "corpus": "demo/corpus-v1",
            "train_protocol": "demo/train-v1",
            "parameters": {"epochs": 1},
            "seed": 42,
            "source_commit": fx["commit"],
            "attempts": training_candidate["attempts"],
            "status": "completed",
            "diagnostic": None,
            "result": {"weights": weights, "model": model_id},
        },
    )
    return {
        "model": model_id,
        "training_result": training_result_id,
        "task": "demo/task-v1",
        "architecture": "demo/arch-v1",
        "weights": weights,
    }


def _evaluation_input(fx: dict[str, object], runtime_model: dict[str, object]) -> dict[str, object]:
    plan = fx["plan"]
    assert type(plan) is dict
    coordinate = plan["trials"][0]["evaluations"][0]
    return {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": "demo/study-result-v1",
        "plan": plan["id"],
        "plan_sha256": plan["content_sha256"],
        "trial": "trial-0001",
        "kind": "evaluation",
        "coordinate": coordinate["coordinate"],
        "source_commit": plan["source_commit"],
        "pins": copy.deepcopy(plan["pins"]),
        "stage": {
            "name": coordinate["stage"],
            "task": coordinate["task"],
            "corpus": coordinate["corpus"],
            "evaluation_protocol": coordinate["evaluation_protocol"],
            "parameters": coordinate["parameters"],
        },
        "runtime_model": runtime_model,
    }


def _rewrite_plan(fx: dict[str, object], mutate) -> dict[str, object]:
    root = fx["root"]
    assert isinstance(root, Path)
    plan = copy.deepcopy(fx["plan"])
    mutate(plan)
    updated = _finalize_plan(root, plan)
    fx["plan"] = updated
    return updated


def test_common_harness_threads_live_telemetry_sink_without_candidate_fields(
    tmp_path: Path,
) -> None:
    fx = _install_fixture(
        tmp_path,
        architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n",
    )
    transport = DictTransport(
        {"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]}
    )
    access = _ObjectByteAccess(transport)
    delivered: list[_AcceptedScalarEvent] = []

    candidate = _harness(
        fx,
        access,
        tmp_path,
        suffix="telemetry",
        telemetry_sink=delivered.append,
    )(_training_input(fx))

    assert candidate["status"] == "completed"
    assert delivered == [
        _AcceptedScalarEvent(group="fixture", series="signal", value=1.0, step=0)
    ]
    assert "telemetry" not in candidate
    assert "telemetry" not in candidate["result"]


@pytest.mark.parametrize(
    "architecture_source",
    [
        "import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n",
        (
            "import torch.nn as nn\n"
            "def build():\n"
            "    return nn.Sequential(nn.Conv2d(1, 2, 1), nn.ReLU(), nn.Conv2d(2, 4, 1))\n"
        ),
    ],
)
def test_common_harness_full_training_evaluation_paths_are_generic(
    tmp_path: Path, architecture_source: str
) -> None:
    fx = _install_fixture(tmp_path, architecture_source=architecture_source)
    transport = DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]})
    access = _ObjectByteAccess(transport)
    training_input = _training_input(fx)
    original = copy.deepcopy(training_input)
    training = _harness(
        fx,
        access,
        tmp_path,
        artifact_uris={"ignored": "not-used-by-training"},
        suffix="train",
    )(training_input)
    assert training_input == original
    assert training["status"] == "completed"
    assert "telemetry" not in training
    assert "telemetry" not in training["result"]
    assert training["stage_key"] == {
        "study_result": training_input["study_result"],
        "plan": training_input["plan"],
        "trial": "trial-0001",
        "kind": "training",
        "coordinate": None,
        "source_commit": training_input["source_commit"],
    }
    assert len(training["attempts"]) == 1
    attempt = training["attempts"][0]
    assert attempt["backend"] == "fixture-backend"
    assert attempt["execution_id"] == "exec-train"
    assert attempt["status"] == "completed" and attempt["diagnostic"] is None
    assert attempt["ended_at"].endswith("Z")
    root = fx["root"]
    assert isinstance(root, Path)
    assert not (root / "demo" / "models").exists()
    assert not (root / "demo" / "training_results").exists()

    runtime_model = _install_runtime_lineage(fx, training)
    evaluation_input = _evaluation_input(fx, runtime_model)
    evaluation = _harness(
        fx,
        access,
        tmp_path,
        weights_uri="this-is-ignored-by-evaluation",
        artifact_uris={"opaque_bundle": "s3://bucket/candidates/opaque.bin"},
        suffix="eval",
    )(evaluation_input)
    assert evaluation["status"] == "completed"
    assert "telemetry" not in evaluation
    assert "telemetry" not in evaluation["result"]
    assert evaluation["stage_key"]["kind"] == "evaluation"
    assert evaluation["stage_key"]["coordinate"] == "eval-0001"
    assert evaluation["result"]["metrics"] == {"strange_scalar": 7.25}
    artifact = evaluation["result"]["artifacts"]["opaque_bundle"]
    assert artifact["format"] == "opaque-bin"
    assert artifact["schema"] == "demo/opaque-output/v7"
    assert transport.objects[artifact["uri"]] == b"opaque-output"
    assert not (root / "demo" / "evaluation_results").exists()


def _assert_failed(candidate: dict[str, object], *, kind: str) -> None:
    assert candidate["state"] == "terminal"
    assert candidate["status"] == "failed"
    assert candidate["result"] is None
    assert candidate["diagnostic"] == {
        "code": "stage_execution_failed",
        "message": "Stage execution failed.",
    }
    assert candidate["stage_key"]["kind"] == kind
    attempts = candidate["attempts"]
    assert isinstance(attempts, list) and len(attempts) == 1
    assert attempts[0]["status"] == "failed"
    assert attempts[0]["diagnostic"] == candidate["diagnostic"]


def test_execution_harness_call_shape_and_operational_identity_are_separate(tmp_path: Path) -> None:
    protocol_signature = inspect.signature(ExecutionHarness.__call__)
    concrete_signature = inspect.signature(CommonExecutionHarness.__call__)
    assert list(protocol_signature.parameters) == ["self", "stage_input"]
    assert list(concrete_signature.parameters) == ["self", "stage_input"]
    fx = _install_fixture(tmp_path, architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n")
    stage_input = _training_input(fx)
    assert "backend" not in stage_input and "execution_id" not in stage_input
    assert "backend" not in stage_input["stage"] and "execution_id" not in stage_input["stage"]


def test_malformed_input_cannot_fabricate_stage_key(tmp_path: Path) -> None:
    fx = _install_fixture(tmp_path, architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n")
    access = _ObjectByteAccess(DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]}))
    stage_input = _training_input(fx)
    stage_input.pop("trial")
    with pytest.raises(ValueError, match="StageInput fields"):
        _harness(fx, access, tmp_path)(stage_input)  # type: ignore[arg-type]


@pytest.mark.parametrize("case", ["plan_digest", "source_commit", "pins", "trial", "training_stage"])
def test_common_plan_envelope_mismatch_becomes_failed_candidate(tmp_path: Path, case: str) -> None:
    fx = _install_fixture(tmp_path, architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n")
    access = _ObjectByteAccess(DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]}))
    stage_input = _training_input(fx)
    if case == "plan_digest":
        stage_input["plan_sha256"] = "0" * 64
    elif case == "source_commit":
        stage_input["source_commit"] = "0" * 40
    elif case == "pins":
        stage_input["pins"] = copy.deepcopy(stage_input["pins"])
        stage_input["pins"][0]["yaml_sha256"] = "0" * 64
    elif case == "trial":
        stage_input["trial"] = "trial-0002"
    else:
        stage_input["stage"] = dict(stage_input["stage"])
        stage_input["stage"]["seed"] = 43
    candidate = _harness(fx, access, tmp_path, suffix=case)(stage_input)
    _assert_failed(candidate, kind="training")


def _mutate_pin(plan: dict[str, object], *, kind: str, field: str, value: object) -> None:
    pins = plan["pins"]
    assert isinstance(pins, list)
    pin = next(item for item in pins if item["kind"] == kind)
    pin[field] = value


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [
        ("task", "yaml_sha256", "0" * 64),
        ("architecture", "companion_sha256", "0" * 64),
        ("architecture", "sources", [{"path": "ml_impl/shared.py", "sha256": "0" * 64}]),
        ("corpus", "manifest_sha256", "0" * 64),
        ("corpus", "manifest_entries", 2),
    ],
)
def test_plan_pin_integrity_mismatch_fails_before_domain_execution(
    tmp_path: Path, kind: str, field: str, value: object
) -> None:
    fx = _install_fixture(tmp_path, architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n")
    _rewrite_plan(fx, lambda plan: _mutate_pin(plan, kind=kind, field=field, value=value))
    access = _ObjectByteAccess(DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]}))
    candidate = _harness(fx, access, tmp_path, suffix=f"{kind}-{field}")(_training_input(fx))
    _assert_failed(candidate, kind="training")


@pytest.mark.parametrize("relative", ["mldb_v2/src/marker.py", "ml_impl/shared.py"])
def test_dirty_required_snapshot_file_fails_before_domain_execution(tmp_path: Path, relative: str) -> None:
    fx = _install_fixture(tmp_path, architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n")
    repo = fx["repo"]
    assert isinstance(repo, Path)
    (repo / relative).write_text("DIRTY = True\n", encoding="utf-8", newline="\n")
    access = _ObjectByteAccess(DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]}))
    candidate = _harness(fx, access, tmp_path, suffix="dirty")(_training_input(fx))
    _assert_failed(candidate, kind="training")


def test_silent_training_protocol_becomes_failed_candidate_without_weights_publication(
    tmp_path: Path,
) -> None:
    fx = _install_fixture(
        tmp_path,
        architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n",
        train_source=b"def train(context):\n    return context.model\n",
    )
    transport = DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]})
    access = _ObjectByteAccess(transport)
    candidate = _harness(fx, access, tmp_path, suffix="silent-train")(_training_input(fx))
    _assert_failed(candidate, kind="training")
    assert "s3://bucket/candidates/weights.pt" not in transport.objects


def test_malformed_training_telemetry_becomes_bounded_failed_candidate(
    tmp_path: Path,
) -> None:
    fx = _install_fixture(
        tmp_path,
        architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n",
        train_source=(
            b"def train(context):\n"
            b"    context.telemetry.report_scalar(group='', series='signal', value=1, step=0)\n"
            b"    return context.model\n"
        ),
    )
    transport = DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]})
    access = _ObjectByteAccess(transport)
    candidate = _harness(fx, access, tmp_path, suffix="bad-telemetry")(_training_input(fx))
    _assert_failed(candidate, kind="training")
    encoded = json.dumps(candidate)
    assert "telemetry group must be" not in encoded
    assert "non-empty exact string" not in encoded
    assert "s3://bucket/candidates/weights.pt" not in transport.objects


def test_training_runtime_exception_is_bounded_failed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = _install_fixture(tmp_path, architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n")
    access = _ObjectByteAccess(DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]}))
    monkeypatch.setattr(
        harness_module.training_runtime,
        "_execute_training_stage",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("credential=SECRET arbitrary traceback detail")),
    )
    candidate = _harness(fx, access, tmp_path, suffix="train-fail")(_training_input(fx))
    _assert_failed(candidate, kind="training")
    assert "SECRET" not in json.dumps(candidate)


def _dummy_runtime_model() -> dict[str, object]:
    return {
        "model": "demo/model-v1",
        "training_result": "demo/training-result-v1",
        "task": "demo/task-v1",
        "architecture": "demo/arch-v1",
        "weights": {
            "uri": "s3://bucket/models/weights.pt",
            "bytes": 1,
            "sha256": "0" * 64,
            "format": "pytorch-state-dict/v1",
        },
    }


def test_evaluation_runtime_exception_is_bounded_failed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = _install_fixture(tmp_path, architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n")
    access = _ObjectByteAccess(DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]}))
    monkeypatch.setattr(
        harness_module.evaluation_runtime,
        "_execute_evaluation_stage",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("private backend detail")),
    )
    candidate = _harness(fx, access, tmp_path, suffix="eval-fail")(
        _evaluation_input(fx, _dummy_runtime_model())
    )
    _assert_failed(candidate, kind="evaluation")
    assert "private backend detail" not in json.dumps(candidate)


@pytest.mark.parametrize("case", ["coordinate", "stage_name"])
def test_evaluation_plan_coordinate_or_stage_mismatch_fails(tmp_path: Path, case: str) -> None:
    fx = _install_fixture(tmp_path, architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n")
    access = _ObjectByteAccess(DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]}))
    stage_input = _evaluation_input(fx, _dummy_runtime_model())
    if case == "coordinate":
        stage_input["coordinate"] = "eval-0002"
    else:
        stage_input["stage"] = dict(stage_input["stage"])
        stage_input["stage"]["name"] = "wrong-stage"
    candidate = _harness(fx, access, tmp_path, suffix=f"eval-{case}")(stage_input)
    _assert_failed(candidate, kind="evaluation")


def test_canonical_plan_id_mismatch_fails_after_stage_key_established(tmp_path: Path) -> None:
    fx = _install_fixture(tmp_path, architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n")
    plan = fx["plan"]
    root = fx["root"]
    assert type(plan) is dict and isinstance(root, Path)
    _, local_id = str(plan["id"]).split("/", 1)
    plan_path = root / "demo" / "study_plans" / f"{local_id}.yaml"
    malformed = copy.deepcopy(plan)
    malformed["id"] = "demo/other-plan-v1"
    _write_json(plan_path, malformed)
    access = _ObjectByteAccess(DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]}))
    candidate = _harness(fx, access, tmp_path, suffix="plan-id")(_training_input(fx))
    _assert_failed(candidate, kind="training")


@pytest.mark.parametrize(("backend", "execution_id"), [("", "exec-1"), ("fixture", "")])
def test_constructor_rejects_empty_attempt_identity(
    tmp_path: Path, backend: str, execution_id: str
) -> None:
    fx = _install_fixture(tmp_path, architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n")
    access = _ObjectByteAccess(DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]}))
    with pytest.raises(ValueError):
        CommonExecutionHarness(
            repository_root=fx["repo"],
            pinned_mldb_data_root=fx["root"],
            runtime_mldb_data_root=fx["root"],
            object_bytes=access,
            corpus_destination_root=tmp_path / "corpus",
            work_dir=tmp_path / "work",
            training_weights_uri="s3://bucket/candidates/weights.pt",
            backend=backend,
            execution_id=execution_id,
        )


def test_missing_training_weights_uri_is_failed_attempt_not_domain_success(tmp_path: Path) -> None:
    fx = _install_fixture(tmp_path, architecture_source="import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n")
    access = _ObjectByteAccess(DictTransport({"s3://bucket/corpus-v1/data.bin": fx["corpus_payload"]}))
    candidate = _harness(fx, access, tmp_path, weights_uri=None, suffix="no-weights")(
        _training_input(fx)
    )
    _assert_failed(candidate, kind="training")
