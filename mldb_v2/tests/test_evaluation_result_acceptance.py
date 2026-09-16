from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from mldb_v2.src.common.ids import _canonical_json_bytes
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study._plan_build import _plan_id, _validate_study_plan
from mldb_v2.src.verification._evaluation_result_acceptance import _accept_evaluation_result


EXECUTION_KEY = "1234567812344abc8def1234567890ab"
STUDY_RESULT_ID = f"demo/run-{EXECUTION_KEY}"
TRAINING_RESULT_ID = f"{STUDY_RESULT_ID}-trial-0001-train"
MODEL_ID = f"{STUDY_RESULT_ID}-trial-0001-model"
SOURCE_COMMIT = "a" * 40


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeTransport:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.publish_calls = 0

    def read_bytes(self, uri: str) -> bytes:
        return self.objects[uri]

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        self.publish_calls += 1
        raise AssertionError("acceptance must not publish bytes")


def _artifact_ref(uri: str, data: bytes, *, format: str, schema: str) -> dict[str, object]:
    return {
        "uri": uri,
        "bytes": len(data),
        "sha256": _sha(data),
        "format": format,
        "schema": schema,
    }


def _pin(kind: str, entity_id: str, *, companion: str | None = None,
         manifest: str | None = None, entries: int | None = None) -> dict[str, object]:
    return {
        "kind": kind,
        "id": entity_id,
        "yaml_sha256": "1" * 64,
        "companion_sha256": companion,
        "sources": [],
        "manifest_sha256": manifest,
        "manifest_entries": entries,
    }


def _install_catalog(root: Path) -> tuple[str, str]:
    _write_json(root / "demo" / "namespace.yaml", {
        "schema": "mjtensu.mldb-v2/namespace/v1", "id": "demo",
        "name": "Demo", "description": "",
    })
    _write_json(root / "demo" / "tasks" / "task-v1.yaml", {
        "schema": "mjtensu.mldb-v2/task/v1", "id": "demo/task-v1", "status": "draft",
        "name": "Task", "problem_type": "regression", "description": "",
        "input": {}, "target": {"type": "continuous"}, "semantics": {}, "scope": {},
    })
    _write_json(root / "demo" / "architectures" / "linear-v1.yaml", {
        "schema": "mjtensu.mldb-v2/architecture/v1", "id": "demo/linear-v1",
        "status": "draft", "task": "demo/task-v1", "name": "Linear", "family": "toy",
        "description": "", "implementation": {"framework": "pytorch", "entrypoint": "build"},
        "interface": {"input": {"kind": "tensor"}, "output": {"kind": "tensor"}},
        "structure": {"summary": "linear"},
    })

    companion = b"def evaluate(context):\n    return {}\n"
    protocol_dir = root / "demo" / "evaluation_protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    (protocol_dir / "eval-v1.py").write_bytes(companion)
    companion_sha = _sha(companion)
    protocol = {
        "schema": "mjtensu.mldb-v2/evaluation-protocol/v1",
        "id": "demo/eval-v1",
        "status": "sealed",
        "task": "demo/task-v1",
        "name": "Eval",
        "description": "",
        "implementation": {"entrypoint": "evaluate", "sha256": companion_sha},
        "parameters": {"threshold": {"type": "number", "default": 0.5}},
        "metrics": {
            "count": {"type": "integer", "required": True},
            "score": {"type": "number", "required": True},
            "aux": {"type": "number", "required": False},
        },
        "artifacts": {
            "predictions": {
                "format": "jsonl", "schema": "demo/predictions/v1", "required": True,
            },
            "report": {
                "format": "text", "schema": "demo/report/v1", "required": False,
            },
        },
    }
    protocol_path = protocol_dir / "eval-v1.yaml"
    _write_json(protocol_path, protocol)
    return _sha(protocol_path.read_bytes()), companion_sha


def _build_plan(protocol_yaml_sha: str, protocol_companion_sha: str) -> dict[str, object]:
    pins = [
        _pin("namespace", "demo"),
        _pin("task", "demo/task-v1"),
        _pin("corpus", "demo/eval-corpus-v1", manifest="2" * 64, entries=1),
        _pin("corpus", "demo/train-corpus-v1", manifest="3" * 64, entries=1),
        _pin("architecture", "demo/linear-v1", companion="4" * 64),
        _pin("train_protocol", "demo/train-v1", companion="5" * 64),
        _pin("evaluation_protocol", "demo/eval-v1", companion=protocol_companion_sha),
        _pin("study", "demo/study-v1"),
    ]
    pins[6]["yaml_sha256"] = protocol_yaml_sha
    payload = {
        "schema": "mjtensu.mldb-v2/study-plan/v1",
        "study": "demo/study-v1",
        "source_commit": SOURCE_COMMIT,
        "pins": pins,
        "trials": [{
            "trial": "trial-0001",
            "source": {
                "kind": "training", "task": "demo/task-v1",
                "corpus": "demo/train-corpus-v1", "architecture": "demo/linear-v1",
                "train_protocol": "demo/train-v1", "parameters": {}, "seed": 42,
            },
            "evaluations": [{
                "coordinate": "eval-0001", "stage": "holdout", "task": "demo/task-v1",
                "corpus": "demo/eval-corpus-v1", "evaluation_protocol": "demo/eval-v1",
                "parameters": {"threshold": 0.5},
            }],
        }],
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    plan = {
        **payload,
        "id": _plan_id("demo/study-v1", digest),
        "content_sha256": digest,
    }
    return dict(_validate_study_plan(plan))


def _install_model_lineage(root: Path, plan: dict[str, object]) -> dict[str, object]:
    weight_data = b"canonical-weights"
    weight_ref = {
        "uri": "s3://bucket/model.pt",
        "bytes": len(weight_data),
        "sha256": _sha(weight_data),
        "format": "pytorch-state-dict/v1",
    }
    training_result = {
        "schema": "mjtensu.mldb-v2/training-result/v1",
        "id": TRAINING_RESULT_ID,
        "study_result": STUDY_RESULT_ID,
        "plan": plan["id"],
        "trial": "trial-0001",
        "task": "demo/task-v1",
        "architecture": "demo/linear-v1",
        "corpus": "demo/train-corpus-v1",
        "train_protocol": "demo/train-v1",
        "parameters": {},
        "seed": 42,
        "source_commit": SOURCE_COMMIT,
        "attempts": [{
            "backend": "fake", "execution_id": "train-1", "status": "completed",
            "started_at": "2026-09-13T00:00:00Z", "ended_at": "2026-09-13T00:00:01Z",
            "diagnostic": None,
        }],
        "status": "completed", "diagnostic": None,
        "result": {"weights": weight_ref, "model": MODEL_ID},
    }
    _write_json(
        root / "demo" / "training_results" / f"{TRAINING_RESULT_ID.split('/', 1)[1]}.yaml",
        training_result,
    )
    _write_json(
        root / "demo" / "models" / f"{MODEL_ID.split('/', 1)[1]}.yaml",
        {"schema": "mjtensu.mldb-v2/model/v1", "id": MODEL_ID, "training_result": TRAINING_RESULT_ID},
    )
    return weight_ref


def _study_result(plan: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/study-result/v1",
        "id": STUDY_RESULT_ID,
        "execution_key": EXECUTION_KEY,
        "plan": plan["id"],
        "study": plan["study"],
        "source_commit": SOURCE_COMMIT,
        "backend": "fake",
        "created_at": "2026-09-13T00:00:00Z",
        "status": "submitted",
        "diagnostic": None,
        "trials": [{
            "trial": "trial-0001",
            "training": {"disposition": "completed", "result": TRAINING_RESULT_ID, "reason": None},
            "evaluations": [{
                "coordinate": "eval-0001", "stage": "holdout",
                "disposition": "pending", "result": None, "reason": None,
            }],
        }],
    }


def _stage_input(plan: dict[str, object], weights: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": STUDY_RESULT_ID,
        "plan": plan["id"],
        "plan_sha256": plan["content_sha256"],
        "trial": "trial-0001",
        "kind": "evaluation",
        "coordinate": "eval-0001",
        "source_commit": SOURCE_COMMIT,
        "pins": copy.deepcopy(plan["pins"]),
        "stage": {
            "name": "holdout", "task": "demo/task-v1", "corpus": "demo/eval-corpus-v1",
            "evaluation_protocol": "demo/eval-v1", "parameters": {"threshold": 0.5},
        },
        "runtime_model": {
            "model": MODEL_ID,
            "training_result": TRAINING_RESULT_ID,
            "task": "demo/task-v1",
            "architecture": "demo/linear-v1",
            "weights": copy.deepcopy(weights),
        },
    }


def _attempt(status: str = "completed") -> dict[str, object]:
    diagnostic = None if status == "completed" else {"code": "backend_failed", "message": "backend terminal"}
    return {
        "backend": "fake", "execution_id": "eval-1", "status": status,
        "started_at": "2026-09-13T00:01:00Z", "ended_at": "2026-09-13T00:01:01Z",
        "diagnostic": diagnostic,
    }


def _completed_candidate(predictions: dict[str, object]) -> dict[str, object]:
    return {
        "state": "terminal",
        "stage_key": {
            "study_result": STUDY_RESULT_ID,
            "plan": None,
            "trial": "trial-0001",
            "kind": "evaluation",
            "coordinate": "eval-0001",
            "source_commit": SOURCE_COMMIT,
        },
        "attempts": [_attempt()],
        "status": "completed",
        "diagnostic": None,
        "result": {
            "metrics": {"count": 3, "score": 0.5},
            "artifacts": {"predictions": predictions},
        },
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "mldb_data"
    protocol_yaml_sha, protocol_companion_sha = _install_catalog(root)
    plan = _build_plan(protocol_yaml_sha, protocol_companion_sha)
    weights = _install_model_lineage(root, plan)
    study_result = _study_result(plan)
    stage_input = _stage_input(plan, weights)
    predictions_data = b'{"prediction": 1}\n'
    predictions = _artifact_ref(
        "s3://bucket/predictions.jsonl", predictions_data,
        format="jsonl", schema="demo/predictions/v1",
    )
    candidate = _completed_candidate(predictions)
    candidate["stage_key"]["plan"] = plan["id"]
    transport = FakeTransport()
    transport.objects[predictions["uri"]] = predictions_data
    request = {
        "study_result": study_result,
        "plan": plan,
        "stage_input": stage_input,
        "candidate": candidate,
    }
    return {
        "root": root,
        "plan": plan,
        "study_result": study_result,
        "stage_input": stage_input,
        "candidate": candidate,
        "transport": transport,
        "object_bytes": _ObjectByteAccess(transport),
        "request": request,
    }


def _accept(fx: dict[str, object]) -> dict[str, object]:
    return _accept_evaluation_result(
        request=fx["request"],
        mldb_data_root=fx["root"],
        object_bytes=fx["object_bytes"],
    )["evaluation_result"]


def test_valid_completed_candidate_preserves_exact_lineage_coordinate_and_attempts(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    result = _accept(fx)
    assert result["status"] == "completed"
    assert result["id"] == f"{STUDY_RESULT_ID}-trial-0001-eval-0001"
    assert result["study_result"] == STUDY_RESULT_ID
    assert result["plan"] == fx["plan"]["id"]
    assert result["trial"] == "trial-0001" and result["coordinate"] == "eval-0001"
    assert result["stage"] == "holdout" and result["model"] == MODEL_ID
    assert result["attempts"] == fx["candidate"]["attempts"]
    assert result["result"]["metrics"] == {"count": 3, "score": 0.5}
    assert fx["transport"].publish_calls == 0


def test_optional_metric_and_artifact_may_be_present_or_absent(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    absent = _accept(fx)
    assert "aux" not in absent["result"]["metrics"]
    assert "report" not in absent["result"]["artifacts"]

    report_data = b"ok\n"
    report = _artifact_ref(
        "s3://bucket/report.txt", report_data, format="text", schema="demo/report/v1"
    )
    fx["candidate"]["result"]["metrics"]["aux"] = 1
    fx["candidate"]["result"]["artifacts"]["report"] = report
    fx["transport"].objects[report["uri"]] = report_data
    present = _accept(fx)
    assert present["result"]["metrics"]["aux"] == 1
    assert present["result"]["artifacts"]["report"] == report


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["result"]["metrics"].pop("count"),
        lambda c: c["result"]["metrics"].__setitem__("extra", 1),
        lambda c: c["result"]["metrics"].__setitem__("count", True),
        lambda c: c["result"]["metrics"].__setitem__("score", True),
        lambda c: c["result"]["metrics"].__setitem__("score", float("nan")),
        lambda c: c["result"]["metrics"].__setitem__("score", float("inf")),
        lambda c: c["result"]["metrics"].__setitem__("score", float("-inf")),
    ],
)
def test_metric_contract_failures_become_formal_failed_result(tmp_path: Path, mutate) -> None:
    fx = _fixture(tmp_path)
    mutate(fx["candidate"])
    result = _accept(fx)
    assert result["status"] == "failed"
    assert result["result"] is None
    assert result["diagnostic"]["code"] == "evaluation_acceptance_failed"


def _assert_acceptance_failed(result: dict[str, object]) -> None:
    assert result["status"] == "failed"
    assert result["result"] is None
    assert result["diagnostic"] == {
        "code": "evaluation_acceptance_failed",
        "message": "Evaluation candidate failed canonical acceptance.",
    }


def test_required_and_undeclared_artifact_failures_become_formal_failed(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    fx["candidate"]["result"]["artifacts"].pop("predictions")
    _assert_acceptance_failed(_accept(fx))

    fx = _fixture(tmp_path / "undeclared")
    fx["candidate"]["result"]["artifacts"]["extra"] = copy.deepcopy(
        fx["candidate"]["result"]["artifacts"]["predictions"]
    )
    _assert_acceptance_failed(_accept(fx))


def test_artifact_format_and_schema_mismatch_become_formal_failed(tmp_path: Path) -> None:
    for field, value in (("format", "csv"), ("schema", "demo/other/v1")):
        fx = _fixture(tmp_path / field)
        fx["candidate"]["result"]["artifacts"]["predictions"][field] = value
        _assert_acceptance_failed(_accept(fx))


def test_artifact_uri_bytes_and_sha_failures_become_formal_failed(tmp_path: Path) -> None:
    mutations = (
        ("uri", "file:///tmp/predictions.jsonl"),
        ("bytes", 999),
        ("sha256", "0" * 64),
    )
    for field, value in mutations:
        fx = _fixture(tmp_path / field)
        fx["candidate"]["result"]["artifacts"]["predictions"][field] = value
        _assert_acceptance_failed(_accept(fx))


def test_supplied_artifact_bytes_must_match_ref(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    uri = fx["candidate"]["result"]["artifacts"]["predictions"]["uri"]
    fx["transport"].objects[uri] = b"different-bytes"
    _assert_acceptance_failed(_accept(fx))


def test_completed_stage_key_mismatch_becomes_formal_failed(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    fx["candidate"]["stage_key"]["coordinate"] = "eval-0002"
    result = _accept(fx)
    _assert_acceptance_failed(result)
    assert result["attempts"] == fx["candidate"]["attempts"]


@pytest.mark.parametrize(
    "mutation",
    ["study_result", "plan", "trial", "coordinate", "source_commit"],
)
def test_stage_input_request_lineage_mismatch_is_rejected(tmp_path: Path, mutation: str) -> None:
    fx = _fixture(tmp_path)
    stage_input = fx["stage_input"]
    if mutation == "study_result":
        stage_input["study_result"] = "demo/run-ffffffffffffffffffffffffffffffff"
    elif mutation == "plan":
        stage_input["plan"] = "demo/other-plan-v1"
    elif mutation == "trial":
        stage_input["trial"] = "trial-0002"
    elif mutation == "coordinate":
        stage_input["coordinate"] = "eval-0002"
    else:
        stage_input["source_commit"] = "b" * 40
    with pytest.raises(ValueError):
        _accept(fx)
    assert fx["transport"].publish_calls == 0


def test_study_result_plan_and_source_lineage_mismatch_are_rejected(tmp_path: Path) -> None:
    for field, value in (("plan", "demo/other-plan-v1"), ("source_commit", "b" * 40)):
        fx = _fixture(tmp_path / field)
        fx["study_result"][field] = value
        with pytest.raises(ValueError):
            _accept(fx)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "other-stage"),
        ("task", "demo/other-task-v1"),
        ("corpus", "demo/other-corpus-v1"),
        ("evaluation_protocol", "demo/other-eval-v1"),
        ("parameters", {"threshold": 0.75}),
    ],
)
def test_evaluation_stage_lineage_mismatch_is_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    fx = _fixture(tmp_path)
    fx["stage_input"]["stage"][field] = value
    with pytest.raises(ValueError, match="EvaluationStage"):
        _accept(fx)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "demo/other-model"),
        ("training_result", "demo/other-training-result"),
        ("task", "demo/other-task-v1"),
        ("architecture", "demo/other-arch-v1"),
    ],
)
def test_runtime_model_lineage_mismatch_is_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    fx = _fixture(tmp_path)
    fx["stage_input"]["runtime_model"][field] = value
    with pytest.raises(ValueError, match="runtime Model"):
        _accept(fx)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("study_result", "demo/run-ffffffffffffffffffffffffffffffff"),
        ("plan", "demo/other-plan-v1"),
        ("trial", "trial-0002"),
        ("kind", "training"),
        ("coordinate", "eval-0002"),
        ("source_commit", "b" * 40),
    ],
)
def test_completed_candidate_stage_key_exact_mismatch_becomes_failed(
    tmp_path: Path, field: str, value: object
) -> None:
    fx = _fixture(tmp_path)
    fx["candidate"]["stage_key"][field] = value
    _assert_acceptance_failed(_accept(fx))


def _terminal_candidate(fx: dict[str, object], status: str) -> dict[str, object]:
    return {
        "state": "terminal",
        "stage_key": copy.deepcopy(fx["candidate"]["stage_key"]),
        "attempts": [_attempt(status)],
        "status": status,
        "diagnostic": {"code": "backend_terminal", "message": status},
        "result": None,
    }


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_backend_failed_or_cancelled_maps_directly(tmp_path: Path, status: str) -> None:
    fx = _fixture(tmp_path)
    candidate = _terminal_candidate(fx, status)
    fx["candidate"] = candidate
    fx["request"]["candidate"] = candidate
    result = _accept(fx)
    assert result["status"] == status
    assert result["result"] is None
    assert result["diagnostic"] == candidate["diagnostic"]
    assert result["attempts"] == candidate["attempts"]


def test_attempt_order_is_preserved_exactly(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    fx["candidate"]["attempts"] = [
        _attempt("failed"),
        {**_attempt("completed"), "execution_id": "eval-2"},
    ]
    result = _accept(fx)
    assert result["status"] == "completed"
    assert result["attempts"] == fx["candidate"]["attempts"]
    assert result["attempts"] is not fx["candidate"]["attempts"]


def test_acceptance_does_not_mutate_repository_or_publish(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    root = fx["root"]
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    _accept(fx)
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    assert after == before
    assert fx["transport"].publish_calls == 0


def test_acceptance_source_has_no_skeleton_clearml_or_persistence_dependency() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "verification" / "_evaluation_result_acceptance.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert all(not module.startswith("mldb_v2.skeleton") for module in imports)
    assert "clearml" not in source.casefold()
    assert "CanonicalRepositoryWriter" not in source
    assert "create_immutable" not in source
    assert "replace_nonterminal_study_result" not in source
    assert "publish_bytes" not in source
