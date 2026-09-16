from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from mldb_v2.src.common.ids import _canonical_json_bytes
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.training.canonical_weights import _serialize_canonical_state_dict
from mldb_v2.src.verification._training_result_acceptance import (
    _accept_training_candidate,
)


COMMIT = "a" * 40
EXECUTION_KEY = "123e4567e89b42d3a456426614174000"
STUDY_RESULT_ID = f"demo/run-{EXECUTION_KEY}"


class DictTransport:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.reads: list[str] = []
        self.publications: list[tuple[str, bytes]] = []

    def read_bytes(self, uri: str) -> bytes:
        self.reads.append(uri)
        return self.objects[uri]

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        self.publications.append((uri, data))
        raise AssertionError("acceptance must not publish bytes")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _install_architecture(tmp_path: Path) -> Path:
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
    source = "import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n"
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    architecture_dir = root / "demo" / "architectures"
    _write_json(
        architecture_dir / "arch-v1.yaml",
        {
            "schema": "mjtensu.mldb-v2/architecture/v1",
            "id": "demo/arch-v1",
            "status": "sealed",
            "task": "demo/task-v1",
            "name": "Linear",
            "family": "toy",
            "description": "",
            "implementation": {
                "framework": "pytorch",
                "entrypoint": "build",
                "sha256": source_sha,
            },
            "interface": {"input": {"kind": "tensor"}, "output": {"kind": "tensor"}},
            "structure": {"summary": "linear"},
        },
    )
    architecture_dir.mkdir(parents=True, exist_ok=True)
    (architecture_dir / "arch-v1.py").write_text(source, encoding="utf-8")
    return root


def _pin(kind: str, entity_id: str, *, executable: bool = False, corpus: bool = False):
    return {
        "kind": kind,
        "id": entity_id,
        "yaml_sha256": "1" * 64,
        "companion_sha256": "2" * 64 if executable else None,
        "sources": [],
        "manifest_sha256": "3" * 64 if corpus else None,
        "manifest_entries": 1 if corpus else None,
    }


def _plan() -> dict[str, object]:
    plan: dict[str, object] = {
        "schema": "mjtensu.mldb-v2/study-plan/v1",
        "id": "demo/placeholder",
        "content_sha256": "0" * 64,
        "study": "demo/study-v1",
        "source_commit": COMMIT,
        "pins": [
            _pin("namespace", "demo"),
            _pin("task", "demo/task-v1"),
            _pin("corpus", "demo/corpus-v1", corpus=True),
            _pin("architecture", "demo/arch-v1", executable=True),
            _pin("train_protocol", "demo/train-v1", executable=True),
            _pin("evaluation_protocol", "demo/eval-v1", executable=True),
            _pin("study", "demo/study-v1"),
        ],
        "trials": [
            {
                "trial": "trial-0001",
                "source": {
                    "kind": "training",
                    "task": "demo/task-v1",
                    "corpus": "demo/corpus-v1",
                    "architecture": "demo/arch-v1",
                    "train_protocol": "demo/train-v1",
                    "parameters": {"epochs": 2, "learning_rate": 0.25},
                    "seed": 42,
                },
                "evaluations": [
                    {
                        "coordinate": "eval-0001",
                        "stage": "holdout",
                        "task": "demo/task-v1",
                        "corpus": "demo/corpus-v1",
                        "evaluation_protocol": "demo/eval-v1",
                        "parameters": {"threshold": 0.5},
                    }
                ],
            }
        ],
    }
    payload = {
        key: plan[key]
        for key in ("schema", "study", "source_commit", "pins", "trials")
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    plan["content_sha256"] = digest
    plan["id"] = f"demo/study-v1-plan-{digest[:16]}"
    return plan


def _study_result(plan: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/study-result/v1",
        "id": STUDY_RESULT_ID,
        "execution_key": EXECUTION_KEY,
        "plan": plan["id"],
        "study": plan["study"],
        "source_commit": COMMIT,
        "backend": "fake",
        "created_at": "2026-09-13T00:00:00Z",
        "status": "submitted",
        "diagnostic": None,
        "trials": [
            {
                "trial": "trial-0001",
                "training": {"disposition": "pending", "result": None, "reason": None},
                "evaluations": [
                    {
                        "coordinate": "eval-0001",
                        "stage": "holdout",
                        "disposition": "pending",
                        "result": None,
                        "reason": None,
                    }
                ],
            }
        ],
    }


def _stage_input(plan: dict[str, object]) -> dict[str, object]:
    source = copy.deepcopy(plan["trials"][0]["source"])
    source.pop("kind")
    return {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": STUDY_RESULT_ID,
        "plan": plan["id"],
        "plan_sha256": plan["content_sha256"],
        "trial": "trial-0001",
        "kind": "training",
        "coordinate": None,
        "source_commit": COMMIT,
        "pins": copy.deepcopy(plan["pins"]),
        "stage": source,
        "runtime_model": None,
    }


def _attempt(status: str, execution_id: str) -> dict[str, object]:
    diagnostic = None
    if status != "completed":
        diagnostic = {"code": "backend_failed", "message": f"{status} attempt"}
    return {
        "backend": "fake",
        "execution_id": execution_id,
        "status": status,
        "started_at": "2026-09-13T00:00:00Z",
        "ended_at": "2026-09-13T00:00:01Z",
        "diagnostic": diagnostic,
    }


def _weight_ref(data: bytes, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "uri": "s3://bucket/candidates/weights.pt",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "format": "pytorch-state-dict/v1",
    }
    value.update(overrides)
    return value


def _completed_candidate(stage_input: dict[str, object], weights: dict[str, object]):
    return {
        "state": "terminal",
        "stage_key": {
            "study_result": stage_input["study_result"],
            "plan": stage_input["plan"],
            "trial": stage_input["trial"],
            "kind": "training",
            "coordinate": None,
            "source_commit": stage_input["source_commit"],
        },
        "attempts": [
            _attempt("failed", "attempt-1"),
            _attempt("completed", "attempt-2"),
        ],
        "status": "completed",
        "diagnostic": None,
        "result": {"weights": weights},
    }


def _terminal_candidate(stage_input: dict[str, object], status: str):
    return {
        "state": "terminal",
        "stage_key": {
            "study_result": stage_input["study_result"],
            "plan": stage_input["plan"],
            "trial": stage_input["trial"],
            "kind": "training",
            "coordinate": None,
            "source_commit": stage_input["source_commit"],
        },
        "attempts": [_attempt(status, "attempt-1")],
        "status": status,
        "diagnostic": {"code": "backend_terminal", "message": status},
        "result": None,
    }


def _accept(tmp_path: Path, *, candidate=None, plan=None, study_result=None, stage_input=None, data=None):
    plan = plan or _plan()
    study_result = study_result or _study_result(plan)
    stage_input = stage_input or _stage_input(plan)
    root = _install_architecture(tmp_path)
    data = data or _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    candidate = candidate or _completed_candidate(stage_input, _weight_ref(data))
    transport = DictTransport({candidate.get("result", {}).get("weights", {}).get("uri", "s3://unused/object"): data})
    result = _accept_training_candidate(
        study_result=study_result,
        plan=plan,
        stage_input=stage_input,
        candidate=candidate,
        mldb_data_root=root,
        object_bytes=_ObjectByteAccess(transport),
    )
    return result, transport, root


def test_valid_completed_candidate_builds_exact_training_result_and_model(tmp_path: Path) -> None:
    plan = _plan()
    stage_input = _stage_input(plan)
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    candidate = _completed_candidate(stage_input, _weight_ref(data))
    result, transport, _root = _accept(
        tmp_path,
        plan=plan,
        stage_input=stage_input,
        data=data,
        candidate=candidate,
    )
    training = result["training_result"]
    expected_training_id = f"{STUDY_RESULT_ID}-trial-0001-train"
    expected_model_id = f"{STUDY_RESULT_ID}-trial-0001-model"

    assert training["id"] == expected_training_id
    assert training["study_result"] == STUDY_RESULT_ID
    assert training["plan"] == plan["id"]
    assert training["trial"] == "trial-0001"
    assert training["task"] == "demo/task-v1"
    assert training["corpus"] == "demo/corpus-v1"
    assert training["architecture"] == "demo/arch-v1"
    assert training["train_protocol"] == "demo/train-v1"
    assert training["parameters"] == {"epochs": 2, "learning_rate": 0.25}
    assert training["seed"] == 42
    assert training["source_commit"] == COMMIT
    assert training["status"] == "completed"
    assert training["diagnostic"] is None
    assert training["result"]["weights"] == candidate["result"]["weights"]
    assert training["result"]["model"] == expected_model_id
    assert result["model"] == {
        "schema": "mjtensu.mldb-v2/model/v1",
        "id": expected_model_id,
        "training_result": expected_training_id,
    }
    assert training["attempts"] == candidate["attempts"]
    assert training["attempts"] is not candidate["attempts"]
    assert transport.reads == [candidate["result"]["weights"]["uri"]]
    assert transport.publications == []


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_backend_failed_or_cancelled_maps_directly_without_model(tmp_path: Path, status: str) -> None:
    plan = _plan()
    stage_input = _stage_input(plan)
    candidate = _terminal_candidate(stage_input, status)
    root = _install_architecture(tmp_path)
    transport = DictTransport({})

    result = _accept_training_candidate(
        study_result=_study_result(plan),
        plan=plan,
        stage_input=stage_input,
        candidate=candidate,
        mldb_data_root=root,
        object_bytes=_ObjectByteAccess(transport),
    )

    assert result["model"] is None
    assert result["training_result"]["status"] == status
    assert result["training_result"]["result"] is None
    assert result["training_result"]["diagnostic"] == candidate["diagnostic"]
    assert result["training_result"]["attempts"] == candidate["attempts"]
    assert transport.reads == []
    assert transport.publications == []


def _assert_acceptance_failed(result: dict[str, object]) -> None:
    training = result["training_result"]
    assert training["status"] == "failed"
    assert training["result"] is None
    assert training["diagnostic"] == {
        "code": "training_acceptance_failed",
        "message": "Training candidate failed canonical acceptance.",
    }
    assert result["model"] is None


def test_completed_stage_key_mismatch_becomes_formal_failed(tmp_path: Path) -> None:
    plan = _plan()
    stage_input = _stage_input(plan)
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    candidate = _completed_candidate(stage_input, _weight_ref(data))
    candidate["stage_key"]["trial"] = "trial-0002"

    result, _transport, _root = _accept(
        tmp_path,
        plan=plan,
        stage_input=stage_input,
        data=data,
        candidate=candidate,
    )
    _assert_acceptance_failed(result)
    assert result["training_result"]["attempts"] == candidate["attempts"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "demo/other-task-v1"),
        ("corpus", "demo/other-corpus-v1"),
        ("architecture", "demo/other-arch-v1"),
        ("train_protocol", "demo/other-train-v1"),
        ("parameters", {"epochs": 99}),
        ("seed", 7),
    ],
)
def test_stage_lineage_mismatch_is_rejected_before_candidate_acceptance(
    tmp_path: Path, field: str, value: object
) -> None:
    plan = _plan()
    stage_input = _stage_input(plan)
    stage_input["stage"][field] = value
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    candidate = _completed_candidate(stage_input, _weight_ref(data))

    with pytest.raises(ValueError, match="TrainingStage"):
        _accept_training_candidate(
            study_result=_study_result(plan),
            plan=plan,
            stage_input=stage_input,
            candidate=candidate,
            mldb_data_root=_install_architecture(tmp_path),
            object_bytes=_ObjectByteAccess(DictTransport({candidate["result"]["weights"]["uri"]: data})),
        )


@pytest.mark.parametrize(
    "mutation",
    ["study_result", "plan", "trial", "source_commit", "pins"],
)
def test_stage_input_request_lineage_mismatch_is_rejected(tmp_path: Path, mutation: str) -> None:
    plan = _plan()
    stage_input = _stage_input(plan)
    if mutation == "study_result":
        stage_input["study_result"] = "demo/run-ffffffffffffffffffffffffffffffff"
    elif mutation == "plan":
        stage_input["plan"] = "demo/other-plan-v1"
    elif mutation == "trial":
        stage_input["trial"] = "trial-0002"
    elif mutation == "source_commit":
        stage_input["source_commit"] = "b" * 40
    else:
        stage_input["pins"] = copy.deepcopy(stage_input["pins"][:-1])
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    candidate = _completed_candidate(stage_input, _weight_ref(data))
    transport = DictTransport({candidate["result"]["weights"]["uri"]: data})

    with pytest.raises(ValueError):
        _accept_training_candidate(
            study_result=_study_result(plan),
            plan=plan,
            stage_input=stage_input,
            candidate=candidate,
            mldb_data_root=_install_architecture(tmp_path),
            object_bytes=_ObjectByteAccess(transport),
        )
    assert transport.reads == []
    assert transport.publications == []


def test_study_result_plan_and_source_lineage_mismatch_are_rejected(tmp_path: Path) -> None:
    plan = _plan()
    stage_input = _stage_input(plan)
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    candidate = _completed_candidate(stage_input, _weight_ref(data))
    root = _install_architecture(tmp_path)

    for field, value in (("plan", "demo/other-plan-v1"), ("source_commit", "b" * 40)):
        study_result = _study_result(plan)
        study_result[field] = value
        with pytest.raises(ValueError):
            _accept_training_candidate(
                study_result=study_result,
                plan=plan,
                stage_input=stage_input,
                candidate=candidate,
                mldb_data_root=root,
                object_bytes=_ObjectByteAccess(DictTransport({candidate["result"]["weights"]["uri"]: data})),
            )


def _accept_completed_payload(
    tmp_path: Path,
    *,
    artifact_data: bytes,
    supplied_data: bytes | None = None,
    ref_overrides: dict[str, object] | None = None,
):
    plan = _plan()
    stage_input = _stage_input(plan)
    ref = _weight_ref(artifact_data)
    if ref_overrides:
        ref.update(ref_overrides)
    candidate = _completed_candidate(stage_input, ref)
    transport = DictTransport({ref["uri"]: supplied_data if supplied_data is not None else artifact_data})
    result = _accept_training_candidate(
        study_result=_study_result(plan),
        plan=plan,
        stage_input=stage_input,
        candidate=candidate,
        mldb_data_root=_install_architecture(tmp_path),
        object_bytes=_ObjectByteAccess(transport),
    )
    return result, candidate, transport


def test_invalid_artifact_ref_and_format_become_formal_failed(tmp_path: Path) -> None:
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    result, _candidate, transport = _accept_completed_payload(
        tmp_path / "invalid-uri",
        artifact_data=data,
        ref_overrides={"uri": "file:///tmp/weights.pt"},
    )
    _assert_acceptance_failed(result)
    assert transport.reads == []

    result, _candidate, transport = _accept_completed_payload(
        tmp_path / "invalid-format",
        artifact_data=data,
        ref_overrides={"format": "checkpoint/v1"},
    )
    _assert_acceptance_failed(result)
    assert transport.reads == []


def test_byte_and_sha_mismatch_become_formal_failed(tmp_path: Path) -> None:
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())

    result, _candidate, _transport = _accept_completed_payload(
        tmp_path / "bytes",
        artifact_data=data,
        ref_overrides={"bytes": len(data) + 1},
    )
    _assert_acceptance_failed(result)

    result, _candidate, _transport = _accept_completed_payload(
        tmp_path / "sha",
        artifact_data=data,
        ref_overrides={"sha256": "0" * 64},
    )
    _assert_acceptance_failed(result)


def test_malformed_canonical_bytes_and_incompatible_architecture_become_failed(tmp_path: Path) -> None:
    malformed = b"not a pytorch state dict"
    result, _candidate, _transport = _accept_completed_payload(
        tmp_path / "malformed",
        artifact_data=malformed,
    )
    _assert_acceptance_failed(result)

    incompatible = _serialize_canonical_state_dict(torch.nn.Linear(4, 2).state_dict())
    result, _candidate, _transport = _accept_completed_payload(
        tmp_path / "incompatible",
        artifact_data=incompatible,
    )
    _assert_acceptance_failed(result)


def test_supplied_bytes_must_match_artifact_ref(tmp_path: Path) -> None:
    expected = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    different = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    result, _candidate, _transport = _accept_completed_payload(
        tmp_path,
        artifact_data=expected,
        supplied_data=different,
    )
    _assert_acceptance_failed(result)


def test_acceptance_does_not_write_repository_or_publish_artifacts(tmp_path: Path) -> None:
    plan = _plan()
    stage_input = _stage_input(plan)
    root = _install_architecture(tmp_path)
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    candidate = _completed_candidate(stage_input, _weight_ref(data))
    transport = DictTransport({candidate["result"]["weights"]["uri"]: data})
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    _accept_training_candidate(
        study_result=_study_result(plan),
        plan=plan,
        stage_input=stage_input,
        candidate=candidate,
        mldb_data_root=root,
        object_bytes=_ObjectByteAccess(transport),
    )
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert transport.publications == []


def test_attempt_backend_and_terminal_status_must_match_candidate(tmp_path: Path) -> None:
    plan = _plan()
    stage_input = _stage_input(plan)
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    candidate = _completed_candidate(stage_input, _weight_ref(data))
    candidate["attempts"][-1]["backend"] = "other"

    with pytest.raises(ValueError, match="backend"):
        _accept_training_candidate(
            study_result=_study_result(plan),
            plan=plan,
            stage_input=stage_input,
            candidate=candidate,
            mldb_data_root=_install_architecture(tmp_path / "backend"),
            object_bytes=_ObjectByteAccess(DictTransport({candidate["result"]["weights"]["uri"]: data})),
        )

    candidate = _completed_candidate(stage_input, _weight_ref(data))
    candidate["attempts"][-1] = _attempt("failed", "attempt-2")
    with pytest.raises(ValueError, match="last training attempt status"):
        _accept_training_candidate(
            study_result=_study_result(plan),
            plan=plan,
            stage_input=stage_input,
            candidate=candidate,
            mldb_data_root=_install_architecture(tmp_path / "status"),
            object_bytes=_ObjectByteAccess(DictTransport({candidate["result"]["weights"]["uri"]: data})),
        )


def test_acceptance_source_has_no_skeleton_clearml_or_persistence_dependency() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "verification"
        / "_training_result_acceptance.py"
    ).read_text(encoding="utf-8")
    assert "mldb_v2.skeleton" not in source
    assert "clearml" not in source.casefold()
    assert "CanonicalRepositoryWriter" not in source
    assert "create_immutable" not in source
    assert "replace_nonterminal_study_result" not in source
    assert "publish_bytes" not in source


def test_stage_parameters_are_type_sensitive_exact_plan_values(tmp_path: Path) -> None:
    plan = _plan()
    plan["trials"][0]["source"]["parameters"] = {"epochs": 1}
    _resign(plan)
    stage_input = _stage_input(plan)
    stage_input["stage"]["parameters"] = {"epochs": 1.0}
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    candidate = _completed_candidate(stage_input, _weight_ref(data))

    with pytest.raises(ValueError, match="does not exactly match Plan trial source"):
        _accept_training_candidate(
            study_result=_study_result(plan),
            plan=plan,
            stage_input=stage_input,
            candidate=candidate,
            mldb_data_root=_install_architecture(tmp_path),
            object_bytes=_ObjectByteAccess(
                DictTransport({candidate["result"]["weights"]["uri"]: data})
            ),
        )


def _resign(plan: dict[str, object]) -> None:
    payload = {
        key: plan[key]
        for key in ("schema", "study", "source_commit", "pins", "trials")
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    plan["content_sha256"] = digest
    plan["id"] = f"demo/study-v1-plan-{digest[:16]}"
