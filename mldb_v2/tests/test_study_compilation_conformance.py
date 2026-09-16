from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from mldb_v2.src.common.ids import _canonical_json_bytes
from mldb_v2.src.study import _plan_build as plan_build_module
from mldb_v2.src.study._plan_build import (
    _StudyPlanError,
    _build_study_plan,
    _create_study_plan,
)
from mldb_v2.src.study._source_pinning import _SourcePinningError
from mldb_v2.tests.test_study_source_pinning import (
    _commit_all,
    _existing_repo,
    _planning,
    _training_repo,
    _write_json,
)


def _compile(repo: Path, study_id: str, commit: str):
    planning = _planning(repo, study_id)
    return _build_study_plan(
        repository_root=repo,
        mldb_data_root=repo / "mldb_data",
        planning=planning,
        selected_commit=commit,
    )


def test_classifier_like_compilation_is_deterministic_and_resolves_defaults(tmp_path: Path) -> None:
    repo, study_id, _commit, _ = _training_repo(tmp_path)
    root = repo / "mldb_data"

    train_path = root / "proto-ns/train_protocols/train-v1.yaml"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    train["parameters"] = {
        "batch_size": {"default": 32, "type": "integer"},
        "learning_rate": {"default": 0.001, "type": "number"},
    }
    _write_json(train_path, train)

    eval_path = root / "proto-ns/evaluation_protocols/eval-v1.yaml"
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    evaluation["parameters"] = {
        "limit": {"default": 10, "type": "integer"},
        "threshold": {"default": 0.5, "type": "number"},
    }
    _write_json(eval_path, evaluation)

    study_path = root / "study-ns/studies/study-v1.yaml"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["model"]["train"]["architectures"] = ["arch-ns/arch-a-v1"]
    study["model"]["train"]["parameters"] = {
        "learning_rate": {"values": [0.001, 0.01]}
    }
    study["model"]["train"]["seeds"] = [41, 43]
    study["evaluations"][0]["parameters"] = {
        "threshold": {"values": [0.4, 0.6]}
    }
    study["evaluations"][1]["parameters"] = {}
    _write_json(study_path, study)
    commit = _commit_all(repo, "classifier-like closure fixture")

    first = _compile(repo, study_id, commit)
    second = _compile(repo, study_id, commit)
    assert first == second
    assert len(first["trials"]) == 4
    assert [trial["source"]["seed"] for trial in first["trials"]] == [41, 43, 41, 43]
    assert [trial["source"]["parameters"]["learning_rate"] for trial in first["trials"]] == [
        0.001, 0.001, 0.01, 0.01
    ]
    assert all(trial["source"]["parameters"]["batch_size"] == 32 for trial in first["trials"])
    assert all(
        [coordinate["coordinate"] for coordinate in trial["evaluations"]]
        == ["eval-0001", "eval-0002", "eval-0003"]
        for trial in first["trials"]
    )
    assert first["trials"][0]["evaluations"][2]["parameters"] == {
        "limit": 10,
        "threshold": 0.5,
    }
    payload = {key: value for key, value in first.items() if key not in {"id", "content_sha256"}}
    expected_digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    assert first["content_sha256"] == expected_digest
    assert first["id"].endswith(f"-plan-{expected_digest[:16]}")


def test_rotated_detector_like_multi_architecture_multi_stage_is_generic(tmp_path: Path) -> None:
    repo, study_id, _commit, _ = _training_repo(tmp_path)
    root = repo / "mldb_data"

    train_path = root / "proto-ns/train_protocols/train-v1.yaml"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    train["parameters"] = {
        "batch_size": {"default": 8, "type": "integer"},
        "rotation_deg": {"default": 45, "type": "integer"},
    }
    _write_json(train_path, train)
    eval_path = root / "proto-ns/evaluation_protocols/eval-v1.yaml"
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    evaluation["parameters"] = {
        "angle": {"default": 0, "type": "integer"},
        "iou": {"default": 0.5, "type": "number"},
    }
    _write_json(eval_path, evaluation)

    study_path = root / "study-ns/studies/study-v1.yaml"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["model"]["train"]["parameters"] = {
        "rotation_deg": {"values": [15, 45]},
        "batch_size": {"values": [16, 8]},
    }
    study["model"]["train"]["seeds"] = [42]
    study["evaluations"][0]["parameters"] = {"iou": {"values": [0.5, 0.75]}}
    study["evaluations"][1]["parameters"] = {"angle": {"values": [0, 30]}}
    _write_json(study_path, study)
    commit = _commit_all(repo, "rotated-like closure fixture")

    plan = _compile(repo, study_id, commit)
    assert len(plan["trials"]) == 8
    assert [trial["source"]["architecture"] for trial in plan["trials"]] == [
        "arch-ns/arch-z-v1",
    ] * 4 + ["arch-ns/arch-a-v1"] * 4
    assert [
        (trial["source"]["parameters"]["batch_size"], trial["source"]["parameters"]["rotation_deg"])
        for trial in plan["trials"][:4]
    ] == [(16, 15), (16, 45), (8, 15), (8, 45)]
    for trial in plan["trials"]:
        assert [item["stage"] for item in trial["evaluations"]] == [
            "eval-a", "eval-a", "eval-b", "eval-b"
        ]
        assert [item["coordinate"] for item in trial["evaluations"]] == [
            "eval-0001", "eval-0002", "eval-0003", "eval-0004"
        ]

    source_root = Path(__file__).resolve().parents[1] / "src/study"
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in source_root.glob("*.py")
        if path.name != "study.py"
    ).lower()
    for family_token in ("rotated-fcos", "tile-classifier", "nanodet", "mobilenet"):
        assert family_token not in production


def test_cross_namespace_pins_dependencies_and_unrelated_dirty_allowed(tmp_path: Path) -> None:
    repo, study_id, commit, _ = _training_repo(tmp_path)
    root = repo / "mldb_data"
    (repo / "README.md").write_text("dirty but unrelated\n", encoding="utf-8")
    unrelated = root / "unrelated-ns/namespace.yaml"
    unrelated.write_bytes(unrelated.read_bytes() + b"\n")

    plan = _compile(repo, study_id, commit)
    namespaces = [pin["id"] for pin in plan["pins"] if pin["kind"] == "namespace"]
    assert namespaces == ["arch-ns", "data-ns", "proto-ns", "study-ns", "task-ns"]
    assert "unrelated-ns" not in namespaces
    pin_keys = {(pin["kind"], pin["id"]) for pin in plan["pins"]}
    assert ("task", "task-ns/task-v1") in pin_keys
    assert ("architecture", "arch-ns/arch-a-v1") in pin_keys
    assert ("architecture", "arch-ns/arch-z-v1") in pin_keys
    assert ("train_protocol", "proto-ns/train-v1") in pin_keys
    assert ("evaluation_protocol", "proto-ns/eval-v1") in pin_keys
    declared_sources = {
        source["path"]
        for pin in plan["pins"]
        for source in pin["sources"]
    }
    assert declared_sources == {
        "mldb_data/arch-ns/lib/a.py",
        "mldb_data/arch-ns/lib/b.py",
        "mldb_data/proto-ns/lib/common.py",
    }


def test_existing_model_order_lineage_pins_and_no_runtime_loading(tmp_path: Path) -> None:
    repo, study_id, _commit, ids = _existing_repo(tmp_path)
    root = repo / "mldb_data"
    model2 = "model-ns/model-v2"
    result2 = "result-ns/result-v2"
    _write_json(
        root / "model-ns/models/model-v2.yaml",
        {"schema": "mjtensu.mldb-v2/model/v1", "id": model2, "training_result": result2},
    )
    _write_json(
        root / "result-ns/training_results/result-v2.yaml",
        {
            "schema": "mjtensu.mldb-v2/training-result/v1",
            "id": result2,
            "task": "task-ns/task-v1",
            "architecture": ids["architecture"],
            "corpus": ids["historical_corpus"],
            "train_protocol": ids["historical_protocol"],
            "status": "completed",
        },
    )
    study_path = root / "study-ns/studies/existing-v1.yaml"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["model"]["existing"] = [model2, ids["model"]]
    _write_json(study_path, study)
    commit = _commit_all(repo, "existing-model closure fixture")

    plan = _compile(repo, study_id, commit)
    assert [trial["source"] for trial in plan["trials"]] == [
        {"kind": "existing_model", "model": model2},
        {"kind": "existing_model", "model": ids["model"]},
    ]
    keys = {(pin["kind"], pin["id"]) for pin in plan["pins"]}
    assert ("model", model2) in keys
    assert ("model", ids["model"]) in keys
    assert ("training_result", result2) in keys
    assert ("training_result", ids["result"]) in keys
    assert ("architecture", ids["architecture"]) in keys
    assert all(len(trial["evaluations"]) == 1 for trial in plan["trials"])

    preflight_source = (
        Path(__file__).resolve().parents[1] / "src/study/_planning_preflight.py"
    ).read_text(encoding="utf-8").lower()
    assert "mldb_v2.src.model" not in preflight_source
    assert "torch" not in preflight_source
    assert "weights" not in preflight_source


def test_source_drift_rejects_required_dirty_and_stale_planning_input(tmp_path: Path) -> None:
    repo, study_id, commit_a, _ = _training_repo(tmp_path)
    root = repo / "mldb_data"
    planning_a = _planning(repo, study_id)
    study_path = root / "study-ns/studies/study-v1.yaml"
    original = study_path.read_bytes()

    study_path.write_bytes(original + b"\n")
    with pytest.raises(_SourcePinningError) as error:
        _build_study_plan(
            repository_root=repo,
            mldb_data_root=root,
            planning=planning_a,
            selected_commit=commit_a,
        )
    assert error.value.code == "required_source_dirty"

    study_path.write_bytes(original)
    study_b = json.loads(study_path.read_text(encoding="utf-8"))
    study_b["model"]["train"]["architectures"] = ["arch-ns/arch-a-v1"]
    _write_json(study_path, study_b)
    commit_b = _commit_all(repo, "stale planning-input fixture")

    with pytest.raises(_SourcePinningError) as error:
        _build_study_plan(
            repository_root=repo,
            mldb_data_root=root,
            planning=planning_a,
            selected_commit=commit_b,
        )
    assert error.value.code == "planning_input_stale"


def test_persistence_first_create_exact_replay_and_malformed_existing_reject(tmp_path: Path) -> None:
    repo, study_id, commit, _ = _training_repo(tmp_path)
    plan = _compile(repo, study_id, commit)
    created = _create_study_plan(repository_root=repo, plan=plan)
    replayed = _create_study_plan(repository_root=repo, plan=plan)
    assert created == replayed == plan

    namespace, local_id = plan["id"].split("/", 1)
    path = repo / "mldb_data" / namespace / "study_plans" / f"{local_id}.yaml"
    malformed = json.loads(path.read_text(encoding="utf-8"))
    malformed["content_sha256"] = "0" * 64
    _write_json(path, malformed)
    with pytest.raises((_StudyPlanError, ValueError)):
        _create_study_plan(repository_root=repo, plan=plan)
    assert json.loads(path.read_text(encoding="utf-8"))["content_sha256"] == "0" * 64


def test_persistence_conflicting_valid_same_identity_rejects(tmp_path: Path, monkeypatch) -> None:
    repo, study_id, commit, _ = _training_repo(tmp_path / "conflict")
    plan = _compile(repo, study_id, commit)
    _create_study_plan(repository_root=repo, plan=plan)

    conflicting = copy.deepcopy(dict(plan))
    conflicting["trials"][0]["evaluations"][0]["stage"] = "conflicting-stage"
    original_digest = plan["content_sha256"]
    monkeypatch.setattr(plan_build_module, "_content_digest", lambda _record: original_digest)
    with pytest.raises(ValueError, match="lifecycle conflict"):
        _create_study_plan(repository_root=repo, plan=conflicting)


def test_public_shape_and_planning_dependency_hygiene() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime = (root / "mldb_v2/src/study/plan.py").read_text(encoding="utf-8")
    frozen = (root / "mldb_v2/skeleton/study/plan.py").read_text(encoding="utf-8")
    import ast

    def public_shape(source: str) -> list[str]:
        tree = ast.parse(source)
        return [
            ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef))
        ]

    assert public_shape(runtime) == public_shape(frozen)
    planning_sources = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "mldb_v2/src/study/_planning_preflight.py",
            "mldb_v2/src/study/_grid_expansion.py",
            "mldb_v2/src/study/_source_pinning.py",
            "mldb_v2/src/study/_plan_build.py",
            "mldb_v2/src/study/plan.py",
        )
    ).lower()
    for forbidden in (
        "mldb_v2.skeleton",
        "from mldb.src",
        "import mldb.src",
        "clearml",
        "torch",
    ):
        assert forbidden not in planning_sources
