from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from mldb_v2.src.common.ids import EntityKind, _canonical_json_bytes
from mldb_v2.src.study._grid_expansion import (
    _ExpandedEvaluationCoordinate,
    _ExpandedExistingModelSource,
    _ExpandedTrainingSource,
    _ExpandedTrial,
)
from mldb_v2.src.study._plan_build import (
    _StudyPlanError,
    _StudyPlanRecordValidator,
    _build_from_parts,
    _build_study_plan,
    _content_digest,
    _create_study_plan,
    _plan_id,
    _validate_study_plan,
)
from mldb_v2.src.study._planning_preflight import _StudyPlanningInput
from mldb_v2.src.study._source_pinning import (
    _CommittedSourcePin,
    _CommittedSourcePinCollection,
    _PinnedExecutableSource,
)
from mldb_v2.tests.test_study_source_pinning import (
    _commit_all,
    _existing_repo,
    _planning,
    _training_repo,
    _write_json,
)


def _hex(char: str) -> str:
    return char * 64


def _pin(
    kind: str,
    entity_id: str,
    *,
    companion: str | None = None,
    sources: tuple[_PinnedExecutableSource, ...] = (),
    manifest: str | None = None,
    entries: int | None = None,
) -> _CommittedSourcePin:
    return _CommittedSourcePin(
        kind=kind,
        id=entity_id,
        yaml_sha256=_hex("a"),
        companion_sha256=companion,
        sources=sources,
        manifest_sha256=manifest,
        manifest_entries=entries,
    )


def _unit_parts():
    planning = _StudyPlanningInput(
        study={"id": "demo/study-v1"},
        task={"id": "demo/task-v1"},
        model=None,  # not consumed by the T003-04 mechanical-conversion seam
        evaluations=(),
    )
    evaluations = (
        _ExpandedEvaluationCoordinate(
            coordinate="eval-0001",
            stage="holdout-a",
            task="demo/task-v1",
            corpus="demo/eval-corpus-v1",
            evaluation_protocol="demo/eval-protocol-v1",
            parameters={"threshold": 0.5},
        ),
        _ExpandedEvaluationCoordinate(
            coordinate="eval-0002",
            stage="holdout-b",
            task="demo/task-v1",
            corpus="demo/eval-corpus-v1",
            evaluation_protocol="demo/eval-protocol-v1",
            parameters={"threshold": 0.75},
        ),
    )
    expansion = (
        _ExpandedTrial(
            trial="trial-0001",
            source=_ExpandedTrainingSource(
                task="demo/task-v1",
                corpus="demo/train-corpus-v1",
                architecture="demo/arch-a-v1",
                train_protocol="demo/train-protocol-v1",
                parameters={"batch_size": 32, "ratio": [1, 1.0, True]},
                seed=7,
            ),
            evaluations=evaluations,
        ),
    )
    sources = (
        _PinnedExecutableSource(path="mldb_data/demo/lib/a.py", sha256=_hex("b")),
        _PinnedExecutableSource(path="mldb_data/demo/lib/z.py", sha256=_hex("c")),
    )
    pins = (
        _pin("namespace", "demo"),
        _pin("task", "demo/task-v1"),
        _pin("corpus", "demo/eval-corpus-v1", manifest=_hex("d"), entries=1),
        _pin("corpus", "demo/train-corpus-v1", manifest=_hex("e"), entries=2),
        _pin("architecture", "demo/arch-a-v1", companion=_hex("b"), sources=sources),
        _pin("architecture", "demo/arch-b-v1", companion=_hex("c")),
        _pin("train_protocol", "demo/train-protocol-v1", companion=_hex("d")),
        _pin("evaluation_protocol", "demo/eval-protocol-v1", companion=_hex("e")),
        _pin("study", "demo/study-v1"),
    )
    collection = _CommittedSourcePinCollection(source_commit="1" * 40, pins=pins)
    return planning, expansion, collection


def _base_plan() -> dict[str, object]:
    planning, expansion, collection = _unit_parts()
    return copy.deepcopy(dict(_build_from_parts(
        planning=planning,
        expansion=expansion,
        pin_collection=collection,
    )))


def _resign(plan: dict[str, object]) -> None:
    payload = {
        "schema": plan["schema"],
        "study": plan["study"],
        "source_commit": plan["source_commit"],
        "pins": plan["pins"],
        "trials": plan["trials"],
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    plan["content_sha256"] = digest
    plan["id"] = _plan_id(str(plan["study"]), digest)


def _expect_invalid(plan: dict[str, object]) -> None:
    with pytest.raises((_StudyPlanError, ValueError)):
        _validate_study_plan(plan)


def test_public_plan_shape_exactly_matches_frozen_skeleton_ast() -> None:
    root = Path(__file__).parents[2]
    frozen = ast.parse((root / "mldb_v2/skeleton/study/plan.py").read_text(encoding="utf-8"))
    runtime = ast.parse((root / "mldb_v2/src/study/plan.py").read_text(encoding="utf-8"))

    def public_shape(tree: ast.Module):
        return [
            ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef))
        ]

    assert public_shape(runtime) == public_shape(frozen)


def test_compiler_binds_same_planning_to_expansion_and_pins_and_uses_pin_commit() -> None:
    planning, expansion, collection = _unit_parts()
    seen: list[object] = []

    def expand(arg):
        seen.append(arg)
        return expansion

    def collect(**kwargs):
        seen.append(kwargs["planning"])
        assert kwargs["selected_commit"] == "f" * 40
        return _CommittedSourcePinCollection(source_commit="1" * 40, pins=collection.pins)

    plan = _build_study_plan(
        repository_root="unused",
        mldb_data_root="unused",
        planning=planning,
        selected_commit="f" * 40,
        grid_expander=expand,
        source_pin_collector=collect,
    )
    assert seen == [planning, planning]
    assert plan["source_commit"] == "1" * 40


def test_digest_excludes_only_identity_fields_and_id_is_study_derived() -> None:
    plan = _base_plan()
    payload = {key: value for key, value in plan.items() if key not in {"id", "content_sha256"}}
    expected = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    assert plan["content_sha256"] == expected == _content_digest(plan)
    assert plan["id"] == f"demo/study-v1-plan-{expected[:16]}"


def test_validator_rejects_top_level_identity_and_commit_failures() -> None:
    for mutate in (
        lambda p: p.__setitem__("schema", "wrong"),
        lambda p: p.pop("trials"),
        lambda p: p.__setitem__("extra", 1),
        lambda p: p.__setitem__("study", "bad"),
        lambda p: p.__setitem__("source_commit", "abc"),
        lambda p: p.__setitem__("content_sha256", "0" * 64),
        lambda p: p.__setitem__("id", "demo/wrong-plan-0000000000000000"),
    ):
        plan = _base_plan()
        mutate(plan)
        _expect_invalid(plan)


def test_validator_rejects_pin_order_structure_and_reference_failures() -> None:
    mutations = []

    def duplicate(p):
        p["pins"].insert(1, copy.deepcopy(p["pins"][0]))
    mutations.append(duplicate)

    def kind_order(p):
        p["pins"][1], p["pins"][2] = p["pins"][2], p["pins"][1]
    mutations.append(kind_order)

    def id_order(p):
        arch = [i for i, pin in enumerate(p["pins"]) if pin["kind"] == "architecture"]
        p["pins"][arch[0]], p["pins"][arch[1]] = p["pins"][arch[1]], p["pins"][arch[0]]
    mutations.append(id_order)

    def missing_namespace(p):
        p["pins"].pop(0)
    mutations.append(missing_namespace)

    def missing_ref(p):
        p["pins"] = [pin for pin in p["pins"] if pin["id"] != "demo/arch-a-v1"]
    mutations.append(missing_ref)

    def bad_task_companion(p):
        next(pin for pin in p["pins"] if pin["kind"] == "task")["companion_sha256"] = _hex("f")
    mutations.append(bad_task_companion)

    def missing_exec_companion(p):
        next(pin for pin in p["pins"] if pin["kind"] == "architecture")["companion_sha256"] = None
    mutations.append(missing_exec_companion)

    def null_manifest(p):
        next(pin for pin in p["pins"] if pin["kind"] == "corpus")["manifest_sha256"] = None
    mutations.append(null_manifest)

    def bool_entries(p):
        next(pin for pin in p["pins"] if pin["kind"] == "corpus")["manifest_entries"] = True
    mutations.append(bool_entries)

    def bad_sha(p):
        p["pins"][0]["yaml_sha256"] = "A" * 64
    mutations.append(bad_sha)

    for mutate in mutations:
        plan = _base_plan()
        mutate(plan)
        _expect_invalid(plan)


def test_validator_accepts_same_namespace_lib_sources() -> None:
    plan = _base_plan()
    assert _validate_study_plan(plan) == plan
    _StudyPlanRecordValidator().validate(
        kind=EntityKind.STUDY_PLAN, entity_id=plan["id"], document=plan
    )


@pytest.mark.parametrize("path", ["product/a.py", "mldb_data/other/lib/a.py", "mldb_data/demo/architectures/helper.py"])
def test_validator_rejects_executable_sources_outside_same_namespace_lib(path: str) -> None:
    plan = _base_plan()
    pin = next(pin for pin in plan["pins"] if pin["id"] == "demo/arch-a-v1")
    pin["sources"][0]["path"] = path
    pin["sources"].sort(key=lambda source: source["path"])
    _resign(plan)
    with pytest.raises(_StudyPlanError, match="invalid_pin_source_path"):
        _validate_study_plan(plan)


def test_validator_rejects_trial_evaluation_and_parameter_failures() -> None:
    mutations = []

    def trial_gap(p):
        p["trials"][0]["trial"] = "trial-0002"
    mutations.append(trial_gap)

    def bad_trial_format(p):
        p["trials"][0]["trial"] = "trial-0000"
    mutations.append(bad_trial_format)

    def mixed_source(p):
        p["trials"][0]["source"]["model"] = "demo/model-v1"
    mutations.append(mixed_source)

    def missing_source_field(p):
        p["trials"][0]["source"].pop("seed")
    mutations.append(missing_source_field)

    def bool_seed(p):
        p["trials"][0]["source"]["seed"] = True
    mutations.append(bool_seed)

    def eval_gap(p):
        p["trials"][0]["evaluations"][0]["coordinate"] = "eval-0002"
    mutations.append(eval_gap)

    def eval_reorder(p):
        p["trials"][0]["evaluations"].reverse()
    mutations.append(eval_reorder)

    def bad_train_parameter(p):
        p["trials"][0]["source"]["parameters"]["bad"] = float("nan")
    mutations.append(bad_train_parameter)

    def bad_eval_parameter(p):
        p["trials"][0]["evaluations"][0]["parameters"]["bad"] = object()
    mutations.append(bad_eval_parameter)

    def missing_eval_ref(p):
        p["pins"] = [pin for pin in p["pins"] if pin["kind"] != "evaluation_protocol"]
    mutations.append(missing_eval_ref)

    for mutate in mutations:
        plan = _base_plan()
        mutate(plan)
        _expect_invalid(plan)


def test_validator_accepts_existing_model_source_and_rejects_mixed_fields() -> None:
    planning, _, collection = _unit_parts()
    pins = list(collection.pins)
    pins.append(_pin("model", "demo/model-v1"))
    expansion = (
        _ExpandedTrial(
            trial="trial-0001",
            source=_ExpandedExistingModelSource(model="demo/model-v1"),
            evaluations=(
                _ExpandedEvaluationCoordinate(
                    coordinate="eval-0001",
                    stage="holdout",
                    task="demo/task-v1",
                    corpus="demo/eval-corpus-v1",
                    evaluation_protocol="demo/eval-protocol-v1",
                    parameters={"threshold": 0.5},
                ),
            ),
        ),
    )
    plan = dict(_build_from_parts(
        planning=planning,
        expansion=expansion,
        pin_collection=_CommittedSourcePinCollection(source_commit="1" * 40, pins=tuple(pins)),
    ))
    assert plan["trials"][0]["source"] == {"kind": "existing_model", "model": "demo/model-v1"}
    broken = copy.deepcopy(plan)
    broken["trials"][0]["source"]["seed"] = 7
    _expect_invalid(broken)


def test_training_synthetic_git_end_to_end_compile_write_replay_and_determinism(tmp_path: Path) -> None:
    repo, study_id, _initial_commit, _ = _training_repo(tmp_path)
    root = repo / "mldb_data"

    train_path = root / "proto-ns/train_protocols/train-v1.yaml"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    train["parameters"] = {
        "batch_size": {"default": 32, "type": "integer"},
        "lr": {"default": 0.001, "type": "number"},
    }
    _write_json(train_path, train)

    eval_path = root / "proto-ns/evaluation_protocols/eval-v1.yaml"
    eval_doc = json.loads(eval_path.read_text(encoding="utf-8"))
    eval_doc["parameters"] = {
        "threshold": {"default": 0.5, "type": "number"},
        "limit": {"default": 10, "type": "integer"},
    }
    _write_json(eval_path, eval_doc)

    study_path = root / "study-ns/studies/study-v1.yaml"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["model"]["train"]["seeds"] = [7, 3]
    study["model"]["train"]["parameters"] = {"lr": {"values": [0.001, 0.01]}}
    study["evaluations"][0]["parameters"] = {"threshold": {"values": [0.4, 0.6]}}
    study["evaluations"][1]["parameters"] = {"limit": {"values": [5, 10]}}
    _write_json(study_path, study)
    commit = _commit_all(repo, "plan grid fixture")

    planning = _planning(repo, study_id)
    first = _build_study_plan(
        repository_root=repo,
        mldb_data_root=root,
        planning=planning,
        selected_commit=commit,
    )
    second = _build_study_plan(
        repository_root=repo,
        mldb_data_root=root,
        planning=planning,
        selected_commit=commit,
    )
    assert first == second
    assert len(first["trials"]) == 8
    assert [trial["trial"] for trial in first["trials"]] == [f"trial-{i:04d}" for i in range(1, 9)]
    assert [trial["source"]["architecture"] for trial in first["trials"]] == [
        "arch-ns/arch-z-v1"
    ] * 4 + ["arch-ns/arch-a-v1"] * 4
    assert [trial["source"]["seed"] for trial in first["trials"]] == [7, 3, 7, 3, 7, 3, 7, 3]
    assert all(trial["source"]["parameters"]["batch_size"] == 32 for trial in first["trials"])
    assert all([e["coordinate"] for e in trial["evaluations"]] == ["eval-0001", "eval-0002", "eval-0003", "eval-0004"] for trial in first["trials"])
    assert [(pin["kind"], pin["id"]) for pin in first["pins"]] == sorted(
        [(pin["kind"], pin["id"]) for pin in first["pins"]],
        key=lambda item: (("namespace", "task", "corpus", "architecture", "train_protocol", "evaluation_protocol", "study", "model", "training_result").index(item[0]), item[1]),
    )

    stored = _create_study_plan(repository_root=repo, plan=first)
    replay = _create_study_plan(repository_root=repo, plan=first)
    assert stored == replay == first
    namespace, local = first["id"].split("/", 1)
    path = root / namespace / "study_plans" / f"{local}.yaml"
    assert path.is_file()
    assert len(list(path.parent.glob("*.yaml"))) == 1


def test_existing_model_synthetic_plan_preserves_order_and_lineage_pins(tmp_path: Path) -> None:
    repo, study_id, _commit, ids = _existing_repo(tmp_path)
    root = repo / "mldb_data"

    model2 = "model-ns/model-v2"
    result2 = "result-ns/result-v2"
    _write_json(root / "model-ns/models/model-v2.yaml", {
        "schema": "mjtensu.mldb-v2/model/v1", "id": model2, "training_result": result2
    })
    _write_json(root / "result-ns/training_results/result-v2.yaml", {
        "schema": "mjtensu.mldb-v2/training-result/v1",
        "id": result2,
        "task": "task-ns/task-v1",
        "architecture": ids["architecture"],
        "corpus": ids["historical_corpus"],
        "train_protocol": ids["historical_protocol"],
        "status": "completed",
    })
    study_path = root / "study-ns/studies/existing-v1.yaml"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["model"]["existing"] = [model2, ids["model"]]
    _write_json(study_path, study)
    commit = _commit_all(repo, "two existing models")

    planning = _planning(repo, study_id)
    plan = _build_study_plan(
        repository_root=repo,
        mldb_data_root=root,
        planning=planning,
        selected_commit=commit,
    )
    assert [trial["source"] for trial in plan["trials"]] == [
        {"kind": "existing_model", "model": model2},
        {"kind": "existing_model", "model": ids["model"]},
    ]
    keys = {(pin["kind"], pin["id"]) for pin in plan["pins"]}
    assert ("model", model2) in keys and ("model", ids["model"]) in keys
    assert ("training_result", result2) in keys and ("training_result", ids["result"]) in keys
    assert ("architecture", ids["architecture"]) in keys
    assert all(len(trial["evaluations"]) == 1 for trial in plan["trials"])
    assert plan == _build_study_plan(
        repository_root=repo,
        mldb_data_root=root,
        planning=planning,
        selected_commit=commit,
    )


def test_invalid_existing_plan_is_not_accepted_as_replay(tmp_path: Path) -> None:
    repo, study_id, commit, _ = _training_repo(tmp_path)
    plan = _build_study_plan(
        repository_root=repo,
        mldb_data_root=repo / "mldb_data",
        planning=_planning(repo, study_id),
        selected_commit=commit,
    )
    _create_study_plan(repository_root=repo, plan=plan)
    namespace, local = plan["id"].split("/", 1)
    path = repo / "mldb_data" / namespace / "study_plans" / f"{local}.yaml"
    malformed = json.loads(path.read_text(encoding="utf-8"))
    malformed["content_sha256"] = "0" * 64
    _write_json(path, malformed)
    with pytest.raises((_StudyPlanError, ValueError)):
        _create_study_plan(repository_root=repo, plan=plan)
    assert json.loads(path.read_text(encoding="utf-8"))["content_sha256"] == "0" * 64


def test_same_identity_conflict_flows_through_w001_writer(tmp_path: Path, monkeypatch) -> None:
    repo, study_id, commit, _ = _training_repo(tmp_path)
    plan = _build_study_plan(
        repository_root=repo,
        mldb_data_root=repo / "mldb_data",
        planning=_planning(repo, study_id),
        selected_commit=commit,
    )
    _create_study_plan(repository_root=repo, plan=plan)
    conflicting = copy.deepcopy(dict(plan))
    conflicting["trials"][0]["evaluations"][0]["stage"] = "different-stage"
    original_digest = plan["content_sha256"]
    import mldb_v2.src.study._plan_build as module
    monkeypatch.setattr(module, "_content_digest", lambda _record: original_digest)
    with pytest.raises(ValueError, match="lifecycle conflict"):
        _create_study_plan(repository_root=repo, plan=conflicting)


def test_study_plan_record_validator_rejects_wrong_immutable_kind() -> None:
    plan = _base_plan()
    validator = _StudyPlanRecordValidator()
    with pytest.raises(ValueError, match="only accepts study_plan"):
        validator.validate(kind=EntityKind.MODEL, entity_id=plan["id"], document=plan)
