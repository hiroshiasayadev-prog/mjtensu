from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from mldb_v2.src.study._grid_expansion import (
    _ExpandedExistingModelSource,
    _ExpandedTrainingSource,
    _GridExpansionError,
    _expand_study_grid,
    _require_sequence_capacity,
)
from mldb_v2.src.study._planning_preflight import (
    _EvaluationPlanningInput,
    _ExistingModelPlanningEntry,
    _ExistingModelPlanningInput,
    _StudyPlanningInput,
    _StudyPlanningPreflight,
    _TrainingPlanningInput,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _evaluation(
    stage: str,
    *,
    axes: dict[str, object] | None = None,
    declarations: dict[str, object] | None = None,
) -> _EvaluationPlanningInput:
    protocol_id = f"demo/{stage}-protocol-v1"
    corpus_id = f"demo/{stage}-corpus-v1"
    stage_doc = {
        "stage": stage,
        "corpus": corpus_id,
        "protocol": protocol_id,
        "parameters": axes or {},
    }
    return _EvaluationPlanningInput(
        stage=stage_doc,
        corpus={"id": corpus_id},
        protocol={
            "id": protocol_id,
            "parameters": declarations
            or {"limit": {"default": 10, "type": "integer"}},
        },
    )


def _training_planning(
    *,
    architectures: tuple[str, ...] = ("demo/arch-a-v1",),
    axes: dict[str, object] | None = None,
    seeds: tuple[int, ...] = (7,),
    declarations: dict[str, object] | None = None,
    evaluations: tuple[_EvaluationPlanningInput, ...] | None = None,
) -> _StudyPlanningInput:
    model = _TrainingPlanningInput(
        corpus={"id": "demo/train-corpus-v1"},
        protocol={
            "id": "demo/train-protocol-v1",
            "parameters": declarations
            or {
                "zeta": {"default": 9, "type": "integer"},
                "alpha": {"default": 1, "type": "integer"},
                "batch_size": {"default": 32, "type": "integer"},
            },
        },
        architectures=tuple({"id": value} for value in architectures),
        parameter_axes=axes or {},
        seeds=seeds,
    )
    return _StudyPlanningInput(
        study={"id": "demo/study-v1"},
        task={"id": "demo/task-v1"},
        model=model,
        evaluations=evaluations or (_evaluation("final"),),
    )


def _existing_planning(
    *,
    models: tuple[str, ...] = ("demo/model-b", "demo/model-a"),
    evaluations: tuple[_EvaluationPlanningInput, ...] | None = None,
) -> _StudyPlanningInput:
    entries = tuple(
        _ExistingModelPlanningEntry(
            model={"id": model_id},
            training_result={"id": f"demo/result-{index}"},
            task_id="demo/task-v1",
            architecture_id="demo/arch-a-v1",
        )
        for index, model_id in enumerate(models, start=1)
    )
    return _StudyPlanningInput(
        study={"id": "demo/study-v1"},
        task={"id": "demo/task-v1"},
        model=_ExistingModelPlanningInput(models=entries),
        evaluations=evaluations or (_evaluation("final"),),
    )


def _parameter_dict(value) -> dict[str, object]:
    return dict(value)


def test_training_no_axes_uses_defaults_and_sequential_id() -> None:
    result = _expand_study_grid(_training_planning())
    assert [trial.trial for trial in result] == ["trial-0001"]
    source = result[0].source
    assert isinstance(source, _ExpandedTrainingSource)
    assert source.architecture == "demo/arch-a-v1"
    assert source.seed == 7
    assert list(source.parameters) == ["zeta", "alpha", "batch_size"]
    assert _parameter_dict(source.parameters) == {
        "zeta": 9,
        "alpha": 1,
        "batch_size": 32,
    }
    assert "seed" not in source.parameters


def test_training_cartesian_order_is_exact() -> None:
    planning = _training_planning(
        architectures=("demo/arch-b-v1", "demo/arch-a-v1"),
        axes={
            "zeta": {"values": [9, 7]},
            "alpha": {"values": [2, 1]},
        },
        seeds=(7, 3),
    )
    result = _expand_study_grid(planning)
    observed = [
        (
            trial.source.architecture,
            trial.source.parameters["alpha"],
            trial.source.parameters["zeta"],
            trial.source.seed,
        )
        for trial in result
        if isinstance(trial.source, _ExpandedTrainingSource)
    ]
    assert observed == [
        (architecture, alpha, zeta, seed)
        for architecture in ("demo/arch-b-v1", "demo/arch-a-v1")
        for alpha in (2, 1)
        for zeta in (9, 7)
        for seed in (7, 3)
    ]
    assert [trial.trial for trial in result] == [
        f"trial-{index:04d}" for index in range(1, 17)
    ]
    for trial in result:
        assert list(trial.source.parameters) == ["zeta", "alpha", "batch_size"]
        assert trial.source.parameters["batch_size"] == 32


def test_sparse_axis_inserts_all_omitted_defaults() -> None:
    planning = _training_planning(
        axes={"alpha": {"values": [4, 5]}},
    )
    result = _expand_study_grid(planning)
    assert [_parameter_dict(trial.source.parameters) for trial in result] == [
        {"zeta": 9, "alpha": 4, "batch_size": 32},
        {"zeta": 9, "alpha": 5, "batch_size": 32},
    ]


def test_type_sensitive_axis_keeps_bool_int_float_distinct() -> None:
    planning = _training_planning(
        axes={"choice": {"values": [True, 1, 1.0]}},
        declarations={"choice": {"default": None}},
    )
    result = _expand_study_grid(planning)
    values = [trial.source.parameters["choice"] for trial in result]
    assert [type(value) for value in values] == [bool, int, float]
    assert values == [True, 1, 1.0]


@pytest.mark.parametrize(
    "values",
    [
        [True, True],
        [1, 1],
        [1.0, 1.0],
        [[1, {"x": True}], [1, {"x": True}]],
        [
            {"x": [1, True], "y": 1.0},
            {"y": 1.0, "x": [1, True]},
        ],
    ],
)
def test_exact_axis_duplicates_fail_type_sensitively(values: list[object]) -> None:
    planning = _training_planning(
        axes={"choice": {"values": values}},
        declarations={"choice": {"default": None}},
    )
    with pytest.raises(_GridExpansionError, match="duplicate_parameter_axis_value"):
        _expand_study_grid(planning)


def test_evaluation_zero_axis_stage_emits_one_complete_coordinate() -> None:
    result = _expand_study_grid(_training_planning())
    coordinates = result[0].evaluations
    assert [item.coordinate for item in coordinates] == ["eval-0001"]
    assert coordinates[0].stage == "final"
    assert _parameter_dict(coordinates[0].parameters) == {"limit": 10}


def test_evaluation_order_spans_stages_and_resets_per_trial() -> None:
    stages = (
        _evaluation(
            "stage-z",
            axes={
                "zeta_eval": {"values": [8, 7]},
                "alpha_eval": {"values": [2, 1]},
            },
            declarations={
                "zeta_eval": {"default": 9, "type": "integer"},
                "alpha_eval": {"default": 0, "type": "integer"},
                "limit": {"default": 10, "type": "integer"},
            },
        ),
        _evaluation("stage-a"),
    )
    result = _expand_study_grid(
        _training_planning(seeds=(7, 3), evaluations=stages)
    )
    assert len(result) == 2
    for trial in result:
        assert [item.coordinate for item in trial.evaluations] == [
            "eval-0001",
            "eval-0002",
            "eval-0003",
            "eval-0004",
            "eval-0005",
        ]
        assert [item.stage for item in trial.evaluations] == [
            "stage-z",
            "stage-z",
            "stage-z",
            "stage-z",
            "stage-a",
        ]
        first_four = [
            (item.parameters["alpha_eval"], item.parameters["zeta_eval"])
            for item in trial.evaluations[:4]
        ]
        assert first_four == [(2, 8), (2, 7), (1, 8), (1, 7)]
        assert list(trial.evaluations[0].parameters) == [
            "zeta_eval",
            "alpha_eval",
            "limit",
        ]
        assert trial.evaluations[0].parameters["limit"] == 10


def test_existing_models_preserve_order_and_receive_evaluations() -> None:
    planning = _existing_planning(models=("demo/model-b", "demo/model-a"))
    result = _expand_study_grid(planning)
    assert [trial.trial for trial in result] == ["trial-0001", "trial-0002"]
    assert [trial.source.model for trial in result] == [
        "demo/model-b",
        "demo/model-a",
    ]
    assert all(isinstance(trial.source, _ExpandedExistingModelSource) for trial in result)
    assert all(set(trial.source.__dict__) == {"model", "kind"} for trial in result)
    assert all(
        [item.coordinate for item in trial.evaluations] == ["eval-0001"]
        for trial in result
    )


@pytest.mark.parametrize(
    ("count", "code"),
    [
        (9999, "trial_limit_exceeded"),
        (9999, "evaluation_coordinate_limit_exceeded"),
    ],
)
def test_sequence_capacity_accepts_exactly_9999(count: int, code: str) -> None:
    assert _require_sequence_capacity(count, code=code) == 9999


@pytest.mark.parametrize(
    "code",
    ["trial_limit_exceeded", "evaluation_coordinate_limit_exceeded"],
)
def test_sequence_capacity_rejects_10000(code: str) -> None:
    with pytest.raises(_GridExpansionError, match=code):
        _require_sequence_capacity(10000, code=code)


def test_expansion_is_pure_and_repeatable() -> None:
    planning = _training_planning(
        architectures=("demo/arch-b-v1", "demo/arch-a-v1"),
        axes={"alpha": {"values": [2, 1]}},
        seeds=(7, 3),
    )
    before = copy.deepcopy(planning)
    first = _expand_study_grid(planning)
    second = _expand_study_grid(planning)
    assert planning == before
    assert first == second


def test_expansion_source_has_no_forbidden_dependencies() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/study/_grid_expansion.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "mldb.src",
        "mldb_v2.skeleton",
        "backend",
        "ClearML",
        "torch",
        "PlanPin",
        "from mldb_v2.src.study.plan",
        "source_commit",
        "subprocess",
        "git ",
    ):
        assert forbidden not in source


def test_evaluation_single_axis_preserves_authored_values() -> None:
    stage = _evaluation(
        "single",
        axes={"threshold": {"values": [0.7, 0.3]}},
        declarations={
            "threshold": {"default": 0.5, "type": "number"},
            "limit": {"default": 10, "type": "integer"},
        },
    )
    result = _expand_study_grid(_training_planning(evaluations=(stage,)))
    assert [item.parameters["threshold"] for item in result[0].evaluations] == [
        0.7,
        0.3,
    ]
    assert [_parameter_dict(item.parameters) for item in result[0].evaluations] == [
        {"threshold": 0.7, "limit": 10},
        {"threshold": 0.3, "limit": 10},
    ]


def test_preflight_to_expansion_integration_is_read_only(tmp_path: Path) -> None:
    from mldb_v2.tests.test_study_planning_preflight import (
        _Verifier,
        _training_repo,
        _tree_bytes,
    )

    root, study_id = _training_repo(tmp_path)
    for path in root.rglob("*.py"):
        path.write_text("raise AssertionError('companion executed')\n", encoding="utf-8")
    before = _tree_bytes(root)
    planning = _StudyPlanningPreflight(
        mldb_data_root=root,
        verifier=_Verifier(),
    ).prepare(study_id)
    result = _expand_study_grid(planning)
    assert len(result) == 16
    first = result[0]
    assert first.trial == "trial-0001"
    assert isinstance(first.source, _ExpandedTrainingSource)
    assert first.source.architecture == "demo/arch-b-v1"
    assert first.source.parameters["alpha"] == 2
    assert first.source.parameters["zeta"] == 9
    assert first.source.seed == 7
    assert [item.coordinate for item in first.evaluations] == [
        f"eval-{index:04d}" for index in range(1, 9)
    ]
    assert [item.stage for item in first.evaluations] == [
        "stage-z",
        "stage-z",
        "stage-z",
        "stage-z",
        "stage-a",
        "stage-a",
        "stage-a",
        "stage-a",
    ]
    assert _tree_bytes(root) == before
