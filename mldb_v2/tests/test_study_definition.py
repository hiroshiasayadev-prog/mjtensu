from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from mldb_v2.src.study._study_validation import _load_study_definition, _validate_study
from mldb_v2.src.study.study import (
    EvaluationStage,
    ExistingStudyModelSource,
    ParameterAxis,
    Study,
    TrainingModelSource,
    TrainingStudyModelSource,
)


def _study() -> dict[str, object]:
    return {
        "schema": "mjtensu.mldb-v2/study/v1",
        "id": "demo/study-v1",
        "status": "draft",
        "name": "Study",
        "description": "description",
        "model": {
            "train": {
                "corpus": "demo/train-corpus-v1",
                "protocol": "demo/train-v1",
                "architectures": ["demo/arch-a-v1", "demo/arch-b-v2"],
                "parameters": {
                    "p": {"values": [True, 1, 1.0, "1", [1, True], {"x": 1}]}
                },
                "seeds": [42, -1],
            }
        },
        "evaluations": [
            {
                "stage": "holdout-a",
                "corpus": "demo/eval-corpus-v1",
                "protocol": "demo/eval-v1",
                "parameters": {"threshold": {"values": [0.25, 0.5]}},
            }
        ],
    }


def _existing_study() -> dict[str, object]:
    value = _study()
    value["model"] = {"existing": ["demo/run-abcd-trial-0001-model", "other/model-xyz"]}
    return value


def test_current_study_examples_parse_locally() -> None:
    root = Path(__file__).resolve().parents[2] / "mldb_data"
    assert _load_study_definition(root, "tile-classifier/rotation-robustness-example-v1")["id"] == "tile-classifier/rotation-robustness-example-v1"
    assert _load_study_definition(root, "rotated-fcos/spatial-screen-example-v1")["id"] == "rotated-fcos/spatial-screen-example-v1"


def test_training_and_existing_sources_parse_and_preserve_authored_order() -> None:
    training = _validate_study(_study())
    assert training["model"]["train"]["architectures"] == ["demo/arch-a-v1", "demo/arch-b-v2"]
    assert training["model"]["train"]["seeds"] == [42, -1]
    existing = _validate_study(_existing_study())
    assert existing["model"]["existing"] == ["demo/run-abcd-trial-0001-model", "other/model-xyz"]


@pytest.mark.parametrize("mutation", [
    {"schema": "wrong"},
    {"id": "demo/study"},
    {"id": "demo/study-v0"},
    {"id": "demo/study-v01"},
    {"id": "Demo/study-v1"},
    {"status": "complete"},
    {"name": ""},
    {"evaluations": []},
])
def test_invalid_top_level_semantics_reject(mutation: dict[str, object]) -> None:
    value = _study(); value.update(mutation)
    with pytest.raises(ValueError):
        _validate_study(value)


def test_unknown_top_level_and_path_id_mismatch_reject() -> None:
    value = _study(); value["unexpected"] = 1
    with pytest.raises(ValueError):
        _validate_study(value)
    with pytest.raises(ValueError, match="path identity"):
        _validate_study(_study(), expected_id="demo/other-v1")


@pytest.mark.parametrize("model", [
    {},
    {"train": _study()["model"]["train"], "existing": ["demo/model"]},
    {"other": {}},
    {"train": _study()["model"]["train"], "extra": 1},
])
def test_model_union_is_exact(model: object) -> None:
    value = _study(); value["model"] = model
    with pytest.raises(ValueError):
        _validate_study(value)


def test_training_source_requires_exact_fields() -> None:
    value = _study(); source = copy.deepcopy(value["model"]["train"])
    del source["seeds"]
    value["model"] = {"train": source}
    with pytest.raises(ValueError): _validate_study(value)
    source = copy.deepcopy(_study()["model"]["train"]); source["extra"] = 1
    value["model"] = {"train": source}
    with pytest.raises(ValueError): _validate_study(value)


@pytest.mark.parametrize(("field", "bad"), [
    ("corpus", "demo/corpus"),
    ("protocol", "demo/train"),
    ("corpus", "Demo/corpus-v1"),
])
def test_training_corpus_and_protocol_are_versioned_refs(field: str, bad: object) -> None:
    value = _study(); value["model"]["train"][field] = bad
    with pytest.raises(ValueError): _validate_study(value)


def test_architectures_are_nonempty_unique_versioned_and_ordered() -> None:
    for architectures in [[], ["demo/arch"], ["demo/arch-v1", "demo/arch-v1"]]:
        value = _study(); value["model"]["train"]["architectures"] = architectures
        with pytest.raises(ValueError): _validate_study(value)
    parsed = _validate_study(_study())
    assert parsed["model"]["train"]["architectures"] == ["demo/arch-a-v1", "demo/arch-b-v2"]


def test_axis_exact_shape_nonempty_and_finite_values() -> None:
    for axis in [{}, {"values": [], "x": 1}, {"values": []}, {"values": [math.nan]}]:
        value = _study(); value["model"]["train"]["parameters"] = {"p": axis}
        with pytest.raises(ValueError): _validate_study(value)


def test_axis_accepts_nested_values_and_type_sensitive_distinct_scalars() -> None:
    value = _study()
    values = [True, 1, 1.0, "1", [1, {"x": False}], {"a": [1.0, True]}]
    value["model"]["train"]["parameters"] = {"p": {"values": values}}
    assert _validate_study(value)["model"]["train"]["parameters"]["p"]["values"] is values


@pytest.mark.parametrize("values", [
    [True, True], [1, 1], [1.0, 1.0], ["x", "x"],
    [[1, True], [1, True]], [{"a": 1, "b": True}, {"b": True, "a": 1}],
])
def test_axis_rejects_exact_recursive_type_sensitive_duplicates(values: list[object]) -> None:
    value = _study(); value["model"]["train"]["parameters"] = {"p": {"values": values}}
    with pytest.raises(ValueError): _validate_study(value)


def test_axis_value_order_is_preserved() -> None:
    value = _study(); values = [3, 1, 2]
    value["model"]["train"]["parameters"] = {"p": {"values": values}}
    assert _validate_study(value)["model"]["train"]["parameters"]["p"]["values"] == [3, 1, 2]


@pytest.mark.parametrize("seeds", [[], [True], [1.0], ["1"], [1, 1]])
def test_seeds_are_nonempty_unique_exact_ints(seeds: list[object]) -> None:
    value = _study(); value["model"]["train"]["seeds"] = seeds
    with pytest.raises(ValueError): _validate_study(value)


def test_existing_models_use_common_typed_ref_without_version_suffix_requirement() -> None:
    value = _existing_study()
    assert _validate_study(value)["model"]["existing"][0] == "demo/run-abcd-trial-0001-model"
    for models in [[], ["model-only"], ["Demo/model"], ["demo/model", "demo/model"]]:
        candidate = _existing_study(); candidate["model"] = {"existing": models}
        with pytest.raises(ValueError): _validate_study(candidate)


def test_evaluation_stage_exact_shape_unique_kebab_refs_and_axes() -> None:
    value = _study(); stage = value["evaluations"][0]
    assert _validate_study(value)["evaluations"][0]["stage"] == "holdout-a"
    for bad_stage in ["", "Holdout", "hold_out", "-holdout"]:
        candidate = _study(); candidate["evaluations"][0]["stage"] = bad_stage
        with pytest.raises(ValueError): _validate_study(candidate)
    candidate = _study(); candidate["evaluations"].append(copy.deepcopy(candidate["evaluations"][0]))
    with pytest.raises(ValueError): _validate_study(candidate)
    for field in ["corpus", "protocol"]:
        candidate = _study(); candidate["evaluations"][0][field] = "demo/unversioned"
        with pytest.raises(ValueError): _validate_study(candidate)
    candidate = _study(); candidate["evaluations"][0]["extra"] = 1
    with pytest.raises(ValueError): _validate_study(candidate)


def test_evaluation_axis_uses_same_contract() -> None:
    value = _study(); value["evaluations"][0]["parameters"] = {"p": {"values": [False, 0, 0.0]}}
    assert _validate_study(value)["evaluations"][0]["parameters"]["p"]["values"] == [False, 0, 0.0]
    value = _study(); value["evaluations"][0]["parameters"] = {"p": {"values": [[1], [1]]}}
    with pytest.raises(ValueError): _validate_study(value)


def test_public_study_shape_is_exact_skeleton_mirror() -> None:
    repo = Path(__file__).resolve().parents[2]
    runtime = (repo / "mldb_v2/src/study/study.py").read_text(encoding="utf-8")
    skeleton = (repo / "mldb_v2/skeleton/study/study.py").read_text(encoding="utf-8")
    assert runtime.replace("\r\n", "\n") == skeleton.replace("mldb_v2.skeleton.", "mldb_v2.src.").replace("\r\n", "\n")
    assert ParameterAxis.__required_keys__ == frozenset({"values"})
    assert TrainingModelSource.__required_keys__ == frozenset({"corpus", "protocol", "architectures", "parameters", "seeds"})
    assert TrainingStudyModelSource.__required_keys__ == frozenset({"train"})
    assert ExistingStudyModelSource.__required_keys__ == frozenset({"existing"})
    assert EvaluationStage.__required_keys__ == frozenset({"stage", "corpus", "protocol", "parameters"})
    assert Study.__required_keys__ == frozenset({"schema", "id", "status", "name", "description", "model", "evaluations"})
