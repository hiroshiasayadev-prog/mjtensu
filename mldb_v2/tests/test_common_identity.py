import math

import pytest

from mldb_v2.src.common.ids import (
    DefinitionKind,
    EntityKind,
    _canonical_json_bytes,
    _validate_evaluation_coordinate_id,
    _validate_namespace_id,
    _validate_trial_id,
    _validate_typed_reference,
)


@pytest.mark.parametrize("value", ["a", "abc-123", "0", "a0-b9"])
def test_namespace_id_accepts_common_kebab_segments(value: str) -> None:
    assert _validate_namespace_id(value) == value


@pytest.mark.parametrize(
    "value", ["", "A", "a_b", "-a", "a-", "a--b", "a/b", "é"]
)
def test_namespace_id_rejects_invalid_segments(value: str) -> None:
    with pytest.raises(ValueError):
        _validate_namespace_id(value)


def test_typed_reference_accepts_exact_namespace_and_local_id() -> None:
    value = "mahjong/tile-classification-v1"
    assert _validate_typed_reference(value) == value


@pytest.mark.parametrize(
    "value",
    ["mahjong", "/task", "ns/", "ns/task/extra", "Name/task", "ns/task_name"],
)
def test_typed_reference_rejects_invalid_forms(value: str) -> None:
    with pytest.raises(ValueError):
        _validate_typed_reference(value)


@pytest.mark.parametrize("value", ["trial-0001", "trial-0042", "trial-9999"])
def test_trial_id_accepts_canonical_form(value: str) -> None:
    assert _validate_trial_id(value) == value


@pytest.mark.parametrize("value", ["trial-0000", "trial-1", "trial-10000", "Trial-0001"])
def test_trial_id_rejects_invalid_forms(value: str) -> None:
    with pytest.raises(ValueError):
        _validate_trial_id(value)


@pytest.mark.parametrize("value", ["eval-0001", "eval-0042", "eval-9999"])
def test_evaluation_coordinate_id_accepts_canonical_form(value: str) -> None:
    assert _validate_evaluation_coordinate_id(value) == value


@pytest.mark.parametrize("value", ["eval-0000", "eval-1", "eval-10000", "Eval-0001"])
def test_evaluation_coordinate_id_rejects_invalid_forms(value: str) -> None:
    with pytest.raises(ValueError):
        _validate_evaluation_coordinate_id(value)


def test_canonical_json_bytes_are_sorted_compact_utf8_and_lf_free() -> None:
    value = {"z": None, "b": 1.0, "a": True, "u": "牌"}
    encoded = _canonical_json_bytes(value)
    assert encoded == '{"a":true,"b":1.0,"u":"牌","z":null}'.encode("utf-8")
    assert not encoded.endswith(b"\n")


def test_canonical_json_preserves_bool_int_float_distinction() -> None:
    assert _canonical_json_bytes([True, 1, 1.0]) == b"[true,1,1.0]"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError):
        _canonical_json_bytes({"value": value})


@pytest.mark.parametrize("value", [{1: "bad"}, (1, 2)])
def test_canonical_json_rejects_non_json_python_values(value: object) -> None:
    with pytest.raises(ValueError):
        _canonical_json_bytes(value)


def test_common_enums_match_frozen_skeleton_values() -> None:
    assert EntityKind.STUDY_RESULT.value == "study_result"
    assert DefinitionKind.EVALUATION_PROTOCOL.value == "evaluation_protocol"
