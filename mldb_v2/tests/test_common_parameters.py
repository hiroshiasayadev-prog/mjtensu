import math

import pytest

from mldb_v2.src.common.parameters import (
    _public_parameter_values_equal,
    _resolve_public_parameters,
    _validate_public_parameter_declaration,
    _validate_public_parameter_value,
    _validate_training_seed,
)


def test_public_parameter_value_accepts_nested_json_values_without_coercion() -> None:
    value = {
        "enabled": True,
        "items": [None, "x", 1, 1.0, {"nested": False}],
    }
    assert _validate_public_parameter_value(value) is value


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_public_parameter_value_rejects_non_finite_float(value: float) -> None:
    with pytest.raises(ValueError):
        _validate_public_parameter_value([{"value": value}])


def test_public_parameter_value_rejects_non_string_mapping_key() -> None:
    with pytest.raises(ValueError):
        _validate_public_parameter_value({1: "bad"})


def test_type_sensitive_recursive_equality_distinguishes_bool_int_float() -> None:
    assert not _public_parameter_values_equal(True, 1)
    assert not _public_parameter_values_equal(1, 1.0)
    assert not _public_parameter_values_equal(True, 1.0)
    assert _public_parameter_values_equal(
        {"a": [True, 1, 1.0], "b": None},
        {"b": None, "a": [True, 1, 1.0]},
    )


def test_declaration_accepts_required_default_only() -> None:
    declaration = {"default": "plain"}
    assert _validate_public_parameter_declaration(declaration) is declaration


def test_declaration_enum_uses_type_sensitive_equality() -> None:
    declaration = {"default": 1, "enum": [True, 1, 1.0]}
    assert _validate_public_parameter_declaration(declaration) is declaration


def test_declaration_rejects_duplicate_enum_values() -> None:
    with pytest.raises(ValueError):
        _validate_public_parameter_declaration({"default": 1, "enum": [1, 1]})


@pytest.mark.parametrize("field", ["type", "enum", "description"])
def test_declaration_rejects_none_for_present_optional_non_null_field(field: str) -> None:
    with pytest.raises(ValueError):
        _validate_public_parameter_declaration({"default": 1, field: None})


@pytest.mark.parametrize(
    "declaration",
    [
        {"default": True, "type": "integer"},
        {"default": True, "type": "number"},
        {"default": 1, "type": "string"},
    ],
)
def test_declaration_rejects_type_mismatch_and_bool_as_numeric(declaration: dict) -> None:
    with pytest.raises(ValueError):
        _validate_public_parameter_declaration(declaration)


def test_declaration_accepts_numeric_bounds_and_checks_default() -> None:
    declaration = {"default": 3, "type": "integer", "minimum": 1, "maximum": 5}
    assert _validate_public_parameter_declaration(declaration) is declaration


@pytest.mark.parametrize(
    "declaration",
    [
        {"default": 0, "type": "integer", "minimum": 1},
        {"default": 6.0, "type": "number", "maximum": 5},
        {"default": 1, "type": "integer", "minimum": 2, "maximum": 1},
        {"default": 1, "minimum": 0},
        {"default": 1, "type": "integer", "minimum": True},
        {"default": 1.0, "type": "number", "maximum": math.inf},
    ],
)
def test_declaration_rejects_invalid_numeric_constraints(declaration: dict) -> None:
    with pytest.raises(ValueError):
        _validate_public_parameter_declaration(declaration)


def test_declaration_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        _validate_public_parameter_declaration({"default": 1, "unexpected": 2})


def test_resolution_fills_defaults_and_applies_overrides() -> None:
    declarations = {
        "epochs": {"default": 10, "type": "integer", "minimum": 1},
        "label": {"default": "base", "type": "string"},
    }
    assert _resolve_public_parameters(declarations, {"epochs": 20}) == {
        "epochs": 20,
        "label": "base",
    }


def test_resolution_rejects_unknown_override_key() -> None:
    declarations = {"epochs": {"default": 10, "type": "integer"}}
    with pytest.raises(ValueError):
        _resolve_public_parameters(declarations, {"unknown": 1})


def test_resolution_does_not_coerce_override_values() -> None:
    declarations = {"epochs": {"default": 10, "type": "integer"}}
    with pytest.raises(ValueError):
        _resolve_public_parameters(declarations, {"epochs": "20"})


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_training_seed_requires_exact_integer(value: object) -> None:
    with pytest.raises(ValueError):
        _validate_training_seed(value)


def test_training_seed_accepts_integer() -> None:
    assert _validate_training_seed(0) == 0
