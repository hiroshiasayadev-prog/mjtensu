"""Shared MLDB v2 public-parameter runtime values and internal validation."""

from __future__ import annotations

import math as _math
from typing import Literal, Mapping, NotRequired, TypeAlias, TypedDict

PublicParameterValue: TypeAlias = (
    None
    | bool
    | str
    | int
    | float
    | list["PublicParameterValue"]
    | dict[str, "PublicParameterValue"]
)

PublicParameterType: TypeAlias = Literal[
    "boolean",
    "integer",
    "number",
    "string",
    "array",
    "object",
    "null",
]


class PublicParameterDeclaration(TypedDict):
    default: PublicParameterValue
    type: NotRequired[PublicParameterType]
    enum: NotRequired[list[PublicParameterValue]]
    minimum: NotRequired[int | float]
    maximum: NotRequired[int | float]
    description: NotRequired[str]


PublicParameterDeclarations: TypeAlias = Mapping[str, PublicParameterDeclaration]
ResolvedPublicParameters: TypeAlias = Mapping[str, PublicParameterValue]

_ALLOWED_DECLARATION_FIELDS = {
    "default",
    "type",
    "enum",
    "minimum",
    "maximum",
    "description",
}
_ALLOWED_PARAMETER_TYPES = {
    "boolean",
    "integer",
    "number",
    "string",
    "array",
    "object",
    "null",
}


def _validate_public_parameter_value(value: object) -> PublicParameterValue:
    value_type = type(value)
    if value is None or value_type in {bool, str, int}:
        return value  # type: ignore[return-value]
    if value_type is float:
        if not _math.isfinite(value):
            raise ValueError("public parameter floats must be finite")
        return value
    if value_type is list:
        for item in value:
            _validate_public_parameter_value(item)
        return value
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("public parameter mapping keys must be strings")
            _validate_public_parameter_value(item)
        return value
    raise ValueError("value is not a public parameter value")


def _public_parameter_values_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(
            _public_parameter_values_equal(a, b) for a, b in zip(left, right)
        )
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _public_parameter_values_equal(left[key], right[key]) for key in left
        )
    return left == right


def _matches_parameter_type(value: object, declared_type: str) -> bool:
    value_type = type(value)
    if declared_type == "boolean":
        return value_type is bool
    if declared_type == "integer":
        return value_type is int
    if declared_type == "number":
        return value_type in {int, float}
    if declared_type == "string":
        return value_type is str
    if declared_type == "array":
        return value_type is list
    if declared_type == "object":
        return value_type is dict
    if declared_type == "null":
        return value is None
    return False


def _validate_finite_bound(value: object, *, field: str) -> int | float:
    if type(value) not in {int, float}:
        raise ValueError(f"{field} must be numeric")
    if type(value) is float and not _math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _validate_value_against_declaration(
    value: object, declaration: PublicParameterDeclaration
) -> PublicParameterValue:
    validated = _validate_public_parameter_value(value)
    declared_type = declaration.get("type")
    if declared_type is not None and not _matches_parameter_type(validated, declared_type):
        raise ValueError("public parameter value does not match declared type")

    enum_values = declaration.get("enum")
    if enum_values is not None and not any(
        _public_parameter_values_equal(validated, candidate) for candidate in enum_values
    ):
        raise ValueError("public parameter value is not in enum")

    minimum = declaration.get("minimum")
    if minimum is not None and validated < minimum:  # type: ignore[operator]
        raise ValueError("public parameter value is below minimum")

    maximum = declaration.get("maximum")
    if maximum is not None and validated > maximum:  # type: ignore[operator]
        raise ValueError("public parameter value is above maximum")

    return validated


def _validate_public_parameter_declaration(value: object) -> PublicParameterDeclaration:
    if type(value) is not dict:
        raise ValueError("public parameter declaration must be a mapping")
    unknown_fields = set(value) - _ALLOWED_DECLARATION_FIELDS
    if unknown_fields:
        raise ValueError("unknown public parameter declaration field")
    if "default" not in value:
        raise ValueError("public parameter declaration requires default")

    declared_type = value.get("type")
    if "type" in value:
        if type(declared_type) is not str or declared_type not in _ALLOWED_PARAMETER_TYPES:
            raise ValueError("invalid public parameter type")

    description = value.get("description")
    if "description" in value and type(description) is not str:
        raise ValueError("description must be a string")

    enum_values = value.get("enum")
    if "enum" in value:
        if type(enum_values) is not list or not enum_values:
            raise ValueError("enum must be a non-empty list")
        for candidate in enum_values:
            _validate_public_parameter_value(candidate)
        for index, candidate in enumerate(enum_values):
            if any(
                _public_parameter_values_equal(candidate, prior)
                for prior in enum_values[:index]
            ):
                raise ValueError("enum contains duplicate values")

    minimum = value.get("minimum")
    maximum = value.get("maximum")
    if "minimum" in value or "maximum" in value:
        if declared_type not in {"integer", "number"}:
            raise ValueError("minimum/maximum require integer or number type")
    if "minimum" in value:
        minimum = _validate_finite_bound(minimum, field="minimum")
    if "maximum" in value:
        maximum = _validate_finite_bound(maximum, field="maximum")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum must not exceed maximum")

    declaration = value
    _validate_value_against_declaration(declaration["default"], declaration)
    return declaration


def _resolve_public_parameters(
    declarations: PublicParameterDeclarations,
    overrides: Mapping[str, object],
) -> dict[str, PublicParameterValue]:
    if not isinstance(declarations, Mapping) or not isinstance(overrides, Mapping):
        raise ValueError("declarations and overrides must be mappings")

    validated_declarations: dict[str, PublicParameterDeclaration] = {}
    for key, declaration in declarations.items():
        if type(key) is not str:
            raise ValueError("public parameter keys must be strings")
        validated_declarations[key] = _validate_public_parameter_declaration(declaration)

    unknown_keys = set(overrides) - set(validated_declarations)
    if unknown_keys:
        raise ValueError("unknown public parameter override key")

    resolved: dict[str, PublicParameterValue] = {}
    for key, declaration in validated_declarations.items():
        candidate = overrides[key] if key in overrides else declaration["default"]
        resolved[key] = _validate_value_against_declaration(candidate, declaration)
    return resolved


def _validate_training_seed(value: object) -> int:
    if type(value) is not int:
        raise ValueError("training seed must be an integer")
    return value
