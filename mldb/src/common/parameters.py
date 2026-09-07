"""Shared MLDB public-parameter signatures and resolution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, TypeAlias, TypeGuard


PublicParameterValue: TypeAlias = (
    None
    | bool
    | str
    | int
    | float
    | list["PublicParameterValue"]
    | dict[str, "PublicParameterValue"]
)
"""One MLDB v1 public parameter value."""


@dataclass(frozen=True, slots=True)
class PublicParameterDeclaration:
    """Normalized common portion of one public parameter declaration."""

    default: PublicParameterValue


PublicParameterDeclarations: TypeAlias = Mapping[str, PublicParameterDeclaration]
"""Published parameter declarations keyed by protocol-local public name."""

PublicParameterOverrides: TypeAlias = Mapping[str, PublicParameterValue]
"""Caller-supplied values for a subset of published parameter names."""

ResolvedPublicParameters: TypeAlias = Mapping[str, PublicParameterValue]
"""Complete resolved mapping containing every published parameter exactly once."""


def is_public_parameter_value(value: object) -> TypeGuard[PublicParameterValue]:
    """Return whether ``value`` belongs to the MLDB public-parameter domain."""
    return _is_public_parameter_value(value, set())


def resolve_public_parameters(
    declarations: PublicParameterDeclarations,
    overrides: PublicParameterOverrides,
) -> ResolvedPublicParameters:
    """Resolve protocol defaults plus caller overrides into one complete mapping."""
    for key in overrides:
        if key not in declarations:
            raise ValueError(f"unknown public parameter: {key!r}")

    resolved: dict[str, PublicParameterValue] = {}
    for key, declaration in declarations.items():
        default = declaration.default
        if not is_public_parameter_value(default):
            raise ValueError(f"invalid public parameter default for {key!r}")

        value = overrides[key] if key in overrides else default
        if not is_public_parameter_value(value):
            raise ValueError(f"invalid public parameter value for {key!r}")
        resolved[key] = _clone_public_parameter_value(value)

    return resolved


def _is_public_parameter_value(value: object, active_container_ids: set[int]) -> bool:
    value_type = type(value)
    if value is None or value_type in {bool, str, int}:
        return True
    if value_type is float:
        return math.isfinite(value)
    if value_type not in {list, dict}:
        return False

    container_id = id(value)
    if container_id in active_container_ids:
        return False
    active_container_ids.add(container_id)
    try:
        if value_type is list:
            return all(
                _is_public_parameter_value(item, active_container_ids)
                for item in value
            )
        return all(
            type(key) is str
            and _is_public_parameter_value(item, active_container_ids)
            for key, item in value.items()
        )
    finally:
        active_container_ids.remove(container_id)


def _clone_public_parameter_value(value: PublicParameterValue) -> PublicParameterValue:
    value_type = type(value)
    if value_type is list:
        return [_clone_public_parameter_value(item) for item in value]
    if value_type is dict:
        return {
            key: _clone_public_parameter_value(item)
            for key, item in value.items()
        }
    return value
