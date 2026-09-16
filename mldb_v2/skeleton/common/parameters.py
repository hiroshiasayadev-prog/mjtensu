"""Shared MLDB v2 public-parameter type boundaries."""

from __future__ import annotations

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
