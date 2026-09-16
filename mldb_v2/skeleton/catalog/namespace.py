"""MLDB v2 Namespace public shape."""

from typing import Literal, TypedDict

from mldb_v2.skeleton.common.ids import NamespaceId


class Namespace(TypedDict):
    schema: Literal["mjtensu.mldb-v2/namespace/v1"]
    id: NamespaceId
    name: str
    description: str
