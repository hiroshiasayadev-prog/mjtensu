"""MLDB v2 Namespace public shape and private validation."""

from typing import Literal, TypedDict, cast

from mldb_v2.src.common.ids import EntityKind, NamespaceId, _validate_namespace_id
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver

from ._core_definition_validation import (
    _require_exact_mapping,
    _require_required_fields,
    _require_string,
)


class Namespace(TypedDict):
    schema: Literal["mjtensu.mldb-v2/namespace/v1"]
    id: NamespaceId
    name: str
    description: str


_SCHEMA = "mjtensu.mldb-v2/namespace/v1"
_REQUIRED_FIELDS = {"schema", "id", "name", "description"}
# Frozen backend records define these exact fields as backend-owned operational identity.
_BACKEND_NAMESPACE_FIELDS = frozenset({"backend", "execution_id", "execution_ids"})


def _validate_namespace(value: object, *, expected_id: str | None = None) -> Namespace:
    document = _require_exact_mapping(value, label="Namespace")
    _require_required_fields(document, _REQUIRED_FIELDS, label="Namespace")
    if document["schema"] != _SCHEMA:
        raise ValueError("unsupported Namespace schema")
    namespace_id = str(_validate_namespace_id(document["id"]))
    if expected_id is not None and namespace_id != expected_id:
        raise ValueError("Namespace id does not match canonical directory identity")
    _require_string(document["name"], label="Namespace name", non_empty=True)
    _require_string(document["description"], label="Namespace description")
    if set(document) & _BACKEND_NAMESPACE_FIELDS:
        raise ValueError("backend metadata is forbidden in Namespace")
    return cast(Namespace, document)


def _load_namespace(
    resolver: CanonicalRepositoryResolver,
    namespace_id: NamespaceId,
) -> Namespace:
    expected_id = str(_validate_namespace_id(namespace_id))
    document = resolver.resolve(kind=EntityKind.NAMESPACE, entity_id=namespace_id)
    return _validate_namespace(document, expected_id=expected_id)
