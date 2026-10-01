"""Immutable deployable runtime-model catalog entries."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, NotRequired, TypedDict

from mldb_v2.src.catalog._core_definition_validation import (
    _require_string,
    _validate_versioned_entity_id,
)
from mldb_v2.src.common.ids import EntityKind, RuntimeModelId
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.storage.artifact_reference import ArtifactRef, _validate_artifact_ref


RuntimeModelRole = Literal["detector", "tile-classifier", "red-five-classifier"]


class RuntimeModel(TypedDict):
    schema: Literal["mjtensu.mldb-v2/runtime-model/v1"]
    id: RuntimeModelId
    name: str
    description: str
    role: RuntimeModelRole
    format: Literal["onnx"]
    runtime_spec: str
    artifact: ArtifactRef
    provenance: NotRequired[Mapping[str, object]]


_SCHEMA = "mjtensu.mldb-v2/runtime-model/v1"
_REQUIRED_FIELDS = {
    "schema", "id", "name", "description", "role", "format", "runtime_spec", "artifact",
}
_OPTIONAL_FIELDS = {"provenance"}
_ROLES = {"detector", "tile-classifier", "red-five-classifier"}


def _validate_runtime_model(value: object, *, expected_id: str | None = None) -> RuntimeModel:
    if type(value) is not dict:
        raise ValueError("RuntimeModel must be a mapping")
    keys = set(value)
    if not _REQUIRED_FIELDS <= keys or not keys <= _REQUIRED_FIELDS | _OPTIONAL_FIELDS:
        raise ValueError("RuntimeModel fields do not match schema")
    if value["schema"] != _SCHEMA:
        raise ValueError("unsupported RuntimeModel schema")
    model_id = _validate_versioned_entity_id(value["id"], expected_id=expected_id)
    name = _require_string(value["name"], label="RuntimeModel name", non_empty=True)
    description = _require_string(value["description"], label="RuntimeModel description")
    role = value["role"]
    if type(role) is not str or role not in _ROLES:
        raise ValueError("RuntimeModel role is invalid")
    if value["format"] != "onnx":
        raise ValueError("RuntimeModel format must be onnx")
    runtime_spec = _require_string(value["runtime_spec"], label="RuntimeModel runtime_spec", non_empty=True)
    artifact = _validate_artifact_ref(value["artifact"])
    result: RuntimeModel = {
        "schema": _SCHEMA,
        "id": RuntimeModelId(model_id),
        "name": name,
        "description": description,
        "role": role,  # type: ignore[typeddict-item]
        "format": "onnx",
        "runtime_spec": runtime_spec,
        "artifact": artifact,
    }
    if "provenance" in value:
        provenance = value["provenance"]
        if type(provenance) is not dict:
            raise ValueError("RuntimeModel provenance must be a mapping")
        result["provenance"] = provenance
    return result


def _load_runtime_model(
    mldb_data_root: str | Path,
    runtime_model_id: RuntimeModelId | str,
) -> RuntimeModel:
    expected_id = _validate_versioned_entity_id(runtime_model_id)
    resolver = CanonicalRepositoryResolver(mldb_data_root)
    document = resolver.resolve(kind=EntityKind.RUNTIME_MODEL, entity_id=RuntimeModelId(expected_id))
    return _validate_runtime_model(document, expected_id=expected_id)
