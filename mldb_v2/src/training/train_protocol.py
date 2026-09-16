"""MLDB v2 reusable Train Protocol definition shape and private parsing."""

from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from mldb_v2.src.common.ids import EntityKind, TaskId, TrainProtocolId
from mldb_v2.src.common.parameters import PublicParameterDeclarations
from mldb_v2.src.verification.executable_integrity import ExecutableSource
from mldb_v2.src.catalog._executable_definition_loading import (
    _require_exact_keys,
    _require_string,
    _resolve_document,
    _validate_definition_id,
    _validate_implementation,
    _validate_parameter_declarations,
    _validate_status,
    _validate_task_reference,
)


class TrainProtocolImplementation(TypedDict):
    entrypoint: Literal["train"]
    sha256: NotRequired[str]
    sources: NotRequired[list[ExecutableSource]]


class TrainProtocol(TypedDict):
    schema: Literal["mjtensu.mldb-v2/train-protocol/v1"]
    id: TrainProtocolId
    status: Literal["draft", "sealed"]
    task: TaskId
    name: str
    description: str
    implementation: TrainProtocolImplementation
    parameters: PublicParameterDeclarations


_REQUIRED_TOP_LEVEL = {
    "schema", "id", "status", "task", "name", "description", "implementation", "parameters"
}


def _parse_train_protocol_document(document: object, *, expected_id: str) -> TrainProtocol:
    mapping = _require_exact_keys(document, required=_REQUIRED_TOP_LEVEL, label="Train Protocol")
    if mapping["schema"] != "mjtensu.mldb-v2/train-protocol/v1":
        raise ValueError("unsupported Train Protocol schema")
    definition_id = _validate_definition_id(mapping["id"])
    if definition_id != expected_id:
        raise ValueError("Train Protocol id does not match canonical path identity")
    status = _validate_status(mapping["status"])
    task = _validate_task_reference(mapping["task"])
    name = _require_string(mapping["name"], label="Train Protocol name", nonempty=True)
    description = _require_string(mapping["description"], label="Train Protocol description")
    implementation = _validate_implementation(
        mapping["implementation"], entrypoint="train", sealed=status == "sealed",
        namespace=definition_id.split("/", 1)[0],
    )
    parameters = _validate_parameter_declarations(mapping["parameters"])
    return {
        "schema": "mjtensu.mldb-v2/train-protocol/v1",
        "id": TrainProtocolId(definition_id),
        "status": status,  # type: ignore[typeddict-item]
        "task": TaskId(task),
        "name": name,
        "description": description,
        "implementation": implementation,  # type: ignore[typeddict-item]
        "parameters": parameters,
    }


def _load_train_protocol_definition(
    mldb_data_root: str | Path, protocol_id: TrainProtocolId | str
) -> TrainProtocol:
    expected_id = _validate_definition_id(protocol_id)
    document, _path = _resolve_document(
        mldb_data_root,
        kind=EntityKind.TRAIN_PROTOCOL,
        entity_id=expected_id,
    )
    return _parse_train_protocol_document(document, expected_id=expected_id)
