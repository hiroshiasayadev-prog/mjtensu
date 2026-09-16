"""MLDB v2 Architecture public shapes and private definition parsing."""

from pathlib import Path
from typing import Literal, NotRequired, TypeAlias, TypedDict

from mldb_v2.src.common.ids import ArchitectureId, EntityKind, TaskId
from mldb_v2.src.common.parameters import PublicParameterValue
from mldb_v2.src.verification.executable_integrity import ExecutableSource

from ._executable_definition_loading import (
    _require_exact_keys,
    _require_string,
    _resolve_document,
    _validate_definition_id,
    _validate_implementation,
    _validate_json_mapping,
    _validate_status,
    _validate_task_reference,
)

ArchitectureInterfaceEndpoint: TypeAlias = dict[str, PublicParameterValue]
ArchitectureParameters: TypeAlias = dict[str, PublicParameterValue]


class ArchitectureImplementation(TypedDict):
    framework: Literal["pytorch"]
    entrypoint: Literal["build"]
    sha256: NotRequired[str]
    sources: NotRequired[list[ExecutableSource]]


class ArchitectureInterface(TypedDict):
    input: ArchitectureInterfaceEndpoint
    output: ArchitectureInterfaceEndpoint


class ArchitectureStructure(TypedDict):
    summary: str
    traits: NotRequired[list[str]]


class Architecture(TypedDict):
    schema: Literal["mjtensu.mldb-v2/architecture/v1"]
    id: ArchitectureId
    status: Literal["draft", "sealed"]
    task: TaskId
    name: str
    family: str
    description: str
    implementation: ArchitectureImplementation
    interface: ArchitectureInterface
    structure: ArchitectureStructure
    parameters: NotRequired[ArchitectureParameters]


_REQUIRED_TOP_LEVEL = {
    "schema",
    "id",
    "status",
    "task",
    "name",
    "family",
    "description",
    "implementation",
    "interface",
    "structure",
}
_OPTIONAL_TOP_LEVEL = {"parameters"}


def _parse_architecture_document(document: object, *, expected_id: str) -> Architecture:
    mapping = _require_exact_keys(
        document,
        required=_REQUIRED_TOP_LEVEL,
        optional=_OPTIONAL_TOP_LEVEL,
        label="Architecture",
    )
    if mapping["schema"] != "mjtensu.mldb-v2/architecture/v1":
        raise ValueError("unsupported Architecture schema")
    definition_id = _validate_definition_id(mapping["id"])
    if definition_id != expected_id:
        raise ValueError("Architecture id does not match canonical path identity")
    status = _validate_status(mapping["status"])
    task = _validate_task_reference(mapping["task"])
    name = _require_string(mapping["name"], label="Architecture name", nonempty=True)
    family = _require_string(mapping["family"], label="Architecture family", nonempty=True)
    description = _require_string(mapping["description"], label="Architecture description")
    implementation = _validate_implementation(
        mapping["implementation"],
        entrypoint="build",
        framework="pytorch",
        sealed=status == "sealed",
        namespace=definition_id.split("/", 1)[0],
    )

    interface = _require_exact_keys(
        mapping["interface"],
        required={"input", "output"},
        label="Architecture interface",
    )
    endpoints: dict[str, ArchitectureInterfaceEndpoint] = {}
    for endpoint_name in ("input", "output"):
        endpoint = _validate_json_mapping(
            interface[endpoint_name], label=f"Architecture interface.{endpoint_name}"
        )
        _require_string(
            endpoint.get("kind"),
            label=f"Architecture interface.{endpoint_name}.kind",
            nonempty=True,
        )
        endpoints[endpoint_name] = endpoint  # type: ignore[assignment]

    structure = _require_exact_keys(
        mapping["structure"],
        required={"summary"},
        optional={"traits"},
        label="Architecture structure",
    )
    summary = _require_string(
        structure["summary"], label="Architecture structure.summary", nonempty=True
    )
    parsed_structure: ArchitectureStructure = {"summary": summary}
    if "traits" in structure:
        traits = structure["traits"]
        if type(traits) is not list:
            raise ValueError("Architecture structure.traits must be a list")
        validated_traits: list[str] = []
        for trait in traits:
            validated_traits.append(
                _require_string(trait, label="Architecture trait", nonempty=True)
            )
        if len(set(validated_traits)) != len(validated_traits):
            raise ValueError("Architecture traits must be unique")
        parsed_structure["traits"] = validated_traits

    parsed: Architecture = {
        "schema": "mjtensu.mldb-v2/architecture/v1",
        "id": ArchitectureId(definition_id),
        "status": status,  # type: ignore[typeddict-item]
        "task": TaskId(task),
        "name": name,
        "family": family,
        "description": description,
        "implementation": implementation,  # type: ignore[typeddict-item]
        "interface": {"input": endpoints["input"], "output": endpoints["output"]},
        "structure": parsed_structure,
    }
    if "parameters" in mapping:
        parsed["parameters"] = _validate_json_mapping(
            mapping["parameters"], label="Architecture parameters"
        )  # type: ignore[typeddict-item]
    return parsed


def _load_architecture_definition(
    mldb_data_root: str | Path, architecture_id: ArchitectureId | str
) -> Architecture:
    expected_id = _validate_definition_id(architecture_id)
    document, _path = _resolve_document(
        mldb_data_root,
        kind=EntityKind.ARCHITECTURE,
        entity_id=expected_id,
    )
    return _parse_architecture_document(document, expected_id=expected_id)
