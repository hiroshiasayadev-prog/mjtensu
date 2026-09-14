"""MLDB v2 Task public shapes and private validation."""

from typing import Literal, TypeAlias, TypedDict, cast

from mldb_v2.src.common.ids import EntityKind, TaskId
from mldb_v2.src.common.parameters import PublicParameterValue
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver

from ._core_definition_validation import (
    _require_exact_fields,
    _require_exact_mapping,
    _require_string,
    _validate_json_mapping,
    _validate_lifecycle,
    _validate_versioned_entity_id,
)

TaskInput: TypeAlias = dict[str, PublicParameterValue]
TaskTarget: TypeAlias = dict[str, PublicParameterValue]
TaskSemantics: TypeAlias = dict[str, PublicParameterValue]
TaskScope: TypeAlias = dict[str, PublicParameterValue]


class Task(TypedDict):
    schema: Literal["mjtensu.mldb-v2/task/v1"]
    id: TaskId
    status: Literal["draft", "sealed"]
    name: str
    problem_type: str
    description: str
    input: TaskInput
    target: TaskTarget
    semantics: TaskSemantics
    scope: TaskScope


_SCHEMA = "mjtensu.mldb-v2/task/v1"
_FIELDS = {
    "schema",
    "id",
    "status",
    "name",
    "problem_type",
    "description",
    "input",
    "target",
    "semantics",
    "scope",
}


def _validate_categorical_target(target: dict[str, object]) -> None:
    labels = target.get("labels")
    if type(labels) is not list or not labels:
        raise ValueError("categorical target requires non-empty labels")
    seen: set[str] = set()
    for label in labels:
        if type(label) is not str or not label:
            raise ValueError("categorical labels must be non-empty strings")
        if label in seen:
            raise ValueError("categorical labels must be unique")
        seen.add(label)


def _validate_task(value: object, *, expected_id: str | None = None) -> Task:
    document = _require_exact_mapping(value, label="Task")
    _require_exact_fields(document, _FIELDS, label="Task")
    if document["schema"] != _SCHEMA:
        raise ValueError("unsupported Task schema")
    _validate_versioned_entity_id(document["id"], expected_id=expected_id)
    _validate_lifecycle(document["status"])
    _require_string(document["name"], label="Task name", non_empty=True)
    _require_string(document["problem_type"], label="Task problem_type", non_empty=True)
    _require_string(document["description"], label="Task description")

    _validate_json_mapping(document["input"], label="Task input")
    target = _validate_json_mapping(document["target"], label="Task target")
    _validate_json_mapping(document["semantics"], label="Task semantics")
    _validate_json_mapping(document["scope"], label="Task scope")

    target_type = target.get("type")
    _require_string(target_type, label="Task target.type", non_empty=True)
    if target_type == "categorical":
        _validate_categorical_target(target)
    return cast(Task, document)


def _load_task(resolver: CanonicalRepositoryResolver, task_id: TaskId) -> Task:
    expected_id = _validate_versioned_entity_id(task_id)
    document = resolver.resolve(kind=EntityKind.TASK, entity_id=task_id)
    return _validate_task(document, expected_id=expected_id)
