"""MLDB v2 Task public shapes."""

from typing import Literal, TypeAlias, TypedDict

from mldb_v2.skeleton.common.ids import TaskId
from mldb_v2.skeleton.common.parameters import PublicParameterValue

TaskInput: TypeAlias = dict[str, PublicParameterValue]
# ``type`` is required and non-empty by the Task format; all other keys remain Task-specific.
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
