"""Task-domain implementation for MLDB Wave I1-A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias

from ..common.errors import ValidationIssue, ValidationReport
from ..common.ids import TaskId


TaskProblemType: TypeAlias = str
"""Task-local prediction-problem identifier."""

TaskSemantics: TypeAlias = Mapping[str, object]
"""Task-specific semantic meaning keyed by Task-local semantic names."""


@dataclass(frozen=True, slots=True)
class TaskInput:
    """Semantic input contract for one Task."""

    semantic_unit: str


@dataclass(frozen=True, slots=True)
class CategoricalTarget:
    """Categorical prediction target with normative ordered labels."""

    type: Literal["categorical"]
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskScope:
    """Explicit semantic inclusion and exclusion boundary for one Task."""

    includes: tuple[str, ...]
    excludes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Task:
    """One MLDB Task semantic prediction contract."""

    schema: Literal["mjtensu.mldb/task/v1"]
    id: TaskId
    name: str
    problem_type: TaskProblemType
    description: str
    input: TaskInput
    target: CategoricalTarget
    semantics: TaskSemantics
    scope: TaskScope


def validate_task(task: Task) -> ValidationReport:
    """Statically validate invariants owned by the Task contract."""
    issues: list[ValidationIssue] = []

    if task.schema != "mjtensu.mldb/task/v1":
        issues.append(
            ValidationIssue(
                code="task.schema.unsupported",
                message="Task schema must be 'mjtensu.mldb/task/v1'.",
                path="schema",
            )
        )

    if task.target.type != "categorical":
        issues.append(
            ValidationIssue(
                code="task.target.type.unsupported",
                message="Task v1 target type must be 'categorical'.",
                path="target.type",
            )
        )

    labels = task.target.labels
    if not labels:
        issues.append(
            ValidationIssue(
                code="task.target.labels.empty",
                message="Categorical target labels must not be empty.",
                path="target.labels",
            )
        )
    else:
        seen: set[str] = set()
        for index, label in enumerate(labels):
            path = f"target.labels[{index}]"
            if type(label) is not str or label == "":
                issues.append(
                    ValidationIssue(
                        code="task.target.label.invalid",
                        message="Categorical labels must be non-empty strings.",
                        path=path,
                    )
                )
                continue
            if label in seen:
                issues.append(
                    ValidationIssue(
                        code="task.target.label.duplicate",
                        message=f"Categorical label {label!r} is duplicated.",
                        path=path,
                    )
                )
            else:
                seen.add(label)

    return ValidationReport(tuple(issues))
