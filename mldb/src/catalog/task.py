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
class RotatedObjectDetectionGeometry:
    """Rotated rectangle geometry contract for detection targets."""

    format: Literal["cx-cy-w-h-angle-deg"]
    angle_period_deg: int


@dataclass(frozen=True, slots=True)
class RotatedObjectDetectionTarget:
    """Object labels plus rotated-rectangle geometry semantics."""

    type: Literal["rotated-object-detection"]
    labels: tuple[str, ...]
    geometry: RotatedObjectDetectionGeometry


TaskTarget: TypeAlias = CategoricalTarget | RotatedObjectDetectionTarget | Mapping[str, object]


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
    target: TaskTarget
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

    if isinstance(task.target, CategoricalTarget):
        _validate_categorical_labels(task.target.labels, issues)
    elif isinstance(task.target, RotatedObjectDetectionTarget):
        _validate_rotated_detection_target(task.target, issues)
    elif isinstance(task.target, Mapping):
        target_type = task.target.get("type")
        if target_type == "categorical":
            labels = task.target.get("labels")
            if isinstance(labels, (list, tuple)):
                _validate_categorical_labels(tuple(labels), issues)
            else:
                issues.append(
                    ValidationIssue(
                        code="task.target.labels.required",
                        message="Categorical target requires target.labels.",
                        path="target.labels",
                    )
                )
        elif target_type == "rotated-object-detection":
            labels = task.target.get("labels")
            geometry = task.target.get("geometry")
            if isinstance(labels, (list, tuple)):
                _validate_categorical_labels(tuple(labels), issues)
            else:
                issues.append(ValidationIssue(code="task.target.labels.required", message="Rotated detection target requires target.labels.", path="target.labels"))
            if not isinstance(geometry, Mapping):
                issues.append(ValidationIssue(code="task.target.geometry.required", message="Rotated detection target requires target.geometry.", path="target.geometry"))
    return ValidationReport(tuple(issues))


def _validate_categorical_labels(
    labels: tuple[object, ...],
    issues: list[ValidationIssue],
) -> None:
    if not labels:
        issues.append(
            ValidationIssue(
                code="task.target.labels.empty",
                message="Categorical target labels must not be empty.",
                path="target.labels",
            )
        )
        return

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


def _validate_rotated_detection_target(
    target: RotatedObjectDetectionTarget,
    issues: list[ValidationIssue],
) -> None:
    _validate_categorical_labels(target.labels, issues)
    if target.geometry.format != "cx-cy-w-h-angle-deg":
        issues.append(ValidationIssue(code="task.target.geometry.format.invalid", message="Rotated detection geometry format must be cx-cy-w-h-angle-deg.", path="target.geometry.format"))
    if type(target.geometry.angle_period_deg) is not int or target.geometry.angle_period_deg <= 0:
        issues.append(ValidationIssue(code="task.target.geometry.angle_period.invalid", message="Rotated detection angle period must be a positive integer.", path="target.geometry.angle_period_deg"))
