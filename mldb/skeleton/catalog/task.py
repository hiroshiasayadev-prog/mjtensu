"""Public Task-domain signatures for MLDB Wave 1-A.

Owns:
- one Task's semantic prediction contract;
- semantic input-unit representation;
- categorical target vocabulary and its normative class-index ordering;
- Task-local semantic meaning, including Task-specific normalization decisions;
- explicit in-scope and out-of-scope meaning;
- static validation of Task-local invariants.

Does not own:
- YAML/filesystem loading or canonical repository placement;
- ID allocation, filename-basename checks, or reference-history immutability checks;
- Corpus paths, splits, materialized image representation, or sample validation;
- tensor shape, channels, normalization, preprocessing, or augmentation;
- Architecture execution or model topology;
- training/evaluation parameters or behavior;
- runtime asset resolution, cross-asset compatibility, deployment, HTTP/CLI, or persistence.

This module is a signature skeleton only. Public bodies intentionally remain unimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias

from ..common.errors import ValidationReport
from ..common.ids import TaskId


TaskProblemType: TypeAlias = str
"""Task-local prediction-problem identifier.

MLDB Task v1 deliberately defines no global ``problem_type`` enum. Concrete values
are authored by Task records and constrained only by contracts that explicitly define
them.
"""


TaskSemantics: TypeAlias = Mapping[str, object]
"""Task-specific semantic meaning keyed by Task-local semantic names.

This is intentionally an open mapping because Task v1 defines no global semantic-key
or semantic-value schema. Decisions such as red-five to base-five normalization and
the meaning of an ``invalid`` outcome belong here when the Task defines them. A
separate generic normalization-rule language is not part of the current contract.
"""


@dataclass(frozen=True, slots=True)
class TaskInput:
    """Semantic input contract for one Task.

    ``semantic_unit`` names what one prediction input means, for example one visible
    tile image. It does not define tensor dimensions, channels, pixel encoding,
    preprocessing, or any other material representation detail.
    """

    semantic_unit: str


@dataclass(frozen=True, slots=True)
class CategoricalTarget:
    """Categorical prediction target with normative ordered labels.

    ``labels`` order is part of the Task ABI. The zero-based tuple position of a label
    is its canonical class index for this Task revision. No independent public
    label-to-index mapping is stored because Task v1 defines that mapping by order.
    """

    type: Literal["categorical"]
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskScope:
    """Explicit semantic inclusion and exclusion boundary for one Task."""

    includes: tuple[str, ...]
    excludes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Task:
    """One MLDB Task semantic prediction contract.

    The current Task v1 target structure specified by Design Records is categorical.
    Future non-categorical target structures require an explicit contract addition;
    this skeleton does not invent a generic target schema in advance.

    ``semantics`` stores Task-owned target meaning without standardizing Task-local
    semantic keys. It must not be used for Corpus representation, training policy,
    model-interface, evaluation, or deployment metadata.
    """

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
    """Statically validate invariants owned by the Task contract.

    This validation scope covers only rules decidable from the in-memory Task value,
    including the supported Task schema and categorical-label invariants such as
    non-empty unique labels while preserving their normative order.

    It does not perform filesystem/YAML loading, filename-to-ID agreement checks,
    historical immutability checks, repository lookup, downstream reference
    resolution, Corpus sample validation, or runtime compatibility checks between a
    Task and Corpus, Architecture, Protocol, Model, or Run.

    ID grammar is not redefined here. Any repository identity agreement or future
    shared ID grammar remains owned by the corresponding repository/entity contract.
    """

    ...
