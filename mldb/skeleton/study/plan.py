"""Public Python signatures for materialized MLDB Study Run plans.

This skeleton fixes the normalized in-memory representation of the immutable
``plan.jsonl`` execution intent produced for one Study Run. Each row describes one
Study-local trial with exactly one Model source and the fully resolved evaluation
stages intended for that trial Model.

It intentionally does not define authored Study metadata, grid expansion, public-
parameter default resolution, JSONL serialization, plan hashing/integrity metadata,
Study Run lifecycle, child Run allocation, or queue/worker behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from ..common.errors import ValidationReport
from ..common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    ModelId,
    TrainProtocolId,
)
from ..common.parameters import PublicParameterValue, ResolvedPublicParameters


@dataclass(frozen=True, slots=True)
class StudyPlanTraining:
    """Fully resolved training coordinate for one training-derived trial.

    ``architecture``, ``corpus``, and ``protocol`` are the concrete reusable inputs
    selected during Study materialization. ``seed`` is the concrete training seed and
    remains separate from public parameters; it must be an integer and must not be a
    boolean.

    ``parameters`` is the complete resolved Train Protocol public-parameter mapping:
    it contains protocol defaults plus the concrete Study grid coordinate, not only
    values explicitly authored by the Study.

    No Training Run ID or future Model ID is part of this value because neither child
    execution identity nor its learned Model exists as plan input.
    """

    architecture: ArchitectureId
    corpus: CorpusId
    protocol: TrainProtocolId
    seed: int
    parameters: ResolvedPublicParameters


@dataclass(frozen=True, slots=True)
class StudyPlanEvaluation:
    """Fully resolved evaluation-stage intent for one trial Model.

    ``stage`` is a Study-local identifier rather than an MLDB entity identity.
    ``parameters`` is the complete resolved Evaluation Protocol public-parameter
    mapping, including defaults for published parameters omitted by the Study.

    The trial Model is implicit from the containing plan row, so Model identity is not
    repeated here. No Evaluation Run ID is present because concrete child Runs are
    allocated only after plan materialization.
    """

    stage: str
    corpus: CorpusId
    protocol: EvaluationProtocolId
    parameters: ResolvedPublicParameters


@dataclass(frozen=True, slots=True)
class StudyPlanRow:
    """One materialized Study Run-local trial row.

    ``trial`` is the persisted Study Run-local ``trial-NNNN`` string. It is deliberately
    not promoted to a shared entity-ID type.

    Exactly one Model source is valid: ``training`` for a training-derived trial or
    ``model`` for an authored existing-Model trial. Existing-Model rows therefore carry
    only the immutable selected :class:`ModelId` as their Model source and do not use a
    dummy training object.

    ``evaluations`` contains the materialized evaluation stages in their persisted
    order and must be non-empty. Every entry is already fully resolved before the plan
    is considered valid.
    """

    trial: str
    evaluations: Sequence[StudyPlanEvaluation]
    training: StudyPlanTraining | None = None
    model: ModelId | None = None


StudyPlan: TypeAlias = Sequence[StudyPlanRow]
"""Ordered materialized rows comprising one Study Run plan.

The plan artifact's exact bytes, SHA-256, byte size, row counts, and persistence
location belong to Study Run persistence rather than this representation. The alias
fixes only the ordered row surface required by Study materialization and downstream
child-execution planning.
"""


def validate_study_plan(plan: StudyPlan) -> ValidationReport:
    """Validate plan-local structural invariants without Study/protocol I/O.

    This validation owns rules decidable from the materialized ordered rows alone:
    the plan is non-empty; trial strings follow ``trial-NNNN`` grammar, are unique,
    begin at ``trial-0001``, and correspond sequentially without gaps to row position;
    every row has exactly one of ``training`` or ``model``; evaluation lists are
    non-empty; evaluation stage strings are unique within each row; training seeds are
    integers but not booleans; and every represented training/evaluation parameter
    value belongs to the recursive JSON-compatible public-parameter domain. Duplicate
    represented Model-source coordinates within the plan are also plan-local
    invalidity. Public-parameter portions of coordinate comparison preserve recursive
    decoded type distinctions, so for example ``True`` and ``1`` are distinct values.

    This function does not prove that rows were generated from a particular Study,
    verify canonical Cartesian-product or authored existing-Model ordering, verify
    that an existing Model was selected by that Study, resolve referenced assets,
    check Task compatibility, or prove that a resolved parameter mapping contains
    exactly the keys/defaults declared by its selected Train/Evaluation Protocol.
    Likewise, agreement of evaluation stage membership/order with the authored Study
    requires Study context. Those checks belong to materialization/execution
    validation with both the Study and referenced protocol/assets available.

    Serialization syntax, persisted JSONL bytes, SHA-256/byte-size integrity, Study Run
    lifecycle, child Run identity, retries, and queue/worker state are outside this
    validator.
    """

    ...
