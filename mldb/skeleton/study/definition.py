"""Public Python signatures for MLDB Study v1 definitions.

This skeleton fixes the reusable Study definition boundary only. A Study selects
exactly one Model source, either a finite training grid or an authored ordered set
of existing Models, and applies one or more fixed evaluation stages to every
materialized Model.

This module intentionally does not resolve referenced assets, resolve public
parameter defaults, expand Cartesian products, assign trial IDs, define Study Run
plan rows, allocate Runs, persist plans, compare sealed revisions, or perform
queue/worker execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

from ..common.errors import ValidationReport
from ..common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    ModelId,
    StudyId,
    TrainProtocolId,
)
from ..common.parameters import PublicParameterOverrides, PublicParameterValue


class StudyStatus(str, Enum):
    """Persisted Study definition lifecycle state."""

    DRAFT = "draft"
    SEALED = "sealed"


@dataclass(frozen=True, slots=True)
class StudyTrainingParameterAxis:
    """One caller-authored Train Protocol public-parameter grid axis.

    ``values`` is a non-empty ordered sequence in every valid Study. Every item
    remains an unresolved caller-supplied :class:`PublicParameterValue`; omitted
    Train Protocol defaults are not inserted at Study-definition time.

    Authored value order is semantically significant. Duplicate values are invalid
    under YAML-decoded, recursive, type-sensitive equality: values of different
    decoded types are not duplicates merely because Python equality may coerce them
    to equal values, for example ``True`` and ``1``. Concrete duplicate comparison
    logic is intentionally outside this signature-only skeleton.
    """

    values: Sequence[PublicParameterValue]


@dataclass(frozen=True, slots=True)
class StudyTrainingModelSource:
    """Study Model source that materializes new Models from a finite training grid.

    ``architectures`` is a non-empty ordered sequence of unique Architecture IDs.
    ``parameters`` maps Train Protocol public keys to Study-owned grid axes and may
    be empty. Mapping key insertion order is not semantic; canonical parameter-key
    sorting belongs to later grid expansion. Each axis preserves authored value
    order.

    ``seeds`` is a non-empty ordered sequence of unique training seeds. Every seed
    is an integer, boolean is invalid, no generic numeric range exists, and no
    coercion is permitted. Seed remains separate from public protocol parameters.

    This type represents the contents of persisted ``model.train``. It does not
    imply that referenced assets exist, are sealed, or are Task-compatible.
    """

    corpus: CorpusId
    protocol: TrainProtocolId
    architectures: Sequence[ArchitectureId]
    parameters: Mapping[str, StudyTrainingParameterAxis]
    seeds: Sequence[int]


@dataclass(frozen=True, slots=True)
class StudyExistingModelSource:
    """Study Model source selecting already-created Models in authored order.

    ``models`` is a non-empty ordered sequence of unique Model IDs in every valid
    Study. Authored order is semantically significant and later determines
    Study-local trial ordering for existing-Model materialization.

    This type represents the persisted ``model.existing`` list. Whether each Model
    exists, resolves through a completed Training Run with valid canonical weights,
    and shares one Task with the other selected Models is execution-preflight work,
    not definition-local validation.
    """

    models: Sequence[ModelId]


StudyModelSource: TypeAlias = StudyTrainingModelSource | StudyExistingModelSource
"""Exactly one Study Model-source alternative.

The union makes the normalized Python representation exclusive by construction:
one :class:`Study` contains either a training source or an existing-Model source,
never both. A YAML parser must therefore reject persisted ``model`` mappings that
contain both ``train`` and ``existing`` or neither before constructing this domain
representation. The parser shape itself is not part of this skeleton.
"""


@dataclass(frozen=True, slots=True)
class StudyEvaluationStage:
    """One fixed evaluation declaration applied to every Study trial Model.

    ``stage`` is a Study-local stable string, not an MLDB entity ID, and must be
    unique within the containing Study. Evaluation-stage declaration order is
    semantically significant.

    ``parameters`` directly reuses :class:`PublicParameterOverrides`: it contains
    fixed caller-supplied values only, may be empty, is not a Cartesian grid, and
    does not include omitted Evaluation Protocol defaults. Unknown parameter keys
    and default resolution require the referenced protocol and therefore belong to
    execution preflight/materialization rather than static Study metadata validation.
    """

    stage: str
    corpus: CorpusId
    protocol: EvaluationProtocolId
    parameters: PublicParameterOverrides


@dataclass(frozen=True, slots=True)
class Study:
    """One reusable MLDB Study v1 experiment definition.

    ``model`` contains exactly one normalized Model-source alternative through
    :class:`StudyModelSource`. ``evaluations`` is a non-empty ordered sequence in
    every valid Study and preserves authored stage order.

    A Study remains declarative experiment intent. It contains no resolved protocol
    defaults, Cartesian coordinates, trial identities, child Run identities,
    StudyRun lifecycle state, queue state, or execution results.
    """

    schema: Literal["mjtensu.mldb/study/v1"]
    id: StudyId
    status: StudyStatus
    name: str
    description: str
    model: StudyModelSource
    evaluations: Sequence[StudyEvaluationStage]


def validate_study_metadata(study: Study) -> ValidationReport:
    """Validate Study-format invariants decidable from one normalized definition.

    Static validation owns only definition-local rules, including:

    - the exact ``mjtensu.mldb/study/v1`` schema declaration;
    - terminal positive-integer ``-vN`` Study ID grammar;
    - ``draft`` / ``sealed`` lifecycle value;
    - the exclusive Model-source alternative represented by :class:`StudyModelSource`;
    - a non-empty evaluation-stage sequence with unique ``stage`` strings;
    - for training sources, non-empty unique Architectures, non-empty seeds whose
      values are integer and not boolean, unique seeds, non-empty parameter axes,
      public-parameter value-domain validity, and no duplicate values within an axis;
    - for existing-Model sources, a non-empty sequence with no duplicate Model IDs;
    - public-parameter value-domain validity for fixed evaluation overrides.

    Duplicate training-axis values use YAML-decoded recursive type-sensitive equality.
    In particular, host-language coercive equality must not make differently typed
    decoded values duplicates (for example ``True`` versus ``1``). This function's
    implementation must preserve that rule rather than relying on ordinary Python
    equality alone. This skeleton intentionally supplies no private comparison helper.

    The normalized union makes ``train`` versus ``existing`` exclusive in Python.
    Raw YAML shape validation must reject a ``model`` mapping containing both
    alternatives or neither before constructing :class:`Study`; YAML parsing itself
    is not defined here.

    This function does not compare the Study ID with a filename basename, resolve any
    referenced Corpus, Architecture, Train Protocol, Model, or Evaluation Protocol,
    verify referenced-asset sealed state or implementation/artifact integrity, verify
    common Task agreement, verify that supplied parameter keys are published by their
    referenced protocols, resolve omitted public-parameter defaults, validate existing
    Model completed lineage, compare mutations of a sealed Study, require ``sealed``
    for execution, expand the training grid, assign trial IDs, create or validate plan
    rows, allocate a Study Run or child Run, persist/hash a plan, or perform queue/
    worker behavior. Those concerns belong to repository resolution, execution
    preflight, Study materialization/plan contracts, lifecycle tooling, Run execution,
    or orchestration as applicable.
    """

    ...
