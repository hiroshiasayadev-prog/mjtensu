"""Public Python signatures for MLDB Evaluation Run records and lifecycle rules.

This skeleton fixes the persisted/domain representation of one concrete evaluation
execution attempt after launch preflight has succeeded. It connects the frozen
Evaluation result-acceptance contract to Evaluation Run v1 persistence without
redefining accepted artifact, unavailable-output, or validation-issue metadata.

It intentionally does not resolve assets or parameters, allocate Run IDs, load or
invoke Evaluation Protocol code, construct EvaluationContext, accept/materialize
candidate results, persist run.yaml, manage Study lifecycle, or implement queue/worker
behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ..common.errors import ValidationReport
from ..common.ids import (
    CorpusId,
    EvaluationProtocolId,
    EvaluationRunId,
    ModelId,
    StudyRunId,
)
from ..common.parameters import PublicParameterValue, ResolvedPublicParameters
from .interface import UnavailableOutput
from .result_validation import AcceptedEvaluationArtifact, EvaluationValidationIssue


class EvaluationRunStatus(str, Enum):
    """Persisted Evaluation Run lifecycle state."""

    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_PARTIAL = "completed_partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class EvaluationRunExecution:
    """Execution facts owned directly by one Evaluation Run.

    Evaluation Run v1 has no universal seed field. A stochastic Evaluation Protocol
    exposes any caller-controlled seed through ordinary resolved public parameters.

    ``started_at`` and ``finished_at`` are persisted timestamps, but current Design
    Records do not standardize a concrete Python timestamp class or normalization
    rule. Their in-memory representation therefore remains intentionally opaque,
    matching the Training Run skeleton. ``finished_at`` is absent while the Run is
    ``RUNNING`` and required for every terminal Run.
    """

    started_at: object
    finished_at: object | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """Persisted accepted formal result surface for one Evaluation Run.

    ``metrics`` contains only accepted finite scalar metric values. ``artifacts``
    directly reuses the frozen :class:`AcceptedEvaluationArtifact` metadata contract;
    no Evaluation-Run-specific artifact wrapper is introduced.

    This wrapper mirrors the persisted ``result.metrics`` / ``result.artifacts``
    hierarchy. Incompleteness records intentionally remain top-level Evaluation Run
    fields rather than being nested inside this value.
    """

    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, AcceptedEvaluationArtifact]


@dataclass(frozen=True, slots=True)
class EvaluationRunFailure:
    """Concise historical failure fact for an Evaluation Run when safely available."""

    type: str
    message: str


@dataclass(frozen=True, slots=True)
class EvaluationRunStudyLineage:
    """Optional downstream-to-upstream Study lineage for one Evaluation Run.

    ``run`` identifies the owning Study Run. ``trial`` and ``stage`` are Study-local
    strings from that Study Run's immutable plan; neither is an MLDB entity identity.
    This module therefore introduces no Trial/Stage ID type and has no dependency on
    Study domain objects.
    """

    run: StudyRunId
    trial: str
    stage: str


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """One persisted MLDB evaluation execution attempt.

    Every represented Run already passed launch preflight before allocation, so the
    selected Model, Corpus, Evaluation Protocol, and complete resolved public-parameter
    mapping are present from the initial ``RUNNING`` record. Task is intentionally not
    duplicated because compatibility is established during preflight.

    ``result`` mirrors persisted accepted formal output metadata. Both ``COMPLETED``
    and ``COMPLETED_PARTIAL`` require it, while current v1 metadata rules do not add a
    symmetric prohibition on its presence for other statuses merely by convention.

    ``unavailable_outputs`` and ``validation_issues`` are persisted at Evaluation Run
    top level and directly reuse the frozen Wave 4-B/4-C contracts. A fully
    ``COMPLETED`` Run has neither kind of incompleteness record. A
    ``COMPLETED_PARTIAL`` Run has at least one such reason and retains at least one
    accepted metric or artifact.

    ``failure`` is optional historical text; v1 does not require it even for every
    ``FAILED`` Run. ``study`` is optional and, when represented, contains all three
    lineage members by construction. ``environment`` remains opaque optional metadata
    without a standardized host/GPU/package schema.
    """

    schema: Literal["mjtensu.mldb/evaluation-run/v1"]
    id: EvaluationRunId
    status: EvaluationRunStatus
    model: ModelId
    corpus: CorpusId
    evaluation_protocol: EvaluationProtocolId
    parameters: ResolvedPublicParameters
    execution: EvaluationRunExecution
    result: EvaluationRunResult | None = None
    unavailable_outputs: Sequence[UnavailableOutput] = ()
    validation_issues: Sequence[EvaluationValidationIssue] = ()
    failure: EvaluationRunFailure | None = None
    study: EvaluationRunStudyLineage | None = None
    environment: Mapping[str, object] | None = None


def validate_evaluation_run(run: EvaluationRun) -> ValidationReport:
    """Validate Evaluation Run metadata-local v1 invariants without external I/O.

    This validation owns rules decidable from one normalized in-memory record,
    including the v1 schema and Evaluation Run event-ID grammar, lifecycle value,
    terminal ``finished_at`` conditional, JSON-compatible resolved public-parameter
    value domain, local accepted scalar/artifact metadata shape, Study-lineage local
    shape, and failure text shape.

    Both ``COMPLETED`` and ``COMPLETED_PARTIAL`` require ``result`` with represented
    ``metrics`` and ``artifacts`` mappings. ``COMPLETED_PARTIAL`` additionally requires
    at least one accepted formal metric or artifact and at least one incompleteness
    record in ``unavailable_outputs`` or ``validation_issues``. ``COMPLETED`` requires
    both incompleteness sequences to be empty. Empty metric and/or artifact mappings
    are otherwise valid. The current contract does not add a status-symmetry rule that
    rejects ``result`` or incompleteness records from ``RUNNING``, ``FAILED``, or
    ``CANCELLED`` solely because those states are not successful terminal states.

    Accepted metric local validation enforces finite ``int``/``float`` values with
    booleans rejected. Accepted artifact local validation checks only represented
    metadata shape; agreement with actual artifact bytes and actual filesystem
    containment remain external integrity responsibilities.

    This function does not resolve Model, Corpus, Evaluation Protocol, or Study
    references; verify Task compatibility; prove that ``parameters`` are the exact
    complete mapping produced by the selected protocol; compare Study lineage and Run
    inputs/parameters with a Study plan; verify accepted artifact bytes against their
    SHA-256/byte metadata or actual storage containment; compare the Run directory
    basename with ``id``; or enforce terminal historical immutability against another
    persisted record. Those checks require runtime, repository, integrity, Study, or
    historical state outside this static domain boundary.
    """

    ...


def validate_evaluation_run_transition(
    source: EvaluationRunStatus,
    target: EvaluationRunStatus,
) -> ValidationReport:
    """Validate one requested Evaluation Run lifecycle status transition.

    The only valid v1 transitions are ``RUNNING -> COMPLETED``,
    ``RUNNING -> COMPLETED_PARTIAL``, ``RUNNING -> FAILED``, and
    ``RUNNING -> CANCELLED``. No transition out of a terminal state is valid.

    This boundary validates lifecycle state movement only. Updating execution facts
    while a Run remains ``RUNNING`` is not a status transition, and actual record
    mutation/persistence plus terminal immutability enforcement belong to later
    execution/repository workflow.
    """

    ...
