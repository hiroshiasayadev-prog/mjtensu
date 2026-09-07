"""Public Python signatures for MLDB Training Run records and lifecycle rules.

This skeleton fixes the persisted/domain representation of one concrete training
execution attempt after launch preflight has succeeded. It intentionally does not
resolve assets, allocate Run IDs, construct TrainContext, load or invoke executable
Train Protocol code, serialize canonical weights, persist run.yaml, create Models,
or implement queue/worker retry behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Mapping

from ..common.errors import ValidationReport
from ..common.ids import (
    ArchitectureId,
    CorpusId,
    StudyRunId,
    TrainingRunId,
    TrainProtocolId,
)
from ..common.parameters import PublicParameterValue, ResolvedPublicParameters
from .weights import CanonicalWeightsArtifact


class TrainingRunStatus(str, Enum):
    """Persisted Training Run lifecycle state."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TrainingRunExecution:
    """Execution facts owned directly by one Training Run.

    ``seed`` preserves the validated integer Run seed. Boolean is invalid, there is
    no universal MLDB numeric range, and no coercion is permitted.

    ``started_at`` and ``finished_at`` are persisted timestamps, but current Design
    Records do not standardize a concrete Python timestamp class or normalization
    rule. Their in-memory representation therefore remains intentionally opaque in
    this public skeleton. ``finished_at`` is absent while the Run is ``RUNNING`` and
    required for every terminal Run.
    """

    seed: int
    started_at: object
    finished_at: object | None = None


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Successful Training Run result metadata.

    The wrapper mirrors the persisted ``result.weights`` shape while delegating the
    canonical learned-weight format/path/hash/byte contract entirely to the frozen
    :class:`CanonicalWeightsArtifact` signature.
    """

    weights: CanonicalWeightsArtifact


@dataclass(frozen=True, slots=True)
class TrainingRunFailure:
    """Concise historical failure fact for a failed Training Run."""

    type: str
    message: str


@dataclass(frozen=True, slots=True)
class TrainingRunStudyLineage:
    """Optional downstream-to-upstream Study lineage for one Training Run.

    ``run`` identifies the owning Study Run. ``trial`` is the Study Run-local
    ``trial-NNNN`` identifier from its immutable plan. Trial is not an MLDB entity,
    so this module deliberately does not introduce a shared ``TrialId`` or depend on
    a future Study-domain object.
    """

    run: StudyRunId
    trial: str


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """One persisted MLDB training execution attempt.

    Every represented Run already passed launch preflight before allocation, so the
    selected Corpus, Architecture, Train Protocol, validated seed, and complete
    resolved public-parameter mapping are present from the initial ``RUNNING``
    record. Task is intentionally not duplicated because agreement is established
    during preflight from the referenced reusable assets.

    ``result`` is present only for ``COMPLETED`` Runs. ``failure`` is optional even
    for ``FAILED`` Runs because the persisted contract records concise failure text
    only when safe and available. ``study`` is optional, but when present its
    ``run`` and ``trial`` members are both required by construction.

    ``environment`` and ``work`` preserve the v1 optional generic mappings without
    standardizing environment vocabulary, GPU/package schemas, or working-file type
    hierarchies.
    """

    schema: Literal["mjtensu.mldb/training-run/v1"]
    id: TrainingRunId
    status: TrainingRunStatus
    corpus: CorpusId
    architecture: ArchitectureId
    train_protocol: TrainProtocolId
    parameters: ResolvedPublicParameters
    execution: TrainingRunExecution
    result: TrainingRunResult | None = None
    failure: TrainingRunFailure | None = None
    study: TrainingRunStudyLineage | None = None
    environment: Mapping[str, object] | None = None
    work: Mapping[str, object] | None = None


def validate_training_run(run: TrainingRun) -> ValidationReport:
    """Validate Training Run metadata-local v1 invariants without external I/O.

    This validation owns rules decidable from one normalized in-memory record,
    including the schema and Training Run ID grammar, lifecycle/result conditional,
    terminal ``finished_at`` conditional, integer-and-not-boolean seed contract,
    JSON-compatible public-parameter value domain, Study-lineage local shape, and
    canonical ``result.weights`` metadata shape.

    In particular, a ``COMPLETED`` Run requires exactly one canonical result;
    non-completed Runs may not carry canonical result metadata; ``RUNNING`` has no
    ``finished_at``; and every terminal state requires it. Failure metadata is not
    required even for ``FAILED`` because the v1 record only says it should be stored
    when safe and available; when represented, its local text shape is validated.

    This function does not resolve Corpus, Architecture, Train Protocol, or Study
    references; verify Task agreement; prove that ``parameters`` are the exact result
    of protocol default/override resolution; compare Study lineage or inputs with a
    Study plan; verify canonical weight bytes against their SHA-256/byte metadata;
    compare the Run directory basename with ``id``; or enforce terminal persistence
    immutability against a historical record. Those checks require runtime,
    repository, integrity, or Study resources outside this static domain boundary.
    """

    ...


def validate_training_run_transition(
    source: TrainingRunStatus,
    target: TrainingRunStatus,
) -> ValidationReport:
    """Validate one requested Training Run lifecycle status transition.

    The only valid v1 transitions are ``RUNNING -> COMPLETED``, ``RUNNING -> FAILED``,
    and ``RUNNING -> CANCELLED``. No transition out of a terminal state is valid.

    This boundary validates lifecycle state movement only. Enriching a still-running
    record with newly known execution facts is not a status transition, and actual
    record mutation/persistence plus terminal immutability enforcement belong to the
    later execution/repository workflow.
    """

    ...
