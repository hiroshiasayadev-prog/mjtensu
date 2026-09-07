"""Public Python signatures for MLDB Study Run records and lifecycle rules.

This skeleton fixes the persisted/domain representation of one concrete execution
of one sealed Study. A Study Run may exist while concrete plan materialization is
still in progress; once a complete valid ``plan.jsonl`` is established, its
integrity/count metadata becomes immutable execution intent for that Study Run.

Optional summary values are derived convenience data over planned coordinates and
child Run lineage. They are not authoritative child history and do not introduce
child Run inventories, retry state, Queue state, or orchestration implementation
state into the Study Run record.

This module intentionally does not validate Study execution preflight, materialize,
serialize, hash, or persist plans, allocate or execute child Runs, derive summaries,
reconcile coordinates, scan child lineage, schedule retries, or manage Queue/Worker
state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ..common.errors import ValidationReport
from ..common.ids import StudyId, StudyRunId


class StudyRunStatus(str, Enum):
    """Persisted Study Run lifecycle state."""

    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class StudyRunExecution:
    """Execution facts owned directly by one Study Run.

    ``started_at`` and ``finished_at`` are persisted timestamps, but current Design
    Records do not standardize a concrete Python timestamp class or normalization
    rule. Their in-memory representation therefore remains intentionally opaque,
    matching the Training Run and Evaluation Run skeletons.

    ``finished_at`` is required for every terminal Study Run. The current Study Run
    contract does not additionally prohibit a represented ``finished_at`` while the
    record is ``RUNNING``.
    """

    started_at: object
    finished_at: object | None = None


@dataclass(frozen=True, slots=True)
class StudyRunPlan:
    """Integrity and count metadata for one complete immutable Study Run plan.

    This value exists only after complete valid plan materialization succeeds.
    ``path`` is the Study Run-relative persisted path and is exactly ``plan.jsonl``
    in v1. ``sha256`` and ``bytes`` describe the exact complete persisted plan bytes.
    ``trials`` is the plan row count for either Study Model-source mode, and
    ``evaluation_jobs`` is the total number of materialized evaluation-stage entries
    across all plan rows.

    This metadata does not freeze JSON serialization details that the current plan
    contract leaves non-normative, including object-key ordering, whitespace, line
    endings, or independent byte-identical regeneration policy.
    """

    path: Literal["plan.jsonl"]
    sha256: str
    bytes: int
    trials: int
    evaluation_jobs: int


@dataclass(frozen=True, slots=True)
class StudyRunTrainingSummary:
    """Derived final outcome counts for planned training coordinates.

    These are coordinate counts, not historical Training Run attempt counts. A
    coordinate whose failed or cancelled attempt is followed by a successful retry
    contributes once to ``completed`` while the earlier child Run remains historical
    evidence through its own Study lineage.

    This summary is meaningful only for a training-derived Study. Existing-Model
    Studies have no planned training coordinates and therefore do not require a dummy
    zero-valued training summary.
    """

    completed: int
    failed: int
    cancelled: int


@dataclass(frozen=True, slots=True)
class StudyRunEvaluationSummary:
    """Derived final outcome counts for planned evaluation coordinates.

    ``completed``, ``completed_partial``, ``failed``, and ``cancelled`` describe final
    planned-coordinate outcomes derived from Evaluation Run history. ``blocked`` is a
    Study-level derived outcome for a planned evaluation coordinate that could not
    produce an Evaluation Run because its required upstream Model was ultimately
    unavailable.

    ``blocked`` is deliberately not an Evaluation Run lifecycle status and this type
    does not require synthetic Evaluation Runs for blocked coordinates.
    """

    completed: int
    completed_partial: int
    failed: int
    cancelled: int
    blocked: int


@dataclass(frozen=True, slots=True)
class StudyRunSummary:
    """Optional derived Study Run summary over final planned-coordinate outcomes.

    Both persisted summary sections remain optional because the current format says a
    summary may include evaluation counts and a training-derived Study may additionally
    include training counts. ``training`` is therefore absent for an existing-Model
    Study and need not be represented merely as a dummy zero counter. Whether
    training-summary presence agrees with the referenced Study Model-source mode
    requires resolving that Study and is not decidable from this value alone.

    The summary is a convenience cache only. Immutable plan intent plus Training Run
    and Evaluation Run lineage remain the source of truth and may be used to rebuild
    this value.
    """

    training: StudyRunTrainingSummary | None = None
    evaluation: StudyRunEvaluationSummary | None = None


@dataclass(frozen=True, slots=True)
class StudyRun:
    """One persisted MLDB Study execution event.

    ``study`` references exactly one Study by :class:`StudyId`; the Study definition
    and its lifecycle state are not embedded or duplicated here.

    ``plan`` is absent before complete valid plan materialization has succeeded and
    may therefore be ``None`` while the Study Run is ``RUNNING``. Once a complete
    plan is established, ``plan`` remains present and immutable for every later state.
    ``COMPLETED`` and ``COMPLETED_WITH_FAILURES`` always require it. ``RUNNING``,
    ``FAILED``, and ``CANCELLED`` can each be represented with or without plan
    metadata because termination or current execution may occur on either side of
    plan finalization.

    ``summary`` is optional derived information and is never an authoritative child
    Run inventory. This v1 record intentionally contains no Training Run ID list,
    Evaluation Run ID list, Model list, attempt/retry counters, next-retry timestamp,
    active Worker/current-job fields, Queue status, or other orchestration state.

    Study Run v1 also defines no persisted ``failure`` field. A ``FAILED`` status is
    sufficient at this contract layer; concise failure metadata is not invented by
    analogy with Training Run or Evaluation Run formats.
    """

    schema: Literal["mjtensu.mldb/study-run/v1"]
    id: StudyRunId
    status: StudyRunStatus
    study: StudyId
    execution: StudyRunExecution
    plan: StudyRunPlan | None = None
    summary: StudyRunSummary | None = None


def validate_study_run(run: StudyRun) -> ValidationReport:
    """Validate Study Run metadata-local v1 invariants without external I/O.

    This validation owns only rules decidable from one normalized Study Run record,
    including:

    - the exact ``mjtensu.mldb/study-run/v1`` schema declaration;
    - Study Run event-ID grammar ``sr-YYYYMMDD-NNN``;
    - the five Study Run lifecycle values;
    - every terminal status requiring ``execution.finished_at``;
    - represented plan metadata having the exact ``plan.jsonl`` relative path, valid
      local SHA-256/integer metadata shape, and non-negative ``bytes``, ``trials``,
      and ``evaluation_jobs`` values with booleans rejected as integers;
    - ``COMPLETED`` and ``COMPLETED_WITH_FAILURES`` requiring plan metadata, while
      ``RUNNING``, ``FAILED``, and ``CANCELLED`` permit either plan presence or
      absence because complete plan materialization may or may not have occurred;
    - represented training/evaluation summary count fields being integers, not
      booleans, and non-negative.

    This metadata-local validator does not impose a total-equality relation between
    represented summary counts and ``plan.trials`` / ``plan.evaluation_jobs``. The
    current Design Records define summary as optional derived convenience data but do
    not normatively fix such a completeness equation at this static record boundary.

    This validator does not resolve the referenced Study or verify that it is sealed;
    compare the Study Run directory basename with ``id``; inspect ``plan.jsonl`` bytes
    or prove SHA-256/byte-size agreement; count plan rows or evaluation entries; prove
    that plan contents came from the referenced Study; determine whether the Study is
    training-derived or existing-Model and therefore whether ``summary.training``
    should be present; scan Training Run/Evaluation Run lineage; derive or reconcile
    summary outcomes; prove final Study Run status from coordinate satisfaction;
    compare terminal records for historical immutability; or inspect Queue/retry/
    Worker state. Those checks require repository, Study, plan, child-history, prior-
    record, or orchestration context outside this static domain boundary.
    """

    ...


def validate_study_run_transition(
    source: StudyRunStatus,
    target: StudyRunStatus,
) -> ValidationReport:
    """Validate one requested Study Run lifecycle status transition.

    The only valid v1 transitions are ``RUNNING -> COMPLETED``,
    ``RUNNING -> COMPLETED_WITH_FAILURES``, ``RUNNING -> FAILED``, and
    ``RUNNING -> CANCELLED``. No transition out of a terminal state is valid.

    This boundary validates lifecycle state movement only. Plan finalization or
    summary enrichment while a Study Run remains ``RUNNING`` is not a status
    transition. Actual mutation/persistence, child execution, retry/reconciliation,
    and terminal historical immutability enforcement belong to later orchestration
    and repository workflows.
    """

    ...
