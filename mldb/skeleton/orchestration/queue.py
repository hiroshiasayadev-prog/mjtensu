"""Public value signatures for the durable MLDB operational Queue.

This skeleton projects Study-derived logical jobs into Controller-local Queue
persistence without redefining Study execution intent. Queue state is replaceable
operational state: immutable Study Run plans plus canonical child Run / Model history
remain authoritative experiment facts.

The concrete v1 SQLite adapter, SQL schema management, transaction mechanics,
Controller dispatch locking, Worker API payloads, child Run allocation, executable
invocation, canonical result acceptance, and full reconciliation algorithm are outside
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..common.ids import EvaluationRunId, TrainingRunId
from .jobs import StudyJob


class QueueJobStatus(str, Enum):
    """Durable operational lifecycle state of one logical Queue job.

    These values are exactly the v1 Queue states and are deliberately distinct from
    Training Run, Evaluation Run, and Study Run lifecycle enums.
    """

    BLOCKED = "blocked"
    READY = "ready"
    ACTIVE = "active"
    RETRY_WAIT = "retry_wait"
    SATISFIED = "satisfied"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class QueueJob:
    """Persisted operational projection of one Study-derived logical job.

    ``job_id`` is the SQLite-local integer primary key. It is not an MLDB entity ID
    and must not be persisted into Study plans or child Run records.

    ``logical`` directly reuses the frozen :class:`StudyJob` union. The SQLite
    ``study_run_id``, ``trial_id``, ``kind``, and nullable ``stage`` columns are the
    storage projection of this value rather than a second generic job identity.
    ``dependency_job_id`` is the local SQLite foreign key corresponding to
    ``EvaluationJob.training_dependency`` when one exists; it is ``None`` for
    Training jobs and existing-Model Evaluation jobs.

    ``retry_not_before`` is present exactly for ``RETRY_WAIT`` and absent for every
    other status. ``created_at``, ``updated_at``, and represented retry timestamps are
    canonical Queue storage timestamps: UTC RFC3339 text with exactly six fractional
    digits and a ``Z`` suffix (``YYYY-MM-DDTHH:MM:SS.ffffffZ``). No shared Python
    timestamp abstraction is introduced here.

    The Queue stores no Architecture, Corpus, Protocol, seed, public parameters,
    Model identity, output declaration, or canonical result payload; those remain
    authoritative in the immutable Study plan and canonical MLDB records.
    """

    job_id: int
    logical: StudyJob
    status: QueueJobStatus
    dependency_job_id: int | None
    retry_not_before: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class QueueAttempt:
    """One persisted concrete Worker attempt after child Run allocation.

    ``attempt_id`` and ``job_id`` are SQLite-local integers. ``attempt_no`` begins at
    one per logical job and increases by one for later attempts represented by the
    current Queue database; it is operational history rather than canonical Run
    lineage identity.

    ``run_id`` is the already-allocated concrete child Run identity. Its Python type
    remains the typed union ``TrainingRunId | EvaluationRunId`` rather than flattening
    both Run kinds to plain ``str``. Agreement between this Run kind and the parent
    Queue job kind is a Queue validation/reconciliation invariant.

    ``worker_id``, ``acquire_token``, and ``lease_token`` are non-empty opaque strings.
    Tokens deliberately receive no new class hierarchy. ``acquire_token`` supports
    replay of a lost acquire response; ``lease_token`` identifies the currently
    authorized lease for this attempt.

    ``lease_until``, ``started_at``, and optional ``finished_at`` use the same exact
    canonical Queue timestamp text format as :class:`QueueJob`. ``finished_at is None``
    means the attempt is open; once closed it must never be reopened. ``close_reason``
    is optional concise machine-readable operational metadata and is not canonical
    child Run failure/status truth.

    No attempt exists for a concrete preflight failure that occurred before child Run
    allocation.
    """

    attempt_id: int
    job_id: int
    attempt_no: int
    run_id: TrainingRunId | EvaluationRunId
    worker_id: str
    acquire_token: str
    lease_token: str
    lease_until: str
    started_at: str
    finished_at: str | None = None
    close_reason: str | None = None


def queue_database_path(repository_root: Path) -> Path:
    """Return the v1 Controller-local Queue database path for ``repository_root``.

    The result is exactly ``<repository_root>/.local/mldb/queue.sqlite``. This path is
    operational machine-local state and intentionally does not extend or modify the
    canonical :class:`RepositoryLayout` / ``mldb_data`` contract.
    """

    ...
