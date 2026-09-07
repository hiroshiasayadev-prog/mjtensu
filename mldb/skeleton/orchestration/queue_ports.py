"""Consumer-oriented persistence boundary for the MLDB operational Queue.

The protocol is intentionally semantic rather than SQL-shaped. A concrete v1 adapter
may use SQLite, but callers operate on Study-derived Queue jobs, attempts, leases, and
lifecycle transitions rather than tables, cursors, transactions, or generic CRUD.

Controller remains responsible for concrete Training/Evaluation preflight, child Run
allocation and canonical ``running`` Run persistence, result acceptance, child Run
terminalization, Model creation, Study finalization, and reconciliation judgment.
Worker API request/response types are deliberately absent so Wave 7-2B can depend on
this boundary without this module assuming its parallel contract.
"""

from __future__ import annotations

from typing import Protocol

from ..common.ids import EvaluationRunId, StudyRunId, TrainingRunId
from .jobs import StudyJob
from .queue import QueueAttempt, QueueJob, QueueJobStatus


class QueuePort(Protocol):
    """Durable Queue-side operations consumed by Controller orchestration.

    All timestamp arguments and returned timestamp fields use the v1 Queue storage
    encoding exactly: UTC RFC3339 with six fractional-second digits and ``Z`` suffix.
    The adapter owns validation/encoding enforcement; this protocol intentionally does
    not introduce a datetime-normalization framework.

    Mutations that touch both a job and its attempt must be atomic within the Queue
    store. The Queue database is not atomically transactional with canonical filesystem
    MLDB state; Controller preserves the cross-store ordering required by the
    orchestration specs and reconciliation repairs any crash gap.
    """

    def admit_study_jobs(
        self,
        study_run_id: StudyRunId,
        jobs: frozenset[StudyJob],
        *,
        admitted_at: str,
    ) -> tuple[QueueJob, ...]:
        """Atomically admit one complete derived Study job set.

        ``jobs`` is the complete result of Wave 7-1 logical job derivation for
        ``study_run_id``. Queue therefore does not receive or independently interpret
        :class:`StudyPlan`, and it must not reconstruct execution payload from the
        logical coordinates.

        In one Queue transaction, new training jobs become ``READY``; training-derived
        Evaluation jobs become ``BLOCKED`` and reference their same-trial Training
        Queue row through ``dependency_job_id``; existing-Model Evaluation jobs become
        ``READY`` with no dependency row.

        Repeated admission is safe when the already-persisted logical coordinates and
        dependency relationships agree. Existing operational status, attempts, retry
        timing, and creation timestamps must not be reset merely because admission is
        repeated. Any existing coordinate whose logical kind/stage/dependency relation
        conflicts with the supplied immutable intent is rejected rather than rewritten.
        Admission is all-or-nothing for the Study Run.
        """

        ...

    def jobs_for_study_run(self, study_run_id: StudyRunId) -> tuple[QueueJob, ...]:
        """Return all persisted Queue jobs owned by one Study Run.

        Result ordering is not a scheduling-priority contract. This read exists for
        Controller inspection, Study progress, and later reconciliation.
        """

        ...

    def job_by_id(self, job_id: int) -> QueueJob | None:
        """Return one Queue-local job, or ``None`` when no such row exists."""

        ...

    def attempts_for_job(self, job_id: int) -> tuple[QueueAttempt, ...]:
        """Return the persisted attempt history currently retained for ``job_id``.

        Attempt history is operational and may be incomplete after total Queue rebuild;
        canonical child Run lineage remains authoritative experiment history.
        """

        ...

    def attempt_by_id(self, attempt_id: int) -> QueueAttempt | None:
        """Return the exact persisted attempt with this Queue-local identity.

        Both open and closed attempts are eligible. ``None`` means that no retained
        Queue row has this ``attempt_id``; a total Queue rebuild may therefore make old
        operational attempt identity unavailable. This is historical inspection only:
        it performs no lease authorization or Queue mutation, never reopens a closed
        attempt, and does not infer canonical child Run state or outcome-replay
        acknowledgement.
        """

        ...

    def open_attempt_for_job(self, job_id: int) -> QueueAttempt | None:
        """Return the single open attempt for ``job_id``, if one exists."""

        ...

    def attempt_by_acquire_token(self, acquire_token: str) -> QueueAttempt | None:
        """Return the historical or open attempt owning one acquire token.

        Controller/Worker API uses this lookup before performing a new concrete
        preflight or allocating another child Run. If the returned attempt is still
        open, a lost assignment response can recover that same Run ID, Worker
        attribution, and current lease rather than duplicate execution. If it is
        already closed, the token is historical and must be rejected for new
        activation rather than reused.

        Returning closed attempts as well as open ones is intentional: ``acquire_token``
        is globally unique for the lifetime of the Queue database, so detecting prior
        use must happen before another canonical child Run is allocated.
        """

        ...

    def open_attempt_for_worker(self, worker_id: str) -> QueueAttempt | None:
        """Return the current open attempt already attributed to ``worker_id``.

        Controller may check this before concrete preflight/Run allocation so the v1
        one-open-attempt-per-Worker invariant does not fail only after a new canonical
        ``running`` Run has already been written.
        """

        ...

    def authorized_open_attempt(
        self,
        attempt_id: int,
        lease_token: str,
        *,
        as_of: str,
    ) -> QueueAttempt | None:
        """Return the attempt only while the supplied lease is currently authorized.

        The attempt must still be open, its current ``lease_token`` must match, and its
        one-hour inactivity deadline must remain valid as of ``as_of``. Closed,
        expired/invalidated, or mismatched/superseded leases return ``None`` and do not
        authorize later heartbeat or result acceptance.
        """

        ...

    def select_ready_job(
        self,
        *,
        accepts_training: bool,
        accepts_evaluation: bool,
        as_of: str,
    ) -> QueueJob | None:
        """Select one compatible eligible logical job without creating an attempt.

        ``accepts_training`` and ``accepts_evaluation`` are the Queue-local scheduling
        projection of one Worker acquire request's non-empty ``accepts`` set. At least
        one flag must be true. The Queue must only return a ``READY`` job whose logical
        kind is accepted by the supplied flags; incompatible ready work must not block
        selection of compatible work that is also ready.

        This boundary deliberately uses only Queue-local booleans rather than importing
        Worker API types or introducing a shared JobKind abstraction. The Controller
        integration layer can mechanically map ``"training" in request.accepts`` and
        ``"evaluation" in request.accepts`` to these two arguments.

        Before selecting, the Queue promotes due ``RETRY_WAIT`` jobs whose
        ``retry_not_before <= as_of`` to ``READY``. It defines no v1 priority algorithm
        or fairness ordering beyond the required capability filtering.

        Selection does not reserve a child Run ID, create an attempt, or make the job
        ``ACTIVE``. Controller must keep the dispatch-start path serialized across
        acquire-token/prior-Worker lookup, compatible ready selection, concrete
        preflight, child Run allocation, canonical ``running`` Run persistence, and
        :meth:`activate_attempt`. The Queue adapter must additionally reject an
        activation when the target job is no longer ``READY`` or already has an open
        attempt, so two concurrent pulls cannot both activate the same logical job.
        """

        ...

    def defer_ready_job(
        self,
        job_id: int,
        *,
        retry_not_before: str,
        at: str,
    ) -> QueueJob:
        """Move a selected but unstarted ``READY`` job to ``RETRY_WAIT``.

        This is the no-attempt path for a retryable failure that occurs before child Run
        allocation, including concrete preflight failure. No Run or attempt row is
        created. Retry limits/backoff are not Queue policy here; Controller supplies the
        chosen ``retry_not_before`` value, which may equal ``at`` for immediate retry
        eligibility.
        """

        ...

    def fail_ready_job(self, job_id: int, *, at: str) -> QueueJob:
        """Move a selected but unstarted ``READY`` job to terminal ``FAILED``.

        This is the no-attempt path when pre-Run work fails and orchestration will not
        schedule another attempt. It allocates no Run and creates no attempt row.
        """

        ...

    def activate_attempt(
        self,
        job_id: int,
        run_id: TrainingRunId | EvaluationRunId,
        worker_id: str,
        acquire_token: str,
        lease_token: str,
        *,
        activated_at: str,
    ) -> QueueAttempt:
        """Create the first-class attempt only after Controller allocated its Run.

        Controller calls this only after concrete preflight succeeded, a child Run ID
        was allocated, and the canonical schema-valid ``running`` child Run was
        persisted. Queue never calls a Run allocator itself.

        In one Queue transaction the adapter inserts exactly one new attempt with the
        next positive per-job ``attempt_no`` and marks the job ``ACTIVE``. It must
        reject a target that is not ``READY``, already has an open attempt, whose Run
        kind disagrees with the logical job kind, whose ``worker_id`` already owns an
        open attempt, or whose Run/acquire/lease identity violates Queue uniqueness.
        No successful path may create two open attempts for one job.

        ``activated_at`` is persisted as ``started_at`` and establishes the initial
        lease inactivity deadline exactly one hour later. The returned attempt is the
        committed record; an assignment must not be exposed to Worker before this
        Queue transaction commits.
        """

        ...

    def heartbeat_attempt(
        self,
        attempt_id: int,
        lease_token: str,
        *,
        accepted_at: str,
    ) -> QueueAttempt:
        """Accept one heartbeat for the current open lease and extend its deadline.

        Successful heartbeat requires the attempt to remain open and the supplied
        ``lease_token`` to equal its current token. A closed, expired/invalidated, or
        mismatched lease is rejected and must not be revived. On success the persisted
        ``lease_until`` becomes exactly one hour after ``accepted_at``.

        This operation persists liveness only. It does not mutate canonical child Run
        state and does not implement Worker-side heartbeat cadence or communication
        retry loops.
        """

        ...

    def expired_open_attempts(self, *, as_of: str) -> tuple[QueueAttempt, ...]:
        """Return open attempts whose current one-hour lease is expired as of ``as_of``.

        Detection alone does not close the attempt or terminalize its child Run.
        Controller first resolves the canonical still-running child Run according to
        its lifecycle, then calls the applicable Queue closure operation.
        """

        ...

    def close_attempt(
        self,
        attempt_id: int,
        *,
        target_status: QueueJobStatus,
        finished_at: str,
        retry_not_before: str | None = None,
        close_reason: str | None = None,
    ) -> tuple[QueueJob, QueueAttempt]:
        """Atomically close one open attempt after Controller chose its Queue outcome.

        ``target_status`` is restricted to ``SATISFIED``, ``RETRY_WAIT``, or ``FAILED``.
        Queue does not infer that outcome from Training/Evaluation result objects or Run
        statuses. Controller must first complete the applicable canonical result
        acceptance/child Run terminalization and, for training success, ensure the
        deterministic Model before requesting ``SATISFIED``.

        The attempt receives ``finished_at`` exactly once and is never reopened. The
        owning job changes from ``ACTIVE`` to the requested target in the same Queue
        transaction. ``retry_not_before`` is required exactly for ``RETRY_WAIT`` and
        forbidden for the other two targets; the Queue does not calculate retry limits
        or backoff. When a Training job becomes ``SATISFIED``, its still-``BLOCKED``
        dependent Evaluation jobs are released to ``READY`` in Queue operational state.

        The operation must reject a closed/non-current attempt or any state combination
        that would violate the one-open-attempt and job/attempt lifecycle invariants.
        """

        ...

    def cancel_job(
        self,
        job_id: int,
        *,
        at: str,
        close_reason: str | None = None,
    ) -> QueueJob:
        """Intentionally stop one nonterminal logical job as ``CANCELLED``.

        ``BLOCKED``, ``READY``, and ``RETRY_WAIT`` jobs are marked cancelled directly.
        For ``ACTIVE``, the single open attempt is closed in the same Queue transaction.
        Controller must first resolve/terminalize any corresponding canonical running
        child Run; Queue cancellation is not canonical Run cancellation authority.

        Already ``SATISFIED`` or ``FAILED`` work must not be rewritten as cancelled.
        Repeating cancellation of an already ``CANCELLED`` job may be treated as
        idempotent without altering its prior operational history.
        """

        ...

    def repair_job_state(
        self,
        job_id: int,
        *,
        status: QueueJobStatus,
        retry_not_before: str | None,
        at: str,
    ) -> QueueJob:
        """Apply Controller-selected operational state repair during reconciliation.

        This narrow repair surface exists because immutable Study plan and canonical
        child history outrank stale Queue state. It may repair a job to ``BLOCKED``,
        ``READY``, ``RETRY_WAIT``, ``SATISFIED``, ``FAILED``, or ``CANCELLED`` after
        Controller has established the authoritative facts. ``ACTIVE`` is deliberately
        excluded: only :meth:`activate_attempt` may create active execution authority.

        The operation never changes ``logical`` identity or ``dependency_job_id`` and
        never invents/closes an attempt. Any incompatible open attempt must be resolved
        through attempt inspection/closure before repair. ``retry_not_before`` is
        present exactly for ``RETRY_WAIT``. Repair to ``SATISFIED`` releases blocked
        dependents just as normal satisfying closure does.

        Missing expected logical jobs are restored through :meth:`admit_study_jobs`,
        not by an untyped row-insert API. This keeps reconciliation able to repair
        operational progress without giving Queue authority to rewrite Study intent.
        """

        ...
