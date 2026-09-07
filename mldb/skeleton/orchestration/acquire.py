"""Controller-side Worker acquire orchestration boundary.

This module connects the frozen Worker acquire request/response values, durable Queue
port, immutable Study plan and canonical Run persistence, successful dispatch boundary,
assignment projection, and minimal retry-disposition policy for one logical Worker pull.
It owns only transport-independent acquire orchestration and pre-Run failure disposition.

The public boundary is deliberately one small standalone function. It introduces no
Controller/AcquireManager class, scheduler object, public lock/mutex protocol, token or
clock generator, generic request validator, retry algorithm, Worker polling loop,
heartbeat, candidate upload, outcome handling, cancellation, or reconciliation workflow.
Wave 8-5B/8-5C symbols are not required here.

Acquire replay is reconstructed from persisted Queue authority plus canonical immutable
MLDB state rather than by persisting Worker assignment payloads in Queue. New dispatch
remains cross-store ordered: successful concrete preflight precedes canonical RUNNING
child Run allocation, which precedes Queue attempt activation and assignment exposure.
Only a failure confirmed to have allocated no child Run may be sent through the frozen
preflight-failure retry policy. Any post-allocation gap is preserved for reconciliation.
"""

from __future__ import annotations

from datetime import date

from ..evaluation.preflight import preflight_evaluation
from ..evaluation.run import EvaluationRunStatus
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.run_persistence import (
    read_evaluation_run,
    read_study_plan,
    read_study_run,
    read_training_run,
)
from ..study.run import StudyRunStatus
from ..training.preflight import preflight_training
from ..training.run import TrainingRunStatus
from .asset_source import ImmutableAssetSource
from .assignment import project_evaluation_assignment, project_training_assignment
from .dispatch import dispatch_evaluation_job, dispatch_training_job
from .jobs import EvaluationJob, TrainingJob
from .queue_ports import QueuePort
from .retry_policy import RetryPolicy
from .worker_api import (
    AcquireRejection,
    AcquireWorkRequest,
    AcquireWorkResponse,
    NoWork,
)


def handle_acquire_work(
    request: AcquireWorkRequest,
    allocation_date: date,
    started_at: object,
    as_of: str,
    lease_token: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    assets: ImmutableAssetSource,
    retry_policy: RetryPolicy,
) -> AcquireWorkResponse:
    """Handle one logical Worker acquire operation without owning its transport.

    ``request`` is the frozen :class:`AcquireWorkRequest`. ``allocation_date`` and
    ``started_at`` are passed unchanged to a fresh child Run allocation through the
    successful dispatch boundary. ``as_of`` is the exact Queue timestamp used for
    lease authorization, ready selection, and any pre-Run Queue disposition.
    ``lease_token`` is a caller-supplied opaque token used only when this request really
    creates a new attempt; this function does not generate lease identities or clocks.
    A new-dispatch caller must supply a non-empty token acceptable to the frozen Queue
    contract. Replay never replaces the persisted attempt's lease token with this value.

    The operation returns only the frozen Worker acquire response union. Deterministic
    request/state incompatibilities described below are represented by
    :class:`AcquireRejection`; absence of compatible ready work returns
    :class:`NoWork`. Retryable communication failure is not a response variant under the
    Worker API contract. Infrastructure/persistence failures for which no definitive
    Controller response can safely be established may therefore remain operation
    failures rather than being mislabeled as ``no_work`` or a domain retry decision.

    **Request validation before mutation.** Before any Queue or canonical mutation, the
    handler requires ``request.worker_id`` and ``request.acquire_token`` to be non-empty,
    ``request.accepts`` to be non-empty, and every represented capability to be exactly
    ``"training"`` or ``"evaluation"``. Invalid input returns a definitive rejection;
    no generic validator framework is introduced. After this validation, acquire-token
    replay lookup is the first orchestration lookup that can decide whether new work may
    begin. The caller-supplied ``lease_token`` is not an acquire-request field and is
    never allowed to alter an already-persisted replay attempt.

    **Acquire-token replay comes first.** The handler first calls
    ``queue.attempt_by_acquire_token(request.acquire_token)`` before checking another
    Worker attempt, selecting READY work, performing concrete preflight, or allocating a
    Run. If a retained attempt exists, this acquire token is already consumed and may
    never create another child Run.

    A returned attempt whose ``finished_at`` is non-``None`` is historical/closed and
    causes a definitive rejection. An open attempt is replayable only when all of the
    following remain true:

    - ``attempt.worker_id == request.worker_id``;
    - ``queue.job_by_id(attempt.job_id)`` returns its parent Queue job;
    - that parent logical kind is included in ``request.accepts``;
    - ``queue.authorized_open_attempt(attempt.attempt_id, attempt.lease_token,
      as_of=as_of)`` returns that same concrete attempt identity rather than ``None`` or
      another authorization.

    Expired, stale, superseded, missing-parent, wrong-Worker, wrong-capability, or closed
    replay state is definitively rejected. It is never repaired by allocating a fresh
    Run under the same acquire token. The authorized attempt returned by the lease check
    is the attempt used for replay projection so its current persisted lease deadline is
    authoritative.

    **Replay assignment reconstruction.** Replay must never call
    :func:`dispatch_training_job` or :func:`dispatch_evaluation_job`, because those are
    fresh Run-allocation boundaries. Instead, the parent Queue job's concrete logical
    type selects a typed canonical read using the persisted ``attempt.run_id``:

    - :class:`TrainingJob` -> :func:`read_training_run`;
    - :class:`EvaluationJob` -> :func:`read_evaluation_run`.

    The canonical child must still be ``RUNNING``, its exact ``id`` must equal the
    attempt's ``run_id``, and its frozen Study lineage must exactly agree with the parent
    Queue coordinate. Training requires the Run lineage ``study.run`` and
    ``study.trial`` to equal the Queue Training coordinate. Evaluation additionally
    requires exact ``study.stage`` agreement. Missing lineage, wrong lineage, wrong Run
    kind/status, or any identity disagreement is a definitive replay incompatibility;
    no replacement attempt is created.

    Training replay re-runs :func:`preflight_training` only from the canonical RUNNING
    Run's persisted exact inputs::

        preflight_training(
            run.corpus,
            run.architecture,
            run.train_protocol,
            run.execution.seed,
            run.parameters,
            layout,
            filesystem,
        )

    Evaluation replay likewise re-runs :func:`preflight_evaluation` only from the
    canonical RUNNING Run's persisted exact inputs::

        preflight_evaluation(
            run.model,
            run.corpus,
            run.evaluation_protocol,
            run.parameters,
            layout,
            filesystem,
        )

    Fresh preflight must reproduce the persisted Run execution identity and complete
    parameter mapping exactly. The handler then calls
    :func:`project_training_assignment` or :func:`project_evaluation_assignment` with
    that fresh preflight, the exact canonical RUNNING Run, the still-authorized
    persisted attempt, and ``assets``. The frozen projection boundary performs its own
    exact Run/preflight/attempt checks and regenerates replay-stable immutable asset
    descriptors. Queue receives no Study-plan execution payload and no stored assignment
    response is introduced. Replay reconstruction failure is not a READY-job preflight
    failure and must not mutate the already-active job through ``RetryPolicy``.

    **One-open-attempt Worker rule.** Only when no acquire-token replay row exists does
    the handler call ``queue.open_attempt_for_worker(request.worker_id)``. Any returned
    open attempt means this Worker already owns another active attempt under v1 and the
    new logical acquire is definitively rejected before ready selection or Run
    allocation. This handler does not expire, close, adopt, or reconcile that attempt.

    **Compatible READY selection.** Capability mapping is purely mechanical::

        accepts_training = "training" in request.accepts
        accepts_evaluation = "evaluation" in request.accepts

    The handler passes those flags and ``as_of`` to ``queue.select_ready_job``. A
    ``None`` result returns :class:`NoWork`; that is a successful definitive acquire
    response, not communication retry. A non-``None`` result must be a READY Queue job
    whose concrete logical type agrees with the accepted capability selected by Queue.

    **Study authority before fresh dispatch.** From the selected job's logical
    coordinate, the handler obtains the owning Study Run ID and reads both canonical
    Study authority surfaces through :func:`read_study_run` and
    :func:`read_study_plan`. Fresh child work is permitted only while the exact Study
    Run is :attr:`StudyRunStatus.RUNNING` and its finalized immutable plan remains
    readable/valid. A terminal Study Run must never start another child Run. This
    acquire boundary does not invent Queue cancellation/cleanup for such stale work;
    the request is definitively rejected and later cancellation/reconciliation owns
    repair. No terminal Study state is converted into a retry-policy event.

    **Successful fresh dispatch.** The selected logical type chooses exactly one frozen
    success boundary and supplies the same authoritative finalized plan::

        dispatch_training_job(...)

    or::

        dispatch_evaluation_job(...)

    The call passes ``request.worker_id``, ``request.acquire_token``, the caller-supplied
    ``lease_token``, and ``activated_at=as_of`` unchanged together with
    ``allocation_date``, ``started_at``, ``layout``, ``filesystem``, ``queue``, and
    ``assets``. No local reimplementation may reorder the frozen success path:
    concrete preflight -> canonical RUNNING child Run allocation/persistence -> Queue
    attempt activation -> assignment projection. Successful return is the assignment
    produced by that dispatch boundary.

    **Pre-Run failure disposition only.** A concrete preflight or selected
    plan-coordinate legality failure may enter the frozen retry policy only when the
    acquire implementation can establish that *no child Run was allocated for this
    dispatch attempt*. In that confirmed pre-Run case it reads existing operational
    history exactly once through ``queue.attempts_for_job(job.job_id)`` and calls::

        decision = retry_policy.after_preflight_failure(
            job,
            attempts,
            at=as_of,
        )

    ``decision.retry_not_before is not None`` maps mechanically to::

        queue.defer_ready_job(
            job.job_id,
            retry_not_before=decision.retry_not_before,
            at=as_of,
        )

    while ``None`` maps to ``queue.fail_ready_job(job.job_id, at=as_of)``. The acquire
    itself may then return a definitive rejection. This policy decision concerns future
    domain attempts only; it does not control the Worker's 10-second communication
    retry semantics.

    The implementation must not use a broad ``except`` around ``dispatch_*`` and assume
    every exception is preflight failure. Because Queue SQLite and canonical Run storage
    are separate stores, a dispatch failure can occur after canonical RUNNING Run commit
    but before Queue activation, or after Queue activation during assignment projection.
    Those post-allocation failures must not call ``after_preflight_failure()``,
    ``defer_ready_job()``, or ``fail_ready_job()`` and must not roll back canonical Run
    history. They remain cross-store/replay gaps for later reconciliation. No new public
    exception taxonomy is required: an implementation may use private phase bookkeeping,
    private exceptions, or authoritative before/after canonical inspection, but it may
    apply pre-Run policy only when absence of a newly allocated child Run is actually
    established. Ambiguous failure is therefore treated as post-allocation/reconciliation
    territory rather than risk rewriting a READY job after canonical allocation.

    **Dispatch-start serialization.** V1 Controller dispatch start is one semantic
    implementation-private critical section spanning acquire-token lookup, existing
    Worker lookup, compatible READY selection, Study/plan confirmation, concrete fresh
    dispatch, and Queue activation. This prevents two concurrent pulls from beginning
    the same selected READY work while preserving ``QueuePort.activate_attempt``
    validation as the final storage-side defense. The serialization primitive is not
    part of this public signature: no Lock/Mutex Protocol, Controller class, manager, or
    generic transaction coordinator is added here. Replay reconstruction reuses
    persisted authority and never turns the caller's fresh ``lease_token`` into a new
    activation.

    Heartbeat, immutable-asset retrieval transport, candidate upload, report-outcome,
    intentional cancellation, reconciliation, retry algorithm/backoff, and Worker
    polling cadence remain outside this operation.
    """

    ...
