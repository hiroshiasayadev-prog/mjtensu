"""Controller-side Study Run cancellation-request orchestration boundary.

This module fixes the transport-independent operation behind public Study cancellation.
Cancellation is durable Study-level intent first and replaceable Queue cleanup second:
a canonical ``RUNNING`` Study Run becomes ``CANCELLED`` before any non-active Queue job
is cancelled. That ordering makes the frozen acquire boundary refuse fresh child work
even across a Controller crash between the canonical and Queue stores.

Active Worker attempts are deliberately preserved by this request boundary. Their
canonical child Runs remain ``RUNNING`` and their Queue jobs remain ``ACTIVE`` so the
later heartbeat/control operation can report cooperative ``cancel_requested`` state.
The Worker may then stop and report ``AttemptCancelled`` through the already-frozen
outcome path, which owns child Run terminalization and ACTIVE Queue cancellation.

The boundary introduces no cancellation manager/token, Worker registry, signal bus,
RetryPolicy dependency, child Run finalization, heartbeat implementation, lease expiry,
forced process control, Study restart, public HTTP/CLI DTO, or cross-store transaction.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from ..common.ids import StudyRunId
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.run_persistence import (
    persist_study_run_transition,
    read_study_run,
)
from ..study.run import StudyRun, StudyRunExecution, StudyRunStatus
from .queue import QueueJobStatus
from .queue_ports import QueuePort


StudyCancellationResult: TypeAlias = Literal["accepted", "already_terminal"]
"""Definitive result of one Study-level cancellation request.

``"accepted"`` means this call transitioned the canonical Study Run from ``RUNNING``
to ``CANCELLED`` and then attempted the required non-active Queue propagation before
returning successfully. Active attempts need not have stopped yet.

``"already_terminal"`` means the Study Run was already terminal when this operation
began. For an already-``CANCELLED`` Study Run, the operation still performs the safe
lagging non-active Queue propagation before returning this result, closing the
canonical-first/Queue-second crash gap. Other terminal Study statuses are returned
unchanged and are not reinterpreted as cancellation intent.
"""


def request_study_run_cancellation(
    study_run_id: StudyRunId,
    finished_at: object,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
) -> StudyCancellationResult:
    """Record one Study cancellation intent and propagate it to non-active Queue work.

    ``study_run_id`` is the only execution identity supplied by the caller. No Queue
    job ID, attempt ID, Worker ID, lease token, Training Run ID, or Evaluation Run ID is
    accepted by this operation. ``finished_at`` is the caller-supplied opaque timestamp
    for the canonical Study Run terminal transition. ``queue_at`` is passed unchanged
    to Queue cancellation and therefore uses the frozen Queue timestamp encoding. This
    boundary introduces no clock or timestamp-normalization service.

    The operation first reads the exact canonical Study Run through
    :func:`read_study_run` and branches only on its canonical lifecycle state.

    **Canonical RUNNING -> CANCELLED first.** For
    :attr:`StudyRunStatus.RUNNING`, construct exactly one terminal :class:`StudyRun`::

        StudyRun(
            schema=current.schema,
            id=current.id,
            status=StudyRunStatus.CANCELLED,
            study=current.study,
            execution=StudyRunExecution(
                started_at=current.execution.started_at,
                finished_at=finished_at,
            ),
            plan=current.plan,
            summary=None,
        )

    The value preserves the canonical schema, Study Run identity, referenced Study,
    execution start fact, and exact finalized plan metadata when such metadata already
    exists. When plan materialization never completed, ``current.plan is None`` remains
    ``None``; cancellation does not inspect partial ``plan.jsonl`` bytes or fabricate
    plan integrity/count metadata. When a complete plan was finalized, the exact
    :class:`~mldb.skeleton.study.run.StudyRunPlan` value is retained unchanged as
    required by the frozen Study Run persistence contract.

    ``summary`` is deliberately set to ``None``. Cancellation does not scan Training or
    Evaluation history merely to manufacture final coordinate counts, and it does not
    preserve a potentially provisional running summary as though cancellation made it
    authoritative. The frozen Study Run format explicitly permits terminal
    ``CANCELLED`` with no summary. No Study-level failure text exists in v1 and none is
    invented.

    The constructed terminal value must be persisted through
    :func:`persist_study_run_transition` *before* any Queue cancellation propagation.
    This ordering is the cancellation authority boundary. Once that canonical write is
    durable, the frozen acquire handler already refuses fresh child dispatch for this
    Study Run even if Controller crashes before Queue cleanup completes. Queue state is
    never used as the sole durable cancellation intent.

    Cancellation must also participate in the Controller implementation's same private
    dispatch-start serialization boundary that the frozen acquire handler uses around
    Study authority confirmation and fresh dispatch. For the first ``RUNNING`` request,
    that private exclusion spans the canonical read/transition and the subsequent
    non-active Queue propagation. Otherwise an acquire that observed ``RUNNING``
    immediately before this transition could allocate a fresh child after cancellation
    committed, or activate a job between Queue inspection and cancellation. No
    Lock/Mutex Protocol is added to this public signature; the serialization mechanism
    remains an implementation-private Controller concern. The required externally
    visible invariant is only that no fresh dispatch crosses a successfully committed
    Study cancellation boundary.

    **Queue propagation after canonical cancellation.** After the canonical transition
    has committed, read retained Queue jobs only through
    ``queue.jobs_for_study_run(study_run_id)``. For every job whose status is exactly
    :attr:`QueueJobStatus.BLOCKED`, :attr:`QueueJobStatus.READY`, or
    :attr:`QueueJobStatus.RETRY_WAIT`, call::

        queue.cancel_job(job.job_id, at=queue_at)

    No :class:`~mldb.skeleton.orchestration.retry_policy.RetryPolicy` is consulted.
    Intentional Study cancellation is not a retry decision. Already
    :attr:`QueueJobStatus.SATISFIED`, :attr:`QueueJobStatus.FAILED`, or
    :attr:`QueueJobStatus.CANCELLED` rows are retained as existing terminal operational
    facts and need not be rewritten for cosmetic uniformity.

    **ACTIVE is intentionally left intact.** A retained
    :attr:`QueueJobStatus.ACTIVE` job is not passed to ``queue.cancel_job`` here. The
    frozen Queue contract would close its open attempt, which would defeat the Worker
    API's cooperative heartbeat cancellation path. This operation therefore does not
    call ``cancel_training_run()``, ``cancel_evaluation_run()``, ``close_attempt()``, or
    any ACTIVE Queue mutation merely because the Study cancellation request arrived.

    After the canonical Study Run is ``CANCELLED``, a later heartbeat/control handler
    can still authorize the existing active lease and derive
    ``cancel_requested=True`` from Study authority. The Worker may cooperatively stop
    and report ``AttemptCancelled``. The frozen outcome handler then observes the
    canonical cancelled Study, terminalizes the still-running child Run, and invokes
    ``queue.cancel_job`` for that ACTIVE logical job. Lease expiry or Worker-loss
    cleanup remains a separate operational recovery concern.

    A successful first request returns ``"accepted"`` only after the canonical
    transition and all requested non-active Queue cancellations performed by this call
    have completed. It does not wait for active Workers to stop.

    **Idempotent cancelled replay and crash-gap repair.** If the initial canonical read
    already returns :attr:`StudyRunStatus.CANCELLED`, do not call
    :func:`persist_study_run_transition` again because terminal Study Runs are immutable.
    Do not return immediately either. Re-read/inspect retained Queue jobs through the
    same ``jobs_for_study_run`` surface and apply the same cancellation only to
    ``BLOCKED``/``READY``/``RETRY_WAIT`` rows, still leaving ``ACTIVE`` intact. This
    repairs the supported crash sequence where canonical cancellation committed but
    Queue propagation did not. After that safe lagging propagation completes, return
    ``"already_terminal"``.

    **Other terminal Study states.** If the canonical Study Run is already
    :attr:`StudyRunStatus.COMPLETED`,
    :attr:`StudyRunStatus.COMPLETED_WITH_FAILURES`, or
    :attr:`StudyRunStatus.FAILED`, return ``"already_terminal"`` without rewriting the
    Study Run to ``CANCELLED`` and without propagating cancellation into Queue rows.
    Those states do not establish Study-cancellation intent, and this operation must not
    mutate their historical or operational facts merely because a late cancellation
    request arrived.

    Infrastructure or persistence failures that prevent the canonical transition or
    required non-active Queue propagation from completing are operation failures rather
    than a fabricated definitive result. Because canonical state is written first, a
    retry after a Queue-side failure observes ``CANCELLED`` and safely finishes the
    lagging Queue propagation before returning ``"already_terminal"``. No cross-store
    transaction or rollback is introduced.

    Heartbeat response construction, lease authorization/extension, Worker process
    control, candidate/result acceptance, child retry, Study restart, lease-expiry
    recovery, and public transport DTO mapping remain outside this Wave.
    """


    from ._coordination import orchestration_exclusion

    with orchestration_exclusion():
        current = read_study_run(study_run_id, layout, filesystem)

        if current.status is StudyRunStatus.RUNNING:
            cancelled = StudyRun(
                schema=current.schema,
                id=current.id,
                status=StudyRunStatus.CANCELLED,
                study=current.study,
                execution=StudyRunExecution(
                    started_at=current.execution.started_at,
                    finished_at=finished_at,
                ),
                plan=current.plan,
                summary=None,
            )
            persist_study_run_transition(cancelled, layout, filesystem)
            _cancel_non_active_jobs(study_run_id, queue_at, queue)
            return "accepted"

        if current.status is StudyRunStatus.CANCELLED:
            _cancel_non_active_jobs(study_run_id, queue_at, queue)
            return "already_terminal"

        if current.status in {
            StudyRunStatus.COMPLETED,
            StudyRunStatus.COMPLETED_WITH_FAILURES,
            StudyRunStatus.FAILED,
        }:
            return "already_terminal"

        raise RuntimeError(f"Unsupported Study Run status: {current.status!r}")


def _cancel_non_active_jobs(
    study_run_id: StudyRunId,
    queue_at: str,
    queue: QueuePort,
) -> None:
    for job in queue.jobs_for_study_run(study_run_id):
        if job.status in {
            QueueJobStatus.BLOCKED,
            QueueJobStatus.READY,
            QueueJobStatus.RETRY_WAIT,
        }:
            queue.cancel_job(job.job_id, at=queue_at)
