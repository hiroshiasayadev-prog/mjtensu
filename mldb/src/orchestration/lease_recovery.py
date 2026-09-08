"""Controller recovery boundary for expired retained Queue attempts.

This module fixes the narrow recovery operation for Queue attempts whose retained
one-hour Worker lease has expired. It connects Queue expiry discovery, exact parent-job
authority, typed canonical child Run truth, deterministic Model repair for completed
training, the frozen retry-policy seam, canonical Study cancellation authority, and
Queue attempt/job disposition.

This is not general reconciliation and does not rebuild Queue state from Study plans or
canonical lineage. It does not scan Workers, define Worker liveness, kill Worker
processes, perform acquire/heartbeat/outcome handling, choose a retry algorithm, finalize
a Study Run, synthesize replacement attempts, or introduce a LeaseManager/Sweeper-style
framework.

The public Controller signature remains small and side-effect-only. Heartbeat/outcome
races are closed by implementation-private Controller serialization around concrete
attempt authority, followed by a fresh Queue recheck inside that exclusion. The frozen
:class:`QueuePort` therefore needs no recovery-claim mutation, lease generation, or new
persisted field. The exclusion mechanism is deliberately not frozen and is never exposed
as a public Lock/Mutex argument.
"""

from __future__ import annotations

from ..evaluation.run import EvaluationRunStatus
from ..model.persistence import ensure_model_for_completed_training_run
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.run_finalization import fail_evaluation_run, fail_training_run
from ..runtime.run_persistence import (
    read_evaluation_run,
    read_study_run,
    read_training_run,
)
from ..study.run import StudyRunStatus
from ..training.run import TrainingRunStatus
from .jobs import EvaluationJob, TrainingJob
from .queue import QueueJobStatus
from .queue_ports import QueuePort
from .retry_policy import RetryPolicy


def recover_expired_attempts(
    *,
    as_of: str,
    run_finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> None:
    """Recover retained expired attempts from canonical child and Study authority.

    ``as_of`` is the exact v1 Queue timestamp used for expiry discovery, retry-policy
    input, and the resulting Queue close/cancel timestamps. ``run_finished_at`` is the
    opaque caller-supplied timestamp passed unchanged to canonical unsuccessful child
    terminalization. This operation derives neither clock value and returns no result
    DTO or diagnostic count: successful completion is side-effect-only.

    **Discovery is Queue-only and exact.** The operation discovers work only through::

        queue.expired_open_attempts(as_of=as_of)

    It must not scan Worker processes, reconstruct a Worker registry, infer liveness
    from filesystem/candidate activity, enumerate all Queue jobs, or perform startup
    Queue reconstruction. Each returned value is only an observation that the retained
    attempt was open and its current one-hour lease was expired at that discovery point.

    **Private same-attempt authority exclusion precedes canonical resolution.** For each
    discovery item, Controller must first enter an implementation-private exclusion for
    that concrete attempt before deciding that Worker authority has been lost. The same
    exclusion must cover concurrent operations that can rely on or extend that old lease
    authority, in particular heartbeat authorization/update and Worker outcome
    authorization followed by canonical child/result mutation and its Queue disposition.
    It must compose with the existing same-Study serialization used by acquire/dispatch,
    Study cancellation, unsatisfied outcome disposition, and reconciliation maintenance.
    A Controller-wide orchestration serialization, correctly composed per-Study plus
    per-attempt serialization, or another private equivalent is sufficient. This module
    does not expose or freeze a Lock/Mutex Protocol, lock ordering API, generation, claim
    token, or transaction coordinator.

    While that concrete-attempt exclusion is held, re-read exactly::

        fresh = queue.attempt_by_id(discovered.attempt_id)

    Recovery may continue only for the same retained authority represented by the
    discovery item. ``fresh`` must exist, remain open (``finished_at is None``), and have
    the same ``job_id``, ``run_id``, and ``lease_token`` as the discovered attempt. Its
    parent must still have ``status is QueueJobStatus.ACTIVE``, and
    ``queue.open_attempt_for_job(fresh.job_id)`` must still identify this same open
    attempt. An already closed/resolved discovery item is skipped and is never reopened
    or recovered again. Missing or contradictory open/current identity is not authority
    to terminalize a child; an established lifecycle inconsistency remains an operation
    failure under the neighboring orchestration semantics.

    Only after those exact fresh-row/current-ACTIVE checks, re-check current lease
    authorization inside the same exclusion::

        authorized = queue.authorized_open_attempt(
            fresh.attempt_id,
            fresh.lease_token,
            as_of=as_of,
        )

    If ``authorized`` returns this attempt, its lease is still authorized at the recovery
    cutoff and this discovery item is skipped: recovery must not terminalize its child or
    close its Queue attempt. If the exact fresh attempt remains open/current but the
    authorization call returns ``None``, it may be treated as expired/unrecoverable for
    this recovery pass because heartbeat/outcome authority mutation for this concrete
    attempt remains excluded until recovery finishes canonical resolution and Queue
    disposition. The original sweep snapshot alone is never sufficient authority.

    This closes the heartbeat race without a new Queue mutation. If heartbeat obtains
    the exclusion first, its authorization/update commits and releases; recovery then
    enters, observes the fresh extended attempt, and skips it when that lease is
    authorized at ``as_of``. If recovery obtains the exclusion first, its fresh checks
    establish that the same current attempt is expired/unrecoverable, it resolves the
    canonical child and closes/cancels the Queue work before releasing; a later heartbeat
    then sees closed/stale Queue authority and must be rejected rather than reviving the
    lease. A heartbeat carrying an earlier ``accepted_at`` does not escape this ordering:
    it either commits its extension before recovery's fresh check or performs its Queue
    authorization/update only after recovery has closed the attempt.

    The same exclusion closes the late-outcome race. An outcome that enters first keeps
    its currently authorized attempt authority through canonical acceptance or
    terminalization and the corresponding Queue disposition before recovery may inspect
    that attempt. Recovery that enters first owns the expired attempt through canonical
    child resolution and Queue disposition; a later outcome can no longer authorize that
    open lease and may only follow the frozen historical replay/rejection semantics. Thus
    recovery never concurrently changes a ``RUNNING`` child to ``FAILED`` while a
    previously authorized Worker outcome is canonicalizing success/failure/cancellation
    for that same attempt.

    **Fresh parent authority selects the child kind.** After the exact fresh attempt has
    been established as expired/unrecoverable inside the exclusion, resolve only::

        fresh.job_id
          -> queue.job_by_id(fresh.job_id)

    The parent must be the same current ACTIVE job established above. A missing parent or
    an established Queue identity/lifecycle inconsistency is an operation failure. The
    concrete parent logical type, never the textual shape of ``fresh.run_id``, selects
    the canonical typed read:

    - :class:`TrainingJob` -> :func:`read_training_run`;
    - :class:`EvaluationJob` -> :func:`read_evaluation_run`.

    The retained attempt's concrete Run identity must agree with that parent kind and
    the canonical child returned by the typed read. This boundary does not parse a Run
    ID string to infer kind and does not create a replacement child Run.

    **Expired RUNNING child becomes Controller-side failure.** Once the same attempt has
    been authoritatively established as expired/unrecoverable inside the exclusion, a
    canonical child still in ``RUNNING`` has lost Worker authority. Recovery
    terminalizes exactly that child, preserving the frozen child lifecycle and inventing
    no Worker failure report:

        fail_training_run(
            run,
            run_finished_at,
            layout,
            filesystem,
            failure=None,
        )

    or::

        fail_evaluation_run(
            run,
            run_finished_at,
            layout,
            filesystem,
            failure=None,
        )

    Canonical failure commits before the logical Queue retry/failure/cancellation
    disposition. Lease expiry is Controller operational recovery, not an accepted
    Worker :class:`AttemptFailed`, so no failure ``type``/``message`` is synthesized.
    The resulting canonical fact is exactly ``FAILED(failure=None)``.

    **Already-terminal children are canonical truth and are never rewritten.** A stale
    Queue ACTIVE/open projection may lag an earlier canonical child transition. Recovery
    handles the existing terminal child directly rather than calling another Run
    terminalization boundary.

    For Training:

    - ``COMPLETED`` is satisfying only after
      :func:`ensure_model_for_completed_training_run` succeeds for that exact canonical
      Run; only then close this retained attempt to
      :attr:`QueueJobStatus.SATISFIED`;
    - ``FAILED`` is unsatisfied with policy outcome ``"failed"``;
    - ``CANCELLED`` is unsatisfied with policy outcome ``"cancelled"``.

    For Evaluation:

    - ``COMPLETED`` closes this retained attempt to
      :attr:`QueueJobStatus.SATISFIED`;
    - ``COMPLETED_PARTIAL`` is unsatisfied with policy outcome
      ``"completed_partial"``;
    - ``FAILED`` is unsatisfied with policy outcome ``"failed"``;
    - ``CANCELLED`` is unsatisfied with policy outcome ``"cancelled"``.

    Training ``COMPLETED`` must never be marked Queue-satisfied before deterministic
    Model ensure. Failure of Model ensure remains an operation failure; it is not
    permission to rewrite the completed Training Run to ``FAILED``. No terminal child
    state is reopened or normalized merely to match stale Queue state.

    Satisfying canonical truth is mechanically repaired with the existing current
    attempt only::

        queue.close_attempt(
            fresh.attempt_id,
            target_status=QueueJobStatus.SATISFIED,
            finished_at=as_of,
        )

    This module invents no ``close_reason`` vocabulary. Training satisfaction retains
    the Queue contract's normal dependent-release semantics.

    **Every unsatisfied child uses fresh Study authority.** After canonical unsatisfied
    child truth has been established (including a ``RUNNING`` child failed here or an
    already-terminal ``FAILED``/``CANCELLED``/``COMPLETED_PARTIAL`` child), derive the
    owning Study only from ``job.logical.coordinate.study_run`` and freshly call
    :func:`read_study_run`. Do not reuse a Study snapshot from before child
    terminalization and do not infer Study cancellation from child cancellation.

    The fresh Study authority read plus resulting unsatisfied Queue disposition must
    participate in the same implementation-private Controller orchestration
    serialization used by acquire/outcome/Study cancellation and compatible with
    reconciliation maintenance exclusion. No public Lock/Mutex/transaction argument is
    added here. At minimum, Study cancellation must not be able to commit durable
    ``CANCELLED`` and finish its propagation while this recovery subsequently creates a
    new ``RETRY_WAIT`` from an older ``RUNNING`` Study observation.

    For :attr:`StudyRunStatus.RUNNING`, and only then, call exactly::

        decision = retry_policy.after_unsatisfied_attempt(
            job,
            queue.attempts_for_job(job.job_id),
            outcome,
            at=as_of,
        )

    where ``outcome`` is the canonical unsatisfied category described above. A permitted
    retry maps mechanically to::

        queue.close_attempt(
            fresh.attempt_id,
            target_status=QueueJobStatus.RETRY_WAIT,
            retry_not_before=decision.retry_not_before,
            finished_at=as_of,
        )

    and ``decision.retry_not_before is None`` maps to::

        queue.close_attempt(
            fresh.attempt_id,
            target_status=QueueJobStatus.FAILED,
            finished_at=as_of,
        )

    Retry creates no Run and no attempt here. A future acquire of the logical
    ``RETRY_WAIT``/later ``READY`` job allocates a fresh child Run and fresh Queue
    attempt under the already-frozen dispatch contract.

    For :attr:`StudyRunStatus.CANCELLED`, do not call RetryPolicy. Intentional Study
    cancellation already owns the no-more-work decision. After any necessary truthful
    child failure has committed, close/cancel the ACTIVE logical work only through::

        queue.cancel_job(job.job_id, at=as_of)

    A lease-expired child failed here remains canonical ``FAILED(failure=None)`` rather
    than becoming ``CANCELLED``: no cooperative Worker cancellation outcome was
    accepted. An already canonical child ``CANCELLED`` also remains child-cancelled;
    Study cancellation controls only the logical Queue disposition.

    Canonical :attr:`StudyRunStatus.COMPLETED`,
    :attr:`StudyRunStatus.COMPLETED_WITH_FAILURES`, or
    :attr:`StudyRunStatus.FAILED` together with this still-open expired ACTIVE attempt
    and an unsatisfied child is an orchestration/cross-store inconsistency. Recovery
    must not call RetryPolicy, create another attempt, reinterpret the Study state, or
    invent a recovery result DTO. The inconsistency remains an operation failure under
    the same existing failure semantics used by neighboring Controller operations.

    **Cancelled-child nuance.** A canonical child ``CANCELLED`` does not prove Study
    cancellation. When the fresh Study is still ``RUNNING``, it enters RetryPolicy with
    ``outcome="cancelled"`` and becomes ``RETRY_WAIT`` or ``FAILED``. Only canonical
    Study ``CANCELLED`` suppresses policy and maps the logical job to Queue
    ``CANCELLED``.

    **Partial Evaluation nuance.** Canonical Evaluation ``COMPLETED_PARTIAL`` remains
    useful terminal history but does not satisfy its logical job. A fresh ``RUNNING``
    Study uses RetryPolicy with ``outcome="completed_partial"``; a fresh ``CANCELLED``
    Study bypasses policy and cancels the logical Queue job. The Evaluation Run itself
    is never rewritten.

    **Outcome replay remains distinguishable.** The Controller-generated
    ``FAILED(failure=None)`` used for lease-loss recovery deliberately does not prove
    that an exact Worker ``AttemptFailed`` outcome was previously accepted. The frozen
    outcome handler may therefore reject a later Worker failure report whose exact
    failure metadata cannot be replay-proven, while reconciliation/recovery can still
    use the terminal ``FAILED`` child as sufficient canonical truth to repair Queue
    state. This preserves the existing outcome acknowledgement invariant rather than
    smuggling lease expiry into Worker failure metadata.

    **Boundary from reconciliation.** This operation is a retained-expired-attempt
    repair only. It does not derive jobs from a Study plan, admit missing jobs, list all
    canonical child lineage, repair arbitrary non-ACTIVE Queue projection, recover a
    Queue database loss, decide blocked/readiness state, resolve duplicate canonical
    children, or finalize a Study. General reconciliation remains the separate
    maintenance-excluded boundary for those responsibilities. Recovery here uses the
    expired attempt that already exists and never synthesizes an attempt for historical
    or orphaned work.

    Infrastructure, repository, Queue, Model-ensure, serialization, or authority
    failures are operation failures. They are not converted into a count/result object
    and must not be hidden by continuing as though the affected expired attempt had been
    safely recovered.
    """


    from ._coordination import orchestration_exclusion

    discovered_attempts = queue.expired_open_attempts(as_of=as_of)
    for discovered in discovered_attempts:
        with orchestration_exclusion():
            fresh = queue.attempt_by_id(discovered.attempt_id)
            if fresh is None:
                raise RuntimeError("Expired discovery attempt disappeared from Queue history.")
            if fresh.finished_at is not None:
                continue
            if not _same_discovered_authority(discovered, fresh):
                raise RuntimeError("Expired discovery disagrees with fresh attempt authority.")

            job = queue.job_by_id(fresh.job_id)
            if job is None:
                raise RuntimeError("Expired open attempt has no parent Queue job.")
            if job.status is not QueueJobStatus.ACTIVE:
                raise RuntimeError("Expired open attempt parent is not ACTIVE.")

            current = queue.open_attempt_for_job(fresh.job_id)
            if current is None:
                raise RuntimeError("ACTIVE expired job has no current open attempt.")
            if not _same_current_attempt(fresh, current):
                raise RuntimeError("Expired attempt is not the current open attempt for its job.")

            authorized = queue.authorized_open_attempt(
                fresh.attempt_id,
                fresh.lease_token,
                as_of=as_of,
            )
            if authorized is not None:
                if not _same_current_attempt(fresh, authorized):
                    raise RuntimeError("Lease authorization returned contradictory attempt authority.")
                continue

            run = _read_child(job, fresh.run_id, layout, filesystem)
            if isinstance(job.logical, TrainingJob):
                _recover_training(
                    run,
                    job,
                    fresh.attempt_id,
                    as_of,
                    run_finished_at,
                    layout,
                    filesystem,
                    queue,
                    retry_policy,
                )
            elif isinstance(job.logical, EvaluationJob):
                _recover_evaluation(
                    run,
                    job,
                    fresh.attempt_id,
                    as_of,
                    run_finished_at,
                    layout,
                    filesystem,
                    queue,
                    retry_policy,
                )
            else:
                raise RuntimeError("Expired attempt parent has unsupported logical kind.")


def _same_discovered_authority(discovered: object, fresh: object) -> bool:
    return (
        fresh.attempt_id == discovered.attempt_id
        and fresh.job_id == discovered.job_id
        and fresh.run_id == discovered.run_id
        and fresh.lease_token == discovered.lease_token
    )


def _same_current_attempt(expected: object, actual: object) -> bool:
    return (
        actual.attempt_id == expected.attempt_id
        and actual.job_id == expected.job_id
        and actual.run_id == expected.run_id
        and actual.lease_token == expected.lease_token
        and actual.finished_at is None
    )


def _read_child(
    job: object,
    run_id: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> object:
    if isinstance(job.logical, TrainingJob):
        run = read_training_run(run_id, layout, filesystem)
    elif isinstance(job.logical, EvaluationJob):
        run = read_evaluation_run(run_id, layout, filesystem)
    else:
        raise RuntimeError("Queue parent has unsupported logical kind.")
    if run.id != run_id:
        raise RuntimeError("Canonical child identity disagrees with Queue attempt.")
    return run


def _recover_training(
    run: object,
    job: object,
    attempt_id: int,
    as_of: str,
    run_finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> None:
    if run.status is TrainingRunStatus.RUNNING:
        fail_training_run(
            run,
            run_finished_at,
            layout,
            filesystem,
            failure=None,
        )
        _dispose_unsatisfied(
            job,
            attempt_id,
            "failed",
            as_of,
            layout,
            filesystem,
            queue,
            retry_policy,
        )
        return

    if run.status is TrainingRunStatus.COMPLETED:
        ensure_model_for_completed_training_run(run, layout, filesystem)
        queue.close_attempt(
            attempt_id,
            target_status=QueueJobStatus.SATISFIED,
            finished_at=as_of,
         )
        return

    if run.status is TrainingRunStatus.FAILED:
        _dispose_unsatisfied(
            job,
            attempt_id,
            "failed",
            as_of,
            layout,
            filesystem,
            queue,
            retry_policy,
        )
        return

    if run.status is TrainingRunStatus.CANCELLED:
        _dispose_unsatisfied(
            job,
            attempt_id,
            "cancelled",
            as_of,
            layout,
            filesystem,
            queue,
            retry_policy,
        )
        return

    raise RuntimeError(f"Unsupported Training Run status: {run.status!r}")


def _recover_evaluation(
    run: object,
    job: object,
    attempt_id: int,
    as_of: str,
    run_finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> None:
    if run.status is EvaluationRunStatus.RUNNING:
        fail_evaluation_run(
            run,
            run_finished_at,
            layout,
            filesystem,
            failure=None,
        )
        _dispose_unsatisfied(
            job,
            attempt_id,
            "failed",
            as_of,
            layout,
            filesystem,
            queue,
            retry_policy,
        )
        return

    if run.status is EvaluationRunStatus.COMPLETED:
        queue.close_attempt(
            attempt_id,
            target_status=QueueJobStatus.SATISFIED,
            finished_at=as_of,
        )
        return

    if run.status is EvaluationRunStatus.COMPLETED_PARTIAL:
        _dispose_unsatisfied(
            job,
            attempt_id,
            "completed_partial",
            as_of,
            layout,
            filesystem,
            queue,
            retry_policy,
        )
        return

    if run.status is EvaluationRunStatus.FAILED:
        _dispose_unsatisfied(
            job,
            attempt_id,
            "failed",
            as_of,
            layout,
            filesystem,
            queue,
            retry_policy,
        )
        return

    if run.status is EvaluationRunStatus.CANCELLED:
        _dispose_unsatisfied(
            job,
            attempt_id,
            "cancelled",
            as_of,
            layout,
            filesystem,
            queue,
            retry_policy,
        )
        return

    raise RuntimeError(f"Unsupported Evaluation Run status: {run.status!r}")


def _dispose_unsatisfied(
    job: object,
    attempt_id: int,
    outcome: str,
    as_of: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> None:
    from ._coordination import orchestration_exclusion

    with orchestration_exclusion():
        study_run = read_study_run(
            job.logical.coordinate.study_run,
            layout,
            filesystem,
        )

        if study_run.status is StudyRunStatus.RUNNING:
            decision = retry_policy.after_unsatisfied_attempt(
                job,
                queue.attempts_for_job(job.job_id),
                outcome,
                at=as_of,
            )
            if decision.retry_not_before is not None:
                queue.close_attempt(
                    attempt_id,
                    target_status=QueueJobStatus.RETRY_WAIT,
                    retry_not_before=decision.retry_not_before,
                    finished_at=as_of,
                )
            else:
                queue.close_attempt(
                    attempt_id,
                    target_status=QueueJobStatus.FAILED,
                    finished_at=as_of,
                )
            return

        if study_run.status is StudyRunStatus.CANCELLED:
            queue.cancel_job(job.job_id, at=as_of)
            return

        if study_run.status in {
            StudyRunStatus.COMPLETED,
            StudyRunStatus.COMPLETED_WITH_FAILURES,
            StudyRunStatus.FAILED,
        }:
            raise RuntimeError(
                "Unsatisfied expired ACTIVE attempt conflicts with terminal Study Run."
            )

        raise RuntimeError(f"Unsupported Study Run status: {study_run.status!r}")
