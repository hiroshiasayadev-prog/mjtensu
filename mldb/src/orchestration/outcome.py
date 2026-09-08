"""Controller orchestration for one Worker-reported attempt outcome.

This module fixes the transport-independent operation that connects the frozen Worker
outcome contract to Queue lease authority, canonical Training/Evaluation result
acceptance or unsuccessful terminalization, deterministic Model ensure, retry policy,
and Queue closure.

The boundary owns one logical outcome-report operation only. It does not acquire work,
heartbeat leases, upload candidate bytes, choose a retry algorithm, finalize a Study,
perform general startup reconciliation, define HTTP behavior, or introduce an outcome
manager/service object. Queue SQLite and canonical MLDB filesystem state remain separate
stores; canonical child facts are established before the Queue consequence and existing
terminal child Runs are never reopened or rewritten.
"""

from __future__ import annotations

from ..evaluation.run import EvaluationRunFailure, EvaluationRunStatus
from ..model.persistence import ensure_model_for_completed_training_run
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.result_acceptance import (
    accept_evaluation_success,
    accept_training_success,
)
from ..runtime.run_finalization import (
    cancel_evaluation_run,
    cancel_training_run,
    fail_evaluation_run,
    fail_training_run,
)
from ..runtime.run_persistence import (
    read_evaluation_run,
    read_study_run,
    read_training_run,
)
from ..study.run import StudyRunStatus
from ..training.run import TrainingRunFailure, TrainingRunStatus
from .candidates import CandidateStore, read_verified_candidate_bytes
from .jobs import EvaluationJob, TrainingJob
from .queue import QueueJobStatus
from .queue_ports import QueuePort
from .retry_policy import RetryPolicy
from .worker_api import (
    AttemptCancelled,
    AttemptFailed,
    AttemptOutcome,
    EvaluationSucceeded,
    OutcomeAcknowledgement,
    TrainingSucceeded,
)


def handle_attempt_outcome(
    outcome: AttemptOutcome,
    run_finished_at: object,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    candidates: CandidateStore,
    retry_policy: RetryPolicy,
) -> OutcomeAcknowledgement:
    """Apply one definitive Worker attempt outcome under current canonical authority.

    This is the Controller-side operation behind Worker API ``report_outcome``. It
    accepts no transport object and exposes no HTTP/status-code contract. ``queue_at``
    is the exact v1 Queue timestamp used for lease authorization, retry-policy input,
    attempt closure, and intentional Queue cancellation. ``run_finished_at`` is the
    opaque timestamp value supplied to the frozen child-Run terminalization/acceptance
    boundaries. This function does not normalize or derive either timestamp.

    **Same-attempt authority exclusion.** The currently authorized mutation path must
    participate in the same implementation-private concrete-attempt authority exclusion
    used by heartbeat and expired-lease recovery. Enter it before current lease
    authorization and retain it through canonical child/result acceptance or
    terminalization and the resulting Queue disposition for that attempt. This prevents
    recovery from freshly deciding that the old lease is expired and changing a
    ``RUNNING`` child to ``FAILED`` while an outcome that already relied on that same
    authorization is concurrently canonicalizing success/failure/cancellation.

    The exclusion mechanism is not part of this signature and introduces no public
    Lock/Mutex Protocol, Queue claim field, generation, or cross-store transaction. It
    must compose with the existing same-Study orchestration serialization below; one
    broader Controller serialization is also sufficient. If outcome enters first,
    recovery waits until its canonical mutation and Queue disposition complete. If
    recovery enters first, outcome later fails current authorization and may use only the
    existing historical replay/rejection rules after recovery has closed the attempt.

    **Authority is checked before outcome semantics.** Inside that concrete-attempt
    exclusion, the normal path first calls
    ``queue.authorized_open_attempt(outcome.attempt_id, outcome.lease_token,
    as_of=queue_at)``. A returned attempt is the only ordinary authority to mutate its
    current canonical child Run and later close that attempt. When authorization returns
    ``None``, the function must not immediately reject: it reads the exact retained row
    through ``queue.attempt_by_id(outcome.attempt_id)`` so a lost acknowledgement for an
    already closed attempt can be recognized. No Queue lookup is replaced by a Worker
    claim about Run identity.

    The non-authorized historical replay path returns ``ALREADY_FINALIZED`` only when
    all of the following are established from retained/canonical state:

    - the exact attempt exists and is closed (``finished_at is not None``);
    - its persisted ``lease_token`` equals ``outcome.lease_token`` exactly;
    - ``queue.job_by_id(attempt.job_id)`` returns its parent logical job;
    - the parent job kind agrees with the attempt Run kind and reported outcome kind;
    - the typed canonical child Run for ``attempt.run_id`` is terminal in the exact way
      represented by this report; and
    - every success/failure fact required below agrees sufficiently to prove an exact
      semantic replay rather than merely the existence of some terminal Run. In
      particular, ``AttemptFailed`` requires represented canonical ``failure`` metadata
      whose ``type`` and ``message`` both equal the report exactly; missing failure
      metadata is not a wildcard.

    A missing retained attempt (including Queue-history loss), missing parent job,
    mismatched lease, open-but-expired/stale attempt, job/outcome kind mismatch, or
    conflicting terminal fact is definitive ``REJECTED``. Historical closed replay does
    not call RetryPolicy or require a new Study-derived disposition merely to return
    ``ALREADY_FINALIZED``; the exact retained/canonical replay evidence above remains
    authoritative even when the owning Study is already ``CANCELLED``. The handler does
    not add an ``attempt_id`` or outcome digest to canonical Training/Evaluation schemas
    to recover Queue-only history. General Queue-loss/reconciliation repair remains
    outside this operation.

    Training replay matching requires a :class:`TrainingJob`, a canonical
    :class:`TrainingRunStatus.COMPLETED` Run with canonical weight result metadata, and
    the deterministic Model to be valid under
    :func:`ensure_model_for_completed_training_run`. For ``TrainingSucceeded``, the
    reported weight reference must also resolve through
    :func:`read_verified_candidate_bytes` for this exact attempt and its SHA-256 plus
    byte count must agree exactly with ``run.result.weights``. A terminal Training Run
    alone is never enough. ``AttemptFailed`` matches only a canonical ``FAILED`` Run
    whose ``failure`` is present and whose represented ``type``/``message`` agree
    exactly with the report. A ``FAILED`` Run with ``failure=None`` is insufficient
    evidence that Controller previously accepted this Worker failure and is therefore
    ``REJECTED`` for replay. ``AttemptCancelled`` matches only canonical ``CANCELLED``.
    Training/Evaluation success outcome classes are never interchangeable.

    Evaluation success replay matching requires a :class:`EvaluationJob` and canonical
    ``COMPLETED`` or ``COMPLETED_PARTIAL`` result. Reported metrics and
    ``unavailable_outputs`` must agree with the corresponding terminal formal facts.
    For every accepted artifact key, the reported candidate reference must be readable
    and independently verified from the immutable attempt-local CandidateStore and its
    ``content_identity``/``bytes`` must equal the canonical accepted artifact
    ``sha256``/``bytes``. Candidate artifact presence must also agree with canonical
    optional-artifact rejection facts represented by ``validation_issues``: a rejected
    returned optional artifact remains represented by its formal output issue plus the
    immutable attempt-local candidate reference/content, not by a new digest field in
    the Evaluation Run. A replay that adds, removes, or conflicts with accepted,
    unavailable, or validation-issue formal facts is ``REJECTED``. ``AttemptFailed``
    matches only canonical ``FAILED`` whose ``failure`` is present and whose represented
    ``type``/``message`` agree exactly with the report. A ``FAILED`` Run with
    ``failure=None`` is insufficient evidence that Controller previously accepted this
    Worker failure and is therefore ``REJECTED`` for replay. ``AttemptCancelled``
    matches only canonical ``CANCELLED``.

    Parent logical kind is authoritative for typed canonical reads. A
    :class:`TrainingJob` is handled only through ``read_training_run`` and accepts only
    ``TrainingSucceeded`` or the common failed/cancelled outcomes. An
    :class:`EvaluationJob` is handled only through ``read_evaluation_run`` and accepts
    only ``EvaluationSucceeded`` or the common failed/cancelled outcomes. A success
    outcome of the other job kind is definitive ``REJECTED`` rather than being coerced.

    **Common Study-stop gate for canonical unsatisfied outcomes.** Every authorized
    path that has established a truthful canonical unsatisfied child outcome and would
    otherwise call ``retry_policy.after_unsatisfied_attempt(...)`` must use the same
    Controller implementation-private dispatch-start serialization boundary required by
    acquire and Study cancellation. This applies to Worker ``FAILED`` and ``CANCELLED``,
    accepted Evaluation ``COMPLETED_PARTIAL``, Controller-side success-acceptance
    failure terminalized as ``FAILED``, and terminal-before-Queue-close recovery of
    those same canonical states. Canonical child terminalization may complete before
    entering this *same-Study stop/disposition* section; it remains inside the outer
    same-attempt authority exclusion described above.

    Once the unsatisfied child fact is canonical, the handler must enter that private
    orchestration serialization *before* the authoritative Study stop decision, derive
    the owning Study Run only from ``job.logical.coordinate.study_run``, and perform a
    fresh ``read_study_run(..., layout, filesystem)``. The handler must not reuse a
    Study value read before child terminalization or accept Study identity from Worker
    input. It must remain inside the same private critical section through the complete
    resulting Queue disposition (or operation failure):

        acquire private orchestration serialization
          -> fresh read_study_run(job.logical.coordinate.study_run)
          -> RUNNING: RetryPolicy + Queue close/repair to RETRY_WAIT or FAILED
             CANCELLED: no RetryPolicy + queue.cancel_job(...)
             other terminal Study: no retry; existing inconsistency semantics
          -> release private orchestration serialization

    The protected unit is therefore the authoritative Study stop decision *plus* its
    resulting unsatisfied Queue disposition. A fresh Study read without this exclusion
    is insufficient: cancellation could otherwise commit ``CANCELLED``, intentionally
    pass an ``ACTIVE`` row for cooperative completion, and then have outcome create a
    later ``RETRY_WAIT`` behind the completed cancellation propagation. No public Lock,
    Mutex, transaction-manager, generation, Queue field, or signature is introduced;
    the concrete serialization mechanism remains a Controller implementation concern.

    Within that section, fresh canonical Study status controls only whether another
    logical attempt may be scheduled; it never rewrites the child fact already
    established:

    - :attr:`StudyRunStatus.RUNNING` -> preserve the existing RetryPolicy path and map
      its decision to ``RETRY_WAIT`` or ``FAILED`` before releasing the serialization;
    - :attr:`StudyRunStatus.CANCELLED` -> do **not** call RetryPolicy; call
      ``queue.cancel_job(job.job_id, at=queue_at)`` so the ACTIVE attempt closes and the
      logical job becomes intentionally ``CANCELLED`` before releasing the
      serialization;
    - :attr:`StudyRunStatus.COMPLETED`,
      :attr:`StudyRunStatus.COMPLETED_WITH_FAILURES`, or
      :attr:`StudyRunStatus.FAILED` while this authorized attempt still needs an
      unsatisfied Queue disposition is an orchestration/cross-store inconsistency. No
      RetryPolicy call or new retry is permitted. A path that has already established a
      separate definitive report conflict may still return its existing ``REJECTED``;
      otherwise the inconsistency remains an operation failure rather than inventing a
      new public acknowledgement or policy/result type.

    Both serialization orders are valid and converge without a retry surviving Study
    cancellation. If outcome obtains the private boundary first while Study is still
    ``RUNNING``, it commits ``RETRY_WAIT``/``FAILED`` and releases; cancellation then
    commits ``CANCELLED`` and its normal non-active propagation observes and cancels any
    retryable Queue state. If cancellation obtains the boundary first, it commits Study
    ``CANCELLED``, deliberately leaves the still-``ACTIVE`` attempt for cooperative
    completion, and releases; outcome then freshly reads ``CANCELLED`` inside the same
    boundary, skips RetryPolicy, and cancels that ACTIVE logical job. Thus an
    unsatisfied outcome cannot establish ``RETRY_WAIT`` after durable Study cancellation.
    This in-process serialization is the normal race fix; reconciliation remains only
    crash/restart or other persisted cross-store-gap recovery and is not a substitute
    for coordinating concurrent cancellation and outcome disposition.

    Study cancellation therefore changes only the logical Queue stop decision. A
    Worker-reported ``FAILED`` child remains ``FAILED`` with its exact failure metadata,
    an accepted Evaluation ``COMPLETED_PARTIAL`` remains partial, and a Controller-side
    success-acceptance failure remains ``FAILED(failure=None)``. The outcome
    acknowledgement also keeps its existing semantic meaning: accepted failure/partial
    reports return ``ACCEPTED`` after Queue cancellation, while an invalid Worker
    success candidate remains ``REJECTED`` even when the cancelled Study suppresses a
    retry.

    **Authorized Training success.** When the exact canonical child is still
    ``RUNNING``, the handler calls ``accept_training_success`` with that Run, this
    attempt ID, the reported candidate, and the supplied repository dependencies.
    Successful return means canonical weights were accepted, the Training Run is
    ``COMPLETED``, and its deterministic Model was ensured. Only then may the handler
    call ``queue.close_attempt(..., target_status=QueueJobStatus.SATISFIED,
    finished_at=queue_at)`` and return ``ACCEPTED``. Queue satisfaction never precedes
    canonical Training acceptance plus Model ensure.

    If ``accept_training_success`` raises, the handler must immediately perform a fresh
    ``read_training_run`` before deciding disposition. If the fresh Run is still
    ``RUNNING``, candidate/result acceptance did not establish terminal success; the
    handler may terminalize that exact fresh Run through ``fail_training_run`` with no
    invented failure vocabulary (``failure=None`` is valid). Only after that ``FAILED``
    child is canonical does the handler apply the common fresh Study-stop gate above.
    A still-``RUNNING`` Study uses
    ``retry_policy.after_unsatisfied_attempt(job,
    queue.attempts_for_job(job.job_id), "failed", at=queue_at)`` and closes the current
    attempt to ``RETRY_WAIT`` with the exact returned timestamp or ``FAILED`` when no
    retry remains. A canonical ``CANCELLED`` Study skips RetryPolicy and instead calls
    ``queue.cancel_job(job.job_id, at=queue_at)``. Because the Worker-reported success
    candidate itself was not accepted, the definitive acknowledgement remains
    ``REJECTED`` after either valid Queue disposition is established. Other terminal
    Study states follow the common inconsistency rule and never schedule another
    attempt.

    If the fresh Training Run after an acceptance exception is already ``COMPLETED``,
    it must never be rewritten to ``FAILED``. The handler reruns only
    ``ensure_model_for_completed_training_run`` to repair/verify a possible post-Run
    Model gap. Once Model ensure succeeds and the reported candidate is an exact match
    for that completed result, the still-open authorized attempt may be closed
    ``SATISFIED`` and the report acknowledged ``ACCEPTED``. A different terminal
    Training state is a conflicting success report and returns ``REJECTED`` without a
    terminal rewrite.

    **Authorized Evaluation success.** A canonical ``RUNNING`` Evaluation child is
    passed to ``accept_evaluation_success``. ``COMPLETED`` closes the attempt to
    ``SATISFIED`` and returns ``ACCEPTED``. ``COMPLETED_PARTIAL`` is canonical accepted
    Evaluation history but does not satisfy the logical job, so only after that partial
    Run is canonical does the handler apply the common fresh Study-stop gate. For a
    still-``RUNNING`` Study it calls
    ``retry_policy.after_unsatisfied_attempt(job,
    queue.attempts_for_job(job.job_id), "completed_partial", at=queue_at)`` and closes
    to ``RETRY_WAIT`` with the returned timestamp or ``FAILED`` when no retry remains.
    For a canonical ``CANCELLED`` Study it does not call RetryPolicy and instead calls
    ``queue.cancel_job(job.job_id, at=queue_at)``. The Evaluation Run remains exactly
    ``COMPLETED_PARTIAL`` and the Worker success outcome remains canonically accepted,
    so either valid Queue disposition returns ``ACCEPTED``. Other terminal Study states
    follow the common inconsistency rule.

    If ``accept_evaluation_success`` raises, the handler freshly calls
    ``read_evaluation_run``. A still-``RUNNING`` Run may be terminalized through
    ``fail_evaluation_run`` with ``failure=None``. Only after that ``FAILED`` child is
    canonical does the handler apply the common fresh Study-stop gate: a ``RUNNING``
    Study uses the existing ``after_unsatisfied_attempt(..., "failed")`` mapping, while
    a ``CANCELLED`` Study skips RetryPolicy and calls
    ``queue.cancel_job(job.job_id, at=queue_at)``. Because the reported success candidate
    was invalid/unaccepted, the acknowledgement remains ``REJECTED`` after either valid
    Queue disposition. Other terminal Study states schedule no retry and follow the
    common inconsistency rule.

    If the fresh Run is already ``COMPLETED`` or ``COMPLETED_PARTIAL``, it is never
    rewritten. The handler first requires the reported candidate to match that canonical
    result exactly under the replay rules above, then recovers the lagging Queue closure:
    ``COMPLETED`` to ``SATISFIED``; ``COMPLETED_PARTIAL`` through the same common fresh
    Study-stop gate, giving RetryPolicy-driven ``RETRY_WAIT``/``FAILED`` only while the
    Study is ``RUNNING`` and ``queue.cancel_job`` without RetryPolicy when the Study is
    ``CANCELLED``. Because this is still the currently authorized open attempt whose
    Queue close lagged terminal acceptance, successful recovery returns ``ACCEPTED``
    rather than historical ``ALREADY_FINALIZED``. Any other terminal child state is
    conflicting and returns ``REJECTED`` without being rewritten.

    **Authorized ``AttemptFailed``.** The job kind selects the typed canonical child.
    If it is ``RUNNING``, the handler uses ``fail_training_run`` with
    ``TrainingRunFailure(type=outcome.type, message=outcome.message)`` or
    ``fail_evaluation_run`` with the analogous ``EvaluationRunFailure``. After that exact
    Worker failure is canonical, the common fresh Study-stop gate applies. A
    ``RUNNING`` Study calls ``after_unsatisfied_attempt(..., "failed")`` over retained
    attempt history and closes to ``RETRY_WAIT`` when a retry timestamp is returned,
    otherwise ``FAILED``. A ``CANCELLED`` Study does not call RetryPolicy and instead
    calls ``queue.cancel_job(job.job_id, at=queue_at)``. In both cases the child remains
    ``FAILED`` with the exact Worker ``type``/``message`` and the report returns
    ``ACCEPTED``. Other terminal Study states are inconsistent with this still-active
    unsatisfied attempt and schedule no retry. This operation never asks Worker to
    execute the same attempt again.

    A validly authorized open attempt may also encounter a child that is already
    terminal because Controller previously committed the child transition and crashed
    before Queue closure. The handler must not invoke another terminal transition. For
    ``AttemptFailed``, an already-``FAILED`` Training/Evaluation child is an exact
    accepted Worker-failure replay only when ``failure is not None`` and its
    ``type``/``message`` equal ``outcome.type``/``outcome.message`` exactly. For that
    exact replay the same common fresh Study-stop gate repairs the lagging Queue close:
    a ``RUNNING`` Study uses the existing failed RetryPolicy disposition; a
    ``CANCELLED`` Study skips policy and calls ``queue.cancel_job``; either successful
    recovery returns ``ACCEPTED`` because the current attempt is still authorized/open.
    A generic ``FAILED`` child with ``failure=None`` (including Controller
    reconciliation, lease-loss recovery, or success-candidate acceptance failure) is
    insufficient evidence that this ``AttemptFailed`` report was previously accepted
    and is therefore ``REJECTED``; this handler does not rewrite that terminal Run or
    close the Queue attempt on the basis of the conflicting failure report. General
    reconciliation may later repair the lagging Queue state. An exact replay of
    ``CANCELLED`` recovers the cancellation disposition described below. This is
    cross-store crash-gap recovery for the same current attempt, not general
    reconciliation.

    **Authorized ``AttemptCancelled``.** A ``RUNNING`` child is terminalized only via
    ``cancel_training_run`` or ``cancel_evaluation_run``. Once that child cancellation
    is canonical, the same common Study-stop gate applies: derive the owning Study only
    from the Queue job coordinate and freshly ``read_study_run`` before any RetryPolicy
    call. ``StudyRunStatus.CANCELLED`` is sufficient under the frozen Study Run lifecycle
    to prove intentional Study-level stop: no RetryPolicy call is made and
    ``queue.cancel_job(job.job_id, at=queue_at, ...)`` closes/cancels the logical work
    before returning ``ACCEPTED``. When the fresh Study Run remains ``RUNNING``, the
    child cancellation alone is not intentional logical-stop authority; the handler
    calls ``after_unsatisfied_attempt(..., "cancelled")`` and closes to ``RETRY_WAIT``
    when retry remains or ``FAILED`` when it does not. Other terminal Study states while
    a Queue attempt is still active are an orchestration invariant inconsistency, not
    grounds to invent a new cancellation flag or push Study-stop meaning into
    RetryPolicy.

    The same Study-status rule applies when an authorized open attempt already has an
    exactly matching canonical ``CANCELLED`` child from a terminal-before-Queue-close
    crash gap: do not cancel the Run again; only recover the applicable Queue
    cancellation/retry/failure consequence. Likewise, an exact canonical Evaluation
    ``COMPLETED_PARTIAL`` child uses the common fresh Study-stop gate, so a cancelled
    owning Study repairs the lagging ACTIVE Queue row directly to ``CANCELLED`` without
    RetryPolicy. No terminal child is reopened.

    Satisfying terminal success is deliberately different. An authorized open Training
    ``COMPLETED`` child with exact weight candidate and valid deterministic Model closes
    ``SATISFIED``; an authorized open Evaluation ``COMPLETED`` child closes
    ``SATISFIED``. These fully satisfying paths need not enter the unsatisfied
    Study-stop/Queue-disposition critical section merely because Study cancellation may
    race: they create no retry. If Training acceptance plus Model ensure or Evaluation
    acceptance establishes the exact fully satisfying canonical success, the child
    remains truthful ``COMPLETED`` and Queue ``SATISFIED`` is allowed even when the
    Study became ``CANCELLED`` concurrently. Training satisfaction may operationally
    release blocked dependent Evaluation jobs; cancellation's normal/idempotent Queue
    propagation and acquire's canonical Study authority check prevent those jobs from
    beginning fresh execution under the cancelled Study. The handler does not rewrite a
    satisfying child to ``CANCELLED`` merely because a fresh Study read would now be
    cancelled. The invariant enforced here is that a cancelled Study never schedules
    *another* attempt from an unsatisfied outcome; satisfying accepted history remains
    truthful.

    Queue mapping is therefore mechanical after canonical facts are established:

    - satisfying Training ``COMPLETED`` + valid Model -> ``SATISFIED`` regardless of a
      concurrent Study cancellation race;
    - satisfying Evaluation ``COMPLETED`` -> ``SATISFIED`` on the same basis;
    - ``FAILED`` + current Study ``RUNNING`` -> policy result ``RETRY_WAIT`` or
      ``FAILED``;
    - ``FAILED`` + current Study ``CANCELLED`` -> ``cancel_job`` and logical
      ``CANCELLED`` without RetryPolicy;
    - Evaluation ``COMPLETED_PARTIAL`` + current Study ``RUNNING`` -> policy result
      ``RETRY_WAIT`` or ``FAILED``;
    - Evaluation ``COMPLETED_PARTIAL`` + current Study ``CANCELLED`` -> ``cancel_job``
      and logical ``CANCELLED`` without RetryPolicy;
    - child ``CANCELLED`` + current Study ``RUNNING`` -> policy result
      ``RETRY_WAIT`` or ``FAILED``;
    - child ``CANCELLED`` + current Study ``CANCELLED`` -> ``cancel_job`` and logical
      ``CANCELLED`` without RetryPolicy;
    - any unsatisfied child + current Study ``COMPLETED``,
      ``COMPLETED_WITH_FAILURES``, or ``FAILED`` while the attempt remains ACTIVE -> no
      RetryPolicy/new retry; surface the applicable existing definitive conflict or
      orchestration operation failure.

    Definitive semantic mismatches return ``REJECTED``. Failures of repository, Queue,
    CandidateStore, Model ensure, or other infrastructure that prevent the handler from
    establishing a definitive acknowledgement are not silently converted into Worker
    ``REJECTED``; they remain operation failures so the transport boundary can preserve
    the Worker API distinction between a definitive response and retryable/no-response
    communication/service failure.
    """


    from ._coordination import orchestration_exclusion

    with orchestration_exclusion():
        attempt = queue.authorized_open_attempt(
            outcome.attempt_id,
            outcome.lease_token,
            as_of=queue_at,
        )
        if attempt is None:
            return _historical_replay(
                outcome, layout, filesystem, queue, candidates
            )

        job = queue.job_by_id(attempt.job_id)
        if job is None:
            raise RuntimeError("Authorized Queue attempt has no parent job.")
        if not _attempt_kind_agrees(job, attempt.run_id):
            raise RuntimeError("Authorized Queue attempt Run kind disagrees with parent job.")
        if not _outcome_compatible(job, outcome):
            return OutcomeAcknowledgement.REJECTED

        run = _read_child(job, attempt.run_id, layout, filesystem)

        if isinstance(outcome, TrainingSucceeded):
            return _handle_training_success(
                outcome, run, job, attempt.attempt_id, run_finished_at, queue_at,
                layout, filesystem, queue, candidates, retry_policy,
            )
        if isinstance(outcome, EvaluationSucceeded):
            return _handle_evaluation_success(
                outcome, run, job, attempt.attempt_id, run_finished_at, queue_at,
                layout, filesystem, queue, candidates, retry_policy,
            )
        if isinstance(outcome, AttemptFailed):
            return _handle_failed(
                outcome, run, job, attempt.attempt_id, run_finished_at, queue_at,
                layout, filesystem, queue, retry_policy,
            )
        if isinstance(outcome, AttemptCancelled):
            return _handle_cancelled(
                run, job, attempt.attempt_id, run_finished_at, queue_at,
                layout, filesystem, queue, retry_policy,
            )
        raise RuntimeError(f"Unsupported attempt outcome: {type(outcome)!r}")


def _attempt_kind_agrees(job: object, run_id: object) -> bool:
    value = str(run_id)
    if isinstance(job.logical, TrainingJob):
        return value.startswith("tr-")
    if isinstance(job.logical, EvaluationJob):
        return value.startswith("ev-")
    return False


def _outcome_compatible(job: object, outcome: AttemptOutcome) -> bool:
    if isinstance(job.logical, TrainingJob):
        return not isinstance(outcome, EvaluationSucceeded)
    if isinstance(job.logical, EvaluationJob):
        return not isinstance(outcome, TrainingSucceeded)
    return False


def _read_child(
    job: object,
    run_id: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> object:
    if isinstance(job.logical, TrainingJob):
        return read_training_run(run_id, layout, filesystem)
    if isinstance(job.logical, EvaluationJob):
        return read_evaluation_run(run_id, layout, filesystem)
    raise RuntimeError("Queue parent has an unsupported logical job kind.")


def _historical_replay(
    outcome: AttemptOutcome,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    candidates: CandidateStore,
) -> OutcomeAcknowledgement:
    attempt = queue.attempt_by_id(outcome.attempt_id)
    if (
        attempt is None
        or attempt.finished_at is None
        or attempt.lease_token != outcome.lease_token
    ):
        return OutcomeAcknowledgement.REJECTED

    job = queue.job_by_id(attempt.job_id)
    if job is None:
        return OutcomeAcknowledgement.REJECTED
    if not _attempt_kind_agrees(job, attempt.run_id):
        return OutcomeAcknowledgement.REJECTED
    if not _outcome_compatible(job, outcome):
        return OutcomeAcknowledgement.REJECTED

    run = _read_child(job, attempt.run_id, layout, filesystem)
    if isinstance(outcome, TrainingSucceeded):
        if not isinstance(job.logical, TrainingJob):
            return OutcomeAcknowledgement.REJECTED
        if not _training_success_matches(run, outcome, attempt.attempt_id, candidates):
            return OutcomeAcknowledgement.REJECTED
        ensure_model_for_completed_training_run(run, layout, filesystem)
        return OutcomeAcknowledgement.ALREADY_FINALIZED
    if isinstance(outcome, EvaluationSucceeded):
        if not _evaluation_success_matches(run, outcome, attempt.attempt_id, candidates):
            return OutcomeAcknowledgement.REJECTED
        return OutcomeAcknowledgement.ALREADY_FINALIZED
    if isinstance(outcome, AttemptFailed):
        failure = getattr(run, "failure", None)
        failed_status = (
            TrainingRunStatus.FAILED
            if isinstance(job.logical, TrainingJob)
            else EvaluationRunStatus.FAILED
        )
        if (
            run.status is failed_status
            and failure is not None
            and failure.type == outcome.type
            and failure.message == outcome.message
        ):
            return OutcomeAcknowledgement.ALREADY_FINALIZED
        return OutcomeAcknowledgement.REJECTED
    if isinstance(outcome, AttemptCancelled):
        cancelled_status = (
            TrainingRunStatus.CANCELLED
            if isinstance(job.logical, TrainingJob)
            else EvaluationRunStatus.CANCELLED
        )
        if run.status is cancelled_status:
            return OutcomeAcknowledgement.ALREADY_FINALIZED
        return OutcomeAcknowledgement.REJECTED
    raise RuntimeError(f"Unsupported attempt outcome: {type(outcome)!r}")


def _handle_training_success(
    outcome: TrainingSucceeded,
    run: object,
    job: object,
    attempt_id: int,
    run_finished_at: object,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    candidates: CandidateStore,
    retry_policy: RetryPolicy,
) -> OutcomeAcknowledgement:
    if run.status is TrainingRunStatus.RUNNING:
        try:
            completed = accept_training_success(
                run,
                attempt_id,
                outcome.candidate,
                run_finished_at,
                layout,
                filesystem,
                candidates,
            )
        except Exception as error:
            fresh = read_training_run(run.id, layout, filesystem)
            if fresh.status is TrainingRunStatus.RUNNING:
                if not isinstance(error, ValueError):
                    raise
                fail_training_run(
                    fresh,
                    run_finished_at,
                    layout,
                    filesystem,
                    failure=None,
                )
                _dispose_unsatisfied(
                    job, attempt_id, "failed", queue_at, layout, filesystem,
                    queue, retry_policy,
                )
                return OutcomeAcknowledgement.REJECTED
            if fresh.status is TrainingRunStatus.COMPLETED:
                ensure_model_for_completed_training_run(fresh, layout, filesystem)
                if not _training_success_matches(
                    fresh, outcome, attempt_id, candidates
                ):
                    return OutcomeAcknowledgement.REJECTED
                queue.close_attempt(
                    attempt_id,
                    target_status=QueueJobStatus.SATISFIED,
                    finished_at=queue_at,
                )
                return OutcomeAcknowledgement.ACCEPTED
            return OutcomeAcknowledgement.REJECTED

        if completed.status is not TrainingRunStatus.COMPLETED:
            raise RuntimeError("Training acceptance returned a non-COMPLETED Run.")
        queue.close_attempt(
            attempt_id,
            target_status=QueueJobStatus.SATISFIED,
            finished_at=queue_at,
        )
        return OutcomeAcknowledgement.ACCEPTED

    if run.status is TrainingRunStatus.COMPLETED:
        ensure_model_for_completed_training_run(run, layout, filesystem)
        if not _training_success_matches(run, outcome, attempt_id, candidates):
            return OutcomeAcknowledgement.REJECTED
        queue.close_attempt(
            attempt_id,
            target_status=QueueJobStatus.SATISFIED,
            finished_at=queue_at,
        )
        return OutcomeAcknowledgement.ACCEPTED

    return OutcomeAcknowledgement.REJECTED


def _handle_evaluation_success(
    outcome: EvaluationSucceeded,
    run: object,
    job: object,
    attempt_id: int,
    run_finished_at: object,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    candidates: CandidateStore,
    retry_policy: RetryPolicy,
) -> OutcomeAcknowledgement:
    if run.status is EvaluationRunStatus.RUNNING:
        try:
            terminal = accept_evaluation_success(
                run,
                attempt_id,
                outcome.candidate,
                run_finished_at,
                layout,
                filesystem,
                candidates,
            )
        except Exception as error:
            fresh = read_evaluation_run(run.id, layout, filesystem)
            if fresh.status is EvaluationRunStatus.RUNNING:
                if not isinstance(error, ValueError):
                    raise
                fail_evaluation_run(
                    fresh,
                    run_finished_at,
                    layout,
                    filesystem,
                    failure=None,
                )
                _dispose_unsatisfied(
                    job, attempt_id, "failed", queue_at, layout, filesystem,
                    queue, retry_policy,
                )
                return OutcomeAcknowledgement.REJECTED
            if fresh.status in {
                EvaluationRunStatus.COMPLETED,
                EvaluationRunStatus.COMPLETED_PARTIAL,
            }:
                if not _evaluation_success_matches(
                    fresh, outcome, attempt_id, candidates
                ):
                    return OutcomeAcknowledgement.REJECTED
                _close_evaluation_terminal(
                    fresh, job, attempt_id, queue_at, layout, filesystem,
                    queue, retry_policy,
                )
                return OutcomeAcknowledgement.ACCEPTED
            return OutcomeAcknowledgement.REJECTED

        if terminal.status not in {
            EvaluationRunStatus.COMPLETED,
            EvaluationRunStatus.COMPLETED_PARTIAL,
        }:
            raise RuntimeError("Evaluation acceptance returned a non-success terminal Run.")
        _close_evaluation_terminal(
            terminal, job, attempt_id, queue_at, layout, filesystem,
            queue, retry_policy,
        )
        return OutcomeAcknowledgement.ACCEPTED

    if run.status in {
        EvaluationRunStatus.COMPLETED,
        EvaluationRunStatus.COMPLETED_PARTIAL,
    }:
        if not _evaluation_success_matches(run, outcome, attempt_id, candidates):
            return OutcomeAcknowledgement.REJECTED
        _close_evaluation_terminal(
            run, job, attempt_id, queue_at, layout, filesystem, queue, retry_policy,
        )
        return OutcomeAcknowledgement.ACCEPTED

    return OutcomeAcknowledgement.REJECTED


def _close_evaluation_terminal(
    run: object,
    job: object,
    attempt_id: int,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> None:
    if run.status is EvaluationRunStatus.COMPLETED:
        queue.close_attempt(
            attempt_id,
            target_status=QueueJobStatus.SATISFIED,
            finished_at=queue_at,
        )
        return
    if run.status is EvaluationRunStatus.COMPLETED_PARTIAL:
        _dispose_unsatisfied(
            job, attempt_id, "completed_partial", queue_at, layout, filesystem,
            queue, retry_policy,
        )
        return
    raise RuntimeError("Evaluation success recovery received a non-success terminal Run.")


def _handle_failed(
    outcome: AttemptFailed,
    run: object,
    job: object,
    attempt_id: int,
    run_finished_at: object,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> OutcomeAcknowledgement:
    if isinstance(job.logical, TrainingJob):
        if run.status is TrainingRunStatus.RUNNING:
            fail_training_run(
                run,
                run_finished_at,
                layout,
                filesystem,
                failure=TrainingRunFailure(
                    type=outcome.type,
                    message=outcome.message,
                ),
            )
        elif not (
            run.status is TrainingRunStatus.FAILED
            and run.failure is not None
            and run.failure.type == outcome.type
            and run.failure.message == outcome.message
        ):
            return OutcomeAcknowledgement.REJECTED
    else:
        if run.status is EvaluationRunStatus.RUNNING:
            fail_evaluation_run(
                run,
                run_finished_at,
                layout,
                filesystem,
                failure=EvaluationRunFailure(
                    type=outcome.type,
                    message=outcome.message,
                ),
            )
        elif not (
            run.status is EvaluationRunStatus.FAILED
            and run.failure is not None
            and run.failure.type == outcome.type
            and run.failure.message == outcome.message
        ):
            return OutcomeAcknowledgement.REJECTED

    _dispose_unsatisfied(
        job, attempt_id, "failed", queue_at, layout, filesystem, queue, retry_policy
    )
    return OutcomeAcknowledgement.ACCEPTED


def _handle_cancelled(
    run: object,
    job: object,
    attempt_id: int,
    run_finished_at: object,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> OutcomeAcknowledgement:
    if isinstance(job.logical, TrainingJob):
        if run.status is TrainingRunStatus.RUNNING:
            cancel_training_run(run, run_finished_at, layout, filesystem)
        elif run.status is not TrainingRunStatus.CANCELLED:
            return OutcomeAcknowledgement.REJECTED
    else:
        if run.status is EvaluationRunStatus.RUNNING:
            cancel_evaluation_run(run, run_finished_at, layout, filesystem)
        elif run.status is not EvaluationRunStatus.CANCELLED:
            return OutcomeAcknowledgement.REJECTED

    _dispose_unsatisfied(
        job, attempt_id, "cancelled", queue_at, layout, filesystem, queue, retry_policy
    )
    return OutcomeAcknowledgement.ACCEPTED


def _dispose_unsatisfied(
    job: object,
    attempt_id: int,
    outcome: str,
    queue_at: str,
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
                at=queue_at,
            )
            if decision.retry_not_before is not None:
                queue.close_attempt(
                    attempt_id,
                    target_status=QueueJobStatus.RETRY_WAIT,
                    retry_not_before=decision.retry_not_before,
                    finished_at=queue_at,
                )
            else:
                queue.close_attempt(
                    attempt_id,
                    target_status=QueueJobStatus.FAILED,
                    finished_at=queue_at,
                )
            return

        if study_run.status is StudyRunStatus.CANCELLED:
            queue.cancel_job(job.job_id, at=queue_at)
            return

        if study_run.status in {
            StudyRunStatus.COMPLETED,
            StudyRunStatus.COMPLETED_WITH_FAILURES,
            StudyRunStatus.FAILED,
        }:
            raise RuntimeError(
                "Unsatisfied ACTIVE Queue attempt conflicts with terminal Study Run."
            )
        raise RuntimeError(f"Unsupported Study Run status: {study_run.status!r}")


def _training_success_matches(
    run: object,
    outcome: TrainingSucceeded,
    attempt_id: int,
    candidates: CandidateStore,
) -> bool:
    if run.status is not TrainingRunStatus.COMPLETED or run.result is None:
        return False
    weights = run.result.weights
    ref = outcome.candidate.weights
    if ref.content_identity != weights.sha256 or ref.bytes != weights.bytes:
        return False
    read_verified_candidate_bytes(attempt_id, ref, candidates)
    return True


def _evaluation_success_matches(
    run: object,
    outcome: EvaluationSucceeded,
    attempt_id: int,
    candidates: CandidateStore,
) -> bool:
    if run.status not in {
        EvaluationRunStatus.COMPLETED,
        EvaluationRunStatus.COMPLETED_PARTIAL,
    } or run.result is None:
        return False

    candidate = outcome.candidate
    if dict(candidate.metrics) != dict(run.result.metrics):
        return False
    if tuple(candidate.unavailable_outputs) != tuple(run.unavailable_outputs):
        return False

    accepted = dict(run.result.artifacts)
    rejected_keys: set[str] = set()
    for issue in run.validation_issues:
        prefix = "artifacts."
        if not issue.output.startswith(prefix):
            return False
        key = issue.output[len(prefix):]
        if not key or key in rejected_keys or key in accepted:
            return False
        rejected_keys.add(key)

    if set(candidate.artifacts) != set(accepted) | rejected_keys:
        return False

    for key, artifact in accepted.items():
        ref = candidate.artifacts[key]
        if ref.content_identity != artifact.sha256 or ref.bytes != artifact.bytes:
            return False
        read_verified_candidate_bytes(attempt_id, ref, candidates)

    for key in rejected_keys:
        read_verified_candidate_bytes(attempt_id, candidate.artifacts[key], candidates)

    return True
