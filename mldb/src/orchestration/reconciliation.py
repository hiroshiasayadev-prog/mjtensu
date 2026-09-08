"""Study-scoped Queue / canonical child-history reconciliation boundary.

This module fixes the Controller operation that repairs one Study Run's replaceable
Queue projection from its immutable finalized plan, canonical Training/Evaluation/Model
history, retained Queue state, and the frozen retry-policy seam.

Canonical MLDB state always outranks Queue progress. Reconciliation may therefore
restore missing logical jobs, repair stale Queue lifecycle state, repair the deterministic
Model gap after completed training, unsuccessfully terminalize an unrecoverable canonical
``RUNNING`` child, and finally decide whether the Study Run has exhausted all planned
work. Queue/MLDB writes are intentionally ordered and restartable rather than wrapped in
one cross-store transaction.

Reconciliation is a Controller maintenance/recovery phase, not a mutation that may race
freely with normal orchestration. The caller must execute one Study reconciliation under
an implementation-private Controller orchestration maintenance exclusion that prevents
same-Study fresh acquire/dispatch start, outcome canonical/Queue mutation, and Study
cancellation mutation/propagation from running concurrently. The exclusion begins before
the initial canonical Study read and remains held through all canonical/Queue repairs and
any final Study transition. It may be Controller-wide or correctly keyed per Study; no
public lock, mutex, manager, or transaction argument is introduced here.

Heartbeat is intentionally not stopped for the whole Study maintenance pass. When one
retained ``ACTIVE``/open attempt's lease authority decides whether a canonical
``RUNNING`` child may continue, reconciliation additionally participates in the same
implementation-private concrete-attempt authority exclusion used by heartbeat, outcome,
and expired-lease recovery. Inside that narrower exclusion it freshly rechecks the exact
current Queue attempt and lease authority, and an expired/unrecoverable path remains
protected through canonical-first interrupted-child resolution and Queue disposition.
The composition is private Controller policy, not a public locking API.

This is not a startup manager, scheduler, Worker acquire/outcome handler, generic state
machine, graph engine, history repository, transaction coordinator, or SQLite adapter.
The application may loop over running Study Runs privately and call the single public
Study-scoped operation here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..common.errors import LifecycleConflictError
from ..common.ids import ModelId, StudyRunId
from ..evaluation.run import EvaluationRunStatus
from ..model.persistence import ensure_model_for_completed_training_run
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.run_finalization import fail_evaluation_run, fail_training_run
from ..runtime.run_persistence import (
    list_evaluation_runs,
    list_training_runs,
    persist_study_run_transition,
    read_study_plan,
    read_study_run,
)
from ..study.plan import StudyPlanEvaluation, StudyPlanRow
from ..study.run import StudyRun, StudyRunExecution, StudyRunStatus
from ..training.run import TrainingRun, TrainingRunStatus
from ._coordination import orchestration_exclusion
from .jobs import EvaluationJob, StudyJob, TrainingJob, TrainingJobCoordinate, derive_study_jobs
from .queue import QueueAttempt, QueueJob, QueueJobStatus
from .queue_ports import QueuePort
from .retry_policy import RetryPolicy


def reconcile_study_run(
    study_run_id: StudyRunId,
    run_finished_at: object,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> StudyRun:
    """Reconcile one Study Run from immutable intent plus canonical child history.

    **Invocation precondition / maintenance exclusion.** This function is valid only
    when the Controller caller has already entered an implementation-private
    orchestration maintenance exclusion for ``study_run_id``. The protected scope starts
    before this function's initial canonical Study read and ends only after this call has
    completed every canonical/Queue repair and any final Study Run transition. If the
    operation fails, the caller must not treat that partial pass as completed maintenance
    and resume normal same-Study mutation from a projection still requiring reconciliation.
    At minimum, the exclusion must prevent these same-Study operations from mutating while
    reconciliation is active:

    - fresh acquire / child dispatch start;
    - outcome canonical acceptance or unsuccessful terminalization together with its
      resulting Queue disposition;
    - Study cancellation canonical mutation and Queue propagation.

    The mechanism is deliberately not frozen. A Controller-wide orchestration exclusion
    or a correctly keyed per-Study equivalent is sufficient. It may reuse/extend the
    existing private dispatch-start serialization or be enforced by a compatible outer
    Controller invocation gate, provided acquire/outcome/cancellation cannot cross the
    maintenance scope above. No lock/mutex parameter, public maintenance-mode DTO,
    manager, generation, CAS token, Queue version, or cross-store transaction is added to
    this signature.

    Heartbeat is intentionally not globally stopped merely because reconciliation is in
    progress. Instead, when one retained ``ACTIVE``/open attempt's Worker lease authority
    determines whether a canonical ``RUNNING`` child is preserved or interrupted, the
    Controller implementation of this boundary must enter the same implementation-private
    concrete-attempt authority exclusion used by ``handle_heartbeat()``,
    ``handle_attempt_outcome()``, and ``recover_expired_attempts()`` for that concrete
    attempt. Unrelated Study/Worker heartbeats need not pause for the full reconciliation
    pass. A broader private orchestration serialization is also valid when it provides
    the same ordering without becoming a public locking contract.

    The same-Study maintenance exclusion remains held and continues to exclude fresh
    dispatch, outcome canonical/Queue mutation, and Study cancellation while this narrow
    attempt-authority section is entered. The two private exclusions must compose without
    permitting an authority/use gap or deadlocking the orchestration implementation; no
    lock-order API is frozen. Outcome requires no additional mechanism beyond its existing
    maintenance participation plus the shared concrete-attempt authority serialization.
    Immutable-asset retrieval and candidate upload require neither exclusion merely for
    data-plane access: they do not extend lease authority or choose canonical child/Study
    or logical Queue disposition.

    Once the required same-Study maintenance exclusion is held, the function begins with
    :func:`read_study_run` for ``study_run_id``. Terminal Study Runs are immutable and
    are returned unchanged. Reconciliation does not reopen them, restart child work, or
    perform aggressive Queue cleanup merely because stale Queue rows may still exist.

    This ordering closes the stale-``RUNNING`` cancellation race. If reconciliation
    enters first, it may repair/admit/finalize from the fresh ``RUNNING`` state, but all
    such work completes before cancellation can enter; cancellation then either changes
    the still-running Study to ``CANCELLED`` and cancels its non-active Queue work, or
    observes the Study already terminal and returns its existing already-terminal
    result. If cancellation enters first, its canonical ``CANCELLED`` transition and
    required non-active Queue propagation complete before reconciliation can read the
    Study; reconciliation then reads terminal ``CANCELLED`` and returns unchanged, so it
    cannot recreate ``READY`` or ``RETRY_WAIT`` work behind the completed cancellation
    pass.

    The same exclusion also forbids reconciliation from acting on an older child-history
    snapshot while normal outcome handling accepts/terminalizes that child's canonical
    result and changes the Queue disposition. Thus a normal outcome cannot establish
    canonical ``COMPLETED`` + Queue ``SATISFIED`` while stale reconciliation repairs the
    same coordinate back to ``READY``/``RETRY_WAIT``, and reconciliation cannot close or
    repurpose active authority while a same-Study fresh acquire is beginning dispatch.
    This Wave does not replace that Controller-level exclusion with Queue CAS/versioning.

    A canonical :class:`StudyRun` may legitimately remain ``RUNNING`` with ``plan=None``
    after interrupted plan materialization. In that state this operation returns the Run
    unchanged. It does not read partial ``plan.jsonl`` bytes, invent plan metadata,
    derive jobs, or restart materialization. Resume/materialization remains owned by the
    existing Study launch/resume boundary.

    For ``RUNNING`` with finalized plan metadata, the required setup is:

    1. call :func:`read_study_plan` and use only the verified immutable plan;
    2. call :func:`derive_study_jobs(study_run_id, plan)`;
    3. call ``queue.admit_study_jobs(study_run_id, jobs, admitted_at=queue_at)`` before
       interpreting progress. Repeated admission restores missing rows without resetting
       retained operational state and rejects logical/dependency disagreement;
    4. require the resulting ``queue.jobs_for_study_run(study_run_id)`` projection to
       contain exactly the expected logical coordinates. Extra Study-owned Queue rows,
       duplicate logical coordinates, or dependency intent outside the plan are
       inconsistencies rather than execution intent that reconciliation may rewrite.

    Canonical child history is obtained only through :func:`list_training_runs` and
    :func:`list_evaluation_runs`, filtered by exact represented Study lineage. A child
    whose ``study.run`` equals ``study_run_id`` must map to exactly one expected plan
    coordinate and must agree with that coordinate's immutable execution intent. A
    Study-linked child outside the plan, or a same-coordinate child whose recorded
    execution inputs disagree with the plan, is an inconsistency rather than a candidate
    to ignore or normalize.

    For one training-derived plan row, every same-coordinate Training Run must have
    exactly ``study.run == study_run_id`` and ``study.trial == row.trial`` and must agree
    with the immutable training intent:

    - ``run.corpus == row.training.corpus``;
    - ``run.architecture == row.training.architecture``;
    - ``run.train_protocol == row.training.protocol``;
    - ``run.execution.seed == row.training.seed``;
    - ``run.parameters == row.training.parameters``.

    A Training coordinate is satisfied only when exactly one matching canonical
    Training Run is ``COMPLETED`` and the deterministic Model for that exact Run exists
    and is valid. :func:`ensure_model_for_completed_training_run` is replay-safe and is
    therefore called for that sole completed Run before Queue satisfaction repair. This
    repairs the cross-store gap in which Training completion committed but Model YAML did
    not. More than one completed Training Run for the coordinate is ambiguous Model
    authority and is an inconsistency; reconciliation never chooses a latest completed
    training result. A satisfying completed Training Run also outranks stale Queue
    ``FAILED``/``CANCELLED`` state: after Model ensure, the Queue job becomes
    ``SATISFIED`` through normal attempt closure when its matching attempt remains open,
    or through ``repair_job_state`` when no open attempt exists. Accepted training is
    never rerun merely because downstream Evaluation remains incomplete.

    For one Evaluation coordinate, every same-coordinate Evaluation Run must have exact
    ``study.run``, ``study.trial``, and ``study.stage`` lineage and must agree with the
    immutable stage intent: ``corpus``, ``evaluation_protocol``, and complete resolved
    ``parameters`` must equal the selected plan stage. Its ``model`` must equal the exact
    plan Model for an existing-Model row, or the deterministic Model ensured from the
    sole completed same-trial Training Run for a training-derived row. A matching
    ``COMPLETED`` Evaluation Run satisfies the coordinate. ``COMPLETED_PARTIAL``,
    ``FAILED``, and ``CANCELLED`` remain canonical historical facts but do not satisfy
    it. If any exact satisfying Evaluation Run exists, satisfaction wins over earlier
    unsuccessful retry history and over stale terminal Queue state.

    At most one canonical ``RUNNING`` child may exist for one logical coordinate. More
    than one is an inconsistency. A coordinate that already has a satisfying canonical
    child must not simultaneously retain another ``RUNNING`` child; that contradictory
    active execution is also surfaced as an inconsistency rather than silently adopted
    or allowed to create another accepted result.

    A retained Queue snapshot may identify a candidate sole same-coordinate ``RUNNING``
    child plus ``ACTIVE``/open attempt, but that snapshot is observation rather than
    authority. When the retained open attempt's lease decides whether the child may keep
    executing, enter that concrete attempt's private authority exclusion and freshly
    establish Queue authority from the exact attempt identity. Conceptually::

        fresh = queue.attempt_by_id(attempt.attempt_id)

    The candidate may be preserved or interrupted from retained Worker authority only
    when ``fresh`` exists, ``fresh.finished_at is None``, and its ``job_id``, ``run_id``,
    and ``lease_token`` exactly equal the retained candidate; its ``run_id`` must still
    equal the canonical ``RUNNING`` child. Freshly resolving ``queue.job_by_id`` must
    show that same parent Queue job still ``ACTIVE``, and
    ``queue.open_attempt_for_job(job.job_id)`` must identify this exact same open attempt.
    An earlier ``jobs_for_study_run``/open-attempt snapshot is never lease authority for
    terminalizing the child.

    Only after those exact current/open checks, and while the same concrete-attempt
    exclusion is still held, re-check lease authorization exactly from the fresh row::

        authorized = queue.authorized_open_attempt(
            fresh.attempt_id,
            fresh.lease_token,
            as_of=queue_at,
        )

    If ``authorized`` returns that same current attempt, preserve the canonical
    ``RUNNING`` child and the active attempt. Controller restart alone therefore does not
    fail a healthy leased Worker or allocate a replacement Run. If the exact same attempt
    remains current/open and the authorization call returns ``None``, its lease may be
    resolved as expired/unrecoverable in this pass because same-attempt heartbeat/outcome
    authority mutation remains excluded until the resulting disposition is complete.

    If the retained candidate was concurrently closed/resolved before reconciliation
    acquired its concrete-attempt exclusion, do not reopen it or independently recover it
    from the stale snapshot. Missing or contradictory current/open identity is likewise
    not authority to terminalize a child; an established Queue lifecycle inconsistency
    remains an operation failure. Any further coordinate conclusion must come from fresh
    canonical/Queue truth rather than treating the earlier retained snapshot as current
    execution authority.

    For an exact current/open attempt freshly confirmed expired/unrecoverable, keep the
    concrete-attempt exclusion held across the complete interrupted-child resolution:
    canonical ``RUNNING`` child -> ``FAILED(failure=None)`` first through
    :func:`fail_training_run` or :func:`fail_evaluation_run`, then the already-defined
    Study-aware RetryPolicy/cancellation or retained terminal-Queue decision, and only
    then closure/cancellation of that exact Queue attempt. ``run_finished_at`` is passed
    unchanged and no failure DTO is invented. The Queue attempt must not be closed first
    merely to simulate a lock; canonical-first ordering remains mandatory. No synthetic
    :class:`QueueAttempt` is ever created.

    This yields the required heartbeat ordering. If heartbeat owns the concrete-attempt
    exclusion first, its authorization and ``heartbeat_attempt()`` lease extension commit
    before release; reconciliation then enters and its fresh authorization recheck sees
    the extended lease and preserves the child whenever that lease is authorized at
    ``queue_at``. If reconciliation owns the exclusion first, it confirms the same
    current attempt expired, terminalizes the child, and closes/cancels the Queue attempt
    before release; a delayed heartbeat then sees closed/stale authority and is
    definitively rejected rather than reviving the lease, even when it began earlier with
    an older ``accepted_at``.

    A sole canonical ``RUNNING`` child with no retained open Queue attempt has no concrete
    Worker lease authority to serialize. Queue loss/rebuild, a missing attempt, or the
    dispatch crash gap after canonical Run persistence but before Queue activation remain
    the existing orphan-interruption path: terminalize the canonical child first and then
    repair Queue state under the same-Study maintenance exclusion, without inventing an
    attempt-level lock or synthetic :class:`QueueAttempt`.

    An already-terminal canonical child is never rewritten. When reconciliation only
    repairs Queue projection from that terminal canonical truth, retain the existing
    canonical-truth rules; the targeted concrete-attempt authority exclusion is required
    where retained Worker lease authority decides whether a canonical ``RUNNING`` child
    may still execute, not as a new blanket gate for every projection repair.

    When no retained terminal Queue decision already governs, retry disposition for an
    unsuccessfully terminal child uses only
    ``retry_policy.after_unsatisfied_attempt(job, queue.attempts_for_job(job.job_id),
    outcome, at=queue_at)``. ``outcome`` is ``"failed"``, ``"cancelled"``, or Evaluation
    ``"completed_partial"`` according to canonical child history. The policy call does
    not require a current attempt row. A non-``None`` retry timestamp maps to
    ``RETRY_WAIT`` and ``None`` maps to ``FAILED``. If the Queue job was already
    ``FAILED`` or ``CANCELLED`` before reconciliation, that committed no-more-work intent
    is preserved after any necessary canonical orphan terminalization and policy is not
    called merely to reconsider it.

    When the interrupted child still has its actual retained open attempt, canonical
    terminalization happens first and then ``queue.close_attempt`` closes that exact row
    to ``RETRY_WAIT`` or ``FAILED``. This covers expired leases without leaving the
    attempt open. When no open attempt survives, the equivalent result is applied with
    ``queue.repair_job_state``. A crash after canonical failure but before Queue closure
    is therefore restartable: the next reconciliation observes the already-terminal
    canonical child and completes only the lagging Queue repair.

    Existing Queue decisions are not casually replayed through policy. In particular,
    ``FAILED`` and ``CANCELLED`` are retained terminal operational intent and remain
    stable while canonical history is unsatisfied; only a stronger satisfying canonical
    fact may repair them to ``SATISFIED``. Queue ``SATISFIED`` is different: it is only a
    projection of canonical success, so when canonical history does not actually satisfy
    the coordinate it must be repaired downward to the state justified by canonical
    history/dependency plus retained retry intent rather than trusted as experiment
    truth. Likewise, an agreed retained ``RETRY_WAIT`` or
    a ``READY`` job whose retained closed attempt already represents the current
    canonical unsuccessful child preserves its committed retry decision/timing instead
    of re-running policy on every Controller restart. Policy is replayed when that
    operational disposition is genuinely unavailable, such as Queue rebuild admission
    with canonical unsuccessful history but no retained attempt corresponding to the
    current child, or when reconciliation itself has just terminalized an orphaned
    ``RUNNING`` child.

    When no satisfying or running child exists, the current canonical unsuccessful child
    is the last same-coordinate terminal Run in the frozen listing's identity order.
    This is not an arbitrary "latest" heuristic: v1 Training/Evaluation Run IDs encode
    the local allocation date plus a per-date allocation sequence
    (``tr-/ev-YYYYMMDD-NNN``), and the frozen list operations return identity order, so
    that order is allocation order for these event IDs. Earlier failed/cancelled/partial
    Runs remain history; they are not deleted or counted as separate planned
    coordinates. If there is no child history at all, no retry policy is invoked merely
    to invent a prior failure.

    After training coordinates have been reconciled, no-child initial operational state
    is repaired from immutable dependency intent:

    - a nonterminal Training job is ``READY``;
    - a nonterminal existing-Model Evaluation job is ``READY``;
    - a nonterminal training-derived Evaluation job remains ``BLOCKED`` while its
      Training dependency can still become satisfied;
    - once the upstream Training coordinate is canonically satisfied, the dependent
      Evaluation becomes/remains runnable (normally released by Queue training
      satisfaction, with reconciliation repair to ``READY`` when necessary).

    Retained logical ``FAILED``/``CANCELLED`` intent is not overwritten simply because a
    job currently has no child Run. A training-derived Evaluation whose upstream
    Training coordinate is permanently unsatisfied remains a final Study-level blocked
    outcome; reconciliation creates no synthetic Evaluation Run and need not rewrite the
    dependent Queue row to a fake Evaluation lifecycle result. In the ordinary retained
    dependency projection it may remain ``BLOCKED``.

    Study finalization is considered only after every expected coordinate has been
    reconciled. The Study stays ``RUNNING`` while any planned coordinate can still make
    normal progress: an active healthy attempt, ``READY`` work, ``RETRY_WAIT`` work, or
    an Evaluation blocked by Training that can still succeed. Once no such progress is
    possible, all expected jobs being canonically ``SATISFIED`` yields
    :attr:`StudyRunStatus.COMPLETED`; any remaining terminal-unsatisfied or permanently
    blocked planned coordinate yields :attr:`StudyRunStatus.COMPLETED_WITH_FAILURES`.
    Child failure alone never produces Study ``FAILED`` or ``CANCELLED`` here.

    The terminal value preserves the canonical Study schema, ID, Study identity,
    ``execution.started_at``, and exact immutable ``plan`` metadata, sets
    ``execution.finished_at=run_finished_at``, and is persisted only through
    :func:`persist_study_run_transition`. Successful finalization returns that terminal
    value; otherwise the still-running canonical Study Run is returned after Queue/model/
    child-history repair.

    This Wave deliberately does not persist a :class:`StudyRunSummary`. The summary is
    optional derived convenience data, while the frozen Queue surface does not retain a
    lossless canonical final-outcome category for every no-Run preflight failure or for
    Queue history lost during rebuild. Guessing ``completed_partial`` versus ``failed``
    versus ``cancelled`` from incomplete operational history would violate the summary
    contract's planned-coordinate semantics. A terminal transition produced here
    therefore uses ``summary=None`` rather than preserving a potentially stale running
    summary or inventing historical-attempt counts. This omission does not affect the
    Study lifecycle decision, which needs only fully satisfied versus permanently
    unsatisfied/blocked coordinates.

    Cross-store ordering is always canonical-first: Model ensure precedes Training Queue
    ``SATISFIED``; canonical Evaluation completion already precedes Evaluation Queue
    ``SATISFIED``; interrupted canonical child failure precedes attempt closure/job
    repair; and Study terminalization happens only after the reconciled coordinate state
    is known. For a retained exact ``ACTIVE``/open attempt freshly confirmed
    expired/unrecoverable, the concrete-attempt authority exclusion additionally remains
    held from the fresh Queue recheck through canonical child failure, the applicable
    Study-aware retry/cancellation decision, and final Queue attempt closure/cancellation.
    Every step is independently replayable. These ordered writes are performed while the
    required same-Study maintenance exclusion remains held; replayability is crash
    recovery, not permission for concurrent normal orchestration mutation.

    Normal startup may therefore open the canonical repository and Queue, enter the
    private reconciliation maintenance phase, reconcile affected running Study Runs, and
    only then enable normal scheduling/API mutation. If Queue reconstruction is required
    after the Controller is already serving work, the implementation must enter the same
    private maintenance exclusion for every affected Study Run before invoking this
    operation and keep normal same-Study acquire/outcome/cancellation mutation excluded
    until reconstruction/reconciliation for that Study completes successfully. A failed
    maintenance pass is not permission to resume scheduling from the partial projection.
    No public ``maintenance mode`` abstraction is required.

    No UnitOfWork, rollback protocol, cross-store transaction, attempt-reconstruction
    DTO, public synchronization abstraction, or future Wave 8-5A/8-5B symbol is
    introduced.
    """

    with orchestration_exclusion():
        return _reconcile_study_run_locked(
            study_run_id,
            run_finished_at,
            queue_at,
            layout,
            filesystem,
            queue,
            retry_policy,
        )



@dataclass(slots=True)
class _TrainingCoordinateState:
    row: StudyPlanRow
    logical: TrainingJob
    history: tuple[TrainingRun, ...]
    completed: TrainingRun | None
    running: TrainingRun | None
    satisfied: bool
    model_id: ModelId | None


@dataclass(slots=True)
class _EvaluationCoordinateState:
    row: StudyPlanRow
    evaluation: StudyPlanEvaluation
    logical: EvaluationJob
    history: tuple[object, ...]
    running: object | None
    satisfying_run_ids: frozenset[object]
    satisfied: bool


def _reconcile_study_run_locked(
    study_run_id: StudyRunId,
    run_finished_at: object,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> StudyRun:
    study_run = read_study_run(study_run_id, layout, filesystem)
    if study_run.status is not StudyRunStatus.RUNNING:
        return study_run
    if study_run.plan is None:
        return study_run

    plan = tuple(read_study_plan(study_run_id, layout, filesystem))
    expected_jobs = derive_study_jobs(study_run_id, plan)
    queue.admit_study_jobs(study_run_id, expected_jobs, admitted_at=queue_at)
    jobs = _validated_queue_projection(
        expected_jobs,
        queue.jobs_for_study_run(study_run_id),
    )

    training_states = _training_states(
        study_run_id,
        plan,
        expected_jobs,
        layout,
        filesystem,
    )
    evaluation_states = _evaluation_states(
        study_run_id,
        plan,
        expected_jobs,
        training_states,
        layout,
        filesystem,
    )

    for state in training_states.values():
        jobs[state.logical] = _reconcile_training_coordinate(
            state,
            jobs[state.logical],
            run_finished_at,
            queue_at,
            layout,
            filesystem,
            queue,
            retry_policy,
        )

    for state in evaluation_states.values():
        jobs[state.logical] = _reconcile_evaluation_coordinate(
            state,
            jobs[state.logical],
            training_states,
            jobs,
            run_finished_at,
            queue_at,
            layout,
            filesystem,
            queue,
            retry_policy,
        )

    all_satisfied = all(state.satisfied for state in training_states.values()) and all(
        state.satisfied for state in evaluation_states.values()
    )
    if all_satisfied:
        target_status = StudyRunStatus.COMPLETED
    elif _progress_is_possible(training_states, evaluation_states, jobs):
        return study_run
    else:
        target_status = StudyRunStatus.COMPLETED_WITH_FAILURES

    terminal = StudyRun(
        schema=study_run.schema,
        id=study_run.id,
        status=target_status,
        study=study_run.study,
        execution=StudyRunExecution(
            started_at=study_run.execution.started_at,
            finished_at=run_finished_at,
        ),
        plan=study_run.plan,
        summary=None,
    )
    persist_study_run_transition(terminal, layout, filesystem)
    return terminal


def _validated_queue_projection(
    expected_jobs: frozenset[StudyJob],
    rows: tuple[QueueJob, ...],
) -> dict[StudyJob, QueueJob]:
    if len(rows) != len(expected_jobs):
        raise LifecycleConflictError("Queue projection does not contain exactly the Study plan coordinates")

    result: dict[StudyJob, QueueJob] = {}
    seen_ids: set[int] = set()
    for row in rows:
        if not isinstance(row, QueueJob):
            raise LifecycleConflictError("Queue projection contains a malformed job row")
        if row.logical not in expected_jobs:
            raise LifecycleConflictError("Queue projection contains work outside the Study plan")
        if row.logical in result:
            raise LifecycleConflictError("Queue projection contains a duplicate logical coordinate")
        if type(row.job_id) is not int or row.job_id <= 0 or row.job_id in seen_ids:
            raise LifecycleConflictError("Queue projection contains an invalid or duplicate job identity")
        if not isinstance(row.status, QueueJobStatus):
            raise LifecycleConflictError("Queue projection contains an invalid job status")
        if not isinstance(row.created_at, str) or not row.created_at:
            raise LifecycleConflictError("Queue projection contains an invalid created_at")
        if not isinstance(row.updated_at, str) or not row.updated_at:
            raise LifecycleConflictError("Queue projection contains an invalid updated_at")
        if row.status is QueueJobStatus.RETRY_WAIT:
            if not isinstance(row.retry_not_before, str) or not row.retry_not_before:
                raise LifecycleConflictError("RETRY_WAIT Queue job lacks retry timing")
        elif row.retry_not_before is not None:
            raise LifecycleConflictError("non-RETRY_WAIT Queue job carries retry timing")
        seen_ids.add(row.job_id)
        result[row.logical] = row

    if set(result) != set(expected_jobs):
        raise LifecycleConflictError("Queue projection is missing a Study plan coordinate")

    for logical, row in result.items():
        if isinstance(logical, TrainingJob):
            if row.dependency_job_id is not None:
                raise LifecycleConflictError("Training Queue job has a dependency")
            continue
        if not isinstance(logical, EvaluationJob):
            raise LifecycleConflictError("Queue projection contains an unsupported logical job kind")
        if logical.training_dependency is None:
            if row.dependency_job_id is not None:
                raise LifecycleConflictError("existing-Model Evaluation Queue job has a dependency")
            continue
        dependency = result.get(TrainingJob(coordinate=logical.training_dependency))
        if dependency is None or row.dependency_job_id != dependency.job_id:
            raise LifecycleConflictError("Evaluation Queue dependency disagrees with Study plan intent")
    return result


def _training_states(
    study_run_id: StudyRunId,
    plan: tuple[StudyPlanRow, ...],
    expected_jobs: frozenset[StudyJob],
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> dict[str, _TrainingCoordinateState]:
    rows = {row.trial: row for row in plan}
    history: dict[str, list[TrainingRun]] = {
        row.trial: [] for row in plan if row.training is not None
    }
    for run in list_training_runs(layout, filesystem):
        if run.study is None or run.study.run != study_run_id:
            continue
        row = rows.get(run.study.trial)
        if row is None or row.training is None or not _training_inputs_match(run, row):
            raise LifecycleConflictError(
                f"Training Run {run.id} disagrees with Study plan lineage or inputs"
            )
        history[row.trial].append(run)

    states: dict[str, _TrainingCoordinateState] = {}
    for row in plan:
        if row.training is None:
            continue
        logical = next(
            job
            for job in expected_jobs
            if isinstance(job, TrainingJob) and job.coordinate.trial == row.trial
        )
        runs = tuple(history[row.trial])
        completed = tuple(run for run in runs if run.status is TrainingRunStatus.COMPLETED)
        running = tuple(run for run in runs if run.status is TrainingRunStatus.RUNNING)
        if len(completed) > 1:
            raise LifecycleConflictError(
                f"multiple completed Training Runs represent {row.trial}"
            )
        if len(running) > 1 or (completed and running):
            raise LifecycleConflictError(
                f"ambiguous active Training Run history represents {row.trial}"
            )

        completed_run = completed[0] if completed else None
        model_id: ModelId | None = None
        if completed_run is not None:
            model = ensure_model_for_completed_training_run(
                completed_run,
                layout,
                filesystem,
            )
            model_id = model.id

        states[row.trial] = _TrainingCoordinateState(
            row=row,
            logical=logical,
            history=runs,
            completed=completed_run,
            running=running[0] if running else None,
            satisfied=completed_run is not None,
            model_id=model_id,
        )
    return states


def _evaluation_states(
    study_run_id: StudyRunId,
    plan: tuple[StudyPlanRow, ...],
    expected_jobs: frozenset[StudyJob],
    training_states: dict[str, _TrainingCoordinateState],
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> dict[tuple[str, str], _EvaluationCoordinateState]:
    rows = {row.trial: row for row in plan}
    histories: dict[tuple[str, str], list[object]] = {
        (row.trial, evaluation.stage): []
        for row in plan
        for evaluation in row.evaluations
    }
    for run in list_evaluation_runs(layout, filesystem):
        if run.study is None or run.study.run != study_run_id:
            continue
        row = rows.get(run.study.trial)
        evaluation = None if row is None else next(
            (item for item in row.evaluations if item.stage == run.study.stage),
            None,
        )
        if row is None or evaluation is None:
            raise LifecycleConflictError(
                f"Evaluation Run {run.id} lies outside the Study plan"
            )
        training = training_states.get(row.trial)
        if not _evaluation_inputs_match(run, row, evaluation, training):
            raise LifecycleConflictError(
                f"Evaluation Run {run.id} disagrees with Study plan lineage or inputs"
            )
        histories[(row.trial, evaluation.stage)].append(run)

    states: dict[tuple[str, str], _EvaluationCoordinateState] = {}
    for row in plan:
        for evaluation in row.evaluations:
            logical = next(
                job
                for job in expected_jobs
                if isinstance(job, EvaluationJob)
                and job.coordinate.trial == row.trial
                and job.coordinate.stage == evaluation.stage
            )
            runs = tuple(histories[(row.trial, evaluation.stage)])
            completed = tuple(
                run for run in runs if run.status is EvaluationRunStatus.COMPLETED
            )
            running = tuple(
                run for run in runs if run.status is EvaluationRunStatus.RUNNING
            )
            if len(running) > 1 or (completed and running):
                raise LifecycleConflictError(
                    f"ambiguous active Evaluation Run history represents {row.trial}/{evaluation.stage}"
                )
            states[(row.trial, evaluation.stage)] = _EvaluationCoordinateState(
                row=row,
                evaluation=evaluation,
                logical=logical,
                history=runs,
                running=running[0] if running else None,
                satisfying_run_ids=frozenset(run.id for run in completed),
                satisfied=bool(completed),
            )
    return states


def _training_inputs_match(run: TrainingRun, row: StudyPlanRow) -> bool:
    training = row.training
    assert training is not None
    return (
        run.study is not None
        and run.study.trial == row.trial
        and run.corpus == training.corpus
        and run.architecture == training.architecture
        and run.train_protocol == training.protocol
        and type(run.execution.seed) is int
        and run.execution.seed == training.seed
        and _same_public_value(run.parameters, training.parameters)
    )


def _evaluation_inputs_match(
    run: object,
    row: StudyPlanRow,
    evaluation: StudyPlanEvaluation,
    training: _TrainingCoordinateState | None,
) -> bool:
    expected_model = row.model
    if row.training is not None:
        if training is None or not training.satisfied or training.model_id is None:
            return False
        expected_model = training.model_id
    return (
        run.study is not None
        and run.study.trial == row.trial
        and run.study.stage == evaluation.stage
        and run.model == expected_model
        and run.corpus == evaluation.corpus
        and run.evaluation_protocol == evaluation.protocol
        and _same_public_value(run.parameters, evaluation.parameters)
    )


def _same_public_value(left: object, right: object) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _same_public_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_public_value(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _reconcile_training_coordinate(
    state: _TrainingCoordinateState,
    job: QueueJob,
    run_finished_at: object,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> QueueJob:
    if state.satisfied:
        assert state.completed is not None
        return _repair_satisfied_job(
            job,
            frozenset({state.completed.id}),
            queue_at,
            queue,
        )

    if state.running is not None:
        job, preserved, failed = _resolve_running_child(
            job,
            state.running,
            run_finished_at,
            queue_at,
            layout,
            filesystem,
            queue,
            retry_policy,
        )
        if preserved:
            return job
        state.history = tuple(
            failed if run.id == failed.id else run for run in state.history
        )
        state.running = None
        return job

    if state.history:
        terminal = state.history[-1]
        return _apply_unsatisfied_terminal(
            job,
            terminal,
            _unsatisfied_outcome(terminal),
            queue_at,
            queue,
            retry_policy,
        )

    return _repair_no_child_job(
        job,
        QueueJobStatus.READY,
        queue_at,
        queue,
        preserve_retry=True,
    )


def _reconcile_evaluation_coordinate(
    state: _EvaluationCoordinateState,
    job: QueueJob,
    training_states: dict[str, _TrainingCoordinateState],
    jobs: dict[StudyJob, QueueJob],
    run_finished_at: object,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> QueueJob:
    if state.satisfied:
        return _repair_satisfied_job(
            job,
            state.satisfying_run_ids,
            queue_at,
            queue,
        )

    if state.running is not None:
        job, preserved, failed = _resolve_running_child(
            job,
            state.running,
            run_finished_at,
            queue_at,
            layout,
            filesystem,
            queue,
            retry_policy,
        )
        if preserved:
            return job
        state.history = tuple(
            failed if run.id == failed.id else run for run in state.history
        )
        state.running = None
        return job

    if state.history:
        terminal = state.history[-1]
        return _apply_unsatisfied_terminal(
            job,
            terminal,
            _unsatisfied_outcome(terminal),
            queue_at,
            queue,
            retry_policy,
        )

    dependency = state.logical.training_dependency
    if dependency is None:
        return _repair_no_child_job(
            job,
            QueueJobStatus.READY,
            queue_at,
            queue,
            preserve_retry=True,
        )

    upstream = training_states[dependency.trial]
    upstream_job = jobs[TrainingJob(coordinate=dependency)]
    if upstream.satisfied:
        target = QueueJobStatus.READY
        preserve_retry = True
    elif upstream_job.status in {
        QueueJobStatus.ACTIVE,
        QueueJobStatus.READY,
        QueueJobStatus.RETRY_WAIT,
    }:
        target = QueueJobStatus.BLOCKED
        preserve_retry = False
    elif upstream_job.status in {
        QueueJobStatus.FAILED,
        QueueJobStatus.CANCELLED,
    }:
        target = QueueJobStatus.BLOCKED
        preserve_retry = False
    else:
        raise LifecycleConflictError("Training dependency has an incoherent reconciled state")
    return _repair_no_child_job(
        job,
        target,
        queue_at,
        queue,
        preserve_retry=preserve_retry,
    )


def _repair_satisfied_job(
    job: QueueJob,
    satisfying_run_ids: frozenset[object],
    queue_at: str,
    queue: QueuePort,
) -> QueueJob:
    open_attempt = queue.open_attempt_for_job(job.job_id)
    if open_attempt is not None:
        if not isinstance(open_attempt, QueueAttempt):
            raise LifecycleConflictError("Queue returned a malformed open attempt")
        if job.status is not QueueJobStatus.ACTIVE:
            raise LifecycleConflictError("non-ACTIVE Queue job retains an open attempt")
        if open_attempt.run_id not in satisfying_run_ids:
            raise LifecycleConflictError("open Queue attempt disagrees with satisfying canonical child")
        repaired, _ = queue.close_attempt(
            open_attempt.attempt_id,
            target_status=QueueJobStatus.SATISFIED,
            finished_at=queue_at,
        )
        return repaired
    if job.status is QueueJobStatus.SATISFIED:
        return job
    return queue.repair_job_state(
        job.job_id,
        status=QueueJobStatus.SATISFIED,
        retry_not_before=None,
        at=queue_at,
    )


def _resolve_running_child(
    job: QueueJob,
    running: object,
    run_finished_at: object,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> tuple[QueueJob, bool, object]:
    retained = queue.open_attempt_for_job(job.job_id)
    if retained is None or job.status is not QueueJobStatus.ACTIVE:
        if retained is not None:
            raise LifecycleConflictError("non-ACTIVE Queue job retains open Worker authority")
        failed = _fail_running_child(running, run_finished_at, layout, filesystem)
        if job.status in {QueueJobStatus.FAILED, QueueJobStatus.CANCELLED}:
            return job, False, failed
        repaired = _policy_disposition(
            job,
            failed,
            "failed",
            queue_at,
            queue,
            retry_policy,
            open_attempt=None,
        )
        return repaired, False, failed

    if not isinstance(retained, QueueAttempt) or retained.run_id != running.id:
        raise LifecycleConflictError("ACTIVE Queue attempt disagrees with canonical RUNNING child")

    with orchestration_exclusion():
        fresh = queue.attempt_by_id(retained.attempt_id)
        if not _same_current_attempt(retained, fresh):
            raise LifecycleConflictError("retained ACTIVE attempt is no longer the current attempt")
        fresh_job = queue.job_by_id(job.job_id)
        if (
            not isinstance(fresh_job, QueueJob)
            or fresh_job.logical != job.logical
            or fresh_job.status is not QueueJobStatus.ACTIVE
        ):
            raise LifecycleConflictError("retained ACTIVE job is no longer current")
        current_open = queue.open_attempt_for_job(job.job_id)
        if not _same_current_attempt(fresh, current_open):
            raise LifecycleConflictError("retained ACTIVE attempt is no longer the open attempt")
        authorized = queue.authorized_open_attempt(
            fresh.attempt_id,
            fresh.lease_token,
            as_of=queue_at,
        )
        if authorized is not None:
            if not _same_current_attempt(fresh, authorized):
                raise LifecycleConflictError("lease authorization returned a different attempt")
            return fresh_job, True, running

        failed = _fail_running_child(running, run_finished_at, layout, filesystem)
        repaired = _policy_disposition(
            fresh_job,
            failed,
            "failed",
            queue_at,
            queue,
            retry_policy,
            open_attempt=fresh,
        )
        return repaired, False, failed


def _same_current_attempt(left: object, right: object) -> bool:
    return (
        isinstance(left, QueueAttempt)
        and isinstance(right, QueueAttempt)
        and right.finished_at is None
        and left.attempt_id == right.attempt_id
        and left.job_id == right.job_id
        and left.run_id == right.run_id
        and left.lease_token == right.lease_token
    )


def _fail_running_child(
    run: object,
    run_finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> object:
    if isinstance(run, TrainingRun):
        return fail_training_run(
            run,
            run_finished_at,
            layout,
            filesystem,
            failure=None,
        )
    return fail_evaluation_run(
        run,
        run_finished_at,
        layout,
        filesystem,
        failure=None,
    )


def _apply_unsatisfied_terminal(
    job: QueueJob,
    run: object,
    outcome: str,
    queue_at: str,
    queue: QueuePort,
    retry_policy: RetryPolicy,
) -> QueueJob:
    open_attempt = queue.open_attempt_for_job(job.job_id)
    if open_attempt is not None:
        if (
            not isinstance(open_attempt, QueueAttempt)
            or job.status is not QueueJobStatus.ACTIVE
            or open_attempt.run_id != run.id
        ):
            raise LifecycleConflictError("open Queue attempt disagrees with terminal canonical child")
        return _policy_disposition(
            job,
            run,
            outcome,
            queue_at,
            queue,
            retry_policy,
            open_attempt=open_attempt,
        )

    if job.status in {QueueJobStatus.FAILED, QueueJobStatus.CANCELLED}:
        return job
    if job.status is QueueJobStatus.RETRY_WAIT:
        return job
    if job.status is QueueJobStatus.READY and _has_closed_attempt_for_run(
        job.job_id,
        run.id,
        queue,
    ):
        return job
    return _policy_disposition(
        job,
        run,
        outcome,
        queue_at,
        queue,
        retry_policy,
        open_attempt=None,
    )


def _has_closed_attempt_for_run(job_id: int, run_id: object, queue: QueuePort) -> bool:
    found = False
    for attempt in queue.attempts_for_job(job_id):
        if not isinstance(attempt, QueueAttempt) or attempt.job_id != job_id:
            raise LifecycleConflictError("Queue returned malformed attempt history")
        if attempt.run_id != run_id:
            continue
        if attempt.finished_at is None:
            return False
        found = True
    return found


def _policy_disposition(
    job: QueueJob,
    run: object,
    outcome: str,
    queue_at: str,
    queue: QueuePort,
    retry_policy: RetryPolicy,
    *,
    open_attempt: QueueAttempt | None,
) -> QueueJob:
    attempts = queue.attempts_for_job(job.job_id)
    decision = retry_policy.after_unsatisfied_attempt(
        job,
        attempts,
        outcome,
        at=queue_at,
    )
    retry_not_before = decision.retry_not_before
    target = (
        QueueJobStatus.RETRY_WAIT
        if retry_not_before is not None
        else QueueJobStatus.FAILED
    )
    if open_attempt is not None:
        repaired, _ = queue.close_attempt(
            open_attempt.attempt_id,
            target_status=target,
            finished_at=queue_at,
            retry_not_before=retry_not_before,
        )
        return repaired
    return queue.repair_job_state(
        job.job_id,
        status=target,
        retry_not_before=retry_not_before,
        at=queue_at,
    )


def _repair_no_child_job(
    job: QueueJob,
    target: QueueJobStatus,
    queue_at: str,
    queue: QueuePort,
    *,
    preserve_retry: bool,
) -> QueueJob:
    if queue.open_attempt_for_job(job.job_id) is not None:
        raise LifecycleConflictError("Queue job without canonical child retains an open attempt")
    if job.status in {QueueJobStatus.FAILED, QueueJobStatus.CANCELLED}:
        return job
    if preserve_retry and job.status is QueueJobStatus.RETRY_WAIT:
        return job
    if job.status is target:
        return job
    return queue.repair_job_state(
        job.job_id,
        status=target,
        retry_not_before=None,
        at=queue_at,
    )


def _unsatisfied_outcome(run: object) -> str:
    if isinstance(run, TrainingRun):
        if run.status is TrainingRunStatus.FAILED:
            return "failed"
        if run.status is TrainingRunStatus.CANCELLED:
            return "cancelled"
    else:
        if run.status is EvaluationRunStatus.FAILED:
            return "failed"
        if run.status is EvaluationRunStatus.CANCELLED:
            return "cancelled"
        if run.status is EvaluationRunStatus.COMPLETED_PARTIAL:
            return "completed_partial"
    raise LifecycleConflictError("canonical child does not represent an unsatisfied terminal outcome")


def _progress_is_possible(
    training_states: dict[str, _TrainingCoordinateState],
    evaluation_states: dict[tuple[str, str], _EvaluationCoordinateState],
    jobs: dict[StudyJob, QueueJob],
) -> bool:
    for state in training_states.values():
        if state.satisfied:
            continue
        status = jobs[state.logical].status
        if status in {
            QueueJobStatus.ACTIVE,
            QueueJobStatus.READY,
            QueueJobStatus.RETRY_WAIT,
        }:
            return True
        if status not in {QueueJobStatus.FAILED, QueueJobStatus.CANCELLED}:
            raise LifecycleConflictError("reconciled Training coordinate has incoherent state")

    for state in evaluation_states.values():
        if state.satisfied:
            continue
        status = jobs[state.logical].status
        if status in {
            QueueJobStatus.ACTIVE,
            QueueJobStatus.READY,
            QueueJobStatus.RETRY_WAIT,
        }:
            return True
        if status is QueueJobStatus.BLOCKED:
            dependency = state.logical.training_dependency
            if dependency is None:
                raise LifecycleConflictError("existing-Model Evaluation remains BLOCKED")
            upstream = jobs[TrainingJob(coordinate=dependency)].status
            if upstream in {
                QueueJobStatus.ACTIVE,
                QueueJobStatus.READY,
                QueueJobStatus.RETRY_WAIT,
            }:
                return True
            if upstream not in {QueueJobStatus.FAILED, QueueJobStatus.CANCELLED}:
                raise LifecycleConflictError("blocked Evaluation has incoherent dependency state")
            continue
        if status not in {QueueJobStatus.FAILED, QueueJobStatus.CANCELLED}:
            raise LifecycleConflictError("reconciled Evaluation coordinate has incoherent state")
    return False
