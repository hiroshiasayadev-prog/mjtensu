"""Controller-side Study launch setup boundary.

This module composes one already-successful :class:`PreparedStudyExecution` through
canonical Study Run allocation, complete Study plan materialization/finalization,
logical job derivation, and all-or-nothing Queue admission. Successful return means
that both the immutable plan metadata and the complete Queue admission are established;
it does not wait for or dispatch any child Training/Evaluation Run.

The boundary deliberately introduces no Study/Launch manager, UnitOfWork, transaction
coordinator, generic orchestration context, retry policy, reconciliation workflow,
Worker assignment, child execution, Study completion judgment, or Public API DTO.
Filesystem-backed canonical Study state and Controller-local Queue state are not wrapped
in one cross-store transaction. Ordered writes plus later reconciliation remain the
recovery model fixed by the lower-wave contracts.
"""

from __future__ import annotations

from datetime import date

from ..common.ids import StudyRunId
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.run_persistence import (
    allocate_study_run,
    finalize_study_plan,
    persist_study_run_transition,
    read_study_run,
)
from ..study.expansion import materialize_study_plan
from ..study.preflight import PreparedStudyExecution
from ..study.run import StudyRun, StudyRunExecution, StudyRunStatus
from .jobs import derive_study_jobs
from .queue_ports import QueuePort


class StudyExecutionSetupError(Exception):
    """Study launch setup failed after one Study Run was successfully allocated.

    ``study_run_id`` is the exact identity returned by the successful
    :func:`allocate_study_run` call in this launch invocation. The exception carries no
    generic orchestration classification, retry policy, transport status, or frozen
    payload for the triggering failure. The relevant underlying exception remains
    available through normal Python exception chaining.

    This exception is never used for failures that occur before
    :func:`allocate_study_run` successfully returns, because this boundary does not yet
    own an allocated Study Run identity in that case.
    """

    study_run_id: StudyRunId

    def __init__(self, study_run_id: StudyRunId) -> None:
        self.study_run_id = study_run_id
        super().__init__()


def launch_study_execution(
    prepared: PreparedStudyExecution,
    allocation_date: date,
    started_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    *,
    admitted_at: str,
    failed_at: object,
) -> StudyRun:
    """Launch one already-preflighted Study through immutable Queue admission.

    ``prepared`` must be the exact successful
    :class:`~mldb.skeleton.study.preflight.PreparedStudyExecution` produced before this
    operation. This function does not perform or repeat Study preflight. Consequently a
    preflight failure never enters this boundary and never allocates a Study Run.

    The successful ordering is fixed exactly as follows:

    1. call :func:`allocate_study_run` with ``prepared``, ``allocation_date``,
       ``started_at``, ``layout``, and ``filesystem``. Successful allocation returns the
       already-persisted canonical ``RUNNING`` Study Run with ``plan is None``;
    2. call :func:`materialize_study_plan` exactly from the retained prepared metadata::

           materialize_study_plan(
               prepared.study.metadata,
               prepared.train_protocol.metadata
               if prepared.train_protocol is not None
               else None,
               {
                   protocol_id: handle.metadata
                   for protocol_id, handle
                   in prepared.evaluation_protocols.items()
               },
           )

       No Study, Protocol, Corpus, Architecture, or Model is re-resolved here and Queue
       does not participate in plan materialization;
    3. call :func:`finalize_study_plan` with the allocated Study Run ID and the complete
       materialized plan. Successful finalization returns the canonical ``RUNNING``
       Study Run whose immutable ``plan`` metadata is present;
    4. call :func:`derive_study_jobs` with that Study Run ID and the same complete
       materialized plan. The resulting complete :class:`frozenset` is the only logical
       work set supplied to Queue;
    5. call ``queue.admit_study_jobs(study_run.id, jobs, admitted_at=admitted_at)``.
       Queue admission is one Queue transaction/all-or-nothing under
       :class:`QueuePort`; Queue receives no :class:`StudyPlan` and must not independently
       reinterpret Study execution intent.

    Only after Queue admission returns successfully does this operation return the
    finalized canonical :class:`StudyRun`. Its status remains
    :attr:`StudyRunStatus.RUNNING` and its ``plan`` metadata is present. Queue rows are
    intentionally not part of the public return value, and this success does not imply
    that any child Training/Evaluation work has been dispatched or completed.

    ``admitted_at`` is passed unchanged to the Queue contract and therefore uses that
    contract's exact UTC RFC3339 representation. ``failed_at`` is the caller-supplied
    opaque Study Run terminal timestamp used only when a setup failure is caught after
    successful Study Run allocation. This module does not introduce a clock or timestamp
    normalization abstraction.

    After :func:`allocate_study_run` has returned successfully, its exact identity is
    retained independently from later local Run values, conceptually::

        allocated_id = run.id

    ``allocated_id`` is stable for the remainder of this launch invocation. Any exception
    from plan materialization, plan finalization, logical job derivation, or Queue
    admission is a Study-level setup/orchestration failure. The allocated Study Run is
    never deleted. Before surfacing that failure, the handler must re-read the exact
    canonical Study Run through
    :func:`read_study_run` and construct one terminal value preserving its immutable
    identity and current plan fact exactly::

        StudyRun(
            schema=current.schema,
            id=current.id,
            status=StudyRunStatus.FAILED,
            study=current.study,
            execution=StudyRunExecution(
                started_at=current.execution.started_at,
                finished_at=failed_at,
            ),
            plan=current.plan,
            summary=current.summary,
        )

    That value is persisted only through :func:`persist_study_run_transition`. There is
    no Study Run v1 failure-text field, so none is invented. Re-reading canonical state
    is required rather than relying on a stale local value: if finalization never made
    plan metadata authoritative, ``current.plan`` remains ``None`` and the ``FAILED``
    Run keeps ``plan=None``; if finalization already committed immutable plan metadata,
    including before a later Queue-admission failure, the ``FAILED`` Run preserves that
    exact plan metadata. Complete plan bytes written without matching Run plan metadata
    remain unfinalized/non-authoritative and do not get promoted during failure handling.

    If that canonical failure transition succeeds, the original setup exception is not
    propagated directly. This boundary raises conceptually::

        raise StudyExecutionSetupError(allocated_id) from setup_exception

    so the triggering setup exception remains available through normal Python exception
    chaining.

    This caught-failure cleanup is not a cross-store transaction and does not attempt to
    roll back Queue or filesystem state. Repeated *agreed* Queue admission remains safe
    under :class:`QueuePort`, but retry/reconciliation policy is outside this function.
    If the cleanup re-read, construction/validation path, or terminal persistence itself
    raises, this boundary still raises :class:`StudyExecutionSetupError` carrying the
    same ``allocated_id``, conceptually chained from that cleanup failure. It must not
    claim that the canonical Run became ``FAILED`` when terminal persistence did not
    commit. Because the cleanup failure occurs while handling the original setup
    exception, ordinary exception context may additionally retain that triggering failure
    for diagnostics.

    If the Controller process crashes before this handler can return an exception, a
    canonical ``RUNNING`` Study Run may remain with either no finalized plan or an
    immutable finalized plan; later Study resume/reconciliation owns that case. No
    durable error receipt or transaction journal is introduced here.

    Failure before :func:`allocate_study_run` successfully returns is not an
    after-allocation setup failure owned here and is propagated unchanged; no
    :class:`StudyExecutionSetupError` is raised because no returned Study Run identity is
    known by this boundary. No Study Run identity is scanned for, inferred, or fabricated.
    Child Run dispatch, Worker execution/assignment, retry, reconciliation, Study
    completion judgment, and Public API response/transport projection remain outside this
    operation.
    """

    allocated_run = allocate_study_run(
        prepared, allocation_date, started_at, layout, filesystem
    )
    allocated_id = allocated_run.id

    try:
        plan = materialize_study_plan(
            prepared.study.metadata,
            prepared.train_protocol.metadata
            if prepared.train_protocol is not None
            else None,
            {
                protocol_id: handle.metadata
                for protocol_id, handle in prepared.evaluation_protocols.items()
            },
        )
        finalized_run = finalize_study_plan(
            allocated_id, plan, layout, filesystem
        )
        jobs = derive_study_jobs(allocated_id, plan)
        queue.admit_study_jobs(
            allocated_id, jobs, admitted_at=admitted_at
        )
        return finalized_run
    except Exception as setup_exception:
        try:
            current = read_study_run(allocated_id, layout, filesystem)
            failed = StudyRun(
                schema=current.schema,
                id=current.id,
                status=StudyRunStatus.FAILED,
                study=current.study,
                execution=StudyRunExecution(
                    started_at=current.execution.started_at,
                    finished_at=failed_at,
                ),
                plan=current.plan,
                summary=current.summary,
            )
            persist_study_run_transition(failed, layout, filesystem)
        except Exception as cleanup_exception:
            raise StudyExecutionSetupError(allocated_id) from cleanup_exception
        raise StudyExecutionSetupError(allocated_id) from setup_exception
