"""Successful Controller dispatch boundary for one selected READY Study-derived job.

This module connects one persisted :class:`QueueJob` plus the authoritative finalized
:class:`StudyPlan` to the already-frozen concrete launch boundaries. It owns only the
successful dispatch path:

``plan coordinate -> concrete preflight -> canonical running child Run -> Queue attempt
activation -> Worker assignment projection``.

Queue carries no execution payload. Training/Evaluation inputs are always recovered from
the exact immutable Study plan row/stage named by the selected logical job coordinate.
For training-derived Evaluation, the upstream Model is recovered from canonical completed
Training child history and deterministic Model identity rather than from Queue state.

This is not a full Worker acquire handler. Acquire-token replay lookup, Worker
one-open-attempt validation, compatible-job selection, lease-token generation, retry
policy, preflight-failure Queue disposition, no-work/rejection mapping, heartbeat,
result handling, and reconciliation repair remain outside this module. No Dispatcher,
generic dispatch abstraction, reservation, scheduler, or cross-store transaction is
introduced.
"""

from __future__ import annotations

from datetime import date

from ..evaluation.preflight import preflight_evaluation
from ..evaluation.run import EvaluationRunStudyLineage
from ..model.identity import model_id_for_training_run
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.run_persistence import (
    allocate_evaluation_run,
    allocate_training_run,
    list_training_runs,
)
from ..study.plan import StudyPlan
from ..training.preflight import preflight_training
from ..training.run import TrainingRunStatus, TrainingRunStudyLineage
from .asset_source import ImmutableAssetSource
from .assignment import project_evaluation_assignment, project_training_assignment
from .jobs import EvaluationJob, TrainingJob, TrainingJobCoordinate
from .queue import QueueJob, QueueJobStatus
from .queue_ports import QueuePort
from .worker_api import EvaluationAssignment, TrainingAssignment


def dispatch_training_job(
    job: QueueJob,
    plan: StudyPlan,
    allocation_date: date,
    started_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    assets: ImmutableAssetSource,
    *,
    worker_id: str,
    acquire_token: str,
    lease_token: str,
    activated_at: str,
) -> TrainingAssignment:
    """Dispatch one selected ``READY`` Training job through the successful boundary.

    ``job`` is the already-selected persisted Queue row and must have
    ``status is QueueJobStatus.READY`` with ``logical`` exactly a :class:`TrainingJob`.
    The function does not select or reserve work and does not look up ``acquire_token``
    or another open Worker attempt; the later policy-aware acquire orchestration owns
    those checks before entering this boundary.

    ``plan`` is the canonical finalized immutable Study plan for
    ``job.logical.coordinate.study_run``. The function locates the exact single row whose
    ``trial`` equals ``job.logical.coordinate.trial``. Missing or duplicate matches are
    operation failures. The selected row must be training-derived exactly:
    ``row.training is not None`` and ``row.model is None``. Queue contributes no Corpus,
    Architecture, Protocol, seed, parameter mapping, Model, or other execution payload.

    Concrete launch input is taken only from that :class:`StudyPlanTraining` and passed
    unchanged to :func:`preflight_training`:

    - ``corpus_id = row.training.corpus``;
    - ``architecture_id = row.training.architecture``;
    - ``train_protocol_id = row.training.protocol``;
    - ``seed = row.training.seed``;
    - ``parameter_overrides = row.training.parameters``.

    Although plan parameters are already complete, concrete preflight must still run.
    Successful ``preflight.parameters`` must equal ``row.training.parameters`` exactly;
    disagreement is an operation failure before any child Run or Queue attempt exists.
    No shortcut may construct a :class:`TrainingPreflight` directly from plan values.

    After successful preflight and exact parameter agreement, ordering is fixed:

    1. :func:`allocate_training_run` is called with the successful preflight and
       ``study=TrainingRunStudyLineage(run=coordinate.study_run, trial=row.trial)``.
       Successful return means the schema-valid canonical ``RUNNING`` Training Run is
       already committed.
    2. ``queue.activate_attempt()`` is called for ``job.job_id`` and that fresh
       ``run.id``, copying exactly the caller-supplied ``worker_id``, ``acquire_token``,
       ``lease_token``, and ``activated_at``.
    3. Only after Queue activation commits may :func:`project_training_assignment`
       expose the frozen Worker assignment from the same preflight, committed Run,
       returned Queue attempt, and ``assets`` source.

    ``allocation_date`` and ``started_at`` are passed unchanged to Run allocation.
    ``lease_token`` generation and timestamp construction/normalization are not owned by
    this function.

    A concrete preflight failure, including failure to reproduce the plan's complete
    parameter mapping, creates no child Run and no Queue attempt. This function does not
    call ``defer_ready_job()`` or ``fail_ready_job()``; the selected READY job's retry or
    terminal disposition belongs to later policy-aware acquire orchestration.

    Queue SQLite and canonical Run persistence are not one transaction. If execution is
    interrupted after canonical ``RUNNING`` Run commit but before Queue activation
    commit, the orphaned-running-Run gap is preserved for later reconciliation; this
    boundary must not invent rollback, Run reservation, or a cross-store transaction.
    Likewise, later activation/projection failure is surfaced as an operation failure
    without compensating mutation invented here.

    Lost acquire replay does not require persisting this returned assignment. A later
    handler can read the persisted :class:`QueueAttempt`, recover its Queue job and
    canonical Study plan, reconstruct the same concrete Training preflight from the plan
    coordinate, read the attempt's canonical Run, and rerun the frozen assignment
    projection for the same still-authorized attempt.
    """

    ...


def dispatch_evaluation_job(
    job: QueueJob,
    plan: StudyPlan,
    allocation_date: date,
    started_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    assets: ImmutableAssetSource,
    *,
    worker_id: str,
    acquire_token: str,
    lease_token: str,
    activated_at: str,
) -> EvaluationAssignment:
    """Dispatch one selected ``READY`` Evaluation job through the successful boundary.

    ``job`` must have ``status is QueueJobStatus.READY`` and ``logical`` exactly an
    :class:`EvaluationJob`. This operation performs no Queue selection, acquire-token
    lookup, prior-Worker-attempt validation, retry judgment, or lease generation.

    ``plan`` is the canonical finalized immutable Study plan for
    ``job.logical.coordinate.study_run``. The function locates exactly one row whose
    ``trial`` equals ``job.logical.coordinate.trial`` and then exactly one evaluation
    entry whose ``stage`` equals ``job.logical.coordinate.stage``. Missing or duplicate
    row/stage matches are operation failures. Corpus, Evaluation Protocol, parameters,
    and Model source are read only from those plan values; Queue remains coordinate-only
    operational state.

    Model resolution has two exact cases:

    **Existing-Model row.** When ``row.model is not None``, ``row.training`` must be
    ``None`` and ``job.logical.training_dependency`` must be ``None``. The exact
    ``row.model`` value is used as the Evaluation ``model_id``; no substitute or latest
    Model may be selected.

    **Training-derived row.** When ``row.training is not None``, ``row.model`` must be
    ``None`` and ``job.logical.training_dependency`` must equal
    ``TrainingJobCoordinate(study_run=coordinate.study_run, trial=row.trial)``. Queue
    readiness is insufficient evidence that the upstream Model exists. The function
    must inspect canonical history through :func:`list_training_runs` and identify the
    completed Training child for the same Study coordinate.

    Among canonical Training Runs, failed/cancelled/running retry history does not
    satisfy the dependency. There must be exactly one ``COMPLETED`` Run whose
    ``study.run`` and ``study.trial`` equal the selected Evaluation coordinate. More
    than one such completed Run is ambiguous authority and is an operation failure;
    none means the upstream dependency is not canonically satisfied. The sole completed
    candidate must additionally agree exactly with the training intent in this plan row:

    - ``run.corpus == row.training.corpus``;
    - ``run.architecture == row.training.architecture``;
    - ``run.train_protocol == row.training.protocol``;
    - ``run.execution.seed == row.training.seed``;
    - ``run.parameters == row.training.parameters``.

    Any mismatch is an operation failure rather than permission to choose another Run.
    The exact Model identity is then derived only through
    :func:`model_id_for_training_run(run.id)`. This dispatch boundary does not create or
    repair a missing Model. Passing that deterministic ID to
    :func:`preflight_evaluation` reuses frozen Model resolution and therefore requires
    the canonical Model plus its completed Training lineage and learned bytes to exist
    and satisfy concrete Evaluation preflight.

    Concrete Evaluation preflight is always performed with exactly:

    - the resolved Model ID from the applicable case above;
    - ``corpus_id = stage.corpus``;
    - ``evaluation_protocol_id = stage.protocol``;
    - ``parameter_overrides = stage.parameters``.

    Successful ``preflight.parameters`` must equal ``stage.parameters`` exactly before
    Run allocation. Complete plan parameters therefore do not bypass the frozen
    preflight boundary.

    After successful preflight and exact parameter agreement, ordering is fixed:

    1. :func:`allocate_evaluation_run` is called with the successful preflight and
       ``study=EvaluationRunStudyLineage(run=coordinate.study_run, trial=row.trial,
       stage=stage.stage)``. Successful return means the schema-valid canonical
       ``RUNNING`` Evaluation Run is already committed.
    2. ``queue.activate_attempt()`` is called for ``job.job_id`` and that fresh
       ``run.id``, copying exactly ``worker_id``, ``acquire_token``, ``lease_token``, and
       ``activated_at`` from this call.
    3. Only after Queue activation commits may :func:`project_evaluation_assignment`
       expose the frozen Worker assignment from the same preflight, committed Run,
       returned Queue attempt, and ``assets`` source.

    ``allocation_date`` and ``started_at`` are passed unchanged to Evaluation Run
    allocation. Lease-token generation and timestamp normalization remain caller-owned.

    Any failure before successful Run allocation—including missing/ambiguous upstream
    canonical Training history, absent deterministic Model, Model/preflight failure, or
    parameter disagreement—creates no Evaluation Run and no Queue attempt. This module
    does not decide between ``retry_wait`` and ``failed`` and must not call
    ``defer_ready_job()`` or ``fail_ready_job()``. Later policy-aware acquire
    orchestration owns that Queue disposition.

    The canonical-Run-before-Queue crash gap is intentional. No cross-store transaction,
    rollback, reservation, or compensation layer is introduced. A failure after Run
    allocation or Queue activation is surfaced as an operation failure and left to the
    existing/later reconciliation and policy boundaries rather than silently rewriting
    canonical history.

    Lost acquire replay remains feasible without assignment persistence. From the
    persisted Queue attempt, a later handler can recover the Queue job and immutable
    Study plan, re-resolve the exact existing Model or repeat the same canonical
    Training-history/deterministic-Model derivation, reconstruct the same successful
    Evaluation preflight, read the attempt's canonical Run, and rerun frozen assignment
    projection for the same still-authorized attempt.
    """

    ...
