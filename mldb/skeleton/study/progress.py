"""Read-only Study Run observation and derived application-progress signatures.

This module fixes the lower-level observation boundary used by the later public
Controller ``get_study_run`` operation. It combines one exact canonical Study Run,
its verified immutable plan when finalized, canonical Study-linked child Run history,
existing deterministic Models, and the replaceable operational Queue projection into a
stable application progress view.

Canonical MLDB history always outranks Queue state for accepted success facts. Queue is
consulted only to classify work that is not canonically satisfied into the Controller
progress vocabulary. ``ACTIVE`` additionally requires current Worker lease authority at
the caller-supplied Queue observation cutoff. Observation never repairs Queue state,
creates a missing Model, terminalizes a Run, retries work, reconciles a Study, or mutates
any canonical or operational record.

The boundary deliberately exposes no ProgressManager, generic snapshot/query service,
DAG engine, polling/watch API, event timeline, Queue-row DTO, Worker/lease identity, or
historical attempt inventory.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.ids import EvaluationRunId, StudyRunId, TrainingRunId
from ..evaluation.run import EvaluationRunStatus
from ..model.identity import model_id_for_training_run
from ..orchestration.jobs import derive_study_jobs
from ..orchestration.queue import QueueJobStatus
from ..orchestration.queue_ports import QueuePort
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.resolution import resolve_model
from ..runtime.run_persistence import (
    list_evaluation_runs,
    list_training_runs,
    read_study_plan,
    read_study_run,
)
from ..training.run import TrainingRunStatus
from .run import StudyRun


@dataclass(frozen=True, slots=True)
class StudyRunProgressCounts:
    """Stable Controller-level progress categories for one execution kind.

    ``total`` is the number of immutable planned coordinates of this kind and
    ``satisfied`` is the number proven complete from canonical MLDB child history.
    Those two values never come from Queue counters.

    ``active``, ``waiting``, ``blocked``, and ``unsatisfied_terminal`` are the only
    operational progress categories exposed by this boundary:

    - ``active`` means an unsatisfied coordinate is represented by a coherent Queue
      ``ACTIVE`` projection whose single open attempt is currently lease-authorized at
      the supplied observation cutoff and refers to the corresponding sole canonical
      ``RUNNING`` child;
    - ``waiting`` covers coherent Queue ``READY`` and ``RETRY_WAIT`` projections that
      are actually runnable (or retry-delayed) under the immutable dependency intent;
    - ``blocked`` covers unsatisfied training-derived Evaluation coordinates that are
      still genuinely waiting for their planned upstream Model dependency to become
      satisfied; a physical Queue ``BLOCKED`` row is not by itself sufficient;
    - ``unsatisfied_terminal`` covers an otherwise unsatisfied coordinate that will not
      receive another attempt in this Study Run, including coherent Queue ``FAILED`` /
      ``CANCELLED`` work and a retained Evaluation ``BLOCKED`` row whose upstream
      Training coordinate is coherently established as permanently unsatisfied.

    When every unsatisfied coordinate of this execution kind has a coherent Queue
    projection, all four operational fields are integers and together with
    ``satisfied`` partition ``total`` exactly.

    Queue is replaceable state and may be missing, stale, or structurally inconsistent.
    If any unsatisfied coordinate of this execution kind cannot be classified honestly,
    all four operational fields are ``None`` rather than fabricating a category or
    adding an ``unknown`` public vocabulary value. ``total`` and ``satisfied`` remain
    available because immutable plan intent and canonical accepted child facts remain
    authoritative even when Queue observation is unusable.

    A canonically satisfied coordinate always contributes to ``satisfied`` regardless
    of a lagging/missing Queue row and is never double-counted as active, waiting,
    blocked, or terminal-unsatisfied.
    """

    total: int
    satisfied: int
    active: int | None
    waiting: int | None
    blocked: int | None
    unsatisfied_terminal: int | None


@dataclass(frozen=True, slots=True)
class TrainingProgressDiagnostic:
    """Optional latest-child diagnostic for one incomplete training coordinate.

    ``trial`` is the exact Study Run-local plan coordinate. ``latest_run_id`` and
    ``latest_status`` are both ``None`` when no Training Run has ever been allocated for
    the coordinate; otherwise both identify the last matching child in the frozen
    Training Run listing's identity/allocation order.

    This value is diagnostic only. It does not expose Queue attempts, retry counts,
    Worker identity, lease state, failure text, or a synthetic current attempt.
    """

    trial: str
    latest_run_id: TrainingRunId | None
    latest_status: TrainingRunStatus | None


@dataclass(frozen=True, slots=True)
class EvaluationProgressDiagnostic:
    """Optional latest-child diagnostic for one incomplete evaluation coordinate.

    ``trial`` and ``stage`` are the exact immutable plan coordinate. When matching
    Evaluation Run history exists, ``latest_run_id`` / ``latest_status`` identify the
    last matching child in frozen Evaluation Run identity/allocation order. Both are
    ``None`` when no child Run exists for the coordinate.

    Historical failed/partial/cancelled retries remain canonical child records but are
    not expanded into attempt history here.
    """

    trial: str
    stage: str
    latest_run_id: EvaluationRunId | None
    latest_status: EvaluationRunStatus | None


@dataclass(frozen=True, slots=True)
class StudyRunProgress:
    """Exact canonical Study Run plus its current derived application progress.

    ``study_run`` is the exact value returned by the canonical typed Study Run read;
    observation never constructs a replacement Study Run or derives another lifecycle
    status.

    ``training`` is present only for a finalized training-derived plan. An
    existing-Model plan consistently omits it because there are no planned training
    coordinates. ``evaluation`` is present whenever a finalized valid plan exists.
    When ``study_run.plan is None``, both sections are ``None`` because no authoritative
    coordinate totals exist.

    Diagnostic tuples contain only incomplete planned coordinates and are ordered by
    immutable plan order (and evaluation-stage order within a trial). A coordinate that
    is canonically satisfied is omitted even if earlier unsuccessful child Runs exist.
    With no finalized plan both tuples are empty.
    """

    study_run: StudyRun
    training: StudyRunProgressCounts | None
    evaluation: StudyRunProgressCounts | None
    training_incomplete: tuple[TrainingProgressDiagnostic, ...] = ()
    evaluation_incomplete: tuple[EvaluationProgressDiagnostic, ...] = ()


def get_study_run_progress(
    study_run_id: StudyRunId,
    as_of: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
) -> StudyRunProgress:
    """Observe one Study Run without reconciliation or any other side effect.

    The operation starts with the frozen typed canonical ``read_study_run`` semantics
    and returns that exact :class:`StudyRun` object as ``result.study_run``. Canonical
    Study Run status remains authoritative whether it is ``RUNNING`` or terminal; this
    function never infers, rewrites, or second-guesses Study lifecycle state from child
    or Queue progress.

    ``as_of`` is the Queue observation cutoff used only for current lease-authority
    checks. It uses the exact frozen Queue timestamp encoding: UTC RFC3339 text with six
    fractional-second digits and a ``Z`` suffix. This boundary does not introduce a
    Clock/NowProvider or derive a current time itself; the later Controller composition
    supplies the observation value unchanged to Queue authority checks.

    **No finalized plan.** ``RUNNING``, ``FAILED``, and ``CANCELLED`` Study Runs may
    legitimately have ``plan=None``. In that case the result contains the exact
    canonical Study Run, ``training=None``, ``evaluation=None``, and empty diagnostic
    tuples. The operation must not read or parse incidental ``plan.jsonl`` bytes,
    derive zero planned totals, inspect child/Queue state as substitute intent, or
    restart materialization. Only frozen ``read_study_plan`` semantics may establish an
    authoritative plan, and that read is used only when Study Run plan metadata exists.

    **Finalized plan and canonical child history.** With plan metadata present, the
    exact verified immutable plan is read through ``read_study_plan``. Canonical child
    history is obtained only through the frozen typed ``list_training_runs`` and
    ``list_evaluation_runs`` operations and filtered by exact represented Study lineage.
    Historical unsuccessful retries do not increase ``total``: totals come only from
    immutable plan coordinates.

    A Study-linked child that cannot map to exactly one planned coordinate, or whose
    recorded execution inputs disagree with that coordinate's immutable plan intent, is
    a canonical-history inconsistency rather than evidence to ignore or normalize. The
    same plan/child agreement rules already frozen for reconciliation apply here:

    - Training child ``study.run`` / ``study.trial`` and its Corpus, Architecture,
      Train Protocol, seed, and complete resolved parameters must equal the selected
      training plan row;
    - Evaluation child ``study.run`` / ``study.trial`` / ``study.stage`` and its Corpus,
      Evaluation Protocol, complete resolved parameters, and exact Model source must
      equal the selected evaluation plan coordinate.

    A finalized Study plan is expected to represent one Study Model-source mode. A
    mixed training/existing-Model plan is not projected as an invented hybrid public
    progress shape.

    **Training satisfaction.** One training-derived trial is satisfied only when its
    canonical matching history contains exactly one ``TrainingRunStatus.COMPLETED`` Run
    and the deterministic Model for that exact completed Run exists and is valid. Derive
    the expected identity only through ``model_id_for_training_run(completed.id)`` and
    observe existence only through
    ``filesystem.file_exists(layout.model_metadata_path(expected_model_id))``. The Model
    directory is never scanned.

    If that exact Model YAML is absent, this is the known repairable cross-store gap:
    the training coordinate remains canonically unsatisfied, observation continues, and
    the incomplete-coordinate diagnostic may truthfully report the completed Training
    Run as its latest child. This operation does not call
    ``ensure_model_for_completed_training_run`` or otherwise create the missing Model.

    If the expected Model YAML exists, ``resolve_model(expected_model_id, layout,
    filesystem)`` must succeed. Resolution failure is represented canonical-state
    invalidity/inconsistency and propagates as an operation failure; corrupted or
    conflicting Model metadata is not downgraded to the benign "not created yet" case.
    The resolved handle must satisfy ``handle.metadata.id == expected_model_id``,
    ``handle.metadata.training_run == completed.id``, and
    ``handle.training_run.id == completed.id``. Only then is the coordinate satisfied.
    Learned state is not loaded.

    Earlier ``FAILED`` or ``CANCELLED`` Training Runs remain history and do not add
    coordinates or prevent later canonical satisfaction. More than one completed
    Training Run for one coordinate is the same ambiguous Model-authority inconsistency
    recognized by reconciliation rather than a reason to pick the latest completed Run.

    **Evaluation satisfaction.** One evaluation coordinate is satisfied when canonical
    matching history contains an ``EvaluationRunStatus.COMPLETED`` Run for the exact
    Study Run, trial, and stage whose immutable execution inputs/Model agree with the
    plan. ``COMPLETED_PARTIAL``, ``FAILED``, and ``CANCELLED`` are historical
    unsatisfied outcomes and do not satisfy the coordinate. A satisfying completed Run
    outranks earlier unsuccessful retries. At most one canonical ``RUNNING`` child may
    represent an incomplete logical coordinate, and a satisfying child plus another
    ``RUNNING`` child is a canonical inconsistency rather than an additional progress
    category.

    **Latest-child diagnostics.** Diagnostics are produced only for incomplete
    coordinates. If child history exists, the latest matching child is selected only by
    the already-frozen deterministic Run listing order: v1 ``tr-/ev-YYYYMMDD-NNN`` IDs
    encode allocation date plus per-date allocation sequence, and the typed list
    operations return identity order. No timestamp heuristic or Queue attempt number is
    invented. A coordinate with no child Run reports both latest identity/status fields
    as ``None``.

    **Queue projection.** Canonical satisfaction is classified before Queue state is
    considered. For each remaining unsatisfied coordinate, Queue observation uses only
    the expected logical job derived from the immutable plan and the existing
    ``QueuePort`` read operations. No raw Queue status escapes this module. A coherent
    projection maps:

    - ``QueueJobStatus.ACTIVE`` -> ``active`` only when
      ``queue.open_attempt_for_job(job.job_id)`` returns the single open attempt, that
      attempt's ``run_id`` equals the corresponding sole canonical ``RUNNING`` child,
      and ``queue.authorized_open_attempt(attempt.attempt_id, attempt.lease_token,
      as_of=as_of)`` returns that same current attempt identity;
    - ``QueueJobStatus.READY`` or ``QueueJobStatus.RETRY_WAIT`` -> ``waiting`` only
      when the coordinate's immutable dependency state makes that work runnable;
    - an unsatisfied Evaluation's own coherent ``QueueJobStatus.FAILED`` or
      ``QueueJobStatus.CANCELLED`` -> ``unsatisfied_terminal`` regardless of whether
      its upstream Model previously existed or could otherwise have existed;
    - ``QueueJobStatus.BLOCKED`` is never mapped mechanically. For a training-derived
      Evaluation, the planned upstream Training coordinate is classified from canonical
      Training/Model history plus its coherent retained Queue disposition. Public
      ``blocked`` means the dependency can still become satisfied in this Study Run;
      public ``unsatisfied_terminal`` means the upstream Training is permanently
      unsatisfied and no Evaluation attempt can still be created.

    ``QueueJobStatus.SATISFIED`` is never accepted as success authority. If canonical
    history already satisfies the coordinate, canonical ``satisfied`` wins regardless
    of Queue lag. If Queue says ``SATISFIED`` while canonical history does not satisfy
    the coordinate, the operational projection is stale/inconsistent.

    Coherence includes Queue structure, not just the status enum. Every non-``ACTIVE``
    unsatisfied job must have no open attempt. Training jobs and existing-Model
    Evaluation jobs must have no dependency row; a training-derived Evaluation job must
    reference exactly its same-trial Training job. Impossible open-attempt or dependency
    relations degrade the affected execution section instead of being counted from
    status alone.

    For a training-derived Evaluation whose own coherent Queue job is ``BLOCKED``, the
    upstream dependency disposition is observed as follows. If upstream Training is not
    canonically satisfied and its coherent Queue job is ``ACTIVE`` with the exact
    current authorized attempt and matching sole canonical ``RUNNING`` Training child,
    or is ``READY`` / ``RETRY_WAIT`` under the ordinary training projection rules, a
    later successful Training attempt can still establish the Model: the Evaluation is
    public ``blocked`` and is not ``waiting`` itself. If the upstream Training remains
    canonically unsatisfied and its coherent retained Queue job is ``FAILED`` or
    ``CANCELLED``, with no stronger canonical satisfying fact and outside the known
    completed-Training/missing-Model gap described below, no further Training work can
    establish that dependency; the retained Evaluation ``BLOCKED`` row therefore
    contributes to public ``unsatisfied_terminal`` rather than public ``blocked``.
    Reconciliation need not rewrite that dependent Queue row merely for presentation: a
    physical Queue ``BLOCKED`` row may remain while application progress projects the
    coordinate as terminally unsatisfied.

    Conversely, if upstream Training is canonically satisfied by its exact completed
    Training Run plus valid deterministic Model, a dependent Evaluation still retained
    as Queue ``BLOCKED`` is stale/inconsistent. Observation does not silently treat it
    as ``READY`` or ``waiting``; the Evaluation section's four operational counts
    degrade to ``None`` unless another already-frozen coherent rule establishes an exact
    classification. The same degradation applies when upstream future satisfiability
    versus permanent unsatisfaction cannot be established honestly: for example the
    upstream Training Queue job is missing, says ``SATISFIED`` without canonical
    satisfaction, has a contradictory status/open-attempt relation, retains ``ACTIVE``
    without current authorization at ``as_of``, or the dependency projection is
    malformed. This upstream-disposition inquiry is unnecessary when the unsatisfied
    Evaluation's own structurally coherent Queue job is already ``FAILED`` or
    ``CANCELLED``: that logical Evaluation work itself has a no-more-work disposition.

    ``READY`` / ``RETRY_WAIT`` Evaluation work is ``waiting`` only when its dependency
    is already runnable: an existing-Model row has no training dependency, and a
    training-derived row requires canonical upstream Training satisfaction. ``ACTIVE``
    likewise requires that runnable dependency state in addition to the exact canonical
    ``RUNNING`` child / open-attempt / authorization checks above. A training-derived
    Evaluation cannot coherently be ``READY``, ``RETRY_WAIT``, or ``ACTIVE`` while its
    upstream Model dependency is genuinely unsatisfied; such a contradiction degrades
    the Evaluation operational counts rather than fabricating progress.

    ``authorized_open_attempt`` is used here strictly as a read-only current-authority
    check. The authorized result must still identify the same ``attempt_id``, ``job_id``,
    ``run_id``, and ``lease_token`` as the open attempt being classified; observation
    does not adopt a different/superseding attempt from this check. If an ``ACTIVE`` row
    is retained but its open attempt is missing, stale, superseded, expired, no longer
    authorized at ``as_of``, or disagrees with the sole canonical ``RUNNING`` child, it
    is not counted as active. The affected execution section degrades all four
    operational counts to ``None`` under the existing stale/inconsistent Queue rule.
    No lease recovery or heartbeat is triggered.

    An exact completed Training Run whose deterministic Model YAML is still missing is
    canonically unsatisfied but does not by itself describe runnable Training work. In
    that known Model-creation crash gap, ``ACTIVE`` cannot be coherent because there is
    no matching canonical ``RUNNING`` child, and ``READY`` / ``RETRY_WAIT`` are not
    accepted merely from their Queue status as though another Training Run should
    automatically be scheduled. If the retained Queue facts cannot be classified
    consistently with that canonical gap, the training operational counts degrade to
    ``None``. Normal coherent terminal ``FAILED`` / ``CANCELLED`` projection remains
    terminal-unsatisfied for the Training section; Queue ``SATISFIED`` still never
    overrides the missing canonical Model.

    The same gap has a different dependency meaning for a retained dependent Evaluation
    ``BLOCKED`` row. Its upstream Training execution has already completed and the
    deterministic Model may still be established by normal result completion or
    reconciliation without another Training attempt, so the Evaluation is genuinely
    waiting for that Model and contributes to public ``blocked``, not
    ``unsatisfied_terminal``. Observation does not create the Model. If surrounding
    Queue facts contradict the already-frozen repairable-gap rules, the Evaluation
    operational counts degrade under the normal stale/inconsistent rule instead of
    guessing terminality. This repairable-gap dependency classification still obeys the
    terminal-Study consistency rule below; a terminal Study is not described as waiting
    for future Evaluation work when the observed facts establish that no such attempt
    can occur.

    If an unsatisfied expected coordinate is missing from Queue, duplicates/conflicts
    another expected logical coordinate, has an impossible dependency/status relation,
    claims satisfaction without canonical success, or otherwise cannot be mapped
    honestly, the affected execution section preserves canonical ``total`` and
    ``satisfied`` but sets all four operational counts to ``None``. No partial
    operational subtotal is presented as complete truth and no new public ``unknown``
    category is introduced. Training and evaluation sections degrade independently when
    only one kind's Queue projection is unusable.

    This is a best-effort read across canonical filesystem state and Queue state, not a
    snapshot transaction. No observation lock, Queue generation, or global consistency
    framework is introduced. Concurrent change between reads may therefore make one
    pass detect inconsistency and degrade nullable operational counts; that is preferred
    to mutation or fabricated precision.

    **Terminal Study Runs.** Terminal Study Run status remains the exact canonical
    status returned above. Derived progress continues to describe observed canonical
    child and Queue facts and never reopens, completes, fails, or cancels the Study.
    In particular a ``CANCELLED`` Study Run may be observed during a cancellation race
    with one still-current authorized Worker attempt; it is reported as active only when
    the exact ``ACTIVE`` / open-attempt / canonical ``RUNNING`` / authorization rule
    above holds at ``as_of``.

    Canonical terminal Study status is also an additional consistency constraint on
    non-active projection: observation must not describe an unsatisfied coordinate as
    public ``blocked`` merely because a retained Queue row is physically ``BLOCKED``
    when canonical/operational facts already establish that no future attempt can occur
    in this Study Run. Where the existing coordinate rules establish a no-more-work
    disposition, report ``unsatisfied_terminal``; where retained Queue facts contradict
    terminal Study state but do not establish an exact coordinate disposition, degrade
    the affected operational counts to ``None``. This is not a second Study-finalization
    algorithm and does not recompute or override the canonical Study status.

    **Mutation prohibition.** This operation must not call ``reconcile_study_run``,
    ``ensure_model_for_completed_training_run``, ``RetryPolicy``, Queue admission,
    ``heartbeat_attempt``, ``expired_open_attempts`` as a recovery driver,
    ``close_attempt``, ``repair_job_state``, ``cancel_job``, activation/defer/fail
    operations, child Run allocation or finalization, Study Run
    persistence/finalization, plan materialization/finalization, or any filesystem write
    operation. ``authorized_open_attempt`` is the only lease-sensitive operation used
    here and is read-only. The function reads canonical records and Queue projection
    only.
    """

    ...
