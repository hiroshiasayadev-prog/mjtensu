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

from collections.abc import Mapping
from dataclasses import dataclass

from ..common.errors import LifecycleConflictError
from ..common.ids import EvaluationRunId, StudyRunId, TrainingRunId
from ..evaluation.run import EvaluationRun, EvaluationRunStatus
from ..model.identity import model_id_for_training_run
from ..orchestration.jobs import (
    EvaluationJob,
    StudyJob,
    TrainingJob,
    derive_study_jobs,
)
from ..orchestration.queue import QueueAttempt, QueueJob, QueueJobStatus
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
from ..training.run import TrainingRun, TrainingRunStatus
from .plan import StudyPlanEvaluation, StudyPlanRow
from .run import StudyRun, StudyRunStatus


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

    study_run = read_study_run(study_run_id, layout, filesystem)
    if study_run.plan is None:
        return StudyRunProgress(
            study_run=study_run,
            training=None,
            evaluation=None,
        )

    plan = tuple(read_study_plan(study_run_id, layout, filesystem))
    training_modes = {row.training is not None for row in plan}
    if len(training_modes) != 1:
        raise LifecycleConflictError(
            "Study plan mixes training-derived and existing-Model rows"
        )
    training_derived = training_modes == {True}

    rows_by_trial = {row.trial: row for row in plan}
    training_history: dict[str, list[TrainingRun]] = {
        row.trial: [] for row in plan if row.training is not None
    }
    for run in list_training_runs(layout, filesystem):
        if run.study is None or run.study.run != study_run_id:
            continue
        row = rows_by_trial.get(run.study.trial)
        if (
            row is None
            or row.training is None
            or not _training_inputs_match(run, row)
        ):
            raise LifecycleConflictError(
                f"Training Run {run.id} disagrees with Study plan lineage or inputs"
            )
        training_history[row.trial].append(run)

    canonical_training: dict[str, _CanonicalTraining] = {}
    for row in plan:
        if row.training is None:
            continue
        history = tuple(training_history[row.trial])
        completed = tuple(
            run for run in history if run.status is TrainingRunStatus.COMPLETED
        )
        running = tuple(
            run for run in history if run.status is TrainingRunStatus.RUNNING
        )
        if len(completed) > 1:
            raise LifecycleConflictError(
                f"multiple completed Training Runs represent {row.trial}"
            )
        if len(running) > 1 or (completed and running):
            raise LifecycleConflictError(
                f"ambiguous active Training Run history represents {row.trial}"
            )

        satisfied = False
        model_gap = False
        completed_run = completed[0] if completed else None
        if completed_run is not None:
            expected_model_id = model_id_for_training_run(completed_run.id)
            model_path = layout.model_metadata_path(expected_model_id)
            if not filesystem.file_exists(model_path):
                model_gap = True
            else:
                handle = resolve_model(expected_model_id, layout, filesystem)
                if (
                    handle.metadata.id != expected_model_id
                    or handle.metadata.training_run != completed_run.id
                    or handle.training_run.id != completed_run.id
                ):
                    raise LifecycleConflictError(
                        f"Model {expected_model_id} disagrees with completed Training Run"
                    )
                satisfied = True

        canonical_training[row.trial] = _CanonicalTraining(
            row=row,
            history=history,
            completed=completed_run,
            running=running[0] if running else None,
            satisfied=satisfied,
            model_gap=model_gap,
        )

    evaluation_history: dict[tuple[str, str], list[EvaluationRun]] = {
        (row.trial, evaluation.stage): []
        for row in plan
        for evaluation in row.evaluations
    }
    for run in list_evaluation_runs(layout, filesystem):
        if run.study is None or run.study.run != study_run_id:
            continue
        row = rows_by_trial.get(run.study.trial)
        evaluation = (
            None
            if row is None
            else next(
                (
                    candidate
                    for candidate in row.evaluations
                    if candidate.stage == run.study.stage
                ),
                None,
            )
        )
        if (
            row is None
            or evaluation is None
            or not _evaluation_inputs_match(
                run,
                row,
                evaluation,
                canonical_training.get(row.trial),
            )
        ):
            raise LifecycleConflictError(
                f"Evaluation Run {run.id} disagrees with Study plan lineage or inputs"
            )
        evaluation_history[(row.trial, evaluation.stage)].append(run)

    canonical_evaluation: dict[tuple[str, str], _CanonicalEvaluation] = {}
    for row in plan:
        for evaluation in row.evaluations:
            coordinate = (row.trial, evaluation.stage)
            history = tuple(evaluation_history[coordinate])
            completed = tuple(
                run for run in history if run.status is EvaluationRunStatus.COMPLETED
            )
            running = tuple(
                run for run in history if run.status is EvaluationRunStatus.RUNNING
            )
            if len(running) > 1 or (completed and running):
                raise LifecycleConflictError(
                    "ambiguous active Evaluation Run history represents "
                    f"{row.trial}/{evaluation.stage}"
                )
            canonical_evaluation[coordinate] = _CanonicalEvaluation(
                row=row,
                evaluation=evaluation,
                history=history,
                running=running[0] if running else None,
                satisfied=bool(completed),
            )

    training_incomplete = tuple(
        TrainingProgressDiagnostic(
            trial=row.trial,
            latest_run_id=(
                canonical_training[row.trial].history[-1].id
                if canonical_training[row.trial].history
                else None
            ),
            latest_status=(
                canonical_training[row.trial].history[-1].status
                if canonical_training[row.trial].history
                else None
            ),
        )
        for row in plan
        if row.training is not None
        and not canonical_training[row.trial].satisfied
    )
    evaluation_incomplete = tuple(
        EvaluationProgressDiagnostic(
            trial=row.trial,
            stage=evaluation.stage,
            latest_run_id=(
                canonical_evaluation[(row.trial, evaluation.stage)].history[-1].id
                if canonical_evaluation[(row.trial, evaluation.stage)].history
                else None
            ),
            latest_status=(
                canonical_evaluation[(row.trial, evaluation.stage)].history[-1].status
                if canonical_evaluation[(row.trial, evaluation.stage)].history
                else None
            ),
        )
        for row in plan
        for evaluation in row.evaluations
        if not canonical_evaluation[(row.trial, evaluation.stage)].satisfied
    )

    training_total = len(canonical_training)
    training_satisfied = sum(
        state.satisfied for state in canonical_training.values()
    )
    evaluation_total = len(canonical_evaluation)
    evaluation_satisfied = sum(
        state.satisfied for state in canonical_evaluation.values()
    )

    if (
        training_satisfied == training_total
        and evaluation_satisfied == evaluation_total
    ):
        training_counts = (
            _counts(training_total, training_satisfied, ())
            if training_derived
            else None
        )
        return StudyRunProgress(
            study_run=study_run,
            training=training_counts,
            evaluation=_counts(evaluation_total, evaluation_satisfied, ()),
            training_incomplete=training_incomplete,
            evaluation_incomplete=evaluation_incomplete,
        )

    expected_jobs = derive_study_jobs(study_run_id, plan)
    queue_rows = queue.jobs_for_study_run(study_run_id)
    malformed_kinds = _unexpected_queue_kinds(queue_rows, expected_jobs)
    terminal_study = study_run.status is not StudyRunStatus.RUNNING

    raw_training_categories: dict[str, str | None] = {}
    training_categories: list[str | None] = []
    for row in plan:
        if row.training is None:
            continue
        state = canonical_training[row.trial]
        if state.satisfied:
            continue
        raw = _classify_training(
            state,
            expected_jobs,
            queue_rows,
            queue,
            as_of,
        )
        raw_training_categories[row.trial] = raw
        if "training" in malformed_kinds:
            training_categories.append(None)
        elif terminal_study and raw == "waiting":
            training_categories.append(None)
        else:
            training_categories.append(raw)

    evaluation_categories: list[str | None] = []
    for row in plan:
        for evaluation in row.evaluations:
            state = canonical_evaluation[(row.trial, evaluation.stage)]
            if state.satisfied:
                continue
            upstream = None
            if row.training is not None:
                training_state = canonical_training[row.trial]
                if training_state.satisfied:
                    upstream = "satisfied"
                else:
                    raw = raw_training_categories.get(row.trial)
                    if raw is None and row.trial not in raw_training_categories:
                        raw = _classify_training(
                            training_state,
                            expected_jobs,
                            queue_rows,
                            queue,
                            as_of,
                        )
                        raw_training_categories[row.trial] = raw
                    if training_state.model_gap:
                        upstream = "repairable_gap" if raw == "terminal" else "unknown"
                    elif raw in {"active", "waiting"}:
                        upstream = "viable"
                    elif raw == "terminal":
                        upstream = "terminal"
                    else:
                        upstream = "unknown"

            category = _classify_evaluation(
                state,
                upstream,
                expected_jobs,
                queue_rows,
                queue,
                as_of,
                terminal_study,
            )
            evaluation_categories.append(
                None if "evaluation" in malformed_kinds else category
            )

    return StudyRunProgress(
        study_run=study_run,
        training=(
            _counts(
                training_total,
                training_satisfied,
                tuple(training_categories),
            )
            if training_derived
            else None
        ),
        evaluation=_counts(
            evaluation_total,
            evaluation_satisfied,
            tuple(evaluation_categories),
        ),
        training_incomplete=training_incomplete,
        evaluation_incomplete=evaluation_incomplete,
    )


@dataclass(frozen=True, slots=True)
class _CanonicalTraining:
    row: StudyPlanRow
    history: tuple[TrainingRun, ...]
    completed: TrainingRun | None
    running: TrainingRun | None
    satisfied: bool
    model_gap: bool


@dataclass(frozen=True, slots=True)
class _CanonicalEvaluation:
    row: StudyPlanRow
    evaluation: StudyPlanEvaluation
    history: tuple[EvaluationRun, ...]
    running: EvaluationRun | None
    satisfied: bool


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
    run: EvaluationRun,
    row: StudyPlanRow,
    evaluation: StudyPlanEvaluation,
    training: _CanonicalTraining | None,
) -> bool:
    expected_model = row.model
    if row.training is not None:
        if training is None or training.completed is None:
            return False
        expected_model = model_id_for_training_run(training.completed.id)
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
        return (
            set(left) == set(right)
            and all(_same_public_value(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_public_value(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _counts(
    total: int,
    satisfied: int,
    categories: tuple[str | None, ...],
) -> StudyRunProgressCounts:
    if any(category is None for category in categories):
        return StudyRunProgressCounts(total, satisfied, None, None, None, None)
    return StudyRunProgressCounts(
        total=total,
        satisfied=satisfied,
        active=categories.count("active"),
        waiting=categories.count("waiting"),
        blocked=categories.count("blocked"),
        unsatisfied_terminal=categories.count("terminal"),
    )


def _unexpected_queue_kinds(
    rows: tuple[QueueJob, ...],
    expected: frozenset[StudyJob],
) -> frozenset[str]:
    bad: set[str] = set()
    for row in rows:
        if not isinstance(row, QueueJob):
            bad.update(("training", "evaluation"))
            continue
        if not any(row.logical == logical for logical in expected):
            if isinstance(row.logical, TrainingJob):
                bad.add("training")
            elif isinstance(row.logical, EvaluationJob):
                bad.add("evaluation")
            else:
                bad.update(("training", "evaluation"))
    return frozenset(bad)


def _one_queue_job(
    logical: StudyJob,
    rows: tuple[QueueJob, ...],
) -> QueueJob | None:
    matches = [
        row
        for row in rows
        if isinstance(row, QueueJob) and row.logical == logical
    ]
    return matches[0] if len(matches) == 1 else None


def _queue_job_is_coherent(
    job: QueueJob,
    logical: StudyJob,
    dependency: TrainingJob | None,
    rows: tuple[QueueJob, ...],
) -> bool:
    if (
        job.logical != logical
        or type(job.job_id) is not int
        or job.job_id <= 0
        or not isinstance(job.status, QueueJobStatus)
        or not isinstance(job.created_at, str)
        or not job.created_at
        or not isinstance(job.updated_at, str)
        or not job.updated_at
        or (
            (
                not isinstance(job.retry_not_before, str)
                or not job.retry_not_before
            )
            if job.status is QueueJobStatus.RETRY_WAIT
            else job.retry_not_before is not None
        )
        or sum(
            isinstance(row, QueueJob) and row.job_id == job.job_id
            for row in rows
        )
        != 1
    ):
        return False

    if dependency is None:
        return job.dependency_job_id is None

    dependency_job = _one_queue_job(dependency, rows)
    return (
        dependency_job is not None
        and type(dependency_job.job_id) is int
        and dependency_job.job_id > 0
        and dependency_job.dependency_job_id is None
        and isinstance(dependency_job.status, QueueJobStatus)
        and (
            (
                isinstance(dependency_job.retry_not_before, str)
                and bool(dependency_job.retry_not_before)
            )
            if dependency_job.status is QueueJobStatus.RETRY_WAIT
            else dependency_job.retry_not_before is None
        )
        and isinstance(dependency_job.created_at, str)
        and bool(dependency_job.created_at)
        and isinstance(dependency_job.updated_at, str)
        and bool(dependency_job.updated_at)
        and sum(
            isinstance(row, QueueJob) and row.job_id == dependency_job.job_id
            for row in rows
        )
        == 1
        and job.dependency_job_id == dependency_job.job_id
    )


def _non_active_has_no_open_attempt(
    job: QueueJob,
    queue: QueuePort,
) -> bool:
    return queue.open_attempt_for_job(job.job_id) is None


def _active_is_authorized(
    job: QueueJob,
    running_run_id: TrainingRunId | EvaluationRunId | None,
    queue: QueuePort,
    as_of: str,
) -> bool:
    if running_run_id is None:
        return False
    attempt = queue.open_attempt_for_job(job.job_id)
    if (
        not isinstance(attempt, QueueAttempt)
        or type(attempt.attempt_id) is not int
        or attempt.attempt_id <= 0
        or attempt.job_id != job.job_id
        or type(attempt.attempt_no) is not int
        or attempt.attempt_no <= 0
        or attempt.run_id != running_run_id
        or not isinstance(attempt.worker_id, str)
        or not attempt.worker_id
        or not isinstance(attempt.acquire_token, str)
        or not attempt.acquire_token
        or not isinstance(attempt.lease_token, str)
        or not attempt.lease_token
        or not isinstance(attempt.lease_until, str)
        or not attempt.lease_until
        or not isinstance(attempt.started_at, str)
        or not attempt.started_at
        or attempt.finished_at is not None
    ):
        return False
    authorized = queue.authorized_open_attempt(
        attempt.attempt_id,
        attempt.lease_token,
        as_of=as_of,
    )
    return (
        isinstance(authorized, QueueAttempt)
        and authorized.finished_at is None
        and authorized.attempt_id == attempt.attempt_id
        and authorized.job_id == attempt.job_id
        and authorized.run_id == attempt.run_id
        and authorized.lease_token == attempt.lease_token
    )


def _classify_training(
    state: _CanonicalTraining,
    expected_jobs: frozenset[StudyJob],
    rows: tuple[QueueJob, ...],
    queue: QueuePort,
    as_of: str,
) -> str | None:
    logical = next(
        (
            job
            for job in expected_jobs
            if isinstance(job, TrainingJob)
            and job.coordinate.trial == state.row.trial
        ),
        None,
    )
    if logical is None:
        return None
    job = _one_queue_job(logical, rows)
    if job is None or not _queue_job_is_coherent(job, logical, None, rows):
        return None

    if job.status is QueueJobStatus.ACTIVE:
        if state.model_gap:
            return None
        return (
            "active"
            if _active_is_authorized(
                job,
                state.running.id if state.running is not None else None,
                queue,
                as_of,
            )
            else None
        )

    if not _non_active_has_no_open_attempt(job, queue) or state.running is not None:
        return None
    if job.status in {QueueJobStatus.READY, QueueJobStatus.RETRY_WAIT}:
        return None if state.model_gap else "waiting"
    if job.status in {QueueJobStatus.FAILED, QueueJobStatus.CANCELLED}:
        return "terminal"
    return None


def _classify_evaluation(
    state: _CanonicalEvaluation,
    upstream: str | None,
    expected_jobs: frozenset[StudyJob],
    rows: tuple[QueueJob, ...],
    queue: QueuePort,
    as_of: str,
    terminal_study: bool,
) -> str | None:
    logical = next(
        (
            job
            for job in expected_jobs
            if isinstance(job, EvaluationJob)
            and job.coordinate.trial == state.row.trial
            and job.coordinate.stage == state.evaluation.stage
        ),
        None,
    )
    if logical is None:
        return None
    dependency = (
        next(
            (
                job
                for job in expected_jobs
                if isinstance(job, TrainingJob)
                and logical.training_dependency == job.coordinate
            ),
            None,
        )
        if logical.training_dependency is not None
        else None
    )
    if logical.training_dependency is not None and dependency is None:
        return None

    job = _one_queue_job(logical, rows)
    if (
        job is None
        or not _queue_job_is_coherent(job, logical, dependency, rows)
    ):
        return None

    if job.status is QueueJobStatus.ACTIVE:
        if terminal_study or upstream in {None, "satisfied"}:
            runnable = logical.training_dependency is None or upstream == "satisfied"
        else:
            runnable = False
        if not runnable:
            return None
        return (
            "active"
            if _active_is_authorized(
                job,
                state.running.id if state.running is not None else None,
                queue,
                as_of,
            )
            else None
        )

    if not _non_active_has_no_open_attempt(job, queue) or state.running is not None:
        return None
    if job.status in {QueueJobStatus.READY, QueueJobStatus.RETRY_WAIT}:
        runnable = logical.training_dependency is None or upstream == "satisfied"
        return "waiting" if runnable and not terminal_study else None
    if job.status in {QueueJobStatus.FAILED, QueueJobStatus.CANCELLED}:
        return "terminal"
    if job.status is QueueJobStatus.BLOCKED:
        if logical.training_dependency is None or upstream == "satisfied":
            return None
        if upstream == "terminal":
            return "terminal"
        if upstream == "repairable_gap":
            return "terminal" if terminal_study else "blocked"
        if upstream == "viable":
            return None if terminal_study else "blocked"
    return None

