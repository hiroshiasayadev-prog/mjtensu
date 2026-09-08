"""Public Python signatures for Study-derived MLDB logical orchestration jobs.

This skeleton fixes the logical work coordinates derived from one complete immutable
Study Run plan. A logical job identifies planned Training or Evaluation work, not a
concrete Training Run / Evaluation Run and not a Worker execution attempt.

Execution payload remains authoritative in :class:`StudyPlan`; this module therefore
stores only Study Run-local coordinates plus the one logical training dependency needed
by training-derived Evaluation work. Queue lifecycle, SQLite identity/persistence,
claims, retries, leases, concrete preflight, child Run allocation, result acceptance,
and reconciliation are intentionally outside this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from ..common.ids import StudyRunId
from ..study.plan import StudyPlan


@dataclass(frozen=True, slots=True)
class TrainingJobCoordinate:
    """Canonical logical identity of one Study-derived Training job.

    ``study_run`` identifies the owning Study Run and ``trial`` is that immutable
    plan's Study Run-local ``trial-NNNN`` string. Trial remains a plain string rather
    than a new entity-ID type.

    Training has no stage coordinate. Architecture, Corpus, Train Protocol, seed,
    resolved public parameters, future Model identity, concrete Training Run IDs, and
    Worker-attempt state are deliberately absent because they belong to the immutable
    Study plan or later execution boundaries.
    """

    study_run: StudyRunId
    trial: str


@dataclass(frozen=True, slots=True)
class EvaluationJobCoordinate:
    """Canonical logical identity of one Study-derived Evaluation job.

    ``study_run`` and ``trial`` select one immutable Study plan row; ``stage`` selects
    exactly one evaluation-stage entry in that row. ``trial`` and ``stage`` remain
    Study-local strings rather than shared entity identities.

    Model, Corpus, Evaluation Protocol, resolved public parameters, concrete Evaluation
    Run IDs, and Worker-attempt state are deliberately absent because those facts are
    resolved from the immutable plan or introduced only by later execution boundaries.
    """

    study_run: StudyRunId
    trial: str
    stage: str


@dataclass(frozen=True, slots=True)
class TrainingJob:
    """One logical v1 ``training`` work unit.

    The concrete Python type is the v1 job-kind discriminator; no separate generic job
    base or Queue storage identity is introduced. One instance may have zero, one, or
    multiple concrete Training Run attempts over its lifetime without changing this
    logical coordinate.
    """

    coordinate: TrainingJobCoordinate


@dataclass(frozen=True, slots=True)
class EvaluationJob:
    """One logical v1 ``evaluation`` work unit and its training dependency intent.

    ``training_dependency`` is the same-row :class:`TrainingJobCoordinate` for a
    training-derived Study plan row. It is ``None`` for an existing-Model row because
    that immutable plan row already identifies the Model required by evaluation.

    This relation expresses only logical dependency identity. ``blocked``/``ready``
    state, SQLite ``dependency_job_id``, satisfaction checks, and dependency release
    belong to Queue admission/lifecycle and later reconciliation rather than this job
    value.
    """

    coordinate: EvaluationJobCoordinate
    training_dependency: TrainingJobCoordinate | None


StudyJob: TypeAlias = TrainingJob | EvaluationJob
"""The complete v1 logical Study-derived job-kind union.

V1 contains exactly Training and Evaluation work. Study-wide, trial-wide, generic DAG,
and Worker-attempt job kinds are not part of this union.
"""


def derive_study_jobs(
    study_run_id: StudyRunId,
    plan: StudyPlan,
) -> frozenset[StudyJob]:
    """Derive the complete logical work set from one complete valid Study Run plan.

    ``plan`` is the exact immutable materialized :class:`StudyPlan` for
    ``study_run_id`` and must already be complete and valid. This pure boundary does not
    author, mutate, resolve, persist, or re-materialize Study execution payload.

    For every training-derived row, the result contains exactly one
    :class:`TrainingJob` at ``study_run_id + trial`` and one :class:`EvaluationJob` per
    materialized evaluation stage at ``study_run_id + trial + stage``. Every such
    Evaluation job carries the same-row Training coordinate as
    ``training_dependency``.

    For every existing-Model row, the result contains no Training job and contains one
    Evaluation job per materialized evaluation stage with ``training_dependency`` set
    to ``None``. The existing Model itself is not copied into the logical job because
    the immutable Study plan remains execution-payload authority.

    The return value is an immutable set because current Design Records define logical
    coordinates and dependencies but do not define Queue scheduling priority or
    admission order. An implementation may traverse the ordered Study plan
    deterministically while constructing the result, but frozenset iteration order is
    not semantic and must not be interpreted as scheduler priority.

    This operation allocates no Queue-local integer job ID, no Training/Evaluation Run
    ID, and no attempt number. It derives no Queue status, timestamps, retry timing,
    Worker identity, claim/lease token, concrete preflight input, or satisfaction state.
    Queue admission is responsible for translating dependency intent into initial
    ``ready``/``blocked`` lifecycle state; later Controller/Queue reconciliation is
    responsible for matching logical coordinates against canonical child history.
    """

    jobs: set[StudyJob] = set()
    for row in plan:
        training_coordinate = None
        if row.training is not None:
            training_coordinate = TrainingJobCoordinate(
                study_run=study_run_id,
                trial=row.trial,
            )
            jobs.add(TrainingJob(coordinate=training_coordinate))

        for evaluation in row.evaluations:
            jobs.add(
                EvaluationJob(
                    coordinate=EvaluationJobCoordinate(
                        study_run=study_run_id,
                        trial=row.trial,
                        stage=evaluation.stage,
                    ),
                    training_dependency=training_coordinate,
                )
            )

    return frozenset(jobs)
