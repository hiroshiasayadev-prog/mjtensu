"""Canonical MLDB execution-state persistence and Run allocation signatures.

This module fixes the Controller-side boundary that begins only after the frozen
Training/Evaluation/Study preflight operations have succeeded. Each public allocation
operation combines typed Run-ID selection with persistence of the first schema-valid
``running`` record, so no successful public allocation exposes an identity without its
canonical initial state. The module also finalizes one complete Study plan, replaces
``running`` records with caller-constructed terminal domain values, and exposes typed
canonical reads/listing needed by later reconciliation and Public API work.

The boundary deliberately reuses the frozen ``TrainingRun``, ``EvaluationRun``,
``StudyRun``, ``StudyPlan``, ``RepositoryLayout``, and ``FilesystemPort`` contracts. It
introduces no persistence DTO, Repository/RunManager/Store abstraction, transaction
framework, ORM, Queue/Worker dependency, result-acceptance policy, or generic string ID
allocator.

Semantic YAML/JSONL parsing and serialization, SHA-256 calculation, and allocation
sequencing mechanics are implementation-private details. Canonical locations are always
derived through :class:`RepositoryLayout`; this module does not restate repository path
grammar in a second path service.
"""

from __future__ import annotations

from datetime import date

from ..common.ids import EntityKind, EvaluationRunId, StudyRunId, TrainingRunId
from ..evaluation.preflight import EvaluationPreflight
from ..evaluation.run import (
    EvaluationRun,
    EvaluationRunExecution,
    EvaluationRunStatus,
    EvaluationRunStudyLineage,
    validate_evaluation_run,
    validate_evaluation_run_transition,
)
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..study.plan import StudyPlan, validate_study_plan
from ..study.preflight import PreparedStudyExecution
from ..study.run import (
    StudyRun,
    StudyRunExecution,
    StudyRunPlan,
    StudyRunStatus,
    validate_study_run,
    validate_study_run_transition,
)
from ..training.preflight import TrainingPreflight
from ..training.run import (
    TrainingRun,
    TrainingRunExecution,
    TrainingRunStatus,
    TrainingRunStudyLineage,
    validate_training_run,
    validate_training_run_transition,
)


def allocate_training_run(
    preflight: TrainingPreflight,
    allocation_date: date,
    started_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    *,
    study: TrainingRunStudyLineage | None = None,
) -> TrainingRun:
    """Allocate and durably establish one initial ``running`` Training Run.

    ``preflight`` must be the exact successful frozen Training launch-preflight value.
    This is the only public Training allocation success boundary: it selects one typed
    v1 :class:`TrainingRunId`, constructs the schema-valid initial domain Run entirely
    from that preflight plus ``started_at`` and optional frozen Study lineage, establishes
    the canonical Run/``work``/``artifacts`` directories, commits authoritative
    ``run.yaml``, and only then returns the exact :class:`TrainingRun` that was committed.

    The constructed Run has ``TrainingRunStatus.RUNNING`` and copies exactly
    ``preflight.corpus.metadata.id``, ``preflight.architecture.metadata.id``,
    ``preflight.protocol.metadata.id``, the complete ``preflight.parameters`` mapping,
    and the exact validated ``preflight.seed``. It has no terminal result, failure, or
    ``finished_at``. The frozen :func:`validate_training_run` must accept the value before
    canonical metadata commit.

    ``allocation_date`` is the caller-selected local calendar date used only for the
    ``tr-YYYYMMDD-NNN`` identity. Candidate selection, collision retry, serialization,
    and any locking/critical-section mechanism are implementation-private. Concurrent
    successful calls must never return the same Run identity, and the caller must not
    coordinate a separate allocate-ID/persist-running critical section. Existing or
    partially established canonical identities are never overwritten merely to reuse a
    sequence number; sequence gaps are not a public contract defect.

    Successful return therefore implies that ``run.id`` is already uniquely established
    by a schema-valid canonical ``running`` record. No typed Run ID is exposed as a
    successful public result before that first durable state exists. Executable loading,
    Worker dispatch/execution, learned-state acceptance, Model creation, Queue mutation,
    and terminal status selection remain outside this operation.
    """

    ...


def allocate_evaluation_run(
    preflight: EvaluationPreflight,
    allocation_date: date,
    started_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    *,
    study: EvaluationRunStudyLineage | None = None,
) -> EvaluationRun:
    """Allocate and durably establish one initial ``running`` Evaluation Run.

    ``preflight`` is the exact successful frozen Evaluation launch-preflight value. This
    single public allocation operation selects one typed v1 :class:`EvaluationRunId`,
    constructs the initial schema-valid domain Run, establishes its canonical Run,
    ``work``, and ``artifacts`` directories, commits authoritative ``run.yaml``, and only
    then returns the exact :class:`EvaluationRun` that was committed.

    The initial Run has ``EvaluationRunStatus.RUNNING`` and copies exactly
    ``preflight.model.metadata.id``, ``preflight.corpus.metadata.id``,
    ``preflight.protocol.metadata.id``, and the complete ``preflight.parameters``
    mapping, together with ``started_at`` and optional frozen Study lineage. It has no
    terminal result, incompleteness decision, failure, or ``finished_at``. The frozen
    :func:`validate_evaluation_run` must accept it before canonical metadata commit.

    ``allocation_date`` supplies only the local date portion of
    ``ev-YYYYMMDD-NNN``. Candidate selection, collision retry, serialization, and any
    locking/critical-section mechanism remain private. Two concurrent successful calls
    cannot establish or return the same canonical Run identity, and callers never own a
    separate allocate-ID/persist-running critical section. Existing or partially
    established canonical identities are not overwritten for sequence reuse.

    Successful return means ``run.id`` is already represented by the canonical
    schema-valid ``running`` record. No successful public Evaluation allocation exposes
    an ID before that durable state. Model loading, Worker execution, formal result
    acceptance, canonical artifact commit, Queue mutation, and terminal status choice
    remain later Controller responsibilities.
    """

    ...


def allocate_study_run(
    prepared: PreparedStudyExecution,
    allocation_date: date,
    started_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> StudyRun:
    """Allocate and durably establish one initial no-plan ``running`` Study Run.

    ``prepared`` must be the exact successful frozen
    :class:`PreparedStudyExecution`. This single public allocation operation selects one
    typed v1 :class:`StudyRunId`, constructs the schema-valid initial Study Run,
    establishes its canonical directory, commits authoritative ``run.yaml``, and only
    then returns the exact :class:`StudyRun` that was committed.

    The initial value has ``StudyRunStatus.RUNNING``,
    ``study = prepared.study.metadata.id``, the supplied ``started_at``, no
    ``finished_at``, ``plan = None``, and no invented summary. The frozen
    :func:`validate_study_run` must accept it before metadata commit. ``plan.jsonl`` is
    not authoritative at allocation and no plan metadata is fabricated.

    ``allocation_date`` supplies only the local date portion of
    ``sr-YYYYMMDD-NNN``. Candidate selection, collision retry, serialization, and any
    locking/critical-section mechanism are private implementation details. Concurrent
    successful calls cannot establish or return the same Study Run identity, and the
    caller does not coordinate separate identity allocation and initial persistence.
    Existing or partially established canonical identities are not overwritten to fill
    sequence gaps.

    Successful return therefore exposes the typed ``run.id`` only after the canonical
    no-plan ``running`` record exists. Plan materialization/finalization, Queue admission,
    child Run creation, reconciliation, and Study terminal-state choice remain outside
    this operation.
    """

    ...


def finalize_study_plan(
    study_run_id: StudyRunId,
    plan: StudyPlan,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> StudyRun:
    """Persist one complete Study plan and establish its immutable Run metadata.

    The caller supplies a complete materialized :class:`StudyPlan`; this boundary does
    not invoke ``materialize_study_plan()`` or repeat Study grid expansion. The
    operation must apply the frozen :func:`validate_study_plan` before any plan becomes
    authoritative.

    The current canonical Study Run is loaded through the same typed read semantics as
    :func:`read_study_run`. Plan finalization is valid only while that Run is
    ``StudyRunStatus.RUNNING`` and its ``plan`` metadata is still absent. A terminal Run
    or an already-finalized plan is immutable and must be rejected rather than reopened
    or replaced.

    JSONL serialization is an implementation-private deterministic encoding of the
    supplied ordered plan. This public boundary deliberately does not standardize a
    reusable codec, object-key-order API, whitespace API, or byte-regeneration service.
    The exact bytes chosen by the implementation become authoritative only when
    finalization succeeds.

    The implementation commits the complete plan bytes to the canonical
    ``layout.study_run_paths(study_run_id).plan_path`` through ``FilesystemPort`` and
    calculates SHA-256, byte size, trial count, and total evaluation-job count from that
    same complete plan. It then constructs an updated frozen :class:`StudyRun` whose
    status remains ``RUNNING`` and whose existing Study identity, execution facts, and
    optional summary are preserved, adding exactly one :class:`StudyRunPlan` with
    ``path='plan.jsonl'`` and those integrity/count facts. The updated Run must pass
    :func:`validate_study_run` before
    authoritative ``run.yaml`` replacement.

    Because :class:`FilesystemPort` guarantees complete replacement per destination but
    does not expose a multi-file transaction, finalization must order writes so an
    interruption cannot make partial plan bytes authoritative. A complete plan file may
    be written before its metadata record; until matching plan metadata is durably
    committed, such bytes remain unfinalized and may be replaced on resume. Successful
    return means both canonical plan bytes and matching Run metadata are established.
    No public UnitOfWork/transaction abstraction is introduced.

    The returned :class:`StudyRun` is the updated canonical ``RUNNING`` value. Plan
    finalization does not admit Queue work, create child Runs, judge reconciliation, or
    terminalize the Study Run.
    """

    ...


def persist_training_run_transition(
    next_run: TrainingRun,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> None:
    """Persist one caller-constructed terminal transition for a Training Run.

    ``next_run`` is the already constructed domain outcome selected by the Controller's
    later execution/result-acceptance workflow. This persistence boundary does not
    decide whether training succeeded, validate a Worker candidate, serialize or accept
    learned weights, or create a Model.

    The operation reads the current canonical Run by ``next_run.id``, applies the frozen
    :func:`validate_training_run` to ``next_run``, and applies
    :func:`validate_training_run_transition` to ``current.status`` and
    ``next_run.status``. This operation is for the single allowed
    ``RUNNING -> terminal`` replacement; it does not
    expose terminal reopen, terminal-to-terminal rewrite, or generic status mutation.

    Before replacement it must also preserve the immutable initial execution identity:
    schema, Run ID, Corpus, Architecture, Train Protocol, complete resolved parameters,
    seed, ``started_at``, and optional Study lineage must agree with the canonical
    ``RUNNING`` record. Terminal-only facts such as ``finished_at``, accepted result
    metadata, failure text, environment, and work inventory come from ``next_run`` and
    remain the caller's domain responsibility subject to the frozen Run validator.

    The authoritative metadata path is derived solely through
    ``RepositoryLayout.training_run_paths`` and committed with
    ``FilesystemPort.replace_text`` complete-replacement semantics. No canonical
    artifact bytes are accepted or copied by this operation. Model creation remains a
    separate post-acceptance Controller responsibility associated with successful
    Training finalization.
    """

    ...


def persist_evaluation_run_transition(
    next_run: EvaluationRun,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> None:
    """Persist one caller-constructed terminal transition for an Evaluation Run.

    The caller is responsible for Evaluation result acceptance and for constructing the
    complete domain outcome, including accepted metrics/artifact metadata,
    ``completed_partial`` incompleteness facts, failure information, and final status.
    This operation neither receives Worker result candidates nor chooses among
    ``completed``, ``completed_partial``, ``failed``, and ``cancelled``.

    The operation reads the current canonical record identified by ``next_run.id``,
    applies :func:`validate_evaluation_run` to the supplied value, and requires the
    frozen :func:`validate_evaluation_run_transition` to accept
    ``current.status -> next_run.status``. No terminal record can be reopened or
    replaced with another terminal state.

    Schema, Run ID, Model, Corpus, Evaluation Protocol, complete resolved parameters,
    ``started_at``, and optional Study lineage are immutable initial facts and must be
    identical to the canonical ``RUNNING`` record. Terminal result/environment/failure/
    incompleteness facts are taken from the caller-constructed ``next_run`` subject to
    the frozen domain validator.

    Only authoritative ``run.yaml`` is replaced here using the canonical path from
    ``RepositoryLayout`` and ``FilesystemPort`` complete-replacement semantics. Formal
    artifact candidate validation/import must already have been handled by the later
    Controller result-acceptance boundary; this function does not commit artifact bytes.
    """

    ...


def persist_study_run_transition(
    next_run: StudyRun,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> None:
    """Persist one caller-constructed terminal transition for a Study Run.

    Study reconciliation/orchestration decides the terminal status and any derived
    summary before calling this operation. Persistence does not inspect Queue state,
    judge planned-coordinate satisfaction, count historical child attempts, or select
    ``completed`` versus ``completed_with_failures``.

    The current canonical Study Run is read by ``next_run.id``. The supplied value must
    pass :func:`validate_study_run`, and
    :func:`validate_study_run_transition` must accept
    ``current.status -> next_run.status`` for the one ``RUNNING -> terminal``
    transition. Terminal Study Runs are never reopened.

    Schema, Study Run ID, referenced Study, and ``started_at`` must agree with the
    canonical current record. ``next_run.plan`` must equal ``current.plan`` exactly: an
    already-finalized plan cannot change, and an absent plan cannot be fabricated during
    terminalization. If no complete plan was ever established, terminal ``FAILED`` or
    ``CANCELLED`` therefore continues to omit plan metadata under the frozen Study Run
    contract. ``summary`` and ``finished_at`` come from the caller-constructed next
    value.

    Only canonical ``run.yaml`` is replaced. Existing authoritative ``plan.jsonl``
    bytes are never mutated by terminalization.
    """

    ...


def read_training_run(
    run_id: TrainingRunId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> TrainingRun:
    """Read one exact canonical Training Run as the frozen domain value.

    The operation derives only ``layout.training_run_paths(run_id).metadata_path``,
    parses the implementation-private YAML representation into :class:`TrainingRun`,
    requires recorded ``id`` to equal the requested typed ID, and applies
    :func:`validate_training_run` before returning.

    It does not resolve referenced assets, verify canonical weight bytes, reconstruct a
    Model, inspect Queue state, or infer retry/reconciliation status. Missing or invalid
    canonical metadata is an operation failure rather than a nullable result.
    """

    ...


def read_evaluation_run(
    run_id: EvaluationRunId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> EvaluationRun:
    """Read one exact canonical Evaluation Run as the frozen domain value.

    Canonical metadata path selection is delegated to
    ``RepositoryLayout.evaluation_run_paths``. Private YAML decoding must preserve the
    existing domain types, require requested/recorded ID agreement, and apply
    :func:`validate_evaluation_run` before return.

    The read does not re-run formal result acceptance or verify accepted artifact bytes;
    those integrity/acceptance concerns remain outside this metadata read surface.
    Missing or invalid canonical metadata is an operation failure.
    """

    ...


def read_study_run(
    run_id: StudyRunId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> StudyRun:
    """Read one exact canonical Study Run as the frozen domain value.

    The operation uses only ``RepositoryLayout.study_run_paths`` for path selection,
    decodes the private YAML representation, requires requested/recorded ID agreement,
    and applies :func:`validate_study_run`.

    This metadata read does not itself parse or hash ``plan.jsonl``. Call
    :func:`read_study_plan` when reconciliation or Public API logic needs the immutable
    materialized plan and its integrity agreement with Study Run metadata.
    """

    ...


def read_study_plan(
    study_run_id: StudyRunId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> StudyPlan:
    """Read and verify one finalized canonical Study Run plan.

    The operation first reads the exact canonical :class:`StudyRun` and requires
    ``run.plan`` to be present. It then reads the exact canonical ``plan.jsonl`` bytes
    selected by ``RepositoryLayout.study_run_paths(study_run_id).plan_path`` and
    requires SHA-256 and byte size to agree with the frozen :class:`StudyRunPlan`
    metadata.

    Private JSONL decoding produces the existing :class:`StudyPlan`; the frozen
    :func:`validate_study_plan` must succeed, decoded trial count must agree with
    ``run.plan.trials``, and the total decoded evaluation-entry count must agree with
    ``run.plan.evaluation_jobs``. No public JSONL codec, plan repository, or query
    object is introduced.

    A Study Run with no finalized plan has no authoritative plan to return even if
    unfinalized bytes happen to exist at ``plan.jsonl`` after an interrupted
    materialization. Such bytes are deliberately ignored as authority until matching
    Run plan metadata exists.
    """

    ...


def list_training_runs(
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> tuple[TrainingRun, ...]:
    """Return all canonical Training Runs in deterministic identity order.

    This is the minimal typed listing surface used by later Study-lineage
    reconciliation and Public API work. It enumerates the canonical Training Run
    container through ``layout.entity_directory(EntityKind.TRAINING_RUN)`` plus
    ``FilesystemPort.list_directory`` and returns domain Runs ordered by
    ``TrainingRunId`` string value, which is deterministic for the same repository
    contents.

    Reconciliation-specific filtering is intentionally not embedded here. Callers may
    inspect each Run's optional frozen ``study`` lineage. The function does not expose a
    generic kind query, predicate language, pagination engine, cache, or repository
    service.

    Each returned record has the same parsing/ID-agreement/metadata-local validation
    semantics as :func:`read_training_run`. A malformed canonical Run record must not be
    silently converted into a partial value or treated as valid history.
    """

    ...


def list_evaluation_runs(
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> tuple[EvaluationRun, ...]:
    """Return all canonical Evaluation Runs in deterministic identity order.

    The operation is the Evaluation counterpart to :func:`list_training_runs`, using
    ``layout.entity_directory(EntityKind.EVALUATION_RUN)`` for the canonical container,
    and provides the child-lineage history needed by later Study reconciliation without
    a generic repository/query abstraction. Results are ordered by
    ``EvaluationRunId`` string value and each record satisfies
    :func:`read_evaluation_run` semantics.

    No stage/trial/Study filtering or result ranking is owned by persistence; callers
    perform those reconciliation or presentation decisions from the returned frozen
    domain records.
    """

    ...


def list_study_runs(
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> tuple[StudyRun, ...]:
    """Return all canonical Study Runs in deterministic identity order.

    This typed list uses ``layout.entity_directory(EntityKind.STUDY_RUN)`` and prevents
    later Public API/reconciliation code from inventing a second Study Run YAML parser
    or a generic ``get_entity(kind, str)`` service. Results are ordered by
    ``StudyRunId`` string value and each record uses
    :func:`read_study_run` validation semantics.

    Plan contents are intentionally not loaded for every listed Study Run. Callers that
    need one finalized plan use :func:`read_study_plan` for that exact typed identity.
    """

    ...
