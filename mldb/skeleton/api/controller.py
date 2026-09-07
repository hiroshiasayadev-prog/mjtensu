"""Transport-independent public Controller application API signatures for MLDB.

This module is the thin public composition boundary over the freeze-existing lower
application/domain operations. It defines exactly the seven v1 Controller operations
from ``spec:mldb.api.controller`` and deliberately contains no second implementation of
validation, sealing, Study planning, Queue lifecycle, progress classification, entity
resolution, or cancellation semantics.

Public kind-oriented operations accept one :class:`~mldb.skeleton.common.ids.EntityKind`
and an exact string identity. They mechanically construct only the frozen entity-specific
``NewType`` selected by that kind before delegating. No generic EntityId, registry,
directory probing, kind guessing, request envelope, Controller/Service object, dependency
context, or generic result wrapper is introduced.

The lower
:func:`mldb.skeleton.orchestration.study_launch.launch_study_execution` boundary owns the
post-allocation setup-failure cleanup and now surfaces
:class:`StudyExecutionSetupError` carrying the exact allocated
:class:`~mldb.skeleton.common.ids.StudyRunId`. This module re-exports that same exception
as the public Python representation of ``execution_setup_failed`` and lets it cross
``execute_study`` unchanged. No Study Run scan, timestamp/sequence inference, or duplicate
API-local wrapper is needed.

Unexpected infrastructure/runtime exceptions are likewise not wrapped here into a new
``internal_failure`` Python exception hierarchy. Future transports may map unexpected
exceptions to that external error code while preserving the existing expected MLDB error
vocabulary for normal application/domain failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, TypeAlias

from ..common.errors import ValidationIssue
from ..common.ids import (
    ArchitectureId,
    CorpusId,
    EntityKind,
    EvaluationProtocolId,
    StudyId,
    StudyRunId,
    TaskId,
    TrainProtocolId,
)
from ..orchestration.queue_ports import QueuePort
from ..orchestration.study_cancellation import (
    StudyCancellationResult,
    request_study_run_cancellation as _request_study_run_cancellation,
)
from ..orchestration.study_launch import (
    StudyExecutionSetupError,
    launch_study_execution as _launch_study_execution,
)
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.entity_query import (
    EntitySummary,
    EntityValue,
    get_entity as _get_entity,
    list_entities as _list_entities,
)
from ..runtime.resolution import resolve_study as _resolve_study
from ..study.preflight import preflight_study_execution as _preflight_study_execution
from ..study.progress import StudyRunProgress, get_study_run_progress as _get_study_run_progress
from ..study.run import StudyRunStatus
from ..verification.definition_validation import (
    validate_architecture_definition as _validate_architecture_definition,
    validate_corpus_definition as _validate_corpus_definition,
    validate_evaluation_protocol_definition as _validate_evaluation_protocol_definition,
    validate_study_definition as _validate_study_definition,
    validate_task_definition as _validate_task_definition,
    validate_train_protocol_definition as _validate_train_protocol_definition,
)
from ..verification.lifecycle import (
    SealResult,
    seal_architecture as _seal_architecture,
    seal_evaluation_protocol as _seal_evaluation_protocol,
    seal_study as _seal_study,
    seal_train_protocol as _seal_train_protocol,
)
from ..verification.sealing import PytestRunner


__all__ = (
    "DefinitionValidationResponse",
    "SealDefinitionResponse",
    "StudyExecutionResponse",
    "StudyExecutionSetupError",
    "validate_definition",
    "seal_definition",
    "execute_study",
    "get_study_run",
    "cancel_study_run",
    "get_entity",
    "list_entities",
)


_ReusableDefinitionKind: TypeAlias = Literal[
    EntityKind.TASK,
    EntityKind.CORPUS,
    EntityKind.ARCHITECTURE,
    EntityKind.TRAIN_PROTOCOL,
    EntityKind.EVALUATION_PROTOCOL,
    EntityKind.STUDY,
]

_ReusableDefinitionId: TypeAlias = (
    TaskId
    | CorpusId
    | ArchitectureId
    | TrainProtocolId
    | EvaluationProtocolId
    | StudyId
)

_SealableDefinitionKind: TypeAlias = Literal[
    EntityKind.ARCHITECTURE,
    EntityKind.TRAIN_PROTOCOL,
    EntityKind.EVALUATION_PROTOCOL,
    EntityKind.STUDY,
]

_SealableDefinitionId: TypeAlias = (
    ArchitectureId | TrainProtocolId | EvaluationProtocolId | StudyId
)


@dataclass(frozen=True, slots=True)
class DefinitionValidationResponse:
    """Thin public projection of one reusable-definition validation result.

    ``kind`` and ``id`` are the exact typed request identity selected by
    :func:`validate_definition`. ``valid`` and ``issues`` are copied directly from the
    lower :class:`DefinitionValidationResult` convenience projections; issue codes,
    messages, paths, and ordering are not reclassified here.

    Exact requested-definition absence remains exceptional ``NotFoundError`` semantics
    from the lower validation boundary. Existing invalid definitions produce this value
    with ``valid=False`` and do not become ``ValidationFailedError`` merely because the
    public Controller boundary was used.
    """

    kind: _ReusableDefinitionKind
    id: _ReusableDefinitionId
    valid: bool
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class SealDefinitionResponse:
    """Public result of one supported definition sealing operation.

    ``status`` is always the Controller contract value ``"sealed"`` after a definitive
    successful lower operation. ``result`` is copied unchanged from the lower sealing
    boundary and therefore distinguishes a new transition from an integrity-valid
    idempotent replay without re-validating or re-resolving the definition here.
    """

    kind: _SealableDefinitionKind
    id: _SealableDefinitionId
    status: Literal["sealed"]
    result: SealResult


@dataclass(frozen=True, slots=True)
class StudyExecutionResponse:
    """Successful public response after complete Study setup and Queue admission.

    This value is constructed only after :func:`_launch_study_execution` returns its
    finalized canonical Study Run. ``status`` is the exact canonical status from that
    returned Run; successful launch is expected to return ``RUNNING`` under the frozen
    lower contract. No child execution is awaited.
    """

    study_run_id: StudyRunId
    study_id: StudyId
    status: StudyRunStatus


def validate_definition(
    kind: EntityKind,
    entity_id: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> DefinitionValidationResponse:
    """Validate one supported reusable definition through its frozen typed boundary.

    Dispatch is mechanical and exact:

    - ``TASK`` -> ``_validate_task_definition(TaskId(entity_id), ...)``;
    - ``CORPUS`` -> ``_validate_corpus_definition(CorpusId(entity_id), ...)``;
    - ``ARCHITECTURE`` ->
      ``_validate_architecture_definition(ArchitectureId(entity_id), ...)``;
    - ``TRAIN_PROTOCOL`` ->
      ``_validate_train_protocol_definition(TrainProtocolId(entity_id), ...)``;
    - ``EVALUATION_PROTOCOL`` ->
      ``_validate_evaluation_protocol_definition(EvaluationProtocolId(entity_id), ...)``;
    - ``STUDY`` -> ``_validate_study_definition(StudyId(entity_id), ...)``.

    The returned public value copies exactly ``result.kind``, ``result.id``,
    ``result.valid``, and ``result.issues``. It does not parse YAML, repeat schema or
    dependency validation, alter issue order, or turn ordinary invalidity into an
    exception.

    ``TRAINING_RUN``, ``MODEL``, ``EVALUATION_RUN``, and ``STUDY_RUN`` are valid
    :class:`EntityKind` members but this operation is not defined for them by the
    Controller contract. They therefore raise ``UnsupportedOperationError`` before any
    lower mutation/read-validation dispatcher is invoked; this function does not invent
    validation semantics for execution records or Models.
    """

    ...


def seal_definition(
    kind: EntityKind,
    entity_id: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    runner: PytestRunner | None = None,
) -> SealDefinitionResponse:
    """Seal one supported reusable definition through its lifecycle-specific boundary.

    Dispatch is exact:

    - ``ARCHITECTURE`` -> ``_seal_architecture(ArchitectureId(entity_id), ..., runner)``;
    - ``TRAIN_PROTOCOL`` ->
      ``_seal_train_protocol(TrainProtocolId(entity_id), ..., runner)``;
    - ``EVALUATION_PROTOCOL`` ->
      ``_seal_evaluation_protocol(EvaluationProtocolId(entity_id), ..., runner)``;
    - ``STUDY`` -> ``_seal_study(StudyId(entity_id), layout, filesystem)``.

    For Architecture, Train Protocol, and Evaluation Protocol, ``runner`` is required;
    absence is an ``InvalidRequestError`` detected before mutation. For Study,
    ``runner`` must be ``None`` because Study sealing has no executable-asset pytest
    gate; supplying one is likewise an ``InvalidRequestError`` rather than silently
    ignoring a request dependency. The Study path never forwards the runner.

    ``TASK`` and ``CORPUS`` have no sealing transition in their frozen lifecycle
    contracts. Run/Model kinds likewise have no public ``seal_definition`` operation.
    All six unsupported kinds raise ``UnsupportedOperationError`` before any mutating
    lower function is called.

    On success, construct :class:`SealDefinitionResponse` directly from the selected
    typed kind/ID and the one returned :class:`SealResult`, with ``status="sealed"``.
    Do not rerun read-only validation or resolution merely to build the response.
    """

    ...


def execute_study(
    study_id: StudyId,
    allocation_date: date,
    started_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
    *,
    admitted_at: str,
    failed_at: object,
) -> StudyExecutionResponse:
    """Resolve, preflight, and launch one exact sealed Study without waiting for work.

    The successful composition is exactly::

        study = _resolve_study(study_id, layout, filesystem)
        prepared = _preflight_study_execution(study, layout, filesystem)
        run = _launch_study_execution(
            prepared,
            allocation_date,
            started_at,
            layout,
            filesystem,
            queue,
            admitted_at=admitted_at,
            failed_at=failed_at,
        )

    Only after the launcher returns does this boundary project::

        StudyExecutionResponse(
            study_run_id=run.id,
            study_id=run.study,
            status=run.status,
        )

    ``resolve_study`` plus ``preflight_study_execution`` therefore complete before the
    lower launcher is entered and before Study Run allocation. This function does not
    re-check sealed status, compatibility, parameter publication/defaults, static asset
    integrity, grid legality, plan expansion, logical jobs, or Queue lifecycle.

    ``allocation_date`` and ``started_at`` are forwarded to Study Run allocation through
    the lower launcher. ``admitted_at`` is forwarded unchanged to Queue admission, and
    ``failed_at`` is forwarded unchanged for the launcher's post-allocation Study Run
    failure transition. No clock/current-time provider is introduced.

    If the lower launcher fails after successful Study Run allocation, it raises the
    re-exported :class:`StudyExecutionSetupError` carrying the exact allocated
    ``study_run_id``. This public boundary does not catch, replace, or reclassify that
    exception; normal application callers and future transport adapters can inspect
    ``exc.study_run_id`` directly. The launcher's underlying setup/cleanup failure remains
    available through normal Python exception chaining.

    Failures before successful Study Run allocation do not become
    :class:`StudyExecutionSetupError` and continue to cross this boundary under their
    existing exception semantics. This function never lists/scans Study Runs, derives an
    identity from ``allocation_date``/``started_at``/sequence state, terminalizes a Run,
    inspects Queue state for recovery, or performs any setup cleanup itself.
    """

    ...


def get_study_run(
    study_run_id: StudyRunId,
    as_of: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
) -> StudyRunProgress:
    """Return the frozen read-only Study Run progress value directly.

    Delegate exactly to::

        _get_study_run_progress(study_run_id, as_of, layout, filesystem, queue)

    :class:`StudyRunProgress` already contains the exact canonical ``StudyRun`` identity
    and status, optional training/evaluation Controller progress sections, and optional
    incomplete-coordinate diagnostics required by ``spec:mldb.api.controller``. A
    second API DTO would only duplicate that frozen application value, so this public
    operation returns it directly.

    No Queue row/attempt is exposed and this function performs no progress
    reclassification, reconciliation, Model repair, or other mutation.
    """

    ...


def cancel_study_run(
    study_run_id: StudyRunId,
    finished_at: object,
    queue_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
) -> StudyCancellationResult:
    """Request Study-level cancellation through the frozen orchestration boundary.

    Delegate exactly to::

        _request_study_run_cancellation(
            study_run_id,
            finished_at,
            queue_at,
            layout,
            filesystem,
            queue,
        )

    The returned ``"accepted"`` / ``"already_terminal"`` value is already the exact
    public Controller response and is returned unchanged. This function never inspects,
    cancels, or closes individual Queue jobs/attempts itself.
    """

    ...


def get_entity(
    kind: EntityKind,
    entity_id: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> EntityValue:
    """Read one canonical entity through the frozen runtime entity-query dispatcher.

    Delegate exactly to ``_get_entity(kind, entity_id, layout, filesystem)`` and return
    its canonical parsed domain value unchanged. All ten :class:`EntityKind` members are
    supported by that lower boundary. This public function performs no duplicate ID
    conversion, resolver selection, directory scan, fallback lookup, or DTO projection.
    """

    ...


def list_entities(
    kind: EntityKind,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> tuple[EntitySummary, ...]:
    """List deterministic compact entity summaries through the frozen query boundary.

    Delegate exactly to ``_list_entities(kind, layout, filesystem)`` and return the
    resulting tuple unchanged. V1 adds no public filtering, pagination, ranking, or
    alternate repository enumeration at this layer.
    """

    ...
