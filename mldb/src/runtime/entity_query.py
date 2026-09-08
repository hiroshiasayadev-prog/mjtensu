"""Read-only typed entity query and deterministic discovery signatures for MLDB.

This module fixes the lower-level runtime boundary used by later Controller
``get_entity`` and ``list_entities`` operations. Exact reads dispatch from one requested
:class:`~mldb.skeleton.common.ids.EntityKind` plus ID string into the already-frozen
entity-specific resolver or Run-read operation, then expose only the canonical parsed
domain value. Resolver provenance paths, executable/resource paths, Run plan bytes, and
other handle-only runtime details do not cross this query boundary.

Listing is deliberately small. Run kinds reuse the frozen typed Run listing operations.
Reusable-definition kinds and Model enumerate only their canonical containing directory
through :meth:`~mldb.skeleton.repository.layout.RepositoryLayout.entity_directory` and
:meth:`~mldb.skeleton.repository.ports.FilesystemPort.list_directory`, derive candidate
IDs only from that kind's canonical physical filename suffixes, and pass every candidate
through the same frozen typed resolution used by exact reads. No generic YAML parser,
generic repository/CRUD service, directory fallback, query language, pagination,
filtering, ranking, tags search, registry, cache, or BaseEntity wrapper is introduced.

For single-file Task, Model, and Study discovery, ``.yaml`` entries are candidates. For
sibling-file definitions, every suffix that belongs to that kind's canonical physical
form contributes its basename as a candidate: Corpus uses ``.yaml``, ``.sqlite``, and
``.py``; Architecture, Train Protocol, and Evaluation Protocol use ``.yaml`` and
``.py``. Duplicate basenames from required siblings represent one candidate ID. Entries
whose suffix is not part of the requested kind's canonical physical form are not MLDB
entity candidates and are ignored.

A canonical candidate is never skipped because it is malformed. Every discovered
candidate must complete exact typed resolution; malformed metadata, invalid identity,
missing required siblings, broken declared references, or resolver-owned integrity/
lineage failure aborts the complete listing as an operation failure. In particular, an
orphan canonical sibling such as an Architecture ``<id>.py`` without ``<id>.yaml`` still
contributes ``<id>`` and therefore fails resolution instead of disappearing from the
list. Run listing inherits the corresponding frozen ``list_*_runs`` failure semantics
and this boundary does not catch-and-drop invalid Run records.

Successful results are ordered lexically by the resolved entity ID string. Candidate
filesystem order and sibling encounter order therefore do not affect the returned
collection. Read and list operations perform no sealing, Run allocation, plan loading,
Queue/Worker mutation, reconciliation, artifact loading, executable import, or other
canonical-state mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from ..catalog.architecture import Architecture
from ..catalog.corpus import Corpus
from ..catalog.task import Task
from ..common.ids import (
    ArchitectureId as _ArchitectureId,
    CorpusId as _CorpusId,
    EntityKind,
    EvaluationProtocolId as _EvaluationProtocolId,
    EvaluationRunId as _EvaluationRunId,
    ModelId as _ModelId,
    StudyId as _StudyId,
    StudyRunId as _StudyRunId,
    TaskId as _TaskId,
    TrainingRunId as _TrainingRunId,
    TrainProtocolId as _TrainProtocolId,
)
from ..evaluation.protocol import EvaluationProtocol
from ..evaluation.run import EvaluationRun
from ..model.identity import Model
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..study.definition import Study
from ..study.run import StudyRun
from ..training.protocol import TrainProtocol
from ..training.run import TrainingRun
from .resolution import (
    resolve_architecture as _resolve_architecture,
    resolve_corpus as _resolve_corpus,
    resolve_evaluation_protocol as _resolve_evaluation_protocol,
    resolve_model as _resolve_model,
    resolve_study as _resolve_study,
    resolve_task as _resolve_task,
    resolve_train_protocol as _resolve_train_protocol,
)
from .run_persistence import (
    list_evaluation_runs as _list_evaluation_runs,
    list_study_runs as _list_study_runs,
    list_training_runs as _list_training_runs,
    read_evaluation_run as _read_evaluation_run,
    read_study_run as _read_study_run,
    read_training_run as _read_training_run,
)


EntityValue: TypeAlias = (
    Task
    | Corpus
    | Architecture
    | TrainProtocol
    | TrainingRun
    | Model
    | EvaluationProtocol
    | EvaluationRun
    | Study
    | StudyRun
)
"""Canonical parsed domain value returned by :func:`get_entity`.

The union contains exactly the ten frozen :class:`EntityKind` members. It is not a
common entity base class and adds no generic metadata protocol. Each value remains the
existing entity-specific domain type selected by ``kind``.
"""


@dataclass(frozen=True, slots=True)
class EntitySummary:
    """Small deterministic list projection for one resolved MLDB entity.

    ``kind`` and ``id`` identify the entity selected from a listing. ``id`` is the
    string form of the already-resolved entity-specific ID; the accompanying ``kind``
    retains the type discriminator without introducing a generic entity-ID class.

    ``status`` is the canonical persisted status string when that domain type defines a
    ``status`` field. It is therefore present for Architecture, Train Protocol,
    Training Run, Evaluation Protocol, Evaluation Run, Study, and Study Run, and is
    ``None`` for Task, Corpus, and Model. No name, description, lineage, result,
    parameter, path, artifact, or complete entity metadata is duplicated here.
    """

    kind: EntityKind
    id: str
    status: str | None = None


def get_entity(
    kind: EntityKind,
    entity_id: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> EntityValue:
    """Read one exact requested entity through the frozen typed runtime boundary.

    ``entity_id`` is converted only to the entity-specific frozen ID type selected by
    ``kind``; it is never used to scan directories or probe another kind as fallback.
    Dispatch is exactly:

    - ``TASK`` -> ``resolve_task(TaskId(entity_id), ...)`` -> ``handle.metadata``;
    - ``CORPUS`` -> ``resolve_corpus(CorpusId(entity_id), ...)`` -> ``handle.metadata``;
    - ``ARCHITECTURE`` -> ``resolve_architecture(ArchitectureId(entity_id), ...)`` ->
      ``handle.metadata``;
    - ``TRAIN_PROTOCOL`` -> ``resolve_train_protocol(TrainProtocolId(entity_id), ...)``
      -> ``handle.metadata``;
    - ``TRAINING_RUN`` -> ``read_training_run(TrainingRunId(entity_id), ...)``;
    - ``MODEL`` -> ``resolve_model(ModelId(entity_id), ...)`` -> ``handle.metadata``;
    - ``EVALUATION_PROTOCOL`` ->
      ``resolve_evaluation_protocol(EvaluationProtocolId(entity_id), ...)`` ->
      ``handle.metadata``;
    - ``EVALUATION_RUN`` -> ``read_evaluation_run(EvaluationRunId(entity_id), ...)``;
    - ``STUDY`` -> ``resolve_study(StudyId(entity_id), ...)`` -> ``handle.metadata``;
    - ``STUDY_RUN`` -> ``read_study_run(StudyRunId(entity_id), ...)``.

    Successful definition/Model reads therefore inherit requested/recorded/canonical
    identity agreement plus every validation, reference, integrity, and lineage check
    owned by the corresponding frozen resolver. Successful Run reads inherit the frozen
    typed ``read_*`` parsing, ID-agreement, and metadata-local validation contract.

    Model intentionally returns only the frozen :class:`Model` metadata from the
    successfully resolved Model handle. Resolution may establish completed Training Run
    lineage, canonical weights-path existence, and Architecture resolution as already
    required by ``resolve_model``; this read does not expose the Training Run,
    Architecture handle, or weights path and does not load or hash learned bytes.

    Study Run exact read returns only canonical :class:`StudyRun` metadata. It does not
    call ``read_study_plan`` merely because plan metadata is present.

    Missing, malformed, invalid, or inconsistent canonical state is an operation failure
    under the selected frozen typed boundary. This function performs no exception
    suppression or translation merely to return a nullable value and performs no
    mutation.
    """

    if kind is EntityKind.TASK:
        return _resolve_task(_TaskId(entity_id), layout, filesystem).metadata
    if kind is EntityKind.CORPUS:
        return _resolve_corpus(_CorpusId(entity_id), layout, filesystem).metadata
    if kind is EntityKind.ARCHITECTURE:
        return _resolve_architecture(_ArchitectureId(entity_id), layout, filesystem).metadata
    if kind is EntityKind.TRAIN_PROTOCOL:
        return _resolve_train_protocol(_TrainProtocolId(entity_id), layout, filesystem).metadata
    if kind is EntityKind.TRAINING_RUN:
        return _read_training_run(_TrainingRunId(entity_id), layout, filesystem)
    if kind is EntityKind.MODEL:
        return _resolve_model(_ModelId(entity_id), layout, filesystem).metadata
    if kind is EntityKind.EVALUATION_PROTOCOL:
        return _resolve_evaluation_protocol(
            _EvaluationProtocolId(entity_id), layout, filesystem
        ).metadata
    if kind is EntityKind.EVALUATION_RUN:
        return _read_evaluation_run(_EvaluationRunId(entity_id), layout, filesystem)
    if kind is EntityKind.STUDY:
        return _resolve_study(_StudyId(entity_id), layout, filesystem).metadata
    if kind is EntityKind.STUDY_RUN:
        return _read_study_run(_StudyRunId(entity_id), layout, filesystem)
    raise ValueError(f"unsupported entity kind: {kind!r}")


def list_entities(
    kind: EntityKind,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> tuple[EntitySummary, ...]:
    """List one requested entity kind as compact summaries in deterministic ID order.

    For ``TRAINING_RUN``, ``EVALUATION_RUN``, and ``STUDY_RUN``, dispatch respectively
    to the frozen ``list_training_runs()``, ``list_evaluation_runs()``, and
    ``list_study_runs()`` operations. Do not rescan Run directories here and do not load
    Study ``plan.jsonl`` while listing Study Runs.

    For Task, Corpus, Architecture, Train Protocol, Model, Evaluation Protocol, and
    Study, derive exactly ``layout.entity_directory(kind)`` and enumerate that one
    directory exactly once through ``filesystem.list_directory(...)``. Candidate
    basenames are formed only from the canonical physical suffixes documented by this
    module. Each unique basename is then read through the same exact typed resolver
    dispatch as :func:`get_entity`; no YAML is parsed directly in listing code and no
    alternative directory is searched when a candidate fails.

    A summary copies only ``kind``, ``str(entity.id)``, and, when the resolved domain
    value defines ``status``, its persisted string value. Task, Corpus, and Model use
    ``status=None``. The complete domain value is not embedded in the summary.

    After every candidate has resolved successfully, summaries are ordered by
    ``summary.id`` using normal Python lexical string ordering. This final resolved-ID
    ordering is authoritative even though ``FilesystemPort.list_directory`` itself is
    already deterministic. If two physical candidates would resolve to the same entity
    ID, they produce one summary only when they are the expected sibling files sharing
    one basename; identity disagreement remains a resolver failure rather than a
    de-duplication rule.

    Any malformed canonical candidate aborts the listing. The function must not catch a
    candidate resolution/Run-read failure and continue with a deceptively partial list.
    Non-canonical suffixes do not constitute entities and are ignored. Filesystem access
    failures likewise are not converted into an empty list by this boundary.

    Listing performs no lifecycle mutation, Run allocation, Queue/reconciliation work,
    artifact loading, executable import, filtering, pagination, ranking, search, or
    cache population.
    """

    if kind is EntityKind.TRAINING_RUN:
        entities = _list_training_runs(layout, filesystem)
        return _summaries(kind, entities)
    if kind is EntityKind.EVALUATION_RUN:
        entities = _list_evaluation_runs(layout, filesystem)
        return _summaries(kind, entities)
    if kind is EntityKind.STUDY_RUN:
        entities = _list_study_runs(layout, filesystem)
        return _summaries(kind, entities)

    suffixes = _CANONICAL_SUFFIXES[kind]
    directory = layout.entity_directory(kind)
    candidates: set[str] = set()
    for entry in filesystem.list_directory(directory):
        for suffix in suffixes:
            if entry.name.endswith(suffix):
                candidates.add(entry.name[: -len(suffix)])
                break

    entities = tuple(
        get_entity(kind, candidate, layout, filesystem)
        for candidate in sorted(candidates)
    )
    return _summaries(kind, entities)


_CANONICAL_SUFFIXES: dict[EntityKind, tuple[str, ...]] = {
    EntityKind.TASK: (".yaml",),
    EntityKind.CORPUS: (".yaml", ".sqlite", ".py"),
    EntityKind.ARCHITECTURE: (".yaml", ".py"),
    EntityKind.TRAIN_PROTOCOL: (".yaml", ".py"),
    EntityKind.MODEL: (".yaml",),
    EntityKind.EVALUATION_PROTOCOL: (".yaml", ".py"),
    EntityKind.STUDY: (".yaml",),
}


def _summaries(kind: EntityKind, entities: tuple[EntityValue, ...]) -> tuple[EntitySummary, ...]:
    summaries = []
    for entity in entities:
        status = getattr(entity, "status", None)
        summaries.append(
            EntitySummary(
                kind=kind,
                id=str(entity.id),
                status=None if status is None else status.value,
            )
        )
    return tuple(sorted(summaries, key=lambda summary: summary.id))
