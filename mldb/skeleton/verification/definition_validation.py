"""Read-only validation query boundary for reusable MLDB definitions.

This module fixes the lower-level query operations intended for a later public Controller
``validate_definition`` projection. It supports exactly Task, Corpus, Architecture,
Train Protocol, Evaluation Protocol, and Study. There is deliberately no generic
``validate(kind, id)`` dispatcher, DefinitionValidator class, entity registry,
ValidationContext graph, repository scan, or second validation-issue vocabulary.

Each entity-specific operation first checks only the exact requested canonical metadata
path derived by :class:`~mldb.skeleton.repository.layout.RepositoryLayout`. A definitive
normal ``filesystem.file_exists(...) == False`` means the requested typed definition
itself does not exist and must raise/propagate ``NotFoundError`` semantics. No alternative
directory is inspected. Once that requested metadata exists, semantic validation delegates
to the frozen typed runtime resolver rather than parsing canonical YAML independently.
Resolver-owned schema/identity/reference checks and entity-local domain validation
therefore remain single-sourced. Failures of dependencies/resources reached after target
existence is established are ordinary definition invalidity when existing structured
validation/error evidence can be faithfully projected into the frozen
:class:`ValidationReport` / :class:`ValidationIssue` vocabulary. Unexpected filesystem,
runtime, adapter, or other infrastructure failure is not converted into ``not_found`` or
a domain issue.

Architecture, Train Protocol, and Evaluation Protocol add one read-only integrity check
that normal resolution intentionally defers: when the resolved definition is sealed, the
recorded ``implementation.sha256`` must equal SHA-256 of the exact bytes at the
resolver-produced canonical sibling ``implementation_path``. Draft executable
definitions have no such hash requirement here. Validation never imports executable
Python, inspects callable shape, invokes an entrypoint, or runs pytest.

Study definition validation deliberately remains weaker than Study execution preflight.
It reuses typed Study resolution, then checks reusable-definition facts that require the
resolved referenced metadata: common Task compatibility and publication of Study-authored
Train/Evaluation Protocol parameter keys. It does not require the Study or referenced
executable definitions to be sealed, verify referenced executable implementation hashes,
verify existing-Model learned-weight bytes/hash/size beyond resolver-owned lineage/path
facts, resolve omitted protocol defaults, expand the grid, materialize a plan, allocate
a Study Run, or touch Queue/Worker state. Those execution-readiness checks remain owned
by :func:`mldb.skeleton.study.preflight.preflight_study_execution` and later Study
materialization.

The current Study format spec labels parameter publication and common Task relationships
as Study validation rules while the frozen Study resolver explicitly leaves those facts
to later Study-level validation/preflight. This module is that reusable read-only
Study-level validation boundary: it checks those two facts without importing the stronger
sealed/integrity/readiness gates from execution preflight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from ..common.errors import ValidationIssue, ValidationReport
from ..common.ids import (
    ArchitectureId,
    CorpusId,
    EntityKind,
    EvaluationProtocolId,
    StudyId,
    TaskId,
    TrainProtocolId,
)
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort


ReusableDefinitionKind: TypeAlias = Literal[
    EntityKind.TASK,
    EntityKind.CORPUS,
    EntityKind.ARCHITECTURE,
    EntityKind.TRAIN_PROTOCOL,
    EntityKind.EVALUATION_PROTOCOL,
    EntityKind.STUDY,
]
"""The exact reusable-definition kinds accepted by this validation boundary."""


ReusableDefinitionId: TypeAlias = (
    TaskId
    | CorpusId
    | ArchitectureId
    | TrainProtocolId
    | EvaluationProtocolId
    | StudyId
)
"""Typed ID union corresponding to :data:`ReusableDefinitionKind`."""


@dataclass(frozen=True, slots=True)
class DefinitionValidationResult:
    """Read-only validation result suitable for later Controller projection.

    ``kind`` and ``id`` identify the exact typed definition query that produced this
    value. The six entity-specific functions in this module guarantee their matching
    pair; no generic dispatch or runtime kind/ID registry is introduced.

    ``report`` is the frozen ordered :class:`ValidationReport`. Existing validator or
    resolver-originated issue codes and ordering must be preserved when already
    available. Checks owned by this boundary, such as sealed implementation hash
    agreement and Study compatibility/parameter publication, append ordinary
    :class:`ValidationIssue` values to the same report vocabulary rather than defining
    another issue/result hierarchy.

    ``valid`` and ``issues`` are convenience projections needed by the later Controller
    response. ``valid`` is exactly ``report.valid`` and ``issues`` is exactly
    ``report.issues``; no warning/severity filtering exists.

    The exact requested definition being absent is not ordinary invalidity. Before
    resolver delegation, each entity-specific operation derives its one typed canonical
    metadata path and calls ``filesystem.file_exists(...)``. Only a definitive normal
    ``False`` establishes requested-target absence and raises/propagates ``NotFoundError``
    semantics; it never produces this result type.

    After requested-target existence is established, ordinary invalidity includes
    resolver-owned malformed/invalid requested metadata, canonical identity/reference
    failure, missing required sibling resources, missing/invalid referenced definitions,
    Corpus integrity failure, this boundary's sealed executable hash mismatch, and the
    Study-level checks described below when the expected runtime evidence permits
    faithful validation projection. A concrete implementation must preserve a resolver's
    structured ``ValidationReport`` when one is available rather than replacing its
    issue codes with generic text. In particular, an absent upstream Task or other
    Study-referenced entity is not reclassified as requested-target ``not_found``.

    Unexpected infrastructure failure must escape this boundary as an operation failure.
    A filesystem adapter exception while performing the target-existence check is neither
    ``not_found`` nor ``valid=False``. Arbitrary ``OSError``/adapter/tooling/runtime
    exceptions must not be caught by a broad ``except Exception`` and reported as
    ordinary validation invalidity.
    """

    kind: ReusableDefinitionKind
    id: ReusableDefinitionId
    report: ValidationReport

    @property
    def valid(self) -> bool:
        """Whether the ordered report contains no validation issues."""

        ...

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        """Return the report's ordered issues without copying or reclassification."""

        ...


def validate_task_definition(
    task_id: TaskId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> DefinitionValidationResult:
    """Validate one Task through the frozen typed Task resolver.

    First derive exactly ``layout.task_metadata_path(task_id)`` and call
    ``filesystem.file_exists(...)`` for that path. A normal ``False`` means the requested
    Task itself is absent and must raise/propagate ``NotFoundError`` semantics rather than
    return ``valid=False``. An exception from that existence check remains an operation
    failure.

    Once the metadata exists, delegate the semantic lookup/validation path only to
    ``resolve_task(task_id, layout, filesystem)``. Resolver success establishes
    parsing/normalization, ``validate_task()`` acceptance, and exact
    requested/recorded/canonical identity agreement. Task has no upstream reference,
    executable sibling, sealing hash, or additional integrity rule to add here.

    Resolver-originated ordinary definition invalidity for the existing requested Task is
    returned in ``report`` when it can be faithfully represented with the existing
    validation-error information. Successful resolution returns an empty report. The
    operation performs no writes and inspects no alternative Task location.
    """

    ...


def validate_corpus_definition(
    corpus_id: CorpusId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> DefinitionValidationResult:
    """Validate one Corpus through the frozen typed Corpus resolver.

    First derive exactly ``layout.corpus_metadata_path(corpus_id)`` and call
    ``filesystem.file_exists(...)`` for that path. A normal ``False`` means the requested
    Corpus itself is absent and must raise/propagate ``NotFoundError`` semantics rather
    than return ``valid=False``. An exception from that existence check remains an
    operation failure.

    Once the metadata exists, delegate semantic validation only to
    ``resolve_corpus(corpus_id, layout, filesystem)``. Resolver success establishes
    static Corpus metadata validity, canonical identity, referenced Task resolution,
    presence of the canonical SQLite/builder siblings, and exact SQLite SHA-256 plus
    optional byte-size agreement required by Corpus v1. After the target metadata guard,
    a missing referenced Task, SQLite sibling, builder sibling, or integrity mismatch is
    requested-Corpus invalidity rather than requested-target ``not_found`` when existing
    runtime evidence can be faithfully projected.

    No builder is imported or executed. SQLite content/schema inspection beyond the
    resolver-owned immutable artifact-integrity contract is not added here. Resolver-
    originated ordinary invalidity becomes the returned report when it can be projected
    faithfully; successful resolution returns an empty report. The operation performs no
    writes and inspects no alternative Corpus location.
    """

    ...


def validate_architecture_definition(
    architecture_id: ArchitectureId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> DefinitionValidationResult:
    """Validate one Architecture, including sealed sibling-byte integrity.

    First derive exactly ``layout.architecture_metadata_path(architecture_id)`` and call
    ``filesystem.file_exists(...)`` for that path. A normal ``False`` means the requested
    Architecture itself is absent and must raise/propagate ``NotFoundError`` semantics;
    an exception from the existence check remains an operation failure.

    Once the metadata exists, call
    ``resolve_architecture(architecture_id, layout, filesystem)``. Resolver success
    supplies canonical identity/path agreement, static Architecture metadata validation,
    sibling presence, and referenced Task validity without importing Python. A missing
    implementation sibling or referenced Task reached after the target guard is ordinary
    requested-Architecture invalidity when it can be faithfully projected, not requested-
    target ``not_found``.

    If the resolved Architecture is ``draft``, return success after those checks; a draft
    has no recorded implementation-hash requirement.

    If it is ``sealed``, read the exact bytes only from the resolver-produced
    ``ArchitectureHandle.implementation_path`` using ``filesystem.read_bytes(...)``,
    calculate SHA-256 privately, and require exact agreement with
    ``metadata.implementation.sha256``. A mismatch is ordinary validation invalidity
    represented by a :class:`ValidationIssue` in the returned report. The concrete
    implementation must not call ``load_architecture_build()``, inspect or invoke
    ``build``, or run asset pytest merely to validate metadata/integrity.

    Hash calculation is implementation-private; this module introduces no HashService.
    No metadata or implementation bytes are rewritten.
    """

    ...


def validate_train_protocol_definition(
    protocol_id: TrainProtocolId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> DefinitionValidationResult:
    """Validate one Train Protocol, including sealed sibling-byte integrity.

    First derive exactly ``layout.train_protocol_metadata_path(protocol_id)`` and call
    ``filesystem.file_exists(...)`` for that path. A normal ``False`` means the requested
    Train Protocol itself is absent and must raise/propagate ``NotFoundError`` semantics;
    an exception from the existence check remains an operation failure.

    Once the metadata exists, call
    ``resolve_train_protocol(protocol_id, layout, filesystem)``. Resolver success supplies
    canonical identity/path agreement, static Train Protocol metadata validation, sibling
    presence, and referenced Task validity. A missing implementation sibling or referenced
    Task reached after the target guard is ordinary requested-Protocol invalidity when it
    can be faithfully projected, not requested-target ``not_found``.

    ``draft`` requires no implementation SHA-256 comparison. For ``sealed``, read the
    exact bytes only from the resolver-produced ``TrainProtocolHandle.implementation_path``,
    calculate SHA-256 privately, and require exact agreement with the recorded
    ``metadata.implementation.sha256``. A mismatch is an ordinary validation issue.

    Do not call ``load_train_entrypoint()``, import or inspect the sibling module, invoke
    ``train``, resolve concrete parameter overrides/defaults, or run pytest. No files are
    written.
    """

    ...


def validate_evaluation_protocol_definition(
    protocol_id: EvaluationProtocolId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> DefinitionValidationResult:
    """Validate one Evaluation Protocol, including sealed sibling-byte integrity.

    First derive exactly ``layout.evaluation_protocol_metadata_path(protocol_id)`` and
    call ``filesystem.file_exists(...)`` for that path. A normal ``False`` means the
    requested Evaluation Protocol itself is absent and must raise/propagate
    ``NotFoundError`` semantics; an exception from the existence check remains an
    operation failure.

    Once the metadata exists, call
    ``resolve_evaluation_protocol(protocol_id, layout, filesystem)``. Resolver success
    supplies canonical identity/path agreement, static Evaluation Protocol metadata
    validation, sibling presence, and referenced Task validity. A missing implementation
    sibling or referenced Task reached after the target guard is ordinary requested-
    Protocol invalidity when it can be faithfully projected, not requested-target
    ``not_found``.

    ``draft`` requires no implementation SHA-256 comparison. For ``sealed``, read the
    exact bytes only from the resolver-produced
    ``EvaluationProtocolHandle.implementation_path``, calculate SHA-256 privately, and
    require exact agreement with the recorded ``metadata.implementation.sha256``. A
    mismatch is an ordinary validation issue.

    Do not call ``load_evaluation_entrypoint()``, import/inspect the sibling module,
    invoke ``evaluate``, validate a concrete Evaluation result, or run pytest. No files
    are written.
    """

    ...


def validate_study_definition(
    study_id: StudyId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> DefinitionValidationResult:
    """Validate one reusable Study without upgrading it to execution preflight.

    First derive exactly ``layout.study_metadata_path(study_id)`` and call
    ``filesystem.file_exists(...)`` for that path. A normal ``False`` means the requested
    Study itself is absent and must raise/propagate ``NotFoundError`` semantics rather
    than return ``valid=False``. An exception from that existence check remains an
    operation failure.

    Once the metadata exists, call ``resolve_study(study_id, layout, filesystem)``.
    Resolver success establishes canonical Study identity,
    ``validate_study_metadata()`` acceptance, and successful typed resolution of every
    direct Study reference appropriate to its Model-source mode. That includes resolver-
    owned Corpus artifact integrity and, for existing Models, the frozen Model resolver's
    completed Training Run lineage, canonical weights metadata/path contract, weights-file
    existence, and lineage Architecture resolution. Any missing/invalid Study-referenced
    Corpus, Protocol, Architecture, Model, nested Task, or required resource reached after
    the target guard is ordinary requested-Study invalidity when existing runtime evidence
    can be faithfully projected, not requested-target ``not_found``.

    Because ``StudyHandle`` intentionally embeds no dependency handles, this operation
    may re-resolve the exact direct references through the existing entity-specific
    resolvers solely to establish reusable Study-definition facts. It must not scan
    repository directories or infer dependencies from arbitrary strings.

    For a training-source Study, additionally verify:

    - training Corpus, Train Protocol, and every selected Architecture record one common
      ``TaskId``;
    - every key in ``model.train.parameters`` is published verbatim by the selected
      Train Protocol's ``parameters`` mapping.

    For an existing-Model Study, additionally verify:

    - every selected Model's Task, defined by its resolver-produced lineage
      Architecture metadata, is the same common ``TaskId``.

    For both source modes, additionally verify:

    - every evaluation Corpus and Evaluation Protocol records that same common Task; and
    - every key in each ``StudyEvaluationStage.parameters`` mapping is published
      verbatim by that stage's selected Evaluation Protocol.

    These failures are ordinary :class:`ValidationIssue` values. Deterministic issue
    order must not depend on mapping insertion order: model-source compatibility follows
    authored source order (Architecture or existing-Model order), Study-authored
    training parameter keys are checked in ascending Python string order, evaluation
    stages follow authored stage order, and each stage's parameter keys are checked in
    ascending Python string order.

    This operation deliberately does **not** call ``preflight_study_execution()`` and
    therefore does not require ``StudyStatus.SEALED``; does not require referenced
    Architecture/Train/Evaluation Protocol status ``SEALED``; does not verify their
    implementation SHA-256 against current bytes; does not verify existing-Model learned
    weights SHA-256/byte count beyond the facts already owned by ``resolve_model()``;
    does not resolve omitted public-parameter defaults; and does not expand/materialize
    the Study grid.

    The distinction is intentional: this operation answers whether the reusable Study
    definition and its typed references/declared interfaces are valid now, not whether a
    new Study Run may be allocated now. No Run, Model, Queue, pytest, executable import,
    entrypoint invocation, or repository mutation occurs.
    """

    ...
