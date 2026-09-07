"""Typed canonical asset-resolution signatures for MLDB runtime definitions.

This module fixes the public boundary from one exact typed MLDB identity to one
freeze-existing runtime handle. Resolution uses only canonical paths supplied by
:class:`~mldb.skeleton.repository.layout.RepositoryLayout`. Filesystem-oriented
existence and byte/text access is performed through
:class:`~mldb.skeleton.repository.ports.FilesystemPort`. When an entity-specific data
schema requires inspection of a database artifact, the concrete resolver may open the
already selected canonical execution path read-only as an implementation-private detail;
that does not create another repository lookup mechanism or public I/O abstraction.

The concrete implementation is responsible for implementation-private YAML parsing and
normalization, entity-specific static validation, requested/recorded/canonical identity
agreement, required-resource checks, declared upstream-reference traversal, applicable
artifact integrity, Model lineage completion, and construction of the frozen handle.
Every handle returned by this module is resolver-produced: repository-provenance paths
are present and execution-resource paths are canonical repository locations. Distributed
Worker materialization is a separate legal handle-construction source and never changes
these resolver guarantees.
No YAML parser/codec abstraction, generic resolver, registry, repository wrapper,
resolved-entity generic, Run handle, or dependency-graph type is public here.

Resolution never imports asset-owned Python code, invokes executable entrypoints, runs
pytest, seals definitions, resolves public parameters, decides cross-asset execution
compatibility, allocates Runs, materializes Study plans, or participates in Queue/Worker
orchestration. Draft Architecture, Train Protocol, Evaluation Protocol, and Study
metadata may resolve successfully when their own static metadata contracts are valid.
"""

from __future__ import annotations

from ..common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    ModelId,
    StudyId,
    TaskId,
    TrainProtocolId,
)
from ..model.loading import ModelHandle
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from .catalog_handles import ArchitectureHandle, CorpusHandle, TaskHandle
from .definition_handles import EvaluationProtocolHandle, StudyHandle, TrainProtocolHandle


def resolve_task(
    task_id: TaskId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> TaskHandle:
    """Resolve one Task from its exact typed identity and canonical YAML location.

    The implementation must derive only ``layout.task_metadata_path(task_id)``, require
    that canonical metadata file through ``filesystem``, parse/normalize it as the
    frozen Task domain value, and pass it through ``validate_task()``. Successful
    resolution requires ``task_id == metadata.id``; because the canonical path itself
    was derived from ``task_id``, this also establishes requested/recorded/canonical
    repository identity agreement without scanning or filename guessing.

    Task has no declared upstream references or required sibling resources in v1.
    Malformed metadata, unsupported schema, static validation failure, or identity
    disagreement is an operation failure. Successful return is the frozen
    :class:`TaskHandle` only, with ``metadata_path`` equal to the canonical YAML path and
    therefore non-``None``; no ``ValidationReport`` or generic result wrapper is returned.
    """

    ...


def resolve_corpus(
    corpus_id: CorpusId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> CorpusHandle:
    """Resolve one immutable, Task-consistent Corpus for runtime consumption.

    The implementation must derive the metadata, SQLite artifact, and Python builder
    paths only through the corresponding ``RepositoryLayout`` operations for
    ``corpus_id`` and require all three as regular files through ``filesystem``. It
    parses/normalizes the YAML as the frozen Corpus value, applies ``validate_corpus()``,
    and requires ``corpus_id == metadata.id``.

    ``metadata.task`` is the only declared upstream MLDB reference and must itself
    succeed through :func:`resolve_task`. The resolved Task is not embedded in the
    returned handle, but it is the normative semantic authority used to validate the
    Corpus data selected by ``metadata.data.schema``. This is intrinsic Corpus validity,
    not the broader Training/Evaluation compatibility decision that combines independently
    selected assets.

    Corpus v1 has no later frozen artifact-loader boundary that owns immutable SQLite
    integrity. Therefore successful Corpus resolution must compare the exact canonical
    SQLite bytes obtained through ``filesystem`` with ``metadata.artifact.sha256`` and,
    when recorded, ``metadata.artifact.bytes``. The builder Python file is never
    imported or executed and has no generic v1 integrity hash contract.

    Before returning a handle, resolution must also validate the domain-critical
    concrete-data contract selected by ``metadata.data.schema`` against that exact
    canonical SQLite artifact. This inspection is implementation-private: the concrete
    implementation may open ``artifact_path`` with a read-only SQLite connection after
    canonical path selection and integrity verification. It must not introduce a
    ``SqlitePort``, Corpus repository/service, schema-registry API, repository scan, or
    another public resolution boundary.

    For ``mjtensu.mldb/image-classification-corpus/v1``, successful resolution requires
    the canonical table named by ``metadata.data.table`` and the schema-required core
    columns ``sample_id``, ``split``, ``target``, ``class_index``, plus the payload column
    named by ``metadata.representation['payload_column']``. The representation must
    satisfy this concrete schema: ``kind`` is ``image``, and its declared ``dtype``,
    ``shape``, and payload column must describe the materialized payload rather than an
    alternative runtime reinterpretation.

    For a referenced categorical Task, every canonical sample row must have ``target``
    equal to one of the Task's normative ordered labels, and ``class_index`` must equal
    the zero-based position of that exact ``target`` in the ordered label tuple. Unknown
    targets, negative/out-of-range indices, or any target/index disagreement invalidate
    the Corpus even when its immutable bytes match the recorded SHA-256.

    Every canonical sample payload must be storage- and byte-compatible with the declared
    image representation. In particular, a declared ``uint8`` BLOB representation must
    contain a non-NULL SQLite BLOB whose byte length is exactly the element count implied
    by the declared shape; a payload that cannot be interpreted as exactly that declared
    dtype and shape is invalid. Unsupported or internally inconsistent representation
    combinations must be rejected rather than guessed, coerced, reshaped, or silently
    interpreted under another representation. No generic tensor/data framework is
    implied by this concrete-schema check.

    Missing canonical siblings, malformed or invalid metadata, identity disagreement,
    missing/invalid Task lineage, SQLite integrity disagreement, unsupported concrete data
    schema, or any selected concrete-schema semantic/representation violation is an
    operation failure. Success returns exactly :class:`CorpusHandle` with the validated
    metadata, non-``None`` canonical ``metadata_path`` and ``builder_path``, and canonical
    ``artifact_path``. Downstream Training/Evaluation preflight may therefore trust the
    returned Corpus's Task label semantics, class-index ABI, and materialized model-input
    representation without reopening SQLite or duplicating these checks.
    """

    ...


def resolve_architecture(
    architecture_id: ArchitectureId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> ArchitectureHandle:
    """Resolve one Architecture and its canonical YAML/Python sibling locations.

    The implementation must derive only the Architecture metadata and implementation
    paths exposed by ``layout`` for ``architecture_id`` and require both regular files
    through ``filesystem``. It parses/normalizes the YAML, applies
    ``validate_architecture_metadata()``, and requires
    ``architecture_id == metadata.id``.

    ``metadata.task`` is the only declared upstream MLDB reference and must resolve
    through :func:`resolve_task`; the Task handle is intentionally not embedded in the
    returned Architecture handle. A statically valid ``draft`` Architecture may resolve
    successfully. Resolution therefore does not impose ``sealed`` status.

    This operation fixes the canonical implementation path but never imports Python or
    exposes ``build``. Sealed implementation SHA-256 verification remains owned by the
    freeze-existing ``load_architecture_build()`` boundary immediately before executable
    code is exposed, avoiding duplicate hash ownership in general metadata resolution.

    Missing siblings, malformed/invalid metadata, identity disagreement, or an invalid
    Task reference is an operation failure. Success returns exactly
    :class:`ArchitectureHandle` with non-``None`` canonical ``metadata_path`` and
    canonical ``implementation_path``.
    """

    ...


def resolve_train_protocol(
    train_protocol_id: TrainProtocolId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> TrainProtocolHandle:
    """Resolve one Train Protocol and its canonical YAML/Python sibling locations.

    The implementation derives only the Train Protocol paths exposed by ``layout`` for
    ``train_protocol_id``, requires both regular files through ``filesystem``,
    parses/normalizes the metadata, applies ``validate_train_protocol_metadata()``, and
    requires ``train_protocol_id == metadata.id``.

    ``metadata.task`` must resolve through :func:`resolve_task`. No Task handle is
    embedded, and no selected Corpus/Architecture compatibility or public-parameter
    resolution occurs here. A statically valid ``draft`` protocol may resolve.

    Resolution never imports the sibling Python file or exposes ``train``. Any sealed
    implementation SHA-256 requirement is verified by the freeze-existing
    ``load_train_entrypoint()`` boundary before executable code exposure rather than
    being duplicated here.

    Missing siblings, malformed/invalid metadata, identity disagreement, or an invalid
    Task reference is an operation failure. Success returns exactly
    :class:`TrainProtocolHandle` with non-``None`` canonical ``metadata_path`` and
    canonical ``implementation_path``.
    """

    ...


def resolve_model(
    model_id: ModelId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> ModelHandle:
    """Resolve one immutable Model through its completed Training Run lineage.

    The implementation must derive ``layout.model_metadata_path(model_id)``, require and
    parse that canonical YAML, apply ``validate_model_metadata()``, and require
    ``model_id == metadata.id``. The Model validator's deterministic v1 identity rule
    must therefore also agree with ``metadata.training_run``.

    It then derives ``layout.training_run_paths(metadata.training_run)`` and reads only
    that canonical ``run.yaml`` through ``filesystem``. The normalized Training Run must
    pass ``validate_training_run()``, its recorded ID must equal
    ``metadata.training_run``, its status must be ``COMPLETED``, and its canonical
    ``result.weights`` metadata must be present. No public TrainingRun resolver or
    TrainingRunHandle is introduced for this internal lineage step.

    ``training_run.architecture`` must resolve through :func:`resolve_architecture`.
    The concrete ``weights_path`` returned in the Model handle is exactly the canonical
    ``TrainingRunPaths.weights_path``. The persisted Run-relative weights path/format
    must agree with the v1 canonical ``artifacts/weights.pt`` contract, and that
    canonical weights file must exist; arbitrary persisted paths, traversal, fallback,
    or filename guessing are invalid.

    Resolution does not import Architecture Python, call ``build()``, ``torch.load``
    weights, or construct a module. The recorded weights SHA-256/byte-size check is not
    duplicated here: the freeze-existing ``load_model()`` contract explicitly owns that
    integrity check immediately before learned bytes are consumed. Likewise sealed
    Architecture implementation integrity remains owned by executable loading.

    Missing or invalid Model/Training Run metadata, identity disagreement, non-completed
    lineage, absent canonical result/weights, canonical path disagreement, missing
    weights bytes, or invalid Architecture resolution is an operation failure. Success
    returns the complete freeze-existing :class:`ModelHandle` with non-``None`` canonical
    Model/Training-Run metadata paths, canonical ``weights_path``, and the canonical
    resolver-produced Architecture handle.
    """

    ...


def resolve_evaluation_protocol(
    evaluation_protocol_id: EvaluationProtocolId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> EvaluationProtocolHandle:
    """Resolve one Evaluation Protocol and canonical YAML/Python sibling locations.

    The implementation derives only the Evaluation Protocol paths exposed by ``layout``
    for ``evaluation_protocol_id``, requires both regular files through ``filesystem``,
    parses/normalizes the YAML, applies ``validate_evaluation_protocol_metadata()``, and
    requires ``evaluation_protocol_id == metadata.id``.

    ``metadata.task`` must resolve through :func:`resolve_task`. The Task handle is not
    embedded, and no selected Corpus/Model compatibility, public-parameter resolution,
    result validation, or execution readiness is decided here. A statically valid
    ``draft`` protocol may resolve.

    Resolution never imports the sibling Python file or exposes ``evaluate``. Any sealed
    implementation SHA-256 requirement remains owned by the freeze-existing
    ``load_evaluation_entrypoint()`` boundary before executable code exposure.

    Missing siblings, malformed/invalid metadata, identity disagreement, or an invalid
    Task reference is an operation failure. Success returns exactly
    :class:`EvaluationProtocolHandle` with non-``None`` canonical ``metadata_path`` and
    canonical ``implementation_path``.
    """

    ...


def resolve_study(
    study_id: StudyId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> StudyHandle:
    """Resolve one Study plus only the upstream references declared by Study v1.

    The implementation must derive only ``layout.study_metadata_path(study_id)``, require
    and parse that canonical YAML through ``filesystem``, apply
    ``validate_study_metadata()``, and require ``study_id == metadata.id``. A statically
    valid ``draft`` Study may resolve; ``sealed`` is an execution-preflight concern.

    For a training Model source, resolution must require successful typed resolution of
    the declared training Corpus, Train Protocol, every selected Architecture, every
    evaluation-stage Corpus, and every evaluation-stage Evaluation Protocol. For an
    existing-Model source, it must require successful resolution of every selected
    Model plus every evaluation-stage Corpus and Evaluation Protocol. Those dependencies
    are checked through the entity-specific operations in this module and are not
    embedded into :class:`StudyHandle`.

    Resolution follows only those explicitly typed Study fields. It must not infer MLDB
    references from arbitrary strings such as stage names, parameter keys, descriptions,
    notes, or parameter values.

    Common Task agreement, executable sealed requirements, protocol parameter-key
    publication, default/override resolution, artifact/executable execution-readiness,
    existing-Model common-Task compatibility, grid expansion, Study Run allocation, and
    plan materialization remain Study preflight/materialization responsibilities rather
    than resolution responsibilities.

    Missing/invalid Study metadata, identity disagreement, or failure of any declared
    typed upstream reference is an operation failure. Success returns exactly
    :class:`StudyHandle`.
    """

    ...
