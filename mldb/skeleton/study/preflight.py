"""Public Python boundary for MLDB Study execution preflight.

This module fixes the operation that accepts one already-resolved Study and proves that
its complete authored execution intent is legal before any Study Run is allocated.
Successful preflight retains only the small resolved handle set needed by downstream
Study launch and frozen plan materialization.

Preflight owns execution-time sealed requirements, required static integrity, exact
cross-asset Task agreement, Study-to-Protocol public-parameter key publication, and
resolution of the dependencies needed to establish those facts. It intentionally does
not expand a training grid, assign trial IDs, resolve parameter defaults, create a
StudyPlan, allocate a StudyRun or child Run, import executable asset implementations,
run asset pytest, serialize/hash a plan, or perform Queue/Worker orchestration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..catalog.architecture import ArchitectureStatus
from ..common.ids import EvaluationProtocolId
from ..evaluation.protocol import EvaluationProtocolStatus
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.catalog_handles import TaskHandle
from ..runtime.definition_handles import (
    EvaluationProtocolHandle,
    StudyHandle,
    TrainProtocolHandle,
)
from ..runtime.resolution import (
    resolve_architecture,
    resolve_corpus,
    resolve_evaluation_protocol,
    resolve_model,
    resolve_task,
    resolve_train_protocol,
)
from ..training.protocol import TrainProtocolStatus
from .definition import (
    StudyExistingModelSource,
    StudyStatus,
    StudyTrainingModelSource,
)


@dataclass(frozen=True, slots=True)
class PreparedStudyExecution:
    """Resolved Study dependencies retained after successful execution preflight.

    ``study`` is the exact resolver-produced Study handle supplied to preflight.
    ``task`` is the one Task shared by every Model-source and evaluation dependency.

    ``train_protocol`` is present only for a training-source Study and is the exact
    sealed Train Protocol referenced by ``study.metadata.model.protocol``. Existing-
    Model Studies retain ``None`` because they imply no new Training Run or Model.

    ``evaluation_protocols`` contains exactly one resolved sealed handle for every
    Evaluation Protocol ID referenced by the Study's evaluation stages. A protocol
    referenced by multiple stages appears once. The mapping is a retained dependency
    for :func:`mldb.skeleton.study.expansion.materialize_study_plan`; it is not a
    general dependency graph or repository cache. Concrete implementations must return
    a snapshot that cannot be mutated through caller-owned mapping state.

    Corpus, Architecture, and existing-Model handles are deliberately not retained.
    They are required to establish preflight legality but are not inputs to frozen Study
    plan materialization. Later concrete child Training/Evaluation launch boundaries
    resolve the assets required by those concrete child executions rather than treating
    this value as a transitive dependency cache.
    """

    study: StudyHandle
    task: TaskHandle
    train_protocol: TrainProtocolHandle | None
    evaluation_protocols: Mapping[EvaluationProtocolId, EvaluationProtocolHandle]


def preflight_study_execution(
    study: StudyHandle,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> PreparedStudyExecution:
    """Validate one resolved sealed Study completely enough to allocate a Study Run.

    ``study`` must be a handle produced by the frozen typed Study resolver. Its handle
    contract therefore already establishes canonical Study identity, Study-format
    metadata validation, and successful typed resolution of every direct Study asset
    reference. This operation does not accept raw :class:`Study`, a repository wrapper,
    a dependency graph, or a pre-resolved caller-supplied collection of child assets.

    The implementation re-resolves the Study's exact direct dependency references
    through the frozen entity-specific resolver operations because ``StudyHandle``
    intentionally does not embed them. These resolved handles are used to establish the
    execution-preflight facts below. After success, protocol handles needed by frozen
    plan materialization are retained in :class:`PreparedStudyExecution`, so the Study
    launcher does not re-resolve those protocol assets before calling
    ``materialize_study_plan()``.

    Common requirements are:

    - ``study.metadata.status`` is exactly :attr:`StudyStatus.SEALED`;
    - every declared evaluation Corpus resolves successfully, reusing
      ``resolve_corpus()``'s immutable Corpus integrity guarantee;
    - every declared Evaluation Protocol resolves successfully and is exactly
      :attr:`EvaluationProtocolStatus.SEALED`;
    - every selected Evaluation Protocol has implementation SHA-256 metadata and its
      exact canonical ``implementation_path`` bytes read through
      ``filesystem.read_bytes(...)`` match that SHA-256;
    - every evaluation Corpus and Evaluation Protocol has exactly the one common Task
      selected by the Study's Model source;
    - each fixed ``StudyEvaluationStage.parameters`` key exists verbatim in the exact
      selected Evaluation Protocol ``parameters`` mapping;
    - Study-authored public-parameter values and protocol defaults are already valid in
      the shared JSON-compatible domain by the typed Study/Protocol resolver guarantees.

    For :class:`StudyTrainingModelSource`, the implementation additionally requires:

    - the training Corpus resolves successfully, reusing ``resolve_corpus()``'s
      immutable Corpus integrity guarantee;
    - the selected Train Protocol resolves successfully and is exactly
      :attr:`TrainProtocolStatus.SEALED`, has implementation SHA-256 metadata, and its
      exact canonical ``implementation_path`` bytes match that SHA-256;
    - every selected Architecture resolves successfully and is exactly
      :attr:`ArchitectureStatus.SEALED`, has implementation SHA-256 metadata, and its
      exact canonical ``implementation_path`` bytes match that SHA-256;
    - training Corpus, Train Protocol, every selected Architecture, every evaluation
      Corpus, and every Evaluation Protocol all record the same ``TaskId``;
    - every key in ``study.metadata.model.parameters`` exists verbatim in the exact
      selected Train Protocol ``parameters`` mapping.

    Study-local grid validity such as non-empty/unique Architectures, parameter-axis
    values, and integer non-boolean unique seeds is not recomputed here: it is already
    guaranteed by the validated ``StudyHandle``. Preflight must not expand Cartesian
    coordinates merely to prove publication. Omitted Train/Evaluation Protocol defaults
    likewise remain unresolved until frozen Study plan materialization applies
    ``resolve_public_parameters()``.

    For :class:`StudyExistingModelSource`, the implementation instead requires every
    selected Model to resolve successfully through the frozen Model resolver. That
    resolver establishes the valid immutable Model record, completed Training Run
    lineage, canonical learned-weight metadata/path contract, canonical weights-file
    existence, and resolved lineage Architecture. Before preflight succeeds, it must
    additionally read the exact canonical ``model_handle.weights_path`` bytes and
    require both SHA-256 and byte-count agreement with
    ``model_handle.training_run.result.weights``.

    Existing-Model execution also depends on the lineage Architecture in order to load
    learned state. Each ``model_handle.architecture`` must therefore be exactly
    :attr:`ArchitectureStatus.SEALED`, carry implementation SHA-256 metadata, and have
    exact canonical implementation bytes matching that SHA-256 before Study Run
    allocation. The original Train Protocol from the completed Training Run is not a
    Model-loading dependency and is not re-resolved or revalidated in this mode.

    The Model Task for Study compatibility is exactly
    ``model_handle.architecture.metadata.task``; Task is not added to Model metadata.
    All selected Model Tasks must agree with each other and with every evaluation Corpus
    and Evaluation Protocol. This mode creates no Training Run and implies no new Model.

    After one common ``TaskId`` has been established, the implementation resolves that
    exact Task through :func:`resolve_task` and returns its :class:`TaskHandle`. The
    resulting prepared value retains no redundant Corpus/Architecture/Model collection.

    Static integrity is part of this pre-allocation boundary. Successful Corpus
    resolution has already verified immutable Corpus SQLite bytes, while this operation
    explicitly verifies the required sealed executable implementation bytes and, for
    existing Models, canonical learned-weight SHA-256/byte-size agreement using the
    resolver-produced handle paths. Hash calculation is an implementation-private
    detail and introduces no second public hash/integrity abstraction.

    This static check does not import sibling Python modules, expose or validate
    executable callables, invoke entrypoints, run asset-specific pytest, call
    ``torch.load``, inspect learned-state mappings, construct Architectures, or apply
    ``load_state_dict``. The frozen executable loaders and ``load_model()`` may repeat
    the same recorded-hash checks after child Run allocation immediately before actual
    code/artifact consumption, covering mutation between preflight and use.

    Any resolution failure, draft Study/executable definition, required static-integrity
    mismatch, learned-weight byte/hash/size mismatch, Task disagreement, or unpublished
    Study parameter key is an operation failure and returns no prepared value. Concrete
    exception/report aggregation is implementation-owned; this skeleton
    does not add a Study-specific error hierarchy or generic preflight result wrapper.
    Failure occurs before Study Run ID allocation, plan materialization, child Run
    allocation, executable invocation, or Queue admission.

    On success, downstream Study launch may allocate its Study Run and then call exactly::

        materialize_study_plan(
            prepared.study.metadata,
            prepared.train_protocol.metadata
            if prepared.train_protocol is not None
            else None,
            {
                protocol_id: handle.metadata
                for protocol_id, handle in prepared.evaluation_protocols.items()
            },
        )

    No repository re-resolution is required for those frozen materialization inputs.
    Study Run allocation, restartable plan persistence, child execution, retry,
    reconciliation, and Queue scheduling remain outside this operation.
    """

    ...
