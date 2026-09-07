"""Public Python signatures for MLDB Evaluation launch preflight.

This skeleton fixes the boundary that turns one concrete Evaluation launch request
into the complete immutable input required before an Evaluation Run may be allocated.
It owns typed Model/Corpus/Evaluation Protocol resolution, Evaluation-specific
sealed-state and required static-integrity checks, common Task compatibility, the
resolved Task handle needed by ``EvaluationContext``, and complete public-parameter
resolution.

It intentionally does not allocate Evaluation Runs or directories, load executable
Python, load Model weights, invoke ``evaluate()``, construct ``EvaluationContext``,
validate or materialize returned results, commit artifacts, finalize Run lifecycle,
or participate in Queue/Worker orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.ids import CorpusId, EvaluationProtocolId, ModelId
from ..common.parameters import (
    PublicParameterOverrides,
    ResolvedPublicParameters,
    resolve_public_parameters,
)
from ..model.loading import ModelHandle
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.catalog_handles import CorpusHandle, TaskHandle
from ..runtime.definition_handles import EvaluationProtocolHandle
from ..runtime.resolution import (
    resolve_corpus,
    resolve_evaluation_protocol,
    resolve_model,
    resolve_task,
)
from .protocol import EvaluationProtocolStatus


@dataclass(frozen=True, slots=True)
class EvaluationPreflight:
    """Complete immutable Evaluation input prepared before Run allocation.

    ``task`` is the one resolved Task shared by the selected Model lineage, Corpus,
    and Evaluation Protocol. ``corpus`` and ``model`` are the exact resolved handles
    that later become ``EvaluationContext`` inputs. ``protocol`` is retained so the
    post-allocation executor can load exactly the preflighted Evaluation Protocol
    without re-resolving the caller's ID.

    ``parameters`` is the complete mapping produced by common public-parameter
    resolution against ``protocol.metadata.parameters``. It contains every published
    key exactly once and no unknown caller keys. Evaluation Run v1 has no universal
    seed, so this value exposes no seed field outside ordinary protocol parameters.

    ``work_dir`` is deliberately absent. It does not exist until after successful
    preflight and Evaluation Run allocation. Supplying this value plus the allocated
    Run's concrete work directory is sufficient for the later executor to construct
    one unambiguous ``EvaluationContext``.
    """

    task: TaskHandle
    corpus: CorpusHandle
    model: ModelHandle
    protocol: EvaluationProtocolHandle
    parameters: ResolvedPublicParameters


def preflight_evaluation(
    model_id: ModelId,
    corpus_id: CorpusId,
    evaluation_protocol_id: EvaluationProtocolId,
    parameter_overrides: PublicParameterOverrides,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> EvaluationPreflight:
    """Prepare one Evaluation launch request before any Evaluation Run allocation.

    The operation must resolve exactly the caller-selected Model, Corpus, and
    Evaluation Protocol through ``resolve_model()``, ``resolve_corpus()``, and
    ``resolve_evaluation_protocol()``. Resolution failures reject the launch request
    and produce no prepared value.

    A resolved Evaluation Protocol is launchable only when
    ``protocol.metadata.status is EvaluationProtocolStatus.SEALED`` and
    ``protocol.metadata.implementation.sha256`` is present. Before success, preflight
    must read the exact canonical sibling bytes through
    ``filesystem.read_bytes(protocol.implementation_path)`` and require their SHA-256
    to equal that recorded value. This is static integrity only: dynamic import,
    entrypoint existence/callability, and callable-shape validation remain owned by
    ``load_evaluation_entrypoint()`` after Run allocation, where the integrity guarantee
    is repeated immediately before code exposure.

    Model's effective Task is derived only through the freeze-existing Model lineage:
    ``model.architecture.metadata.task``. ``resolve_model()`` has already established
    that ``model.training_run.architecture`` resolves to ``model.architecture``; the
    Model record itself therefore needs and must gain no duplicated Task field.

    Compatibility succeeds only when that Model-lineage Task ID, ``corpus.metadata.task``,
    and ``protocol.metadata.task`` are exactly equal. After agreement, the common Task
    ID must resolve through ``resolve_task()`` and the resulting ``TaskHandle`` becomes
    ``EvaluationPreflight.task``. No model-family, tensor-interface, metric, split, or
    protocol-specific semantic compatibility rule is invented by this generic boundary.

    ``resolve_corpus()`` remains the owner of immutable Corpus artifact integrity.
    Preflight must not reopen SQLite or duplicate Corpus hashing.

    ``resolve_model()`` establishes the canonical completed lineage, canonical
    ``model.weights_path`` and its existence, and the resolved lineage Architecture;
    it does not verify the recorded learned-weight hash/size. Before success, preflight
    must read the exact bytes at ``model.weights_path`` and require both
    ``model.training_run.result.weights.sha256`` SHA-256 agreement and
    ``model.training_run.result.weights.bytes`` exact byte-count agreement. It must not
    call ``torch.load`` or inspect the state mapping/content.

    The learned Model can only be loaded after constructing the Architecture recorded
    by its completed Training Run. Therefore the lineage
    ``model.architecture`` is a required executable asset for this Evaluation launch.
    Before success, preflight must require
    ``model.architecture.metadata.status is ArchitectureStatus.SEALED``, require its
    ``implementation.sha256`` metadata, and compare that hash with the exact canonical
    bytes read through
    ``filesystem.read_bytes(model.architecture.implementation_path)``. This requirement
    follows Model learned-state resolution plus the Evaluation pre-allocation static-
    integrity contract; it is not added merely for symmetry. Architecture import,
    ``build()``, ``torch.load``, state-mapping validation, and strict
    ``load_state_dict`` remain post-allocation ``load_model()`` work.

    Hash calculation for protocol, Architecture, and learned-weight bytes is an
    implementation-private detail. No hash/integrity abstraction is introduced.

    Once compatibility succeeds, the operation applies ``resolve_public_parameters()``
    to ``protocol.metadata.parameters`` and ``parameter_overrides``. Unknown caller
    keys, invalid public values, invalid defaults, or any other common parameter
    resolution failure reject the launch and yield no partial mapping. Success stores
    the complete ``ResolvedPublicParameters`` in the prepared value; caller overrides
    alone are never persisted or passed onward as the resolved parameter surface.

    Successful return is the only point after which an Evaluation Run may be allocated.
    Run ID allocation, Run-directory/work-directory creation, initial ``running``
    persistence, executable loading, ``EvaluationContext`` construction, Model loading,
    ``evaluate()`` invocation, result acceptance/materialization, artifact commit, and
    terminal status handling all occur after this boundary. Any failure described above
    occurs before allocation and therefore creates no Evaluation Run and consumes no
    Evaluation Run identity.

    This skeleton fixes neither a feature-specific exception hierarchy nor a generic
    preflight base/result abstraction. Failures are operation failures; success returns
    exactly one :class:`EvaluationPreflight`.
    """

    ...
