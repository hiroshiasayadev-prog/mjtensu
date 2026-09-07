"""Public Python signatures for MLDB Training launch preflight.

This skeleton fixes the boundary that turns one concrete Training launch request into
the complete immutable input required before a Training Run may be allocated. It owns
typed Corpus/Architecture/Train Protocol resolution, Training-specific sealed-state
and static-integrity requirements, common Task compatibility, the resolved Task handle
needed by ``TrainContext``, concrete seed validation, and complete public-parameter
resolution.

It intentionally does not allocate Training Runs or directories, load executable
Python, invoke Architecture ``build()`` or Train Protocol ``train()``, construct
``TrainContext``, validate or serialize learned weights, create Models, finalize Run
lifecycle, or participate in Queue/Worker orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..catalog.architecture import ArchitectureStatus
from ..common.ids import ArchitectureId, CorpusId, TrainProtocolId
from ..common.parameters import (
    PublicParameterOverrides,
    ResolvedPublicParameters,
    resolve_public_parameters,
)
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.catalog_handles import ArchitectureHandle, CorpusHandle, TaskHandle
from ..runtime.definition_handles import TrainProtocolHandle
from ..runtime.resolution import (
    resolve_architecture,
    resolve_corpus,
    resolve_task,
    resolve_train_protocol,
)
from .protocol import TrainProtocolStatus


@dataclass(frozen=True, slots=True)
class TrainingPreflight:
    """Complete immutable Training input prepared before Run allocation.

    ``task`` is the one resolved Task shared by the selected Corpus, Architecture,
    and Train Protocol. ``corpus`` and ``architecture`` are the exact resolved handles
    that later become ``TrainContext`` inputs. ``protocol`` is retained so the
    post-allocation executor can load exactly the preflighted Train Protocol without
    re-resolving the caller's ID.

    ``seed`` is the validated concrete Training Run seed. It is an integer, boolean is
    invalid, no common numeric range is imposed, and the exact value is preserved
    without coercion. It remains separate from ``parameters``.

    ``parameters`` is the complete mapping produced by common public-parameter
    resolution against ``protocol.metadata.parameters``. It contains every published
    key exactly once and no unknown caller keys.

    ``work_dir`` is deliberately absent. It does not exist until after successful
    preflight and Training Run allocation. Supplying this value plus the allocated
    Run's concrete work directory is sufficient for the later executor to construct
    one unambiguous ``TrainContext``.
    """

    task: TaskHandle
    corpus: CorpusHandle
    architecture: ArchitectureHandle
    protocol: TrainProtocolHandle
    seed: int
    parameters: ResolvedPublicParameters


def preflight_training(
    corpus_id: CorpusId,
    architecture_id: ArchitectureId,
    train_protocol_id: TrainProtocolId,
    seed: int,
    parameter_overrides: PublicParameterOverrides,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> TrainingPreflight:
    """Prepare one Training launch request before any Training Run allocation.

    The operation must resolve exactly the caller-selected Corpus, Architecture, and
    Train Protocol through ``resolve_corpus()``, ``resolve_architecture()``, and
    ``resolve_train_protocol()``. Resolution failures reject the launch request and
    produce no prepared value. The operation accepts typed IDs rather than pre-resolved
    handles so launch tooling has one public boundary that owns required resolution and
    returns the exact handles the post-allocation executor must reuse.

    A resolved Architecture is launchable only when
    ``architecture.metadata.status is ArchitectureStatus.SEALED`` and its
    ``architecture.metadata.implementation.sha256`` is present. Before success,
    preflight must read the exact canonical sibling bytes through
    ``filesystem.read_bytes(architecture.implementation_path)`` and require their
    SHA-256 to equal that recorded value.

    A resolved Train Protocol is launchable only when
    ``protocol.metadata.status is TrainProtocolStatus.SEALED`` and its
    ``protocol.metadata.implementation.sha256`` is present. Before success, preflight
    must likewise read ``filesystem.read_bytes(protocol.implementation_path)`` and
    require exact SHA-256 agreement. Hash calculation is an implementation-private
    detail; this skeleton introduces no hash/integrity service or public abstraction.

    These pre-allocation byte checks do not load executable Python. The frozen
    ``load_architecture_build()`` and ``load_train_entrypoint()`` boundaries still
    repeat the applicable integrity guarantee immediately before code exposure and then
    own dynamic import, entrypoint existence/callability, and callable-shape validation.

    Compatibility succeeds only when ``corpus.metadata.task``,
    ``architecture.metadata.task``, and ``protocol.metadata.task`` are exactly equal.
    After agreement, that common Task ID must resolve through ``resolve_task()`` and the
    resulting ``TaskHandle`` becomes ``TrainingPreflight.task``. No additional model-
    family, tensor-interface, Corpus-representation, loss, or protocol-specific semantic
    compatibility rule is invented by this generic boundary because current frozen
    contracts do not define one common rule beyond Task agreement.

    ``resolve_corpus()`` remains the owner of immutable Corpus artifact integrity.
    Successful Corpus resolution has already accepted the canonical SQLite artifact's
    recorded SHA-256 and optional byte-size contract, so preflight must not reopen
    SQLite or duplicate Corpus hashing. Architecture and Train Protocol executable
    *static byte integrity* is nevertheless required here by the Run-allocation
    boundary described above. Dynamic import, callable loading/shape validation,
    ``build()``, and ``train()`` remain post-allocation work.

    ``seed`` must satisfy ``isinstance(seed, int)`` while not being an instance of
    ``bool``. A floating-point value, string, boolean, or other object is invalid. The
    implementation imposes no universal numeric range and performs no coercion. The
    accepted integer is copied unchanged into ``TrainingPreflight.seed`` and must never
    be inserted into the public-parameter mapping.

    Once the seed is valid, the operation applies ``resolve_public_parameters()`` to
    ``protocol.metadata.parameters`` and ``parameter_overrides``. Unknown caller keys,
    invalid public values, invalid defaults, or any other common parameter-resolution
    failure reject the launch and yield no partial mapping. Success stores the complete
    ``ResolvedPublicParameters`` returned by that frozen operation; caller overrides
    alone are never persisted or passed onward as the resolved parameter surface.

    Successful return is the only point after which a Training Run may be allocated.
    Run ID allocation, Run-directory/work-directory creation, ``started_at`` capture,
    initial ``running`` persistence, executable loading, ``TrainContext`` construction,
    Architecture construction, ``train()`` invocation, learned-state validation,
    canonical weights serialization, Model creation, and terminal status handling all
    occur after this boundary. Any failure described above occurs before allocation and
    therefore creates no Training Run and consumes no Training Run identity.

    This skeleton fixes neither a feature-specific exception hierarchy nor a generic
    preflight base/result/request abstraction. Failures are operation failures; success
    returns exactly one :class:`TrainingPreflight`.
    """

    ...
