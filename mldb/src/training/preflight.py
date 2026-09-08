"""Training launch preflight implementation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

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
    """Complete immutable Training input prepared before Run allocation."""

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
    """Prepare one Training launch request before any Training Run allocation."""

    corpus = resolve_corpus(corpus_id, layout, filesystem)
    architecture = resolve_architecture(architecture_id, layout, filesystem)
    protocol = resolve_train_protocol(train_protocol_id, layout, filesystem)

    if architecture.metadata.status is not ArchitectureStatus.SEALED:
        raise ValueError("Architecture must be sealed before Training launch.")
    architecture_sha = architecture.metadata.implementation.sha256
    if architecture_sha is None:
        raise ValueError("Sealed Architecture requires implementation SHA-256.")
    architecture_bytes = filesystem.read_bytes(architecture.implementation_path)
    if hashlib.sha256(architecture_bytes).hexdigest() != architecture_sha.lower():
        raise ValueError("Architecture implementation SHA-256 mismatch.")

    if protocol.metadata.status is not TrainProtocolStatus.SEALED:
        raise ValueError("Train Protocol must be sealed before Training launch.")
    protocol_sha = protocol.metadata.implementation.sha256
    if protocol_sha is None:
        raise ValueError("Sealed Train Protocol requires implementation SHA-256.")
    protocol_bytes = filesystem.read_bytes(protocol.implementation_path)
    if hashlib.sha256(protocol_bytes).hexdigest() != protocol_sha.lower():
        raise ValueError("Train Protocol implementation SHA-256 mismatch.")

    task_id = corpus.metadata.task
    if architecture.metadata.task != task_id or protocol.metadata.task != task_id:
        raise ValueError(
            "Corpus, Architecture, and Train Protocol Task IDs must match exactly."
        )
    task = resolve_task(task_id, layout, filesystem)

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Training seed must be an integer and must not be boolean.")

    parameters = resolve_public_parameters(
        protocol.metadata.parameters,
        parameter_overrides,
    )
    return TrainingPreflight(
        task=task,
        corpus=corpus,
        architecture=architecture,
        protocol=protocol,
        seed=seed,
        parameters=parameters,
    )
