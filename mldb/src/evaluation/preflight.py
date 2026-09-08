"""Public Python signatures for MLDB Evaluation launch preflight.

Concrete implementation of the frozen pre-allocation Evaluation boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..catalog.architecture import ArchitectureStatus
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
    """Complete immutable Evaluation input prepared before Run allocation."""

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
    """Prepare one Evaluation launch request before any Evaluation Run allocation."""

    model = resolve_model(model_id, layout, filesystem)
    corpus = resolve_corpus(corpus_id, layout, filesystem)
    protocol = resolve_evaluation_protocol(
        evaluation_protocol_id,
        layout,
        filesystem,
    )

    if protocol.metadata.status is not EvaluationProtocolStatus.SEALED:
        raise ValueError("Evaluation Protocol must be sealed before launch.")
    protocol_sha = protocol.metadata.implementation.sha256
    if protocol_sha is None:
        raise ValueError("Sealed Evaluation Protocol requires implementation SHA-256.")
    protocol_bytes = filesystem.read_bytes(protocol.implementation_path)
    if hashlib.sha256(protocol_bytes).hexdigest() != protocol_sha.lower():
        raise ValueError("Evaluation Protocol implementation SHA-256 mismatch.")

    architecture = model.architecture
    if architecture.metadata.status is not ArchitectureStatus.SEALED:
        raise ValueError("Model Architecture must be sealed before Evaluation launch.")
    architecture_sha = architecture.metadata.implementation.sha256
    if architecture_sha is None:
        raise ValueError("Sealed Architecture requires implementation SHA-256.")
    architecture_bytes = filesystem.read_bytes(architecture.implementation_path)
    if hashlib.sha256(architecture_bytes).hexdigest() != architecture_sha.lower():
        raise ValueError("Architecture implementation SHA-256 mismatch.")

    weights = model.training_run.result
    if weights is None:
        raise ValueError("Resolved Model lineage must contain completed weights metadata.")
    weight_bytes = filesystem.read_bytes(model.weights_path)
    expected_weights = weights.weights
    if hashlib.sha256(weight_bytes).hexdigest() != expected_weights.sha256.lower():
        raise ValueError("Model weights SHA-256 mismatch.")
    if len(weight_bytes) != expected_weights.bytes:
        raise ValueError("Model weights byte count mismatch.")

    task_id = architecture.metadata.task
    if corpus.metadata.task != task_id or protocol.metadata.task != task_id:
        raise ValueError("Model, Corpus, and Evaluation Protocol Task IDs must match exactly.")
    task = resolve_task(task_id, layout, filesystem)

    parameters = resolve_public_parameters(
        protocol.metadata.parameters,
        parameter_overrides,
    )
    return EvaluationPreflight(
        task=task,
        corpus=corpus,
        model=model,
        protocol=protocol,
        parameters=parameters,
    )
