"""Study execution preflight implementation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from types import MappingProxyType

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
    """Resolved Study dependencies retained after successful execution preflight."""

    study: StudyHandle
    task: TaskHandle
    train_protocol: TrainProtocolHandle | None
    evaluation_protocols: Mapping[
        EvaluationProtocolId,
        EvaluationProtocolHandle,
    ]


def preflight_study_execution(
    study: StudyHandle,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> PreparedStudyExecution:
    """Validate one resolved sealed Study completely enough to allocate a Study Run."""

    if not isinstance(study, StudyHandle):
        raise TypeError("study must be a resolver-produced StudyHandle")
    canonical_path = layout.study_metadata_path(study.metadata.id)
    if study.metadata_path != canonical_path:
        raise ValueError("StudyHandle metadata path is not canonical")
    if study.metadata.status is not StudyStatus.SEALED:
        raise ValueError("Study must be sealed before execution")

    model_source = study.metadata.model
    train_protocol: TrainProtocolHandle | None = None
    common_task_id = None

    if isinstance(model_source, StudyTrainingModelSource):
        training_corpus = resolve_corpus(
            model_source.corpus,
            layout,
            filesystem,
        )
        common_task_id = training_corpus.metadata.task

        train_protocol = resolve_train_protocol(
            model_source.protocol,
            layout,
            filesystem,
        )
        _verify_executable(
            train_protocol,
            TrainProtocolStatus.SEALED,
            "Train Protocol",
            filesystem,
        )
        _require_task_match(
            common_task_id,
            train_protocol.metadata.task,
            "Train Protocol",
        )
        _require_published_keys(
            model_source.parameters,
            train_protocol.metadata.parameters,
            "training",
        )

        for architecture_id in model_source.architectures:
            architecture = resolve_architecture(
                architecture_id,
                layout,
                filesystem,
            )
            _verify_executable(
                architecture,
                ArchitectureStatus.SEALED,
                "Architecture",
                filesystem,
            )
            _require_task_match(
                common_task_id,
                architecture.metadata.task,
                "Architecture",
            )

    elif isinstance(model_source, StudyExistingModelSource):
        for model_id in model_source.models:
            model = resolve_model(model_id, layout, filesystem)
            architecture = model.architecture
            _verify_executable(
                architecture,
                ArchitectureStatus.SEALED,
                "Model lineage Architecture",
                filesystem,
            )
            weights_result = model.training_run.result
            if weights_result is None:
                raise ValueError(
                    "Resolved Model lineage must contain completed weights metadata"
                )
            weights = weights_result.weights
            weight_bytes = filesystem.read_bytes(model.weights_path)
            if hashlib.sha256(weight_bytes).hexdigest() != weights.sha256.lower():
                raise ValueError("Model weights SHA-256 mismatch")
            if len(weight_bytes) != weights.bytes:
                raise ValueError("Model weights byte count mismatch")

            model_task_id = architecture.metadata.task
            if common_task_id is None:
                common_task_id = model_task_id
            else:
                _require_task_match(
                    common_task_id,
                    model_task_id,
                    "Model",
                )
    else:
        raise ValueError("unsupported Study Model source")

    if common_task_id is None:
        raise ValueError("Study Model source did not establish a Task")
    evaluation_protocols: dict[
        EvaluationProtocolId,
        EvaluationProtocolHandle,
    ] = {}
    for stage in study.metadata.evaluations:
        corpus = resolve_corpus(stage.corpus, layout, filesystem)
        _require_task_match(
            common_task_id,
            corpus.metadata.task,
            "Evaluation Corpus",
        )

        protocol = evaluation_protocols.get(stage.protocol)
        if protocol is None:
            protocol = resolve_evaluation_protocol(
                stage.protocol,
                layout,
                filesystem,
            )
            _verify_executable(
                protocol,
                EvaluationProtocolStatus.SEALED,
                "Evaluation Protocol",
                filesystem,
            )
            evaluation_protocols[stage.protocol] = protocol

        _require_task_match(
            common_task_id,
            protocol.metadata.task,
            "Evaluation Protocol",
        )
        _require_published_keys(
            stage.parameters,
            protocol.metadata.parameters,
            f"evaluation stage {stage.stage!r}",
        )
    task = resolve_task(common_task_id, layout, filesystem)
    return PreparedStudyExecution(
        study=study,
        task=task,
        train_protocol=train_protocol,
        evaluation_protocols=MappingProxyType(dict(evaluation_protocols)),
    )


def _verify_executable(
    handle: object,
    sealed_status: object,
    kind: str,
    filesystem: FilesystemPort,
) -> None:
    metadata = handle.metadata
    if metadata.status is not sealed_status:
        raise ValueError(f"{kind} must be sealed before Study execution")
    expected_sha = metadata.implementation.sha256
    if expected_sha is None:
        raise ValueError(f"Sealed {kind} requires implementation SHA-256")
    implementation_bytes = filesystem.read_bytes(handle.implementation_path)
    if hashlib.sha256(implementation_bytes).hexdigest() != expected_sha.lower():
        raise ValueError(f"{kind} implementation SHA-256 mismatch")


def _require_task_match(
    expected: object,
    actual: object,
    kind: str,
) -> None:
    if actual != expected:
        raise ValueError(f"{kind} Task ID must match the Study Task exactly")


def _require_published_keys(
    supplied: Mapping[str, object],
    published: Mapping[str, object],
    context: str,
) -> None:
    for key in supplied:
        if key not in published:
            raise ValueError(
                f"{context} parameter {key!r} is not a published public parameter"
            )
