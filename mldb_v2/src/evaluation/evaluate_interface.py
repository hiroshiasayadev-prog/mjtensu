"""MLDB v2 common Evaluation callable boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol, cast

from mldb_v2.src.catalog._executable_definition_loading import _load_executable_callable
from mldb_v2.src.common.ids import EntityKind, EvaluationProtocolId
from mldb_v2.src.common.parameters import ResolvedPublicParameters
from mldb_v2.src.evaluation.evaluation_protocol import _load_evaluation_protocol_definition

if TYPE_CHECKING:
    import torch
    from mldb_v2.src.catalog.architecture import Architecture
    from mldb_v2.src.catalog.corpus import Corpus
    from mldb_v2.src.catalog.task import Task
    from mldb_v2.src.training.model import Model
    from mldb_v2.src.training.training_result import TrainingResult


@dataclass(frozen=True)
class MaterializedCorpus:
    definition: Corpus
    root: Path


@dataclass(frozen=True)
class LoadedModel:
    definition: Model
    training_result: TrainingResult
    architecture: Architecture
    module: torch.nn.Module


@dataclass(frozen=True)
class EvaluationContext:
    task: Task
    corpus: MaterializedCorpus
    model: LoadedModel
    parameters: ResolvedPublicParameters
    work_dir: Path


@dataclass(frozen=True)
class EvaluationCandidate:
    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, Path]


class EvaluationCallable(Protocol):
    def __call__(self, context: EvaluationContext) -> EvaluationCandidate:
        ...


def _load_evaluation_callable(
    mldb_data_root: str | Path, protocol_id: EvaluationProtocolId | str
) -> EvaluationCallable:
    _load_evaluation_protocol_definition(mldb_data_root, protocol_id)
    callable_value = _load_executable_callable(
        mldb_data_root,
        kind=EntityKind.EVALUATION_PROTOCOL,
        entity_id=str(protocol_id),
        entrypoint="evaluate",
        arity=1,
    )
    return cast(EvaluationCallable, callable_value)
