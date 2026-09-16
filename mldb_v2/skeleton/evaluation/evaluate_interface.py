"""MLDB v2 common Evaluation callable boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol

from mldb_v2.skeleton.common.parameters import ResolvedPublicParameters
from mldb_v2.skeleton.common.telemetry import TelemetryReporter

if TYPE_CHECKING:
    import torch
    from mldb_v2.skeleton.catalog.architecture import Architecture
    from mldb_v2.skeleton.catalog.corpus import Corpus
    from mldb_v2.skeleton.catalog.task import Task
    from mldb_v2.skeleton.training.model import Model
    from mldb_v2.skeleton.training.training_result import TrainingResult


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
    telemetry: TelemetryReporter
    work_dir: Path


@dataclass(frozen=True)
class EvaluationCandidate:
    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, Path]


class EvaluationCallable(Protocol):
    def __call__(self, context: EvaluationContext) -> EvaluationCandidate:
        ...
