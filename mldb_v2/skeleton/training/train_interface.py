"""MLDB v2 common Train callable boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from mldb_v2.skeleton.common.parameters import ResolvedPublicParameters
from mldb_v2.skeleton.common.telemetry import TelemetryReporter

if TYPE_CHECKING:
    import torch
    from mldb_v2.skeleton.catalog.architecture import Architecture
    from mldb_v2.skeleton.catalog.corpus import Corpus
    from mldb_v2.skeleton.catalog.task import Task


@dataclass(frozen=True)
class MaterializedCorpus:
    definition: Corpus
    root: Path


@dataclass(frozen=True)
class TrainContext:
    task: Task
    corpus: MaterializedCorpus
    architecture: Architecture
    model: torch.nn.Module
    seed: int
    parameters: ResolvedPublicParameters
    telemetry: TelemetryReporter
    work_dir: Path


class TrainCallable(Protocol):
    def __call__(self, context: TrainContext) -> torch.nn.Module:
        ...
