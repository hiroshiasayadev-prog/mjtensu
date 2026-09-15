"""MLDB v2 common Train callable boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from mldb_v2.src.catalog._executable_definition_loading import _load_executable_callable
from mldb_v2.src.common.ids import EntityKind, TrainProtocolId
from mldb_v2.src.common.parameters import ResolvedPublicParameters
from mldb_v2.src.common.telemetry import TelemetryReporter
from mldb_v2.src.training.train_protocol import _load_train_protocol_definition

if TYPE_CHECKING:
    import torch
    from mldb_v2.src.catalog.architecture import Architecture
    from mldb_v2.src.catalog.corpus import Corpus
    from mldb_v2.src.catalog.task import Task


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


def _load_train_callable(
    mldb_data_root: str | Path, protocol_id: TrainProtocolId | str
) -> TrainCallable:
    _load_train_protocol_definition(mldb_data_root, protocol_id)
    callable_value = _load_executable_callable(
        mldb_data_root,
        kind=EntityKind.TRAIN_PROTOCOL,
        entity_id=str(protocol_id),
        entrypoint="train",
        arity=1,
    )
    return cast(TrainCallable, callable_value)
