"""Immutable Model identity and canonical runtime lineage loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from mldb_v2.src.catalog.architecture import Architecture, _load_architecture_definition
from mldb_v2.src.catalog.task import Task, _load_task
from mldb_v2.src.common.ids import (
    EntityKind,
    ModelId,
    TaskId,
    TrainingResultId,
    _validate_typed_reference,
)
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.training.canonical_weights import (
    CanonicalWeightsArtifactRef,
    _load_canonical_state_dict_bytes,
    _load_state_into_fresh_architecture,
)
from mldb_v2.src.training.training_result import TrainingResult, _load_training_result

if TYPE_CHECKING:
    import torch.nn


class Model(TypedDict):
    schema: Literal["mjtensu.mldb-v2/model/v1"]
    id: ModelId
    training_result: TrainingResultId


_SCHEMA = "mjtensu.mldb-v2/model/v1"
_FIELDS = {"schema", "id", "training_result"}


@dataclass(frozen=True)
class _ResolvedModelLineage:
    model: Model
    training_result: TrainingResult
    architecture: Architecture
    task: Task
    weights: CanonicalWeightsArtifactRef


@dataclass(frozen=True)
class _LoadedModelRuntime:
    definition: Model
    training_result: TrainingResult
    architecture: Architecture
    module: "torch.nn.Module"


def _validate_model(value: object, *, expected_id: str | None = None) -> Model:
    if type(value) is not dict or set(value) != _FIELDS:
        raise ValueError("Model fields do not match schema")
    if value["schema"] != _SCHEMA:
        raise ValueError("unsupported Model schema")
    model_id = _validate_typed_reference(value["id"])
    if expected_id is not None and model_id != expected_id:
        raise ValueError("Model id does not match canonical path identity")
    training_result = _validate_typed_reference(value["training_result"])
    return {
        "schema": _SCHEMA,
        "id": ModelId(model_id),
        "training_result": TrainingResultId(training_result),
    }


def _load_model(
    resolver: CanonicalRepositoryResolver,
    model_id: ModelId | str,
) -> Model:
    expected_id = _validate_typed_reference(model_id)
    document = resolver.resolve(kind=EntityKind.MODEL, entity_id=ModelId(expected_id))
    return _validate_model(document, expected_id=expected_id)


def _resolve_model_lineage(
    mldb_data_root: str | Path,
    model_id: ModelId | str,
) -> _ResolvedModelLineage:
    resolver = CanonicalRepositoryResolver(mldb_data_root)
    model = _load_model(resolver, model_id)
    training_result = _load_training_result(resolver, model["training_result"])
    if training_result["status"] != "completed":
        raise ValueError("Model requires a completed TrainingResult")
    if training_result["diagnostic"] is not None or training_result["result"] is None:
        raise ValueError("completed TrainingResult payload is inconsistent")
    if training_result["result"]["model"] != model["id"]:
        raise ValueError("TrainingResult result.model does not match Model id")

    architecture = _load_architecture_definition(
        mldb_data_root,
        training_result["architecture"],
    )
    task = _load_task(resolver, TaskId(training_result["task"]))
    if architecture["task"] != training_result["task"]:
        raise ValueError("Architecture task does not match TrainingResult task")

    return _ResolvedModelLineage(
        model=model,
        training_result=training_result,
        architecture=architecture,
        task=task,
        weights=training_result["result"]["weights"],
    )


def _load_model_runtime_from_weight_bytes(
    mldb_data_root: str | Path,
    model_id: ModelId | str,
    weight_bytes: bytes,
) -> _LoadedModelRuntime:
    lineage = _resolve_model_lineage(mldb_data_root, model_id)
    state = _load_canonical_state_dict_bytes(weight_bytes, ref=lineage.weights)
    module = _load_state_into_fresh_architecture(
        mldb_data_root,
        lineage.architecture["id"],
        state,
    )
    return _LoadedModelRuntime(
        definition=lineage.model,
        training_result=lineage.training_result,
        architecture=lineage.architecture,
        module=module,
    )
