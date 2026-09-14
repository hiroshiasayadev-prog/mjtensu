"""Private validation/loading for MLDB v2 declarative Study definitions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from mldb_v2.src.catalog._core_definition_validation import (
    _require_exact_fields,
    _require_exact_mapping,
    _require_string,
    _validate_lifecycle,
    _validate_versioned_entity_id,
)
from mldb_v2.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EntityKind,
    EvaluationProtocolId,
    ModelId,
    StudyId,
    TrainProtocolId,
    _validate_typed_reference,
)
from mldb_v2.src.common.parameters import (
    _public_parameter_values_equal,
    _validate_public_parameter_value,
    _validate_training_seed,
)
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver

from .study import Study


_SCHEMA = "mjtensu.mldb-v2/study/v1"
_TOP_LEVEL_FIELDS = {"schema", "id", "status", "name", "description", "model", "evaluations"}
_TRAIN_FIELDS = {"corpus", "protocol", "architectures", "parameters", "seeds"}
_EVALUATION_FIELDS = {"stage", "corpus", "protocol", "parameters"}
_STAGE_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.ASCII)


def _validate_parameter_grid(value: object, *, label: str) -> dict[str, object]:
    grid = _require_exact_mapping(value, label=label)
    for key, raw_axis in grid.items():
        if type(key) is not str:
            raise ValueError(f"{label} keys must be strings")
        axis = _require_exact_mapping(raw_axis, label=f"{label}.{key}")
        _require_exact_fields(axis, {"values"}, label=f"{label}.{key} axis")
        values = axis["values"]
        if type(values) is not list or not values:
            raise ValueError(f"{label}.{key}.values must be a non-empty list")
        for item in values:
            _validate_public_parameter_value(item)
        for index, item in enumerate(values):
            if any(_public_parameter_values_equal(item, prior) for prior in values[:index]):
                raise ValueError(f"{label}.{key}.values contains duplicate values")
    return grid


def _validate_training_source(value: object) -> None:
    source = _require_exact_mapping(value, label="Study model.train")
    _require_exact_fields(source, _TRAIN_FIELDS, label="Study model.train")
    _validate_versioned_entity_id(source["corpus"])
    _validate_versioned_entity_id(source["protocol"])

    architectures = source["architectures"]
    if type(architectures) is not list or not architectures:
        raise ValueError("Study model.train.architectures must be a non-empty list")
    validated_architectures: list[str] = []
    for item in architectures:
        validated_architectures.append(_validate_versioned_entity_id(item))
    if len(validated_architectures) != len(set(validated_architectures)):
        raise ValueError("Study model.train.architectures must be unique")

    _validate_parameter_grid(source["parameters"], label="Study model.train.parameters")

    seeds = source["seeds"]
    if type(seeds) is not list or not seeds:
        raise ValueError("Study model.train.seeds must be a non-empty list")
    validated_seeds = [_validate_training_seed(seed) for seed in seeds]
    if len(validated_seeds) != len(set(validated_seeds)):
        raise ValueError("Study model.train.seeds must be unique")


def _validate_existing_source(value: object) -> None:
    if type(value) is not list or not value:
        raise ValueError("Study model.existing must be a non-empty list")
    validated: list[str] = []
    for item in value:
        validated.append(_validate_typed_reference(item))
    if len(validated) != len(set(validated)):
        raise ValueError("Study model.existing must be unique")


def _validate_model_source(value: object) -> None:
    model = _require_exact_mapping(value, label="Study model")
    keys = set(model)
    if keys == {"train"}:
        _validate_training_source(model["train"])
        return
    if keys == {"existing"}:
        _validate_existing_source(model["existing"])
        return
    raise ValueError("Study model must contain exactly one of train or existing")


def _validate_evaluations(value: object) -> None:
    if type(value) is not list or not value:
        raise ValueError("Study evaluations must be a non-empty list")
    stages: list[str] = []
    for raw_stage in value:
        stage = _require_exact_mapping(raw_stage, label="Study evaluation stage")
        _require_exact_fields(stage, _EVALUATION_FIELDS, label="Study evaluation stage")
        stage_id = _require_string(stage["stage"], label="Study evaluation stage", non_empty=True)
        if _STAGE_RE.fullmatch(stage_id) is None:
            raise ValueError("Study evaluation stage must be lowercase kebab-case")
        stages.append(stage_id)
        _validate_versioned_entity_id(stage["corpus"])
        _validate_versioned_entity_id(stage["protocol"])
        _validate_parameter_grid(stage["parameters"], label=f"Study evaluation {stage_id} parameters")
    if len(stages) != len(set(stages)):
        raise ValueError("Study evaluation stage identifiers must be unique")


def _validate_study(value: object, *, expected_id: str | None = None) -> Study:
    document = _require_exact_mapping(value, label="Study")
    _require_exact_fields(document, _TOP_LEVEL_FIELDS, label="Study")
    if document["schema"] != _SCHEMA:
        raise ValueError("unsupported Study schema")
    _validate_versioned_entity_id(document["id"], expected_id=expected_id)
    _validate_lifecycle(document["status"])
    _require_string(document["name"], label="Study name", non_empty=True)
    _require_string(document["description"], label="Study description")
    _validate_model_source(document["model"])
    _validate_evaluations(document["evaluations"])
    return cast(Study, document)


def _load_study_definition(
    mldb_data_root: str | Path,
    study_id: StudyId | str,
) -> Study:
    expected_id = _validate_versioned_entity_id(study_id)
    resolver = CanonicalRepositoryResolver(mldb_data_root)
    document = resolver.resolve(kind=EntityKind.STUDY, entity_id=StudyId(expected_id))
    return _validate_study(document, expected_id=expected_id)
