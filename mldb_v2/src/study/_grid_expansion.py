"""Deterministic private Study grid expansion for MLDB v2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Iterator, Literal, Mapping, TypeAlias

from mldb_v2.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationCoordinateId,
    EvaluationProtocolId,
    ModelId,
    TaskId,
    TrainProtocolId,
    TrialId,
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
)
from mldb_v2.src.common.parameters import (
    PublicParameterValue,
    _public_parameter_values_equal,
    _resolve_public_parameters,
    _validate_training_seed,
)
from mldb_v2.src.study._planning_preflight import (
    _ExistingModelPlanningInput,
    _StudyPlanningInput,
    _TrainingPlanningInput,
)

_MAX_SEQUENCE_ID = 9999


class _GridExpansionError(ValueError):
    """Bounded private failure raised by deterministic grid expansion."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class _ExpandedTrainingSource:
    task: TaskId
    corpus: CorpusId
    architecture: ArchitectureId
    train_protocol: TrainProtocolId
    parameters: Mapping[str, PublicParameterValue]
    seed: int
    kind: Literal["training"] = "training"


@dataclass(frozen=True)
class _ExpandedExistingModelSource:
    model: ModelId
    kind: Literal["existing_model"] = "existing_model"


_ExpandedTrialSource: TypeAlias = _ExpandedTrainingSource | _ExpandedExistingModelSource


@dataclass(frozen=True)
class _ExpandedEvaluationCoordinate:
    coordinate: EvaluationCoordinateId
    stage: str
    task: TaskId
    corpus: CorpusId
    evaluation_protocol: EvaluationProtocolId
    parameters: Mapping[str, PublicParameterValue]


@dataclass(frozen=True)
class _ExpandedTrial:
    trial: TrialId
    source: _ExpandedTrialSource
    evaluations: tuple[_ExpandedEvaluationCoordinate, ...]


def _freeze_parameters(
    parameters: Mapping[str, PublicParameterValue],
) -> Mapping[str, PublicParameterValue]:
    return MappingProxyType(deepcopy(dict(parameters)))


def _ordered_axes(
    axes: Mapping[str, object],
) -> tuple[tuple[str, tuple[PublicParameterValue, ...]], ...]:
    ordered: list[tuple[str, tuple[PublicParameterValue, ...]]] = []
    if any(type(key) is not str for key in axes):
        raise _GridExpansionError("invalid_parameter_axis")
    for key in sorted(axes):
        raw_axis = axes[key]
        if type(raw_axis) is not dict or set(raw_axis) != {"values"}:
            raise _GridExpansionError("invalid_parameter_axis")
        values = raw_axis["values"]
        if type(values) is not list or not values:
            raise _GridExpansionError("invalid_parameter_axis")
        for index, value in enumerate(values):
            prior_values = values[:index]
            if any(
                _public_parameter_values_equal(value, prior)
                for prior in prior_values
            ):
                raise _GridExpansionError("duplicate_parameter_axis_value")
        ordered.append((key, tuple(values)))
    return tuple(ordered)


def _axis_cardinality(axes: Mapping[str, object]) -> int:
    count = 1
    for _, values in _ordered_axes(axes):
        count *= len(values)
    return count


def _axis_selections(
    axes: Mapping[str, object],
) -> Iterator[dict[str, PublicParameterValue]]:
    ordered = _ordered_axes(axes)
    if not ordered:
        yield {}
        return
    keys = tuple(key for key, _ in ordered)
    value_axes = tuple(values for _, values in ordered)
    for selection in product(*value_axes):
        yield dict(zip(keys, selection))


def _validate_unique_strings(values: tuple[str, ...], *, code: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise _GridExpansionError(code)
        seen.add(value)


def _training_trial_count(model: _TrainingPlanningInput) -> int:
    architecture_ids = tuple(str(item["id"]) for item in model.architectures)
    if not architecture_ids:
        raise _GridExpansionError("missing_architecture")
    _validate_unique_strings(architecture_ids, code="duplicate_architecture")
    if not model.seeds:
        raise _GridExpansionError("missing_seed")
    validated_seeds = tuple(_validate_training_seed(seed) for seed in model.seeds)
    if len(validated_seeds) != len(set(validated_seeds)):
        raise _GridExpansionError("duplicate_seed")
    return len(architecture_ids) * _axis_cardinality(model.parameter_axes) * len(validated_seeds)


def _existing_trial_count(model: _ExistingModelPlanningInput) -> int:
    model_ids = tuple(str(entry.model["id"]) for entry in model.models)
    if not model_ids:
        raise _GridExpansionError("missing_model")
    _validate_unique_strings(model_ids, code="duplicate_model")
    return len(model_ids)


def _require_sequence_capacity(count: int, *, code: str) -> int:
    if type(count) is not int or count < 1 or count > _MAX_SEQUENCE_ID:
        raise _GridExpansionError(code)
    return count


def _evaluation_coordinate_count(planning: _StudyPlanningInput) -> int:
    if not planning.evaluations:
        raise _GridExpansionError("missing_evaluation_stage")
    count = sum(_axis_cardinality(stage.parameter_axes) for stage in planning.evaluations)
    return _require_sequence_capacity(
        count, code="evaluation_coordinate_limit_exceeded"
    )


def _trial_count(planning: _StudyPlanningInput) -> int:
    if isinstance(planning.model, _TrainingPlanningInput):
        count = _training_trial_count(planning.model)
    else:
        count = _existing_trial_count(planning.model)
    return _require_sequence_capacity(count, code="trial_limit_exceeded")


def _validate_expansion_cardinality(planning: _StudyPlanningInput) -> tuple[int, int]:
    trial_count = _trial_count(planning)
    evaluation_count = _evaluation_coordinate_count(planning)
    return trial_count, evaluation_count


def _expand_evaluations(
    planning: _StudyPlanningInput,
) -> tuple[_ExpandedEvaluationCoordinate, ...]:
    coordinates: list[_ExpandedEvaluationCoordinate] = []
    sequence = 1
    task_id = TaskId(str(planning.task["id"]))
    for stage in planning.evaluations:
        for overrides in _axis_selections(stage.parameter_axes):
            resolved = _resolve_public_parameters(stage.parameter_declarations, overrides)
            coordinate_id = _validate_evaluation_coordinate_id(f"eval-{sequence:04d}")
            coordinates.append(
                _ExpandedEvaluationCoordinate(
                    coordinate=coordinate_id,
                    stage=str(stage.stage["stage"]),
                    task=task_id,
                    corpus=CorpusId(str(stage.corpus["id"])),
                    evaluation_protocol=EvaluationProtocolId(str(stage.protocol["id"])),
                    parameters=_freeze_parameters(resolved),
                )
            )
            sequence += 1
    return tuple(coordinates)


def _expand_training_trials(
    planning: _StudyPlanningInput,
    model: _TrainingPlanningInput,
) -> tuple[_ExpandedTrial, ...]:
    trials: list[_ExpandedTrial] = []
    sequence = 1
    task_id = TaskId(str(planning.task["id"]))
    corpus_id = CorpusId(str(model.corpus["id"]))
    protocol_id = TrainProtocolId(str(model.protocol["id"]))
    for architecture in model.architectures:
        architecture_id = ArchitectureId(str(architecture["id"]))
        for overrides in _axis_selections(model.parameter_axes):
            resolved = _resolve_public_parameters(model.parameter_declarations, overrides)
            for seed in model.seeds:
                validated_seed = _validate_training_seed(seed)
                trial_id = _validate_trial_id(f"trial-{sequence:04d}")
                trials.append(
                    _ExpandedTrial(
                        trial=trial_id,
                        source=_ExpandedTrainingSource(
                            kind="training",
                            task=task_id,
                            corpus=corpus_id,
                            architecture=architecture_id,
                            train_protocol=protocol_id,
                            parameters=_freeze_parameters(resolved),
                            seed=validated_seed,
                        ),
                        evaluations=_expand_evaluations(planning),
                    )
                )
                sequence += 1
    return tuple(trials)


def _expand_existing_trials(
    planning: _StudyPlanningInput,
    model: _ExistingModelPlanningInput,
) -> tuple[_ExpandedTrial, ...]:
    trials: list[_ExpandedTrial] = []
    for sequence, entry in enumerate(model.models, start=1):
        trial_id = _validate_trial_id(f"trial-{sequence:04d}")
        trials.append(
            _ExpandedTrial(
                trial=trial_id,
                source=_ExpandedExistingModelSource(
                    kind="existing_model",
                    model=ModelId(str(entry.model["id"])),
                ),
                evaluations=_expand_evaluations(planning),
            )
        )
    return tuple(trials)


def _expand_study_grid(
    planning: _StudyPlanningInput,
) -> tuple[_ExpandedTrial, ...]:
    """Expand one validated planning input without canonical/runtime side effects."""

    _validate_expansion_cardinality(planning)
    try:
        if isinstance(planning.model, _TrainingPlanningInput):
            return _expand_training_trials(planning, planning.model)
        if isinstance(planning.model, _ExistingModelPlanningInput):
            return _expand_existing_trials(planning, planning.model)
    except _GridExpansionError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise _GridExpansionError("invalid_expansion_input") from error
    raise _GridExpansionError("invalid_model_planning_input")
