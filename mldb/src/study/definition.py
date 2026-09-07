"""MLDB Study v1 definition domain implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import re
from typing import Literal, TypeAlias

from ..common.errors import ValidationIssue, ValidationReport
from ..common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    ModelId,
    StudyId,
    TrainProtocolId,
)
from ..common.parameters import (
    PublicParameterOverrides,
    PublicParameterValue,
    is_public_parameter_value,
)


class StudyStatus(str, Enum):
    """Persisted Study definition lifecycle state."""

    DRAFT = "draft"
    SEALED = "sealed"


@dataclass(frozen=True, slots=True)
class StudyTrainingParameterAxis:
    """One caller-authored Train Protocol public-parameter grid axis."""

    values: Sequence[PublicParameterValue]


@dataclass(frozen=True, slots=True)
class StudyTrainingModelSource:
    """Study Model source that materializes new Models from a finite training grid."""

    corpus: CorpusId
    protocol: TrainProtocolId
    architectures: Sequence[ArchitectureId]
    parameters: Mapping[str, StudyTrainingParameterAxis]
    seeds: Sequence[int]


@dataclass(frozen=True, slots=True)
class StudyExistingModelSource:
    """Study Model source selecting already-created Models in authored order."""

    models: Sequence[ModelId]


StudyModelSource: TypeAlias = StudyTrainingModelSource | StudyExistingModelSource


@dataclass(frozen=True, slots=True)
class StudyEvaluationStage:
    """One fixed evaluation declaration applied to every Study trial Model."""

    stage: str
    corpus: CorpusId
    protocol: EvaluationProtocolId
    parameters: PublicParameterOverrides


@dataclass(frozen=True, slots=True)
class Study:
    """One reusable MLDB Study v1 experiment definition."""

    schema: Literal["mjtensu.mldb/study/v1"]
    id: StudyId
    status: StudyStatus
    name: str
    description: str
    model: StudyModelSource
    evaluations: Sequence[StudyEvaluationStage]


def validate_study_metadata(study: Study) -> ValidationReport:
    """Validate Study-format invariants decidable from one normalized definition."""

    issues: list[ValidationIssue] = []

    if study.schema != "mjtensu.mldb/study/v1":
        _add_issue(issues, "study.schema", "unsupported Study schema", "schema")

    if not isinstance(study.id, str) or _VERSIONED_ID.fullmatch(study.id) is None:
        _add_issue(
            issues,
            "study.id",
            "Study id must end in -v<positive-integer>",
            "id",
        )

    if not isinstance(study.status, StudyStatus):
        _add_issue(issues, "study.status", "invalid Study status", "status")

    if isinstance(study.model, StudyTrainingModelSource):
        _validate_training_source(study.model, issues)
    elif isinstance(study.model, StudyExistingModelSource):
        _validate_existing_source(study.model, issues)
    else:
        _add_issue(
            issues,
            "study.model_source",
            "Study model must be exactly one supported Model source",
            "model",
        )

    if not _is_sequence_value(study.evaluations):
        _add_issue(
            issues,
            "study.evaluations.shape",
            "evaluations must be an ordered sequence",
            "evaluations",
        )
    elif not study.evaluations:
        _add_issue(
            issues,
            "study.evaluations.empty",
            "evaluations must be non-empty",
            "evaluations",
        )
    else:
        seen_stages: set[str] = set()
        for index, evaluation in enumerate(study.evaluations):
            path = f"evaluations[{index}]"
            if not isinstance(evaluation, StudyEvaluationStage):
                _add_issue(
                    issues,
                    "study.evaluation.shape",
                    "evaluation entry must be a StudyEvaluationStage",
                    path,
                )
                continue
            if not isinstance(evaluation.stage, str):
                _add_issue(
                    issues,
                    "study.evaluation.stage",
                    "evaluation stage must be a string",
                    f"{path}.stage",
                )
            elif evaluation.stage in seen_stages:
                _add_issue(
                    issues,
                    "study.evaluation.stage_duplicate",
                    "evaluation stage must be unique within the Study",
                    f"{path}.stage",
                )
            else:
                seen_stages.add(evaluation.stage)
            _validate_public_parameter_mapping(
                evaluation.parameters,
                issues,
                f"{path}.parameters",
                "study.evaluation.parameters",
            )

    return ValidationReport(tuple(issues))


_VERSIONED_ID = re.compile(r".+-v[1-9][0-9]*\Z")


def _add_issue(
    issues: list[ValidationIssue],
    code: str,
    message: str,
    path: str | None = None,
) -> None:
    issues.append(ValidationIssue(code=code, message=message, path=path))


def _is_sequence_value(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _validate_training_source(
    source: StudyTrainingModelSource,
    issues: list[ValidationIssue],
) -> None:
    if not _is_sequence_value(source.architectures):
        _add_issue(
            issues,
            "study.training.architectures.shape",
            "architectures must be an ordered sequence",
            "model.train.architectures",
        )
    elif not source.architectures:
        _add_issue(
            issues,
            "study.training.architectures.empty",
            "architectures must be non-empty",
            "model.train.architectures",
        )
    else:
        seen_architectures: set[str] = set()
        for index, architecture in enumerate(source.architectures):
            path = f"model.train.architectures[{index}]"
            if not isinstance(architecture, str):
                _add_issue(
                    issues,
                    "study.training.architecture",
                    "architecture must be an ArchitectureId string",
                    path,
                )
                continue
            if architecture in seen_architectures:
                _add_issue(
                    issues,
                    "study.training.architecture_duplicate",
                    "architectures must be unique",
                    path,
                )
            else:
                seen_architectures.add(architecture)

    if not isinstance(source.parameters, Mapping):
        _add_issue(
            issues,
            "study.training.parameters.shape",
            "parameters must be a mapping",
            "model.train.parameters",
        )
    else:
        for key, axis in source.parameters.items():
            key_path = f"model.train.parameters[{key!r}]"
            if not isinstance(key, str):
                _add_issue(
                    issues,
                    "study.training.parameter_key",
                    "training parameter keys must be strings",
                    key_path,
                )
            if not isinstance(axis, StudyTrainingParameterAxis):
                _add_issue(
                    issues,
                    "study.training.parameter_axis",
                    "training parameter entry must be a StudyTrainingParameterAxis",
                    key_path,
                )
                continue
            if not _is_sequence_value(axis.values):
                _add_issue(
                    issues,
                    "study.training.parameter_values.shape",
                    "training parameter axis values must be an ordered sequence",
                    f"{key_path}.values",
                )
                continue
            if not axis.values:
                _add_issue(
                    issues,
                    "study.training.parameter_values.empty",
                    "training parameter axis values must be non-empty",
                    f"{key_path}.values",
                )
                continue

            seen_values: set[object] = set()
            for value_index, value in enumerate(axis.values):
                value_path = f"{key_path}.values[{value_index}]"
                if not is_public_parameter_value(value):
                    _add_issue(
                        issues,
                        "study.training.parameter_value",
                        "training parameter value is outside the public value domain",
                        value_path,
                    )
                    continue
                fingerprint = _typed_public_value_fingerprint(value)
                if fingerprint in seen_values:
                    _add_issue(
                        issues,
                        "study.training.parameter_value_duplicate",
                        "training parameter axis contains a duplicate typed value",
                        value_path,
                    )
                else:
                    seen_values.add(fingerprint)

    if not _is_sequence_value(source.seeds):
        _add_issue(
            issues,
            "study.training.seeds.shape",
            "seeds must be an ordered sequence",
            "model.train.seeds",
        )
    elif not source.seeds:
        _add_issue(
            issues,
            "study.training.seeds.empty",
            "seeds must be non-empty",
            "model.train.seeds",
        )
    else:
        seen_seeds: set[int] = set()
        for index, seed in enumerate(source.seeds):
            path = f"model.train.seeds[{index}]"
            if isinstance(seed, bool) or not isinstance(seed, int):
                _add_issue(
                    issues,
                    "study.training.seed",
                    "training seed must be an integer and not boolean",
                    path,
                )
                continue
            if seed in seen_seeds:
                _add_issue(
                    issues,
                    "study.training.seed_duplicate",
                    "training seeds must be unique",
                    path,
                )
            else:
                seen_seeds.add(seed)


def _validate_existing_source(
    source: StudyExistingModelSource,
    issues: list[ValidationIssue],
) -> None:
    if not _is_sequence_value(source.models):
        _add_issue(
            issues,
            "study.existing.models.shape",
            "existing models must be an ordered sequence",
            "model.existing",
        )
        return
    if not source.models:
        _add_issue(
            issues,
            "study.existing.models.empty",
            "existing models must be non-empty",
            "model.existing",
        )
        return

    seen_models: set[str] = set()
    for index, model in enumerate(source.models):
        path = f"model.existing[{index}]"
        if not isinstance(model, str):
            _add_issue(
                issues,
                "study.existing.model",
                "existing model must be a ModelId string",
                path,
            )
            continue
        if model in seen_models:
            _add_issue(
                issues,
                "study.existing.model_duplicate",
                "existing models must be unique",
                path,
            )
        else:
            seen_models.add(model)


def _validate_public_parameter_mapping(
    parameters: object,
    issues: list[ValidationIssue],
    path: str,
    code_prefix: str,
) -> None:
    if not isinstance(parameters, Mapping):
        _add_issue(
            issues,
            f"{code_prefix}.shape",
            "public parameters must be a mapping",
            path,
        )
        return
    for key, value in parameters.items():
        value_path = f"{path}[{key!r}]"
        if not isinstance(key, str):
            _add_issue(
                issues,
                f"{code_prefix}.key",
                "public parameter keys must be strings",
                value_path,
            )
        if not is_public_parameter_value(value):
            _add_issue(
                issues,
                f"{code_prefix}.value",
                "public parameter value is outside the public value domain",
                value_path,
            )


def _typed_public_value_fingerprint(value: PublicParameterValue) -> object:
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, list):
        return (
            "list",
            tuple(_typed_public_value_fingerprint(item) for item in value),
        )
    return (
        "dict",
        tuple(
            (key, _typed_public_value_fingerprint(value[key]))
            for key in sorted(value)
        ),
    )
