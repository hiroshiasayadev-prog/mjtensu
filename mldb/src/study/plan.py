"""Materialized MLDB Study Run plan domain implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import TypeAlias

from ..common.errors import ValidationIssue, ValidationReport
from ..common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    ModelId,
    TrainProtocolId,
)
from ..common.parameters import (
    PublicParameterValue,
    ResolvedPublicParameters,
    is_public_parameter_value,
)


@dataclass(frozen=True, slots=True)
class StudyPlanTraining:
    """Fully resolved training coordinate for one training-derived trial."""

    architecture: ArchitectureId
    corpus: CorpusId
    protocol: TrainProtocolId
    seed: int
    parameters: ResolvedPublicParameters


@dataclass(frozen=True, slots=True)
class StudyPlanEvaluation:
    """Fully resolved evaluation-stage intent for one trial Model."""

    stage: str
    corpus: CorpusId
    protocol: EvaluationProtocolId
    parameters: ResolvedPublicParameters


@dataclass(frozen=True, slots=True)
class StudyPlanRow:
    """One materialized Study Run-local trial row."""

    trial: str
    evaluations: Sequence[StudyPlanEvaluation]
    training: StudyPlanTraining | None = None
    model: ModelId | None = None


StudyPlan: TypeAlias = Sequence[StudyPlanRow]


def validate_study_plan(plan: StudyPlan) -> ValidationReport:
    """Validate plan-local structural invariants without Study/protocol I/O."""

    issues: list[ValidationIssue] = []
    if not _is_sequence_value(plan):
        _add_issue(issues, "study_plan.shape", "plan must be an ordered sequence")
        return ValidationReport(tuple(issues))
    if not plan:
        _add_issue(issues, "study_plan.empty", "plan must be non-empty")
        return ValidationReport(tuple(issues))

    seen_trials: set[str] = set()
    seen_coordinates: set[object] = set()

    for index, row in enumerate(plan, start=1):
        row_path = f"plan[{index - 1}]"
        if not isinstance(row, StudyPlanRow):
            _add_issue(
                issues,
                "study_plan.row.shape",
                "plan entry must be a StudyPlanRow",
                row_path,
            )
            continue

        expected_trial = f"trial-{index:04d}"
        if not isinstance(row.trial, str) or _TRIAL_ID.fullmatch(row.trial) is None:
            _add_issue(
                issues,
                "study_plan.trial",
                "trial must follow trial-NNNN numbering",
                f"{row_path}.trial",
            )
        else:
            if row.trial in seen_trials:
                _add_issue(
                    issues,
                    "study_plan.trial_duplicate",
                    "trial identifiers must be unique",
                    f"{row_path}.trial",
                )
            else:
                seen_trials.add(row.trial)
            if row.trial != expected_trial:
                _add_issue(
                    issues,
                    "study_plan.trial_sequence",
                    "trial identifiers must begin at trial-0001 and increase without gaps",
                    f"{row_path}.trial",
                )

        has_training = row.training is not None
        has_model = row.model is not None
        if has_training == has_model:
            _add_issue(
                issues,
                "study_plan.model_source",
                "row must contain exactly one of training or model",
                row_path,
            )

        coordinate: object | None = None
        if row.training is not None:
            coordinate = _validate_training(row.training, issues, f"{row_path}.training")
        if row.model is not None:
            if not isinstance(row.model, str):
                _add_issue(
                    issues,
                    "study_plan.model",
                    "model must be a ModelId string",
                    f"{row_path}.model",
                )
            else:
                coordinate = ("model", row.model)

        if coordinate is not None:
            if coordinate in seen_coordinates:
                _add_issue(
                    issues,
                    "study_plan.coordinate_duplicate",
                    "materialized Model-source coordinates must be unique",
                    row_path,
                )
            else:
                seen_coordinates.add(coordinate)

        _validate_evaluations(row.evaluations, issues, f"{row_path}.evaluations")

    return ValidationReport(tuple(issues))


_TRIAL_ID = re.compile(r"trial-(?!0000)[0-9]{4}\Z")


def _add_issue(
    issues: list[ValidationIssue],
    code: str,
    message: str,
    path: str | None = None,
) -> None:
    issues.append(ValidationIssue(code=code, message=message, path=path))


def _is_sequence_value(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _validate_training(
    training: object,
    issues: list[ValidationIssue],
    path: str,
) -> object | None:
    if not isinstance(training, StudyPlanTraining):
        _add_issue(
            issues,
            "study_plan.training.shape",
            "training must be a StudyPlanTraining",
            path,
        )
        return None

    valid_coordinate = True
    if not isinstance(training.architecture, str):
        _add_issue(
            issues,
            "study_plan.training.architecture",
            "architecture must be an ArchitectureId string",
            f"{path}.architecture",
        )
        valid_coordinate = False
    if not isinstance(training.corpus, str):
        _add_issue(
            issues,
            "study_plan.training.corpus",
            "corpus must be a CorpusId string",
            f"{path}.corpus",
        )
        valid_coordinate = False
    if not isinstance(training.protocol, str):
        _add_issue(
            issues,
            "study_plan.training.protocol",
            "protocol must be a TrainProtocolId string",
            f"{path}.protocol",
        )
        valid_coordinate = False
    if isinstance(training.seed, bool) or not isinstance(training.seed, int):
        _add_issue(
            issues,
            "study_plan.training.seed",
            "training seed must be an integer and not boolean",
            f"{path}.seed",
        )
        valid_coordinate = False

    parameters_fingerprint = _validate_parameter_mapping(
        training.parameters,
        issues,
        f"{path}.parameters",
    )
    if parameters_fingerprint is None:
        valid_coordinate = False

    if not valid_coordinate:
        return None
    return (
        "training",
        training.architecture,
        training.corpus,
        training.protocol,
        training.seed,
        parameters_fingerprint,
    )


def _validate_evaluations(
    evaluations: object,
    issues: list[ValidationIssue],
    path: str,
) -> None:
    if not _is_sequence_value(evaluations):
        _add_issue(
            issues,
            "study_plan.evaluations.shape",
            "evaluations must be an ordered sequence",
            path,
        )
        return
    if not evaluations:
        _add_issue(
            issues,
            "study_plan.evaluations.empty",
            "evaluations must be non-empty",
            path,
        )
        return

    seen_stages: set[str] = set()
    for index, evaluation in enumerate(evaluations):
        evaluation_path = f"{path}[{index}]"
        if not isinstance(evaluation, StudyPlanEvaluation):
            _add_issue(
                issues,
                "study_plan.evaluation.shape",
                "evaluation entry must be a StudyPlanEvaluation",
                evaluation_path,
            )
            continue
        if not isinstance(evaluation.stage, str):
            _add_issue(
                issues,
                "study_plan.evaluation.stage",
                "evaluation stage must be a string",
                f"{evaluation_path}.stage",
            )
        elif evaluation.stage in seen_stages:
            _add_issue(
                issues,
                "study_plan.evaluation.stage_duplicate",
                "evaluation stages must be unique within each row",
                f"{evaluation_path}.stage",
            )
        else:
            seen_stages.add(evaluation.stage)
        if not isinstance(evaluation.corpus, str):
            _add_issue(
                issues,
                "study_plan.evaluation.corpus",
                "evaluation corpus must be a CorpusId string",
                f"{evaluation_path}.corpus",
            )
        if not isinstance(evaluation.protocol, str):
            _add_issue(
                issues,
                "study_plan.evaluation.protocol",
                "evaluation protocol must be an EvaluationProtocolId string",
                f"{evaluation_path}.protocol",
            )
        _validate_parameter_mapping(
            evaluation.parameters,
            issues,
            f"{evaluation_path}.parameters",
        )


def _validate_parameter_mapping(
    parameters: object,
    issues: list[ValidationIssue],
    path: str,
) -> object | None:
    if not isinstance(parameters, Mapping):
        _add_issue(
            issues,
            "study_plan.parameters.shape",
            "resolved public parameters must be a mapping",
            path,
        )
        return None

    entries: list[tuple[str, object]] = []
    valid = True
    for key, value in parameters.items():
        value_path = f"{path}[{key!r}]"
        if not isinstance(key, str):
            _add_issue(
                issues,
                "study_plan.parameters.key",
                "resolved public parameter keys must be strings",
                value_path,
            )
            valid = False
            continue
        if not is_public_parameter_value(value):
            _add_issue(
                issues,
                "study_plan.parameters.value",
                "resolved public parameter value is outside the public value domain",
                value_path,
            )
            valid = False
            continue
        entries.append((key, _typed_public_value_fingerprint(value)))

    if not valid:
        return None
    entries.sort(key=lambda item: item[0])
    return tuple(entries)


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
