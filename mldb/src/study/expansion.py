"""Deterministic MLDB Study plan materialization."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from itertools import product

from ..common.ids import EvaluationProtocolId
from ..common.parameters import PublicParameterValue, resolve_public_parameters
from ..evaluation.protocol import EvaluationProtocol
from ..training.protocol import TrainProtocol
from .definition import (
    Study,
    StudyExistingModelSource,
    StudyStatus,
    StudyTrainingModelSource,
    validate_study_metadata,
)
from .plan import (
    StudyPlan,
    StudyPlanEvaluation,
    StudyPlanRow,
    StudyPlanTraining,
    validate_study_plan,
)


def materialize_study_plan(
    study: Study,
    train_protocol: TrainProtocol | None,
    evaluation_protocols: Mapping[EvaluationProtocolId, EvaluationProtocol],
) -> StudyPlan:
    """Materialize one preflight-ready sealed Study into its ordered resolved plan."""

    if study.status is not StudyStatus.SEALED:
        raise ValueError("Study materialization requires a sealed Study")

    metadata_report = validate_study_metadata(study)
    if not metadata_report.valid:
        codes = ", ".join(issue.code for issue in metadata_report.issues)
        raise ValueError(f"invalid Study metadata: {codes}")

    if not isinstance(evaluation_protocols, Mapping):
        raise ValueError("evaluation_protocols must be a mapping")

    resolved_evaluations = _resolve_evaluations(study, evaluation_protocols)
    rows: list[StudyPlanRow] = []

    if isinstance(study.model, StudyTrainingModelSource):
        if train_protocol is None:
            raise ValueError("training-source Study requires its Train Protocol")
        if train_protocol.id != study.model.protocol:
            raise ValueError("Train Protocol id does not match Study model.train.protocol")
        rows.extend(
            _materialize_training_rows(
                study.model,
                train_protocol,
                resolved_evaluations,
            )
        )
    elif isinstance(study.model, StudyExistingModelSource):
        if train_protocol is not None:
            raise ValueError("existing-Model Study must not receive a Train Protocol")
        for model in study.model.models:
            rows.append(
                StudyPlanRow(
                    trial="",
                    model=model,
                    evaluations=_copy_evaluations(resolved_evaluations),
                )
            )
    else:
        raise ValueError("unsupported Study Model source")

    plan = tuple(
        StudyPlanRow(
            trial=f"trial-{index:04d}",
            training=row.training,
            model=row.model,
            evaluations=row.evaluations,
        )
        for index, row in enumerate(rows, start=1)
    )

    plan_report = validate_study_plan(plan)
    if not plan_report.valid:
        codes = ", ".join(issue.code for issue in plan_report.issues)
        raise ValueError(f"materialized Study plan is invalid: {codes}")
    return plan


def _resolve_evaluations(
    study: Study,
    evaluation_protocols: Mapping[EvaluationProtocolId, EvaluationProtocol],
) -> tuple[StudyPlanEvaluation, ...]:
    resolved: list[StudyPlanEvaluation] = []
    for stage in study.evaluations:
        try:
            protocol = evaluation_protocols[stage.protocol]
        except KeyError as exc:
            raise ValueError(
                f"missing Evaluation Protocol metadata for {stage.protocol}"
            ) from exc
        if not isinstance(protocol, EvaluationProtocol):
            raise ValueError(
                f"evaluation protocol mapping entry for {stage.protocol} has invalid type"
            )
        if protocol.id != stage.protocol:
            raise ValueError(
                f"Evaluation Protocol id does not match stage reference {stage.protocol}"
            )
        parameters = resolve_public_parameters(protocol.parameters, stage.parameters)
        resolved.append(
            StudyPlanEvaluation(
                stage=stage.stage,
                corpus=stage.corpus,
                protocol=stage.protocol,
                parameters=_copy_parameter_mapping(parameters),
            )
        )
    return tuple(resolved)


def _materialize_training_rows(
    source: StudyTrainingModelSource,
    train_protocol: TrainProtocol,
    evaluations: tuple[StudyPlanEvaluation, ...],
) -> list[StudyPlanRow]:
    parameter_keys = sorted(source.parameters)
    parameter_axes = [source.parameters[key].values for key in parameter_keys]

    rows: list[StudyPlanRow] = []
    for architecture in source.architectures:
        for values in product(*parameter_axes):
            overrides = dict(zip(parameter_keys, values, strict=True))
            resolved = resolve_public_parameters(train_protocol.parameters, overrides)
            for seed in source.seeds:
                rows.append(
                    StudyPlanRow(
                        trial="",
                        training=StudyPlanTraining(
                            architecture=architecture,
                            corpus=source.corpus,
                            protocol=source.protocol,
                            seed=seed,
                            parameters=_copy_parameter_mapping(resolved),
                        ),
                        evaluations=_copy_evaluations(evaluations),
                    )
                )
    return rows


def _copy_evaluations(
    evaluations: tuple[StudyPlanEvaluation, ...],
) -> tuple[StudyPlanEvaluation, ...]:
    return tuple(
        StudyPlanEvaluation(
            stage=evaluation.stage,
            corpus=evaluation.corpus,
            protocol=evaluation.protocol,
            parameters=_copy_parameter_mapping(evaluation.parameters),
        )
        for evaluation in evaluations
    )


def _copy_parameter_mapping(parameters: Mapping[str, PublicParameterValue]) -> dict[str, PublicParameterValue]:
    return {key: deepcopy(value) for key, value in parameters.items()}
