"""Deterministic MLDB Model identity and metadata validation."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from ..common.errors import (
    LifecycleConflictError,
    ValidationFailedError,
    ValidationIssue,
    ValidationReport,
)
from ..common.ids import ModelId, TrainingRunId
from ..training.run import (
    TrainingRun,
    TrainingRunStatus,
    validate_training_run,
)


@dataclass(frozen=True, slots=True)
class Model:
    """One immutable MLDB Model v1 identity record."""

    schema: Literal["mjtensu.mldb/model/v1"]
    id: ModelId
    training_run: TrainingRunId


def model_id_for_training_run(
    training_run_id: TrainingRunId,
) -> ModelId:
    """Derive the deterministic Model v1 identity for one Training Run v1 ID."""

    return ModelId(f"mdl-{str(training_run_id)[3:]}")


def model_for_completed_training_run(
    training_run: TrainingRun,
) -> Model:
    """Construct the exact Model v1 record for an already-valid completed Training Run."""

    report = validate_training_run(training_run)
    if not report.valid:
        raise ValidationFailedError(report)
    if (
        training_run.status is not TrainingRunStatus.COMPLETED
        or training_run.result is None
    ):
        raise LifecycleConflictError(
            "a Model can be constructed only for a completed Training Run"
        )
    return Model(
        schema="mjtensu.mldb/model/v1",
        id=model_id_for_training_run(training_run.id),
        training_run=training_run.id,
    )


def validate_model_metadata(model: Model) -> ValidationReport:
    """Validate Model v1 metadata-local invariants without external I/O."""

    issues: list[ValidationIssue] = []
    if model.schema != "mjtensu.mldb/model/v1":
        issues.append(
            ValidationIssue(
                code="model.schema.unsupported",
                message="Model schema must be 'mjtensu.mldb/model/v1'.",
                path="schema",
            )
        )

    model_id_valid = (
        type(model.id) is str and _MODEL_ID_PATTERN.fullmatch(model.id) is not None
    )
    if not model_id_valid:
        issues.append(
            ValidationIssue(
                code="model.id.invalid",
                message=(
                    "Model id must match 'mdl-YYYYMMDD-NNN' "
                    "with a positive sequence."
                ),
                path="id",
            )
        )

    training_run_id_valid = (
        type(model.training_run) is str
        and _TRAINING_RUN_ID_PATTERN.fullmatch(model.training_run) is not None
    )
    if not training_run_id_valid:
        issues.append(
            ValidationIssue(
                code="model.training_run.invalid",
                message=(
                    "Model training_run must match 'tr-YYYYMMDD-NNN' "
                    "with a positive sequence."
                ),
                path="training_run",
            )
        )

    if (
        model_id_valid
        and training_run_id_valid
        and model.id[4:] != model.training_run[3:]
    ):
        issues.append(
            ValidationIssue(
                code="model.training_run.identity_mismatch",
                message=(
                    "Model id and Training Run id must have the same "
                    "date and sequence suffix."
                ),
                path="training_run",
            )
        )

    return ValidationReport(tuple(issues))


_MODEL_ID_PATTERN = re.compile(r"mdl-[0-9]{8}-(?!000)[0-9]{3}\Z")
_TRAINING_RUN_ID_PATTERN = re.compile(r"tr-[0-9]{8}-(?!000)[0-9]{3}\Z")
