"""Runtime loading for deterministic MLDB Models."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from ..catalog.architecture import ArchitectureStatus
from ..common.errors import MldbError, ValidationFailedError
from ..runtime.catalog_handles import ArchitectureHandle
from ..runtime.executable_loader import load_architecture_build
from ..training.run import (
    TrainingRun,
    TrainingRunStatus,
    validate_training_run,
)
from ..training.weights import load_canonical_weights
from .identity import Model, validate_model_metadata


@dataclass(frozen=True, slots=True)
class ModelHandle:
    """Validated Model lineage plus the resources required to load learned state."""

    metadata: Model
    metadata_path: Path | None
    training_run: TrainingRun
    training_run_metadata_path: Path | None
    weights_path: Path
    architecture: ArchitectureHandle


def load_model(model: ModelHandle) -> torch.nn.Module:
    """Load the learned module represented by one runtime ModelHandle."""

    model_report = validate_model_metadata(model.metadata)
    if not model_report.valid:
        raise ValidationFailedError(model_report)

    run_report = validate_training_run(model.training_run)
    if not run_report.valid:
        raise ValidationFailedError(run_report)
    if model.metadata.training_run != model.training_run.id:
        raise MldbError("Model and Training Run lineage IDs disagree")
    if (
        model.training_run.status is not TrainingRunStatus.COMPLETED
        or model.training_run.result is None
    ):
        raise MldbError("Model lineage Training Run is not completed")

    architecture = model.architecture
    if model.training_run.architecture != architecture.metadata.id:
        raise MldbError("Training Run and resolved Architecture IDs disagree")
    if architecture.metadata.status is not ArchitectureStatus.SEALED:
        raise MldbError("Model Architecture is not sealed")
    if architecture.metadata.implementation.sha256 is None:
        raise MldbError("sealed Model Architecture has no implementation SHA-256")

    weights = model.training_run.result.weights
    if weights.format != "pytorch-state-dict":
        raise MldbError("Model weights format is not canonical")
    if weights.path != "artifacts/weights.pt":
        raise MldbError("Model weights path is not canonical")

    try:
        weight_bytes = model.weights_path.read_bytes()
    except OSError as error:
        raise MldbError(
            f"could not read Model weights: {model.weights_path}"
        ) from error
    if len(weight_bytes) != weights.bytes:
        raise MldbError("Model weights byte count mismatch")
    if hashlib.sha256(weight_bytes).hexdigest() != weights.sha256.lower():
        raise MldbError("Model weights SHA-256 mismatch")

    architecture_build = load_architecture_build(architecture)
    return load_canonical_weights(model.weights_path, architecture_build)
