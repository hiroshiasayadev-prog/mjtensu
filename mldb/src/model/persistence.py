"""Canonical persistence for deterministic MLDB Model records."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..common.errors import LifecycleConflictError
from ..common.ids import EntityKind, ModelId, TrainingRunId
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime._yaml import _YamlError, _load_yaml
from ..training.run import TrainingRun
from .identity import (
    Model,
    model_for_completed_training_run,
    validate_model_metadata,
)


def ensure_model_for_completed_training_run(
    training_run: TrainingRun,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> Model:
    """Ensure the one exact deterministic Model for a canonical completed Training Run."""

    canonical_run = _read_canonical_training_run(
        training_run.id,
        layout,
        filesystem,
    )
    if canonical_run != training_run:
        raise LifecycleConflictError(
            "supplied Training Run disagrees with its canonical record"
        )

    expected = model_for_completed_training_run(training_run)
    destination = layout.model_metadata_path(expected.id)
    models_directory = layout.entity_directory(EntityKind.MODEL)

    existing: Model | None = None
    if filesystem.file_exists(destination):
        existing = _read_model(destination, filesystem)
        if existing != expected:
            raise LifecycleConflictError(
                "existing deterministic Model conflicts with completed Training Run"
            )

    if filesystem.directory_exists(models_directory):
        for entry in filesystem.list_directory(models_directory):
            if entry == destination or entry.suffix != ".yaml":
                continue
            if not filesystem.file_exists(entry):
                continue
            if _claims_training_run(entry, expected.training_run, filesystem):
                raise LifecycleConflictError(
                    "another Model already claims the completed Training Run"
                )

    if existing is not None:
        return expected

    filesystem.ensure_directory(models_directory)
    filesystem.replace_text(
        destination,
        _serialize_model(expected),
        encoding="utf-8",
    )
    return expected


def _read_canonical_training_run(
    run_id: TrainingRunId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> TrainingRun:
    from ..runtime.run_persistence import read_training_run

    return read_training_run(run_id, layout, filesystem)


def _read_model(path: Path, filesystem: FilesystemPort) -> Model:
    try:
        raw = _load_yaml(filesystem.read_text(path, encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, _YamlError) as error:
        raise LifecycleConflictError(
            f"existing Model metadata is malformed: {path}"
        ) from error
    if not isinstance(raw, dict) or not all(type(key) is str for key in raw):
        raise LifecycleConflictError(
            f"existing Model metadata root is invalid: {path}"
        )
    if set(raw) != {"schema", "id", "training_run"}:
        raise LifecycleConflictError(
            f"existing Model metadata fields are invalid: {path}"
        )
    if not all(type(raw[key]) is str for key in raw):
        raise LifecycleConflictError(
            f"existing Model metadata values are invalid: {path}"
        )

    model = Model(
        schema=raw["schema"],
        id=ModelId(raw["id"]),
        training_run=TrainingRunId(raw["training_run"]),
    )
    report = validate_model_metadata(model)
    if not report.valid:
        raise LifecycleConflictError(
            f"existing Model metadata is invalid: {path}"
        )
    if path.stem != model.id:
        raise LifecycleConflictError(
            f"Model filename and recorded identity disagree: {path}"
        )
    return model


def _claims_training_run(
    path: Path,
    training_run_id: TrainingRunId,
    filesystem: FilesystemPort,
) -> bool:
    try:
        raw = _load_yaml(filesystem.read_text(path, encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, _YamlError):
        return False
    if not isinstance(raw, Mapping):
        return False
    return raw.get("training_run") == training_run_id


def _serialize_model(model: Model) -> str:
    return (
        f"schema: {model.schema}\n"
        f"id: {model.id}\n"
        f"training_run: {model.training_run}\n"
    )
