"""Public Python signatures for MLDB Model identity and persisted metadata.

This skeleton fixes the immutable Model v1 record, the deterministic identity mapping
from one Training Run v1 ID, and metadata-local Model validation. It intentionally does
not resolve or mutate Training Runs, create or reconcile Model YAML files, calculate
repository paths, load canonical learned weights, resolve Architectures, construct
``torch.nn.Module`` values, evaluate Models, or define promotion/deployment lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..common.errors import ValidationReport
from ..common.ids import ModelId, TrainingRunId
from ..training.run import TrainingRun


@dataclass(frozen=True, slots=True)
class Model:
    """One immutable MLDB Model v1 identity record.

    Model represents the canonical learned result of exactly one completed Training
    Run without duplicating any learned bytes or Training Run-owned metadata. The
    complete persisted v1 shape is exactly ``schema``, ``id``, and ``training_run``.

    Model has no draft/sealed state, revision, quality status, promotion state,
    Architecture reference, canonical-weight metadata, execution metadata, timestamps,
    tags, or other independently mutable fields.
    """

    schema: Literal["mjtensu.mldb/model/v1"]
    id: ModelId
    training_run: TrainingRunId


def model_id_for_training_run(
    training_run_id: TrainingRunId,
) -> ModelId:
    """Derive the deterministic Model v1 identity for one Training Run v1 ID.

    For ``tr-YYYYMMDD-NNN`` the result is ``mdl-YYYYMMDD-NNN``. The date and sequence
    suffix are preserved exactly; callers must not reproduce this mapping with ad hoc
    string replacement in downstream features.

    This is a pure identity boundary. Repository lookup, Training Run status/result
    validation, Model existence checks, path calculation, persistence, and
    reconciliation are outside this function. The supplied value is expected to satisfy
    the Training Run v1 ID grammar; metadata validation surfaces grammar errors through
    :func:`validate_model_metadata` when validating a persisted Model record.
    """

    ...


def model_for_completed_training_run(
    training_run: TrainingRun,
) -> Model:
    """Construct the exact Model v1 record for an already-valid completed Training Run.

    The input precondition is a metadata-locally valid :class:`TrainingRun` whose status
    is ``COMPLETED`` and whose canonical result therefore satisfies the frozen Wave 2
    Training Run contract. Training Run lifecycle/result validation remains owned by
    ``training.run.validate_training_run`` rather than being redefined in this module.

    Successful construction is pure: it fixes the Model schema, derives ``id`` through
    :func:`model_id_for_training_run`, and copies only ``training_run.id`` as lineage.
    It does not write YAML, ensure idempotent filesystem creation, reconcile a missing
    Model, inspect another Model, or resolve canonical weights or Architecture state.
    """

    ...


def validate_model_metadata(model: Model) -> ValidationReport:
    """Validate Model v1 metadata-local invariants without external I/O.

    This validation owns rules decidable from one normalized in-memory Model record:
    the exact v1 schema identifier, Model ID grammar, Training Run ID grammar, and exact
    deterministic date/sequence suffix agreement between ``id`` and ``training_run``.

    Because this function receives the normalized three-field :class:`Model` value,
    rejection of undeclared raw YAML fields must occur during parsing/normalization
    rather than by silently discarding them before Model construction.

    This function does not resolve ``training_run``; prove that it exists, is
    ``COMPLETED``, or contains valid canonical weight result metadata; detect another
    Model referencing the same Training Run; compare a filename basename with
    ``model.id``; calculate canonical Model paths; enforce persisted-file immutability;
    write or reconcile Model YAML; verify learned-weight bytes; resolve Architecture;
    or load learned state. Those checks belong to Training Run validation,
    repository/application workflows, integrity verification, and later Model loading.
    """

    ...
