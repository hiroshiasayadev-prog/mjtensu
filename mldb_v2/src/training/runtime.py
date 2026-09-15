"""Backend-neutral execution boundary for one MLDB v2 training stage."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from mldb_v2.src.backend.candidate_outcome import TrainingCandidateResult
from mldb_v2.src.backend.stage_input import TrainingStage, TrainingStageInput
from mldb_v2.src.catalog.architecture import _load_architecture_definition
from mldb_v2.src.catalog.corpus import _load_corpus
from mldb_v2.src.catalog.task import _load_task
from mldb_v2.src.common.ids import _validate_trial_id, _validate_typed_reference
from mldb_v2.src.common.telemetry import _AcceptedScalarEvent, _RecordingTelemetryReporter
from mldb_v2.src.common.parameters import (
    _resolve_public_parameters,
    _validate_public_parameter_value,
    _validate_training_seed,
)
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.storage.artifact_runtime import publish_candidate_artifact_bytes
from mldb_v2.src.storage.corpus_runtime import materialize_sealed_corpus
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.training.canonical_weights import (
    CanonicalWeightsArtifactRef,
    _build_fresh_architecture_module,
    _canonicalize_trained_module_state,
    _serialize_canonical_state_dict,
    _validate_canonical_weights_artifact_ref,
)
from mldb_v2.src.training.train_interface import (
    MaterializedCorpus,
    TrainContext,
    _load_train_callable,
)
from mldb_v2.src.training.train_protocol import _load_train_protocol_definition

_STAGE_INPUT_FIELDS = {
    "schema",
    "study_result",
    "plan",
    "plan_sha256",
    "trial",
    "kind",
    "coordinate",
    "source_commit",
    "pins",
    "stage",
    "runtime_model",
}
_TRAINING_STAGE_FIELDS = {
    "task",
    "corpus",
    "architecture",
    "train_protocol",
    "parameters",
    "seed",
}
_STAGE_INPUT_SCHEMA = "mjtensu.mldb-v2/stage-input/v1"
_CANONICAL_WEIGHTS_FORMAT = "pytorch-state-dict/v1"


def _validate_training_stage_input(value: object) -> TrainingStageInput:
    if type(value) is not dict or set(value) != _STAGE_INPUT_FIELDS:
        raise ValueError("TrainingStageInput fields do not match schema")
    if value["schema"] != _STAGE_INPUT_SCHEMA:
        raise ValueError("unsupported StageInput schema")
    if value["kind"] != "training":
        raise ValueError("training runtime requires kind == 'training'")
    if value["coordinate"] is not None:
        raise ValueError("training StageInput coordinate must be null")
    if value["runtime_model"] is not None:
        raise ValueError("training StageInput runtime_model must be null")

    _validate_typed_reference(value["study_result"])
    _validate_typed_reference(value["plan"])
    if type(value["plan_sha256"]) is not str:
        raise ValueError("StageInput plan_sha256 must be a string")
    _validate_trial_id(value["trial"])
    if type(value["source_commit"]) is not str:
        raise ValueError("StageInput source_commit must be a string")
    if type(value["pins"]) is not list:
        raise ValueError("StageInput pins must be a list")

    stage = value["stage"]
    if type(stage) is not dict or set(stage) != _TRAINING_STAGE_FIELDS:
        raise ValueError("TrainingStage fields do not match schema")
    for field in ("task", "corpus", "architecture", "train_protocol"):
        _validate_typed_reference(stage[field])
    if type(stage["parameters"]) is not dict:
        raise ValueError("training parameters must be a mapping")
    _validate_public_parameter_value(stage["parameters"])
    _validate_training_seed(stage["seed"])
    return cast(TrainingStageInput, value)


def _require_sealed(label: str, definition: object) -> None:
    if type(definition) is not dict or definition.get("status") != "sealed":
        raise ValueError(f"{label} must be sealed before training execution")


def _validate_training_lineage(stage: TrainingStage, *, task, corpus, architecture, protocol) -> None:
    task_id = task["id"]
    if stage["task"] != task_id:
        raise ValueError("TrainingStage task does not match resolved Task")
    if corpus["task"] != task_id:
        raise ValueError("Corpus task does not match TrainingStage Task")
    if architecture["task"] != task_id:
        raise ValueError("Architecture task does not match TrainingStage Task")
    if protocol["task"] != task_id:
        raise ValueError("TrainProtocol task does not match TrainingStage Task")


def _resolve_complete_parameters(stage: TrainingStage, protocol) -> dict[str, object]:
    supplied = stage["parameters"]
    declarations = protocol["parameters"]
    if set(supplied) != set(declarations):
        missing = set(declarations) - set(supplied)
        unknown = set(supplied) - set(declarations)
        if missing:
            raise ValueError("training parameters are incomplete")
        if unknown:
            raise ValueError("training parameters contain unknown key")
        raise ValueError("training parameters do not match TrainProtocol declarations")
    return _resolve_public_parameters(declarations, supplied)


def _execute_training_stage(
    stage_input: TrainingStageInput,
    *,
    mldb_data_root: str | Path,
    object_bytes: _ObjectByteAccess,
    corpus_destination_root: str | Path,
    work_dir: str | Path,
    weights_uri: str,
    telemetry_sink: Callable[[_AcceptedScalarEvent], None] | None = None,
) -> TrainingCandidateResult:
    """Execute one validated training stage and return only its provisional result payload."""
    validated = _validate_training_stage_input(stage_input)
    stage = cast(TrainingStage, validated["stage"])
    root = Path(mldb_data_root)
    resolver = CanonicalRepositoryResolver(root)

    task = _load_task(resolver, stage["task"])
    corpus = _load_corpus(resolver, stage["corpus"])
    architecture = _load_architecture_definition(root, stage["architecture"])
    protocol = _load_train_protocol_definition(root, stage["train_protocol"])

    _require_sealed("Task", task)
    _require_sealed("Corpus", corpus)
    _require_sealed("Architecture", architecture)
    _require_sealed("TrainProtocol", protocol)
    _validate_training_lineage(
        stage,
        task=task,
        corpus=corpus,
        architecture=architecture,
        protocol=protocol,
    )
    parameters = _resolve_complete_parameters(stage, protocol)
    seed = _validate_training_seed(stage["seed"])

    materialized_definition, materialized_root = materialize_sealed_corpus(
        mldb_data_root=root,
        corpus_id=stage["corpus"],
        object_bytes=object_bytes,
        destination_root=corpus_destination_root,
    )
    if materialized_definition != corpus:
        raise ValueError("materialized Corpus does not match resolved Corpus")

    model = _build_fresh_architecture_module(root, stage["architecture"])
    telemetry = _RecordingTelemetryReporter(sink=telemetry_sink)
    context = TrainContext(
        task=task,
        corpus=MaterializedCorpus(
            definition=materialized_definition,
            root=Path(materialized_root),
        ),
        architecture=architecture,
        model=model,
        seed=seed,
        parameters=parameters,
        telemetry=telemetry,
        work_dir=Path(work_dir),
    )
    train = _load_train_callable(root, stage["train_protocol"])
    try:
        trained_module = train(context)
    except Exception as error:
        raise ValueError("TrainProtocol raised during training execution") from error
    if telemetry.accepted_count == 0:
        raise ValueError("TrainProtocol must emit at least one valid scalar telemetry event")

    state = _canonicalize_trained_module_state(root, stage["architecture"], trained_module)
    data = _serialize_canonical_state_dict(state)
    published = publish_candidate_artifact_bytes(
        object_bytes=object_bytes,
        uri=weights_uri,
        data=data,
    )
    weights: CanonicalWeightsArtifactRef = _validate_canonical_weights_artifact_ref(
        {
            "uri": published["uri"],
            "bytes": published["bytes"],
            "sha256": published["sha256"],
            "format": _CANONICAL_WEIGHTS_FORMAT,
        }
    )
    return {"weights": weights}
