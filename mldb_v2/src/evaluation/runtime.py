"""Backend-neutral execution boundary for one MLDB v2 evaluation stage."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from mldb_v2.src.backend.candidate_outcome import EvaluationCandidateResult
from mldb_v2.src.backend.stage_input import (
    EvaluationStage,
    EvaluationStageInput,
    RuntimeModel,
)
from mldb_v2.src.catalog.architecture import _load_architecture_definition
from mldb_v2.src.catalog.corpus import _load_corpus
from mldb_v2.src.catalog.task import _load_task
from mldb_v2.src.common.ids import (
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
    _validate_typed_reference,
)
from mldb_v2.src.common.telemetry import _AcceptedScalarEvent, _RecordingTelemetryReporter
from mldb_v2.src.common.parameters import (
    _resolve_public_parameters,
    _validate_public_parameter_value,
)
from mldb_v2.src.evaluation.evaluate_interface import (
    EvaluationCandidate,
    EvaluationContext,
    LoadedModel,
    MaterializedCorpus,
    _load_evaluation_callable,
)
from mldb_v2.src.evaluation.evaluation_protocol import (
    EvaluationArtifactDeclaration,
    EvaluationMetricDeclaration,
    EvaluationProtocol,
    _load_evaluation_protocol_definition,
)
from mldb_v2.src.evaluation.evaluation_result import EvaluationArtifactRef
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.storage.artifact_reference import _validate_logical_object_uri
from mldb_v2.src.storage.artifact_runtime import publish_candidate_artifact_file
from mldb_v2.src.storage.corpus_runtime import materialize_sealed_corpus
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.training.canonical_weights import (
    _load_canonical_state_dict_bytes,
    _load_state_into_fresh_architecture,
    _validate_canonical_weights_artifact_ref,
)
from mldb_v2.src.training.model import _load_model
from mldb_v2.src.training.training_result import _load_training_result

_STAGE_INPUT_SCHEMA = "mjtensu.mldb-v2/stage-input/v1"
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
_EVALUATION_STAGE_FIELDS = {
    "name",
    "task",
    "corpus",
    "evaluation_protocol",
    "parameters",
}
_RUNTIME_MODEL_FIELDS = {
    "model",
    "training_result",
    "task",
    "architecture",
    "weights",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)


def _validate_evaluation_stage_input(value: object) -> EvaluationStageInput:
    if type(value) is not dict or set(value) != _STAGE_INPUT_FIELDS:
        raise ValueError("EvaluationStageInput fields do not match schema")
    if value["schema"] != _STAGE_INPUT_SCHEMA:
        raise ValueError("unsupported StageInput schema")
    if value["kind"] != "evaluation":
        raise ValueError("evaluation runtime requires kind == 'evaluation'")
    coordinate = _validate_evaluation_coordinate_id(value["coordinate"])
    _validate_typed_reference(value["study_result"])
    _validate_typed_reference(value["plan"])
    if type(value["plan_sha256"]) is not str or _SHA256_RE.fullmatch(value["plan_sha256"]) is None:
        raise ValueError("StageInput plan_sha256 must be a lowercase sha256")
    _validate_trial_id(value["trial"])
    if type(value["source_commit"]) is not str or _COMMIT_RE.fullmatch(value["source_commit"]) is None:
        raise ValueError("StageInput source_commit must be a full Git object id")
    if type(value["pins"]) is not list:
        raise ValueError("StageInput pins must be a list")
    if coordinate != value["coordinate"]:
        raise ValueError("invalid evaluation coordinate")

    stage = value["stage"]
    if type(stage) is not dict or set(stage) != _EVALUATION_STAGE_FIELDS:
        raise ValueError("EvaluationStage fields do not match schema")
    if type(stage["name"]) is not str or not stage["name"]:
        raise ValueError("EvaluationStage name must be a non-empty string")
    for field in ("task", "corpus", "evaluation_protocol"):
        _validate_typed_reference(stage[field])
    if type(stage["parameters"]) is not dict:
        raise ValueError("evaluation parameters must be a mapping")
    _validate_public_parameter_value(stage["parameters"])
    runtime_model = value["runtime_model"]
    if type(runtime_model) is not dict or set(runtime_model) != _RUNTIME_MODEL_FIELDS:
        raise ValueError("RuntimeModel fields do not match schema")
    for field in ("model", "training_result", "task", "architecture"):
        _validate_typed_reference(runtime_model[field])
    _validate_canonical_weights_artifact_ref(runtime_model["weights"])
    return cast(EvaluationStageInput, value)


def _require_sealed(label: str, definition: object) -> None:
    if type(definition) is not dict or definition.get("status") != "sealed":
        raise ValueError(f"{label} must be sealed before evaluation execution")


def _resolve_complete_parameters(
    stage: EvaluationStage,
    protocol: EvaluationProtocol,
) -> dict[str, object]:
    supplied = stage["parameters"]
    declarations = protocol["parameters"]
    if set(supplied) != set(declarations):
        missing = set(declarations) - set(supplied)
        unknown = set(supplied) - set(declarations)
        if missing:
            raise ValueError("evaluation parameters are incomplete")
        if unknown:
            raise ValueError("evaluation parameters contain unknown key")
        raise ValueError("evaluation parameters do not match EvaluationProtocol declarations")
    return _resolve_public_parameters(declarations, supplied)


def _resolve_runtime_lineage(
    *,
    runtime_root: Path,
    snapshot: RuntimeModel,
):
    resolver = CanonicalRepositoryResolver(runtime_root)
    model = _load_model(resolver, snapshot["model"])
    if model["training_result"] != snapshot["training_result"]:
        raise ValueError("RuntimeModel training_result does not match Model lineage")
    training_result = _load_training_result(resolver, model["training_result"])
    if training_result["status"] != "completed":
        raise ValueError("RuntimeModel requires a completed TrainingResult")
    if training_result["diagnostic"] is not None or training_result["result"] is None:
        raise ValueError("completed TrainingResult payload is inconsistent")
    if training_result["result"]["model"] != model["id"]:
        raise ValueError("TrainingResult result.model does not match Model id")
    if snapshot["model"] != model["id"]:
        raise ValueError("RuntimeModel model id does not match accepted Model")
    if snapshot["task"] != training_result["task"]:
        raise ValueError("RuntimeModel task does not match TrainingResult task")
    if snapshot["architecture"] != training_result["architecture"]:
        raise ValueError("RuntimeModel architecture does not match TrainingResult architecture")
    if snapshot["weights"] != training_result["result"]["weights"]:
        raise ValueError("RuntimeModel weights do not match accepted TrainingResult weights")
    return model, training_result


def _validate_task_compatibility(
    stage: EvaluationStage,
    runtime_model: RuntimeModel,
    *,
    task,
    corpus,
    architecture,
    protocol,
    training_result,
) -> None:
    task_id = task["id"]
    if stage["task"] != task_id:
        raise ValueError("EvaluationStage task does not match resolved Task")
    if corpus["task"] != task_id:
        raise ValueError("Corpus task does not match EvaluationStage Task")
    if protocol["task"] != task_id:
        raise ValueError("EvaluationProtocol task does not match EvaluationStage Task")
    if runtime_model["task"] != task_id:
        raise ValueError("RuntimeModel task does not match EvaluationStage Task")
    if architecture["task"] != task_id:
        raise ValueError("Architecture task does not match EvaluationStage Task")
    if training_result["task"] != task_id:
        raise ValueError("TrainingResult task does not match EvaluationStage Task")
    if architecture["id"] != runtime_model["architecture"]:
        raise ValueError("Architecture id does not match RuntimeModel lineage")
    if training_result["architecture"] != architecture["id"]:
        raise ValueError("TrainingResult architecture does not match pinned Architecture")


def _validate_candidate_metrics(
    metrics: object,
    declarations: Mapping[str, EvaluationMetricDeclaration],
) -> dict[str, int | float]:
    if not isinstance(metrics, Mapping):
        raise ValueError("EvaluationCandidate metrics must be a mapping")
    if any(type(key) is not str for key in metrics):
        raise ValueError("EvaluationCandidate metric keys must be strings")
    unknown = set(metrics) - set(declarations)
    if unknown:
        raise ValueError("EvaluationCandidate contains undeclared metric")
    missing = {key for key, declaration in declarations.items() if declaration["required"]} - set(metrics)
    if missing:
        raise ValueError("EvaluationCandidate is missing required metric")
    validated: dict[str, int | float] = {}
    for key, raw_value in metrics.items():
        declaration = declarations[key]
        if declaration["type"] == "integer":
            if type(raw_value) is not int:
                raise ValueError(f"metric {key!r} must be an exact integer")
        else:
            if type(raw_value) not in {int, float}:
                raise ValueError(f"metric {key!r} must be an exact number")
            if type(raw_value) is float and not math.isfinite(raw_value):
                raise ValueError(f"metric {key!r} must be finite")
        validated[key] = raw_value  # type: ignore[assignment]
    return validated


def _validate_candidate_artifact_path(path: object, *, work_dir: Path) -> Path:
    if not isinstance(path, Path):
        raise ValueError("EvaluationCandidate artifact path must be a Path")
    if path.is_symlink():
        raise ValueError("EvaluationCandidate artifact path must not be a symlink")
    try:
        work_root = work_dir.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("EvaluationCandidate artifact path must exist") from error
    try:
        resolved.relative_to(work_root)
    except ValueError as error:
        raise ValueError("EvaluationCandidate artifact path escapes work_dir") from error
    if not resolved.is_file():
        raise ValueError("EvaluationCandidate artifact path must be a regular file")
    return resolved


def _validate_artifact_uris(
    artifact_uris: object,
    declarations: Mapping[str, EvaluationArtifactDeclaration],
) -> Mapping[str, str]:
    if not isinstance(artifact_uris, Mapping):
        raise ValueError("artifact_uris must be a mapping")
    if any(type(key) is not str for key in artifact_uris):
        raise ValueError("artifact_uris keys must be strings")
    if set(artifact_uris) != set(declarations):
        raise ValueError("artifact_uris must contain exactly the declared artifact keys")
    for uri in artifact_uris.values():
        _validate_logical_object_uri(uri)
    return cast(Mapping[str, str], artifact_uris)


def _publish_candidate_artifacts(
    artifacts: object,
    declarations: Mapping[str, EvaluationArtifactDeclaration],
    *,
    work_dir: Path,
    artifact_uris: Mapping[str, str],
    object_bytes: _ObjectByteAccess,
) -> dict[str, EvaluationArtifactRef]:
    if not isinstance(artifacts, Mapping):
        raise ValueError("EvaluationCandidate artifacts must be a mapping")
    if any(type(key) is not str for key in artifacts):
        raise ValueError("EvaluationCandidate artifact keys must be strings")
    unknown = set(artifacts) - set(declarations)
    if unknown:
        raise ValueError("EvaluationCandidate contains undeclared artifact")
    missing = {key for key, declaration in declarations.items() if declaration["required"]} - set(artifacts)
    if missing:
        raise ValueError("EvaluationCandidate is missing required artifact")
    published: dict[str, EvaluationArtifactRef] = {}
    for key, raw_path in artifacts.items():
        path = _validate_candidate_artifact_path(raw_path, work_dir=work_dir)
        base_ref = publish_candidate_artifact_file(
            object_bytes=object_bytes,
            uri=artifact_uris[key],
            path=path,
        )
        declaration = declarations[key]
        published[key] = {
            "uri": base_ref["uri"],
            "bytes": base_ref["bytes"],
            "sha256": base_ref["sha256"],
            "format": declaration["format"],
            "schema": declaration["schema"],
        }
    return published


def _execute_evaluation_stage(
    stage_input: EvaluationStageInput,
    *,
    pinned_mldb_data_root: str | Path,
    runtime_mldb_data_root: str | Path,
    object_bytes: _ObjectByteAccess,
    corpus_destination_root: str | Path,
    work_dir: str | Path,
    artifact_uris: Mapping[str, str],
    telemetry_sink: Callable[[_AcceptedScalarEvent], None] | None = None,
) -> EvaluationCandidateResult:
    """Execute one validated evaluation stage and return its provisional result payload."""
    validated = _validate_evaluation_stage_input(stage_input)
    stage = cast(EvaluationStage, validated["stage"])
    runtime_model = cast(RuntimeModel, validated["runtime_model"])
    pinned_root = Path(pinned_mldb_data_root)
    runtime_root = Path(runtime_mldb_data_root)
    pinned_resolver = CanonicalRepositoryResolver(pinned_root)

    task = _load_task(pinned_resolver, stage["task"])
    corpus = _load_corpus(pinned_resolver, stage["corpus"])
    protocol = _load_evaluation_protocol_definition(
        pinned_root, stage["evaluation_protocol"]
    )
    model, training_result = _resolve_runtime_lineage(
        runtime_root=runtime_root,
        snapshot=runtime_model,
    )
    architecture = _load_architecture_definition(
        pinned_root, training_result["architecture"]
    )

    _require_sealed("Task", task)
    _require_sealed("Corpus", corpus)
    _require_sealed("EvaluationProtocol", protocol)
    _require_sealed("Architecture", architecture)
    _validate_task_compatibility(
        stage,
        runtime_model,
        task=task,
        corpus=corpus,
        architecture=architecture,
        protocol=protocol,
        training_result=training_result,
    )
    parameters = _resolve_complete_parameters(stage, protocol)
    validated_artifact_uris = _validate_artifact_uris(
        artifact_uris, protocol["artifacts"]
    )
    materialized_definition, materialized_root = materialize_sealed_corpus(
        mldb_data_root=pinned_root,
        corpus_id=stage["corpus"],
        object_bytes=object_bytes,
        destination_root=corpus_destination_root,
    )
    if materialized_definition != corpus:
        raise ValueError("materialized Corpus does not match resolved Corpus")

    weight_bytes = object_bytes.read_verified(runtime_model["weights"])
    state = _load_canonical_state_dict_bytes(
        weight_bytes,
        ref=runtime_model["weights"],
    )
    module = _load_state_into_fresh_architecture(
        pinned_root,
        architecture["id"],
        state,
    )
    loaded_model = LoadedModel(
        definition=model,
        training_result=training_result,
        architecture=architecture,
        module=module,
    )
    telemetry = _RecordingTelemetryReporter(sink=telemetry_sink)
    context = EvaluationContext(
        task=task,
        corpus=MaterializedCorpus(
            definition=materialized_definition,
            root=Path(materialized_root),
        ),
        model=loaded_model,
        parameters=parameters,
        telemetry=telemetry,
        work_dir=Path(work_dir),
    )
    evaluate = _load_evaluation_callable(
        pinned_root, stage["evaluation_protocol"]
    )
    try:
        candidate = evaluate(context)
    except Exception as error:
        raise ValueError("EvaluationProtocol raised during evaluation execution") from error
    if not isinstance(candidate, EvaluationCandidate):
        raise ValueError("EvaluationProtocol must return EvaluationCandidate")

    metrics = _validate_candidate_metrics(candidate.metrics, protocol["metrics"])
    for key, value in metrics.items():
        telemetry.report_scalar(
            group=stage["name"], series=key, value=value, step=0
        )
    if telemetry.accepted_count == 0:
        raise ValueError("Evaluation must establish at least one valid scalar telemetry event")

    artifacts = _publish_candidate_artifacts(
        candidate.artifacts,
        protocol["artifacts"],
        work_dir=Path(work_dir),
        artifact_uris=validated_artifact_uris,
        object_bytes=object_bytes,
    )
    return {"metrics": metrics, "artifacts": artifacts}
