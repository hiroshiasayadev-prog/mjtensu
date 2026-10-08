"""Lazy production ClearML SDK activation seam for MLDB v2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, urlsplit, urlunsplit

# ClearML executes this file directly in remote Task containers.  Workers that
# opt into the global runtime registry converge and re-exec before importing the
# rest of MLDB so the stage actually runs under the managed environment.
if __name__ == "__main__":
    from mldb_v2.src.runtime_registry import maybe_reexec_managed_worker_runtime

    maybe_reexec_managed_worker_runtime()

from mldb_v2.src.backend._clearml_admission import (
    ClearMLCreateRequest,
    ClearMLTaskRecord,
    _restore_stage_input,
)
from mldb_v2.src.backend._clearml_cancellation import (
    ClearMLCancellationState,
    ClearMLLogData,
    ClearMLOwnedTask,
)
from mldb_v2.src.backend._clearml_observation import (
    ClearMLRuntimeProjection,
    _stage_key_from_input,
)
from mldb_v2.src.backend._clearml_pipeline import (
    ClearMLPipelineCreateRequest,
    ClearMLPipelineRecord,
)
from mldb_v2.src.backend._config import BackendConfig
from mldb_v2.src.common.ids import EntityKind, _validate_typed_reference
from mldb_v2.src.evaluation.evaluation_protocol import _load_evaluation_protocol_definition
from mldb_v2.src.storage.artifact_reference import _validate_artifact_ref
from mldb_v2.src.common.telemetry import _AcceptedScalarEvent
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.runtime_registry import RuntimeRegistryClient, RuntimeRegistryError
from mldb_v2.src.study._plan_build import _validate_study_plan
from mldb_v2.src.common.display_names import validate_user_facing_display_name
from mldb_v2.src.study.display_labels import build_plan_condition_labels
from mldb_v2.src.training.model import _validate_model
from mldb_v2.src.training.training_result import _validate_training_result

_MLDB_CONFIG = "mldb"
_RUNTIME_CONFIG = "mldb.runtime"
_RUNTIME_SNAPSHOTS_CONFIG = "mldb.runtime_snapshots"
_PROJECTION_CONFIG = "mldb.runtime_projection"
_PIPELINE_CONFIG = "mldb.pipeline"
_PIPELINE_SUMMARY_CONFIG = "mldb.study_summary"
_NATIVE_PIPELINE_CONFIG = "Pipeline"
_ACTIVE_STATUSES = {"created", "queued", "in_progress", "publishing"}
_TERMINAL_STATUSES = {"completed", "published", "closed", "failed", "stopped"}
_REMOTE_PACKAGES = (
    "clearml==2.1.12",
    "boto3==1.43.93",
    "PyYAML==6.0.3",
    "numpy==1.26.4",
    "onnx==1.22.0",
    "onnxruntime==1.28.0",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "escnn==1.0.11",
)


class ClearMLSDKError(RuntimeError):
    """Bounded production ClearML SDK activation/projection failure."""


class _ClearMLScalarSink:
    """Lazy child scalar projection with optional best-effort Pipeline mirroring."""

    def __init__(
        self,
        task: object,
        *,
        pipeline_task_id: str | None = None,
        pipeline_series: str | None = None,
    ) -> None:
        self._task = task
        self._logger: object | None = None
        self._pipeline_task_id = pipeline_task_id
        self._pipeline_series = pipeline_series
        self._pipeline_logger: object | None = None

    def _parent_logger(self) -> object | None:
        if self._pipeline_task_id is None or self._pipeline_series is None:
            return None
        if self._pipeline_logger is not None:
            return self._pipeline_logger
        get_task = getattr(type(self._task), "get_task", None)
        if not callable(get_task):
            return None
        try:
            parent = get_task(task_id=self._pipeline_task_id)
            get_logger = getattr(parent, "get_logger", None)
            if not callable(get_logger):
                return None
            logger = get_logger()
        except Exception:
            return None
        self._pipeline_logger = logger
        return logger

    def __call__(self, event: _AcceptedScalarEvent) -> None:
        logger = self._logger
        if logger is None:
            get_logger = getattr(self._task, "get_logger", None)
            if not callable(get_logger):
                raise ClearMLSDKError("ClearML Task does not expose get_logger()")
            logger = get_logger()
            if logger is None:
                raise ClearMLSDKError("ClearML Task logger is unavailable")
            self._logger = logger
        report_scalar = getattr(logger, "report_scalar", None)
        if not callable(report_scalar):
            raise ClearMLSDKError("ClearML logger does not expose report_scalar()")
        report_scalar(
            title=event.group,
            series=event.series,
            value=event.value,
            iteration=event.step,
        )

        parent_logger = self._parent_logger()
        parent_report = getattr(parent_logger, "report_scalar", None)
        if not callable(parent_report):
            return
        series = (
            self._pipeline_series
            if event.series in {"Train", "Val"}
            else f"{self._pipeline_series} | {event.series}"
        )
        try:
            parent_report(
                title=event.group,
                series=series,
                value=event.value,
                iteration=event.step,
            )
        except Exception:
            # Pipeline mirroring is observational and must never affect the
            # authoritative child telemetry or training execution.
            pass


_ARTIFACT_EXTENSIONS = {
    "png": ".png",
    "csv": ".csv",
    "plotly-json": ".plotly.json",
    "json": ".json",
    "jsonl": ".jsonl",
    "html": ".html",
    "mp4": ".mp4",
}


def _artifact_projection_filename(name: str, artifact_format: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in name)
    if not safe:
        safe = "artifact"
    return safe + _ARTIFACT_EXTENSIONS.get(artifact_format, ".bin")


def _clearml_logger(task: object) -> object:
    get_logger = getattr(task, "get_logger", None)
    if not callable(get_logger):
        raise ClearMLSDKError("ClearML Task does not expose get_logger()")
    logger = get_logger()
    if logger is None:
        raise ClearMLSDKError("ClearML Task logger is unavailable")
    return logger


def _clearml_files_proxy_url(url: str) -> str:
    """Route ClearML files through the authenticated Web UI proxy when configured."""
    web_host = os.environ.get("CLEARML_WEB_HOST")
    files_host = os.environ.get("CLEARML_FILES_HOST")
    if not web_host or not files_host:
        return url
    source = urlsplit(url)
    files = urlsplit(files_host)
    if source.scheme != files.scheme or source.netloc != files.netloc:
        return url
    web = urlsplit(web_host)
    proxy_path = "/files" + source.path
    return urlunsplit((web.scheme, web.netloc, proxy_path, source.query, source.fragment))


def _project_evaluation_artifacts(
    *,
    task: object,
    candidate: Mapping[str, object],
    object_bytes: object,
    projection_root: Path,
) -> None:
    """Best-effort ClearML UI projection of canonical evaluation artifacts."""
    if candidate.get("status") != "completed":
        return
    result = candidate.get("result")
    if type(result) is not dict:
        return
    artifacts = result.get("artifacts")
    if type(artifacts) is not dict or not artifacts:
        return

    projection_root.mkdir(parents=True, exist_ok=True)
    logger: object | None = None
    for name, raw_ref in artifacts.items():
        if type(name) is not str or type(raw_ref) is not dict:
            continue
        try:
            read_verified = getattr(object_bytes, "read_verified", None)
            if not callable(read_verified):
                raise ClearMLSDKError("object-byte access does not expose read_verified()")
            data = read_verified(raw_ref)
            artifact_format = str(raw_ref.get("format") or "opaque")
            path = projection_root / _artifact_projection_filename(name, artifact_format)
            path.write_bytes(data)
            upload_artifact = getattr(task, "upload_artifact", None)
            if not callable(upload_artifact):
                raise ClearMLSDKError("ClearML Task does not expose upload_artifact()")
            artifact_name = f"evaluation/{name}"
            upload_ok = upload_artifact(
                name=artifact_name,
                artifact_object=str(path),
                metadata={
                    "canonical_uri": str(raw_ref.get("uri") or ""),
                    "sha256": str(raw_ref.get("sha256") or ""),
                    "format": artifact_format,
                    "schema": str(raw_ref.get("schema") or ""),
                },
                wait_on_upload=artifact_format == "mp4",
            )
            if upload_ok is False:
                raise ClearMLSDKError(f"ClearML artifact upload failed for {artifact_name!r}")
            if artifact_format not in {"png", "csv", "plotly-json", "mp4"}:
                continue
            if logger is None:
                logger = _clearml_logger(task)
            if artifact_format == "png":
                report_image = getattr(logger, "report_image", None)
                if not callable(report_image):
                    raise ClearMLSDKError("ClearML logger does not expose report_image()")
                report_image(title="evaluation artifacts", series=name, iteration=0, local_path=str(path))
            elif artifact_format == "csv":
                report_table = getattr(logger, "report_table", None)
                if not callable(report_table):
                    raise ClearMLSDKError("ClearML logger does not expose report_table()")
                report_table(title="evaluation tables", series=name, iteration=0, csv=str(path))
            elif artifact_format == "plotly-json":
                report_plotly = getattr(logger, "report_plotly", None)
                if not callable(report_plotly):
                    raise ClearMLSDKError("ClearML logger does not expose report_plotly()")
                report_plotly(
                    title="evaluation plots",
                    series=name,
                    iteration=0,
                    figure=json.loads(data.decode("utf-8")),
                )
            else:
                report_media = getattr(logger, "report_media", None)
                if not callable(report_media):
                    raise ClearMLSDKError("ClearML logger does not expose report_media()")
                task_artifacts = getattr(task, "artifacts", None)
                projected = task_artifacts.get(artifact_name) if isinstance(task_artifacts, Mapping) else None
                media_url = getattr(projected, "url", None)
                if type(media_url) is not str or not media_url:
                    raise ClearMLSDKError(f"ClearML uploaded artifact URL unavailable for {artifact_name!r}")
                report_media(
                    title="evaluation media",
                    series=name,
                    iteration=0,
                    url=_clearml_files_proxy_url(media_url),
                )
        except Exception as exc:
            print(f"MLDB ClearML artifact projection skipped {name!r}: {exc}")


@dataclass(frozen=True)
class ClearMLSDKSettings:
    api_host: str | None = None
    web_host: str | None = None
    files_host: str | None = None
    access_key: str | None = field(default=None, repr=False)
    secret_key: str | None = field(default=None, repr=False)
    repository: str | None = None
    local_repository_root: str | None = None
    docker_image: str | None = None
    docker_env_file: str | None = None
    docker_gpu: str | None = None
    docker_shm_size: str | None = None
    runtime_registry_url: str | None = None
    runtime_registry_ca_bundle: str | None = None
    runtime_image_profile: str | None = None
    runtime_image_wait_seconds: int = 1800
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    script: str = "mldb_v2/src/backend/_clearml_sdk.py"
    pipeline_script: str = "mldb_v2/src/backend/_clearml_pipeline_controller.py"
    working_directory: str = "."
    pinned_data_root: str = "mldb_data"
    runtime_data_root: str | None = None
    artifact_uri_prefix: str | None = None
    step_queue: str | None = None
    stage_routes: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    prebuilt_runtime: bool = False
    work_root: str = ".mldb-v2-clearml"
    log_reports: int = 20

    def __post_init__(self) -> None:
        for name in (
            "api_host", "web_host", "files_host", "repository", "local_repository_root",
            "docker_image", "docker_env_file", "docker_gpu", "docker_shm_size",
            "runtime_registry_url", "runtime_registry_ca_bundle", "runtime_image_profile",
            "s3_endpoint_url", "s3_region", "runtime_data_root", "artifact_uri_prefix", "step_queue",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value or value.strip() != value):
                raise ValueError(f"{name} must be null or a non-empty trimmed string")
        for name in ("script", "pipeline_script", "working_directory", "pinned_data_root", "work_root"):
            value = getattr(self, name)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty trimmed string")
        if (self.access_key is None) != (self.secret_key is None):
            raise ValueError("ClearML access_key and secret_key must be supplied together")
        if type(self.prebuilt_runtime) is not bool:
            raise ValueError("prebuilt_runtime must be boolean")
        if type(self.runtime_image_wait_seconds) is not int or self.runtime_image_wait_seconds < 0:
            raise ValueError("runtime_image_wait_seconds must be a non-negative integer")
        if self.runtime_image_profile is not None and self.runtime_registry_url is None:
            raise ValueError("runtime image profile requires runtime_registry_url")
        for stage, route in self.stage_routes.items():
            if type(stage) is not str or not stage or stage.strip() != stage or not isinstance(route, Mapping):
                raise ValueError("stage_routes entries must use non-empty stage names and mappings")
            unknown = set(route) - {"queue", "docker_gpu", "runtime_image_profile"}
            if unknown:
                raise ValueError(f"stage route {stage!r} has unknown fields: {sorted(unknown)!r}")
            if "queue" in route:
                queue = route["queue"]
                if type(queue) is not str or not queue or queue.strip() != queue:
                    raise ValueError(f"stage route {stage!r} queue must be a non-empty trimmed string")
            if "docker_gpu" in route:
                docker_gpu = route["docker_gpu"]
                if docker_gpu is not None and (
                    type(docker_gpu) is not str or not docker_gpu or docker_gpu.strip() != docker_gpu
                ):
                    raise ValueError(f"stage route {stage!r} docker_gpu must be null or a non-empty trimmed string")
            if "runtime_image_profile" in route:
                runtime_image_profile = route["runtime_image_profile"]
                if runtime_image_profile is not None and (
                    type(runtime_image_profile) is not str
                    or not runtime_image_profile
                    or runtime_image_profile.strip() != runtime_image_profile
                ):
                    raise ValueError(
                        f"stage route {stage!r} runtime_image_profile must be null "
                        "or a non-empty trimmed string"
                    )
        if type(self.log_reports) is not int or self.log_reports < 1:
            raise ValueError("log_reports must be a positive integer")


def _optional_string(options: Mapping[str, object], name: str) -> str | None:
    value = options.get(name)
    if value is None:
        return None
    if type(value) is not str or not value or value.strip() != value:
        raise ClearMLSDKError(f"ClearML option {name!r} must be a non-empty trimmed string")
    return value


def _string_option(options: Mapping[str, object], name: str, default: str) -> str:
    value = _optional_string(options, name)
    return default if value is None else value


def _stage_routes_option(options: Mapping[str, object]) -> dict[str, dict[str, object]]:
    value = options.get("stage_routes", {})
    if type(value) is not dict:
        raise ClearMLSDKError("ClearML option 'stage_routes' must be a mapping")
    routes: dict[str, dict[str, object]] = {}
    for stage, route in value.items():
        if type(stage) is not str or not stage or type(route) is not dict:
            raise ClearMLSDKError("ClearML stage_routes entries are malformed")
        routes[stage] = dict(route)
    return routes


def _stage_name_from_input(stage_input: Mapping[str, object]) -> str:
    if stage_input.get("kind") == "training":
        return "training"
    stage = stage_input.get("stage")
    if not isinstance(stage, Mapping):
        raise ClearMLSDKError("ClearML StageInput stage is malformed")
    name = stage.get("name")
    if type(name) is not str or not name:
        raise ClearMLSDKError("ClearML Evaluation StageInput stage name is malformed")
    return name


def _stage_route(
    stage_routes: Mapping[str, Mapping[str, object]], stage: str
) -> Mapping[str, object]:
    route = stage_routes.get(stage)
    return {} if route is None else route


def _local_runtime_root(settings: ClearMLSDKSettings) -> Path | None:
    if settings.local_repository_root is None:
        return None
    repository_root = Path(settings.local_repository_root).resolve()
    configured = settings.runtime_data_root or "mldb_data"
    path = Path(configured)
    return path if path.is_absolute() else repository_root / path


def _local_pinned_data_root(settings: ClearMLSDKSettings) -> Path | None:
    if settings.local_repository_root is None:
        return None
    repository_root = Path(settings.local_repository_root).resolve()
    path = Path(settings.pinned_data_root)
    return path if path.is_absolute() else repository_root / path


def _local_reference_name(reference: object) -> str:
    text = str(reference)
    return text.split("/", 1)[1] if "/" in text else text


def _pipeline_trial_labels(
    settings: ClearMLSDKSettings,
    plan: Mapping[str, object],
) -> dict[str, str]:
    root = _local_pinned_data_root(settings)
    resolver = CanonicalRepositoryResolver(root) if root is not None else None
    try:
        return build_plan_condition_labels(plan=plan, resolver=resolver)
    except ValueError as error:
        raise ClearMLSDKError(str(error)) from error


def _onnx_source_id_for_stage(
    pinned_root: Path, stage_input: Mapping[str, object]
) -> str | None:
    if stage_input["kind"] != "evaluation":
        return None
    stage = cast(Mapping[str, object], stage_input["stage"])
    protocol = _load_evaluation_protocol_definition(
        pinned_root, cast(str, stage["evaluation_protocol"])
    )
    parameter = protocol.get("onnx_input_parameter")
    if parameter is None:
        return None
    parameters = cast(Mapping[str, object], stage["parameters"])
    return _validate_typed_reference(parameters[parameter])


def _validate_onnx_source_snapshot(
    source: Mapping[str, object], *, source_id: str, model_id: str, task_id: str
) -> None:
    if (
        source.get("schema") != "mjtensu.mldb-v2/evaluation-result/v1"
        or source.get("id") != source_id
        or source.get("status") != "completed"
        or source.get("diagnostic") is not None
        or source.get("model") != model_id
        or source.get("task") != task_id
    ):
        raise ClearMLSDKError("ONNX source EvaluationResult identity/status/lineage mismatch")
    payload = source.get("result")
    if not isinstance(payload, Mapping):
        raise ClearMLSDKError("ONNX source EvaluationResult has no result")
    artifacts = payload.get("artifacts")
    ref = artifacts.get("onnx_model") if isinstance(artifacts, Mapping) else None
    if not isinstance(ref, dict) or ref.get("format") != "onnx":
        raise ClearMLSDKError("ONNX source EvaluationResult has no formal ONNX artifact")
    _validate_artifact_ref(ref)


def _dependency_sources(stage_input: Mapping[str, object]) -> dict[str, dict[str, object]]:
    if stage_input["kind"] != "evaluation":
        return {}
    stage = cast(Mapping[str, object], stage_input["stage"])
    raw = stage.get("inputs", {})
    if type(raw) is not dict:
        raise ClearMLSDKError("EvaluationStage dependency input mapping is malformed")
    sources: dict[str, dict[str, object]] = {}
    for alias, entry in raw.items():
        if (
            type(alias) is not str or not alias
            or type(entry) is not dict
            or set(entry) != {"source_evaluation_result", "artifact", "ref"}
        ):
            raise ClearMLSDKError("EvaluationStage dependency input fields are malformed")
        source_id = _validate_typed_reference(entry["source_evaluation_result"])
        _validate_artifact_ref(entry["ref"])
        if type(entry["artifact"]) is not str or not entry["artifact"]:
            raise ClearMLSDKError("EvaluationStage dependency artifact is malformed")
        sources[alias] = entry
    return sources


def _validate_dependency_snapshot(
    source: Mapping[str, object], *, entry: Mapping[str, object],
    model_id: str, task_id: str,
) -> None:
    source_id = entry["source_evaluation_result"]
    if (
        source.get("schema") != "mjtensu.mldb-v2/evaluation-result/v1"
        or source.get("id") != source_id
        or source.get("status") != "completed"
        or source.get("diagnostic") is not None
        or source.get("model") != model_id
        or source.get("task") != task_id
    ):
        raise ClearMLSDKError("dependency source EvaluationResult is not accepted for same Model/Task")
    payload = source.get("result")
    artifacts = payload.get("artifacts") if isinstance(payload, Mapping) else None
    actual = artifacts.get(entry["artifact"]) if isinstance(artifacts, Mapping) else None
    if type(actual) is not dict or actual != entry["ref"]:
        raise ClearMLSDKError("dependency source artifact does not match canonical EvaluationResult")
    _validate_artifact_ref(actual)


def _build_runtime_snapshots(
    settings: ClearMLSDKSettings, stage_input: Mapping[str, object]
) -> dict[str, object] | None:
    root = _local_runtime_root(settings)
    if root is None:
        return None
    resolver = CanonicalRepositoryResolver(root)
    plan_id = cast(str, stage_input["plan"])
    plan = _validate_study_plan(
        resolver.resolve(kind=EntityKind.STUDY_PLAN, entity_id=plan_id)
    )
    if plan["id"] != plan_id:
        raise ClearMLSDKError("runtime StudyPlan id does not match StageInput")
    if plan["content_sha256"] != stage_input["plan_sha256"]:
        raise ClearMLSDKError("runtime StudyPlan digest does not match StageInput")
    if plan["source_commit"] != stage_input["source_commit"]:
        raise ClearMLSDKError("runtime StudyPlan source_commit does not match StageInput")
    snapshots: dict[str, object] = {"study_plan": dict(plan)}
    if stage_input["kind"] == "evaluation":
        runtime_model = stage_input["runtime_model"]
        if type(runtime_model) is not dict:
            raise ClearMLSDKError("evaluation StageInput runtime_model is missing")
        model_id = cast(str, runtime_model["model"])
        training_result_id = cast(str, runtime_model["training_result"])
        model = _validate_model(
            dict(resolver.resolve(kind=EntityKind.MODEL, entity_id=model_id)),
            expected_id=model_id,
        )
        training_result = _validate_training_result(
            dict(
                resolver.resolve(
                    kind=EntityKind.TRAINING_RESULT, entity_id=training_result_id
                )
            ),
            expected_id=training_result_id,
        )
        if model["training_result"] != training_result_id:
            raise ClearMLSDKError("runtime Model training_result does not match StageInput")
        if training_result["status"] != "completed" or training_result["result"] is None:
            raise ClearMLSDKError("runtime TrainingResult is not completed")
        if training_result["result"]["model"] != model_id:
            raise ClearMLSDKError("runtime TrainingResult model does not match StageInput")
        if training_result["result"]["weights"] != runtime_model["weights"]:
            raise ClearMLSDKError("runtime TrainingResult weights do not match StageInput")
        if training_result["task"] != runtime_model["task"]:
            raise ClearMLSDKError("runtime TrainingResult task does not match StageInput")
        if training_result["architecture"] != runtime_model["architecture"]:
            raise ClearMLSDKError("runtime TrainingResult architecture does not match StageInput")
        snapshots["training_result"] = dict(training_result)
        snapshots["model"] = dict(model)
        pinned_root = _local_pinned_data_root(settings)
        if pinned_root is None:
            raise ClearMLSDKError("pinned Evaluation Protocol root is missing")
        source_id = _onnx_source_id_for_stage(pinned_root, stage_input)
        if source_id is not None:
            source = dict(resolver.resolve(
                kind=EntityKind.EVALUATION_RESULT, entity_id=source_id
            ))
            _validate_onnx_source_snapshot(
                source, source_id=source_id, model_id=model_id,
                task_id=cast(str, runtime_model["task"]),
            )
            snapshots["onnx_source_evaluation_result"] = source
        deps = _dependency_sources(stage_input)
        if deps:
            records: dict[str, dict[str, object]] = {}
            for entry in deps.values():
                key = entry["source_evaluation_result"]
                source = records.get(key)
                if source is None:
                    source = dict(resolver.resolve(
                        kind=EntityKind.EVALUATION_RESULT, entity_id=key
                    ))
                    records[key] = source
                _validate_dependency_snapshot(
                    source, entry=entry, model_id=model_id,
                    task_id=cast(str, runtime_model["task"])
                )
            snapshots["artifact_source_evaluation_results"] = records
    return snapshots


def _snapshot_path(root: Path, *, domain: str, entity_id: str) -> Path:
    namespace, local_id = entity_id.split("/", 1)
    return root / namespace / domain / f"{local_id}.yaml"


def _write_runtime_snapshot(path: Path, document: Mapping[str, object]) -> None:
    record = dict(document)
    payload = (
        json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ClearMLSDKError("existing runtime snapshot is malformed") from error
        if existing != record:
            raise ClearMLSDKError("existing runtime snapshot conflicts with Task snapshot")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _materialize_runtime_snapshots(
    *,
    runtime_root: Path,
    pinned_root: Path,
    stage_input: Mapping[str, object],
    snapshots: Mapping[str, object],
) -> None:
    expected = {"study_plan"}
    source_id = _onnx_source_id_for_stage(pinned_root, stage_input)
    if stage_input["kind"] == "evaluation":
        expected |= {"training_result", "model"}
    if source_id is not None:
        expected.add("onnx_source_evaluation_result")
    deps = _dependency_sources(stage_input)
    if deps:
        expected.add("artifact_source_evaluation_results")
    if set(snapshots) != expected:
        raise ClearMLSDKError("runtime snapshot bundle fields do not match stage kind")

    plan_raw = snapshots["study_plan"]
    if not isinstance(plan_raw, Mapping):
        raise ClearMLSDKError("runtime StudyPlan snapshot is malformed")
    plan = _validate_study_plan(plan_raw)
    if (
        plan["id"] != stage_input["plan"]
        or plan["content_sha256"] != stage_input["plan_sha256"]
        or plan["source_commit"] != stage_input["source_commit"]
    ):
        raise ClearMLSDKError("runtime StudyPlan snapshot does not match StageInput")

    namespace = cast(str, stage_input["study_result"]).split("/", 1)[0]
    pinned_namespace = pinned_root / namespace / "namespace.yaml"
    runtime_namespace = runtime_root / namespace / "namespace.yaml"
    if not runtime_namespace.exists():
        if not pinned_namespace.is_file():
            raise ClearMLSDKError("pinned namespace is missing for runtime snapshots")
        runtime_namespace.parent.mkdir(parents=True, exist_ok=True)
        runtime_namespace.write_bytes(pinned_namespace.read_bytes())
    _write_runtime_snapshot(
        _snapshot_path(runtime_root, domain="study_plans", entity_id=cast(str, plan["id"])),
        plan,
    )

    if stage_input["kind"] != "evaluation":
        return
    runtime_model = stage_input["runtime_model"]
    if type(runtime_model) is not dict:
        raise ClearMLSDKError("evaluation StageInput runtime_model is missing")
    training_raw = snapshots["training_result"]
    model_raw = snapshots["model"]
    training = _validate_training_result(
        dict(cast(Mapping[str, object], training_raw)),
        expected_id=cast(str, runtime_model["training_result"]),
    )
    model = _validate_model(
        dict(cast(Mapping[str, object], model_raw)),
        expected_id=cast(str, runtime_model["model"]),
    )
    if model["training_result"] != training["id"]:
        raise ClearMLSDKError("runtime Model snapshot lineage is inconsistent")
    if training["status"] != "completed" or training["result"] is None:
        raise ClearMLSDKError("runtime TrainingResult snapshot is not completed")
    if training["result"]["model"] != model["id"]:
        raise ClearMLSDKError("runtime TrainingResult snapshot model is inconsistent")
    if training["result"]["weights"] != runtime_model["weights"]:
        raise ClearMLSDKError("runtime TrainingResult snapshot weights are inconsistent")
    _write_runtime_snapshot(
        _snapshot_path(
            runtime_root, domain="training_results", entity_id=cast(str, training["id"])
        ),
        training,
    )
    _write_runtime_snapshot(
        _snapshot_path(runtime_root, domain="models", entity_id=cast(str, model["id"])),
        model,
    )
    if source_id is not None:
        source = cast(Mapping[str, object], snapshots["onnx_source_evaluation_result"])
        _validate_onnx_source_snapshot(
            source, source_id=source_id, model_id=cast(str, model["id"]),
            task_id=cast(str, runtime_model["task"]),
        )
        if source_id.split("/", 1)[0] != cast(str, model["id"]).split("/", 1)[0]:
            raise ClearMLSDKError("ONNX source and Model must share a namespace")
        _write_runtime_snapshot(
            _snapshot_path(runtime_root, domain="evaluation_results", entity_id=source_id),
            source,
        )
    if deps:
        records = snapshots["artifact_source_evaluation_results"]
        if type(records) is not dict or set(records) != {
            entry["source_evaluation_result"] for entry in deps.values()
        }:
            raise ClearMLSDKError("artifact source snapshots do not match StageInput")
        for entry in deps.values():
            source = records[entry["source_evaluation_result"]]
            if not isinstance(source, Mapping):
                raise ClearMLSDKError("artifact source snapshot is malformed")
            _validate_dependency_snapshot(
                source, entry=entry, model_id=cast(str, model["id"]),
                task_id=cast(str, runtime_model["task"]),
            )
            _write_runtime_snapshot(
                _snapshot_path(
                    runtime_root, domain="evaluation_results",
                    entity_id=entry["source_evaluation_result"]
                ),
                source,
            )


def _load_task_class() -> Any:
    try:
        from clearml import Task  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise ClearMLSDKError(
            "clearml package is required to activate the production ClearML backend"
        ) from error
    return Task


def _load_pipeline_controller_class() -> Any:
    try:
        from clearml.automation import PipelineController  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise ClearMLSDKError(
            "clearml package is required to activate the production ClearML backend"
        ) from error
    return PipelineController


def _task_id(task: object) -> str:
    value = getattr(task, "id", None) or getattr(task, "task_id", None)
    if type(value) is not str or not value:
        raise ClearMLSDKError("ClearML Task object has no non-empty opaque ID")
    return value


def _search_tag(key: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"mldb-v2-search:{key}:{digest}"


def _metadata(task: object) -> dict[str, str]:
    getter = getattr(task, "get_user_properties", None)
    if not callable(getter):
        raise ClearMLSDKError("ClearML Task does not expose user properties")
    raw = getter(value_only=True)
    if not isinstance(raw, Mapping):
        raise ClearMLSDKError("ClearML user properties projection is malformed")
    result: dict[str, str] = {}
    for key, value in raw.items():
        if type(key) is str and type(value) is str and key.startswith("mldb."):
            result[key] = value
    return result


def _configuration(task: object, name: str) -> dict[str, object] | None:
    getter = getattr(task, "get_configuration_object_as_dict", None)
    if not callable(getter):
        raise ClearMLSDKError("ClearML Task does not expose configuration objects")
    raw = getter(name)
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or any(type(key) is not str for key in raw):
        raise ClearMLSDKError(f"ClearML configuration {name!r} is malformed")
    try:
        normalized = json.loads(
            json.dumps(raw, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ClearMLSDKError(
            f"ClearML configuration {name!r} is not canonical JSON-compatible"
        ) from error
    if type(normalized) is not dict:
        raise ClearMLSDKError(f"ClearML configuration {name!r} is malformed")
    return normalized


def _set_native_pipeline_configuration(
    task: object, dag: Mapping[str, object]
) -> None:
    payload = json.dumps(
        dict(dag),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    native_setter = getattr(task, "_set_configuration", None)
    if callable(native_setter):
        native_setter(
            name=_NATIVE_PIPELINE_CONFIG,
            config_type="dictionary",
            config_text=payload,
        )
        return
    setter = getattr(task, "set_configuration_object", None)
    if not callable(setter):
        raise ClearMLSDKError("ClearML Task does not expose configuration mutation")
    setter(name=_NATIVE_PIPELINE_CONFIG, config_dict=dict(dag))


def _project_name(task: object) -> str | None:
    getter = getattr(task, "get_project_name", None)
    value = getter() if callable(getter) else None
    return value if type(value) is str else None


def _pipeline_task_from_plan(plan: Mapping[str, object]) -> str:
    pins = plan.get("pins")
    if type(pins) is not list:
        raise ClearMLSDKError("ClearML Pipeline plan pins are missing")
    task_ids = [
        pin.get("id")
        for pin in pins
        if isinstance(pin, Mapping) and pin.get("kind") == "task"
    ]
    if len(task_ids) != 1 or type(task_ids[0]) is not str or "/" not in task_ids[0]:
        raise ClearMLSDKError("ClearML Pipeline plan must pin exactly one MLDB Task")
    return cast(str, task_ids[0])


def _native_pipeline_projects(*, logical_project: str, task: str) -> tuple[str, str]:
    if type(logical_project) is not str or not logical_project:
        raise ValueError("logical_project must be a non-empty string")
    if type(task) is not str or "/" not in task:
        raise ValueError("task must be one typed reference")
    task_name = task.split("/", 1)[1]
    parent = f"{logical_project}/.pipelines"
    return parent, f"{parent}/{task_name}"


def _is_pipeline_project(*, project: str | None, logical_project: str) -> bool:
    return project == logical_project or (
        type(project) is str and project.startswith(f"{logical_project}/.pipelines/")
    )


def _status(task: object) -> str:
    getter = getattr(task, "get_status", None)
    value = getter() if callable(getter) else None
    if type(value) is not str or not value:
        raise ClearMLSDKError("ClearML Task status is unavailable")
    return value


def _cancellation_state(status: str) -> ClearMLCancellationState:
    if status in _ACTIVE_STATUSES:
        return ClearMLCancellationState.ACTIVE
    if status == "stopped":
        return ClearMLCancellationState.CANCELLED
    if status in _TERMINAL_STATUSES:
        return ClearMLCancellationState.TERMINAL
    return ClearMLCancellationState.TERMINAL


def _native_pipeline_dag(
    topology: Mapping[str, object],
    *,
    queue: str | None,
    stage_routes: Mapping[str, Mapping[str, object]] | None = None,
    trial_labels: Mapping[str, str] | None = None,
) -> dict[str, object]:
    raw_steps = topology.get("steps")
    if type(raw_steps) is not list:
        raise ClearMLSDKError("MLDB Pipeline topology steps are malformed")
    labels = dict(trial_labels or {})
    routes = stage_routes or {}
    prepared: list[dict[str, object]] = []
    display_names: dict[str, str] = {}
    used_display_names: set[str] = set()
    for raw in raw_steps:
        if type(raw) is not dict:
            raise ClearMLSDKError("MLDB Pipeline topology step is malformed")
        name = raw.get("name")
        parents = raw.get("parents")
        stage = raw.get("stage")
        trial = raw.get("trial")
        kind = raw.get("kind")
        coordinate = raw.get("coordinate")
        if (
            type(name) is not str
            or not name
            or type(parents) is not list
            or type(stage) is not str
            or type(trial) is not str
            or type(kind) is not str
        ):
            raise ClearMLSDKError("MLDB Pipeline topology step fields are malformed")
        suffix = "training" if kind == "training" else stage
        trial_label = labels.get(trial)
        if type(trial_label) is not str:
            raise ClearMLSDKError("ClearML Pipeline trial is missing a user-facing condition label")
        display = f"{trial_label} | {suffix}"
        if display in used_display_names:
            extra = coordinate if type(coordinate) is str else name
            display = f"{display} | {extra}"
        if display in used_display_names:
            raise ClearMLSDKError("ClearML Pipeline node display names are not unique")
        try:
            validate_user_facing_display_name(display, field="ClearML Pipeline node display name")
        except ValueError as error:
            raise ClearMLSDKError(str(error)) from error
        used_display_names.add(display)
        display_names[name] = display
        prepared.append(raw)

    Node = _load_pipeline_controller_class().Node
    dag: dict[str, object] = {}
    for raw in prepared:
        logical_name = cast(str, raw["name"])
        parents = cast(list[object], raw["parents"])
        stage = cast(str, raw["stage"])
        trial = cast(str, raw["trial"])
        display_name = display_names[logical_name]
        translated_parents = [display_names[str(parent)] for parent in parents]
        route = _stage_route(routes, stage)
        node_queue = route.get("queue", queue)
        if node_queue is not None and type(node_queue) is not str:
            raise ClearMLSDKError("ClearML stage route queue is malformed")
        node = Node(
            name=display_name,
            parents=translated_parents,
            queue=cast(str | None, node_queue),
            cache_executed_step=False,
            stage=stage,
        )
        serialized = {
            key: value
            for key, value in node.__dict__.items()
            if key not in {"job", "name", "task_factory_func"}
        }
        serialized["job_id"] = node.executed or (node.job.task_id() if node.job else None)
        serialized["mldb.logical_step"] = logical_name
        serialized["mldb.trial"] = trial
        dag[display_name] = serialized
    return dag


def _native_pipeline_node(
    native: Mapping[str, object], logical_step: str
) -> dict[str, object] | None:
    direct = native.get(logical_step)
    if type(direct) is dict:
        return direct
    matches = [
        raw
        for raw in native.values()
        if type(raw) is dict and raw.get("mldb.logical_step") == logical_step
    ]
    if len(matches) > 1:
        raise ClearMLSDKError("ClearML Pipeline logical step is ambiguous")
    return matches[0] if matches else None


def _native_pipeline_node_display_name(
    native: Mapping[str, object], logical_step: str
) -> str | None:
    matches = [
        name
        for name, raw in native.items()
        if type(name) is str
        and type(raw) is dict
        and raw.get("mldb.logical_step") == logical_step
    ]
    if len(matches) > 1:
        raise ClearMLSDKError("ClearML Pipeline logical step display name is ambiguous")
    if not matches:
        return None
    try:
        return validate_user_facing_display_name(
            matches[0], field="ClearML Pipeline node display name"
        )
    except ValueError as error:
        raise ClearMLSDKError(str(error)) from error


_COMPARISON_RANK_PALETTE = ("#2F6FED", "#9CC7FF", "#F6B0B0", "#D9534F")
_COMPARISON_MISSING_COLOR = "rgba(0,0,0,0)"


def _comparison_rank_colors(
    values: Sequence[float | None], *, preference: str
) -> list[str] | None:
    if preference not in {"higher", "lower"}:
        return None
    finite = sorted(
        {value for value in values if value is not None and math.isfinite(value)},
        reverse=preference == "higher",
    )
    if not finite:
        return None
    if len(finite) == 1:
        by_value = {finite[0]: _COMPARISON_RANK_PALETTE[0]}
    else:
        by_value: dict[float, str] = {}
        last_palette = len(_COMPARISON_RANK_PALETTE) - 1
        last_rank = len(finite) - 1
        for rank, value in enumerate(finite):
            palette_index = int(math.floor((rank * last_palette / last_rank) + 0.5))
            by_value[value] = _COMPARISON_RANK_PALETTE[palette_index]
    return [
        _COMPARISON_MISSING_COLOR if value is None else by_value[value]
        for value in values
    ]


def _study_comparison_projection_iteration(task: object) -> int:
    getter = getattr(task, "get_reported_plots", None)
    if not callable(getter):
        return 0
    try:
        plots = getter() or []
    except Exception:
        return 0
    latest = -1
    for plot in plots:
        if not isinstance(plot, Mapping):
            continue
        metric = plot.get("metric")
        if type(metric) is not str:
            continue
        if not (
            metric == "Study Comparison"
            or metric == "Parameter Sweep"
            or metric.startswith("Model Comparison - ")
            or metric.startswith("Parameter Sweep - ")
        ):
            continue
        raw_iteration = plot.get("iter", 0)
        if type(raw_iteration) is int and raw_iteration >= 0:
            latest = max(latest, raw_iteration)
    return latest + 1 if latest >= 0 else 0


def _comparison_parameter_value(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if type(value) is float:
        return f"{value:.12g}"
    return str(value)


def _comparison_varying_parameter_names(
    rows: Sequence[Mapping[str, object]],
    *,
    field: str,
) -> list[str]:
    if len(rows) < 2:
        return []
    mappings: list[Mapping[str, object]] = []
    for row in rows:
        value = row.get(field)
        mappings.append(value if isinstance(value, Mapping) else {})
    keys = sorted(
        {
            key
            for parameters in mappings
            for key in parameters
            if type(key) is str
        }
    )
    varying: list[str] = []
    missing = object()
    for key in keys:
        values = [parameters.get(key, missing) for parameters in mappings]
        serialized = {
            "<missing>"
            if value is missing
            else json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            for value in values
        }
        if len(serialized) > 1:
            varying.append(key)
    return varying


def _comparison_parameter_fragment(
    row: Mapping[str, object],
    *,
    field: str,
    names: Sequence[str],
) -> str | None:
    parameters = row.get(field)
    if not isinstance(parameters, Mapping):
        return None
    parts = [
        f"{name}={_comparison_parameter_value(parameters[name])}"
        for name in names
        if name in parameters
    ]
    return ", ".join(parts) or None


def _comparison_base_label(row: Mapping[str, object]) -> str:
    raw = row.get("trial_label")
    try:
        return validate_user_facing_display_name(raw, field="Study comparison condition label")
    except ValueError as error:
        raise ClearMLSDKError(str(error)) from error


def _comparison_series_labels(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return [_comparison_base_label(row) for row in rows]


def _comparison_row_labels(rows: Sequence[Mapping[str, object]]) -> list[str]:
    series_labels = _comparison_series_labels(rows)
    evaluation_names = _comparison_varying_parameter_names(rows, field="parameters")
    evaluation_conditions = [
        _comparison_parameter_fragment(row, field="parameters", names=evaluation_names)
        for row in rows
    ]
    distinct_series = {label for label in series_labels if label}
    labels: list[str] = []
    for series, condition in zip(series_labels, evaluation_conditions, strict=True):
        if condition is None:
            labels.append(series)
        elif len(distinct_series) <= 1:
            labels.append(condition)
        elif series:
            labels.append(f"{series} · {condition}")
        else:
            labels.append(condition)
    return labels


def _comparison_varying_numeric_parameter(
    rows: Sequence[Mapping[str, object]],
) -> str | None:
    varying = _comparison_varying_parameter_names(rows, field="parameters")
    if len(varying) != 1:
        return None
    axis = varying[0]
    for row in rows:
        parameters = row.get("parameters")
        if not isinstance(parameters, Mapping):
            return None
        value = parameters.get(axis)
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            return None
    return axis


def _comparison_sweep_figure(
    *,
    stage: str,
    evaluation_name: str | None,
    metric_name: str,
    metric_preference: str,
    parameter_name: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if type(metric_name) is not str or not metric_name:
        return None
    by_series: dict[str, list[tuple[float, float]]] = {}
    series_labels = _comparison_series_labels(rows)
    for row, series_label in zip(rows, series_labels, strict=True):
        parameters = row.get("parameters")
        metrics = row.get("metrics")
        if not series_label or not isinstance(parameters, Mapping) or not isinstance(metrics, Mapping):
            continue
        raw_x = parameters.get(parameter_name)
        raw_y = metrics.get(metric_name)
        if type(raw_x) not in {int, float} or type(raw_y) not in {int, float}:
            continue
        x = float(raw_x)
        y = float(raw_y)
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        by_series.setdefault(series_label, []).append((x, y))
    if not by_series:
        return None

    traces: list[dict[str, object]] = []
    for label, points in by_series.items():
        ordered = sorted(points)
        traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": label,
            "x": [point[0] for point in ordered],
            "y": [point[1] for point in ordered],
            "showlegend": len(by_series) > 1,
            "hovertemplate": (
                f"{parameter_name}: %{{x:.6g}}<br>"
                f"{metric_name}: %{{y:.6g}}<extra>{label}</extra>"
            ),
        })

    preference = metric_preference if metric_preference in {"higher", "lower", "neutral"} else "neutral"
    preference_label = {
        "higher": "higher is better",
        "lower": "lower is better",
        "neutral": "neutral",
    }[preference]
    return {
        "data": traces,
        "layout": {
            "title": {"text": f"{metric_name} vs {parameter_name} / {preference_label}"},
            "showlegend": len(by_series) > 1,
            "height": 360,
            "margin": {"l": 75, "r": 40, "t": 100, "b": 80},
            "hovermode": "x unified",
            "xaxis": {"title": {"text": parameter_name}, "automargin": True},
            "yaxis": {"title": {"text": metric_name}, "automargin": True},
            "meta": {
                "evaluation": stage,
                "evaluation_name": evaluation_name or stage,
                "metric": metric_name,
                "preference": preference,
                "sweep_parameter": parameter_name,
            },
        },
    }


def _comparison_bar_figure(
    *,
    stage: str,
    evaluation_name: str | None,
    metric_name: str,
    metric_preference: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if type(metric_name) is not str or not metric_name:
        return None

    labels: list[str] = []
    values: list[float | None] = []
    rendered_labels = _comparison_row_labels(rows)
    for row, display_label in zip(rows, rendered_labels, strict=True):
        raw_metrics = row.get("metrics")
        if not display_label or not isinstance(raw_metrics, Mapping):
            continue
        value = raw_metrics.get(metric_name)
        numeric: float | None = None
        if type(value) in {int, float}:
            candidate = float(value)
            if math.isfinite(candidate):
                numeric = candidate
        labels.append(display_label)
        values.append(numeric)

    if not labels or not any(value is not None for value in values):
        return None

    preference = metric_preference if metric_preference in {"higher", "lower", "neutral"} else "neutral"
    preference_label = {
        "higher": "higher is better",
        "lower": "lower is better",
        "neutral": "neutral",
    }[preference]
    hover_preference = {
        "higher": "higher is better",
        "lower": "lower is better",
        "neutral": "no preferred direction",
    }[preference]
    trace: dict[str, object] = {
        "type": "bar",
        "orientation": "h",
        "x": values,
        "y": labels,
        "showlegend": False,
        "name": metric_name,
        "hovertemplate": (
            f"%{{y}}<br>{metric_name}: %{{x:.6g}}<br>{hover_preference}<extra></extra>"
        ),
    }
    colors = _comparison_rank_colors(values, preference=preference)
    if colors is not None:
        trace["marker"] = {"color": colors}

    title_prefix = evaluation_name or stage
    layout: dict[str, object] = {
        "title": {"text": f"{metric_name} / {preference_label}"},
        "showlegend": False,
        "height": 320,
        "margin": {"l": 80, "r": 30, "t": 100, "b": 75},
        "hovermode": "closest",
        "hoverlabel": {
            "bgcolor": "#1F2937",
            "bordercolor": "#6B7280",
            "font": {"color": "#FFFFFF"},
        },
        "xaxis": {"title": {"text": metric_name}, "automargin": True},
        "yaxis": {
            "automargin": True,
            "categoryorder": "array",
            "categoryarray": list(reversed(labels)),
        },
        "meta": {
            "evaluation": stage,
            "evaluation_name": title_prefix,
            "metric": metric_name,
            "preference": preference,
        },
    }
    if preference in {"higher", "lower"}:
        layout["annotations"] = [{
            "xref": "paper",
            "yref": "paper",
            "x": 1,
            "y": -0.18,
            "xanchor": "right",
            "yanchor": "top",
            "showarrow": False,
            "text": "Blue = better / Red = worse",
        }]
    return {"data": [trace], "layout": layout}



_TIMING_BREAKDOWN_COMPONENTS = (
    ("latency", "Total pipeline"),
    ("detector_preprocessing", "Detector preprocess"),
    ("detector_inference", "Detector inference"),
    ("detector_postprocessing", "Detector postprocess"),
    ("crop_extraction", "Crop extraction"),
    ("base_classifier_preprocessing", "Base classifier preprocess"),
    ("base_classifier_inference", "Base classifier inference"),
    ("red_five_classifier_preprocessing", "Red-five preprocess"),
    ("red_five_classifier_inference", "Red-five inference"),
    ("pipeline_unattributed_overhead", "Other pipeline overhead"),
)
_TIMING_STATS = (
    ("mean", "Mean"),
    ("p50", "P50"),
    ("p95", "P95"),
    ("max", "Max"),
)


def _human_metric_label(metric: str) -> str:
    return metric.replace("_", " ").strip().capitalize()


def _grouped_metric_figure(
    *,
    title: str,
    description: str | None,
    categories: Sequence[tuple[str, str]],
    rows: Sequence[Mapping[str, object]],
    y_axis_title: str,
    value_suffix: str = "",
) -> dict[str, object] | None:
    labels = [label for _metric, label in categories]
    traces: list[dict[str, object]] = []
    display_labels = _comparison_row_labels(rows)
    for row, trial in zip(rows, display_labels, strict=True):
        raw_metrics = row.get("metrics")
        if not trial or not isinstance(raw_metrics, Mapping):
            continue
        values: list[float | None] = []
        for metric, _label in categories:
            raw = raw_metrics.get(metric)
            numeric: float | None = None
            if type(raw) in {int, float}:
                candidate = float(raw)
                if math.isfinite(candidate):
                    numeric = candidate
            values.append(numeric)
        if not any(value is not None for value in values):
            continue
        traces.append({
            "type": "bar",
            "name": trial,
            "x": labels,
            "y": values,
            "showlegend": True,
            "hovertemplate": (
                "%{x}<br>%{y:.6g}" + value_suffix + "<extra>%{fullData.name}</extra>"
            ),
        })
    if not traces:
        return None
    title_text = title
    if type(description) is str and description:
        title_text += f"<br><sup>{description}</sup>"
    return {
        "data": traces,
        "layout": {
            "title": {"text": title_text},
            "showlegend": True,
            "barmode": "group",
            "height": 430,
            "margin": {"l": 75, "r": 150, "t": 110, "b": 120},
            "hovermode": "closest",
            "legend": {
                "orientation": "v",
                "x": 1.0,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(38,42,49,0.85)",
                "bordercolor": "#8D9199",
                "borderwidth": 1,
                "font": {"color": "#E3E2E6"},
            },
            "xaxis": {"automargin": True, "tickangle": -25},
            "yaxis": {"title": {"text": y_axis_title}, "automargin": True, "rangemode": "tozero"},
        },
    }


def _comparison_grouped_metric_figures(
    *,
    evaluation_name: str | None,
    evaluation_description: str | None,
    metrics: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[tuple[str, dict[str, object]]], set[str]]:
    figures: list[tuple[str, dict[str, object]]] = []
    consumed: set[str] = set()
    metric_set = set(metrics)
    prefix = evaluation_name or "Evaluation"

    for stat, stat_label in _TIMING_STATS:
        categories = [
            (f"{component}_{stat}_ms", label)
            for component, label in _TIMING_BREAKDOWN_COMPONENTS
            if f"{component}_{stat}_ms" in metric_set
        ]
        if len(categories) >= 2:
            figure = _grouped_metric_figure(
                title=f"{prefix} - {stat_label} latency by stage",
                description=evaluation_description,
                categories=categories,
                rows=rows,
                y_axis_title="Time (ms)",
                value_suffix=" ms",
            )
            if figure is not None:
                figures.append((f"Latency breakdown - {stat_label}", figure))
                consumed.update(metric for metric, _label in categories)

    count_categories = [
        (metric, label)
        for stat, stat_label in _TIMING_STATS
        for metric, label in (
            (f"candidate_count_{stat}", f"All candidates {stat_label}"),
            (f"red_five_candidate_count_{stat}", f"Red-five candidates {stat_label}"),
        )
        if metric in metric_set
    ]
    if len(count_categories) >= 2:
        figure = _grouped_metric_figure(
            title=f"{prefix} - Candidate counts",
            description=evaluation_description,
            categories=count_categories,
            rows=rows,
            y_axis_title="Tiles per evaluation",
        )
        if figure is not None:
            figures.append(("Candidate counts", figure))
            consumed.update(metric for metric, _label in count_categories)

    cadence_rate_metrics = [
        ("cadence_fulfillment_rate", "Cadence fulfillment"),
        ("cadence_skip_rate", "Skipped in-flight ticks"),
    ]
    cadence_rate_metrics = [item for item in cadence_rate_metrics if item[0] in metric_set]
    if len(cadence_rate_metrics) >= 2:
        figure = _grouped_metric_figure(
            title=f"{prefix} - Cadence rates",
            description=evaluation_description,
            categories=cadence_rate_metrics,
            rows=rows,
            y_axis_title="Rate",
        )
        if figure is not None:
            figures.append(("Cadence rates", figure))
            consumed.update(metric for metric, _label in cadence_rate_metrics)

    gap_metrics = [
        ("eval_gap_p95_ms", "Evaluation gap P95"),
        ("eval_gap_max_ms", "Evaluation gap max"),
    ]
    gap_metrics = [item for item in gap_metrics if item[0] in metric_set]
    if len(gap_metrics) >= 2:
        figure = _grouped_metric_figure(
            title=f"{prefix} - Evaluation gaps",
            description=evaluation_description,
            categories=gap_metrics,
            rows=rows,
            y_axis_title="Time (ms)",
            value_suffix=" ms",
        )
        if figure is not None:
            figures.append(("Evaluation gaps", figure))
            consumed.update(metric for metric, _label in gap_metrics)

    rate_metrics = [metric for metric in metrics if metric.endswith("_rate") and metric not in consumed]
    if len(rate_metrics) >= 2:
        categories = [(metric, _human_metric_label(metric.removesuffix("_rate"))) for metric in rate_metrics]
        figure = _grouped_metric_figure(
            title=f"{prefix} - Rates",
            description=evaluation_description,
            categories=categories,
            rows=rows,
            y_axis_title="Rate",
        )
        if figure is not None:
            figures.append(("Rates", figure))
            consumed.update(rate_metrics)

    return figures, consumed


def _study_artifact(task: object, name: str) -> object | None:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        return None
    return artifacts.get(f"evaluation/{name}") or artifacts.get(name)


def _study_artifact_url(task: object, name: str) -> str | None:
    artifact = _study_artifact(task, name)
    if artifact is None:
        return None
    url = getattr(artifact, "url", None)
    if type(url) is not str or not url:
        return None
    return _clearml_files_proxy_url(url)


def _study_artifact_local_path(task: object, name: str) -> Path | None:
    artifact = _study_artifact(task, name)
    if artifact is None:
        return None
    getter = getattr(artifact, "get_local_copy", None)
    if not callable(getter):
        return None
    try:
        local = getter()
    except Exception:
        return None
    if type(local) is not str or not local:
        return None
    path = Path(local)
    return path if path.is_file() else None


def _study_artifact_plotly_figure(task: object, name: str) -> dict[str, object] | None:
    getter = getattr(task, "get_reported_plots", None)
    if callable(getter):
        try:
            plots = getter() or []
        except Exception:
            plots = []
        for plot in plots:
            if not isinstance(plot, Mapping):
                continue
            if plot.get("metric") != "evaluation plots" or plot.get("variant") != name:
                continue
            raw = plot.get("plot_str")
            if type(raw) is not str:
                continue
            try:
                figure = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(figure, dict):
                return cast(dict[str, object], figure)
    path = _study_artifact_local_path(task, name)
    if path is None:
        return None
    try:
        figure = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    return cast(dict[str, object], figure) if isinstance(figure, dict) else None


def _selector_labels(labels: Sequence[str]) -> list[str]:
    if len(labels) < 2:
        return list(labels)
    tokens = [label.split() for label in labels]
    common = 0
    while all(common < len(parts) for parts in tokens):
        token = tokens[0][common]
        if any(parts[common] != token for parts in tokens[1:]):
            break
        common += 1
    if common == 0 or any(common >= len(parts) for parts in tokens):
        return list(labels)
    shortened = [" ".join(parts[common:]) for parts in tokens]
    return shortened if all(shortened) and len(set(shortened)) == len(shortened) else list(labels)


def _selectable_plotly_figure(
    items: Sequence[tuple[str, Mapping[str, object]]],
    *,
    artifact_name: str | None = None,
) -> dict[str, object] | None:
    if not items:
        return None
    data: list[dict[str, object]] = []
    ranges: list[tuple[int, int]] = []
    accepted_labels: list[str] = []
    for index, (label, figure) in enumerate(items):
        raw_data = figure.get("data")
        if type(raw_data) is not list:
            continue
        start = len(data)
        for raw_trace in raw_data:
            if not isinstance(raw_trace, Mapping):
                continue
            trace = dict(raw_trace)
            trace["visible"] = index == 0
            data.append(trace)
        if len(data) == start:
            continue
        ranges.append((start, len(data)))
        accepted_labels.append(label)
    if not ranges or not data:
        return None
    display_labels = _selector_labels(accepted_labels)
    first_layout = items[0][1].get("layout")
    layout = dict(first_layout) if isinstance(first_layout, Mapping) else {}
    buttons: list[dict[str, object]] = []
    for item_index, label in enumerate(display_labels):
        visible = [False] * len(data)
        start, end = ranges[item_index]
        for trace_index in range(start, end):
            visible[trace_index] = True
        buttons.append({
            "label": label,
            "method": "update",
            "args": [{"visible": visible}, {}],
        })
    layout["updatemenus"] = [{
        "type": "dropdown", "direction": "down", "showactive": True,
        "x": 0.0, "y": 1.0, "xanchor": "left", "yanchor": "bottom",
        "bgcolor": "#262A31", "bordercolor": "#8D9199",
        "font": {"color": "#E3E2E6"},
        "buttons": buttons,
    }]
    margin = dict(layout.get("margin")) if isinstance(layout.get("margin"), Mapping) else {}
    margin.setdefault("l", 70)
    margin.setdefault("r", 30)
    margin["t"] = max(int(margin.get("t", 0) or 0), 120)
    margin.setdefault("b", 60)

    # ClearML clips Plotly overflow at the card boundary. Dual-right-axis Study
    # artifacts therefore need explicit right room instead of relying on SVG
    # overflow, and the robustness chart needs extra height/bottom room for its
    # long categorical condition labels.
    if isinstance(layout.get("yaxis2"), Mapping):
        margin["r"] = max(int(margin.get("r", 0) or 0), 110)
        yaxis2 = dict(layout["yaxis2"])
        yaxis2["automargin"] = True
        layout["yaxis2"] = yaxis2
    if artifact_name is not None and "robustness" in artifact_name:
        margin["b"] = max(int(margin.get("b", 0) or 0), 150)
        layout["height"] = max(int(layout.get("height", 0) or 0), 560)
        named_visible_traces = [
            trace
            for trace in data
            if trace.get("visible") is True
            and isinstance(trace.get("name"), str)
            and str(trace["name"]).strip()
        ]
        if len(named_visible_traces) >= 2:
            layout["showlegend"] = True
            legend = (
                dict(layout.get("legend"))
                if isinstance(layout.get("legend"), Mapping)
                else {}
            )
            legend.update({
                "orientation": "v",
                "x": 1.0,
                "y": 1.0,
                "xanchor": "right",
                "yanchor": "top",
                "bgcolor": "rgba(38,42,49,0.85)",
                "bordercolor": "#8D9199",
                "borderwidth": 1,
                "font": {"color": "#E3E2E6"},
            })
            layout["legend"] = legend
        for axis_name in ("xaxis", "yaxis"):
            axis = layout.get(axis_name)
            if isinstance(axis, Mapping):
                axis_copy = dict(axis)
                axis_copy["automargin"] = True
                if axis_name == "xaxis":
                    axis_copy.setdefault("tickangle", -30)
                layout[axis_name] = axis_copy
    layout["margin"] = margin
    return {"data": data, "layout": layout}


def _selectable_image_figure(items: Sequence[tuple[str, str]]) -> dict[str, object] | None:
    if not items:
        return None
    display_labels = _selector_labels([label for label, _source in items])

    def image_layout(source: str) -> list[dict[str, object]]:
        return [{
            "source": source, "xref": "x", "yref": "y",
            "x": 0.0, "y": 1.0, "sizex": 1.0, "sizey": 1.0,
            "xanchor": "left", "yanchor": "top", "sizing": "contain", "layer": "above",
        }]

    buttons = [{
        "label": display_labels[index],
        "method": "relayout",
        "args": [{"images": image_layout(source)}],
    } for index, (_label, source) in enumerate(items)]
    return {
        "data": [{
            "type": "scatter", "x": [0, 1], "y": [0, 1], "mode": "markers",
            "marker": {"opacity": 0}, "hoverinfo": "skip", "showlegend": False,
        }],
        "layout": {
            "height": 620,
            "dragmode": "zoom",
            "xaxis": {"visible": False, "range": [0, 1], "fixedrange": False},
            "yaxis": {"visible": False, "range": [0, 1], "scaleanchor": "x", "fixedrange": False},
            "images": image_layout(items[0][1]),
            "updatemenus": [{
                "type": "dropdown", "direction": "down", "showactive": True,
                "x": 0.0, "y": 1.0, "xanchor": "left", "yanchor": "bottom",
                "bgcolor": "#262A31", "bordercolor": "#8D9199",
                "font": {"color": "#E3E2E6"},
                "buttons": buttons,
            }],
            "margin": {"l": 20, "r": 20, "t": 120, "b": 20},
        },
    }

def _selectable_video_figure(items: Sequence[tuple[str, str]]) -> dict[str, object] | None:
    if not items:
        return None
    display_labels = _selector_labels([label for label, _source in items])
    return {
        "data": [{
            "type": "scatter",
            "x": [0],
            "y": [0],
            "mode": "markers",
            "marker": {"opacity": 0},
            "hoverinfo": "skip",
            "showlegend": False,
        }],
        "layout": {
            "height": 620,
            "xaxis": {"visible": False, "fixedrange": True},
            "yaxis": {"visible": False, "fixedrange": True},
            "margin": {"l": 20, "r": 20, "t": 90, "b": 20},
            "meta": {
                "mldb_study_media": {
                    "kind": "video",
                    "items": [
                        {"label": display_labels[index], "url": source}
                        for index, (_label, source) in enumerate(items)
                    ],
                }
            },
        },
    }


def _csv_table_figure(path: Path) -> dict[str, object] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error):
        return None
    if not rows:
        return None
    header = rows[0]
    body = rows[1:]
    columns = [[row[index] if index < len(row) else "" for row in body] for index in range(len(header))]
    return {
        "data": [{"type": "table", "header": {"values": header}, "cells": {"values": columns}}],
        "layout": {"height": min(700, 120 + 28 * max(len(body), 1))},
    }


def _pipeline_evaluation_comment(comparisons: Sequence[object]) -> str | None:
    sections: list[str] = []
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            continue
        stage = comparison.get("stage")
        if type(stage) is not str:
            continue
        name = comparison.get("evaluation_name")
        protocol = comparison.get("evaluation_protocol")
        description = comparison.get("evaluation_description")
        heading = stage if type(name) is not str or not name else f"{stage} - {name}"
        details = [heading]
        if type(protocol) is str and protocol:
            details.append(f"Protocol: {protocol}")
        if type(description) is str and description:
            details.append(description)
        sections.append("\n".join(details))
    if not sections:
        return None
    return "MLDB evaluation guide\n\n" + "\n\n".join(sections)


class ClearMLSDKAdapter:
    """One lazy SDK adapter satisfying admission/observation/cancel/log Protocols."""

    def __init__(
        self,
        settings: ClearMLSDKSettings | None = None,
        *,
        task_class: object | None = None,
    ) -> None:
        self._settings = settings or ClearMLSDKSettings()
        self._task_class = task_class
        self._configured = False

    @classmethod
    def from_backend_config(cls, config: BackendConfig) -> "ClearMLSDKAdapter":
        if not isinstance(config, BackendConfig) or config.backend_type != "clearml":
            raise ClearMLSDKError("ClearML SDK adapter requires clearml BackendConfig")
        options = config.options
        reports = options.get("log_reports", 20)
        if type(reports) is not int or reports < 1:
            raise ClearMLSDKError("ClearML log_reports must be a positive integer")
        prebuilt_runtime = options.get("prebuilt_runtime", False)
        if type(prebuilt_runtime) is not bool:
            raise ClearMLSDKError("ClearML prebuilt_runtime must be boolean")
        runtime_image_wait_seconds = options.get("runtime_image_wait_seconds", 1800)
        if type(runtime_image_wait_seconds) is not int or runtime_image_wait_seconds < 0:
            raise ClearMLSDKError("ClearML runtime_image_wait_seconds must be a non-negative integer")
        settings = ClearMLSDKSettings(
            api_host=_optional_string(options, "api_host"),
            web_host=_optional_string(options, "web_host"),
            files_host=_optional_string(options, "files_host"),
            access_key=_optional_string(options, "access_key"),
            secret_key=_optional_string(options, "secret_key"),
            repository=_optional_string(options, "repository"),
            local_repository_root=_optional_string(options, "local_repository_root"),
            docker_image=_optional_string(options, "docker_image"),
            docker_env_file=_optional_string(options, "docker_env_file"),
            docker_gpu=_optional_string(options, "docker_gpu"),
            docker_shm_size=_optional_string(options, "docker_shm_size"),
            runtime_registry_url=_optional_string(options, "runtime_registry_url"),
            runtime_registry_ca_bundle=_optional_string(options, "runtime_registry_ca_bundle"),
            runtime_image_profile=_optional_string(options, "runtime_image_profile"),
            runtime_image_wait_seconds=runtime_image_wait_seconds,
            s3_endpoint_url=_optional_string(options, "s3_endpoint_url"),
            s3_region=_optional_string(options, "s3_region"),
            script=_string_option(options, "script", "mldb_v2/src/backend/_clearml_sdk.py"),
            pipeline_script=_string_option(
                options,
                "pipeline_script",
                "mldb_v2/src/backend/_clearml_pipeline_controller.py",
            ),
            working_directory=_string_option(options, "working_directory", "."),
            pinned_data_root=_string_option(options, "pinned_data_root", "mldb_data"),
            runtime_data_root=_optional_string(options, "runtime_data_root"),
            artifact_uri_prefix=_optional_string(options, "artifact_uri_prefix"),
            step_queue=_optional_string(options, "queue"),
            stage_routes=_stage_routes_option(options),
            prebuilt_runtime=prebuilt_runtime,
            work_root=_string_option(options, "work_root", ".mldb-v2-clearml"),
            log_reports=reports,
        )
        injected = options.get("task_class")
        return cls(settings, task_class=injected)

    def _Task(self) -> Any:
        task_class = self._task_class or _load_task_class()
        if not self._configured:
            setter = getattr(task_class, "set_credentials", None)
            if not callable(setter):
                raise ClearMLSDKError("ClearML Task.set_credentials is unavailable")
            if any(
                value is not None
                for value in (
                    self._settings.api_host,
                    self._settings.web_host,
                    self._settings.files_host,
                    self._settings.access_key,
                    self._settings.secret_key,
                )
            ):
                setter(
                    api_host=self._settings.api_host,
                    web_host=self._settings.web_host,
                    files_host=self._settings.files_host,
                    key=self._settings.access_key,
                    secret=self._settings.secret_key,
                    store_conf_file=False,
                )
            self._configured = True
        return task_class

    def _get_task(self, task_id: str) -> object:
        if type(task_id) is not str or not task_id:
            raise ValueError("task_id must be a non-empty opaque string")
        task = self._Task().get_task(task_id=task_id)
        if task is None:
            raise ClearMLSDKError("ClearML Task no longer exists")
        return task

    def _ensure_pipeline_project_layout(
        self,
        *,
        task: object,
        logical_project: str,
        mldb_task: str,
    ) -> None:
        parent_project, pipeline_project = _native_pipeline_projects(
            logical_project=logical_project,
            task=mldb_task,
        )
        Task = self._Task()
        get_session = getattr(Task, "_get_default_session", None)
        if callable(get_session):
            try:
                from clearml.backend_interface.util import get_or_create_project

                session = get_session()
                get_or_create_project(
                    session,
                    project_name=parent_project,
                    system_tags=["hidden"],
                )
                project_id = getattr(task, "project", None)
                if _project_name(task) == pipeline_project and type(project_id) is str:
                    get_or_create_project(
                        session,
                        project_name=pipeline_project,
                        project_id=project_id,
                        system_tags=["pipeline", "hidden"],
                    )
                    return
            except Exception as error:
                raise ClearMLSDKError("ClearML Pipeline Project creation failed") from error

        if _project_name(task) == pipeline_project:
            return
        mover = getattr(task, "move_to_project", None)
        if not callable(mover):
            return
        try:
            mover(
                new_project_name=pipeline_project,
                system_tags=["pipeline", "hidden"],
            )
        except Exception as error:
            raise ClearMLSDKError("ClearML Pipeline Project placement failed") from error

    def _search(self, *, project: str, key: str, value: str) -> tuple[object, ...]:
        search_tag = _search_tag(key, value)
        tasks = self._Task().get_tasks(
            project_name=project,
            tags=[search_tag],
            allow_archived=True,
        )
        exact = tuple(
            task
            for task in tasks
            if _project_name(task) == project and _metadata(task).get(key) == value
        )
        if exact:
            return exact

        # ClearML may relocate a Task after it is attached to a Pipeline. The
        # MLDB ownership/search tag remains authoritative, so recover globally
        # when the logical project lookup is empty instead of losing the owner.
        relocated = self._Task().get_tasks(
            project_name=None,
            tags=[search_tag],
            allow_archived=True,
        )
        return tuple(
            task for task in relocated if _metadata(task).get(key) == value
        )

    def search_pipeline_runs(
        self, *, project: str, ownership_key: str
    ) -> Sequence[ClearMLPipelineRecord]:
        tasks = self._Task().get_tasks(
            project_name=None,
            tags=[_search_tag("mldb.pipeline_ownership_key", ownership_key)],
            allow_archived=True,
        )
        tasks = tuple(
            task
            for task in tasks
            if _is_pipeline_project(project=_project_name(task), logical_project=project)
            and _metadata(task).get("mldb.pipeline_ownership_key") == ownership_key
        )
        records: list[ClearMLPipelineRecord] = []
        for task in tasks:
            config = _configuration(task, _PIPELINE_CONFIG) or {}
            records.append(
                ClearMLPipelineRecord(
                    task_id=_task_id(task),
                    metadata=_metadata(task),
                    configuration=config,
                    status=_status(task),
                )
            )
        return records

    def create_pipeline_run(self, request: ClearMLPipelineCreateRequest) -> str | None:
        topology = request.configuration.get("mldb.topology")
        if not isinstance(topology, Mapping):
            raise ClearMLSDKError("ClearML Pipeline request topology is missing")
        study = request.metadata.get("mldb.study")
        if type(study) is not str or not study:
            raise ClearMLSDKError("ClearML Pipeline request study identity is missing")
        plan_config = request.configuration.get("mldb.plan")
        if not isinstance(plan_config, Mapping):
            raise ClearMLSDKError("ClearML Pipeline request plan is missing")
        mldb_task = _pipeline_task_from_plan(plan_config)
        _parent_project, pipeline_project = _native_pipeline_projects(
            logical_project=request.project,
            task=mldb_task,
        )
        trial_labels = _pipeline_trial_labels(self._settings, plan_config)
        native_dag = _native_pipeline_dag(
            topology,
            queue=self._settings.step_queue,
            stage_routes=self._settings.stage_routes,
            trial_labels=trial_labels,
        )
        Task = self._Task()
        try:
            controller_task_name = validate_user_facing_display_name(
                request.task_name, field="ClearML Pipeline Run display name"
            )
        except ValueError as error:
            raise ClearMLSDKError(str(error)) from error
        kwargs: dict[str, object] = {
            "project_name": pipeline_project,
            "task_name": controller_task_name,
            "task_type": "controller",
            "commit": request.source_commit,
            "script": self._settings.pipeline_script,
            "working_directory": self._settings.working_directory,
            "add_task_init_call": False,
        }
        if self._settings.repository is not None:
            kwargs["repo"] = self._settings.repository
        task = Task.create(**kwargs)
        if task is None:
            return None
        self._ensure_pipeline_project_layout(
            task=task,
            logical_project=request.project,
            mldb_task=mldb_task,
        )
        task_id = _task_id(task)
        properties = [
            {"name": key, "value": value}
            for key, value in request.metadata.items()
        ]
        task.set_user_properties(*properties)
        tags = ["mldb-v2", "mldb-v2-pipeline"] + [
            _search_tag(key, value)
            for key, value in request.metadata.items()
            if key in {"mldb.pipeline_ownership_key", "mldb.study_result"}
        ]
        task.set_tags(tags)
        get_system_tags = getattr(task, "get_system_tags", None)
        set_system_tags = getattr(task, "set_system_tags", None)
        if callable(get_system_tags) and callable(set_system_tags):
            system_tags = list(get_system_tags() or [])
            if "pipeline" not in system_tags:
                system_tags.append("pipeline")
            set_system_tags(system_tags)
        task.set_configuration_object(
            name=_PIPELINE_CONFIG,
            config_dict=dict(request.configuration),
        )
        _set_native_pipeline_configuration(task, native_dag)
        version = "1.0.0"
        set_parameters = getattr(task, "set_parameters_as_dict", None)
        if callable(set_parameters):
            set_parameters(
                {
                    "pipeline/default_queue": self._settings.step_queue or "",
                    "pipeline/add_pipeline_tags": "True",
                    "pipeline/target_project": "True",
                    "properties/version": version,
                }
            )
        set_runtime = getattr(task, "_set_runtime_properties", None)
        if callable(set_runtime):
            pipeline_hash = hashlib.sha256(
                json.dumps(
                    {
                        "source_commit": request.source_commit,
                        "configuration": dict(request.configuration),
                        "pipeline": native_dag,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            set_runtime(
                {
                    "_pipeline_hash": f"{pipeline_hash}:{version}",
                    "version": version,
                }
            )
        try:
            task.set_user_properties({"name": "version", "value": version})
        except Exception:
            pass
        task.set_packages(["clearml==2.1.12"])
        return task_id

    def _sync_pipeline_node_statuses(
        self,
        *,
        pipeline: object,
        summary: Mapping[str, object],
    ) -> None:
        native = _configuration(pipeline, _NATIVE_PIPELINE_CONFIG)
        rows = summary.get("rows")
        if native is None or type(rows) is not list:
            return
        changed = False
        for row in rows:
            if type(row) is not dict:
                continue
            trial = row.get("trial")
            kind = row.get("kind")
            coordinate = row.get("coordinate")
            disposition = row.get("disposition")
            if type(trial) is not str:
                continue
            if kind == "training":
                step = f"{trial}-train"
            elif kind == "evaluation" and type(coordinate) is str:
                step = f"{trial}-{coordinate}"
            else:
                continue
            node = _native_pipeline_node(native, step)
            if node is None:
                continue
            executed = node.get("executed")
            if type(executed) is str and executed and node.get("job_id") != executed:
                node["job_id"] = executed
                changed = True

            status: str | None = None
            if disposition == "completed":
                status = "completed"
            elif disposition == "failed":
                status = "failed"
            elif disposition == "cancelled":
                status = "aborted"
            elif disposition == "skipped":
                status = "skipped"
            elif disposition == "pending":
                task_id = node.get("executed")
                if type(task_id) is str and task_id:
                    try:
                        child_status = _status(self._get_task(task_id))
                    except Exception:
                        child_status = None
                    status = {
                        "created": "pending",
                        "queued": "queued",
                        "in_progress": "running",
                        "publishing": "running",
                        "completed": "completed",
                        "published": "completed",
                        "closed": "completed",
                        "failed": "failed",
                        "stopped": "aborted",
                    }.get(child_status)
                else:
                    status = "pending"
            if status is not None and node.get("status") != status:
                node["status"] = status
                changed = True
        if changed:
            _set_native_pipeline_configuration(pipeline, native)

    def _project_pipeline_execution_views(self, *, task: object) -> None:
        """Best-effort native ClearML Pipeline flow/table projection."""
        native = _configuration(task, _NATIVE_PIPELINE_CONFIG)
        if native is None:
            return
        get_logger = getattr(task, "get_logger", None)
        if not callable(get_logger):
            return
        try:
            logger = get_logger()
            report_plotly = getattr(logger, "report_plotly", None)
            report_table = getattr(logger, "report_table", None)
            if not callable(report_plotly) or not callable(report_table):
                return

            pending = dict(native)
            ordered: list[str] = []
            while pending:
                progressed = False
                for name, raw in list(pending.items()):
                    if type(raw) is not dict:
                        pending.pop(name)
                        progressed = True
                        continue
                    parents = raw.get("parents") or []
                    if not all(parent in ordered for parent in parents):
                        continue
                    ordered.append(name)
                    pending.pop(name)
                    progressed = True
                if not progressed:
                    return

            index = {name: i for i, name in enumerate(ordered)}
            labels: list[str] = []
            colors: list[str] = []
            sources: list[int] = []
            targets: list[int] = []
            values: list[int] = []
            color_lookup = {
                "failed": "red",
                "cached": "darkslateblue",
                "completed": "blue",
                "aborted": "royalblue",
                "queued": "#bdf5bd",
                "running": "green",
                "skipped": "gray",
                "pending": "lightsteelblue",
            }
            table = [["Pipeline Step", "Task ID", "Status", "Stage", "Parents"]]
            for name in ordered:
                raw = cast(dict[str, object], native[name])
                status = str(raw.get("status") or "pending")
                stage = str(raw.get("stage") or "")
                task_id = str(raw.get("job_id") or raw.get("executed") or "")
                parents = [str(parent) for parent in (raw.get("parents") or [])]
                labels.append(f"{name}<br />")
                colors.append(color_lookup.get(status, ""))
                for parent in parents:
                    sources.append(index[parent])
                    targets.append(index[name])
                    values.append(1)
                table.append([name, task_id, status, stage, ", ".join(parents)])

            linked = set(sources) | set(targets)
            flow = {
                "link": {
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "hovertemplate": "<extra></extra>",
                },
                "node": {
                    "label": labels,
                    "color": colors,
                    "hovertemplate": "%{label}<extra></extra>",
                },
                "textfont": {"color": "rgba(0,0,0,0)", "size": 1},
                "type": "sankey",
                "orientation": "h",
            }
            data: list[dict[str, object]] = [flow]
            singles = [i for i in range(len(ordered)) if i not in linked]
            if singles:
                data.append(
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": list(range(len(singles))),
                        "y": [1] * len(singles),
                        "text": [labels[i] for i in singles],
                        "hovertemplate": "%{text}<extra></extra>",
                        "marker": {"size": [40] * len(singles)},
                        "showlegend": False,
                    }
                )
            report_plotly(
                title="Pipeline",
                series="Execution Flow",
                iteration=0,
                figure={
                    "data": data,
                    "layout": {
                        "xaxis": {"visible": False},
                        "yaxis": {"visible": False},
                    },
                },
            )
            report_table(
                title="Pipeline Details",
                series="Execution Details",
                iteration=0,
                table_plot=table,
            )
        except Exception:
            return

    def _project_pipeline_summary_tables(
        self,
        *,
        task: object,
        summary: Mapping[str, object],
    ) -> None:
        get_logger = getattr(task, "get_logger", None)
        if not callable(get_logger):
            return
        try:
            logger = get_logger()
        except Exception:
            return
        report_table = getattr(logger, "report_table", None)
        report_plotly = getattr(logger, "report_plotly", None)
        comparisons = summary.get("comparisons")
        if type(comparisons) is not list:
            return

        if callable(report_table):
            guide: list[list[object]] = [["Evaluation", "Metric", "Description", "Better"]]
            for comparison in comparisons:
                if type(comparison) is not dict:
                    continue
                stage = comparison.get("stage")
                metric_names = comparison.get("metrics")
                metric_preferences = comparison.get("metric_preferences")
                metric_descriptions = comparison.get("metric_descriptions")
                if type(stage) is not str or type(metric_names) is not list:
                    continue
                preferences = (
                    metric_preferences if isinstance(metric_preferences, Mapping) else {}
                )
                descriptions = (
                    metric_descriptions if isinstance(metric_descriptions, Mapping) else {}
                )
                for metric in metric_names:
                    if type(metric) is not str:
                        continue
                    description = descriptions.get(metric, "")
                    preference = preferences.get(metric, "neutral")
                    guide.append([
                        stage,
                        metric,
                        description if type(description) is str else "",
                        preference if type(preference) is str else "neutral",
                    ])
            if len(guide) > 1:
                try:
                    report_table(
                        title="Evaluation Metrics",
                        series="Guide",
                        iteration=0,
                        table_plot=guide,
                    )
                except Exception:
                    pass

        projection_iteration = _study_comparison_projection_iteration(task)

        for comparison in comparisons:
            if type(comparison) is not dict:
                continue
            stage = comparison.get("stage")
            evaluation_name = comparison.get("evaluation_name")
            metric_names = comparison.get("metrics")
            metric_preferences = comparison.get("metric_preferences")
            rows = comparison.get("rows")
            if type(stage) is not str or type(metric_names) is not list or type(rows) is not list:
                continue
            preferences = (
                metric_preferences if isinstance(metric_preferences, Mapping) else {}
            )
            metrics = [name for name in metric_names if type(name) is str]
            normalized_rows: list[Mapping[str, object]] = []
            for row in rows:
                if type(row) is not dict:
                    continue
                disposition = row.get("disposition")
                values = row.get("metrics")
                if type(disposition) is not str or not isinstance(values, Mapping):
                    continue
                normalized_rows.append(row)

            evaluation_parameter_names = _comparison_varying_parameter_names(
                normalized_rows,
                field="parameters",
            )
            sweep_parameter = _comparison_varying_numeric_parameter(normalized_rows)
            series_labels = _comparison_series_labels(normalized_rows)
            table: list[list[object]] = [[
                "Condition / model",
                *evaluation_parameter_names,
                "Status",
                *metrics,
            ]]
            for row, series_label in zip(normalized_rows, series_labels, strict=True):
                disposition = row["disposition"]
                values = cast(Mapping[str, object], row["metrics"])
                parameters = row.get("parameters")
                rendered: list[object] = [series_label]
                for parameter in evaluation_parameter_names:
                    value = (
                        parameters.get(parameter, "")
                        if isinstance(parameters, Mapping)
                        else ""
                    )
                    rendered.append(value)
                rendered.append(disposition)
                for metric in metrics:
                    value = values.get(metric, "")
                    if type(value) in {int, float}:
                        numeric = float(value)
                        value = numeric if math.isfinite(numeric) else ""
                    rendered.append(value)
                table.append(rendered)

            if len(table) > 1 and callable(report_table):
                try:
                    report_table(
                        title="Study Comparison",
                        series=stage,
                        iteration=projection_iteration,
                        table_plot=table,
                        extra_layout={"height": 320},
                    )
                except Exception:
                    pass

            if callable(report_plotly):
                if sweep_parameter is not None:
                    for metric in metrics:
                        preference = preferences.get(metric, "neutral")
                        figure = _comparison_sweep_figure(
                            stage=stage,
                            evaluation_name=(
                                evaluation_name if type(evaluation_name) is str else None
                            ),
                            metric_name=metric,
                            metric_preference=(
                                preference if type(preference) is str else "neutral"
                            ),
                            parameter_name=sweep_parameter,
                            rows=normalized_rows,
                        )
                        if figure is None:
                            continue
                        try:
                            report_plotly(
                                title=f"Model Comparison - {stage}",
                                series=metric,
                                iteration=projection_iteration,
                                figure=figure,
                            )
                        except Exception:
                            pass
                    continue

                evaluation_description = comparison.get("evaluation_description")
                grouped, consumed = _comparison_grouped_metric_figures(
                    evaluation_name=(
                        evaluation_name if type(evaluation_name) is str else None
                    ),
                    evaluation_description=(
                        evaluation_description
                        if type(evaluation_description) is str
                        else None
                    ),
                    metrics=metrics,
                    rows=normalized_rows,
                )
                for series_name, figure in grouped:
                    try:
                        report_plotly(
                            title=f"Model Comparison - {stage}",
                            series=series_name,
                            iteration=projection_iteration,
                            figure=figure,
                        )
                    except Exception:
                        pass

                remaining = [metric for metric in metrics if metric not in consumed]
                if len(remaining) == 1 and grouped:
                    # A grouped evaluation often has one scalar with a different
                    # unit (for example effective_eval_hz). Keep it in the
                    # comparison table instead of creating a one-item chart.
                    remaining = []
                for metric in remaining:
                    preference = preferences.get(metric, "neutral")
                    figure = _comparison_bar_figure(
                        stage=stage,
                        evaluation_name=(
                            evaluation_name if type(evaluation_name) is str else None
                        ),
                        metric_name=metric,
                        metric_preference=(
                            preference if type(preference) is str else "neutral"
                        ),
                        rows=normalized_rows,
                    )
                    if figure is None:
                        continue
                    try:
                        report_plotly(
                            title=f"Model Comparison - {stage}",
                            series=metric,
                            iteration=projection_iteration,
                            figure=figure,
                        )
                    except Exception:
                        pass

    def _project_study_artifacts(
        self,
        *,
        task: object,
        summary: Mapping[str, object],
    ) -> None:
        comparisons = summary.get("comparisons")
        if type(comparisons) is not list:
            return
        try:
            logger = _clearml_logger(task)
        except Exception:
            return
        report_plotly = getattr(logger, "report_plotly", None)
        report_image = getattr(logger, "report_image", None)
        report_table = getattr(logger, "report_table", None)
        report_media = getattr(logger, "report_media", None)
        if not callable(report_plotly):
            return

        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                continue
            stage = comparison.get("stage")
            declarations = comparison.get("artifacts")
            rows = comparison.get("rows")
            if type(stage) is not str or not isinstance(declarations, Mapping) or type(rows) is not list:
                continue
            for artifact_name, raw_declaration in declarations.items():
                if type(artifact_name) is not str or not isinstance(raw_declaration, Mapping):
                    continue
                study_view = raw_declaration.get("study_view", "hidden")
                artifact_format = raw_declaration.get("format")
                if study_view not in {"select", "all"} or type(artifact_format) is not str:
                    continue
                loaded: list[tuple[str, object, object]] = []
                for row in rows:
                    if not isinstance(row, Mapping) or row.get("disposition") != "completed":
                        continue
                    execution_id = row.get("execution_id")
                    trial_label = row.get("trial_label")
                    accepted_artifacts = row.get("artifacts")
                    if (type(execution_id) is not str or type(trial_label) is not str
                            or not isinstance(accepted_artifacts, Mapping)
                            or artifact_name not in accepted_artifacts):
                        continue
                    try:
                        child = self._get_task(execution_id)
                    except Exception:
                        continue
                    try:
                        trial_label = validate_user_facing_display_name(
                            trial_label, field="Study artifact condition label"
                        )
                    except ValueError as error:
                        raise ClearMLSDKError(str(error)) from error
                    loaded.append((trial_label, child, accepted_artifacts[artifact_name]))
                if not loaded:
                    continue

                title = f"Study Artifact - {stage}"
                if study_view == "select":
                    figure: dict[str, object] | None = None
                    if artifact_format == "plotly-json":
                        figures = [
                            (label, item)
                            for label, child, _ref in loaded
                            if (item := _study_artifact_plotly_figure(child, artifact_name)) is not None
                        ]
                        figure = _selectable_plotly_figure(figures, artifact_name=artifact_name)
                    elif artifact_format == "png":
                        images: list[tuple[str, str]] = []
                        for label, child, _ref in loaded:
                            source = _study_artifact_url(child, artifact_name)
                            if source is not None:
                                images.append((label, source))
                        figure = _selectable_image_figure(images)
                    elif artifact_format == "csv":
                        figures = []
                        for label, child, _ref in loaded:
                            path = _study_artifact_local_path(child, artifact_name)
                            if path is None:
                                continue
                            table_figure = _csv_table_figure(path)
                            if table_figure is not None:
                                figures.append((label, table_figure))
                        figure = _selectable_plotly_figure(figures)
                    elif artifact_format == "mp4":
                        if callable(report_media):
                            for label, child, _ref in loaded:
                                source = _study_artifact_url(child, artifact_name)
                                if source is None:
                                    continue
                                try:
                                    report_media(
                                        title=f"Study Video - {stage}",
                                        series=f"{artifact_name} | {label}",
                                        iteration=0,
                                        url=source,
                                    )
                                except Exception:
                                    pass
                        continue
                    if figure is not None:
                        try:
                            report_plotly(
                                title=title,
                                series=artifact_name,
                                iteration=0,
                                figure=figure,
                            )
                        except Exception:
                            pass
                    continue

                # `all` mirrors every trial artifact directly onto the Study controller.
                for label, child, _ref in loaded:
                    try:
                        if artifact_format == "plotly-json":
                            figure = _study_artifact_plotly_figure(child, artifact_name)
                            if figure is not None:
                                report_plotly(
                                    title=f"{title} - {artifact_name}",
                                    series=label,
                                    iteration=0,
                                    figure=figure,
                                )
                        elif artifact_format == "png" and callable(report_image):
                            path = _study_artifact_local_path(child, artifact_name)
                            if path is not None:
                                report_image(
                                    title=f"{title} - {artifact_name}",
                                    series=label,
                                    iteration=0,
                                    local_path=str(path),
                                )
                        elif artifact_format == "csv" and callable(report_table):
                            path = _study_artifact_local_path(child, artifact_name)
                            if path is not None:
                                report_table(
                                    title=f"{title} - {artifact_name}",
                                    series=label,
                                    iteration=0,
                                    csv=str(path),
                                )
                        elif artifact_format == "mp4" and callable(report_media):
                            source = _study_artifact_url(child, artifact_name)
                            if source is not None:
                                report_media(
                                    title=f"Study Video - {stage}",
                                    series=f"{artifact_name} | {label}",
                                    iteration=0,
                                    url=source,
                                )
                    except Exception:
                        pass

    def project_pipeline_summary(
        self,
        *,
        execution_id: str,
        summary: Mapping[str, object],
    ) -> None:
        task = self._get_task(execution_id)
        payload = dict(summary)
        study_status = payload.get("status")
        task_status = _status(task)
        if study_status == "completed" and task_status == "failed":
            # Same-run stage retry may recover a canonical Study after this
            # controller was already closed as failed. ClearML only allows
            # configuration/report edits while a Task is editable, so reopen the
            # same controller with a forced StartedRequest. This is not Reset:
            # the Task identity, historical logs, artifacts and failed child
            # attempt remain intact. The controller is closed completed again
            # after the final canonical projection is written.
            task.mark_started(force=True)
            task_status = _status(task)

        previous = _configuration(task, _PIPELINE_SUMMARY_CONFIG)
        if previous != payload:
            task.set_configuration_object(
                name=_PIPELINE_SUMMARY_CONFIG,
                config_dict=payload,
            )
        comparisons = payload.get("comparisons")
        if type(comparisons) is list:
            comment = _pipeline_evaluation_comment(comparisons)
            set_comment = getattr(task, "set_comment", None)
            if comment is not None and callable(set_comment):
                try:
                    set_comment(comment)
                except Exception:
                    pass

        # ClearML Plot events with the same metric/variant/iteration are append-like
        # rather than a reliable overwrite surface. Emitting comparison plots for
        # every in-progress projection can therefore leave a stale partial plot
        # visible after the final Study summary arrives. Keep the live canonical
        # summary/config and native Pipeline node statuses updated continuously,
        # but emit controller comparison tables/plots only for a terminal Study.
        if study_status in {"completed", "completed_with_failures", "failed", "cancelled"}:
            self._project_pipeline_summary_tables(task=task, summary=payload)
            self._project_study_artifacts(task=task, summary=payload)
            # ClearML report_table/report_plotly enqueue events asynchronously,
            # while mark_completed/mark_failed/mark_stopped only send a status
            # transition and do not flush pending reports. Wait for the terminal
            # comparison events before closing the controller Task so the final
            # projection cannot disappear at process teardown.
            flush = getattr(task, "flush", None)
            if callable(flush):
                try:
                    flush(wait_for_uploads=True)
                except Exception:
                    pass
        self._sync_pipeline_node_statuses(pipeline=task, summary=payload)

        task_status = _status(task)
        if study_status in {"submitted", "cancelling"} and task_status == "created":
            task.mark_started(force=True)
        elif study_status == "completed" and task_status not in _TERMINAL_STATUSES:
            status_message = (
                "MLDB Study completed after stage retry"
                if task_status == "in_progress"
                else "MLDB Study completed"
            )
            task.mark_completed(force=True, status_message=status_message)
        elif study_status in {"completed_with_failures", "failed"} and task_status not in _TERMINAL_STATUSES:
            task.mark_failed(force=True, status_message=f"MLDB Study {study_status}")
        elif study_status == "cancelled" and task_status not in _TERMINAL_STATUSES:
            task.mark_stopped(force=True, status_message="MLDB Study cancelled")

    def search_tasks(
        self, *, project: str, ownership_key: str
    ) -> Sequence[ClearMLTaskRecord]:
        tasks = self._search(
            project=project,
            key="mldb.ownership_key",
            value=ownership_key,
        )
        records: list[ClearMLTaskRecord] = []
        for task in tasks:
            config = _configuration(task, _MLDB_CONFIG)
            if config is None:
                config = {}
            records.append(
                ClearMLTaskRecord(
                    task_id=_task_id(task),
                    metadata=_metadata(task),
                    configuration=config,
                )
            )
        return records

    def bind_task_to_pipeline(
        self,
        *,
        task_id: str,
        pipeline_execution_id: str,
        pipeline_step: str,
    ) -> None:
        child = self._get_task(task_id)
        pipeline = self._get_task(pipeline_execution_id)
        if _status(pipeline) in {"stopped", "failed"}:
            mark_started = getattr(pipeline, "mark_started", None)
            if not callable(mark_started):
                raise ClearMLSDKError(
                    "terminal ClearML Pipeline cannot be reopened for MLDB resume"
                )
            mark_started(force=True)
        child_meta = _metadata(child)
        pipeline_meta = _metadata(pipeline)
        if child_meta.get("mldb.study_result") != pipeline_meta.get("mldb.study_result"):
            raise ClearMLSDKError("ClearML child Task and Pipeline StudyResult do not match")

        projected_pipeline = child_meta.get("mldb.pipeline_execution")
        projected_step = child_meta.get("mldb.pipeline_step")
        if projected_pipeline not in {None, pipeline_execution_id}:
            raise ClearMLSDKError("ClearML child Task Pipeline identity does not match")
        if projected_step not in {None, pipeline_step}:
            raise ClearMLSDKError("ClearML child Task Pipeline step does not match")

        native = _configuration(pipeline, _NATIVE_PIPELINE_CONFIG)
        if native is None:
            raise ClearMLSDKError("ClearML Pipeline native DAG configuration is missing")
        node = _native_pipeline_node(native, pipeline_step)
        if node is None:
            raise ClearMLSDKError("ClearML Pipeline step does not exist in native DAG")
        existing = node.get("executed")
        if existing is not None and existing != task_id:
            raise ClearMLSDKError("ClearML Pipeline step is already bound to another Task")

        child_data = getattr(child, "data", None)
        parent = getattr(child_data, "parent", None)
        if parent is None:
            parent = getattr(child, "parent", None)
        if parent == pipeline_execution_id:
            if existing != task_id or node.get("job_id") != task_id:
                node["executed"] = task_id
                node["job_id"] = task_id
                _set_native_pipeline_configuration(pipeline, native)
            return
        if parent not in {None, ""}:
            raise ClearMLSDKError("ClearML child Task is already bound to another parent")
        if _status(child) not in {"created", "in_progress"}:
            raise ClearMLSDKError("ClearML unbound child Task is no longer mutable")

        set_parent = getattr(child, "set_parent", None)
        if not callable(set_parent):
            raise ClearMLSDKError("ClearML child Task does not expose set_parent()")
        set_parent(pipeline_execution_id)
        get_tags = getattr(child, "get_tags", None)
        set_tags = getattr(child, "set_tags", None)
        if callable(get_tags) and callable(set_tags):
            tags = list(get_tags() or [])
            pipeline_tag = f"pipe:{pipeline_execution_id}"
            if pipeline_tag not in tags:
                tags.append(pipeline_tag)
                set_tags(tags)
        node["executed"] = task_id
        node["job_id"] = task_id
        _set_native_pipeline_configuration(pipeline, native)

    def bind_retry_task_to_pipeline(
        self,
        *,
        task_id: str,
        owner_task_id: str,
        pipeline_execution_id: str,
        pipeline_step: str,
    ) -> None:
        """Associate a retry Task with the same Pipeline step without changing ownership."""
        child = self._get_task(task_id)
        owner = self._get_task(owner_task_id)
        pipeline = self._get_task(pipeline_execution_id)
        child_meta = _metadata(child)
        owner_meta = _metadata(owner)
        pipeline_meta = _metadata(pipeline)
        if child_meta.get("mldb.retry_owner") != owner_task_id:
            raise ClearMLSDKError("ClearML retry Task owner identity does not match")
        study_result = owner_meta.get("mldb.study_result")
        if (
            type(study_result) is not str
            or child_meta.get("mldb.study_result") != study_result
            or pipeline_meta.get("mldb.study_result") != study_result
        ):
            raise ClearMLSDKError("ClearML retry Task and Pipeline StudyResult do not match")
        if child_meta.get("mldb.pipeline_execution") != pipeline_execution_id:
            raise ClearMLSDKError("ClearML retry Pipeline execution identity does not match")
        if child_meta.get("mldb.pipeline_step") != pipeline_step:
            raise ClearMLSDKError("ClearML retry Pipeline step identity does not match")

        native = _configuration(pipeline, _NATIVE_PIPELINE_CONFIG)
        if native is None:
            raise ClearMLSDKError("ClearML Pipeline native DAG configuration is missing")
        node = _native_pipeline_node(native, pipeline_step)
        if node is None:
            raise ClearMLSDKError("ClearML retry Pipeline step does not exist")
        existing = node.get("executed")
        if existing != owner_task_id:
            raise ClearMLSDKError(
                "ClearML retry Pipeline step is not bound to the logical owner Task"
            )

        child_data = getattr(child, "data", None)
        parent = getattr(child_data, "parent", None)
        if parent is None:
            parent = getattr(child, "parent", None)
        if parent not in {None, "", pipeline_execution_id}:
            raise ClearMLSDKError("ClearML retry Task is already bound to another parent")
        if parent != pipeline_execution_id:
            set_parent = getattr(child, "set_parent", None)
            if not callable(set_parent):
                raise ClearMLSDKError("ClearML retry Task does not expose set_parent()")
            set_parent(pipeline_execution_id)
        get_tags = getattr(child, "get_tags", None)
        set_tags = getattr(child, "set_tags", None)
        if callable(get_tags) and callable(set_tags):
            tags = list(get_tags() or [])
            pipeline_tag = f"pipe:{pipeline_execution_id}"
            if pipeline_tag not in tags:
                tags.append(pipeline_tag)
                set_tags(tags)
        # Keep the native Pipeline step bound to the original logical owner.
        # A retry is a second physical execution under the same controller, not
        # a replacement owner. This preserves ordinary admit() idempotence.

    def create_task(self, request: ClearMLCreateRequest) -> str | None:
        stage_input = _restore_stage_input(request.launch.stage_input_json)
        runtime_snapshots = _build_runtime_snapshots(self._settings, stage_input)
        Task = self._Task()
        task_name = request.task_name
        if request.launch.pipeline_execution_id is not None:
            if request.launch.pipeline_step is None:
                raise ClearMLSDKError("Pipeline-bound Task is missing pipeline step identity")
            pipeline = self._get_task(request.launch.pipeline_execution_id)
            native = _configuration(pipeline, _NATIVE_PIPELINE_CONFIG)
            if native is None:
                raise ClearMLSDKError("ClearML Pipeline native DAG configuration is missing")
            node_display = _native_pipeline_node_display_name(
                native, request.launch.pipeline_step
            )
            if node_display is None:
                raise ClearMLSDKError("ClearML Pipeline step display name is missing")
            study_id = request.metadata.get("mldb.study")
            if type(study_id) is not str:
                raise ClearMLSDKError("ClearML Task Study identity is missing")
            task_name = f"{node_display} | {_local_reference_name(study_id)}"
        try:
            task_name = validate_user_facing_display_name(
                task_name, field="ClearML child Task display name"
            )
        except ValueError as error:
            raise ClearMLSDKError(str(error)) from error
        kwargs: dict[str, object] = {
            "project_name": request.project,
            "task_name": task_name,
            "task_type": "custom",
            "commit": request.launch.source_commit,
            "script": self._settings.script,
            "working_directory": self._settings.working_directory,
            "add_task_init_call": False,
        }
        if self._settings.repository is not None:
            kwargs["repo"] = self._settings.repository
        task = Task.create(**kwargs)
        if task is None:
            return None
        task_id = _task_id(task)
        stage_name = _stage_name_from_input(stage_input)
        route = _stage_route(self._settings.stage_routes, stage_name)
        docker_gpu = self._settings.docker_gpu
        if "docker_gpu" in route:
            routed_gpu = route["docker_gpu"]
            if routed_gpu is not None and type(routed_gpu) is not str:
                raise ClearMLSDKError("ClearML stage route docker_gpu is malformed")
            docker_gpu = cast(str | None, routed_gpu)
        runtime_registry_version = stage_input.get("runtime_registry_version")
        if type(runtime_registry_version) is not int or runtime_registry_version <= 0:
            raise ClearMLSDKError("StageInput runtime registry version is invalid")
        runtime_image_profile = self._settings.runtime_image_profile
        if "runtime_image_profile" in route:
            routed_profile = route["runtime_image_profile"]
            if routed_profile is not None and type(routed_profile) is not str:
                raise ClearMLSDKError("ClearML stage route runtime_image_profile is malformed")
            runtime_image_profile = cast(str | None, routed_profile)
        docker_image = self._settings.docker_image
        if runtime_image_profile is not None:
            assert self._settings.runtime_registry_url is not None
            try:
                runtime_image = RuntimeRegistryClient(
                    self._settings.runtime_registry_url,
                    ca_bundle=self._settings.runtime_registry_ca_bundle,
                ).runtime_image(
                    runtime_registry_version,
                    runtime_image_profile,
                    wait_seconds=self._settings.runtime_image_wait_seconds,
                )
            except RuntimeRegistryError as error:
                raise ClearMLSDKError(
                    f"runtime image resolution failed for registry version "
                    f"{runtime_registry_version}"
                ) from error
            docker_image = runtime_image.image_ref
        if docker_image is not None:
            docker_arguments: list[str] = []
            if docker_gpu is not None:
                docker_arguments.extend(["--gpus", docker_gpu])
            if self._settings.docker_shm_size is not None:
                docker_arguments.extend(["--shm-size", self._settings.docker_shm_size])
            docker_arguments.extend([
                "-e", f"MLDB_RUNTIME_REGISTRY_VERSION={runtime_registry_version}",
                "-e", "AWS_ACCESS_KEY_ID",
                "-e", "AWS_SECRET_ACCESS_KEY",
                "-e", "AWS_SESSION_TOKEN",
                "-e", "MINIO_ROOT_USER",
                "-e", "MINIO_ROOT_PASSWORD",
            ])
            if self._settings.prebuilt_runtime:
                docker_arguments.extend([
                    "-e", "CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/opt/conda/bin/python",
                ])
            if self._settings.docker_env_file is not None:
                docker_arguments.append(f"--env-file={self._settings.docker_env_file}")
            task.set_base_docker(
                docker_image=docker_image,
                docker_arguments=docker_arguments,
            )
        task.set_packages(list(_REMOTE_PACKAGES))

        properties = [
            {"name": key, "value": value}
            for key, value in request.metadata.items()
        ]
        task.set_user_properties(*properties)
        tags = ["mldb-v2"] + [
            _search_tag(key, value)
            for key, value in request.metadata.items()
            if key in {
                "mldb.ownership_key",
                "mldb.study_result",
                "mldb.retry_key",
                "mldb.retry_owner",
            }
        ]
        task.set_tags(tags)
        task.set_configuration_object(
            name=_MLDB_CONFIG,
            config_dict=dict(request.configuration),
        )
        if runtime_snapshots is not None:
            task.set_configuration_object(
                name=_RUNTIME_SNAPSHOTS_CONFIG,
                config_dict=runtime_snapshots,
            )
        task.set_configuration_object(
            name=_RUNTIME_CONFIG,
            config_dict={
                "pinned_data_root": self._settings.pinned_data_root,
                "runtime_data_root": self._settings.runtime_data_root,
                "artifact_uri_prefix": self._settings.artifact_uri_prefix,
                "s3_endpoint_url": self._settings.s3_endpoint_url,
                "s3_region": self._settings.s3_region,
                "work_root": self._settings.work_root,
            },
        )
        parameters = request.configuration.get("mldb.public_parameters")
        if type(parameters) is dict:
            task.set_parameters_as_dict({"mldb": parameters})
        if request.launch.pipeline_execution_id is not None:
            if request.launch.pipeline_step is None:
                raise ClearMLSDKError("Pipeline-bound Task is missing pipeline step identity")
            self.bind_task_to_pipeline(
                task_id=task_id,
                pipeline_execution_id=request.launch.pipeline_execution_id,
                pipeline_step=request.launch.pipeline_step,
            )
        retry_owner = request.configuration.get("mldb.retry_owner")
        retry_pipeline = request.configuration.get("mldb.pipeline_execution")
        retry_step = request.configuration.get("mldb.pipeline_step")
        if retry_owner is not None:
            if not all(type(value) is str and value for value in (retry_owner, retry_pipeline, retry_step)):
                if retry_pipeline is not None or retry_step is not None:
                    raise ClearMLSDKError("ClearML retry Pipeline identity is malformed")
            elif request.launch.pipeline_execution_id is not None:
                raise ClearMLSDKError("ClearML retry must not use ordinary Pipeline node binding")
            else:
                self.bind_retry_task_to_pipeline(
                    task_id=task_id,
                    owner_task_id=cast(str, retry_owner),
                    pipeline_execution_id=cast(str, retry_pipeline),
                    pipeline_step=cast(str, retry_step),
                )
        Task.enqueue(task=task, queue_name=request.launch.queue)
        return task_id

    def create_retry_task(self, request: ClearMLCreateRequest) -> str | None:
        """Create a physical retry Task that never claims logical ownership."""
        if "mldb.ownership_key" in request.metadata:
            raise ClearMLSDKError("ClearML retry Task must not claim logical ownership metadata")
        if type(request.metadata.get("mldb.retry_key")) is not str:
            raise ClearMLSDKError("ClearML retry Task is missing retry identity")
        return self.create_task(request)

    def search_retry_tasks(
        self, *, project: str, retry_key: str
    ) -> Sequence[ClearMLTaskRecord]:
        tasks = self._search(project=project, key="mldb.retry_key", value=retry_key)
        records: list[ClearMLTaskRecord] = []
        for task in tasks:
            records.append(
                ClearMLTaskRecord(
                    task_id=_task_id(task),
                    metadata=_metadata(task),
                    configuration=_configuration(task, _MLDB_CONFIG) or {},
                )
            )
        return records

    def _read_single_runtime_projection(
        self, *, task_id: str
    ) -> ClearMLRuntimeProjection:
        task = self._get_task(task_id)
        stored = _configuration(task, _PROJECTION_CONFIG)
        if stored is not None:
            try:
                return ClearMLRuntimeProjection(
                    state=cast(Any, stored["state"]),
                    execution_ids=cast(Sequence[str], stored["execution_ids"]),
                    terminal_candidates=cast(Sequence[object], stored["terminal_candidates"]),
                )
            except KeyError as error:
                raise ClearMLSDKError("ClearML runtime projection is incomplete") from error

        status = _status(task)
        if status in _ACTIVE_STATUSES:
            return ClearMLRuntimeProjection(
                state="active",
                execution_ids=[task_id],
                terminal_candidates=[],
            )
        if status not in _TERMINAL_STATUSES:
            raise ClearMLSDKError(f"unsupported ClearML Task status: {status}")
        successful_without_projection = status in {"completed", "published", "closed"}

        config = _configuration(task, _MLDB_CONFIG) or {}
        transport = config.get("mldb.stage_input")
        stage_input = _restore_stage_input(cast(str, transport))
        stage_key = _stage_key_from_input(stage_input)
        candidate_status = "cancelled" if status == "stopped" else "failed"
        diagnostic = {
            "code": (
                "clearml_task_completed_without_projection"
                if successful_without_projection
                else f"clearml_task_{candidate_status}"
            ),
            "message": (
                "ClearML Task reported successful terminal status but did not record "
                "the authoritative MLDB harness projection."
                if successful_without_projection
                else "ClearML Task terminated before a harness candidate was recorded."
            ),
        }
        candidate = {
            "state": "terminal",
            "stage_key": stage_key,
            "attempts": [
                {
                    "backend": "clearml",
                    "execution_id": task_id,
                    "status": candidate_status,
                    "started_at": None,
                    "ended_at": None,
                    "diagnostic": diagnostic,
                }
            ],
            "status": candidate_status,
            "diagnostic": diagnostic,
            "result": None,
        }
        return ClearMLRuntimeProjection(
            state="terminal",
            execution_ids=[task_id],
            terminal_candidates=[candidate],
        )

    def read_runtime_projection(
        self, *, task_id: str
    ) -> ClearMLRuntimeProjection:
        """Aggregate one ownership Task and its ordered physical retry Tasks."""
        owner = self._get_task(task_id)
        base = self._read_single_runtime_projection(task_id=task_id)
        retries = list(
            self._search(
                project=_project_name(owner),
                key="mldb.retry_owner",
                value=task_id,
            )
        )
        if not retries:
            return base

        owner_config = _configuration(owner, _MLDB_CONFIG) or {}
        owner_stage_input = owner_config.get("mldb.stage_input")
        owner_key = owner_config.get("mldb.ownership_key")
        ordered: list[tuple[int, object]] = []
        seen_indexes: set[int] = set()
        for retry in retries:
            metadata = _metadata(retry)
            config = _configuration(retry, _MLDB_CONFIG) or {}
            if "mldb.ownership_key" in metadata:
                raise ClearMLSDKError("ClearML retry Task must not claim logical ownership")
            try:
                index = int(metadata["mldb.retry_index"])
            except (KeyError, ValueError) as error:
                raise ClearMLSDKError("ClearML retry index is malformed") from error
            if index < 2 or index in seen_indexes:
                raise ClearMLSDKError("ClearML retry indexes are ambiguous")
            seen_indexes.add(index)
            retry_id = _task_id(retry)
            expected_retry_key = metadata.get("mldb.retry_key")
            if (
                metadata.get("mldb.retry_owner") != task_id
                or config.get("mldb.retry_owner") != task_id
                or config.get("mldb.retry_index") != str(index)
                or config.get("mldb.retry_key") != expected_retry_key
                or config.get("mldb.ownership_key") != owner_key
                or config.get("mldb.stage_input") != owner_stage_input
                or config.get("mldb.harness") != owner_config.get("mldb.harness")
                or config.get("mldb.source_commit") != owner_config.get("mldb.source_commit")
                or type(expected_retry_key) is not str
                or not expected_retry_key
                or not retry_id
            ):
                raise ClearMLSDKError("ClearML retry Task lineage is inconsistent")
            ordered.append((index, retry))
        ordered.sort(key=lambda item: item[0])
        expected_indexes = list(range(2, 2 + len(ordered)))
        if [index for index, _ in ordered] != expected_indexes:
            raise ClearMLSDKError("ClearML retry attempt indexes are not contiguous")
        if list(base.execution_ids) != [task_id]:
            raise ClearMLSDKError(
                "ClearML ownership Task already contains non-native retry projection"
            )

        execution_ids = [task_id]
        terminal_candidates = list(base.terminal_candidates)
        previous = base
        for index, retry in ordered:
            del index
            retry_id = _task_id(retry)
            if previous.state != "terminal" or len(previous.terminal_candidates) != 1:
                raise ClearMLSDKError("ClearML retry follows non-terminal prior attempt")
            prior_candidate = previous.terminal_candidates[0]
            if not isinstance(prior_candidate, Mapping) or prior_candidate.get("status") == "completed":
                raise ClearMLSDKError("ClearML completed attempt cannot be retried")
            projected = self._read_single_runtime_projection(task_id=retry_id)
            if list(projected.execution_ids) != [retry_id]:
                raise ClearMLSDKError("ClearML retry Task projection is not single-attempt")
            if len(projected.terminal_candidates) > 1:
                raise ClearMLSDKError("ClearML retry Task has ambiguous terminal projection")
            execution_ids.append(retry_id)
            terminal_candidates.extend(projected.terminal_candidates)
            previous = projected

        return ClearMLRuntimeProjection(
            state=previous.state,
            execution_ids=execution_ids,
            terminal_candidates=terminal_candidates,
        )

    def search_tasks_by_metadata(
        self, *, project: str, key: str, value: str
    ) -> Sequence[ClearMLOwnedTask]:
        tasks = self._search(project=project, key=key, value=value)
        return [
            ClearMLOwnedTask(
                task_id=_task_id(task),
                metadata=_metadata(task),
                cancellation_state=_cancellation_state(_status(task)),
            )
            for task in tasks
        ]

    def request_cancellation(self, *, task_id: str) -> None:
        task = self._get_task(task_id)
        status = _status(task)
        if status == "queued":
            self._Task().dequeue(task)
            task.mark_stopped(
                force=True,
                status_message="MLDB cancellation requested before execution",
            )
            return
        if status == "created":
            task.mark_stopped(
                force=True,
                status_message="MLDB cancellation requested before execution",
            )
            return
        if status in {"in_progress", "publishing"}:
            task.mark_stop_request(
                force=True,
                status_message="MLDB cancellation requested",
            )

    def read_task_logs(self, *, task_id: str) -> ClearMLLogData | None:
        task = self._get_task(task_id)
        getter = getattr(task, "get_reported_console_output", None)
        if not callable(getter):
            return None
        raw = getter(number_of_reports=self._settings.log_reports)
        if raw is None:
            return None
        if isinstance(raw, (str, bytes)):
            chunks = (str(raw),)
        else:
            chunks = tuple(item for item in raw if type(item) is str)
        if not chunks:
            return None
        locator_getter = getattr(task, "get_output_log_web_page", None)
        locator = locator_getter() if callable(locator_getter) else None
        if locator is not None and type(locator) is not str:
            locator = None
        return ClearMLLogData(chunks=chunks, locator=locator)


def _runtime_path(repository_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _remote_artifact_prefix(runtime: Mapping[str, object]) -> str:
    configured = runtime.get("artifact_uri_prefix")
    if type(configured) is str and configured:
        return configured.rstrip("/")
    env_prefix = os.environ.get("MLDB_V2_ARTIFACT_URI_PREFIX")
    if env_prefix:
        return env_prefix.rstrip("/")
    bucket = os.environ.get("MLDB_S3_BUCKET")
    if bucket:
        return f"s3://{bucket}/mldb-v2"
    raise ClearMLSDKError(
        "remote harness requires artifact_uri_prefix or MLDB_S3_BUCKET"
    )


def _remote_object_bytes(runtime: Mapping[str, object]):
    from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
    from mldb_v2.src.storage.s3_transport import (
        _S3TransportConfig,
        _create_s3_transport,
    )

    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("MINIO_ROOT_USER")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("MINIO_ROOT_PASSWORD")
    runtime_endpoint = runtime.get("s3_endpoint_url")
    runtime_region = runtime.get("s3_region")
    config = _S3TransportConfig(
        endpoint_url=(
            runtime_endpoint if type(runtime_endpoint) is str and runtime_endpoint
            else os.environ.get("MLDB_S3_ENDPOINT_URL")
        ),
        region_name=(
            runtime_region if type(runtime_region) is str and runtime_region
            else os.environ.get("MLDB_S3_REGION")
        ),
        access_key_id=access_key,
        secret_access_key=secret_key,
        session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )
    return _ObjectByteAccess(_create_s3_transport(config))


def _physical_attempt_token(ownership: object, retry_index: object) -> str:
    if type(ownership) is not str or not ownership.startswith("mldb-v2-stage:"):
        raise ClearMLSDKError("remote harness ownership projection is missing")
    ownership_token = ownership.split(":", 1)[1]
    if not ownership_token:
        raise ClearMLSDKError("remote harness ownership projection is missing")
    if retry_index is None:
        return ownership_token
    if type(retry_index) is not str or not retry_index.isdigit() or int(retry_index) < 2:
        raise ClearMLSDKError("remote harness retry index is malformed")
    return f"{ownership_token}/retry-{int(retry_index):04d}"


def _evaluation_artifact_uris(
    *, stage_input: Mapping[str, object], pinned_data_root: Path, prefix: str
) -> dict[str, str]:
    from mldb_v2.src.evaluation.evaluation_protocol import (
        _load_evaluation_protocol_definition,
    )

    stage = cast(dict[str, object], stage_input["stage"])
    protocol = _load_evaluation_protocol_definition(
        pinned_data_root, cast(str, stage["evaluation_protocol"])
    )
    artifacts = protocol["artifacts"]
    return {
        key: f"{prefix}/artifacts/{quote(key, safe='')}"
        for key in artifacts
    }


def _run_remote_harness() -> None:
    """ClearML-agent entrypoint; executes only the generic CommonExecutionHarness."""
    from clearml import Task  # type: ignore[import-not-found]

    from mldb_v2.src.backend.execution_harness import CommonExecutionHarness

    task = Task.init(project_name="mldb-v2", task_name="MLDB remote harness")
    task_id = _task_id(task)
    mldb_config = _configuration(task, _MLDB_CONFIG) or {}
    runtime = _configuration(task, _RUNTIME_CONFIG) or {}
    stage_input = _restore_stage_input(cast(str, mldb_config.get("mldb.stage_input")))

    repository_root = Path.cwd().resolve()
    pinned_data_root = _runtime_path(
        repository_root,
        cast(str, runtime.get("pinned_data_root") or "mldb_data"),
    )
    runtime_value = runtime.get("runtime_data_root") or os.environ.get(
        "MLDB_V2_RUNTIME_DATA_ROOT"
    ) or "mldb_data"
    runtime_data_root = _runtime_path(repository_root, cast(str, runtime_value))
    runtime_snapshots = _configuration(task, _RUNTIME_SNAPSHOTS_CONFIG)
    if runtime_snapshots is None:
        raise ClearMLSDKError("remote harness runtime snapshots are missing")
    _materialize_runtime_snapshots(
        runtime_root=runtime_data_root,
        pinned_root=pinned_data_root,
        stage_input=stage_input,
        snapshots=runtime_snapshots,
    )
    work_root = _runtime_path(
        repository_root,
        cast(str, runtime.get("work_root") or ".mldb-v2-clearml"),
    )
    ownership = mldb_config.get("mldb.ownership_key")
    physical_attempt_token = _physical_attempt_token(
        ownership,
        mldb_config.get("mldb.retry_index"),
    )
    attempt_root = work_root / physical_attempt_token
    prefix = f"{_remote_artifact_prefix(runtime)}/{physical_attempt_token}"
    artifact_uris: dict[str, str] = {}
    weights_uri: str | None = None
    if stage_input["kind"] == "training":
        weights_uri = f"{prefix}/weights.pt"
    else:
        artifact_uris = _evaluation_artifact_uris(
            stage_input=stage_input,
            pinned_data_root=pinned_data_root,
            prefix=prefix,
        )

    object_bytes = _remote_object_bytes(runtime)
    harness = CommonExecutionHarness(
        repository_root=repository_root,
        pinned_mldb_data_root=pinned_data_root,
        runtime_mldb_data_root=runtime_data_root,
        object_bytes=object_bytes,
        corpus_destination_root=attempt_root / "corpus",
        work_dir=attempt_root / "work",
        training_weights_uri=weights_uri,
        evaluation_artifact_uris=artifact_uris,
        backend="clearml",
        execution_id=task_id,
        started_at=None,
        telemetry_sink=_ClearMLScalarSink(
            task,
            pipeline_task_id=(
                cast(str, mldb_config.get("mldb.pipeline_execution"))
                if stage_input["kind"] == "training"
                and type(mldb_config.get("mldb.pipeline_execution")) is str
                else None
            ),
            pipeline_series=(
                (
                    f"bs{stage_input['stage']['parameters']['batch_size']}"
                    if type(stage_input["stage"]["parameters"].get("batch_size")) is int
                    else str(stage_input["trial"])
                )
                if stage_input["kind"] == "training"
                else None
            ),
        ),
    )
    candidate = harness(stage_input)
    if stage_input["kind"] == "evaluation":
        _project_evaluation_artifacts(
            task=task,
            candidate=candidate,
            object_bytes=object_bytes,
            projection_root=attempt_root / "clearml-projection",
        )
    task.set_configuration_object(
        name=_PROJECTION_CONFIG,
        config_dict={
            "state": "terminal",
            "execution_ids": [task_id],
            "terminal_candidates": [candidate],
        },
    )
    task.flush(wait_for_uploads=True)


if __name__ == "__main__":
    _run_remote_harness()
