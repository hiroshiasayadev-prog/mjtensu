"""Lazy production ClearML SDK activation seam for MLDB v2."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

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
from mldb_v2.src.backend._config import BackendConfig
from mldb_v2.src.common.ids import EntityKind
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.study._plan_build import _validate_study_plan
from mldb_v2.src.training.model import _validate_model
from mldb_v2.src.training.training_result import _validate_training_result

_MLDB_CONFIG = "mldb"
_RUNTIME_CONFIG = "mldb.runtime"
_RUNTIME_SNAPSHOTS_CONFIG = "mldb.runtime_snapshots"
_PROJECTION_CONFIG = "mldb.runtime_projection"
_ACTIVE_STATUSES = {"created", "queued", "in_progress", "publishing"}
_TERMINAL_STATUSES = {"completed", "published", "closed", "failed", "stopped"}
_REMOTE_PACKAGES = (
    "clearml==2.1.12",
    "boto3==1.43.93",
    "PyYAML==6.0.3",
    "numpy==1.26.4",
    "torch==2.5.1",
    "torchvision==0.20.1",
)


class ClearMLSDKError(RuntimeError):
    """Bounded production ClearML SDK activation/projection failure."""


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
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    script: str = "mldb_v2/src/backend/_clearml_sdk.py"
    working_directory: str = "."
    pinned_data_root: str = "mldb_data"
    runtime_data_root: str | None = None
    artifact_uri_prefix: str | None = None
    work_root: str = ".mldb-v2-clearml"
    log_reports: int = 20

    def __post_init__(self) -> None:
        for name in (
            "api_host", "web_host", "files_host", "repository", "local_repository_root",
            "docker_image", "docker_env_file", "docker_gpu", "s3_endpoint_url", "s3_region",
            "runtime_data_root", "artifact_uri_prefix",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value or value.strip() != value):
                raise ValueError(f"{name} must be null or a non-empty trimmed string")
        for name in ("script", "working_directory", "pinned_data_root", "work_root"):
            value = getattr(self, name)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty trimmed string")
        if (self.access_key is None) != (self.secret_key is None):
            raise ValueError("ClearML access_key and secret_key must be supplied together")
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


def _local_runtime_root(settings: ClearMLSDKSettings) -> Path | None:
    if settings.local_repository_root is None:
        return None
    repository_root = Path(settings.local_repository_root).resolve()
    configured = settings.runtime_data_root or "mldb_data"
    path = Path(configured)
    return path if path.is_absolute() else repository_root / path


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
    if stage_input["kind"] == "evaluation":
        expected |= {"training_result", "model"}
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


def _load_task_class() -> Any:
    try:
        from clearml import Task  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise ClearMLSDKError(
            "clearml package is required to activate the production ClearML backend"
        ) from error
    return Task


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


def _project_name(task: object) -> str | None:
    getter = getattr(task, "get_project_name", None)
    value = getter() if callable(getter) else None
    return value if type(value) is str else None


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
            s3_endpoint_url=_optional_string(options, "s3_endpoint_url"),
            s3_region=_optional_string(options, "s3_region"),
            script=_string_option(options, "script", "mldb_v2/src/backend/_clearml_sdk.py"),
            working_directory=_string_option(options, "working_directory", "."),
            pinned_data_root=_string_option(options, "pinned_data_root", "mldb_data"),
            runtime_data_root=_optional_string(options, "runtime_data_root"),
            artifact_uri_prefix=_optional_string(options, "artifact_uri_prefix"),
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

    def _search(self, *, project: str, key: str, value: str) -> tuple[object, ...]:
        tasks = self._Task().get_tasks(
            project_name=project,
            tags=[_search_tag(key, value)],
            allow_archived=True,
        )
        return tuple(
            task
            for task in tasks
            if _project_name(task) == project and _metadata(task).get(key) == value
        )

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

    def create_task(self, request: ClearMLCreateRequest) -> str | None:
        stage_input = _restore_stage_input(request.launch.stage_input_json)
        runtime_snapshots = _build_runtime_snapshots(self._settings, stage_input)
        Task = self._Task()
        kwargs: dict[str, object] = {
            "project_name": request.project,
            "task_name": request.task_name,
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
        if self._settings.docker_image is not None:
            docker_arguments: list[str] = []
            if self._settings.docker_gpu is not None:
                docker_arguments.extend(["--gpus", self._settings.docker_gpu])
            docker_arguments.extend([
                "-e", "AWS_ACCESS_KEY_ID",
                "-e", "AWS_SECRET_ACCESS_KEY",
                "-e", "AWS_SESSION_TOKEN",
                "-e", "MINIO_ROOT_USER",
                "-e", "MINIO_ROOT_PASSWORD",
            ])
            if self._settings.docker_env_file is not None:
                docker_arguments.append(f"--env-file={self._settings.docker_env_file}")
            task.set_base_docker(
                docker_image=self._settings.docker_image,
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
            if key in {"mldb.ownership_key", "mldb.study_result"}
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
        Task.enqueue(task=task, queue_name=request.launch.queue)
        return task_id

    def read_runtime_projection(
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
        if status not in {"failed", "stopped"}:
            raise ClearMLSDKError(
                "ClearML Task reached successful terminal state without harness projection"
            )

        config = _configuration(task, _MLDB_CONFIG) or {}
        transport = config.get("mldb.stage_input")
        stage_input = _restore_stage_input(cast(str, transport))
        stage_key = _stage_key_from_input(stage_input)
        candidate_status = "cancelled" if status == "stopped" else "failed"
        diagnostic = {
            "code": f"clearml_task_{candidate_status}",
            "message": "ClearML Task terminated before a harness candidate was recorded.",
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
    ownership = cast(str, mldb_config.get("mldb.ownership_key"))
    if type(ownership) is not str or not ownership.startswith("mldb-v2-stage:"):
        raise ClearMLSDKError("remote harness ownership projection is missing")
    attempt_root = work_root / ownership.split(":", 1)[1]
    prefix = f"{_remote_artifact_prefix(runtime)}/{ownership.split(':', 1)[1]}"
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

    harness = CommonExecutionHarness(
        repository_root=repository_root,
        pinned_mldb_data_root=pinned_data_root,
        runtime_mldb_data_root=runtime_data_root,
        object_bytes=_remote_object_bytes(runtime),
        corpus_destination_root=attempt_root / "corpus",
        work_dir=attempt_root / "work",
        training_weights_uri=weights_uri,
        evaluation_artifact_uris=artifact_uris,
        backend="clearml",
        execution_id=task_id,
        started_at=None,
    )
    candidate = harness(stage_input)
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
