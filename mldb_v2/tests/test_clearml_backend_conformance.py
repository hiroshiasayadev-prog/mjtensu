from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from mldb_v2.src.backend._clearml_admission import (
    ClearMLCreateRequest,
    ClearMLRemoteLaunch,
    ClearMLTaskRecord,
    _ownership_key,
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
from mldb_v2.src.backend._clearml_sdk import (
    ClearMLSDKAdapter,
    ClearMLSDKSettings,
    _ClearMLScalarSink,
    _configuration,
)
from mldb_v2.src.backend._config import BackendConfig
from mldb_v2.src.backend._registry import BackendRegistry
from mldb_v2.src.backend.clearml_backend import (
    ClearMLBackend,
    clearml_backend_factory,
    register_clearml_backend,
)
from mldb_v2.src.backend.stage_input import (
    EvaluationStageInput,
    StageInput,
    TrainingStageInput,
)
from mldb_v2.src.common.telemetry import _AcceptedScalarEvent
from mldb_v2.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationCoordinateId,
    EvaluationProtocolId,
    ModelId,
    StudyPlanId,
    StudyResultId,
    TaskId,
    TrainProtocolId,
    TrainingResultId,
    TrialId,
)


def _pin(kind: str, entity_id: str) -> dict[str, object]:
    return {
        "kind": kind,
        "id": entity_id,
        "yaml_sha256": "a" * 64,
        "companion_sha256": None,
        "sources": [],
        "manifest_sha256": None,
        "manifest_entries": None,
    }


def _training_stage_input() -> TrainingStageInput:
    return {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": StudyResultId("demo/run-abcd"),
        "plan": StudyPlanId("demo/study-a-plan-0123456789abcdef"),
        "plan_sha256": "1" * 64,
        "trial": TrialId("trial-0001"),
        "kind": "training",
        "coordinate": None,
        "source_commit": "2" * 40,
        "pins": [cast(object, _pin("study", "demo/study-a"))],
        "stage": {
            "task": TaskId("demo/task"),
            "corpus": CorpusId("demo/corpus"),
            "architecture": ArchitectureId("demo/architecture"),
            "train_protocol": TrainProtocolId("demo/train-protocol"),
            "parameters": {"batch_size": 8},
            "seed": 42,
        },
        "runtime_model": None,
    }


def _evaluation_stage_input() -> EvaluationStageInput:
    return {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": StudyResultId("demo/run-abcd"),
        "plan": StudyPlanId("demo/study-a-plan-0123456789abcdef"),
        "plan_sha256": "1" * 64,
        "trial": TrialId("trial-0001"),
        "kind": "evaluation",
        "coordinate": EvaluationCoordinateId("eval-0001"),
        "source_commit": "2" * 40,
        "pins": [cast(object, _pin("study", "demo/study-a"))],
        "stage": {
            "name": "holdout",
            "task": TaskId("demo/task"),
            "corpus": CorpusId("demo/eval-corpus"),
            "evaluation_protocol": EvaluationProtocolId("demo/eval-protocol"),
            "parameters": {"iou_threshold": 0.5},
        },
        "runtime_model": {
            "model": ModelId("demo/model-a"),
            "training_result": TrainingResultId("demo/training-a"),
            "task": TaskId("demo/task"),
            "architecture": ArchitectureId("demo/architecture"),
            "weights": {
                "uri": "s3://bucket/weights.pt",
                "bytes": 123,
                "sha256": "3" * 64,
                "format": "pytorch-state-dict/v1",
            },
        },
    }


def _diagnostic(status: str) -> dict[str, str]:
    return {"code": f"{status}_attempt", "message": f"{status} attempt"}


def _candidate(stage_input: StageInput, task_id: str, status: str) -> dict[str, object]:
    stage_key = _stage_key_from_input(stage_input)
    diagnostic = None if status == "completed" else _diagnostic(status)
    if status != "completed":
        result = None
    elif stage_input["kind"] == "training":
        result = {
            "weights": {
                "uri": "s3://bucket/result.pt",
                "bytes": 77,
                "sha256": "4" * 64,
                "format": "pytorch-state-dict/v1",
            }
        }
    else:
        result = {
            "metrics": {"f1": 0.9},
            "artifacts": {
                "predictions": {
                    "uri": "s3://bucket/predictions.jsonl",
                    "bytes": 8,
                    "sha256": "5" * 64,
                    "format": "jsonl",
                    "schema": "demo/predictions/v1",
                }
            },
        }
    return {
        "state": "terminal",
        "stage_key": stage_key,
        "attempts": [
            {
                "backend": "clearml",
                "execution_id": task_id,
                "status": status,
                "started_at": None,
                "ended_at": None,
                "diagnostic": diagnostic,
            }
        ],
        "status": status,
        "diagnostic": diagnostic,
        "result": result,
    }


class CombinedClient:
    def __init__(self, *, ambiguous_create: bool = False) -> None:
        self.records: list[ClearMLTaskRecord] = []
        self.projections: dict[str, ClearMLRuntimeProjection] = {}
        self.ambiguous_create = ambiguous_create
        self.create_count = 0
        self.cancelled: list[str] = []
        self.logs: dict[str, ClearMLLogData | None] = {}

    def search_tasks(self, *, project: str, ownership_key: str):
        return [
            record
            for record in self.records
            if record.metadata.get("mldb.ownership_key") == ownership_key
            and project == "mldb/" + record.metadata["mldb.namespace"]
        ]

    def create_task(self, request: ClearMLCreateRequest) -> str | None:
        self.create_count += 1
        task_id = f"opaque/task?id={self.create_count}"
        self.records.append(
            ClearMLTaskRecord(
                task_id=task_id,
                metadata=dict(request.metadata),
                configuration=deepcopy(dict(request.configuration)),
            )
        )
        self.projections[task_id] = ClearMLRuntimeProjection(
            state="active",
            execution_ids=[task_id],
            terminal_candidates=[],
        )
        if self.ambiguous_create:
            raise TimeoutError("server created task, response lost")
        return task_id

    def read_runtime_projection(self, *, task_id: str) -> ClearMLRuntimeProjection:
        return self.projections[task_id]

    def search_tasks_by_metadata(self, *, project: str, key: str, value: str):
        result = []
        for record in self.records:
            if project != "mldb/" + record.metadata["mldb.namespace"]:
                continue
            if record.metadata.get(key) != value:
                continue
            result.append(
                ClearMLOwnedTask(
                    task_id=record.task_id,
                    metadata=record.metadata,
                    cancellation_state=ClearMLCancellationState.ACTIVE,
                )
            )
        return result

    def request_cancellation(self, *, task_id: str) -> None:
        self.cancelled.append(task_id)

    def read_task_logs(self, *, task_id: str) -> ClearMLLogData | None:
        return self.logs.get(task_id)


def _backend(client: CombinedClient) -> ClearMLBackend:
    return clearml_backend_factory(
        BackendConfig("clearml", {"client": client, "queue": "gpu-a"})
    )


def test_generic_backend_composes_admission_observation_collection_and_restart() -> None:
    client = CombinedClient()
    stage_input = _training_stage_input()
    backend = _backend(client)

    admitted = backend.admit(stage_input=stage_input)
    stage_key = _stage_key_from_input(stage_input)
    assert admitted == {
        "state": "active",
        "stage_key": stage_key,
        "backend": "clearml",
        "execution_ids": ["opaque/task?id=1"],
    }
    assert client.create_count == 1

    replay = backend.admit(stage_input=stage_input)
    assert replay == admitted
    assert client.create_count == 1

    restarted = _backend(client)
    assert restarted.observe(stage_key=stage_key) == admitted

    terminal = _candidate(stage_input, "opaque/task?id=1", "completed")
    client.projections["opaque/task?id=1"] = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=["opaque/task?id=1"],
        terminal_candidates=[terminal],
    )
    collected = restarted.collect(stage_key=stage_key)
    assert collected is not None
    assert collected["stage_key"] == stage_key
    assert collected["attempts"][0]["execution_id"] == "opaque/task?id=1"
    assert collected["status"] == "completed"


def test_ambiguous_create_recovers_without_duplicate_through_concrete_backend() -> None:
    client = CombinedClient(ambiguous_create=True)
    observed = _backend(client).admit(stage_input=_training_stage_input())
    assert observed["state"] == "active"
    assert client.create_count == 1
    assert len(client.records) == 1


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("stage_input", [_training_stage_input(), _evaluation_stage_input()])
def test_concrete_collect_preserves_terminal_training_and_evaluation_statuses(
    stage_input: StageInput, status: str
) -> None:
    client = CombinedClient()
    backend = _backend(client)
    admitted = backend.admit(stage_input=deepcopy(stage_input))
    task_id = admitted["execution_ids"][0]
    client.projections[task_id] = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=[task_id],
        terminal_candidates=[_candidate(stage_input, task_id, status)],
    )
    result = backend.collect(stage_key=_stage_key_from_input(stage_input))
    assert result is not None
    assert result["status"] == status


def test_concrete_cancel_and_optional_logs_remain_outside_required_port() -> None:
    client = CombinedClient()
    backend = _backend(client)
    stage_input = _training_stage_input()
    observed = backend.admit(stage_input=stage_input)
    task_id = observed["execution_ids"][0]
    client.logs[task_id] = ClearMLLogData(
        chunks=("hello",), locator="clearml://opaque/log"
    )

    backend.cancel_study(study_result=stage_input["study_result"])
    assert client.cancelled == [task_id]
    log = backend.read_task_logs(
        study_result=stage_input["study_result"], task_id=task_id
    )
    assert log is not None
    assert log.chunks == ("hello",)


def test_registry_activation_is_explicit_generic_clearml_type() -> None:
    client = CombinedClient()
    registry = BackendRegistry()
    register_clearml_backend(registry)
    resolved = registry.resolve(BackendConfig("clearml", {"client": client}))
    assert isinstance(resolved, ClearMLBackend)
    assert resolved.admit(stage_input=_training_stage_input())["state"] == "active"


def test_duplicate_ownership_remains_bounded_through_concrete_backend() -> None:
    client = CombinedClient()
    backend = _backend(client)
    stage_input = _training_stage_input()
    backend.admit(stage_input=stage_input)
    client.records.append(deepcopy(client.records[0]))
    with pytest.raises(RuntimeError, match="multiple ClearML Tasks"):
        backend.observe(stage_key=_stage_key_from_input(stage_input))


def test_sdk_configuration_recursively_normalizes_mapping_subclasses() -> None:
    class NestedConfigTask:
        def get_configuration_object_as_dict(self, name):
            assert name == "mldb.runtime_snapshots"
            return OrderedDict(
                {
                    "study_plan": OrderedDict(
                        {
                            "pins": [
                                OrderedDict({"kind": "study", "id": "demo/study"})
                            ]
                        }
                    )
                }
            )

    normalized = _configuration(NestedConfigTask(), "mldb.runtime_snapshots")

    assert normalized is not None
    assert type(normalized) is dict
    plan = normalized["study_plan"]
    assert type(plan) is dict
    pins = plan["pins"]
    assert type(pins) is list
    assert type(pins[0]) is dict


class FakeScalarLogger:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def report_scalar(self, **kwargs) -> None:
        self.calls.append(dict(kwargs))


class FakeScalarTask:
    def __init__(self, logger: FakeScalarLogger) -> None:
        self.logger = logger
        self.get_logger_calls = 0

    def get_logger(self) -> FakeScalarLogger:
        self.get_logger_calls += 1
        return self.logger


def test_clearml_scalar_sink_maps_generic_events_exactly_and_lazily() -> None:
    logger = FakeScalarLogger()
    task = FakeScalarTask(logger)
    sink = _ClearMLScalarSink(task)

    assert task.get_logger_calls == 0
    sink(
        _AcceptedScalarEvent(
            group="optimization",
            series="cross_entropy_loss",
            value=1.25,
            step=3,
        )
    )
    sink(
        _AcceptedScalarEvent(
            group="validation",
            series="mean_iou",
            value=0.7,
            step=5,
        )
    )

    assert task.get_logger_calls == 1
    assert logger.calls == [
        {
            "title": "optimization",
            "series": "cross_entropy_loss",
            "value": 1.25,
            "iteration": 3,
        },
        {
            "title": "validation",
            "series": "mean_iou",
            "value": 0.7,
            "iteration": 5,
        },
    ]


class FakeSDKTask:
    tasks: list["FakeSDKTask"] = []
    credentials_calls: list[dict[str, object]] = []
    create_calls: list[dict[str, object]] = []
    enqueue_calls: list[tuple[str, str | None]] = []
    dequeue_calls: list[str] = []
    docker_calls: list[dict[str, object]] = []
    package_calls: list[list[str]] = []

    def __init__(self, *, project: str, task_id: str) -> None:
        self.project = project
        self.id = task_id
        self.status = "created"
        self.tags: list[str] = []
        self.properties: dict[str, str] = {}
        self.configs: dict[str, dict[str, object]] = {}
        self.parameters: dict[str, object] = {}
        self.logs: list[str] = ["sdk-log"]

    @classmethod
    def reset(cls) -> None:
        cls.tasks = []
        cls.credentials_calls = []
        cls.create_calls = []
        cls.enqueue_calls = []
        cls.dequeue_calls = []
        cls.docker_calls = []
        cls.package_calls = []

    @classmethod
    def set_credentials(cls, **kwargs):
        cls.credentials_calls.append(kwargs)

    @classmethod
    def create(cls, **kwargs):
        cls.create_calls.append(kwargs)
        task = cls(project=cast(str, kwargs["project_name"]), task_id=f"sdk-{len(cls.tasks)+1}")
        cls.tasks.append(task)
        return task

    @classmethod
    def get_tasks(cls, *, project_name, tags, allow_archived):
        return [
            task for task in cls.tasks
            if task.project == project_name and all(tag in task.tags for tag in tags)
        ]

    @classmethod
    def get_task(cls, *, task_id):
        return next((task for task in cls.tasks if task.id == task_id), None)

    @classmethod
    def enqueue(cls, *, task, queue_name=None):
        task.status = "queued"
        cls.enqueue_calls.append((task.id, queue_name))

    @classmethod
    def dequeue(cls, task):
        task.status = "created"
        cls.dequeue_calls.append(task.id)

    def set_base_docker(self, **kwargs):
        type(self).docker_calls.append(deepcopy(kwargs))

    def set_packages(self, packages):
        type(self).package_calls.append(list(packages))

    def set_user_properties(self, *properties):
        for item in properties:
            self.properties[item["name"]] = str(item["value"])

    def get_user_properties(self, value_only=False):
        return dict(self.properties)

    def set_tags(self, tags):
        self.tags = list(tags)

    def set_configuration_object(self, *, name, config_dict):
        self.configs[name] = deepcopy(config_dict)

    def get_configuration_object_as_dict(self, name):
        value = self.configs.get(name)
        return OrderedDict(deepcopy(value)) if value is not None else None

    def set_parameters_as_dict(self, value):
        self.parameters = deepcopy(value)

    def get_project_name(self):
        return self.project

    def get_status(self):
        return self.status

    def mark_stop_request(self, **kwargs):
        self.status = "stopped"

    def mark_stopped(self, **kwargs):
        self.status = "stopped"

    def get_reported_console_output(self, number_of_reports=1):
        return self.logs[-number_of_reports:]

    def get_output_log_web_page(self):
        return f"https://clearml.invalid/{self.id}/logs"


def _sdk_request(stage_input: StageInput) -> ClearMLCreateRequest:
    ownership = _ownership_key(stage_input)
    metadata = {
        "mldb.namespace": "demo",
        "mldb.study_result": str(stage_input["study_result"]),
        "mldb.trial": str(stage_input["trial"]),
        "mldb.stage_kind": str(stage_input["kind"]),
        "mldb.ownership_key": ownership,
    }
    if stage_input["kind"] == "evaluation":
        metadata["mldb.evaluation_stage"] = "holdout"
        metadata["mldb.evaluation_coordinate"] = str(stage_input["coordinate"])
    from mldb_v2.src.backend._clearml_admission import _transport_stage_input
    configuration = {
        "mldb.ownership_key": ownership,
        "mldb.stage_input": _transport_stage_input(stage_input),
        "mldb.public_parameters": deepcopy(stage_input["stage"]["parameters"]),
        "mldb.harness": "mldb_v2.src.backend.execution_harness.CommonExecutionHarness",
        "mldb.source_commit": stage_input["source_commit"],
    }
    return ClearMLCreateRequest(
        project="mldb/demo",
        task_name="presentation only",
        metadata=metadata,
        configuration=configuration,
        launch=ClearMLRemoteLaunch(
            source_commit=stage_input["source_commit"],
            harness_symbol="mldb_v2.src.backend.execution_harness.CommonExecutionHarness",
            stage_input_json=cast(str, configuration["mldb.stage_input"]),
            queue="gpu-a",
        ),
    )


def test_production_sdk_adapter_uses_lazy_credentials_searchable_metadata_and_queue() -> None:
    FakeSDKTask.reset()
    settings = ClearMLSDKSettings(
        api_host="https://api.invalid",
        web_host="https://web.invalid",
        files_host="https://files.invalid",
        access_key="access-secret-value",
        secret_key="secret-secret-value",
        repository="https://git.invalid/repo.git",
        docker_image="python:3.10-slim-bookworm",
        docker_env_file="/srv/bugrat/clearml/.env",
        docker_gpu="all",
        docker_shm_size="2g",
        s3_endpoint_url="https://s3.invalid",
        s3_region="test-region",
    )
    adapter = ClearMLSDKAdapter(settings, task_class=FakeSDKTask)
    request = _sdk_request(_training_stage_input())
    task_id = adapter.create_task(request)

    assert task_id == "sdk-1"
    assert FakeSDKTask.enqueue_calls == [("sdk-1", "gpu-a")]
    assert FakeSDKTask.create_calls[0]["commit"] == "2" * 40
    assert FakeSDKTask.create_calls[0]["script"] == "mldb_v2/src/backend/_clearml_sdk.py"
    assert FakeSDKTask.docker_calls == [{
        "docker_image": "python:3.10-slim-bookworm",
        "docker_arguments": [
            "--gpus", "all",
            "--shm-size", "2g",
            "-e", "AWS_ACCESS_KEY_ID",
            "-e", "AWS_SECRET_ACCESS_KEY",
            "-e", "AWS_SESSION_TOKEN",
            "-e", "MINIO_ROOT_USER",
            "-e", "MINIO_ROOT_PASSWORD",
            "--env-file=/srv/bugrat/clearml/.env",
        ],
    }]
    assert FakeSDKTask.package_calls and "torch==2.5.1" in FakeSDKTask.package_calls[0]
    assert FakeSDKTask.tasks[0].configs["mldb.runtime"]["s3_endpoint_url"] == "https://s3.invalid"
    assert FakeSDKTask.tasks[0].configs["mldb.runtime"]["s3_region"] == "test-region"
    assert FakeSDKTask.credentials_calls[0]["store_conf_file"] is False
    assert "access-secret-value" not in repr(settings)
    assert "secret-secret-value" not in repr(settings)

    found = adapter.search_tasks(project="mldb/demo", ownership_key=request.metadata["mldb.ownership_key"])
    assert [record.task_id for record in found] == ["sdk-1"]
    assert found[0].metadata["mldb.study_result"] == "demo/run-abcd"
    assert found[0].configuration["mldb.stage_input"] == request.configuration["mldb.stage_input"]
    assert adapter.read_runtime_projection(task_id="sdk-1").state == "active"

    owned = adapter.search_tasks_by_metadata(
        project="mldb/demo", key="mldb.study_result", value="demo/run-abcd"
    )
    assert [item.task_id for item in owned] == ["sdk-1"]
    assert owned[0].cancellation_state is ClearMLCancellationState.ACTIVE
    log = adapter.read_task_logs(task_id="sdk-1")
    assert log is not None and log.chunks == ("sdk-log",)
    adapter.request_cancellation(task_id="sdk-1")
    assert FakeSDKTask.dequeue_calls == ["sdk-1"]


def test_sdk_terminal_failure_without_harness_payload_is_safe_failure_not_success() -> None:
    FakeSDKTask.reset()
    adapter = ClearMLSDKAdapter(task_class=FakeSDKTask)
    request = _sdk_request(_training_stage_input())
    task_id = cast(str, adapter.create_task(request))
    task = FakeSDKTask.get_task(task_id=task_id)
    assert task is not None
    task.status = "failed"
    projection = adapter.read_runtime_projection(task_id=task_id)
    assert projection.state == "terminal"
    candidate = cast(dict[str, object], projection.terminal_candidates[0])
    assert candidate["status"] == "failed"
    assert candidate["result"] is None


def test_sdk_success_without_common_harness_projection_is_rejected() -> None:
    FakeSDKTask.reset()
    adapter = ClearMLSDKAdapter(task_class=FakeSDKTask)
    task_id = cast(str, adapter.create_task(_sdk_request(_training_stage_input())))
    task = FakeSDKTask.get_task(task_id=task_id)
    assert task is not None
    task.status = "completed"
    with pytest.raises(RuntimeError, match="without harness projection"):
        adapter.read_runtime_projection(task_id=task_id)


def test_genericity_and_secret_boundaries() -> None:
    root = Path(__file__).parents[1]
    backend_source = (root / "src/backend/clearml_backend.py").read_text(encoding="utf-8")
    sdk_source = (root / "src/backend/_clearml_sdk.py").read_text(encoding="utf-8")
    combined = backend_source + sdk_source

    assert "mldb_v2.skeleton" not in combined
    assert "result_acceptance" not in combined
    assert "study_driver" not in combined
    assert "classifier" not in combined.lower()
    assert "detector" not in combined.lower()
    assert "rotated-fcos" not in combined.lower()
    assert "def read_task_logs" in backend_source
    assert "CommonExecutionHarness" in sdk_source
    assert "Task.create" not in backend_source
    assert "from clearml import Task" not in backend_source
    assert sdk_source.index("from clearml import Task") > sdk_source.index("def _load_task_class")
    assert "split(request.task_name" not in combined
    assert "parse" not in backend_source.lower()


def test_sdk_publishing_task_is_cancelled_as_active_work() -> None:
    FakeSDKTask.reset()
    adapter = ClearMLSDKAdapter(task_class=FakeSDKTask)
    task_id = cast(str, adapter.create_task(_sdk_request(_training_stage_input())))
    task = FakeSDKTask.get_task(task_id=task_id)
    assert task is not None
    task.status = "publishing"

    adapter.request_cancellation(task_id=task_id)

    assert task.status == "stopped"
