from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from mldb_v2.src.backend._clearml_admission import ClearMLAdmissionService
from mldb_v2.src.backend._clearml_pipeline import (
    ClearMLPipelineCreateRequest,
    ClearMLPipelineError,
    ClearMLPipelineRecord,
    ClearMLPipelineService,
    _configuration,
    _metadata,
    _pipeline_ownership_key,
    _pipeline_topology,
)
from mldb_v2.src.backend._clearml_sdk import ClearMLSDKAdapter, ClearMLSDKSettings
from mldb_v2.src.backend._config import BackendConfig
from mldb_v2.src.backend.clearml_backend import clearml_backend_factory
from mldb_v2.src.results.study_result import StudyResult, _validate_study_result
from mldb_v2.src.study._plan_build import _content_digest, _plan_id, _validate_study_plan
from mldb_v2.src.study.execution_readiness import _materialize_stage_input
from mldb_v2.src.study.plan import StudyPlan


def _pin(kind: str, entity_id: str, *, companion: str | None = None,
         manifest: str | None = None, entries: int | None = None) -> dict[str, object]:
    return {
        "kind": kind, "id": entity_id, "yaml_sha256": "a" * 64,
        "companion_sha256": companion, "sources": [],
        "manifest_sha256": manifest, "manifest_entries": entries,
    }


def _plan() -> StudyPlan:
    record: dict[str, object] = {
        "schema": "mjtensu.mldb-v2/study-plan/v1",
        "id": "demo/placeholder",
        "content_sha256": "0" * 64,
        "study": "demo/study-v1",
        "source_commit": "1" * 40,
        "pins": [
            _pin("namespace", "demo"),
            _pin("task", "demo/task-v1"),
            _pin("corpus", "demo/eval-corpus-v1", manifest="b" * 64, entries=1),
            _pin("corpus", "demo/train-corpus-v1", manifest="c" * 64, entries=1),
            _pin("architecture", "demo/arch-v1", companion="d" * 64),
            _pin("train_protocol", "demo/train-v1", companion="e" * 64),
            _pin("evaluation_protocol", "demo/eval-v1", companion="f" * 64),
            _pin("study", "demo/study-v1"),
        ],
        "trials": [{
            "trial": "trial-0001",
            "source": {
                "kind": "training", "task": "demo/task-v1",
                "corpus": "demo/train-corpus-v1", "architecture": "demo/arch-v1",
                "train_protocol": "demo/train-v1", "parameters": {"epochs": 2}, "seed": 42,
            },
            "evaluations": [
                {"coordinate": "eval-0001", "stage": "quality", "task": "demo/task-v1",
                 "corpus": "demo/eval-corpus-v1", "evaluation_protocol": "demo/eval-v1", "parameters": {}},
                {"coordinate": "eval-0002", "stage": "deployment", "task": "demo/task-v1",
                 "corpus": "demo/eval-corpus-v1", "evaluation_protocol": "demo/eval-v1", "parameters": {}},
            ],
        }],
    }
    digest = _content_digest(record)
    record["content_sha256"] = digest
    record["id"] = _plan_id(cast(str, record["study"]), digest)
    return _validate_study_plan(record)


def _result(plan: StudyPlan, *, status: str = "submitted") -> StudyResult:
    execution_key = "123e4567e89b42d3a456426614174000"
    record = {
        "schema": "mjtensu.mldb-v2/study-result/v1",
        "id": f"demo/run-{execution_key}",
        "execution_key": execution_key,
        "plan": plan["id"],
        "study": plan["study"],
        "source_commit": plan["source_commit"],
        "backend": "clearml",
        "created_at": "2026-09-17T10:00:00Z",
        "status": status,
        "diagnostic": None,
        "trials": [{
            "trial": "trial-0001",
            "training": {"disposition": "pending", "result": None, "reason": None},
            "evaluations": [
                {"coordinate": "eval-0001", "stage": "quality", "disposition": "pending", "result": None, "reason": None},
                {"coordinate": "eval-0002", "stage": "deployment", "disposition": "pending", "result": None, "reason": None},
            ],
        }],
    }
    return _validate_study_result(record)


class FakePipelineClient:
    def __init__(self, *, ambiguous: bool = False) -> None:
        self.records: list[ClearMLPipelineRecord] = []
        self.requests: list[ClearMLPipelineCreateRequest] = []
        self.cancelled: list[str] = []
        self.ambiguous = ambiguous

    def search_pipeline_runs(self, *, project: str, ownership_key: str):
        return [record for record in self.records
                if record.metadata.get("mldb.pipeline_ownership_key") == ownership_key
                and project == "mldb/" + record.metadata["mldb.namespace"]]

    def create_pipeline_run(self, request: ClearMLPipelineCreateRequest) -> str | None:
        self.requests.append(request)
        task_id = f"pipeline-{len(self.records) + 1}"
        self.records.append(ClearMLPipelineRecord(
            task_id=task_id, metadata=dict(request.metadata),
            configuration=deepcopy(dict(request.configuration)), status="created",
        ))
        if self.ambiguous:
            raise TimeoutError("controller committed but response was lost")
        return task_id

    def request_cancellation(self, *, task_id: str) -> None:
        self.cancelled.append(task_id)


def test_pipeline_service_creates_one_controller_and_recovers_after_result_progress() -> None:
    plan = _plan()
    client = FakePipelineClient()
    service = ClearMLPipelineService(client=client)

    created = service.ensure(plan=plan, study_result=_result(plan))
    assert created["execution_id"] == "pipeline-1"
    assert created["status"] == "active"
    assert len(client.requests) == 1

    progressing = _result(plan, status="cancelling")
    replay = service.ensure(plan=plan, study_result=progressing)
    assert replay["execution_id"] == "pipeline-1"
    assert len(client.requests) == 1
    config = client.records[0].configuration
    assert "mldb.study_result" not in config
    assert config["mldb.study_result_identity"]["id"] == progressing["id"]


def test_pipeline_topology_preserves_trial_eval_stages_and_dependencies() -> None:
    plan = _plan()
    topology = _pipeline_topology(plan, _result(plan))
    steps = cast(list[dict[str, object]], topology["steps"])
    assert [(s["name"], s["stage"], s["parents"]) for s in steps] == [
        ("trial-0001-train", "training", []),
        ("trial-0001-eval-0001", "quality", ["trial-0001-train"]),
        ("trial-0001-eval-0002", "deployment", ["trial-0001-train"]),
    ]


def test_pipeline_ambiguous_create_recovers_without_duplicate() -> None:
    plan = _plan()
    client = FakePipelineClient(ambiguous=True)
    observed = ClearMLPipelineService(client=client).ensure(
        plan=plan, study_result=_result(plan)
    )
    assert observed["execution_id"] == "pipeline-1"
    assert len(client.records) == 1
    assert len(client.requests) == 1


def test_pipeline_cancellation_targets_controller_execution() -> None:
    plan = _plan()
    client = FakePipelineClient()
    service = ClearMLPipelineService(client=client)
    observed = service.ensure(plan=plan, study_result=_result(plan))

    service.cancel_execution(execution_id=observed["execution_id"])

    assert client.cancelled == ["pipeline-1"]


def test_pipeline_duplicate_ownership_is_bounded_failure() -> None:
    plan = _plan()
    result = _result(plan)
    client = FakePipelineClient()
    service = ClearMLPipelineService(client=client)
    service.ensure(plan=plan, study_result=result)
    client.records.append(deepcopy(client.records[0]))
    with pytest.raises(ClearMLPipelineError, match="multiple ClearML Pipeline"):
        service.observe(plan=plan, study_result=result)


class FakeSDKTask:
    tasks: list["FakeSDKTask"] = []
    create_calls: list[dict[str, object]] = []
    enqueue_calls: list[tuple[str, str | None]] = []

    def __init__(self, *, project: str, task_id: str) -> None:
        self.project = project
        self.id = task_id
        self.status = "created"
        self.tags: list[str] = []
        self.system_tags: list[str] = []
        self.properties: dict[str, str] = {}
        self.configs: dict[str, dict[str, object]] = {}
        self.packages: list[str] = []
        self.parameters: dict[str, object] = {}
        self.parent: str | None = None
        self.uploads: list[dict[str, object]] = []

    @classmethod
    def reset(cls) -> None:
        cls.tasks = []
        cls.create_calls = []
        cls.enqueue_calls = []

    @classmethod
    def set_credentials(cls, **kwargs) -> None:
        pass

    @classmethod
    def create(cls, **kwargs):
        cls.create_calls.append(dict(kwargs))
        task = cls(project=cast(str, kwargs["project_name"]), task_id=f"sdk-pipeline-{len(cls.tasks)+1}")
        cls.tasks.append(task)
        return task

    @classmethod
    def get_tasks(cls, *, project_name, tags, allow_archived):
        return [t for t in cls.tasks if t.project == project_name and all(tag in t.tags for tag in tags)]

    @classmethod
    def get_task(cls, *, task_id):
        return next((task for task in cls.tasks if task.id == task_id), None)

    @classmethod
    def enqueue(cls, *, task, queue_name=None) -> None:
        task.status = "queued"
        cls.enqueue_calls.append((task.id, queue_name))

    def set_user_properties(self, *properties) -> None:
        for item in properties:
            self.properties[item["name"]] = str(item["value"])

    def get_user_properties(self, value_only=False):
        return dict(self.properties)

    def set_tags(self, tags) -> None:
        self.tags = list(tags)

    def get_tags(self):
        return list(self.tags)

    def set_parent(self, parent) -> None:
        self.parent = str(parent)

    def set_parameters_as_dict(self, value) -> None:
        self.parameters = deepcopy(value)

    def get_system_tags(self):
        return list(self.system_tags)

    def set_system_tags(self, tags) -> None:
        self.system_tags = list(tags)

    def set_configuration_object(self, *, name, config_dict) -> None:
        self.configs[name] = deepcopy(config_dict)

    def get_configuration_object_as_dict(self, name):
        value = self.configs.get(name)
        return deepcopy(value) if value is not None else None

    def set_packages(self, packages) -> None:
        self.packages = list(packages)

    def upload_artifact(self, **kwargs) -> bool:
        self.uploads.append(deepcopy(kwargs))
        return True

    def get_project_name(self):
        return self.project

    def get_status(self):
        return self.status

    def mark_started(self, force=False) -> None:
        self.status = "in_progress"

    def mark_completed(self, ignore_errors=True, status_message=None, force=False):
        self.status = "completed"

    def mark_failed(self, ignore_errors=True, status_reason=None, status_message=None, force=False):
        self.status = "failed"

    def mark_stopped(self, force=False, status_message=None) -> None:
        self.status = "stopped"


def _pipeline_request(plan: StudyPlan, result: StudyResult) -> ClearMLPipelineCreateRequest:
    ownership = _pipeline_ownership_key(plan, result)
    return ClearMLPipelineCreateRequest(
        project="mldb/demo",
        task_name="study-v1 | run",
        metadata=_metadata(plan, result, ownership_key=ownership),
        configuration=_configuration(plan, result, ownership_key=ownership),
        source_commit=plan["source_commit"],
    )


def test_sdk_adapter_creates_native_controller_task_without_enqueuing_children() -> None:
    FakeSDKTask.reset()
    plan = _plan()
    result = _result(plan)
    adapter = ClearMLSDKAdapter(
        ClearMLSDKSettings(step_queue="gpu-a"),
        task_class=FakeSDKTask,
    )
    task_id = adapter.create_pipeline_run(_pipeline_request(plan, result))
    assert task_id == "sdk-pipeline-1"
    task = FakeSDKTask.tasks[0]
    assert FakeSDKTask.create_calls[0]["task_type"] == "controller"
    assert FakeSDKTask.create_calls[0]["script"].endswith("_clearml_pipeline_controller.py")
    assert "pipeline" in task.system_tags
    assert "Pipeline" in task.configs
    native = task.configs["Pipeline"]
    assert native["trial-0001-train"]["stage"] == "training"
    assert native["trial-0001-eval-0001"]["parents"] == ["trial-0001-train"]
    assert native["trial-0001-eval-0002"]["cache_executed_step"] is False
    assert task.status == "created"

    ownership = _pipeline_ownership_key(plan, result)
    found = adapter.search_pipeline_runs(project="mldb/demo", ownership_key=ownership)
    assert [record.task_id for record in found] == ["sdk-pipeline-1"]
    assert found[0].configuration == _pipeline_request(plan, result).configuration


def test_sdk_adapter_projects_one_bounded_study_summary_artifact() -> None:
    FakeSDKTask.reset()
    plan = _plan()
    result = _result(plan)
    adapter = ClearMLSDKAdapter(ClearMLSDKSettings(), task_class=FakeSDKTask)
    pipeline_id = cast(str, adapter.create_pipeline_run(_pipeline_request(plan, result)))
    summary = {
        "schema": "mjtensu.mldb-v2/study-summary-projection/v1",
        "study_result": result["id"],
        "study": result["study"],
        "status": "submitted",
        "rows": [{"trial": "trial-0001", "stage": "quality", "metrics": {"accuracy": 0.9}}],
    }

    adapter.project_pipeline_summary(execution_id=pipeline_id, summary=summary)

    controller = FakeSDKTask.get_task(task_id=pipeline_id)
    assert controller is not None
    assert controller.configs["mldb.study_summary"] == summary
    assert len(controller.uploads) == 1
    assert controller.uploads[0]["name"] == "mldb-study-summary"
    assert controller.uploads[0]["artifact_object"] == summary
    assert controller.status == "in_progress"


def test_pipeline_summary_artifact_upload_failure_is_observational() -> None:
    FakeSDKTask.reset()
    plan = _plan()
    result = _result(plan)
    adapter = ClearMLSDKAdapter(ClearMLSDKSettings(), task_class=FakeSDKTask)
    pipeline_id = cast(str, adapter.create_pipeline_run(_pipeline_request(plan, result)))
    controller = FakeSDKTask.get_task(task_id=pipeline_id)
    assert controller is not None

    def fail_upload(**_kwargs):
        raise RuntimeError("fileserver unavailable")

    controller.upload_artifact = fail_upload  # type: ignore[method-assign]
    summary = {
        "schema": "mjtensu.mldb-v2/study-summary-projection/v1",
        "study_result": result["id"],
        "study": result["study"],
        "status": "submitted",
        "rows": [],
    }

    adapter.project_pipeline_summary(execution_id=pipeline_id, summary=summary)

    assert controller.configs["mldb.study_summary"] == summary
    assert controller.status == "in_progress"


def test_concrete_backend_exposes_pipeline_capability_without_replacing_stage_port() -> None:
    plan = _plan()
    result = _result(plan)
    client = FakePipelineClient()
    backend = clearml_backend_factory(BackendConfig("clearml", {"client": client}))
    observed = backend.ensure_study_execution(plan=plan, study_result=result)
    assert observed["execution_id"] == "pipeline-1"
    assert backend.observe_study_execution(plan=plan, study_result=result) == observed


def test_generic_study_execution_seam_has_no_clearml_dependency() -> None:
    root = Path(__file__).parents[1]
    generic = (root / "src/backend/study_execution.py").read_text(encoding="utf-8")
    assert "clearml" not in generic.lower()
    assert "result_acceptance" not in generic
    assert "execution_readiness" not in generic


def test_ready_child_is_bound_to_pipeline_before_queue_and_replay_is_idempotent() -> None:
    FakeSDKTask.reset()
    plan = _plan()
    result = _result(plan)
    adapter = ClearMLSDKAdapter(
        ClearMLSDKSettings(step_queue="gpu-a"),
        task_class=FakeSDKTask,
    )
    pipeline_id = cast(str, adapter.create_pipeline_run(_pipeline_request(plan, result)))
    stage_input = _materialize_stage_input(
        plan=plan,
        result=result,
        stage={"kind": "training", "trial": "trial-0001"},
        mldb_data_root="unused",
    )
    service = ClearMLAdmissionService(client=adapter, queue="gpu-a")
    first = service.admit(
        stage_input=stage_input,
        pipeline_execution_id=pipeline_id,
    )
    replay = service.admit(
        stage_input=stage_input,
        pipeline_execution_id=pipeline_id,
    )
    assert first.task_id == replay.task_id == "sdk-pipeline-2"
    assert replay.recovered is True
    assert len(FakeSDKTask.create_calls) == 2
    assert FakeSDKTask.enqueue_calls == [("sdk-pipeline-2", "gpu-a")]

    child = FakeSDKTask.get_task(task_id="sdk-pipeline-2")
    controller = FakeSDKTask.get_task(task_id=pipeline_id)
    assert child is not None and controller is not None
    assert child.parent == pipeline_id
    assert f"pipe:{pipeline_id}" in child.tags
    assert child.properties["mldb.pipeline_execution"] == pipeline_id
    assert child.properties["mldb.pipeline_step"] == "trial-0001-train"
    node = controller.configs["Pipeline"]["trial-0001-train"]
    assert "job_id" not in node
    assert node["executed"] == child.id

    pending_summary = {
        "schema": "mjtensu.mldb-v2/study-summary-projection/v1",
        "study_result": result["id"],
        "study": result["study"],
        "status": "submitted",
        "rows": [{
            "trial": "trial-0001", "kind": "training", "stage": "training",
            "coordinate": None, "disposition": "pending", "result": None, "metrics": {},
        }],
    }
    adapter.project_pipeline_summary(execution_id=pipeline_id, summary=pending_summary)
    assert controller.configs["Pipeline"]["trial-0001-train"]["status"] == "queued"

    child.status = "in_progress"
    adapter.project_pipeline_summary(execution_id=pipeline_id, summary=pending_summary)
    assert controller.configs["Pipeline"]["trial-0001-train"]["status"] == "running"

    completed_summary = deepcopy(pending_summary)
    completed_summary["rows"][0]["disposition"] = "completed"
    completed_summary["rows"][0]["result"] = "demo/training-result-v1"
    adapter.project_pipeline_summary(execution_id=pipeline_id, summary=completed_summary)
    assert controller.configs["Pipeline"]["trial-0001-train"]["status"] == "completed"
