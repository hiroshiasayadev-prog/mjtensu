"""ClearML-only deterministic admission for one exact MLDB StageInput."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from mldb_v2.src.backend.execution_harness import CommonExecutionHarness
from mldb_v2.src.backend.stage_input import StageInput
from mldb_v2.src.common.ids import (
    _canonical_json_bytes,
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
    _validate_typed_reference,
)

_HARNESS_SYMBOL = (
    f"{CommonExecutionHarness.__module__}.{CommonExecutionHarness.__qualname__}"
)
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


class ClearMLAdmissionError(RuntimeError):
    """Bounded ClearML admission/recovery failure."""


@dataclass(frozen=True)
class ClearMLTaskRecord:
    """SDK-neutral searchable projection of one ClearML Task."""

    task_id: str
    metadata: Mapping[str, str]
    configuration: Mapping[str, object]


@dataclass(frozen=True)
class ClearMLRemoteLaunch:
    """Operational-only remote launch description."""

    source_commit: str
    harness_symbol: str
    stage_input_json: str
    queue: str | None
    pipeline_execution_id: str | None = None
    pipeline_step: str | None = None


@dataclass(frozen=True)
class ClearMLCreateRequest:
    """SDK-neutral request consumed by the production ClearML activation seam."""

    project: str
    task_name: str
    metadata: Mapping[str, str]
    configuration: Mapping[str, object]
    launch: ClearMLRemoteLaunch


@dataclass(frozen=True)
class ClearMLAdmissionResult:
    task_id: str
    project: str
    ownership_key: str
    recovered: bool


class ClearMLAdmissionClient(Protocol):
    """Minimal mocked/production seam needed by admission only."""

    def search_tasks(
        self, *, project: str, ownership_key: str
    ) -> Sequence[ClearMLTaskRecord]: ...

    def create_task(self, request: ClearMLCreateRequest) -> str | None: ...


def _namespace_of(reference: object) -> str:
    return _validate_typed_reference(reference).split("/", 1)[0]


def _study_id_from_pins(stage_input: StageInput) -> str:
    pins = stage_input["pins"]
    if type(pins) is not list:
        raise ValueError("StageInput pins must be a list")
    studies = [
        pin.get("id")
        for pin in pins
        if type(pin) is dict and pin.get("kind") == "study"
    ]
    if len(studies) != 1:
        raise ValueError("StageInput must contain exactly one Study pin")
    return _validate_typed_reference(studies[0])


def _validate_stage_input_identity(stage_input: StageInput) -> tuple[str, str]:
    if type(stage_input) is not dict or set(stage_input) != _STAGE_INPUT_FIELDS:
        raise ValueError("StageInput fields do not match schema")
    if stage_input["schema"] != "mjtensu.mldb-v2/stage-input/v1":
        raise ValueError("unsupported StageInput schema")
    study_result = _validate_typed_reference(stage_input["study_result"])
    _validate_typed_reference(stage_input["plan"])
    _validate_trial_id(stage_input["trial"])
    kind = stage_input["kind"]
    if kind == "training":
        if stage_input["coordinate"] is not None or stage_input["runtime_model"] is not None:
            raise ValueError("training StageInput has invalid coordinate/runtime_model")
    elif kind == "evaluation":
        _validate_evaluation_coordinate_id(stage_input["coordinate"])
        if type(stage_input["runtime_model"]) is not dict:
            raise ValueError("evaluation StageInput requires runtime_model")
    else:
        raise ValueError("StageInput kind must be training or evaluation")
    study_id = _study_id_from_pins(stage_input)
    namespace = _namespace_of(study_result)
    if _namespace_of(study_id) != namespace:
        raise ValueError("Study pin namespace does not match StudyResult namespace")
    return namespace, study_id


def _ownership_key(stage_input: StageInput) -> str:
    _validate_stage_input_identity(stage_input)
    payload = {
        "study_result": stage_input["study_result"],
        "trial": stage_input["trial"],
        "kind": stage_input["kind"],
        "coordinate": stage_input["coordinate"],
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return f"mldb-v2-stage:{digest}"


def _transport_stage_input(stage_input: StageInput) -> str:
    _validate_stage_input_identity(stage_input)
    return _canonical_json_bytes(stage_input).decode("utf-8")


def _restore_stage_input(transport: str) -> StageInput:
    if type(transport) is not str:
        raise ValueError("StageInput transport must be JSON text")
    value = json.loads(transport)
    validated = cast(StageInput, value)
    _validate_stage_input_identity(validated)
    if _transport_stage_input(validated) != transport:
        raise ValueError("StageInput transport is not canonical exact JSON")
    return validated


def _metadata(stage_input: StageInput, *, study_id: str, ownership_key: str) -> dict[str, str]:
    stage = stage_input["stage"]
    if type(stage) is not dict:
        raise ValueError("StageInput stage must be a mapping")
    metadata = {
        "mldb.namespace": _namespace_of(stage_input["study_result"]),
        "mldb.study": study_id,
        "mldb.plan": _validate_typed_reference(stage_input["plan"]),
        "mldb.study_result": _validate_typed_reference(stage_input["study_result"]),
        "mldb.trial": str(_validate_trial_id(stage_input["trial"])),
        "mldb.stage_kind": str(stage_input["kind"]),
        "mldb.task": _validate_typed_reference(stage.get("task")),
        "mldb.corpus": _validate_typed_reference(stage.get("corpus")),
        "mldb.source_commit": str(stage_input["source_commit"]),
        "mldb.ownership_key": ownership_key,
    }
    if stage_input["kind"] == "training":
        metadata["mldb.architecture"] = _validate_typed_reference(stage.get("architecture"))
        metadata["mldb.protocol"] = _validate_typed_reference(stage.get("train_protocol"))
    else:
        runtime_model = stage_input["runtime_model"]
        if type(runtime_model) is not dict:
            raise ValueError("evaluation StageInput requires runtime_model")
        stage_name = stage.get("name")
        if type(stage_name) is not str or not stage_name:
            raise ValueError("evaluation stage name must be non-empty")
        metadata["mldb.evaluation_stage"] = stage_name
        metadata["mldb.evaluation_coordinate"] = str(
            _validate_evaluation_coordinate_id(stage_input["coordinate"])
        )
        metadata["mldb.model"] = _validate_typed_reference(runtime_model.get("model"))
        metadata["mldb.architecture"] = _validate_typed_reference(
            runtime_model.get("architecture")
        )
        metadata["mldb.protocol"] = _validate_typed_reference(
            stage.get("evaluation_protocol")
        )
    return metadata


def _local_id(reference: object) -> str:
    return _validate_typed_reference(reference).split("/", 1)[1]


def _task_name(stage_input: StageInput, *, study_id: str) -> str:
    stage = stage_input["stage"]
    if type(stage) is not dict:
        raise ValueError("StageInput stage must be a mapping")
    if stage_input["kind"] == "training":
        architecture = _local_id(stage.get("architecture"))
        stage_label = "train"
    else:
        runtime_model = stage_input["runtime_model"]
        if type(runtime_model) is not dict:
            raise ValueError("evaluation StageInput requires runtime_model")
        architecture = _local_id(runtime_model.get("architecture"))
        stage_label = stage.get("name")
        if type(stage_label) is not str or not stage_label:
            raise ValueError("evaluation stage name must be non-empty")
    return f"{architecture} | {stage_label} | {_local_id(study_id)} | {stage_input['trial']}"


def _pipeline_step_name(stage_input: StageInput) -> str:
    if stage_input["kind"] == "training":
        return f"{stage_input['trial']}-train"
    return f"{stage_input['trial']}-{stage_input['coordinate']}"


def _canonical_copy(value: object) -> object:
    return json.loads(_canonical_json_bytes(value))


def _matches(
    task: ClearMLTaskRecord,
    *,
    metadata: Mapping[str, str],
    stage_input_json: str,
    ownership_key: str,
) -> bool:
    if type(task.task_id) is not str or not task.task_id:
        return False
    if any(task.metadata.get(key) != value for key, value in metadata.items()):
        return False
    return (
        task.configuration.get("mldb.ownership_key") == ownership_key
        and task.configuration.get("mldb.stage_input") == stage_input_json
        and task.configuration.get("mldb.harness") == _HARNESS_SYMBOL
        and task.configuration.get("mldb.source_commit") == metadata["mldb.source_commit"]
    )


def _select_existing(
    records: Sequence[ClearMLTaskRecord],
    *,
    metadata: Mapping[str, str],
    stage_input_json: str,
    ownership_key: str,
) -> ClearMLTaskRecord | None:
    records = tuple(records)
    if not records:
        return None
    if len(records) != 1:
        raise ClearMLAdmissionError("multiple ClearML Tasks claim one logical ownership key")
    record = records[0]
    if not _matches(
        record,
        metadata=metadata,
        stage_input_json=stage_input_json,
        ownership_key=ownership_key,
    ):
        raise ClearMLAdmissionError("ClearML ownership metadata or StageInput identity mismatch")
    return record


def _stage_name(stage_input: StageInput) -> str:
    if stage_input["kind"] == "training":
        return "training"
    stage = stage_input["stage"]
    if type(stage) is not dict:
        raise ValueError("StageInput stage must be a mapping")
    name = stage.get("name")
    if type(name) is not str or not name:
        raise ValueError("Evaluation StageInput stage name must be a non-empty string")
    return name


def _queue_for_stage(
    stage_input: StageInput,
    *,
    default_queue: str | None,
    stage_routes: Mapping[str, Mapping[str, object]],
) -> str | None:
    route = stage_routes.get(_stage_name(stage_input))
    if route is None or "queue" not in route:
        return default_queue
    queue = route["queue"]
    if type(queue) is not str or not queue:
        raise ValueError("ClearML stage route queue must be a non-empty string")
    return queue


class ClearMLAdmissionService:
    """Idempotently admit exactly one ready StageInput to ClearML."""

    def __init__(
        self,
        *,
        client: ClearMLAdmissionClient,
        queue: str | None = None,
        stage_routes: Mapping[str, Mapping[str, object]] | None = None,
        recovery_search_attempts: int = 3,
    ) -> None:
        if queue is not None and (type(queue) is not str or not queue):
            raise ValueError("queue must be null or a non-empty string")
        if type(recovery_search_attempts) is not int or recovery_search_attempts < 1:
            raise ValueError("recovery_search_attempts must be a positive integer")
        self._client = client
        self._queue = queue
        self._stage_routes = {key: dict(value) for key, value in (stage_routes or {}).items()}
        self._recovery_search_attempts = recovery_search_attempts

    def _search(
        self,
        *,
        project: str,
        ownership_key: str,
        metadata: Mapping[str, str],
        stage_input_json: str,
    ) -> ClearMLTaskRecord | None:
        try:
            records = self._client.search_tasks(
                project=project, ownership_key=ownership_key
            )
        except Exception as error:
            raise ClearMLAdmissionError("ClearML ownership search failed") from error
        return _select_existing(
            records,
            metadata=metadata,
            stage_input_json=stage_input_json,
            ownership_key=ownership_key,
        )

    def _bind_pipeline_task(
        self,
        *,
        task_id: str,
        pipeline_execution_id: str | None,
        pipeline_step: str | None,
    ) -> None:
        if pipeline_execution_id is None:
            return
        if pipeline_step is None:
            raise ClearMLAdmissionError("Pipeline-bound Task is missing pipeline step identity")
        binder = getattr(self._client, "bind_task_to_pipeline", None)
        if not callable(binder):
            raise ClearMLAdmissionError("ClearML client cannot bind Task to Pipeline")
        try:
            binder(
                task_id=task_id,
                pipeline_execution_id=pipeline_execution_id,
                pipeline_step=pipeline_step,
            )
        except Exception as error:
            raise ClearMLAdmissionError("ClearML Pipeline Task binding failed") from error

    def admit(
        self,
        *,
        stage_input: StageInput,
        pipeline_execution_id: str | None = None,
    ) -> ClearMLAdmissionResult:
        if pipeline_execution_id is not None and (
            type(pipeline_execution_id) is not str or not pipeline_execution_id
        ):
            raise ValueError("pipeline_execution_id must be null or a non-empty string")
        namespace, study_id = _validate_stage_input_identity(stage_input)
        project = f"mldb/{namespace}"
        ownership_key = _ownership_key(stage_input)
        stage_input_json = _transport_stage_input(stage_input)
        pipeline_step = (
            None if pipeline_execution_id is None else _pipeline_step_name(stage_input)
        )
        metadata = _metadata(
            stage_input, study_id=study_id, ownership_key=ownership_key
        )
        if pipeline_execution_id is not None:
            metadata["mldb.pipeline_execution"] = pipeline_execution_id
            metadata["mldb.pipeline_step"] = cast(str, pipeline_step)
        existing = self._search(
            project=project,
            ownership_key=ownership_key,
            metadata=metadata,
            stage_input_json=stage_input_json,
        )
        if existing is not None:
            self._bind_pipeline_task(
                task_id=existing.task_id,
                pipeline_execution_id=pipeline_execution_id,
                pipeline_step=pipeline_step,
            )
            return ClearMLAdmissionResult(
                task_id=existing.task_id,
                project=project,
                ownership_key=ownership_key,
                recovered=True,
            )

        parameters = cast(dict[str, object], stage_input["stage"])["parameters"]
        configuration = {
            "mldb.ownership_key": ownership_key,
            "mldb.stage_input": stage_input_json,
            "mldb.public_parameters": _canonical_copy(parameters),
            "mldb.harness": _HARNESS_SYMBOL,
            "mldb.source_commit": stage_input["source_commit"],
        }
        if pipeline_execution_id is not None:
            configuration["mldb.pipeline_execution"] = pipeline_execution_id
            configuration["mldb.pipeline_step"] = cast(str, pipeline_step)
        request = ClearMLCreateRequest(
            project=project,
            task_name=_task_name(stage_input, study_id=study_id),
            metadata=metadata,
            configuration=configuration,
            launch=ClearMLRemoteLaunch(
                source_commit=str(stage_input["source_commit"]),
                harness_symbol=_HARNESS_SYMBOL,
                stage_input_json=stage_input_json,
                queue=_queue_for_stage(
                    stage_input,
                    default_queue=self._queue,
                    stage_routes=self._stage_routes,
                ),
                pipeline_execution_id=pipeline_execution_id,
                pipeline_step=pipeline_step,
            ),
        )

        create_id: str | None = None
        create_error: Exception | None = None
        try:
            create_id = self._client.create_task(request)
            if create_id is not None and (type(create_id) is not str or not create_id):
                raise TypeError("ClearML create response Task ID must be null or non-empty string")
        except Exception as error:
            create_error = error
            create_id = None

        found: ClearMLTaskRecord | None = None
        for _ in range(self._recovery_search_attempts):
            found = self._search(
                project=project,
                ownership_key=ownership_key,
                metadata=metadata,
                stage_input_json=stage_input_json,
            )
            if found is not None:
                break
        if found is None:
            if create_error is not None:
                raise ClearMLAdmissionError(
                    "ClearML create outcome is ambiguous and ownership could not be recovered"
                ) from create_error
            raise ClearMLAdmissionError(
                "ClearML Task creation did not yield a recoverable ownership record"
            )
        if create_id is not None and found.task_id != create_id:
            raise ClearMLAdmissionError(
                "ClearML create response Task ID does not match recovered ownership"
            )
        self._bind_pipeline_task(
            task_id=found.task_id,
            pipeline_execution_id=pipeline_execution_id,
            pipeline_step=pipeline_step,
        )
        return ClearMLAdmissionResult(
            task_id=found.task_id,
            project=project,
            ownership_key=ownership_key,
            recovered=create_error is not None or create_id is None,
        )