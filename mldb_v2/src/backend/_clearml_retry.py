"""ClearML same-logical-stage retry without creating a second ownership Task."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from mldb_v2.src.backend._clearml_admission import (
    ClearMLCreateRequest,
    ClearMLRemoteLaunch,
    ClearMLTaskRecord,
    _HARNESS_SYMBOL,
    _canonical_copy,
    _metadata,
    _ownership_key,
    _pipeline_step_name,
    _queue_for_stage,
    _select_existing,
    _task_name,
    _transport_stage_input,
    _validate_stage_input_identity,
)
from mldb_v2.src.backend._clearml_observation import (
    ClearMLObservationService,
    _stage_key_from_input,
)
from mldb_v2.src.backend.candidate_outcome import BackendObservation
from mldb_v2.src.backend.stage_input import StageInput
from mldb_v2.src.common.ids import _canonical_json_bytes


class ClearMLRetryError(RuntimeError):
    """Bounded retry creation/recovery failure."""


class ClearMLRetryClient(Protocol):
    def search_tasks(
        self, *, project: str, ownership_key: str
    ) -> Sequence[ClearMLTaskRecord]: ...

    def search_retry_tasks(
        self, *, project: str, retry_key: str
    ) -> Sequence[ClearMLTaskRecord]: ...

    def create_retry_task(self, request: ClearMLCreateRequest) -> str | None: ...

    def bind_retry_task_to_pipeline(
        self,
        *,
        task_id: str,
        owner_task_id: str,
        pipeline_execution_id: str,
        pipeline_step: str,
    ) -> None: ...

    def read_runtime_projection(self, *, task_id: str): ...


def _retry_key(ownership_key: str, retry_index: int) -> str:
    payload = {"ownership_key": ownership_key, "retry_index": retry_index}
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return f"mldb-v2-stage-retry:{digest}"


def _execution_ids(observation: BackendObservation) -> list[str]:
    if observation["state"] == "active":
        return list(observation["execution_ids"])
    return [str(item["execution_id"]) for item in observation["attempts"]]


def _retry_record_matches(
    record: ClearMLTaskRecord,
    *,
    metadata: Mapping[str, str],
    configuration: Mapping[str, object],
) -> bool:
    if type(record.task_id) is not str or not record.task_id:
        return False
    if "mldb.ownership_key" in record.metadata:
        return False
    if any(record.metadata.get(key) != value for key, value in metadata.items()):
        return False
    expected_keys = (
        "mldb.ownership_key",
        "mldb.stage_input",
        "mldb.harness",
        "mldb.source_commit",
        "mldb.retry_owner",
        "mldb.retry_key",
        "mldb.retry_index",
    )
    return all(record.configuration.get(key) == configuration.get(key) for key in expected_keys)


class ClearMLRetryService:
    """Create/recover exactly one additional physical attempt for a failed evaluation."""

    def __init__(
        self,
        *,
        client: ClearMLRetryClient,
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
        self._observation = ClearMLObservationService(client=cast(object, client))

    def _owner(
        self,
        *,
        stage_input: StageInput,
        pipeline_execution_id: str | None,
    ) -> tuple[ClearMLTaskRecord, str, str, str, str | None]:
        namespace, study_id = _validate_stage_input_identity(stage_input)
        if stage_input["kind"] != "evaluation":
            raise ClearMLRetryError("retry-stage currently supports evaluation stages only")
        project = f"mldb/{namespace}"
        ownership_key = _ownership_key(stage_input)
        stage_input_json = _transport_stage_input(stage_input)
        pipeline_step = (
            None if pipeline_execution_id is None else _pipeline_step_name(stage_input)
        )
        metadata = _metadata(
            stage_input, study_id=study_id, ownership_key=ownership_key
        )
        # Pipeline linkage is operational association, not logical ownership.
        # Recover the owner from the immutable stage identity even if an older
        # admission did not project pipeline metadata onto that Task.
        try:
            owner = _select_existing(
                self._client.search_tasks(
                    project=project, ownership_key=ownership_key
                ),
                metadata=metadata,
                stage_input_json=stage_input_json,
                ownership_key=ownership_key,
            )
        except Exception as error:
            raise ClearMLRetryError("ClearML retry owner lookup failed") from error
        if owner is None:
            raise ClearMLRetryError("ClearML retry owner Task was not found")
        return owner, project, ownership_key, stage_input_json, pipeline_step

    def _search_retry(
        self,
        *,
        project: str,
        retry_key: str,
        metadata: Mapping[str, str],
        configuration: Mapping[str, object],
    ) -> ClearMLTaskRecord | None:
        try:
            records = tuple(
                self._client.search_retry_tasks(project=project, retry_key=retry_key)
            )
        except Exception as error:
            raise ClearMLRetryError("ClearML retry Task search failed") from error
        if not records:
            return None
        if len(records) != 1:
            raise ClearMLRetryError("multiple ClearML Tasks claim one retry attempt")
        record = records[0]
        if not _retry_record_matches(
            record, metadata=metadata, configuration=configuration
        ):
            raise ClearMLRetryError("ClearML retry Task identity does not match request")
        return record

    def _bind(
        self,
        *,
        task_id: str,
        owner_task_id: str,
        pipeline_execution_id: str | None,
        pipeline_step: str | None,
    ) -> None:
        if pipeline_execution_id is None:
            return
        if pipeline_step is None:
            raise ClearMLRetryError("Pipeline-bound retry is missing pipeline step")
        try:
            self._client.bind_retry_task_to_pipeline(
                task_id=task_id,
                owner_task_id=owner_task_id,
                pipeline_execution_id=pipeline_execution_id,
                pipeline_step=pipeline_step,
            )
        except Exception as error:
            raise ClearMLRetryError("ClearML retry Pipeline binding failed") from error

    def recover_pipeline_execution_id(
        self,
        *,
        stage_input: StageInput,
    ) -> str | None:
        """Recover the existing controller identity from the logical owner Task."""
        owner, _project, _ownership_key_value, _stage_input_json, _pipeline_step = self._owner(
            stage_input=stage_input,
            pipeline_execution_id=None,
        )
        pipeline_execution = owner.metadata.get("mldb.pipeline_execution")
        pipeline_step = owner.metadata.get("mldb.pipeline_step")
        if pipeline_execution is None and pipeline_step is None:
            return None
        if (
            type(pipeline_execution) is not str
            or not pipeline_execution
            or type(pipeline_step) is not str
            or not pipeline_step
        ):
            raise ClearMLRetryError(
                "ClearML retry owner Pipeline linkage is incomplete"
            )
        expected_step = _pipeline_step_name(stage_input)
        if pipeline_step != expected_step:
            raise ClearMLRetryError(
                "ClearML retry owner Pipeline step does not match logical stage"
            )
        return pipeline_execution

    def retry(
        self,
        *,
        stage_input: StageInput,
        prior_execution_ids: Sequence[str],
        pipeline_execution_id: str | None = None,
    ) -> BackendObservation:
        if not prior_execution_ids or any(
            type(item) is not str or not item for item in prior_execution_ids
        ):
            raise ValueError("prior_execution_ids must be a non-empty sequence of opaque ids")
        if len(set(prior_execution_ids)) != len(prior_execution_ids):
            raise ValueError("prior_execution_ids must be unique")
        if pipeline_execution_id is not None and (
            type(pipeline_execution_id) is not str or not pipeline_execution_id
        ):
            raise ValueError("pipeline_execution_id must be null or a non-empty string")

        owner, project, ownership_key, stage_input_json, pipeline_step = self._owner(
            stage_input=stage_input,
            pipeline_execution_id=pipeline_execution_id,
        )
        stage_key = _stage_key_from_input(stage_input)
        try:
            observed = self._observation.observe(stage_key=stage_key)
        except Exception as error:
            raise ClearMLRetryError("ClearML retry owner projection is ambiguous") from error
        if observed is None:
            raise ClearMLRetryError("ClearML retry owner became unobservable")

        prior = list(prior_execution_ids)
        current_ids = _execution_ids(observed)
        next_index = len(prior) + 1
        retry_key = _retry_key(ownership_key, next_index)
        namespace, study_id = _validate_stage_input_identity(stage_input)
        del namespace
        retry_metadata = _metadata(
            stage_input, study_id=study_id, ownership_key=ownership_key
        )
        retry_metadata.pop("mldb.ownership_key")
        retry_metadata.update(
            {
                "mldb.retry_owner": owner.task_id,
                "mldb.retry_key": retry_key,
                "mldb.retry_index": str(next_index),
            }
        )
        configuration: dict[str, object] = {
            "mldb.ownership_key": ownership_key,
            "mldb.stage_input": stage_input_json,
            "mldb.public_parameters": _canonical_copy(
                cast(dict[str, object], stage_input["stage"])["parameters"]
            ),
            "mldb.harness": _HARNESS_SYMBOL,
            "mldb.source_commit": stage_input["source_commit"],
            "mldb.retry_owner": owner.task_id,
            "mldb.retry_key": retry_key,
            "mldb.retry_index": str(next_index),
        }
        if pipeline_execution_id is not None:
            retry_metadata["mldb.pipeline_execution"] = pipeline_execution_id
            retry_metadata["mldb.pipeline_step"] = cast(str, pipeline_step)
            configuration["mldb.pipeline_execution"] = pipeline_execution_id
            configuration["mldb.pipeline_step"] = cast(str, pipeline_step)

        existing_retry = self._search_retry(
            project=project,
            retry_key=retry_key,
            metadata=retry_metadata,
            configuration=configuration,
        )
        if current_ids == prior + ([existing_retry.task_id] if existing_retry else []):
            if existing_retry is not None:
                self._bind(
                    task_id=existing_retry.task_id,
                    owner_task_id=owner.task_id,
                    pipeline_execution_id=pipeline_execution_id,
                    pipeline_step=pipeline_step,
                )
                return observed
        elif current_ids != prior:
            raise ClearMLRetryError(
                "ClearML retry history does not exactly extend canonical attempts"
            )

        if current_ids != prior:
            raise ClearMLRetryError("ClearML retry projection is inconsistent")
        if observed["state"] != "terminal" or observed["status"] not in {"failed", "cancelled"}:
            raise ClearMLRetryError("ClearML retry requires failed terminal logical history")
        if existing_retry is not None:
            raise ClearMLRetryError(
                "ClearML retry Task exists but owner projection does not include it"
            )

        request = ClearMLCreateRequest(
            project=project,
            task_name=f"{_task_name(stage_input, study_id=study_id)} | retry {next_index}",
            metadata=retry_metadata,
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
                # Retry association is explicit: ordinary native-node binding would reject
                # a second physical Task for the same logical Pipeline step.
                pipeline_execution_id=None,
                pipeline_step=None,
            ),
        )

        create_id: str | None = None
        create_error: Exception | None = None
        try:
            create_id = self._client.create_retry_task(request)
            if create_id is not None and (type(create_id) is not str or not create_id):
                raise TypeError("ClearML retry create response id is malformed")
        except Exception as error:
            create_error = error
            create_id = None

        found: ClearMLTaskRecord | None = None
        for _ in range(self._recovery_search_attempts):
            found = self._search_retry(
                project=project,
                retry_key=retry_key,
                metadata=retry_metadata,
                configuration=configuration,
            )
            if found is not None:
                break
        if found is None:
            if create_error is not None:
                raise ClearMLRetryError(
                    "ClearML retry create outcome is ambiguous and could not be recovered"
                ) from create_error
            raise ClearMLRetryError("ClearML retry creation did not yield one recoverable Task")
        if create_id is not None and found.task_id != create_id:
            raise ClearMLRetryError(
                "ClearML retry create response id does not match recovered Task"
            )

        self._bind(
            task_id=found.task_id,
            owner_task_id=owner.task_id,
            pipeline_execution_id=pipeline_execution_id,
            pipeline_step=pipeline_step,
        )
        try:
            observed = self._observation.observe(stage_key=stage_key)
        except Exception as error:
            raise ClearMLRetryError(
                "ClearML retry Task was created but ordered attempt projection is ambiguous"
            ) from error
        if observed is None or _execution_ids(observed) != prior + [found.task_id]:
            raise ClearMLRetryError(
                "ClearML retry Task was created but ordered attempt projection is ambiguous"
            )
        return observed
