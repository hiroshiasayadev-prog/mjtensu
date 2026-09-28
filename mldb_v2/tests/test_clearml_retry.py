from __future__ import annotations

import copy

import pytest

import mldb_v2.tests.test_clearml_observation as obsfx
from mldb_v2.src.backend._clearml_admission import (
    ClearMLAdmissionService,
    ClearMLCreateRequest,
    ClearMLTaskRecord,
)
from mldb_v2.src.backend._clearml_observation import (
    ClearMLObservationService,
    ClearMLRuntimeProjection,
    _stage_key_from_input,
)
from mldb_v2.src.backend._clearml_retry import ClearMLRetryError, ClearMLRetryService


class RetryClient:
    def __init__(self, *, ambiguous_retry_create: bool = False) -> None:
        self.records: list[ClearMLTaskRecord] = []
        self.projections: dict[str, ClearMLRuntimeProjection] = {}
        self.owner_creates = 0
        self.retry_creates = 0
        self.ambiguous_retry_create = ambiguous_retry_create
        self.binds: list[tuple[str, str, str, str]] = []

    def search_tasks(self, *, project: str, ownership_key: str):
        return [
            item for item in self.records
            if item.metadata.get("mldb.ownership_key") == ownership_key
            and project == "mldb/" + item.metadata["mldb.namespace"]
        ]

    def create_task(self, request: ClearMLCreateRequest):
        self.owner_creates += 1
        task_id = f"owner-{self.owner_creates}"
        self.records.append(
            ClearMLTaskRecord(
                task_id=task_id,
                metadata=dict(request.metadata),
                configuration=copy.deepcopy(dict(request.configuration)),
            )
        )
        self.projections[task_id] = ClearMLRuntimeProjection(
            state="active", execution_ids=[task_id], terminal_candidates=[]
        )
        return task_id

    def search_retry_tasks(self, *, project: str, retry_key: str):
        return [
            item for item in self.records
            if item.metadata.get("mldb.retry_key") == retry_key
            and project == "mldb/" + item.metadata["mldb.namespace"]
        ]

    def create_retry_task(self, request: ClearMLCreateRequest):
        self.retry_creates += 1
        task_id = f"retry-{self.retry_creates + 1}"
        self.records.append(
            ClearMLTaskRecord(
                task_id=task_id,
                metadata=dict(request.metadata),
                configuration=copy.deepcopy(dict(request.configuration)),
            )
        )
        self.projections[task_id] = ClearMLRuntimeProjection(
            state="active", execution_ids=[task_id], terminal_candidates=[]
        )
        if self.ambiguous_retry_create:
            raise TimeoutError("created but response lost")
        return task_id

    def bind_retry_task_to_pipeline(
        self,
        *,
        task_id: str,
        owner_task_id: str,
        pipeline_execution_id: str,
        pipeline_step: str,
    ) -> None:
        self.binds.append((task_id, owner_task_id, pipeline_execution_id, pipeline_step))

    def read_runtime_projection(self, *, task_id: str):
        base = self.projections[task_id]
        retries = sorted(
            (
                item for item in self.records
                if item.metadata.get("mldb.retry_owner") == task_id
            ),
            key=lambda item: int(item.metadata["mldb.retry_index"]),
        )
        if not retries:
            return base
        execution_ids = list(base.execution_ids)
        terminal_candidates = list(base.terminal_candidates)
        last = base
        for record in retries:
            projected = self.projections[record.task_id]
            execution_ids.extend(projected.execution_ids)
            terminal_candidates.extend(projected.terminal_candidates)
            last = projected
        return ClearMLRuntimeProjection(
            state=last.state,
            execution_ids=execution_ids,
            terminal_candidates=terminal_candidates,
        )


def _setup(client: RetryClient):
    stage_input = obsfx._evaluation_stage_input()
    owner = ClearMLAdmissionService(client=client).admit(stage_input=stage_input)
    key = _stage_key_from_input(stage_input)
    client.projections[owner.task_id] = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=[owner.task_id],
        terminal_candidates=[
            obsfx._candidate(key, owner.task_id, "failed", index=0)
        ],
    )
    return stage_input, key, owner.task_id


def test_retry_creates_new_physical_task_without_second_logical_owner() -> None:
    client = RetryClient()
    stage_input, key, owner_id = _setup(client)
    service = ClearMLRetryService(client=client)

    observed = service.retry(
        stage_input=stage_input,
        prior_execution_ids=[owner_id],
    )

    assert observed["state"] == "active"
    assert observed["execution_ids"] == [owner_id, "retry-2"]
    assert client.owner_creates == 1
    assert client.retry_creates == 1
    owners = [
        item for item in client.records
        if item.metadata.get("mldb.ownership_key") is not None
    ]
    assert [item.task_id for item in owners] == [owner_id]
    retry = next(item for item in client.records if item.task_id == "retry-2")
    assert "mldb.ownership_key" not in retry.metadata
    assert retry.configuration["mldb.ownership_key"].startswith("mldb-v2-stage:")

    # Ordinary admit remains idempotent and recovers only the original owner.
    recovered = ClearMLAdmissionService(client=client).admit(stage_input=stage_input)
    assert recovered.task_id == owner_id
    assert recovered.recovered is True
    assert client.owner_creates == 1

    client.projections["retry-2"] = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=["retry-2"],
        terminal_candidates=[
            obsfx._candidate(key, "retry-2", "completed", index=1)
        ],
    )
    terminal = service.retry(
        stage_input=stage_input,
        prior_execution_ids=[owner_id],
    )
    assert terminal["state"] == "terminal"
    assert client.retry_creates == 1
    collected = ClearMLObservationService(client=client).collect(stage_key=key)
    assert collected is not None
    assert [item["execution_id"] for item in collected["attempts"]] == [
        owner_id, "retry-2"
    ]
    assert [item["status"] for item in collected["attempts"]] == [
        "failed", "completed"
    ]
    assert collected["status"] == "completed"


def test_retry_ambiguous_create_recovers_exactly_one_created_attempt() -> None:
    client = RetryClient(ambiguous_retry_create=True)
    stage_input, _key, owner_id = _setup(client)

    observed = ClearMLRetryService(client=client).retry(
        stage_input=stage_input,
        prior_execution_ids=[owner_id],
    )

    assert observed["execution_ids"] == [owner_id, "retry-2"]
    assert client.retry_creates == 1
    assert len([
        item for item in client.records
        if item.metadata.get("mldb.retry_owner") == owner_id
    ]) == 1


def test_retry_duplicate_retry_identity_fails_closed_without_creating_another() -> None:
    client = RetryClient()
    stage_input, _key, owner_id = _setup(client)
    service = ClearMLRetryService(client=client)
    service.retry(stage_input=stage_input, prior_execution_ids=[owner_id])
    duplicate = copy.deepcopy(
        next(item for item in client.records if item.metadata.get("mldb.retry_key"))
    )
    client.records.append(duplicate)

    with pytest.raises(ClearMLRetryError):
        service.retry(stage_input=stage_input, prior_execution_ids=[owner_id])
    assert client.retry_creates == 1


def test_retry_recovers_existing_pipeline_identity_from_owner_metadata() -> None:
    client = RetryClient()
    stage_input, _key, owner_id = _setup(client)
    owner = next(item for item in client.records if item.task_id == owner_id)
    owner.metadata["mldb.pipeline_execution"] = "pipeline-existing"
    owner.metadata["mldb.pipeline_step"] = "trial-0001-eval-0001"

    recovered = ClearMLRetryService(client=client).recover_pipeline_execution_id(
        stage_input=stage_input
    )

    assert recovered == "pipeline-existing"


def test_retry_pipeline_recovery_rejects_incomplete_or_wrong_owner_linkage() -> None:
    client = RetryClient()
    stage_input, _key, owner_id = _setup(client)
    owner = next(item for item in client.records if item.task_id == owner_id)
    owner.metadata["mldb.pipeline_execution"] = "pipeline-existing"

    with pytest.raises(ClearMLRetryError, match="linkage is incomplete"):
        ClearMLRetryService(client=client).recover_pipeline_execution_id(
            stage_input=stage_input
        )

    owner.metadata["mldb.pipeline_step"] = "wrong-step"
    with pytest.raises(ClearMLRetryError, match="does not match logical stage"):
        ClearMLRetryService(client=client).recover_pipeline_execution_id(
            stage_input=stage_input
        )
