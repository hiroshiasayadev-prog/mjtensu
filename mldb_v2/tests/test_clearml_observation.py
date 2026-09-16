from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from mldb_v2.src.backend._clearml_admission import (
    ClearMLAdmissionService,
    ClearMLCreateRequest,
    ClearMLTaskRecord,
    _ownership_key as _admission_ownership_key,
)
from mldb_v2.src.backend._clearml_observation import (
    ClearMLObservationError,
    ClearMLObservationService,
    ClearMLRuntimeProjection,
    _ownership_key_for_stage_key,
    _stage_key_from_input,
)
from mldb_v2.src.backend.candidate_outcome import StageKey
from mldb_v2.src.backend.stage_input import (
    EvaluationStageInput,
    StageInput,
    TrainingStageInput,
)
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
            "parameters": {"batch_size": 8, "learning_rate": 0.001},
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
            "name": "final-holdout",
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
                "uri": "s3://bucket/model.pt",
                "bytes": 123,
                "sha256": "3" * 64,
                "format": "pytorch-state-dict/v1",
            },
        },
    }


class AdmissionClient:
    def __init__(self) -> None:
        self.records: list[ClearMLTaskRecord] = []

    def search_tasks(self, *, project: str, ownership_key: str):
        return [
            record
            for record in self.records
            if record.metadata.get("mldb.ownership_key") == ownership_key
        ]

    def create_task(self, request: ClearMLCreateRequest) -> str:
        task_id = f"clearml-owner-{len(self.records) + 1}"
        self.records.append(
            ClearMLTaskRecord(
                task_id=task_id,
                metadata=dict(request.metadata),
                configuration=deepcopy(dict(request.configuration)),
            )
        )
        return task_id


def _admitted_record(stage_input: StageInput) -> ClearMLTaskRecord:
    client = AdmissionClient()
    ClearMLAdmissionService(client=client).admit(stage_input=stage_input)
    assert len(client.records) == 1
    return client.records[0]


class ObservationClient:
    def __init__(
        self,
        *,
        records: list[ClearMLTaskRecord] | None = None,
        projections: dict[str, ClearMLRuntimeProjection] | None = None,
    ) -> None:
        self.records = records or []
        self.projections = projections or {}
        self.searches: list[tuple[str, str]] = []
        self.reads: list[str] = []

    def search_tasks(self, *, project: str, ownership_key: str):
        self.searches.append((project, ownership_key))
        return [
            record
            for record in self.records
            if record.metadata.get("mldb.ownership_key") == ownership_key
        ]

    def read_runtime_projection(self, *, task_id: str) -> ClearMLRuntimeProjection:
        self.reads.append(task_id)
        return self.projections[task_id]


def _diagnostic(status: str) -> dict[str, str]:
    return {"code": f"{status}_attempt", "message": f"{status} attempt"}


def _attempt(
    execution_id: str,
    status: str,
    *,
    index: int = 0,
) -> dict[str, object]:
    minute = index * 2
    diagnostic = None if status == "completed" else _diagnostic(status)
    return {
        "backend": "clearml",
        "execution_id": execution_id,
        "status": status,
        "started_at": f"2026-09-13T00:{minute:02d}:00Z",
        "ended_at": f"2026-09-13T00:{minute + 1:02d}:00Z",
        "diagnostic": diagnostic,
    }


def _completed_result(kind: str) -> dict[str, object]:
    if kind == "training":
        return {
            "weights": {
                "uri": "s3://bucket/weights.pt",
                "bytes": 123,
                "sha256": "4" * 64,
                "format": "pytorch-state-dict/v1",
            }
        }
    return {
        "metrics": {"f1": 0.967, "count": 7},
        "artifacts": {
            "predictions": {
                "uri": "s3://bucket/predictions.jsonl",
                "bytes": 987,
                "sha256": "5" * 64,
                "format": "jsonl",
                "schema": "demo/predictions/v1",
            }
        },
    }


def _candidate(
    stage_key: StageKey,
    execution_id: str,
    status: str,
    *,
    index: int = 0,
) -> dict[str, object]:
    diagnostic = None if status == "completed" else _diagnostic(status)
    result = _completed_result(stage_key["kind"]) if status == "completed" else None
    return {
        "state": "terminal",
        "stage_key": deepcopy(stage_key),
        "attempts": [_attempt(execution_id, status, index=index)],
        "status": status,
        "diagnostic": diagnostic,
        "result": result,
    }


def _client_for(
    stage_input: StageInput,
    projection: ClearMLRuntimeProjection,
) -> tuple[ObservationClient, StageKey, ClearMLTaskRecord]:
    record = _admitted_record(stage_input)
    client = ObservationClient(
        records=[record],
        projections={record.task_id: projection},
    )
    return client, _stage_key_from_input(stage_input), record


def test_ownership_key_matches_t005_02_exact_rule_for_training_and_evaluation() -> None:
    for stage_input in (_training_stage_input(), _evaluation_stage_input()):
        stage_key = _stage_key_from_input(stage_input)
        assert _ownership_key_for_stage_key(stage_key) == _admission_ownership_key(stage_input)


def test_zero_match_returns_none_for_observe_and_collect() -> None:
    stage_key = _stage_key_from_input(_training_stage_input())
    client = ObservationClient()
    service = ClearMLObservationService(client=client)

    assert service.observe(stage_key=stage_key) is None
    assert service.collect(stage_key=stage_key) is None
    assert client.reads == []
    assert client.searches[0][0] == "mldb/demo"


def test_active_training_observation_and_collect_none() -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    projection = ClearMLRuntimeProjection(
        state="active",
        execution_ids=[record.task_id],
        terminal_candidates=[],
    )
    client = ObservationClient(records=[record], projections={record.task_id: projection})
    service = ClearMLObservationService(client=client)

    observed = service.observe(stage_key=stage_key)
    assert observed == {
        "state": "active",
        "stage_key": stage_key,
        "backend": "clearml",
        "execution_ids": [record.task_id],
    }
    assert service.collect(stage_key=stage_key) is None


def test_active_evaluation_preserves_opaque_ordered_retry_ids() -> None:
    stage_input = _evaluation_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    retry_id = "opaque/retry#2?not-an-identity"
    projection = ClearMLRuntimeProjection(
        state="active",
        execution_ids=[record.task_id, retry_id],
        terminal_candidates=[_candidate(stage_key, record.task_id, "failed", index=0)],
    )
    client = ObservationClient(records=[record], projections={record.task_id: projection})

    observed = ClearMLObservationService(client=client).observe(stage_key=stage_key)

    assert observed is not None
    assert observed["state"] == "active"
    assert observed["execution_ids"] == [record.task_id, retry_id]


def test_duplicate_ownership_is_bounded_failure() -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    client = ObservationClient(records=[record, deepcopy(record)])

    with pytest.raises(ClearMLObservationError, match="multiple ClearML Tasks"):
        ClearMLObservationService(client=client).observe(stage_key=stage_key)


def test_stage_key_metadata_mismatch_is_rejected() -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    requested = _stage_key_from_input(stage_input)
    requested["plan"] = StudyPlanId("demo/other-plan-0123456789abcdef")
    client = ObservationClient(records=[record])

    with pytest.raises(ClearMLObservationError, match="StageKey"):
        ClearMLObservationService(client=client).observe(stage_key=requested)


def test_source_commit_configuration_mismatch_is_rejected() -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    broken_config = dict(record.configuration)
    broken_config["mldb.source_commit"] = "9" * 40
    broken = ClearMLTaskRecord(
        task_id=record.task_id,
        metadata=record.metadata,
        configuration=broken_config,
    )
    stage_key = _stage_key_from_input(stage_input)
    client = ObservationClient(records=[broken])

    with pytest.raises(ClearMLObservationError, match="source_commit"):
        ClearMLObservationService(client=client).observe(stage_key=stage_key)


def test_common_harness_configuration_mismatch_is_rejected() -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    broken_config = dict(record.configuration)
    broken_config["mldb.harness"] = "example.OtherHarness"
    broken = ClearMLTaskRecord(
        task_id=record.task_id,
        metadata=record.metadata,
        configuration=broken_config,
    )
    stage_key = _stage_key_from_input(stage_input)
    client = ObservationClient(records=[broken])

    with pytest.raises(ClearMLObservationError, match="harness"):
        ClearMLObservationService(client=client).observe(stage_key=stage_key)

def test_human_task_name_is_not_required_or_parsed() -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    projection = ClearMLRuntimeProjection(
        state="active",
        execution_ids=[record.task_id],
        terminal_candidates=[],
    )
    client = ObservationClient(records=[record], projections={record.task_id: projection})

    assert ClearMLObservationService(client=client).observe(stage_key=stage_key) is not None
    source = (
        Path(__file__).parents[1] / "src" / "backend" / "_clearml_observation.py"
    ).read_text(encoding="utf-8")
    assert "task_name" not in source


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_terminal_training_candidate_reconstruction(status: str) -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    payload = _candidate(stage_key, record.task_id, status)
    projection = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=[record.task_id],
        terminal_candidates=[payload],
    )
    client = ObservationClient(records=[record], projections={record.task_id: projection})
    service = ClearMLObservationService(client=client)

    observed = service.observe(stage_key=stage_key)
    collected = service.collect(stage_key=stage_key)

    assert observed == collected
    assert collected is not None
    assert collected["state"] == "terminal"
    assert collected["status"] == status
    assert collected["attempts"][0]["execution_id"] == record.task_id
    if status == "completed":
        assert collected["diagnostic"] is None
        assert collected["result"] == _completed_result("training")
    else:
        assert collected["diagnostic"] == _diagnostic(status)
        assert collected["result"] is None


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_terminal_evaluation_candidate_reconstruction(status: str) -> None:
    stage_input = _evaluation_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    payload = _candidate(stage_key, record.task_id, status)
    projection = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=[record.task_id],
        terminal_candidates=[payload],
    )
    client = ObservationClient(records=[record], projections={record.task_id: projection})
    collected = ClearMLObservationService(client=client).collect(stage_key=stage_key)

    assert collected is not None
    assert collected["status"] == status
    assert collected["attempts"][0]["execution_id"] == record.task_id
    if status == "completed":
        assert collected["diagnostic"] is None
        assert collected["result"] == _completed_result("evaluation")
    else:
        assert collected["diagnostic"] == _diagnostic(status)
        assert collected["result"] is None


def test_failed_retry_completed_preserves_ordered_attempt_history() -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    retry_id = "clearml-retry-2"
    projection = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=[record.task_id, retry_id],
        terminal_candidates=[
            _candidate(stage_key, record.task_id, "failed", index=0),
            _candidate(stage_key, retry_id, "completed", index=1),
        ],
    )
    client = ObservationClient(records=[record], projections={record.task_id: projection})

    collected = ClearMLObservationService(client=client).collect(stage_key=stage_key)

    assert collected is not None
    assert collected["status"] == "completed"
    assert [attempt["execution_id"] for attempt in collected["attempts"]] == [
        record.task_id,
        retry_id,
    ]
    assert [attempt["status"] for attempt in collected["attempts"]] == [
        "failed",
        "completed",
    ]


def test_invalid_attempt_order_is_rejected() -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    retry_id = "clearml-retry-2"
    projection = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=[record.task_id, retry_id],
        terminal_candidates=[
            _candidate(stage_key, retry_id, "failed", index=0),
            _candidate(stage_key, record.task_id, "completed", index=1),
        ],
    )
    client = ObservationClient(records=[record], projections={record.task_id: projection})

    with pytest.raises(ClearMLObservationError, match="malformed"):
        ClearMLObservationService(client=client).collect(stage_key=stage_key)


def test_malformed_diagnostic_is_rejected() -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    payload = _candidate(stage_key, record.task_id, "failed")
    payload["diagnostic"] = {"code": "BAD-CODE", "message": "broken"}
    cast(dict[str, object], payload["attempts"][0])["diagnostic"] = payload["diagnostic"]
    projection = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=[record.task_id],
        terminal_candidates=[payload],
    )
    client = ObservationClient(records=[record], projections={record.task_id: projection})

    with pytest.raises(ClearMLObservationError, match="malformed"):
        ClearMLObservationService(client=client).collect(stage_key=stage_key)


def test_malformed_training_candidate_payload_is_rejected() -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    payload = _candidate(stage_key, record.task_id, "completed")
    cast(dict[str, object], payload["result"])["weights"] = {
        "uri": "s3://bucket/weights.pt",
        "bytes": 123,
        "sha256": "NOT-A-SHA",
        "format": "pytorch-state-dict/v1",
    }
    projection = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=[record.task_id],
        terminal_candidates=[payload],
    )
    client = ObservationClient(records=[record], projections={record.task_id: projection})

    with pytest.raises(ClearMLObservationError, match="malformed"):
        ClearMLObservationService(client=client).collect(stage_key=stage_key)


def test_malformed_evaluation_candidate_payload_is_rejected() -> None:
    stage_input = _evaluation_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    payload = _candidate(stage_key, record.task_id, "completed")
    cast(dict[str, object], cast(dict[str, object], payload["result"])["metrics"])[
        "f1"
    ] = float("nan")
    projection = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=[record.task_id],
        terminal_candidates=[payload],
    )
    client = ObservationClient(records=[record], projections={record.task_id: projection})

    with pytest.raises(ClearMLObservationError, match="malformed"):
        ClearMLObservationService(client=client).collect(stage_key=stage_key)


def test_terminal_status_without_harness_payload_is_rejected() -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    projection = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=[record.task_id],
        terminal_candidates=[],
    )
    client = ObservationClient(records=[record], projections={record.task_id: projection})

    with pytest.raises(ClearMLObservationError, match="malformed"):
        ClearMLObservationService(client=client).collect(stage_key=stage_key)


def test_terminal_candidate_status_payload_contradiction_is_rejected() -> None:
    stage_input = _training_stage_input()
    record = _admitted_record(stage_input)
    stage_key = _stage_key_from_input(stage_input)
    payload = _candidate(stage_key, record.task_id, "failed")
    payload["result"] = _completed_result("training")
    projection = ClearMLRuntimeProjection(
        state="terminal",
        execution_ids=[record.task_id],
        terminal_candidates=[payload],
    )
    client = ObservationClient(records=[record], projections={record.task_id: projection})

    with pytest.raises(ClearMLObservationError, match="malformed"):
        ClearMLObservationService(client=client).collect(stage_key=stage_key)


def test_observation_module_is_sdk_optional_and_read_only() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "backend" / "_clearml_observation.py"
    ).read_text(encoding="utf-8")
    assert "import clearml" not in source.lower()
    assert "from clearml" not in source.lower()
    assert "mldb_v2.skeleton" not in source
    assert "result_acceptance" not in source
    assert "canonical_writes" not in source
    assert "study_result_repository" not in source
    assert "def admit(" not in source
    assert "def cancel" not in source
    assert "write_" not in source
