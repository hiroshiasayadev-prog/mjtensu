from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from mldb_v2.src.backend._clearml_admission import (
    ClearMLAdmissionError,
    ClearMLAdmissionService,
    ClearMLCreateRequest,
    ClearMLTaskRecord,
    _ownership_key,
    _restore_stage_input,
)
from mldb_v2.src.backend.stage_input import EvaluationStageInput, TrainingStageInput
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


def _training_stage_input(
    *, study_result: str = "demo/run-abcd", study: str = "demo/study-a"
) -> TrainingStageInput:
    return {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": StudyResultId(study_result),
        "plan": StudyPlanId("demo/study-a-plan-0123456789abcdef"),
        "plan_sha256": "1" * 64,
        "trial": TrialId("trial-0001"),
        "kind": "training",
        "coordinate": None,
        "source_commit": "2" * 40,
        "pins": [cast(object, _pin("study", study))],
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


def _evaluation_stage_input(
    *,
    coordinate: str = "eval-0001",
    study_result: str = "demo/run-abcd",
) -> EvaluationStageInput:
    return {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": StudyResultId(study_result),
        "plan": StudyPlanId("demo/study-a-plan-0123456789abcdef"),
        "plan_sha256": "1" * 64,
        "trial": TrialId("trial-0001"),
        "kind": "evaluation",
        "coordinate": EvaluationCoordinateId(coordinate),
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
                "uri": "s3://bucket/weights.pt",
                "bytes": 123,
                "sha256": "3" * 64,
                "format": "pytorch-state-dict/v1",
            },
        },
    }


class FakeClient:
    def __init__(self, *, create_mode: str = "normal") -> None:
        self.records: list[ClearMLTaskRecord] = []
        self.searches: list[tuple[str, str]] = []
        self.requests: list[ClearMLCreateRequest] = []
        self.create_mode = create_mode

    def search_tasks(self, *, project: str, ownership_key: str):
        self.searches.append((project, ownership_key))
        return [
            record
            for record in self.records
            if record.metadata.get("mldb.ownership_key") == ownership_key
        ]

    def create_task(self, request: ClearMLCreateRequest) -> str | None:
        self.requests.append(request)
        if self.create_mode == "unresolved-error":
            raise TimeoutError("request outcome unknown")
        task_id = f"clearml-{len(self.records) + 1}"
        self.records.append(
            ClearMLTaskRecord(
                task_id=task_id,
                metadata=dict(request.metadata),
                configuration=deepcopy(dict(request.configuration)),
            )
        )
        if self.create_mode == "ambiguous-error":
            raise TimeoutError("server committed but response was lost")
        if self.create_mode == "ambiguous-none":
            return None
        return task_id


def test_training_admission_maps_namespace_metadata_and_public_parameters() -> None:
    client = FakeClient()
    stage_input = _training_stage_input()
    result = ClearMLAdmissionService(client=client, queue="gpu-a").admit(
        stage_input=stage_input
    )

    assert result.project == "mldb/demo"
    assert result.task_id == "clearml-1"
    assert result.recovered is False
    request = client.requests[0]
    assert request.project == "mldb/demo"
    assert request.metadata == {
        "mldb.namespace": "demo",
        "mldb.study": "demo/study-a",
        "mldb.plan": "demo/study-a-plan-0123456789abcdef",
        "mldb.study_result": "demo/run-abcd",
        "mldb.trial": "trial-0001",
        "mldb.stage_kind": "training",
        "mldb.task": "demo/task",
        "mldb.corpus": "demo/corpus",
        "mldb.source_commit": "2" * 40,
        "mldb.ownership_key": result.ownership_key,
        "mldb.architecture": "demo/architecture",
        "mldb.protocol": "demo/train-protocol",
    }
    assert request.configuration["mldb.public_parameters"] == {
        "batch_size": 8,
        "learning_rate": 0.001,
    }
    assert "mldb.evaluation_stage" not in request.metadata
    assert "mldb.evaluation_coordinate" not in request.metadata


def test_evaluation_admission_maps_coordinate_model_and_protocol_metadata() -> None:
    client = FakeClient()
    stage_input = _evaluation_stage_input()
    result = ClearMLAdmissionService(client=client).admit(stage_input=stage_input)
    metadata = client.requests[0].metadata

    assert result.project == "mldb/demo"
    assert metadata["mldb.evaluation_stage"] == "final-holdout"
    assert metadata["mldb.evaluation_coordinate"] == "eval-0001"
    assert metadata["mldb.model"] == "demo/model-a"
    assert metadata["mldb.architecture"] == "demo/architecture"
    assert metadata["mldb.protocol"] == "demo/eval-protocol"


def test_namespace_project_is_shared_across_studies() -> None:
    first = FakeClient()
    second = FakeClient()
    a = ClearMLAdmissionService(client=first).admit(
        stage_input=_training_stage_input(study_result="demo/run-a", study="demo/study-a")
    )
    stage_b = _training_stage_input(study_result="demo/run-b", study="demo/study-b")
    stage_b["plan"] = StudyPlanId("demo/study-b-plan-0123456789abcdef")
    b = ClearMLAdmissionService(client=second).admit(stage_input=stage_b)
    assert a.project == b.project == "mldb/demo"


def test_ownership_is_deterministic_and_distinguishes_exact_logical_coordinate() -> None:
    training = _training_stage_input()
    same_training = deepcopy(training)
    evaluation_a = _evaluation_stage_input(coordinate="eval-0001")
    evaluation_b = _evaluation_stage_input(coordinate="eval-0002")
    other_result = _training_stage_input(study_result="demo/run-efgh")

    assert _ownership_key(training) == _ownership_key(same_training)
    assert _ownership_key(training) != _ownership_key(evaluation_a)
    assert _ownership_key(evaluation_a) != _ownership_key(evaluation_b)
    assert _ownership_key(training) != _ownership_key(other_result)


def test_existing_exact_task_is_recovered_without_duplicate_create() -> None:
    client = FakeClient()
    stage_input = _training_stage_input()
    first = ClearMLAdmissionService(client=client).admit(stage_input=stage_input)
    client.requests.clear()

    replay = ClearMLAdmissionService(client=client).admit(stage_input=stage_input)

    assert replay.task_id == first.task_id
    assert replay.ownership_key == first.ownership_key
    assert replay.recovered is True
    assert client.requests == []
    assert len(client.records) == 1


def test_duplicate_exact_ownership_is_bounded_failure() -> None:
    client = FakeClient()
    stage_input = _training_stage_input()
    ClearMLAdmissionService(client=client).admit(stage_input=stage_input)
    client.records.append(deepcopy(client.records[0]))

    with pytest.raises(ClearMLAdmissionError, match="multiple ClearML Tasks"):
        ClearMLAdmissionService(client=client).admit(stage_input=stage_input)
    assert len(client.records) == 2


def test_same_ownership_with_mismatched_stage_input_is_rejected_before_create() -> None:
    client = FakeClient()
    stage_input = _training_stage_input()
    first = ClearMLAdmissionService(client=client).admit(stage_input=stage_input)
    client.requests.clear()
    broken = client.records[0]
    config = dict(broken.configuration)
    config["mldb.stage_input"] = "{}"
    client.records[:] = [
        ClearMLTaskRecord(
            task_id=broken.task_id,
            metadata=broken.metadata,
            configuration=config,
        )
    ]

    with pytest.raises(ClearMLAdmissionError, match="identity mismatch"):
        ClearMLAdmissionService(client=client).admit(stage_input=stage_input)
    assert first.task_id == "clearml-1"
    assert client.requests == []


@pytest.mark.parametrize("mode", ["ambiguous-error", "ambiguous-none"])
def test_ambiguous_create_response_researches_and_recovers(mode: str) -> None:
    client = FakeClient(create_mode=mode)
    result = ClearMLAdmissionService(client=client).admit(
        stage_input=_training_stage_input()
    )

    assert result.task_id == "clearml-1"
    assert result.recovered is True
    assert len(client.records) == 1
    assert len(client.searches) >= 2


def test_ambiguous_create_without_recoverable_ownership_fails_bounded() -> None:
    client = FakeClient(create_mode="unresolved-error")
    with pytest.raises(ClearMLAdmissionError, match="could not be recovered"):
        ClearMLAdmissionService(
            client=client, recovery_search_attempts=2
        ).admit(stage_input=_training_stage_input())
    assert len(client.records) == 0
    assert len(client.searches) == 3


def test_stage_input_transport_is_exact_and_operational_fields_do_not_leak() -> None:
    client = FakeClient()
    stage_input = _evaluation_stage_input()
    original = deepcopy(stage_input)
    sdk_object = object()
    service = ClearMLAdmissionService(client=client, queue="gpu-a")
    service.admit(stage_input=stage_input)
    request = client.requests[0]

    assert stage_input == original
    assert _restore_stage_input(request.launch.stage_input_json) == stage_input
    assert request.launch.stage_input_json == request.configuration["mldb.stage_input"]
    assert request.launch.queue == "gpu-a"
    restored = _restore_stage_input(request.launch.stage_input_json)
    for forbidden in ("endpoint", "credentials", "queue", "worker", "clearml_task_id"):
        assert forbidden not in restored
    assert sdk_object not in restored.values()
    assert "queue" not in request.configuration


def test_remote_launch_uses_exact_common_execution_harness_and_pinned_commit() -> None:
    client = FakeClient()
    stage_input = _training_stage_input()
    ClearMLAdmissionService(client=client).admit(stage_input=stage_input)
    request = client.requests[0]

    assert request.launch.harness_symbol == (
        "mldb_v2.src.backend.execution_harness.CommonExecutionHarness"
    )
    assert request.configuration["mldb.harness"] == request.launch.harness_symbol
    assert request.launch.source_commit == stage_input["source_commit"]
    assert request.configuration["mldb.source_commit"] == stage_input["source_commit"]
    assert "classifier" not in request.launch.harness_symbol
    assert "detector" not in request.launch.harness_symbol
    assert "rotated" not in request.launch.harness_symbol


def test_task_name_is_human_readable_but_not_recovery_identity() -> None:
    client = FakeClient()
    stage_input = _training_stage_input()
    ClearMLAdmissionService(client=client).admit(stage_input=stage_input)
    request = client.requests[0]
    assert request.task_name == "architecture | train | study-a | trial-0001"

    client.requests.clear()
    replay = ClearMLAdmissionService(client=client).admit(stage_input=stage_input)
    assert replay.recovered is True
    assert client.requests == []


def test_evaluation_task_name_shows_architecture_and_stage() -> None:
    client = FakeClient()
    stage_input = _evaluation_stage_input()
    ClearMLAdmissionService(client=client).admit(stage_input=stage_input)

    assert client.requests[0].task_name == (
        "architecture | final-holdout | study-a | trial-0001"
    )


def test_admission_module_remains_clearml_sdk_optional_and_admission_only() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "backend" / "_clearml_admission.py"
    ).read_text(encoding="utf-8")
    assert "import clearml" not in source.lower()
    assert "from clearml" not in source.lower()
    assert "mldb_v2.skeleton" not in source
    assert "result_acceptance" not in source
    assert "canonical_writes" not in source
    assert "execution_readiness" not in source
    assert "def observe(" not in source
    assert "def collect(" not in source
    assert "def cancel" not in source
    assert "training_runtime" not in source
    assert "evaluation_runtime" not in source
    assert "classifier" not in source.lower()
    assert "detector" not in source.lower()
    assert "rotated-fcos" not in source.lower()