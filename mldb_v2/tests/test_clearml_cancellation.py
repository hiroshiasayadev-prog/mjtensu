from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import cast

import pytest

from mldb_v2.src.backend._clearml_admission import _ownership_key
from mldb_v2.src.backend._clearml_cancellation import (
    ClearMLCancellationService,
    ClearMLCancellationState,
    ClearMLLogData,
    ClearMLLogRecord,
    ClearMLLogService,
    ClearMLOwnedTask,
    _expected_ownership_key,
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
    *, study_result: str = "demo/run-abcd"
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
        "pins": [cast(object, _pin("study", "demo/study-a"))],
        "stage": {
            "task": TaskId("demo/task"),
            "corpus": CorpusId("demo/corpus"),
            "architecture": ArchitectureId("demo/architecture"),
            "train_protocol": TrainProtocolId("demo/train-protocol"),
            "parameters": {},
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
            "parameters": {},
        },
        "runtime_model": {
            "model": ModelId("demo/model-a"),
            "training_result": TrainingResultId("demo/training-a"),
            "task": TaskId("demo/task"),
            "architecture": ArchitectureId("demo/architecture"),
            "weights": {
                "uri": "s3://bucket/weights.pt",
                "bytes": 1,
                "sha256": "3" * 64,
                "format": "pytorch-state-dict/v1",
            },
        },
    }


def _metadata(
    *,
    study_result: str = "demo/run-abcd",
    trial: str = "trial-0001",
    kind: str = "training",
    coordinate: str | None = None,
) -> dict[str, str]:
    metadata = {
        "mldb.namespace": study_result.split("/", 1)[0],
        "mldb.study_result": study_result,
        "mldb.trial": trial,
        "mldb.stage_kind": kind,
        "mldb.ownership_key": _expected_ownership_key(
            study_result=study_result,
            trial=trial,
            kind=kind,
            coordinate=coordinate,
        ),
    }
    if kind == "evaluation":
        assert coordinate is not None
        metadata["mldb.evaluation_stage"] = "holdout"
        metadata["mldb.evaluation_coordinate"] = coordinate
    return metadata


def _task(
    task_id: str,
    *,
    study_result: str = "demo/run-abcd",
    state: ClearMLCancellationState = ClearMLCancellationState.ACTIVE,
    kind: str = "training",
    coordinate: str | None = None,
) -> ClearMLOwnedTask:
    return ClearMLOwnedTask(
        task_id=task_id,
        metadata=_metadata(
            study_result=study_result, kind=kind, coordinate=coordinate
        ),
        cancellation_state=state,
    )


class FakeClient:
    def __init__(
        self,
        records: list[ClearMLOwnedTask] | None = None,
        *,
        leaky_search: bool = False,
        logs: dict[str, ClearMLLogData | None] | None = None,
    ) -> None:
        self.records = list(records or [])
        self.leaky_search = leaky_search
        self.logs = dict(logs or {})
        self.searches: list[tuple[str, str, str]] = []
        self.cancelled: list[str] = []
        self.credentials = "secret-token"
        self.endpoint = "https://secret.example.invalid"

    def search_tasks_by_metadata(self, *, project: str, key: str, value: str):
        self.searches.append((project, key, value))
        if self.leaky_search:
            return list(self.records)
        return [record for record in self.records if record.metadata.get(key) == value]

    def request_cancellation(self, *, task_id: str) -> None:
        self.cancelled.append(task_id)
        self.records = [
            replace(record, cancellation_state=ClearMLCancellationState.CANCELLED)
            if record.task_id == task_id
            else record
            for record in self.records
        ]

    def read_task_logs(self, *, task_id: str) -> ClearMLLogData | None:
        return self.logs.get(task_id)


class NoLogClient:
    def __init__(self, records: list[ClearMLOwnedTask]) -> None:
        self.records = records

    def search_tasks_by_metadata(self, *, project: str, key: str, value: str):
        return [record for record in self.records if record.metadata.get(key) == value]


def test_ownership_key_matches_t005_02_for_training_and_evaluation() -> None:
    training = _training_stage_input()
    evaluation = _evaluation_stage_input()
    assert _expected_ownership_key(
        study_result=str(training["study_result"]),
        trial=str(training["trial"]),
        kind="training",
        coordinate=None,
    ) == _ownership_key(training)
    assert _expected_ownership_key(
        study_result=str(evaluation["study_result"]),
        trial=str(evaluation["trial"]),
        kind="evaluation",
        coordinate=str(evaluation["coordinate"]),
    ) == _ownership_key(evaluation)


def test_cancel_scopes_by_exact_study_result_and_namespace_project() -> None:
    client = FakeClient([_task("task-a")])
    ClearMLCancellationService(client=client).cancel_study(
        study_result=StudyResultId("demo/run-abcd")
    )
    assert client.searches == [
        ("mldb/demo", "mldb.study_result", "demo/run-abcd")
    ]
    assert client.cancelled == ["task-a"]


def test_zero_owned_work_is_successful_noop() -> None:
    client = FakeClient()
    ClearMLCancellationService(client=client).cancel_study(
        study_result=StudyResultId("demo/run-abcd")
    )
    assert client.cancelled == []


def test_multiple_exact_owned_active_tasks_are_cancelled() -> None:
    client = FakeClient(
        [
            _task("train"),
            _task("eval", kind="evaluation", coordinate="eval-0001"),
        ]
    )
    ClearMLCancellationService(client=client).cancel_study(
        study_result=StudyResultId("demo/run-abcd")
    )
    assert client.cancelled == ["train", "eval"]


@pytest.mark.parametrize(
    "state",
    [
        ClearMLCancellationState.TERMINAL,
        ClearMLCancellationState.CANCELLING,
        ClearMLCancellationState.CANCELLED,
    ],
)
def test_terminal_or_already_cancelling_work_is_successful_noop(
    state: ClearMLCancellationState,
) -> None:
    client = FakeClient([_task("task-a", state=state)])
    ClearMLCancellationService(client=client).cancel_study(
        study_result=StudyResultId("demo/run-abcd")
    )
    assert client.cancelled == []


def test_repeat_cancellation_does_not_issue_second_request() -> None:
    client = FakeClient([_task("task-a")])
    service = ClearMLCancellationService(client=client)
    service.cancel_study(study_result=StudyResultId("demo/run-abcd"))
    service.cancel_study(study_result=StudyResultId("demo/run-abcd"))
    assert client.cancelled == ["task-a"]


def test_wrong_study_record_from_leaky_search_is_not_cancelled() -> None:
    client = FakeClient(
        [_task("other", study_result="demo/run-other")], leaky_search=True
    )
    ClearMLCancellationService(client=client).cancel_study(
        study_result=StudyResultId("demo/run-abcd")
    )
    assert client.cancelled == []


def test_malformed_or_mismatched_ownership_is_skipped_safely() -> None:
    valid = _task("valid")
    wrong_owner = dict(valid.metadata)
    wrong_owner["mldb.ownership_key"] = "mldb-v2-stage:" + "0" * 64
    wrong_namespace = dict(valid.metadata)
    wrong_namespace["mldb.namespace"] = "other"
    bad_coordinate = _task("bad-coordinate", kind="evaluation", coordinate="eval-0001")
    bad_coordinate_metadata = dict(bad_coordinate.metadata)
    bad_coordinate_metadata["mldb.evaluation_coordinate"] = "not-a-coordinate"
    client = FakeClient(
        [
            replace(valid, task_id="wrong-owner", metadata=wrong_owner),
            replace(valid, task_id="wrong-namespace", metadata=wrong_namespace),
            replace(bad_coordinate, metadata=bad_coordinate_metadata),
        ],
        leaky_search=True,
    )
    ClearMLCancellationService(client=client).cancel_study(
        study_result=StudyResultId("demo/run-abcd")
    )
    assert client.cancelled == []


def test_supported_logs_return_immutable_observational_record() -> None:
    client = FakeClient(
        [_task("task-a")],
        logs={
            "task-a": ClearMLLogData(
                chunks=("line 1", "line 2"), locator="clearml://task/task-a/logs"
            )
        },
    )
    record = ClearMLLogService(client=client).read_task_logs(
        study_result=StudyResultId("demo/run-abcd"), task_id="task-a"
    )
    assert record == ClearMLLogRecord(
        task_id="task-a",
        chunks=("line 1", "line 2"),
        locator="clearml://task/task-a/logs",
    )
    assert record is not None
    with pytest.raises(FrozenInstanceError):
        record.task_id = "other"  # type: ignore[misc]


def test_log_absence_is_none() -> None:
    client = FakeClient([_task("task-a")], logs={"task-a": None})
    assert (
        ClearMLLogService(client=client).read_task_logs(
            study_result=StudyResultId("demo/run-abcd"), task_id="task-a"
        )
        is None
    )


def test_unsupported_log_capability_is_none() -> None:
    client = NoLogClient([_task("task-a")])
    assert (
        ClearMLLogService(client=client).read_task_logs(
            study_result=StudyResultId("demo/run-abcd"), task_id="task-a"
        )
        is None
    )


def test_logs_require_exact_task_ownership() -> None:
    client = FakeClient(
        [_task("other", study_result="demo/run-other")],
        leaky_search=True,
        logs={"other": ClearMLLogData(chunks=("wrong study",))},
    )
    assert (
        ClearMLLogService(client=client).read_task_logs(
            study_result=StudyResultId("demo/run-abcd"), task_id="other"
        )
        is None
    )


def test_logs_do_not_expose_client_credentials_endpoint_or_sdk_object() -> None:
    client = FakeClient(
        [_task("task-a")],
        logs={"task-a": ClearMLLogData(chunks=("safe log",))},
    )
    record = ClearMLLogService(client=client).read_task_logs(
        study_result=StudyResultId("demo/run-abcd"), task_id="task-a"
    )
    assert record is not None
    assert set(record.__dataclass_fields__) == {"task_id", "chunks", "locator"}
    assert client.credentials not in repr(record)
    assert client.endpoint not in repr(record)
    assert not hasattr(record, "client")
    assert not hasattr(record, "credentials")
    assert not hasattr(record, "endpoint")


def test_cancellation_module_is_sdk_optional_and_has_no_canonical_authority() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "backend" / "_clearml_cancellation.py"
    ).read_text(encoding="utf-8")
    lower = source.lower()
    assert "import clearml" not in lower
    assert "from clearml" not in lower
    assert "mldb_v2.skeleton" not in source
    assert "mldb_v2.src.results" not in source
    assert "mldb_v2.src.study" not in source
    assert "candidate_outcome" not in source
    assert "result_acceptance" not in source
    assert "canonical" not in lower.replace("_canonical_json_bytes", "")
    assert "task_name" not in source
    assert "def admit(" not in source
    assert "def observe(" not in source
    assert "def collect(" not in source
