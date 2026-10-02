from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import mldb_v2.tests.test_evaluation_result_acceptance as eval_fx
from mldb_v2.src.api._errors import _ApplicationBoundaryError
from mldb_v2.src.api.application import Application
from mldb_v2.src.backend._registry import BackendRegistry
from mldb_v2.src.common.ids import EntityKind, EvaluationResultId, StudyResultId
from mldb_v2.src.repository.canonical_writes import CanonicalRepositoryWriter
from mldb_v2.src.verification.result_acceptance import AcceptedResultRecordValidator, ResultAcceptor


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _attempt(execution_id: str, status: str, index: int) -> dict[str, object]:
    diagnostic = None if status == "completed" else {
        "code": "backend_failed",
        "message": "backend terminal",
    }
    return {
        "backend": "fake",
        "execution_id": execution_id,
        "status": status,
        "started_at": f"2026-09-13T00:0{index}:00Z",
        "ended_at": f"2026-09-13T00:0{index}:01Z",
        "diagnostic": diagnostic,
    }


def _install_failed_evaluation(tmp_path: Path):
    fx = eval_fx._fixture(tmp_path)
    root = fx["root"]
    plan = fx["plan"]
    study_result = copy.deepcopy(fx["study_result"])
    failed_candidate = copy.deepcopy(fx["candidate"])
    failed_candidate["attempts"] = [_attempt("eval-attempt-1", "failed", 1)]
    failed_candidate["status"] = "failed"
    failed_candidate["diagnostic"] = {
        "code": "backend_failed",
        "message": "backend terminal",
    }
    failed_candidate["result"] = None
    failed = ResultAcceptor(
        mldb_data_root=root,
        object_bytes=fx["object_bytes"],
    ).accept_evaluation(
        request={
            "study_result": study_result,
            "plan": plan,
            "stage_input": fx["stage_input"],
            "candidate": failed_candidate,
        }
    )["evaluation_result"]

    _write(
        root / "demo" / "study_plans" / f"{plan['id'].split('/', 1)[1]}.yaml",
        plan,
    )
    _write(
        root / "demo" / "evaluation_results" / f"{failed['id'].split('/', 1)[1]}.yaml",
        failed,
    )
    study_result["status"] = "completed_with_failures"
    slot = study_result["trials"][0]["evaluations"][0]
    slot.update({"disposition": "failed", "result": failed["id"], "reason": None})
    _write(
        root / "demo" / "study_results" / f"{study_result['id'].split('/', 1)[1]}.yaml",
        study_result,
    )
    return fx, study_result, failed


def _completed_replacement(fx, failed):
    replacement = copy.deepcopy(failed)
    replacement["attempts"].append(_attempt("eval-attempt-2", "completed", 2))
    replacement["status"] = "completed"
    replacement["diagnostic"] = None
    replacement["result"] = copy.deepcopy(fx["candidate"]["result"])
    return replacement


def _failed_replacement(failed):
    replacement = copy.deepcopy(failed)
    replacement["attempts"].append(_attempt("eval-attempt-2", "failed", 2))
    replacement["status"] = "failed"
    replacement["diagnostic"] = {
        "code": "backend_failed",
        "message": "backend terminal",
    }
    replacement["result"] = None
    return replacement


def _writer(tmp_path: Path) -> CanonicalRepositoryWriter:
    return CanonicalRepositoryWriter(
        tmp_path,
        record_validator=AcceptedResultRecordValidator(
            mldb_data_root=tmp_path / "mldb_data"
        ),
    )


def test_failed_evaluation_retry_can_append_failed_attempt(tmp_path: Path) -> None:
    _fx, _parent, failed = _install_failed_evaluation(tmp_path)
    stored = _writer(tmp_path).replace_failed_evaluation_result(
        entity_id=EvaluationResultId(failed["id"]),
        replacement=_failed_replacement(failed),
    )
    assert stored["status"] == "failed"
    assert [item["execution_id"] for item in stored["attempts"]] == [
        "eval-attempt-1",
        "eval-attempt-2",
    ]


def test_failed_evaluation_retry_can_append_completed_attempt(tmp_path: Path) -> None:
    fx, _parent, failed = _install_failed_evaluation(tmp_path)
    stored = _writer(tmp_path).replace_failed_evaluation_result(
        entity_id=EvaluationResultId(failed["id"]),
        replacement=_completed_replacement(fx, failed),
    )
    assert stored["status"] == "completed"
    assert stored["id"] == failed["id"]


@pytest.mark.parametrize("mutation", ["edit", "remove", "reorder"])
def test_retry_rejects_non_prefix_attempt_history(tmp_path: Path, mutation: str) -> None:
    fx, _parent, failed = _install_failed_evaluation(tmp_path)
    replacement = _completed_replacement(fx, failed)
    if mutation == "edit":
        replacement["attempts"][0]["execution_id"] = "edited"
    elif mutation == "remove":
        replacement["attempts"] = replacement["attempts"][1:]
    else:
        replacement["attempts"] = list(reversed(replacement["attempts"]))
    with pytest.raises(ValueError):
        _writer(tmp_path).replace_failed_evaluation_result(
            entity_id=EvaluationResultId(failed["id"]),
            replacement=replacement,
        )


def test_retry_rejects_lineage_mutation_and_completed_result_retry(tmp_path: Path) -> None:
    fx, _parent, failed = _install_failed_evaluation(tmp_path)
    replacement = _completed_replacement(fx, failed)
    broken = copy.deepcopy(replacement)
    broken["source_commit"] = "9" * 40
    with pytest.raises(ValueError):
        _writer(tmp_path).replace_failed_evaluation_result(
            entity_id=EvaluationResultId(failed["id"]),
            replacement=broken,
        )

    writer = _writer(tmp_path)
    writer.replace_failed_evaluation_result(
        entity_id=EvaluationResultId(failed["id"]),
        replacement=replacement,
    )
    third = copy.deepcopy(replacement)
    third["attempts"].append(_attempt("eval-attempt-3", "failed", 3))
    third["status"] = "failed"
    third["diagnostic"] = {"code": "backend_failed", "message": "backend terminal"}
    third["result"] = None
    with pytest.raises(ValueError, match="only failed EvaluationResult"):
        writer.replace_failed_evaluation_result(
            entity_id=EvaluationResultId(failed["id"]),
            replacement=third,
        )


def test_parent_retry_closure_failed_stays_failed_and_recovery_completes(tmp_path: Path) -> None:
    _fx, parent, failed = _install_failed_evaluation(tmp_path)
    writer = _writer(tmp_path)

    unchanged = writer.replace_study_result_after_evaluation_retry(
        entity_id=StudyResultId(parent["id"]),
        trial="trial-0001",
        coordinate="eval-0001",
        replacement=copy.deepcopy(parent),
    )
    assert unchanged["status"] == "completed_with_failures"

    replacement = copy.deepcopy(parent)
    replacement["trials"][0]["evaluations"][0]["disposition"] = "completed"
    replacement["status"] = "completed"
    stored = writer.replace_study_result_after_evaluation_retry(
        entity_id=StudyResultId(parent["id"]),
        trial="trial-0001",
        coordinate="eval-0001",
        replacement=replacement,
    )
    assert stored["status"] == "completed"
    assert stored["trials"][0]["evaluations"][0]["result"] == failed["id"]


class RetryBackend:
    def __init__(self, candidate: dict[str, object]) -> None:
        self.candidate = candidate
        self.retry_calls: list[dict[str, object]] = []
        self.admit_calls: list[object] = []

    def admit(self, *, stage_input):
        self.admit_calls.append(copy.deepcopy(stage_input))
        raise AssertionError("ordinary admit must not run during retry-stage")

    def retry_stage(self, *, stage_input, prior_execution_ids):
        self.retry_calls.append(
            {
                "stage_input": copy.deepcopy(stage_input),
                "prior_execution_ids": list(prior_execution_ids),
            }
        )
        return copy.deepcopy(self.candidate)

    def ensure_study_execution(self, *, plan, study_result):
        raise AssertionError(
            "retry-stage must recover the existing stage/controller lineage, "
            "not generic Study execution ensure"
        )

    def observe(self, *, stage_key):
        return copy.deepcopy(self.candidate)

    def collect(self, *, stage_key):
        return copy.deepcopy(self.candidate)

    def cancel_study(self, *, study_result):
        raise AssertionError("retry-stage must not cancel")


def _application(tmp_path: Path, fx, backend: RetryBackend) -> Application:
    registry = BackendRegistry()
    registry.register("fake", lambda _config: backend)
    return Application(
        repository_root=tmp_path,
        backend_registry=registry,
        object_bytes=fx["object_bytes"],
        runtime_registry_version_resolver=lambda: 1,
    )


def _aggregate_candidate(fx, status: str) -> dict[str, object]:
    candidate = copy.deepcopy(fx["candidate"])
    candidate["attempts"] = [
        _attempt("eval-attempt-1", "failed", 1),
        _attempt("eval-attempt-2", status, 2),
    ]
    candidate["status"] = status
    if status == "completed":
        candidate["diagnostic"] = None
    else:
        candidate["diagnostic"] = {"code": "backend_failed", "message": "backend terminal"}
        candidate["result"] = None
    return candidate


def test_application_retry_stage_updates_same_run_without_ordinary_admission(tmp_path: Path) -> None:
    fx, parent, failed = _install_failed_evaluation(tmp_path)
    backend = RetryBackend(_aggregate_candidate(fx, "completed"))
    app = _application(tmp_path, fx, backend)

    updated = app.retry_stage(
        study_result=parent["id"],
        trial="trial-0001",
        coordinate="eval-0001",
    )

    assert updated["id"] == parent["id"]
    assert updated["execution_key"] == parent["execution_key"]
    assert updated["status"] == "completed"
    assert backend.admit_calls == []
    assert len(backend.retry_calls) == 1
    assert backend.retry_calls[0]["prior_execution_ids"] == ["eval-attempt-1"]
    stored = app.get_entity(
        kind=EntityKind.EVALUATION_RESULT,
        entity_id=failed["id"],
    )
    assert stored["id"] == failed["id"]
    assert [item["execution_id"] for item in stored["attempts"]] == [
        "eval-attempt-1", "eval-attempt-2"
    ]


def test_application_failed_retry_grows_history_and_parent_remains_failed(tmp_path: Path) -> None:
    fx, parent, failed = _install_failed_evaluation(tmp_path)
    backend = RetryBackend(_aggregate_candidate(fx, "failed"))
    app = _application(tmp_path, fx, backend)

    updated = app.retry_stage(
        study_result=parent["id"],
        trial="trial-0001",
        coordinate="eval-0001",
    )
    assert updated["status"] == "completed_with_failures"
    stored = app.get_entity(kind=EntityKind.EVALUATION_RESULT, entity_id=failed["id"])
    assert stored["status"] == "failed"
    assert len(stored["attempts"]) == 2


def test_application_retry_completed_or_nonexistent_target_is_bounded(tmp_path: Path) -> None:
    fx, parent, _failed = _install_failed_evaluation(tmp_path)
    backend = RetryBackend(_aggregate_candidate(fx, "completed"))
    app = _application(tmp_path, fx, backend)
    app.retry_stage(
        study_result=parent["id"],
        trial="trial-0001",
        coordinate="eval-0001",
    )
    with pytest.raises(_ApplicationBoundaryError) as completed:
        app.retry_stage(
            study_result=parent["id"],
            trial="trial-0001",
            coordinate="eval-0001",
        )
    assert completed.value.error["code"] == "lifecycle_conflict"

    other_root = tmp_path / "missing"
    fx2, parent2, _ = _install_failed_evaluation(other_root)
    app2 = _application(other_root, fx2, RetryBackend(_aggregate_candidate(fx2, "failed")))
    with pytest.raises(_ApplicationBoundaryError) as missing:
        app2.retry_stage(
            study_result=parent2["id"],
            trial="trial-9999",
            coordinate="eval-0001",
        )
    assert missing.value.error["code"] == "not_found"
