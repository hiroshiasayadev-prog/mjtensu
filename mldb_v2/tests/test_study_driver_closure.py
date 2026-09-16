from __future__ import annotations

import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

import mldb_v2.src.study.study_driver as driver
import mldb_v2.tests.test_evaluation_result_acceptance as eval_fx
import mldb_v2.tests.test_execution_readiness as ready_fx
import mldb_v2.tests.test_study_driver as fx
import mldb_v2.tests.test_training_result_acceptance as train_fx
from mldb_v2.src.common.ids import _canonical_json_bytes
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study._plan_build import _plan_id, _validate_study_plan
from mldb_v2.src.training.canonical_weights import _serialize_canonical_state_dict


def _use_fake_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    fx.FakeAcceptor.force_training_failure = False
    fx.FakeAcceptor.force_evaluation_failure = False
    monkeypatch.setattr(driver, "ResultAcceptor", fx.FakeAcceptor)
    monkeypatch.setattr(driver, "AcceptedResultRecordValidator", fx.PermissiveValidator)


def _child_path(root: Path, kind: str, entity_id: str) -> Path:
    directory = {
        "training_result": "training_results",
        "evaluation_result": "evaluation_results",
    }[kind]
    return root / "demo" / directory / f"{entity_id.split('/', 1)[1]}.yaml"


def _write_failed_training_child(root: Path, plan: dict[str, object], status: str) -> str:
    result = fx.FakeAcceptor().accept_training(
        request={
            "candidate": {"status": status},
            "stage_input": {"trial": "trial-0001"},
            "study_result": fx._study_result(plan),
            "plan": plan,
        }
    )["training_result"]
    child_id = str(result["id"])
    fx._write_json(_child_path(root, "training_result", child_id), result)
    return child_id


def _write_evaluation_child(root: Path, *, status: str) -> str:
    child_id = f"{fx.STUDY_RESULT_ID}-trial-0001-eval-0001"
    fx._write_json(
        _child_path(root, "evaluation_result", child_id),
        {"schema": "mjtensu.mldb-v2/evaluation-result/v1", "id": child_id, "status": status},
    )
    return child_id


@pytest.mark.parametrize(
    ("status", "skip_reason"),
    [("failed", "upstream_failed"), ("cancelled", "upstream_cancelled")],
)
def test_interrupted_terminal_training_child_resumes_parent_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, skip_reason: str
) -> None:
    _use_fake_composition(monkeypatch)
    root, plan, _result = fx._install_repo(tmp_path)
    child_id = _write_failed_training_child(root, plan, status)
    before = _child_path(root, "training_result", child_id).read_bytes()

    response = fx._advance(tmp_path, fx.FakeBackend())
    current = fx._read_result(root)

    assert current["trials"][0]["training"] == {
        "disposition": status, "result": child_id, "reason": None
    }
    assert current["trials"][0]["evaluations"][0] == {
        "coordinate": "eval-0001", "stage": "holdout-a",
        "disposition": "skipped", "result": None, "reason": skip_reason,
    }
    assert _child_path(root, "training_result", child_id).read_bytes() == before
    assert response["status"] == "completed_with_failures"


def test_interrupted_evaluation_child_resumes_parent_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    root, plan, _result = fx._install_repo(tmp_path)
    fx._seed_training_children(root, plan)
    fx._mark_training_completed(root)
    child_id = _write_evaluation_child(root, status="failed")
    before = _child_path(root, "evaluation_result", child_id).read_bytes()

    response = fx._advance(tmp_path, fx.FakeBackend())
    current = fx._read_result(root)

    assert current["trials"][0]["evaluations"][0]["disposition"] == "failed"
    assert current["trials"][0]["evaluations"][0]["result"] == child_id
    assert _child_path(root, "evaluation_result", child_id).read_bytes() == before
    assert response["status"] == "completed_with_failures"


def test_conflicting_deterministic_child_is_bounded_global_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    root, plan, _result = fx._install_repo(tmp_path)
    child_id = _write_failed_training_child(root, plan, "failed")
    before = _child_path(root, "training_result", child_id).read_bytes()
    backend = fx.FakeBackend()
    key = fx._stage_key(plan)
    candidate = fx._terminal_observation(key, status="completed")
    backend.observations[backend._token(key)] = copy.deepcopy(candidate)
    backend.candidates[backend._token(key)] = copy.deepcopy(candidate)

    response = fx._advance(tmp_path, backend)
    current = fx._read_result(root)

    assert response["status"] == "failed"
    assert current["diagnostic"]["code"] == "study_progression_failed"
    assert current["trials"][0]["training"]["reason"] == "global_failure"
    assert _child_path(root, "training_result", child_id).read_bytes() == before


def test_terminal_study_result_is_immutable_and_backend_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    root, plan, _result = fx._install_repo(tmp_path)
    fx._seed_training_children(root, plan)
    fx._mark_training_completed(root)
    child_id = _write_evaluation_child(root, status="completed")
    first = fx._advance(tmp_path, fx.FakeBackend())
    assert first["status"] == "completed"
    before = (root / "demo" / "study_results" / f"{fx.STUDY_RESULT_ID.split('/', 1)[1]}.yaml").read_bytes()

    class ExplodingBackend(fx.FakeBackend):
        def observe(self, *, stage_key):
            raise AssertionError("terminal Study must not observe backend")

        def collect(self, *, stage_key):
            raise AssertionError("terminal Study must not collect backend")

        def admit(self, *, stage_input):
            raise AssertionError("terminal Study must not admit backend")

        def cancel_study(self, *, study_result):
            raise AssertionError("terminal Study must not cancel backend")

    second = fx._advance(tmp_path, ExplodingBackend())
    after = (root / "demo" / "study_results" / f"{fx.STUDY_RESULT_ID.split('/', 1)[1]}.yaml").read_bytes()
    assert second["changed"] is False
    assert second["terminal"] is True
    assert after == before


def test_all_backend_methods_run_outside_mutation_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    _root, plan, _result = fx._install_repo(tmp_path, status="cancelling")
    held = {"value": False}
    real_coordinator = driver.StudyResultMutationCoordinator

    class TrackingCoordinator(real_coordinator):
        def acquire(self, *, study_result_id):
            inner = super().acquire(study_result_id=study_result_id)

            class Guard:
                def __enter__(self_nonlocal):
                    inner.__enter__()
                    held["value"] = True

                def __exit__(self_nonlocal, exc_type, exc, tb):
                    held["value"] = False
                    return inner.__exit__(exc_type, exc, tb)

            return Guard()

    class AssertUnlockedBackend(fx.FakeBackend):
        def observe(self, *, stage_key):
            assert held["value"] is False
            return super().observe(stage_key=stage_key)

        def collect(self, *, stage_key):
            assert held["value"] is False
            return super().collect(stage_key=stage_key)

        def admit(self, *, stage_input):
            assert held["value"] is False
            return super().admit(stage_input=stage_input)

        def cancel_study(self, *, study_result):
            assert held["value"] is False
            return super().cancel_study(study_result=study_result)

    monkeypatch.setattr(driver, "StudyResultMutationCoordinator", TrackingCoordinator)

    active_backend = AssertUnlockedBackend()
    training_key = fx._stage_key(plan)
    active_backend.observations[active_backend._token(training_key)] = fx._active_observation(training_key)
    fx._advance(tmp_path, active_backend)
    assert active_backend.cancel_calls == [fx.STUDY_RESULT_ID]

    collect_base = tmp_path / "collect"
    _root2, plan2, _result2 = fx._install_repo(collect_base, status="cancelling")
    collect_backend = AssertUnlockedBackend()
    key2 = fx._stage_key(plan2)
    terminal = fx._terminal_observation(key2, status="cancelled")
    collect_backend.observations[collect_backend._token(key2)] = copy.deepcopy(terminal)
    collect_backend.candidates[collect_backend._token(key2)] = copy.deepcopy(terminal)
    fx._advance(collect_base, collect_backend)
    assert collect_backend.collect_calls == [key2]


def test_cancelling_mixed_active_and_never_admitted_drains_without_new_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    root, plan, _result = fx._install_repo(tmp_path, evaluations=2, status="cancelling")
    fx._seed_training_children(root, plan)
    fx._mark_training_completed(root)
    backend = fx.FakeBackend()
    active_key = fx._stage_key(plan, kind="evaluation", coordinate="eval-0001")
    backend.observations[backend._token(active_key)] = fx._active_observation(active_key)

    first = fx._advance(tmp_path, backend)
    current = fx._read_result(root)

    assert first["status"] == "cancelling"
    assert backend.cancel_calls == [fx.STUDY_RESULT_ID]
    assert backend.admit_calls == []
    assert current["trials"][0]["evaluations"][0]["disposition"] == "pending"
    assert current["trials"][0]["evaluations"][1]["reason"] == "study_cancelled"

    terminal = fx._terminal_observation(active_key, status="cancelled")
    backend.observations[backend._token(active_key)] = copy.deepcopy(terminal)
    backend.candidates[backend._token(active_key)] = copy.deepcopy(terminal)
    second = fx._advance(tmp_path, backend)
    current = fx._read_result(root)

    assert second["status"] == "cancelled"
    assert current["trials"][0]["evaluations"][0]["disposition"] == "cancelled"
    assert current["trials"][0]["evaluations"][1]["reason"] == "study_cancelled"
    assert backend.admit_calls == []


def test_transient_cancel_failure_preserves_state_and_later_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    root, plan, _result = fx._install_repo(tmp_path, status="cancelling")
    backend = fx.FakeBackend()
    key = fx._stage_key(plan)
    backend.observations[backend._token(key)] = fx._active_observation(key)
    backend.raise_on = "cancel"

    first = fx._advance(tmp_path, backend)
    after_first = fx._read_result(root)
    assert first["status"] == "cancelling"
    assert after_first["trials"][0]["training"]["disposition"] == "pending"
    assert after_first["trials"][0]["evaluations"][0]["reason"] == "study_cancelled"

    backend.raise_on = None
    second = fx._advance(tmp_path, backend)
    assert second["status"] == "cancelling"
    assert backend.cancel_calls == [fx.STUDY_RESULT_ID, fx.STUDY_RESULT_ID]
    assert fx._read_result(root)["status"] == "cancelling"


def test_two_callers_racing_new_readiness_share_one_logical_backend_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    _root, plan, _result = fx._install_repo(tmp_path)

    class IdempotentRaceBackend(fx.FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.barrier = threading.Barrier(2)
            self.guard = threading.Lock()
            self.physical_creates = 0
            self.owned: dict[tuple[str, str, str | None], dict[str, object]] = {}

        def admit(self, *, stage_input):
            self.barrier.wait(timeout=5)
            key = self._key_from_input(stage_input)
            token = self._token(key)
            with self.guard:
                self.admit_calls.append(copy.deepcopy(stage_input))
                if token not in self.owned:
                    self.physical_creates += 1
                    self.owned[token] = fx._active_observation(key)
                observation = copy.deepcopy(self.owned[token])
                self.observations[token] = copy.deepcopy(observation)
            return observation

    backend = IdempotentRaceBackend()

    def run_once():
        return fx._advance(tmp_path, backend)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_once) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]

    assert backend.physical_creates == 1
    assert len(backend.admit_calls) == 2
    assert all(item["status"] == "submitted" for item in results)
    assert all(item["terminal"] is False for item in results)
    current = fx._read_result(tmp_path / "mldb_data")
    assert current["trials"][0]["training"]["disposition"] == "pending"


def test_repeated_terminal_child_replay_is_exact_and_no_duplicate_parent_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    root, plan, _result = fx._install_repo(tmp_path)
    child_id = _write_failed_training_child(root, plan, "failed")
    before_child = _child_path(root, "training_result", child_id).read_bytes()

    first = fx._advance(tmp_path, fx.FakeBackend())
    parent_path = root / "demo" / "study_results" / f"{fx.STUDY_RESULT_ID.split('/', 1)[1]}.yaml"
    before_parent = parent_path.read_bytes()
    second = fx._advance(tmp_path, fx.FakeBackend())

    assert first["terminal"] is True
    assert second["changed"] is False
    assert second["terminal"] is True
    assert _child_path(root, "training_result", child_id).read_bytes() == before_child
    assert parent_path.read_bytes() == before_parent


def test_conflicting_same_id_terminal_child_without_model_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    root, plan, _result = fx._install_repo(tmp_path)
    child_id = _write_failed_training_child(root, plan, "failed")
    before = _child_path(root, "training_result", child_id).read_bytes()
    backend = fx.FakeBackend()
    key = fx._stage_key(plan)
    candidate = fx._terminal_observation(key, status="cancelled")
    backend.observations[backend._token(key)] = copy.deepcopy(candidate)
    backend.candidates[backend._token(key)] = copy.deepcopy(candidate)

    response = fx._advance(tmp_path, backend)
    current = fx._read_result(root)

    assert response["status"] == "failed"
    assert current["diagnostic"]["code"] == "study_progression_failed"
    assert current["trials"][0]["training"]["reason"] == "global_failure"
    assert _child_path(root, "training_result", child_id).read_bytes() == before


def _install_three_eval_repo(tmp_path: Path):
    root = tmp_path / "mldb_data"
    ready_fx._install_catalog(root)
    base = fx._single_trial_plan(evaluations=2)
    payload = {
        "schema": base["schema"],
        "study": base["study"],
        "source_commit": base["source_commit"],
        "pins": copy.deepcopy(base["pins"]),
        "trials": copy.deepcopy(base["trials"]),
    }
    third = copy.deepcopy(payload["trials"][0]["evaluations"][1])
    third["coordinate"] = "eval-0003"
    third["stage"] = "holdout-c"
    payload["trials"][0]["evaluations"].append(third)
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    plan = dict(_validate_study_plan({
        **payload,
        "id": _plan_id(str(payload["study"]), digest),
        "content_sha256": digest,
    }))
    result = fx._study_result(plan)
    fx._write_json(root / "demo" / "study_plans" / f"{plan['id'].split('/', 1)[1]}.yaml", plan)
    fx._write_json(root / "demo" / "study_results" / f"{result['id'].split('/', 1)[1]}.yaml", result)
    return root, plan, result


def test_three_evaluation_siblings_completed_failed_and_pending_are_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    root, plan, _result = _install_three_eval_repo(tmp_path)
    fx._seed_training_children(root, plan)
    fx._mark_training_completed(root)

    completed_id = f"{fx.STUDY_RESULT_ID}-trial-0001-eval-0001"
    fx._write_json(
        _child_path(root, "evaluation_result", completed_id),
        {"schema": "mjtensu.mldb-v2/evaluation-result/v1", "id": completed_id, "status": "completed"},
    )
    current = fx._read_result(root)
    current["trials"][0]["evaluations"][0].update(
        {"disposition": "completed", "result": completed_id, "reason": None}
    )
    fx._write_json(
        root / "demo" / "study_results" / f"{fx.STUDY_RESULT_ID.split('/', 1)[1]}.yaml",
        current,
    )

    backend = fx.FakeBackend()
    failed_key = fx._stage_key(plan, kind="evaluation", coordinate="eval-0002")
    failed = fx._terminal_observation(failed_key, status="failed")
    backend.observations[backend._token(failed_key)] = copy.deepcopy(failed)
    backend.candidates[backend._token(failed_key)] = copy.deepcopy(failed)
    response = fx._advance(tmp_path, backend)
    current = fx._read_result(root)

    assert current["trials"][0]["evaluations"][0]["disposition"] == "completed"
    assert current["trials"][0]["evaluations"][1]["disposition"] == "failed"
    assert current["trials"][0]["evaluations"][2]["disposition"] == "pending"
    assert response["admitted"] == [
        fx._stage_key(plan, kind="evaluation", coordinate="eval-0003")
    ]
    assert response["status"] == "submitted"


def test_cancelling_terminal_and_active_siblings_collect_and_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    root, plan, _result = fx._install_repo(tmp_path, evaluations=2, status="cancelling")
    fx._seed_training_children(root, plan)
    fx._mark_training_completed(root)
    backend = fx.FakeBackend()
    terminal_key = fx._stage_key(plan, kind="evaluation", coordinate="eval-0001")
    active_key = fx._stage_key(plan, kind="evaluation", coordinate="eval-0002")
    terminal = fx._terminal_observation(terminal_key, status="completed")
    backend.observations[backend._token(terminal_key)] = copy.deepcopy(terminal)
    backend.candidates[backend._token(terminal_key)] = copy.deepcopy(terminal)
    backend.observations[backend._token(active_key)] = fx._active_observation(active_key)
    first = fx._advance(tmp_path, backend)
    current = fx._read_result(root)

    assert first["status"] == "cancelling"
    assert current["trials"][0]["evaluations"][0]["disposition"] == "completed"
    assert current["trials"][0]["evaluations"][1]["disposition"] == "pending"
    assert backend.cancel_calls == [fx.STUDY_RESULT_ID]
    assert backend.admit_calls == []

    cancelled = fx._terminal_observation(active_key, status="cancelled")
    backend.observations[backend._token(active_key)] = copy.deepcopy(cancelled)
    backend.candidates[backend._token(active_key)] = copy.deepcopy(cancelled)
    second = fx._advance(tmp_path, backend)
    current = fx._read_result(root)

    assert second["status"] == "cancelled"
    assert current["trials"][0]["evaluations"][1]["disposition"] == "cancelled"
    assert backend.admit_calls == []


def test_two_callers_collecting_same_terminal_candidate_do_not_duplicate_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    root, plan, _result = fx._install_repo(tmp_path)
    key = fx._stage_key(plan)
    candidate = fx._terminal_observation(key, status="failed")
    class RacingTerminalBackend(fx.FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.barrier = threading.Barrier(2)

        def collect(self, *, stage_key):
            self.collect_calls.append(copy.deepcopy(stage_key))
            self.barrier.wait(timeout=5)
            return copy.deepcopy(self.candidates.get(self._token(stage_key)))

    backend = RacingTerminalBackend()
    backend.observations[backend._token(key)] = copy.deepcopy(candidate)
    backend.candidates[backend._token(key)] = copy.deepcopy(candidate)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fx._advance, tmp_path, backend) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]

    child_id = f"{fx.STUDY_RESULT_ID}-trial-0001-train"
    current = fx._read_result(root)
    assert len(backend.collect_calls) == 2
    assert _child_path(root, "training_result", child_id).is_file()
    assert current["trials"][0]["training"]["disposition"] == "failed"
    assert current["status"] == "completed_with_failures"
    assert all(item["status"] == "completed_with_failures" for item in results)


def _install_existing_model_repo(tmp_path: Path):
    root, base_plan, _base_result = ready_fx._fixture(tmp_path)
    existing_trial = copy.deepcopy(base_plan["trials"][1])
    existing_trial["trial"] = "trial-0001"
    payload = {
        "schema": base_plan["schema"],
        "study": base_plan["study"],
        "source_commit": base_plan["source_commit"],
        "pins": copy.deepcopy(base_plan["pins"]),
        "trials": [existing_trial],
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    plan = dict(_validate_study_plan({
        **payload,
        "id": _plan_id(str(payload["study"]), digest),
        "content_sha256": digest,
    }))
    result = {
        "schema": "mjtensu.mldb-v2/study-result/v1",
        "id": fx.STUDY_RESULT_ID,
        "execution_key": fx.EXECUTION_KEY,
        "plan": plan["id"],
        "study": plan["study"],
        "source_commit": plan["source_commit"],
        "backend": "fake",
        "created_at": "2026-09-13T00:00:00Z",
        "status": "submitted",
        "diagnostic": None,
        "trials": [{
            "trial": "trial-0001",
            "training": None,
            "evaluations": [
                {
                    "coordinate": item["coordinate"],
                    "stage": item["stage"],
                    "disposition": "pending",
                    "result": None,
                    "reason": None,
                }
                for item in existing_trial["evaluations"]
            ],
        }],
    }
    fx._write_json(
        root / "demo" / "study_plans" / f"{plan['id'].split('/', 1)[1]}.yaml", plan
    )
    fx._write_json(
        root / "demo" / "study_results" / f"{fx.STUDY_RESULT_ID.split('/', 1)[1]}.yaml",
        result,
    )
    return root, plan, result


def test_existing_model_siblings_complete_fail_and_repeated_advance_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_composition(monkeypatch)
    root, plan, _result = _install_existing_model_repo(tmp_path)
    backend = fx.FakeBackend()
    completed_key = fx._stage_key(plan, kind="evaluation", coordinate="eval-0001")
    failed_key = fx._stage_key(plan, kind="evaluation", coordinate="eval-0002")
    completed = fx._terminal_observation(completed_key, status="completed")
    failed = fx._terminal_observation(failed_key, status="failed")
    for key, candidate in ((completed_key, completed), (failed_key, failed)):
        backend.observations[backend._token(key)] = copy.deepcopy(candidate)
        backend.candidates[backend._token(key)] = copy.deepcopy(candidate)

    first = fx._advance(tmp_path, backend)
    current = fx._read_result(root)

    assert current["trials"][0]["training"] is None
    assert [slot["disposition"] for slot in current["trials"][0]["evaluations"]] == [
        "completed", "failed"
    ]
    assert first["status"] == "completed_with_failures"
    assert first["terminal"] is True

    class ExplodingBackend(fx.FakeBackend):
        def observe(self, *, stage_key):
            raise AssertionError("terminal existing-model Study must not observe")

    second = fx._advance(tmp_path, ExplodingBackend())
    assert second["changed"] is False
    assert second["status"] == "completed_with_failures"
    assert second["terminal"] is True
