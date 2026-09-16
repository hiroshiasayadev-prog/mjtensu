from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import mldb_v2.src.study.study_driver as driver
from mldb_v2.src.common.ids import EntityKind, _canonical_json_bytes
from mldb_v2.src.repository.canonical_writes import CanonicalRepositoryWriter as RealWriter
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study._plan_build import _plan_id, _validate_study_plan
import mldb_v2.tests.test_execution_readiness as ready_fx

SOURCE_COMMIT = ready_fx.SOURCE_COMMIT
EXECUTION_KEY = ready_fx.EXECUTION_KEY
STUDY_RESULT_ID = ready_fx.STUDY_RESULT_ID


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class NullTransport:
    def read_bytes(self, uri: str) -> bytes:
        raise AssertionError("fake driver acceptance must not read object bytes")

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        raise AssertionError("driver must not publish object bytes")


class PermissiveValidator:
    def __init__(self, *, mldb_data_root: str | Path) -> None:
        self.root = Path(mldb_data_root)

    def validate(self, *, kind, entity_id, document) -> None:
        return None


class FakeAcceptor:
    force_training_failure = False
    force_evaluation_failure = False

    def __init__(self, **_kwargs) -> None:
        pass

    def accept_training(self, *, request):
        candidate = request["candidate"]
        status = candidate.get("status", "completed")
        if self.force_training_failure and status == "completed":
            status = "failed"
        trial = request["stage_input"]["trial"]
        result = request["study_result"]
        plan = request["plan"]
        training_id = f"{result['id']}-{trial}-train"
        model_id = f"{result['id']}-{trial}-model"
        source = next(item for item in plan["trials"] if item["trial"] == trial)["source"]
        attempt = {
            "backend": result["backend"],
            "execution_id": "train-1",
            "status": status,
            "started_at": "2026-09-13T00:00:00Z",
            "ended_at": "2026-09-13T00:00:01Z",
            "diagnostic": None if status == "completed" else {"code": "terminal", "message": status},
        }
        training = {
            "schema": "mjtensu.mldb-v2/training-result/v1",
            "id": training_id,
            "study_result": result["id"],
            "plan": plan["id"],
            "trial": trial,
            "task": source["task"],
            "architecture": source["architecture"],
            "corpus": source["corpus"],
            "train_protocol": source["train_protocol"],
            "parameters": copy.deepcopy(source["parameters"]),
            "seed": source["seed"],
            "source_commit": plan["source_commit"],
            "attempts": [attempt],
            "status": status,
            "diagnostic": None if status == "completed" else {"code": "terminal", "message": status},
            "result": None,
        }
        model = None
        if status == "completed":
            training["result"] = {
                "weights": ready_fx._weights(),
                "model": model_id,
            }
            model = {
                "schema": "mjtensu.mldb-v2/model/v1",
                "id": model_id,
                "training_result": training_id,
            }
        return {"training_result": training, "model": model}

    def accept_evaluation(self, *, request):
        candidate = request["candidate"]
        status = candidate.get("status", "completed")
        if self.force_evaluation_failure and status == "completed":
            status = "failed"
        stage_input = request["stage_input"]
        result = request["study_result"]
        plan = request["plan"]
        result_id = f"{result['id']}-{stage_input['trial']}-{stage_input['coordinate']}"
        evaluation = {
            "schema": "mjtensu.mldb-v2/evaluation-result/v1",
            "id": result_id,
            "study_result": result["id"],
            "plan": plan["id"],
            "trial": stage_input["trial"],
            "coordinate": stage_input["coordinate"],
            "stage": stage_input["stage"]["name"],
            "model": stage_input["runtime_model"]["model"],
            "task": stage_input["stage"]["task"],
            "corpus": stage_input["stage"]["corpus"],
            "evaluation_protocol": stage_input["stage"]["evaluation_protocol"],
            "parameters": copy.deepcopy(stage_input["stage"]["parameters"]),
            "source_commit": plan["source_commit"],
            "attempts": [],
            "status": status,
            "diagnostic": None if status == "completed" else {"code": "terminal", "message": status},
            "result": {} if status == "completed" else None,
        }
        return {"evaluation_result": evaluation}


class FakeBackend:
    def __init__(self) -> None:
        self.observations: dict[tuple[str, str, str | None], dict[str, object] | None] = {}
        self.candidates: dict[tuple[str, str, str | None], dict[str, object] | None] = {}
        self.admit_calls: list[dict[str, object]] = []
        self.observe_calls: list[dict[str, object]] = []
        self.collect_calls: list[dict[str, object]] = []
        self.cancel_calls: list[str] = []
        self.raise_on: str | None = None

    @staticmethod
    def _token(stage_key):
        return (stage_key["trial"], stage_key["kind"], stage_key["coordinate"])

    @staticmethod
    def _key_from_input(stage_input):
        return {
            "study_result": stage_input["study_result"],
            "plan": stage_input["plan"],
            "trial": stage_input["trial"],
            "kind": stage_input["kind"],
            "coordinate": stage_input["coordinate"],
            "source_commit": stage_input["source_commit"],
        }

    def observe(self, *, stage_key):
        self.observe_calls.append(copy.deepcopy(stage_key))
        if self.raise_on == "observe":
            raise ConnectionError("transient")
        return copy.deepcopy(self.observations.get(self._token(stage_key)))

    def collect(self, *, stage_key):
        self.collect_calls.append(copy.deepcopy(stage_key))
        if self.raise_on == "collect":
            raise ConnectionError("transient")
        return copy.deepcopy(self.candidates.get(self._token(stage_key)))

    def admit(self, *, stage_input):
        if self.raise_on == "admit":
            raise ConnectionError("transient")
        self.admit_calls.append(copy.deepcopy(stage_input))
        key = self._key_from_input(stage_input)
        observation = {
            "state": "active",
            "stage_key": key,
            "backend": "fake",
            "execution_ids": [f"exec-{len(self.admit_calls)}"],
        }
        self.observations[self._token(key)] = copy.deepcopy(observation)
        return observation

    def cancel_study(self, *, study_result):
        self.cancel_calls.append(str(study_result))
        if self.raise_on == "cancel":
            raise ConnectionError("transient")


def _single_trial_plan(*, evaluations: int = 1) -> dict[str, object]:
    base = ready_fx._plan()
    payload = {
        "schema": base["schema"],
        "study": base["study"],
        "source_commit": base["source_commit"],
        "pins": copy.deepcopy(base["pins"]),
        "trials": [copy.deepcopy(base["trials"][0])],
    }
    payload["trials"][0]["evaluations"] = payload["trials"][0]["evaluations"][:evaluations]
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return dict(
        _validate_study_plan(
            {
                **payload,
                "id": _plan_id(str(payload["study"]), digest),
                "content_sha256": digest,
            }
        )
    )


def _study_result(plan: dict[str, object], *, status: str = "submitted") -> dict[str, object]:
    trial = plan["trials"][0]
    return {
        "schema": "mjtensu.mldb-v2/study-result/v1",
        "id": STUDY_RESULT_ID,
        "execution_key": EXECUTION_KEY,
        "plan": plan["id"],
        "study": plan["study"],
        "source_commit": plan["source_commit"],
        "backend": "fake",
        "created_at": "2026-09-13T00:00:00Z",
        "status": status,
        "diagnostic": None,
        "trials": [{
            "trial": trial["trial"],
            "training": {"disposition": "pending", "result": None, "reason": None},
            "evaluations": [
                {"coordinate": item["coordinate"], "stage": item["stage"], "disposition": "pending", "result": None, "reason": None}
                for item in trial["evaluations"]
            ],
        }],
    }


def _install_repo(tmp_path: Path, *, evaluations: int = 1, status: str = "submitted"):
    root = tmp_path / "mldb_data"
    ready_fx._install_catalog(root)
    plan = _single_trial_plan(evaluations=evaluations)
    result = _study_result(plan, status=status)
    _write_json(root / "demo" / "study_plans" / f"{plan['id'].split('/', 1)[1]}.yaml", plan)
    _write_json(root / "demo" / "study_results" / f"{result['id'].split('/', 1)[1]}.yaml", result)
    return root, plan, result


def _read_result(root: Path) -> dict[str, object]:
    path = root / "demo" / "study_results" / f"{STUDY_RESULT_ID.split('/', 1)[1]}.yaml"
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_key(plan, *, kind="training", coordinate=None):
    return {
        "study_result": STUDY_RESULT_ID,
        "plan": plan["id"],
        "trial": "trial-0001",
        "kind": kind,
        "coordinate": coordinate,
        "source_commit": SOURCE_COMMIT,
    }


def _terminal_observation(key, *, status="completed"):
    return {
        "state": "terminal",
        "stage_key": copy.deepcopy(key),
        "attempts": [],
        "status": status,
        "diagnostic": None if status == "completed" else {"code": "terminal", "message": status},
        "result": {} if status == "completed" else None,
    }


def _active_observation(key):
    return {
        "state": "active",
        "stage_key": copy.deepcopy(key),
        "backend": "fake",
        "execution_ids": ["exec-1"],
    }


@pytest.fixture(autouse=True)
def _patch_driver_composition(monkeypatch):
    FakeAcceptor.force_training_failure = False
    FakeAcceptor.force_evaluation_failure = False
    monkeypatch.setattr(driver, "ResultAcceptor", FakeAcceptor)
    monkeypatch.setattr(driver, "AcceptedResultRecordValidator", PermissiveValidator)


def _advance(tmp_path: Path, backend: FakeBackend):
    return driver.advance_study(
        repository_root=tmp_path,
        study_result_id=STUDY_RESULT_ID,
        backend=backend,
        object_bytes=_ObjectByteAccess(NullTransport()),
    )


def test_active_pending_work_is_noop_and_reports_exact_stage_key(tmp_path: Path) -> None:
    _root, plan, _result = _install_repo(tmp_path)
    backend = FakeBackend()
    key = _stage_key(plan)
    backend.observations[backend._token(key)] = _active_observation(key)

    response = _advance(tmp_path, backend)

    assert response["changed"] is False
    assert response["active"] == [key]
    assert response["admitted"] == []
    assert response["status"] == "submitted"


def test_initial_training_admission_then_repeated_pass_is_idempotent(tmp_path: Path) -> None:
    _root, plan, _result = _install_repo(tmp_path)
    backend = FakeBackend()

    first = _advance(tmp_path, backend)
    second = _advance(tmp_path, backend)

    assert first["admitted"] == [_stage_key(plan)]
    assert first["changed"] is True
    assert len(backend.admit_calls) == 1
    assert second["admitted"] == []
    assert second["changed"] is False
    assert second["active"] == [_stage_key(plan)]


@pytest.mark.parametrize(
    ("status", "expected_study", "model_count"),
    [
        ("completed", "submitted", 1),
        ("failed", "completed_with_failures", 0),
        ("cancelled", "completed_with_failures", 0),
    ],
)
def test_terminal_training_reconciles_child_before_parent_and_closure(
    tmp_path: Path, status: str, expected_study: str, model_count: int
) -> None:
    root, plan, _result = _install_repo(tmp_path)
    backend = FakeBackend()
    key = _stage_key(plan)
    candidate = _terminal_observation(key, status=status)
    backend.observations[backend._token(key)] = copy.deepcopy(candidate)
    backend.candidates[backend._token(key)] = copy.deepcopy(candidate)

    response = _advance(tmp_path, backend)
    current = _read_result(root)

    assert len(response["finalized_results"]) == 1
    assert len(response["finalized_models"]) == model_count
    assert current["trials"][0]["training"]["disposition"] == status
    assert current["status"] == expected_study
    if status == "completed":
        assert len(response["admitted"]) == 1
        assert response["admitted"][0]["kind"] == "evaluation"
    else:
        assert current["trials"][0]["evaluations"][0]["disposition"] == "skipped"


def test_completed_training_acceptance_failure_becomes_formal_failed(tmp_path: Path) -> None:
    root, plan, _result = _install_repo(tmp_path)
    backend = FakeBackend()
    key = _stage_key(plan)
    candidate = _terminal_observation(key, status="completed")
    backend.observations[backend._token(key)] = copy.deepcopy(candidate)
    backend.candidates[backend._token(key)] = copy.deepcopy(candidate)
    FakeAcceptor.force_training_failure = True

    response = _advance(tmp_path, backend)
    current = _read_result(root)

    assert response["status"] == "completed_with_failures"
    assert current["trials"][0]["training"]["disposition"] == "failed"
    assert current["trials"][0]["evaluations"][0]["reason"] == "upstream_failed"


def _seed_training_children(root: Path, plan: dict[str, object], *, model: bool = True) -> None:
    training = ready_fx._training_result(
        training_result_id=ready_fx.RUNTIME_TRAINING_RESULT_ID,
        model_id=ready_fx.RUNTIME_MODEL_ID,
        plan_id=str(plan["id"]),
        study_result_id=STUDY_RESULT_ID,
        trial="trial-0001",
        source_commit=SOURCE_COMMIT,
        seed=42,
    )
    _write_json(
        root / "demo" / "training_results" / f"{ready_fx.RUNTIME_TRAINING_RESULT_ID.split('/', 1)[1]}.yaml",
        training,
    )
    if model:
        _write_json(
            root / "demo" / "models" / f"{ready_fx.RUNTIME_MODEL_ID.split('/', 1)[1]}.yaml",
            {
                "schema": "mjtensu.mldb-v2/model/v1",
                "id": ready_fx.RUNTIME_MODEL_ID,
                "training_result": ready_fx.RUNTIME_TRAINING_RESULT_ID,
            },
        )


def _mark_training_completed(root: Path) -> None:
    current = _read_result(root)
    current["trials"][0]["training"] = {
        "disposition": "completed",
        "result": ready_fx.RUNTIME_TRAINING_RESULT_ID,
        "reason": None,
    }
    _write_json(
        root / "demo" / "study_results" / f"{STUDY_RESULT_ID.split('/', 1)[1]}.yaml",
        current,
    )


def test_interrupted_after_training_result_recovers_model_then_parent(tmp_path: Path) -> None:
    root, plan, _result = _install_repo(tmp_path)
    _seed_training_children(root, plan, model=False)
    backend = FakeBackend()

    response = _advance(tmp_path, backend)
    current = _read_result(root)

    model_path = root / "demo" / "models" / f"{ready_fx.RUNTIME_MODEL_ID.split('/', 1)[1]}.yaml"
    assert model_path.is_file()
    assert current["trials"][0]["training"]["disposition"] == "completed"
    assert response["finalized_results"] == [ready_fx.RUNTIME_TRAINING_RESULT_ID]
    assert response["finalized_models"] == [ready_fx.RUNTIME_MODEL_ID]


def test_interrupted_after_model_recovers_parent_without_rewrite(tmp_path: Path) -> None:
    root, plan, _result = _install_repo(tmp_path)
    _seed_training_children(root, plan, model=True)
    model_path = root / "demo" / "models" / f"{ready_fx.RUNTIME_MODEL_ID.split('/', 1)[1]}.yaml"
    before = model_path.read_bytes()

    response = _advance(tmp_path, FakeBackend())
    current = _read_result(root)

    assert current["trials"][0]["training"]["disposition"] == "completed"
    assert model_path.read_bytes() == before
    assert response["finalized_models"] == [ready_fx.RUNTIME_MODEL_ID]


@pytest.mark.parametrize(
    ("status", "expected_study"),
    [
        ("completed", "completed"),
        ("failed", "completed_with_failures"),
        ("cancelled", "completed_with_failures"),
    ],
)
def test_terminal_evaluation_reconciles_and_closes(
    tmp_path: Path, status: str, expected_study: str
) -> None:
    root, plan, _result = _install_repo(tmp_path)
    _seed_training_children(root, plan)
    _mark_training_completed(root)
    backend = FakeBackend()
    key = _stage_key(plan, kind="evaluation", coordinate="eval-0001")
    candidate = _terminal_observation(key, status=status)
    backend.observations[backend._token(key)] = copy.deepcopy(candidate)
    backend.candidates[backend._token(key)] = copy.deepcopy(candidate)

    response = _advance(tmp_path, backend)
    current = _read_result(root)

    assert current["trials"][0]["evaluations"][0]["disposition"] == status
    assert response["status"] == expected_study
    assert response["terminal"] is True


def test_completed_evaluation_acceptance_failure_becomes_failed(tmp_path: Path) -> None:
    root, plan, _result = _install_repo(tmp_path)
    _seed_training_children(root, plan)
    _mark_training_completed(root)
    backend = FakeBackend()
    key = _stage_key(plan, kind="evaluation", coordinate="eval-0001")
    candidate = _terminal_observation(key)
    backend.observations[backend._token(key)] = copy.deepcopy(candidate)
    backend.candidates[backend._token(key)] = copy.deepcopy(candidate)
    FakeAcceptor.force_evaluation_failure = True

    response = _advance(tmp_path, backend)
    assert response["status"] == "completed_with_failures"
    assert _read_result(root)["trials"][0]["evaluations"][0]["disposition"] == "failed"


def test_training_persistence_order_is_result_model_parent(tmp_path: Path, monkeypatch) -> None:
    _root, plan, _result = _install_repo(tmp_path)
    backend = FakeBackend()
    key = _stage_key(plan)
    candidate = _terminal_observation(key)
    backend.observations[backend._token(key)] = copy.deepcopy(candidate)
    backend.candidates[backend._token(key)] = copy.deepcopy(candidate)
    events: list[str] = []

    class RecordingWriter(RealWriter):
        def create_immutable(self, *, kind, entity_id, document):
            events.append(kind.value)
            return super().create_immutable(kind=kind, entity_id=entity_id, document=document)

        def replace_nonterminal_study_result(self, *, entity_id, replacement):
            events.append("study_result")
            return super().replace_nonterminal_study_result(entity_id=entity_id, replacement=replacement)

    monkeypatch.setattr(driver, "CanonicalRepositoryWriter", RecordingWriter)
    _advance(tmp_path, backend)
    assert events[:3] == ["training_result", "model", "study_result"]


def test_evaluation_persistence_order_is_child_before_parent(tmp_path: Path, monkeypatch) -> None:
    root, plan, _result = _install_repo(tmp_path)
    _seed_training_children(root, plan)
    _mark_training_completed(root)
    backend = FakeBackend()
    key = _stage_key(plan, kind="evaluation", coordinate="eval-0001")
    candidate = _terminal_observation(key)
    backend.observations[backend._token(key)] = copy.deepcopy(candidate)
    backend.candidates[backend._token(key)] = copy.deepcopy(candidate)
    events: list[str] = []

    class RecordingWriter(RealWriter):
        def create_immutable(self, *, kind, entity_id, document):
            events.append(kind.value)
            return super().create_immutable(kind=kind, entity_id=entity_id, document=document)

        def replace_nonterminal_study_result(self, *, entity_id, replacement):
            events.append("study_result")
            return super().replace_nonterminal_study_result(entity_id=entity_id, replacement=replacement)

    monkeypatch.setattr(driver, "CanonicalRepositoryWriter", RecordingWriter)
    _advance(tmp_path, backend)
    assert events[:2] == ["evaluation_result", "study_result"]


def test_interrupted_after_evaluation_child_write_finishes_parent(tmp_path: Path) -> None:
    root, plan, _result = _install_repo(tmp_path)
    _seed_training_children(root, plan)
    _mark_training_completed(root)
    child_id = f"{STUDY_RESULT_ID}-trial-0001-eval-0001"
    _write_json(
        root / "demo" / "evaluation_results" / f"{child_id.split('/', 1)[1]}.yaml",
        {"schema": "mjtensu.mldb-v2/evaluation-result/v1", "id": child_id, "status": "completed"},
    )

    response = _advance(tmp_path, FakeBackend())
    current = _read_result(root)

    assert current["trials"][0]["evaluations"][0]["disposition"] == "completed"
    assert response["status"] == "completed"
    assert response["finalized_results"] == [child_id]


def test_evaluation_sibling_failure_does_not_block_other_sibling(tmp_path: Path) -> None:
    root, plan, _result = _install_repo(tmp_path, evaluations=2)
    _seed_training_children(root, plan)
    _mark_training_completed(root)
    backend = FakeBackend()
    failed_key = _stage_key(plan, kind="evaluation", coordinate="eval-0001")
    candidate = _terminal_observation(failed_key, status="failed")
    backend.observations[backend._token(failed_key)] = copy.deepcopy(candidate)
    backend.candidates[backend._token(failed_key)] = copy.deepcopy(candidate)

    response = _advance(tmp_path, backend)
    current = _read_result(root)

    assert current["trials"][0]["evaluations"][0]["disposition"] == "failed"
    assert current["trials"][0]["evaluations"][1]["disposition"] == "pending"
    assert response["admitted"] == [_stage_key(plan, kind="evaluation", coordinate="eval-0002")]
    assert response["status"] == "submitted"


def test_cancelling_never_admitted_stages_skip_and_close_cancelled(tmp_path: Path) -> None:
    root, _plan, _result = _install_repo(tmp_path, status="cancelling")
    backend = FakeBackend()

    response = _advance(tmp_path, backend)
    current = _read_result(root)

    assert response["status"] == "cancelled"
    assert response["terminal"] is True
    assert backend.cancel_calls == []
    assert backend.admit_calls == []
    assert current["trials"][0]["training"] == {
        "disposition": "skipped", "result": None, "reason": "study_cancelled"
    }
    assert current["trials"][0]["evaluations"][0]["reason"] == "study_cancelled"


def test_cancelling_active_admitted_work_requests_backend_cancel_only(tmp_path: Path) -> None:
    root, plan, _result = _install_repo(tmp_path, status="cancelling")
    backend = FakeBackend()
    training_key = _stage_key(plan)
    backend.observations[backend._token(training_key)] = _active_observation(training_key)

    response = _advance(tmp_path, backend)
    current = _read_result(root)

    assert backend.cancel_calls == [STUDY_RESULT_ID]
    assert backend.admit_calls == []
    assert response["active"] == [training_key]
    assert response["status"] == "cancelling"
    assert current["trials"][0]["training"]["disposition"] == "pending"
    assert current["trials"][0]["evaluations"][0]["reason"] == "study_cancelled"


def test_cancelling_terminal_admitted_work_is_collected_normally(tmp_path: Path) -> None:
    root, plan, _result = _install_repo(tmp_path, status="cancelling")
    backend = FakeBackend()
    training_key = _stage_key(plan)
    candidate = _terminal_observation(training_key, status="cancelled")
    backend.observations[backend._token(training_key)] = copy.deepcopy(candidate)
    backend.candidates[backend._token(training_key)] = copy.deepcopy(candidate)

    response = _advance(tmp_path, backend)
    current = _read_result(root)

    assert len(backend.collect_calls) == 1
    assert backend.admit_calls == []
    assert response["status"] == "cancelled"
    assert current["trials"][0]["training"]["disposition"] == "cancelled"
    assert current["trials"][0]["evaluations"][0]["reason"] == "upstream_cancelled"


def test_stale_prepared_candidate_re_resolves_and_does_not_overwrite_newer_parent(tmp_path: Path) -> None:
    root, plan, _result = _install_repo(tmp_path)
    training_key = _stage_key(plan)
    candidate = _terminal_observation(training_key, status="completed")

    class StaleBackend(FakeBackend):
        def collect(self, *, stage_key):
            current = _read_result(root)
            current["trials"][0]["training"] = {
                "disposition": "failed",
                "result": f"{STUDY_RESULT_ID}-trial-0001-train",
                "reason": None,
            }
            _write_json(
                root / "demo" / "study_results" / f"{STUDY_RESULT_ID.split('/', 1)[1]}.yaml",
                current,
            )
            return copy.deepcopy(candidate)

    backend = StaleBackend()
    backend.observations[backend._token(training_key)] = copy.deepcopy(candidate)
    response = _advance(tmp_path, backend)
    current = _read_result(root)

    assert current["trials"][0]["training"]["disposition"] == "failed"
    assert current["trials"][0]["evaluations"][0]["reason"] == "upstream_failed"
    assert response["status"] == "completed_with_failures"
    training_path = root / "demo" / "training_results" / f"{STUDY_RESULT_ID.split('/', 1)[1]}-trial-0001-train.yaml"
    assert not training_path.exists()


def test_backend_calls_are_outside_study_result_mutation_lock(tmp_path: Path, monkeypatch) -> None:
    _root, _plan, _result = _install_repo(tmp_path)
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

    class AssertUnlockedBackend(FakeBackend):
        def observe(self, *, stage_key):
            assert held["value"] is False
            return super().observe(stage_key=stage_key)
        def admit(self, *, stage_input):
            assert held["value"] is False
            return super().admit(stage_input=stage_input)
        def cancel_study(self, *, study_result):
            assert held["value"] is False
            return super().cancel_study(study_result=study_result)

    monkeypatch.setattr(driver, "StudyResultMutationCoordinator", TrackingCoordinator)
    _advance(tmp_path, AssertUnlockedBackend())


@pytest.mark.parametrize("mode", ["observe", "collect", "admit"])
def test_transient_backend_unavailable_never_global_fails(tmp_path: Path, mode: str) -> None:
    root, plan, _result = _install_repo(tmp_path)
    backend = FakeBackend()
    backend.raise_on = mode
    if mode == "collect":
        key = _stage_key(plan)
        terminal = _terminal_observation(key)
        backend.observations[backend._token(key)] = terminal

    response = _advance(tmp_path, backend)
    current = _read_result(root)

    assert response["status"] == "submitted"
    assert response["terminal"] is False
    assert current["status"] == "submitted"
    assert all(
        slot["reason"] != "global_failure"
        for slot in [current["trials"][0]["training"], *current["trials"][0]["evaluations"]]
    )


def test_invalid_backend_stage_key_is_bounded_global_failure(tmp_path: Path) -> None:
    root, plan, _result = _install_repo(tmp_path)
    backend = FakeBackend()
    key = _stage_key(plan)
    wrong = _active_observation(key)
    wrong["stage_key"]["source_commit"] = "b" * 40
    backend.observations[backend._token(key)] = wrong

    response = _advance(tmp_path, backend)
    current = _read_result(root)

    assert response["status"] == "failed"
    assert response["terminal"] is True
    assert current["diagnostic"]["code"] == "study_progression_failed"
    assert current["trials"][0]["training"]["reason"] == "global_failure"
    assert current["trials"][0]["evaluations"][0]["reason"] == "global_failure"


def test_unexpected_backend_exception_is_not_converted_to_global_failure(tmp_path: Path) -> None:
    root, _plan, _result = _install_repo(tmp_path)

    class ExplodingBackend(FakeBackend):
        def observe(self, *, stage_key):
            raise RuntimeError("programming bug")

    with pytest.raises(RuntimeError, match="programming bug"):
        _advance(tmp_path, ExplodingBackend())
    assert _read_result(root)["status"] == "submitted"


def test_existing_model_evaluation_is_initially_admitted(tmp_path: Path) -> None:
    root, base_plan, _base_result = ready_fx._fixture(tmp_path)
    existing_trial = copy.deepcopy(base_plan["trials"][1])
    existing_trial["trial"] = "trial-0001"
    existing_trial["evaluations"] = existing_trial["evaluations"][:1]
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
        "id": STUDY_RESULT_ID,
        "execution_key": EXECUTION_KEY,
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
            "evaluations": [{"coordinate": "eval-0001", "stage": existing_trial["evaluations"][0]["stage"], "disposition": "pending", "result": None, "reason": None}],
        }],
    }
    _write_json(root / "demo" / "study_plans" / f"{plan['id'].split('/', 1)[1]}.yaml", plan)
    _write_json(root / "demo" / "study_results" / f"{STUDY_RESULT_ID.split('/', 1)[1]}.yaml", result)

    response = _advance(tmp_path, FakeBackend())
    assert response["admitted"] == [_stage_key(plan, kind="evaluation", coordinate="eval-0001")]


def test_driver_source_has_no_clearml_api_scheduler_or_retry_policy_dependency() -> None:
    source = Path(driver.__file__).read_text(encoding="utf-8")
    folded = source.casefold()
    assert "clearml" not in folded
    assert "mldb_v2.src.api" not in source
    assert "mldb_v2.skeleton" not in source
    assert "gpu" not in folded
    assert "queue" not in folded
    assert "retry" not in folded
