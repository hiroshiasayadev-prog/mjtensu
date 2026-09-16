import copy
import json
import threading

import pytest

from mldb_v2.src.common.ids import EntityKind, StudyPlanId, StudyResultId
from mldb_v2.src.repository import canonical_writes as writes
from mldb_v2.src.repository.canonical_writes import CanonicalRepositoryWriter


def _repo(tmp_path):
    namespace = tmp_path / "mldb_data" / "demo"
    namespace.mkdir(parents=True)
    (namespace / "namespace.yaml").write_text(
        json.dumps({"schema": "mjtensu.mldb-v2/namespace/v1", "id": "demo"}),
        encoding="utf-8",
    )
    return tmp_path


def _immutable(entity_id="demo/example-plan", value=1):
    return {
        "schema": "mjtensu.mldb-v2/study-plan/v1",
        "id": entity_id,
        "value": value,
    }


class FakeValidationError(ValueError):
    pass


class RecordingValidator:
    def __init__(self, *, reject=None) -> None:
        self.calls = []
        self._reject = reject

    def validate(self, *, kind, entity_id, document) -> None:
        self.calls.append((kind, entity_id, document))
        if self._reject is not None and self._reject(document):
            raise FakeValidationError("fake canonical record rejection")


def _writer(repo, *, record_validator=None):
    if record_validator is None:
        record_validator = RecordingValidator()
    return CanonicalRepositoryWriter(repo, record_validator=record_validator)


def _study_result():
    key = "123e4567e89b42d3a456426614174000"
    return {
        "schema": "mjtensu.mldb-v2/study-result/v1",
        "id": f"demo/run-{key}",
        "execution_key": key,
        "plan": "demo/example-plan",
        "study": "demo/example-study",
        "source_commit": "b" * 40,
        "backend": "clearml",
        "created_at": "2026-09-09T09:00:00Z",
        "status": "submitted",
        "diagnostic": None,
        "trials": [
            {
                "trial": "trial-0001",
                "training": {
                    "disposition": "pending",
                    "result": None,
                    "reason": None,
                },
                "evaluations": [
                    {
                        "coordinate": "eval-0001",
                        "stage": "holdout",
                        "disposition": "pending",
                        "result": None,
                        "reason": None,
                    }
                ],
            }
        ],
    }


def _study_path(repo, document):
    return (
        repo
        / "mldb_data"
        / "demo"
        / "study_results"
        / f"{document['id'].split('/', 1)[1]}.yaml"
    )


def _complete(document):
    replacement = copy.deepcopy(document)
    key = replacement["execution_key"]
    trial = replacement["trials"][0]
    trial["training"] = {
        "disposition": "completed",
        "result": f"demo/run-{key}-trial-0001-train",
        "reason": None,
    }
    trial["evaluations"][0].update(
        disposition="completed",
        result=f"demo/run-{key}-trial-0001-eval-0001",
        reason=None,
    )
    replacement["status"] = "completed"
    return replacement


def test_writer_requires_non_optional_record_validator(tmp_path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(TypeError):
        CanonicalRepositoryWriter(repo)
    with pytest.raises(TypeError, match="record_validator"):
        CanonicalRepositoryWriter(repo, record_validator=None)


def test_new_immutable_validation_receives_exact_proposed_document(tmp_path) -> None:
    repo = _repo(tmp_path)
    validator = RecordingValidator()
    writer = _writer(repo, record_validator=validator)
    entity_id = StudyPlanId("demo/example-plan")
    document = _immutable()

    assert writer.create_immutable(
        kind=EntityKind.STUDY_PLAN,
        entity_id=entity_id,
        document=document,
    ) == document
    assert len(validator.calls) == 1
    kind, called_id, called_document = validator.calls[0]
    assert kind == EntityKind.STUDY_PLAN
    assert called_id == entity_id
    assert called_document is document


def test_new_immutable_validation_failure_propagates_without_creating_file(tmp_path) -> None:
    repo = _repo(tmp_path)
    validator = RecordingValidator(reject=lambda document: True)
    writer = _writer(repo, record_validator=validator)
    entity_id = StudyPlanId("demo/example-plan")
    document = _immutable()
    path = repo / "mldb_data" / "demo" / "study_plans" / "example-plan.yaml"

    with pytest.raises(FakeValidationError, match="fake canonical record rejection"):
        writer.create_immutable(
            kind=EntityKind.STUDY_PLAN,
            entity_id=entity_id,
            document=document,
        )
    assert len(validator.calls) == 1
    assert not path.exists()


def test_immutable_replay_validates_proposed_then_existing_before_success(tmp_path) -> None:
    repo = _repo(tmp_path)
    entity_id = StudyPlanId("demo/example-plan")
    document = _immutable()
    _writer(repo).create_immutable(
        kind=EntityKind.STUDY_PLAN,
        entity_id=entity_id,
        document=document,
    )
    proposed = copy.deepcopy(document)
    validator = RecordingValidator()
    writer = _writer(repo, record_validator=validator)

    assert writer.create_immutable(
        kind=EntityKind.STUDY_PLAN,
        entity_id=entity_id,
        document=proposed,
    ) == document
    assert len(validator.calls) == 2
    assert validator.calls[0][2] is proposed
    assert validator.calls[1][2] is not proposed
    assert validator.calls[1][2] == document


def test_immutable_replay_rejects_invalid_existing_even_when_content_matches(tmp_path) -> None:
    repo = _repo(tmp_path)
    entity_id = StudyPlanId("demo/example-plan")
    document = _immutable()
    _writer(repo).create_immutable(
        kind=EntityKind.STUDY_PLAN,
        entity_id=entity_id,
        document=document,
    )
    path = repo / "mldb_data" / "demo" / "study_plans" / "example-plan.yaml"
    original = path.read_bytes()
    proposed = copy.deepcopy(document)
    validator = RecordingValidator(reject=lambda candidate: candidate is not proposed)
    writer = _writer(repo, record_validator=validator)

    with pytest.raises(FakeValidationError, match="fake canonical record rejection"):
        writer.create_immutable(
            kind=EntityKind.STUDY_PLAN,
            entity_id=entity_id,
            document=proposed,
        )
    assert len(validator.calls) == 2
    assert path.read_bytes() == original


def test_immutable_proposed_validation_failure_precedes_existing_conflict(tmp_path) -> None:
    repo = _repo(tmp_path)
    entity_id = StudyPlanId("demo/example-plan")
    existing = _immutable(value=1)
    _writer(repo).create_immutable(
        kind=EntityKind.STUDY_PLAN,
        entity_id=entity_id,
        document=existing,
    )
    path = repo / "mldb_data" / "demo" / "study_plans" / "example-plan.yaml"
    original = path.read_bytes()
    proposed = _immutable(value=2)
    validator = RecordingValidator(reject=lambda candidate: candidate is proposed)
    writer = _writer(repo, record_validator=validator)

    with pytest.raises(FakeValidationError, match="fake canonical record rejection"):
        writer.create_immutable(
            kind=EntityKind.STUDY_PLAN,
            entity_id=entity_id,
            document=proposed,
        )
    assert len(validator.calls) == 1
    assert validator.calls[0][2] is proposed
    assert path.read_bytes() == original


def test_immutable_valid_differing_replay_is_lifecycle_conflict(tmp_path) -> None:
    repo = _repo(tmp_path)
    entity_id = StudyPlanId("demo/example-plan")
    _writer(repo).create_immutable(
        kind=EntityKind.STUDY_PLAN,
        entity_id=entity_id,
        document=_immutable(value=1),
    )
    path = repo / "mldb_data" / "demo" / "study_plans" / "example-plan.yaml"
    original = path.read_bytes()
    validator = RecordingValidator()
    writer = _writer(repo, record_validator=validator)

    with pytest.raises(ValueError, match="lifecycle conflict"):
        writer.create_immutable(
            kind=EntityKind.STUDY_PLAN,
            entity_id=entity_id,
            document=_immutable(value=2),
        )
    assert len(validator.calls) == 2
    assert path.read_bytes() == original


def test_immutable_repository_generic_checks_remain_repository_owned(tmp_path) -> None:
    repo = _repo(tmp_path)
    validator = RecordingValidator()
    writer = _writer(repo, record_validator=validator)
    entity_id = StudyPlanId("demo/example-plan")

    with pytest.raises(ValueError, match="identity mismatch"):
        writer.create_immutable(
            kind=EntityKind.STUDY_PLAN,
            entity_id=entity_id,
            document=_immutable(entity_id="demo/other-plan"),
        )
    with pytest.raises(ValueError, match="immutable canonical kind"):
        writer.create_immutable(
            kind=EntityKind.STUDY_RESULT,
            entity_id=entity_id,
            document=_immutable(),
        )
    unserializable = _immutable()
    unserializable["value"] = object()
    with pytest.raises(ValueError, match="JSON-compatible"):
        writer.create_immutable(
            kind=EntityKind.STUDY_PLAN,
            entity_id=entity_id,
            document=unserializable,
        )
    assert validator.calls == []


def test_immutable_create_replay_and_conflict_preserves_existing(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    entity_id = StudyPlanId("demo/example-plan")
    document = _immutable()

    assert writer.create_immutable(
        kind=EntityKind.STUDY_PLAN,
        entity_id=entity_id,
        document=document,
    ) == document
    assert writer.create_immutable(
        kind=EntityKind.STUDY_PLAN,
        entity_id=entity_id,
        document=dict(reversed(list(document.items()))),
    ) == document

    path = repo / "mldb_data" / "demo" / "study_plans" / "example-plan.yaml"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="lifecycle conflict"):
        writer.create_immutable(
            kind=EntityKind.STUDY_PLAN,
            entity_id=entity_id,
            document=_immutable(value=2),
        )
    assert path.read_bytes() == original


def test_study_result_initial_replay_and_identity_mismatch(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])

    assert writer.create_study_result(entity_id=entity_id, document=document) == document
    assert writer.create_study_result(entity_id=entity_id, document=copy.deepcopy(document)) == document

    mismatched = copy.deepcopy(document)
    mismatched["id"] = "demo/run-" + "c" * 32
    with pytest.raises(ValueError, match="identity mismatch"):
        writer.create_study_result(entity_id=entity_id, document=mismatched)


def test_valid_nonterminal_replacement_and_backwards_stage_rejected(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    progressed = copy.deepcopy(document)
    key = progressed["execution_key"]
    progressed["trials"][0]["training"] = {
        "disposition": "completed",
        "result": f"demo/run-{key}-trial-0001-train",
        "reason": None,
    }
    assert writer.replace_nonterminal_study_result(
        entity_id=entity_id,
        replacement=progressed,
    ) == progressed

    backwards = copy.deepcopy(progressed)
    backwards["trials"][0]["training"] = {
        "disposition": "pending",
        "result": None,
        "reason": None,
    }
    with pytest.raises(ValueError, match="terminal stage disposition is immutable"):
        writer.replace_nonterminal_study_result(
            entity_id=entity_id,
            replacement=backwards,
        )


def test_terminal_current_rejects_change_but_exact_replay_is_idempotent(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)
    terminal = _complete(document)
    writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=terminal)

    assert writer.replace_nonterminal_study_result(
        entity_id=entity_id,
        replacement=copy.deepcopy(terminal),
    ) == terminal
    changed = copy.deepcopy(terminal)
    changed["diagnostic"] = {"code": "late_change", "message": "not allowed"}
    with pytest.raises(ValueError):
        writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=changed)


def test_atomic_replacement_never_exposes_partial_or_temp_yaml(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)
    replacement = _complete(document)
    path = _study_path(repo, document)

    entered = threading.Event()
    release = threading.Event()
    real_replace = writes.os.replace

    def delayed_replace(source, destination):
        entered.set()
        assert release.wait(timeout=5)
        real_replace(source, destination)

    monkeypatch.setattr(writes.os, "replace", delayed_replace)
    errors = []

    def mutate():
        try:
            writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=replacement)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=mutate)
    thread.start()
    assert entered.wait(timeout=5)
    assert json.loads(path.read_text(encoding="utf-8")) == document
    assert [p.name for p in path.parent.glob("*.yaml")] == [path.name]
    release.set()
    thread.join(timeout=5)
    assert not errors
    assert json.loads(path.read_text(encoding="utf-8")) == replacement


def test_failure_before_replace_leaves_old_canonical_intact(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)
    path = _study_path(repo, document)
    original = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("injected before replace")

    monkeypatch.setattr(writes.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=_complete(document))
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))


def test_immutable_replay_ignores_yaml_incidental_formatting(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    entity_id = StudyPlanId("demo/example-plan")
    path = repo / "mldb_data" / "demo" / "study_plans" / "example-plan.yaml"
    path.parent.mkdir(parents=True)
    original = """schema: mjtensu.mldb-v2/study-plan/v1
id: demo/example-plan
value: 1
"""
    path.write_text(original, encoding="utf-8")

    assert writer.create_immutable(
        kind=EntityKind.STUDY_PLAN,
        entity_id=entity_id,
        document=_immutable(),
    ) == _immutable()
    assert path.read_text(encoding="utf-8") == original


def test_study_result_rejects_non_uuid4_execution_key(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    document["execution_key"] = "a" * 32
    document["id"] = "demo/run-" + "a" * 32
    entity_id = StudyResultId(document["id"])
    with pytest.raises(ValueError, match="UUID4"):
        writer.create_study_result(entity_id=entity_id, document=document)


def test_failed_closure_requires_global_failure_skip(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    invalid = copy.deepcopy(document)
    key = invalid["execution_key"]
    invalid["trials"][0]["training"] = {
        "disposition": "failed",
        "result": f"demo/run-{key}-trial-0001-train",
        "reason": None,
    }
    invalid["trials"][0]["evaluations"][0].update(
        disposition="skipped", result=None, reason="upstream_failed"
    )
    invalid["status"] = "failed"
    invalid["diagnostic"] = {"code": "execution_failed", "message": "failure"}
    with pytest.raises(ValueError, match="global_failure"):
        writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=invalid)


def test_completed_with_failures_rejects_study_cancelled_skip(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    invalid = copy.deepcopy(document)
    key = invalid["execution_key"]
    invalid["trials"][0]["training"] = {
        "disposition": "completed",
        "result": f"demo/run-{key}-trial-0001-train",
        "reason": None,
    }
    invalid["trials"][0]["evaluations"][0].update(
        disposition="skipped", result=None, reason="study_cancelled"
    )
    invalid["status"] = "completed_with_failures"
    with pytest.raises(ValueError, match="Study cancellation"):
        writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=invalid)


def test_submitted_with_terminal_slots_is_rejected(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    invalid = _complete(document)
    invalid["status"] = "submitted"
    with pytest.raises(ValueError, match="at least one pending"):
        writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=invalid)


def test_cancelling_with_terminal_slots_is_rejected(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    cancelling = copy.deepcopy(document)
    cancelling["status"] = "cancelling"
    writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=cancelling)

    invalid = _complete(cancelling)
    invalid["status"] = "cancelling"
    with pytest.raises(ValueError, match="at least one pending"):
        writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=invalid)


def test_submitted_and_cancelling_with_pending_stage_are_valid(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    submitted = _study_result()
    entity_id = StudyResultId(submitted["id"])
    assert writer.create_study_result(entity_id=entity_id, document=submitted) == submitted

    cancelling = copy.deepcopy(submitted)
    cancelling["status"] = "cancelling"
    assert writer.replace_nonterminal_study_result(
        entity_id=entity_id,
        replacement=cancelling,
    ) == cancelling


def test_completed_requires_and_accepts_exact_all_completed_closure(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    completed = _complete(document)
    assert writer.replace_nonterminal_study_result(
        entity_id=entity_id,
        replacement=completed,
    ) == completed


def test_completed_with_failures_accepts_ordinary_child_failure(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    replacement = copy.deepcopy(document)
    key = replacement["execution_key"]
    replacement["trials"][0]["training"] = {
        "disposition": "failed",
        "result": f"demo/run-{key}-trial-0001-train",
        "reason": None,
    }
    replacement["trials"][0]["evaluations"][0].update(
        disposition="skipped", result=None, reason="upstream_failed"
    )
    replacement["status"] = "completed_with_failures"
    assert writer.replace_nonterminal_study_result(
        entity_id=entity_id,
        replacement=replacement,
    ) == replacement


def test_completed_with_failures_rejects_global_failure_skip(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    invalid = copy.deepcopy(document)
    key = invalid["execution_key"]
    invalid["trials"][0]["training"] = {
        "disposition": "completed",
        "result": f"demo/run-{key}-trial-0001-train",
        "reason": None,
    }
    invalid["trials"][0]["evaluations"][0].update(
        disposition="skipped", result=None, reason="global_failure"
    )
    invalid["status"] = "completed_with_failures"
    with pytest.raises(ValueError, match="global failure"):
        writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=invalid)


def test_failed_accepts_global_failure_closure_with_diagnostic(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    failed = copy.deepcopy(document)
    key = failed["execution_key"]
    failed["trials"][0]["training"] = {
        "disposition": "completed",
        "result": f"demo/run-{key}-trial-0001-train",
        "reason": None,
    }
    failed["trials"][0]["evaluations"][0].update(
        disposition="skipped", result=None, reason="global_failure"
    )
    failed["status"] = "failed"
    failed["diagnostic"] = {"code": "progression_failed", "message": "cannot progress"}
    assert writer.replace_nonterminal_study_result(
        entity_id=entity_id,
        replacement=failed,
    ) == failed


def test_failed_accepts_global_failure_only_closure_with_diagnostic(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    failed = copy.deepcopy(document)
    failed["trials"][0]["training"] = {
        "disposition": "skipped",
        "result": None,
        "reason": "global_failure",
    }
    failed["trials"][0]["evaluations"][0].update(
        disposition="skipped", result=None, reason="global_failure"
    )
    failed["status"] = "failed"
    failed["diagnostic"] = {"code": "progression_failed", "message": "cannot progress"}
    assert writer.replace_nonterminal_study_result(
        entity_id=entity_id,
        replacement=failed,
    ) == failed


def test_failed_rejects_study_cancelled_even_with_global_failure(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    invalid = copy.deepcopy(document)
    invalid["trials"][0]["training"] = {
        "disposition": "skipped",
        "result": None,
        "reason": "global_failure",
    }
    invalid["trials"][0]["evaluations"][0].update(
        disposition="skipped", result=None, reason="study_cancelled"
    )
    invalid["status"] = "failed"
    invalid["diagnostic"] = {"code": "progression_failed", "message": "cannot progress"}
    with pytest.raises(ValueError, match="Study cancellation"):
        writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=invalid)


def test_failed_accepts_ordinary_child_failure_with_global_failure(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    failed = copy.deepcopy(document)
    key = failed["execution_key"]
    failed["trials"][0]["training"] = {
        "disposition": "failed",
        "result": f"demo/run-{key}-trial-0001-train",
        "reason": None,
    }
    failed["trials"][0]["evaluations"][0].update(
        disposition="skipped", result=None, reason="global_failure"
    )
    failed["status"] = "failed"
    failed["diagnostic"] = {"code": "progression_failed", "message": "cannot progress"}
    assert writer.replace_nonterminal_study_result(
        entity_id=entity_id,
        replacement=failed,
    ) == failed


def test_cancelled_accepts_all_child_cancelled_without_study_cancelled_skip(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    cancelling = copy.deepcopy(document)
    cancelling["status"] = "cancelling"
    writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=cancelling)

    cancelled = copy.deepcopy(cancelling)
    key = cancelled["execution_key"]
    cancelled["trials"][0]["training"] = {
        "disposition": "cancelled",
        "result": f"demo/run-{key}-trial-0001-train",
        "reason": None,
    }
    cancelled["trials"][0]["evaluations"][0].update(
        disposition="cancelled",
        result=f"demo/run-{key}-trial-0001-eval-0001",
        reason=None,
    )
    cancelled["status"] = "cancelled"
    assert writer.replace_nonterminal_study_result(
        entity_id=entity_id,
        replacement=cancelled,
    ) == cancelled


def test_cancelled_accepts_study_cancelled_closure(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)

    cancelling = copy.deepcopy(document)
    cancelling["status"] = "cancelling"
    writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=cancelling)

    cancelled = copy.deepcopy(cancelling)
    key = cancelled["execution_key"]
    cancelled["trials"][0]["training"] = {
        "disposition": "cancelled",
        "result": f"demo/run-{key}-trial-0001-train",
        "reason": None,
    }
    cancelled["trials"][0]["evaluations"][0].update(
        disposition="skipped", result=None, reason="study_cancelled"
    )
    cancelled["status"] = "cancelled"
    assert writer.replace_nonterminal_study_result(
        entity_id=entity_id,
        replacement=cancelled,
    ) == cancelled


def test_existing_model_training_null_is_not_counted_as_pending_stage(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    document["trials"][0]["training"] = None
    entity_id = StudyResultId(document["id"])
    assert writer.create_study_result(entity_id=entity_id, document=document) == document

    completed = copy.deepcopy(document)
    key = completed["execution_key"]
    completed["trials"][0]["evaluations"][0].update(
        disposition="completed",
        result=f"demo/run-{key}-trial-0001-eval-0001",
        reason=None,
    )
    completed["status"] = "completed"
    assert writer.replace_nonterminal_study_result(
        entity_id=entity_id,
        replacement=completed,
    ) == completed


@pytest.mark.parametrize("status", ["submitted", "cancelling"])
def test_nonterminal_study_result_rejects_zero_stage_topology(tmp_path, status) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    document["status"] = status
    document["trials"] = []
    entity_id = StudyResultId(document["id"])

    with pytest.raises(ValueError, match="pending planned stage"):
        writer.create_study_result(entity_id=entity_id, document=document)


def test_submitted_existing_model_trial_rejects_zero_evaluation_topology(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    document["trials"][0]["training"] = None
    document["trials"][0]["evaluations"] = []
    entity_id = StudyResultId(document["id"])

    with pytest.raises(ValueError, match="pending planned stage"):
        writer.create_study_result(entity_id=entity_id, document=document)


@pytest.mark.parametrize(
    "created_at",
    [
        "2026-09-09T09:00:00Z",
        "2026-09-09T09:00:00.1Z",
        "2026-09-09T09:00:00.123456789Z",
    ],
)
def test_study_result_accepts_rfc3339_utc_z_created_at(tmp_path, created_at) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    document["created_at"] = created_at
    entity_id = StudyResultId(document["id"])

    assert writer.create_study_result(entity_id=entity_id, document=document) == document


@pytest.mark.parametrize(
    "created_at",
    [
        "2026-W37-3T09:00:00Z",
        "2026-252T09:00:00Z",
        "2026-09-09T09Z",
        "2026-09-09T09:00Z",
        "2026-09-09 09:00:00Z",
        "2026-09-09T18:00:00+09:00",
        "2026-09-09T09:00:00",
        "2026-09-09T09:00:00z",
        "2026-02-30T09:00:00Z",
    ],
)
def test_study_result_rejects_non_rfc3339_utc_z_created_at(tmp_path, created_at) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    document["created_at"] = created_at
    entity_id = StudyResultId(document["id"])

    with pytest.raises(ValueError, match="RFC3339 UTC using Z"):
        writer.create_study_result(entity_id=entity_id, document=document)


def test_study_result_accepts_matching_study_and_plan_namespaces(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])

    assert writer.create_study_result(entity_id=entity_id, document=document) == document


@pytest.mark.parametrize("field", ["study", "plan"])
def test_study_result_rejects_cross_namespace_study_or_plan(tmp_path, field) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    document[field] = f"other/example-{field}"
    entity_id = StudyResultId(document["id"])

    with pytest.raises(ValueError, match="namespaces must match"):
        writer.create_study_result(entity_id=entity_id, document=document)


def test_canonical_write_accepts_valid_namespace_metadata(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _immutable()

    assert writer.create_immutable(
        kind=EntityKind.STUDY_PLAN,
        entity_id=StudyPlanId(document["id"]),
        document=document,
    ) == document


def test_immutable_create_rejects_namespace_id_directory_mismatch_before_target_create(
    tmp_path,
) -> None:
    repo = _repo(tmp_path)
    namespace_path = repo / "mldb_data" / "demo" / "namespace.yaml"
    namespace_path.write_text(
        json.dumps({"schema": "mjtensu.mldb-v2/namespace/v1", "id": "other"}),
        encoding="utf-8",
    )
    unrelated = repo / "mldb_data" / "demo" / "models" / "unrelated.yaml"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"unrelated canonical bytes\n")
    original = unrelated.read_bytes()
    target = repo / "mldb_data" / "demo" / "study_plans" / "example-plan.yaml"

    with pytest.raises(ValueError, match="Namespace id.*does not match directory"):
        _writer(repo).create_immutable(
            kind=EntityKind.STUDY_PLAN,
            entity_id=StudyPlanId("demo/example-plan"),
            document=_immutable(),
        )

    assert not target.exists()
    assert unrelated.read_bytes() == original


def test_study_result_create_rejects_namespace_id_directory_mismatch(tmp_path) -> None:
    repo = _repo(tmp_path)
    (repo / "mldb_data" / "demo" / "namespace.yaml").write_text(
        json.dumps({"schema": "mjtensu.mldb-v2/namespace/v1", "id": "other"}),
        encoding="utf-8",
    )
    document = _study_result()
    target = _study_path(repo, document)

    with pytest.raises(ValueError, match="Namespace id.*does not match directory"):
        _writer(repo).create_study_result(
            entity_id=StudyResultId(document["id"]),
            document=document,
        )
    assert not target.exists()


def test_study_result_replace_revalidates_canonical_namespace_metadata(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)
    path = _study_path(repo, document)
    original = path.read_bytes()
    (repo / "mldb_data" / "demo" / "namespace.yaml").write_text(
        json.dumps({"schema": "mjtensu.mldb-v2/namespace/v1", "id": "other"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Namespace id.*does not match directory"):
        writer.replace_nonterminal_study_result(
            entity_id=entity_id,
            replacement=_complete(document),
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "namespace_text",
    [
        json.dumps({"schema": "mjtensu.mldb-v2/task/v1", "id": "demo"}),
        "schema: [unterminated\nid: demo\n",
        json.dumps({"schema": "mjtensu.mldb-v2/namespace/v1", "id": "Demo"}),
    ],
)
def test_canonical_write_rejects_invalid_namespace_metadata(tmp_path, namespace_text) -> None:
    repo = _repo(tmp_path)
    (repo / "mldb_data" / "demo" / "namespace.yaml").write_text(
        namespace_text,
        encoding="utf-8",
    )
    target = repo / "mldb_data" / "demo" / "study_plans" / "example-plan.yaml"

    with pytest.raises(ValueError):
        _writer(repo).create_immutable(
            kind=EntityKind.STUDY_PLAN,
            entity_id=StudyPlanId("demo/example-plan"),
            document=_immutable(),
        )
    assert not target.exists()


def test_canonical_write_rejects_missing_namespace_metadata(tmp_path) -> None:
    repo = _repo(tmp_path)
    (repo / "mldb_data" / "demo" / "namespace.yaml").unlink()
    target = repo / "mldb_data" / "demo" / "study_plans" / "example-plan.yaml"

    with pytest.raises(ValueError, match="namespace is not canonical"):
        _writer(repo).create_immutable(
            kind=EntityKind.STUDY_PLAN,
            entity_id=StudyPlanId("demo/example-plan"),
            document=_immutable(),
        )
    assert not target.exists()


def test_study_result_accepts_first_trial_and_evaluation_sequence_ids(tmp_path) -> None:
    repo = _repo(tmp_path)
    document = _study_result()

    assert _writer(repo).create_study_result(
        entity_id=StudyResultId(document["id"]),
        document=document,
    ) == document


@pytest.mark.parametrize("trial_id", ["trial-0000", "trial-1", "trial-00001", "Trial-0001"])
def test_study_result_rejects_invalid_actual_trial_id_grammar(tmp_path, trial_id) -> None:
    repo = _repo(tmp_path)
    document = _study_result()
    document["trials"][0]["trial"] = trial_id

    with pytest.raises(ValueError, match="trial id"):
        _writer(repo).create_study_result(
            entity_id=StudyResultId(document["id"]),
            document=document,
        )


@pytest.mark.parametrize("coordinate", ["eval-0000", "eval-1", "eval-00001", "Eval-0001"])
def test_study_result_rejects_invalid_actual_evaluation_id_grammar(
    tmp_path, coordinate
) -> None:
    repo = _repo(tmp_path)
    document = _study_result()
    document["trials"][0]["evaluations"][0]["coordinate"] = coordinate

    with pytest.raises(ValueError, match="evaluation coordinate id"):
        _writer(repo).create_study_result(
            entity_id=StudyResultId(document["id"]),
            document=document,
        )


def test_study_result_rejects_trial_10000_after_contiguous_prefix() -> None:
    document = _study_result()
    document["status"] = "completed"
    document["trials"] = [
        {"trial": f"trial-{index:04d}", "training": None, "evaluations": []}
        for index in range(1, 10001)
    ]

    with pytest.raises(ValueError, match="trial id"):
        writes._validate_study_result_record(
            document,
            entity_id=StudyResultId(document["id"]),
        )


def test_study_result_rejects_eval_10000_after_contiguous_prefix() -> None:
    document = _study_result()
    document["status"] = "completed"
    document["trials"][0]["training"] = None
    key = document["execution_key"]
    document["trials"][0]["evaluations"] = [
        {
            "coordinate": f"eval-{index:04d}",
            "stage": "holdout",
            "disposition": "completed",
            "result": f"demo/run-{key}-trial-0001-eval-{index:04d}",
            "reason": None,
        }
        for index in range(1, 10001)
    ]

    with pytest.raises(ValueError, match="evaluation coordinate id"):
        writes._validate_study_result_record(
            document,
            entity_id=StudyResultId(document["id"]),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("trial", "trial-0002", "trial IDs must be contiguous and ordered"),
        ("coordinate", "eval-0002", "evaluation coordinates must be contiguous and ordered"),
    ],
)
def test_study_result_preserves_contiguous_authored_sequence_order(
    tmp_path, field, value, message
) -> None:
    repo = _repo(tmp_path)
    document = _study_result()
    if field == "trial":
        document["trials"][0]["trial"] = value
    else:
        document["trials"][0]["evaluations"][0]["coordinate"] = value

    with pytest.raises(ValueError, match=message):
        _writer(repo).create_study_result(
            entity_id=StudyResultId(document["id"]),
            document=document,
        )


def test_study_result_preserves_deterministic_child_result_identity(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    document = _study_result()
    entity_id = StudyResultId(document["id"])
    writer.create_study_result(entity_id=entity_id, document=document)
    invalid = _complete(document)
    invalid["trials"][0]["evaluations"][0]["result"] = (
        f"demo/run-{document['execution_key']}-trial-0001-eval-0002"
    )

    with pytest.raises(ValueError, match="deterministic result"):
        writer.replace_nonterminal_study_result(
            entity_id=entity_id,
            replacement=invalid,
        )