from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mldb.src.api import controller
from mldb.src.common.errors import (
    InvalidRequestError,
    UnsupportedOperationError,
    ValidationIssue,
)
from mldb.src.common.ids import EntityKind, StudyId, StudyRunId, TaskId
from mldb.src.orchestration.study_launch import (
    StudyExecutionSetupError as LowerStudyExecutionSetupError,
)
from mldb.src.study.run import StudyRunStatus


_VALIDATION_CASES = (
    (EntityKind.TASK, "TaskId", "_validate_task_definition"),
    (EntityKind.CORPUS, "CorpusId", "_validate_corpus_definition"),
    (EntityKind.ARCHITECTURE, "ArchitectureId", "_validate_architecture_definition"),
    (EntityKind.TRAIN_PROTOCOL, "TrainProtocolId", "_validate_train_protocol_definition"),
    (
        EntityKind.EVALUATION_PROTOCOL,
        "EvaluationProtocolId",
        "_validate_evaluation_protocol_definition",
    ),
    (EntityKind.STUDY, "StudyId", "_validate_study_definition"),
)
_VALIDATION_LOWERS = tuple(case[2] for case in _VALIDATION_CASES)
_UNSUPPORTED_VALIDATION_KINDS = (
    EntityKind.TRAINING_RUN,
    EntityKind.MODEL,
    EntityKind.EVALUATION_RUN,
    EntityKind.STUDY_RUN,
)


@pytest.mark.parametrize("kind,converter_name,lower_name", _VALIDATION_CASES)
def test_validate_definition_exact_dispatch_and_projection(
    monkeypatch, kind, converter_name, lower_name
):
    layout = object()
    filesystem = object()
    typed_id = object()
    issues = (
        ValidationIssue("first", "first message", "a"),
        ValidationIssue("second", "second message", "b"),
    )
    converter = Mock(return_value=typed_id)
    monkeypatch.setattr(controller, converter_name, converter)

    lowers = {name: Mock() for name in _VALIDATION_LOWERS}
    for name, lower in lowers.items():
        monkeypatch.setattr(controller, name, lower)
    lowers[lower_name].return_value = SimpleNamespace(
        kind=kind,
        id=typed_id,
        valid=False,
        issues=issues,
    )

    response = controller.validate_definition(kind, "entity", layout, filesystem)

    converter.assert_called_once_with("entity")
    lowers[lower_name].assert_called_once_with(typed_id, layout, filesystem)
    for name, lower in lowers.items():
        if name != lower_name:
            lower.assert_not_called()
    assert response.kind is kind
    assert response.id is typed_id
    assert response.valid is False
    assert response.issues is issues
    assert response.issues == issues


@pytest.mark.parametrize("kind", _UNSUPPORTED_VALIDATION_KINDS)
def test_validate_definition_rejects_unsupported_before_lower(monkeypatch, kind):
    lowers = []
    for name in _VALIDATION_LOWERS:
        lower = Mock()
        lowers.append(lower)
        monkeypatch.setattr(controller, name, lower)

    with pytest.raises(UnsupportedOperationError):
        controller.validate_definition(kind, "entity", object(), object())

    assert all(not lower.called for lower in lowers)


def test_validate_definition_lower_exception_propagates_same_object(monkeypatch):
    failure = RuntimeError("lower failure")
    lower = Mock(side_effect=failure)
    monkeypatch.setattr(controller, "_validate_task_definition", lower)

    with pytest.raises(RuntimeError) as caught:
        controller.validate_definition(EntityKind.TASK, "task", object(), object())

    assert caught.value is failure


_SEAL_EXECUTABLE_CASES = (
    (EntityKind.ARCHITECTURE, "ArchitectureId", "_seal_architecture"),
    (EntityKind.TRAIN_PROTOCOL, "TrainProtocolId", "_seal_train_protocol"),
    (
        EntityKind.EVALUATION_PROTOCOL,
        "EvaluationProtocolId",
        "_seal_evaluation_protocol",
    ),
)
_SEAL_LOWERS = (
    "_seal_architecture",
    "_seal_train_protocol",
    "_seal_evaluation_protocol",
    "_seal_study",
)
_UNSUPPORTED_SEAL_KINDS = (
    EntityKind.TASK,
    EntityKind.CORPUS,
    EntityKind.TRAINING_RUN,
    EntityKind.MODEL,
    EntityKind.EVALUATION_RUN,
    EntityKind.STUDY_RUN,
)


@pytest.mark.parametrize("kind,converter_name,lower_name", _SEAL_EXECUTABLE_CASES)
def test_seal_definition_executable_exact_dispatch_with_runner(
    monkeypatch, kind, converter_name, lower_name
):
    layout = object()
    filesystem = object()
    runner = object()
    typed_id = object()
    converter = Mock(return_value=typed_id)
    monkeypatch.setattr(controller, converter_name, converter)

    lowers = {name: Mock() for name in _SEAL_LOWERS}
    for name, lower in lowers.items():
        monkeypatch.setattr(controller, name, lower)
    lowers[lower_name].return_value = "already_sealed"

    response = controller.seal_definition(
        kind, "entity", layout, filesystem, runner=runner
    )

    converter.assert_called_once_with("entity")
    lowers[lower_name].assert_called_once_with(typed_id, layout, filesystem, runner)
    for name, lower in lowers.items():
        if name != lower_name:
            lower.assert_not_called()
    assert response.kind is kind
    assert response.id is typed_id
    assert response.status == "sealed"
    assert response.result == "already_sealed"


def test_seal_definition_study_exact_dispatch_without_runner(monkeypatch):
    layout = object()
    filesystem = object()
    typed_id = object()
    converter = Mock(return_value=typed_id)
    lower = Mock(return_value="sealed")
    monkeypatch.setattr(controller, "StudyId", converter)
    monkeypatch.setattr(controller, "_seal_study", lower)

    response = controller.seal_definition(
        EntityKind.STUDY, "study", layout, filesystem
    )

    converter.assert_called_once_with("study")
    lower.assert_called_once_with(typed_id, layout, filesystem)
    assert response.kind is EntityKind.STUDY
    assert response.id is typed_id
    assert response.status == "sealed"
    assert response.result == "sealed"


@pytest.mark.parametrize("kind,_,lower_name", _SEAL_EXECUTABLE_CASES)
def test_seal_definition_requires_runner_before_mutation(monkeypatch, kind, _, lower_name):
    lower = Mock()
    monkeypatch.setattr(controller, lower_name, lower)

    with pytest.raises(InvalidRequestError):
        controller.seal_definition(kind, "entity", object(), object(), runner=None)

    lower.assert_not_called()


def test_seal_definition_rejects_runner_for_study_before_mutation(monkeypatch):
    lower = Mock()
    monkeypatch.setattr(controller, "_seal_study", lower)

    with pytest.raises(InvalidRequestError):
        controller.seal_definition(
            EntityKind.STUDY, "study", object(), object(), runner=object()
        )

    lower.assert_not_called()


@pytest.mark.parametrize("kind", _UNSUPPORTED_SEAL_KINDS)
def test_seal_definition_rejects_unsupported_before_mutation(monkeypatch, kind):
    lowers = []
    for name in _SEAL_LOWERS:
        lower = Mock()
        lowers.append(lower)
        monkeypatch.setattr(controller, name, lower)

    with pytest.raises(UnsupportedOperationError):
        controller.seal_definition(kind, "entity", object(), object(), runner=object())

    assert all(not lower.called for lower in lowers)


def test_seal_definition_lower_exception_propagates_and_no_duplicate_validation(monkeypatch):
    failure = RuntimeError("seal lower failure")
    lower = Mock(side_effect=failure)
    monkeypatch.setattr(controller, "_seal_architecture", lower)
    validation_spies = []
    for name in _VALIDATION_LOWERS:
        spy = Mock()
        validation_spies.append(spy)
        monkeypatch.setattr(controller, name, spy)

    with pytest.raises(RuntimeError) as caught:
        controller.seal_definition(
            EntityKind.ARCHITECTURE,
            "architecture",
            object(),
            object(),
            runner=object(),
        )

    assert caught.value is failure
    assert all(not spy.called for spy in validation_spies)


def _execute_args():
    return (
        StudyId("study"),
        date(2026, 9, 8),
        object(),
        Mock(name="layout"),
        Mock(name="filesystem"),
        Mock(name="queue"),
        "2026-09-08T09:00:00.000000Z",
        object(),
    )


def test_execute_study_exact_order_identity_and_projection(monkeypatch):
    (
        study_id,
        allocation_date,
        started_at,
        layout,
        filesystem,
        queue,
        admitted_at,
        failed_at,
    ) = _execute_args()
    study = object()
    prepared = object()
    run = SimpleNamespace(
        id=StudyRunId("sr-20260908-001"),
        study=StudyId("canonical-study"),
        status=StudyRunStatus.RUNNING,
    )
    returned = object()
    calls = []

    def resolve(*args):
        calls.append(("resolve", args))
        return study

    def preflight(*args):
        calls.append(("preflight", args))
        return prepared

    def launch(*args, **kwargs):
        calls.append(("launch", args, kwargs))
        return run

    def response(**kwargs):
        calls.append(("response", kwargs))
        return returned

    monkeypatch.setattr(controller, "_resolve_study", resolve)
    monkeypatch.setattr(controller, "_preflight_study_execution", preflight)
    monkeypatch.setattr(controller, "_launch_study_execution", launch)
    monkeypatch.setattr(controller, "StudyExecutionResponse", response)

    result = controller.execute_study(
        study_id,
        allocation_date,
        started_at,
        layout,
        filesystem,
        queue,
        admitted_at=admitted_at,
        failed_at=failed_at,
    )

    assert result is returned
    assert [call[0] for call in calls] == ["resolve", "preflight", "launch", "response"]
    assert calls[0][1] == (study_id, layout, filesystem)
    assert calls[1][1] == (study, layout, filesystem)
    assert calls[2][1] == (
        prepared,
        allocation_date,
        started_at,
        layout,
        filesystem,
        queue,
    )
    assert calls[2][2] == {"admitted_at": admitted_at, "failed_at": failed_at}
    assert calls[3][1] == {
        "study_run_id": run.id,
        "study_id": run.study,
        "status": run.status,
    }
    assert layout.mock_calls == []
    assert filesystem.mock_calls == []
    assert queue.mock_calls == []


def test_execute_study_resolve_failure_stops_composition(monkeypatch):
    failure = RuntimeError("resolve failure")
    preflight = Mock()
    launch = Mock()
    monkeypatch.setattr(controller, "_resolve_study", Mock(side_effect=failure))
    monkeypatch.setattr(controller, "_preflight_study_execution", preflight)
    monkeypatch.setattr(controller, "_launch_study_execution", launch)
    args = _execute_args()

    with pytest.raises(RuntimeError) as caught:
        controller.execute_study(
            *args[:6], admitted_at=args[6], failed_at=args[7]
        )

    assert caught.value is failure
    preflight.assert_not_called()
    launch.assert_not_called()


def test_execute_study_preflight_failure_stops_before_launch(monkeypatch):
    failure = RuntimeError("preflight failure")
    launch = Mock()
    monkeypatch.setattr(controller, "_resolve_study", Mock(return_value=object()))
    monkeypatch.setattr(controller, "_preflight_study_execution", Mock(side_effect=failure))
    monkeypatch.setattr(controller, "_launch_study_execution", launch)
    args = _execute_args()

    with pytest.raises(RuntimeError) as caught:
        controller.execute_study(
            *args[:6], admitted_at=args[6], failed_at=args[7]
        )

    assert caught.value is failure
    launch.assert_not_called()


def test_execute_study_launch_preallocation_failure_propagates_unchanged(monkeypatch):
    failure = RuntimeError("allocation failed")
    monkeypatch.setattr(controller, "_resolve_study", Mock(return_value=object()))
    monkeypatch.setattr(controller, "_preflight_study_execution", Mock(return_value=object()))
    monkeypatch.setattr(controller, "_launch_study_execution", Mock(side_effect=failure))
    args = _execute_args()

    with pytest.raises(RuntimeError) as caught:
        controller.execute_study(
            *args[:6], admitted_at=args[6], failed_at=args[7]
        )

    assert caught.value is failure


def test_execute_study_setup_error_same_object_class_and_id(monkeypatch):
    run_id = StudyRunId("sr-20260908-777")
    failure = LowerStudyExecutionSetupError(run_id)
    monkeypatch.setattr(controller, "_resolve_study", Mock(return_value=object()))
    monkeypatch.setattr(controller, "_preflight_study_execution", Mock(return_value=object()))
    monkeypatch.setattr(controller, "_launch_study_execution", Mock(side_effect=failure))
    args = _execute_args()

    with pytest.raises(LowerStudyExecutionSetupError) as caught:
        controller.execute_study(
            *args[:6], admitted_at=args[6], failed_at=args[7]
        )

    assert caught.value is failure
    assert caught.value.study_run_id == run_id
    assert controller.StudyExecutionSetupError is LowerStudyExecutionSetupError


def test_execute_study_success_returns_exact_run_projection(monkeypatch):
    run = SimpleNamespace(
        id=StudyRunId("sr-20260908-002"),
        study=StudyId("study"),
        status=StudyRunStatus.RUNNING,
    )
    monkeypatch.setattr(controller, "_resolve_study", Mock(return_value=object()))
    monkeypatch.setattr(controller, "_preflight_study_execution", Mock(return_value=object()))
    monkeypatch.setattr(controller, "_launch_study_execution", Mock(return_value=run))
    args = _execute_args()

    response = controller.execute_study(
        *args[:6], admitted_at=args[6], failed_at=args[7]
    )

    assert response == controller.StudyExecutionResponse(run.id, run.study, run.status)


def test_get_study_run_exact_delegate_identity_and_no_direct_mutation(monkeypatch):
    study_run_id = StudyRunId("sr-20260908-003")
    layout = Mock(name="layout")
    filesystem = Mock(name="filesystem")
    queue = Mock(name="queue")
    progress = object()
    lower = Mock(return_value=progress)
    monkeypatch.setattr(controller, "_get_study_run_progress", lower)

    result = controller.get_study_run(
        study_run_id, "as-of", layout, filesystem, queue
    )

    assert result is progress
    lower.assert_called_once_with(study_run_id, "as-of", layout, filesystem, queue)
    assert layout.mock_calls == []
    assert filesystem.mock_calls == []
    assert queue.mock_calls == []


@pytest.mark.parametrize("lower_result", ("accepted", "already_terminal"))
def test_cancel_study_run_exact_delegate_and_result(monkeypatch, lower_result):
    study_run_id = StudyRunId("sr-20260908-004")
    finished_at = object()
    layout = Mock(name="layout")
    filesystem = Mock(name="filesystem")
    queue = Mock(name="queue")
    lower = Mock(return_value=lower_result)
    monkeypatch.setattr(controller, "_request_study_run_cancellation", lower)

    result = controller.cancel_study_run(
        study_run_id, finished_at, "queue-at", layout, filesystem, queue
    )

    assert result == lower_result
    lower.assert_called_once_with(
        study_run_id, finished_at, "queue-at", layout, filesystem, queue
    )
    assert queue.mock_calls == []


def test_cancel_study_run_lower_failure_propagates(monkeypatch):
    failure = RuntimeError("cancel failure")
    monkeypatch.setattr(
        controller, "_request_study_run_cancellation", Mock(side_effect=failure)
    )

    with pytest.raises(RuntimeError) as caught:
        controller.cancel_study_run(
            StudyRunId("sr-20260908-005"), object(), "at", object(), object(), object()
        )

    assert caught.value is failure


@pytest.mark.parametrize("kind", tuple(EntityKind))
def test_get_entity_all_kinds_delegate_unchanged_without_kind_guessing(monkeypatch, kind):
    canonical = object()
    lower = Mock(return_value=canonical)
    monkeypatch.setattr(controller, "_get_entity", lower)
    for converter_name in (
        "TaskId",
        "CorpusId",
        "ArchitectureId",
        "TrainProtocolId",
        "EvaluationProtocolId",
        "StudyId",
        "StudyRunId",
    ):
        monkeypatch.setattr(controller, converter_name, Mock(side_effect=AssertionError))
    layout = object()
    filesystem = object()

    result = controller.get_entity(kind, "exact-id", layout, filesystem)

    assert result is canonical
    lower.assert_called_once_with(kind, "exact-id", layout, filesystem)


def test_get_entity_lower_failure_propagates(monkeypatch):
    failure = RuntimeError("entity failure")
    monkeypatch.setattr(controller, "_get_entity", Mock(side_effect=failure))

    with pytest.raises(RuntimeError) as caught:
        controller.get_entity(EntityKind.MODEL, "model", object(), object())

    assert caught.value is failure


def test_list_entities_exact_delegate_preserves_tuple_identity(monkeypatch):
    summaries = (object(), object())
    lower = Mock(return_value=summaries)
    monkeypatch.setattr(controller, "_list_entities", lower)
    layout = object()
    filesystem = object()

    result = controller.list_entities(EntityKind.STUDY, layout, filesystem)

    assert result is summaries
    lower.assert_called_once_with(EntityKind.STUDY, layout, filesystem)


def test_list_entities_empty_tuple_identity(monkeypatch):
    empty = ()
    monkeypatch.setattr(controller, "_list_entities", Mock(return_value=empty))

    result = controller.list_entities(EntityKind.TASK, object(), object())

    assert result is empty


def test_list_entities_lower_failure_propagates(monkeypatch):
    failure = RuntimeError("list failure")
    monkeypatch.setattr(controller, "_list_entities", Mock(side_effect=failure))

    with pytest.raises(RuntimeError) as caught:
        controller.list_entities(EntityKind.TASK, object(), object())

    assert caught.value is failure


def test_public_all_is_exact_and_setup_error_is_lower_identity():
    assert controller.__all__ == (
        "DefinitionValidationResponse",
        "SealDefinitionResponse",
        "StudyExecutionResponse",
        "StudyExecutionSetupError",
        "validate_definition",
        "seal_definition",
        "execute_study",
        "get_study_run",
        "cancel_study_run",
        "get_entity",
        "list_entities",
    )
    assert controller.StudyExecutionSetupError is LowerStudyExecutionSetupError


def test_public_dtos_are_value_equal_and_frozen():
    issues = (ValidationIssue("code", "message", "path"),)
    validation = controller.DefinitionValidationResponse(
        EntityKind.TASK, TaskId("task"), False, issues
    )
    assert validation == controller.DefinitionValidationResponse(
        EntityKind.TASK, TaskId("task"), False, issues
    )
    with pytest.raises(FrozenInstanceError):
        validation.valid = True

    sealed = controller.SealDefinitionResponse(
        EntityKind.STUDY, StudyId("study"), "sealed", "already_sealed"
    )
    assert sealed == controller.SealDefinitionResponse(
        EntityKind.STUDY, StudyId("study"), "sealed", "already_sealed"
    )
    with pytest.raises(FrozenInstanceError):
        sealed.result = "sealed"

    execution = controller.StudyExecutionResponse(
        StudyRunId("sr-20260908-006"), StudyId("study"), StudyRunStatus.RUNNING
    )
    assert execution == controller.StudyExecutionResponse(
        StudyRunId("sr-20260908-006"), StudyId("study"), StudyRunStatus.RUNNING
    )
    with pytest.raises(FrozenInstanceError):
        execution.status = StudyRunStatus.FAILED
