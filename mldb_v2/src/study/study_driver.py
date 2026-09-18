"""One-pass backend-neutral Study progression and canonical reconciliation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypedDict, cast

from mldb_v2.src.backend.backend_port import BackendPort
from mldb_v2.src.backend.candidate_outcome import BackendObservation, StageKey, TerminalCandidate
from mldb_v2.src.common.diagnostic import Diagnostic
from mldb_v2.src.common.ids import (
    EntityKind,
    EvaluationResultId,
    ModelId,
    StudyResultId,
    TrainingResultId,
    _canonical_json_bytes,
    _validate_typed_reference,
)
from mldb_v2.src.repository.canonical_writes import CanonicalRepositoryWriter
from mldb_v2.src.repository.mutation_coordination import StudyResultMutationCoordinator
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.results.study_result import StudyResult, StudyResultStatus, _validate_study_result
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study._plan_build import _validate_study_plan
from mldb_v2.src.study.execution_readiness import (
    ExecutionReadinessResolver,
    PlannedStageRef,
    ReadinessSkip,
    _build_stage_input,
    _materialize_stage_input,
    _validate_result_plan_lineage,
)
from mldb_v2.src.study.plan import StudyPlan
from mldb_v2.src.verification.result_acceptance import (
    AcceptedResultRecordValidator,
    EvaluationAcceptanceRequest,
    ResultAcceptor,
    TrainingAcceptanceRequest,
)

FinalizedResultId: TypeAlias = TrainingResultId | EvaluationResultId


class StudyAdvanceResult(TypedDict):
    study_result: StudyResultId
    status: StudyResultStatus
    changed: bool
    admitted: list[StageKey]
    finalized_results: list[FinalizedResultId]
    finalized_models: list[ModelId]
    active: list[StageKey]
    terminal: bool


_TERMINAL_STUDY_STATUSES = {
    "completed",
    "completed_with_failures",
    "failed",
    "cancelled",
}
_BACKEND_TRANSIENT_ERRORS = (ConnectionError, TimeoutError, OSError)
_GLOBAL_FAILURE_DIAGNOSTIC: Diagnostic = {
    "code": "study_progression_failed",
    "message": "Study progression cannot continue from canonical inputs.",
}


class _UnrecoverableProgressionError(ValueError):
    pass


@dataclass(frozen=True)
class _PreparedTerminal:
    stage: PlannedStageRef
    candidate: TerminalCandidate
    accepted: Mapping[str, object]


@dataclass(frozen=True)
class _ObservationState:
    observation: BackendObservation | None
    unavailable: bool


def _stage_token(stage: PlannedStageRef) -> tuple[str, str, str | None]:
    return (
        str(stage["trial"]),
        str(stage["kind"]),
        None if stage["kind"] == "training" else str(stage["coordinate"]),
    )


def _stage_key(*, plan: StudyPlan, result: StudyResult, stage: PlannedStageRef) -> StageKey:
    if stage["kind"] == "training":
        return cast(
            StageKey,
            {
                "study_result": result["id"],
                "plan": plan["id"],
                "trial": stage["trial"],
                "kind": "training",
                "coordinate": None,
                "source_commit": plan["source_commit"],
            },
        )
    return cast(
        StageKey,
        {
            "study_result": result["id"],
            "plan": plan["id"],
            "trial": stage["trial"],
            "kind": "evaluation",
            "coordinate": stage["coordinate"],
            "source_commit": plan["source_commit"],
        },
    )


def _pending_stages(plan: StudyPlan, result: StudyResult) -> list[PlannedStageRef]:
    stages: list[PlannedStageRef] = []
    for plan_trial, result_trial in zip(plan["trials"], result["trials"]):
        trial_id = plan_trial["trial"]
        training = result_trial["training"]
        if training is not None and training["disposition"] == "pending":
            stages.append({"kind": "training", "trial": trial_id})
        for plan_coordinate, result_slot in zip(
            plan_trial["evaluations"], result_trial["evaluations"]
        ):
            if result_slot["disposition"] == "pending":
                stages.append(
                    {
                        "kind": "evaluation",
                        "trial": trial_id,
                        "coordinate": plan_coordinate["coordinate"],
                    }
                )
    return stages


def _all_slots(result: StudyResult) -> list[dict[str, object]]:
    slots: list[dict[str, object]] = []
    for trial in result["trials"]:
        if trial["training"] is not None:
            slots.append(cast(dict[str, object], trial["training"]))
        slots.extend(cast(list[dict[str, object]], trial["evaluations"]))
    return slots


def _terminal(result: StudyResult) -> bool:
    return result["status"] in _TERMINAL_STUDY_STATUSES


def _load_context(
    resolver: CanonicalRepositoryResolver,
    *,
    study_result_id: StudyResultId,
) -> tuple[StudyResult, StudyPlan]:
    result = _validate_study_result(
        resolver.resolve(kind=EntityKind.STUDY_RESULT, entity_id=study_result_id)
    )
    plan = _validate_study_plan(
        resolver.resolve(kind=EntityKind.STUDY_PLAN, entity_id=result["plan"])
    )
    _validate_result_plan_lineage(result, plan)
    return result, plan


def _ensure_backend_study_execution(
    *,
    backend: BackendPort,
    plan: StudyPlan,
    result: StudyResult,
) -> None:
    ensure = getattr(backend, "ensure_study_execution", None)
    if not callable(ensure):
        return
    observation = ensure(plan=plan, study_result=result)
    if type(observation) is not dict:
        raise _UnrecoverableProgressionError(
            "backend Study execution observation must be a mapping"
        )
    expected_key = {
        "study_result": result["id"],
        "plan": plan["id"],
        "source_commit": plan["source_commit"],
    }
    if observation.get("key") != expected_key:
        raise _UnrecoverableProgressionError(
            "backend Study execution identity does not match canonical StudyResult"
        )
    if observation.get("state") not in {"active", "terminal"}:
        raise _UnrecoverableProgressionError(
            "backend Study execution observation has invalid state"
        )


def _local_reference_name(value: object) -> str:
    text = str(value)
    return text.split("/", 1)[1] if "/" in text else text


def _trial_projection_context(
    *,
    resolver: CanonicalRepositoryResolver,
    plan_trial: Mapping[str, object] | None,
    result_trial: Mapping[str, object],
) -> tuple[str, str | None, str | None]:
    trial_id = str(result_trial["trial"])
    architecture_id: str | None = None
    model_id: str | None = None
    source = plan_trial.get("source") if isinstance(plan_trial, Mapping) else None
    if isinstance(source, Mapping) and source.get("kind") == "training":
        raw_architecture = source.get("architecture")
        if type(raw_architecture) is str:
            architecture_id = raw_architecture
        training = result_trial.get("training")
        if isinstance(training, Mapping) and type(training.get("result")) is str:
            try:
                training_result = resolver.resolve(
                    kind=EntityKind.TRAINING_RESULT,
                    entity_id=training["result"],
                )
            except (FileNotFoundError, ValueError):
                training_result = None
            if isinstance(training_result, Mapping):
                payload = training_result.get("result")
                if isinstance(payload, Mapping) and type(payload.get("model")) is str:
                    model_id = str(payload["model"])
    elif isinstance(source, Mapping) and source.get("kind") == "existing_model":
        raw_model = source.get("model")
        if type(raw_model) is str:
            model_id = raw_model
            try:
                model = resolver.resolve(kind=EntityKind.MODEL, entity_id=model_id)
            except (FileNotFoundError, ValueError):
                model = None
            if isinstance(model, Mapping) and type(model.get("training_result")) is str:
                try:
                    training_result = resolver.resolve(
                        kind=EntityKind.TRAINING_RESULT,
                        entity_id=model["training_result"],
                    )
                except (FileNotFoundError, ValueError):
                    training_result = None
                if isinstance(training_result, Mapping) and type(training_result.get("architecture")) is str:
                    architecture_id = str(training_result["architecture"])

    label = trial_id
    if architecture_id is not None:
        try:
            architecture = resolver.resolve(
                kind=EntityKind.ARCHITECTURE,
                entity_id=architecture_id,
            )
        except (FileNotFoundError, ValueError):
            architecture = None
        if isinstance(architecture, Mapping) and type(architecture.get("name")) is str:
            label = str(architecture["name"])
        else:
            label = _local_reference_name(architecture_id)
    elif model_id is not None:
        label = _local_reference_name(model_id)
    return label, architecture_id, model_id


def _evaluation_projection_context(
    *,
    resolver: CanonicalRepositoryResolver,
    plan_trial: Mapping[str, object] | None,
    coordinate: object,
) -> tuple[str | None, str | None, str | None, list[str], dict[str, str]]:
    if not isinstance(plan_trial, Mapping):
        return None, None, None, [], {}
    evaluations = plan_trial.get("evaluations")
    if type(evaluations) is not list:
        return None, None, None, [], {}
    plan_evaluation = next(
        (
            item
            for item in evaluations
            if isinstance(item, Mapping) and item.get("coordinate") == coordinate
        ),
        None,
    )
    if not isinstance(plan_evaluation, Mapping):
        return None, None, None, [], {}
    raw_protocol = plan_evaluation.get("evaluation_protocol")
    if type(raw_protocol) is not str:
        return None, None, None, [], {}

    protocol_id = raw_protocol
    protocol_name = _local_reference_name(protocol_id)
    protocol_description: str | None = None
    metric_names: list[str] = []
    metric_preferences: dict[str, str] = {}
    try:
        protocol = resolver.resolve(
            kind=EntityKind.EVALUATION_PROTOCOL,
            entity_id=protocol_id,
        )
    except (FileNotFoundError, ValueError):
        protocol = None
    if isinstance(protocol, Mapping):
        if type(protocol.get("name")) is str and protocol["name"]:
            protocol_name = str(protocol["name"])
        if type(protocol.get("description")) is str and protocol["description"]:
            protocol_description = str(protocol["description"])
        raw_metrics = protocol.get("metrics")
        if isinstance(raw_metrics, Mapping):
            for metric, declaration in raw_metrics.items():
                if type(metric) is not str:
                    continue
                metric_names.append(metric)
                preference = "neutral"
                if isinstance(declaration, Mapping):
                    raw_preference = declaration.get("preference")
                    if type(raw_preference) is str and raw_preference in {"higher", "lower", "neutral"}:
                        preference = raw_preference
                metric_preferences[metric] = preference
    return protocol_id, protocol_name, protocol_description, metric_names, metric_preferences


def _study_summary_projection(
    *,
    resolver: CanonicalRepositoryResolver,
    result: StudyResult,
) -> dict[str, object]:
    try:
        plan = _validate_study_plan(
            resolver.resolve(kind=EntityKind.STUDY_PLAN, entity_id=result["plan"])
        )
    except (FileNotFoundError, ValueError):
        plan = None
    plan_trials = {
        str(trial["trial"]): trial
        for trial in (plan["trials"] if plan is not None else [])
    }

    contexts: dict[str, tuple[str, str | None, str | None]] = {}
    base_counts: dict[str, int] = {}
    for trial in result["trials"]:
        trial_id = str(trial["trial"])
        context = _trial_projection_context(
            resolver=resolver,
            plan_trial=plan_trials.get(trial_id),
            result_trial=trial,
        )
        contexts[trial_id] = context
        base_counts[context[0]] = base_counts.get(context[0], 0) + 1

    rows: list[dict[str, object]] = []
    comparisons_by_stage: dict[tuple[str, str | None], dict[str, object]] = {}
    for trial in result["trials"]:
        trial_id = str(trial["trial"])
        base_label, architecture_id, model_id = contexts[trial_id]
        trial_label = (
            base_label
            if base_counts[base_label] == 1
            else f"{base_label} · {trial_id}"
        )
        training = trial["training"]
        if training is not None:
            rows.append(
                {
                    "trial": trial_id,
                    "trial_label": trial_label,
                    "architecture": architecture_id,
                    "model": model_id,
                    "kind": "training",
                    "stage": "training",
                    "coordinate": None,
                    "disposition": training["disposition"],
                    "result": training["result"],
                    "metrics": {},
                }
            )
        plan_trial = plan_trials.get(trial_id)
        for evaluation in trial["evaluations"]:
            metrics: dict[str, object] = {}
            result_id = evaluation["result"]
            evaluation_model = model_id
            if evaluation["disposition"] == "completed" and result_id is not None:
                try:
                    document = resolver.resolve(
                        kind=EntityKind.EVALUATION_RESULT,
                        entity_id=result_id,
                    )
                except (FileNotFoundError, ValueError):
                    document = None
                if isinstance(document, Mapping):
                    if type(document.get("model")) is str:
                        evaluation_model = str(document["model"])
                    payload = document.get("result")
                    if isinstance(payload, Mapping) and isinstance(payload.get("metrics"), Mapping):
                        metrics = copy.deepcopy(dict(payload["metrics"]))
            stage = str(evaluation["stage"])
            (
                evaluation_protocol,
                evaluation_name,
                evaluation_description,
                protocol_metric_names,
                protocol_metric_preferences,
            ) = _evaluation_projection_context(
                resolver=resolver,
                plan_trial=plan_trial,
                coordinate=evaluation["coordinate"],
            )
            row = {
                "trial": trial_id,
                "trial_label": trial_label,
                "architecture": architecture_id,
                "model": evaluation_model,
                "kind": "evaluation",
                "stage": stage,
                "coordinate": evaluation["coordinate"],
                "evaluation_protocol": evaluation_protocol,
                "evaluation_name": evaluation_name,
                "evaluation_description": evaluation_description,
                "disposition": evaluation["disposition"],
                "result": result_id,
                "metrics": metrics,
            }
            rows.append(row)

            comparison = comparisons_by_stage.setdefault(
                (stage, evaluation_protocol),
                {
                    "stage": stage,
                    "evaluation_protocol": evaluation_protocol,
                    "evaluation_name": evaluation_name,
                    "evaluation_description": evaluation_description,
                    "metrics": list(protocol_metric_names),
                    "metric_preferences": dict(protocol_metric_preferences),
                    "rows": [],
                },
            )
            metric_names = cast(list[str], comparison["metrics"])
            metric_preferences = cast(dict[str, str], comparison["metric_preferences"])
            for metric in metrics:
                if type(metric) is str and metric not in metric_names:
                    metric_names.append(metric)
                if type(metric) is str:
                    metric_preferences.setdefault(metric, "neutral")
            cast(list[dict[str, object]], comparison["rows"]).append(
                {
                    "trial": trial_id,
                    "trial_label": trial_label,
                    "architecture": architecture_id,
                    "model": evaluation_model,
                    "disposition": evaluation["disposition"],
                    "metrics": copy.deepcopy(metrics),
                }
            )
    return {
        "schema": "mjtensu.mldb-v2/study-summary-projection/v3",
        "study_result": result["id"],
        "study": result["study"],
        "status": result["status"],
        "rows": rows,
        "comparisons": list(comparisons_by_stage.values()),
    }


def _project_backend_summary_best_effort(
    *,
    backend: BackendPort,
    resolver: CanonicalRepositoryResolver,
    result: StudyResult,
) -> None:
    project = getattr(backend, "project_study_summary", None)
    if not callable(project):
        return
    try:
        project(
            study_result=result,
            summary=_study_summary_projection(resolver=resolver, result=result),
        )
    except Exception:
        # Pipeline/UI projection is observational and must never rewrite canonical success.
        return


def _validate_backend_value(
    value: BackendObservation | TerminalCandidate,
    *,
    expected_key: StageKey,
    label: str,
) -> None:
    if type(value) is not dict:
        raise _UnrecoverableProgressionError(f"{label} must be a mapping")
    if value.get("stage_key") != expected_key:
        raise _UnrecoverableProgressionError(
            f"{label} StageKey does not exactly match canonical logical stage"
        )
    if value.get("state") not in {"active", "terminal"}:
        raise _UnrecoverableProgressionError(f"{label} has invalid state")


def _observe_pending(
    *,
    backend: BackendPort,
    plan: StudyPlan,
    result: StudyResult,
) -> tuple[dict[tuple[str, str, str | None], _ObservationState], list[StageKey]]:
    observations: dict[tuple[str, str, str | None], _ObservationState] = {}
    active: list[StageKey] = []
    for stage in _pending_stages(plan, result):
        key = _stage_key(plan=plan, result=result, stage=stage)
        token = _stage_token(stage)
        try:
            observation = backend.observe(stage_key=key)
        except _BACKEND_TRANSIENT_ERRORS:
            observations[token] = _ObservationState(observation=None, unavailable=True)
            continue
        if observation is not None:
            _validate_backend_value(
                observation,
                expected_key=key,
                label="backend observation",
            )
            if observation["state"] == "active":
                active.append(copy.deepcopy(key))
        observations[token] = _ObservationState(observation=observation, unavailable=False)
    return observations, active


def _collect_candidates(
    *,
    backend: BackendPort,
    plan: StudyPlan,
    result: StudyResult,
    observations: Mapping[tuple[str, str, str | None], _ObservationState],
) -> list[tuple[PlannedStageRef, TerminalCandidate]]:
    collected: list[tuple[PlannedStageRef, TerminalCandidate]] = []
    for stage in _pending_stages(plan, result):
        state = observations[_stage_token(stage)]
        observation = state.observation
        if observation is None or observation["state"] != "terminal":
            continue
        key = _stage_key(plan=plan, result=result, stage=stage)
        try:
            candidate = backend.collect(stage_key=key)
        except _BACKEND_TRANSIENT_ERRORS:
            continue
        if candidate is None:
            continue
        _validate_backend_value(candidate, expected_key=key, label="terminal candidate")
        if candidate["state"] != "terminal":
            raise _UnrecoverableProgressionError("collected candidate must be terminal")
        collected.append((stage, candidate))
    return collected


def _prepare_terminal_acceptance(
    *,
    plan: StudyPlan,
    result: StudyResult,
    candidates: list[tuple[PlannedStageRef, TerminalCandidate]],
    acceptor: ResultAcceptor,
    mldb_data_root: Path,
) -> list[_PreparedTerminal]:
    prepared: list[_PreparedTerminal] = []
    for stage, candidate in candidates:
        try:
            stage_input = _materialize_stage_input(
                plan=plan,
                result=result,
                stage=stage,
                mldb_data_root=mldb_data_root,
            )
            if stage["kind"] == "training":
                accepted = acceptor.accept_training(
                    request=cast(
                        TrainingAcceptanceRequest,
                        {
                            "study_result": result,
                            "plan": plan,
                            "stage_input": stage_input,
                            "candidate": candidate,
                        },
                    )
                )
            else:
                accepted = acceptor.accept_evaluation(
                    request=cast(
                        EvaluationAcceptanceRequest,
                        {
                            "study_result": result,
                            "plan": plan,
                            "stage_input": stage_input,
                            "candidate": candidate,
                        },
                    )
                )
        except (ValueError, KeyError, FileNotFoundError) as error:
            raise _UnrecoverableProgressionError(
                "terminal candidate could not be reconciled with canonical lineage"
            ) from error
        prepared.append(_PreparedTerminal(stage=stage, candidate=candidate, accepted=accepted))
    return prepared


def _try_resolve(
    resolver: CanonicalRepositoryResolver,
    *,
    kind: EntityKind,
    entity_id: str,
) -> Mapping[str, object] | None:
    try:
        return resolver.resolve(kind=kind, entity_id=entity_id)
    except FileNotFoundError:
        return None


def _find_result_slot(result: StudyResult, stage: PlannedStageRef) -> dict[str, object]:
    trials = [trial for trial in result["trials"] if trial["trial"] == stage["trial"]]
    if len(trials) != 1:
        raise _UnrecoverableProgressionError("StudyResult trial disappeared during reconciliation")
    trial = trials[0]
    if stage["kind"] == "training":
        if trial["training"] is None:
            raise _UnrecoverableProgressionError("training slot disappeared during reconciliation")
        return cast(dict[str, object], trial["training"])
    slots = [
        slot for slot in trial["evaluations"] if slot["coordinate"] == stage["coordinate"]
    ]
    if len(slots) != 1:
        raise _UnrecoverableProgressionError(
            "evaluation slot disappeared during reconciliation"
        )
    return cast(dict[str, object], slots[0])


def _set_slot_terminal(
    result: StudyResult,
    *,
    stage: PlannedStageRef,
    disposition: str,
    result_id: str,
) -> None:
    slot = _find_result_slot(result, stage)
    if slot["disposition"] != "pending":
        return
    slot["disposition"] = disposition
    slot["result"] = result_id
    slot["reason"] = None


def _set_slot_skipped(result: StudyResult, *, stage: PlannedStageRef, reason: str) -> None:
    slot = _find_result_slot(result, stage)
    if slot["disposition"] != "pending":
        return
    slot["disposition"] = "skipped"
    slot["result"] = None
    slot["reason"] = reason


def _finish_status_if_closed(result: StudyResult) -> None:
    slots = _all_slots(result)
    if any(slot["disposition"] == "pending" for slot in slots):
        return
    if result["status"] == "cancelling":
        result["status"] = "cancelled"
        result["diagnostic"] = None
        return
    if all(slot["disposition"] == "completed" for slot in slots):
        result["status"] = "completed"
        result["diagnostic"] = None
        return
    if any(
        slot["disposition"] == "skipped" and slot["reason"] == "global_failure"
        for slot in slots
    ):
        result["status"] = "failed"
        result["diagnostic"] = copy.deepcopy(_GLOBAL_FAILURE_DIAGNOSTIC)
        return
    result["status"] = "completed_with_failures"
    result["diagnostic"] = None


def _prepared_by_token(
    prepared: list[_PreparedTerminal],
) -> dict[tuple[str, str, str | None], _PreparedTerminal]:
    return {_stage_token(item.stage): item for item in prepared}


def _training_child_id(result: StudyResult, stage: PlannedStageRef) -> TrainingResultId:
    return TrainingResultId(_validate_typed_reference(f"{result['id']}-{stage['trial']}-train"))


def _model_child_id(result: StudyResult, stage: PlannedStageRef) -> ModelId:
    return ModelId(_validate_typed_reference(f"{result['id']}-{stage['trial']}-model"))


def _evaluation_child_id(result: StudyResult, stage: PlannedStageRef) -> EvaluationResultId:
    if stage["kind"] != "evaluation":
        raise ValueError("evaluation child identity requires evaluation stage")
    return EvaluationResultId(
        _validate_typed_reference(
            f"{result['id']}-{stage['trial']}-{stage['coordinate']}"
        )
    )


def _reconcile_terminal_children(
    *,
    resolver: CanonicalRepositoryResolver,
    coordinator: StudyResultMutationCoordinator,
    writer: CanonicalRepositoryWriter,
    validator: AcceptedResultRecordValidator,
    study_result_id: StudyResultId,
    plan: StudyPlan,
    prepared: list[_PreparedTerminal],
) -> tuple[list[FinalizedResultId], list[ModelId]]:
    prepared_map = _prepared_by_token(prepared)
    finalized_results: list[FinalizedResultId] = []
    finalized_models: list[ModelId] = []

    with coordinator.acquire(study_result_id=study_result_id):
        current, current_plan = _load_context(resolver, study_result_id=study_result_id)
        if current_plan["id"] != plan["id"]:
            raise _UnrecoverableProgressionError("StudyResult Plan changed during reconciliation")
        if _terminal(current):
            return finalized_results, finalized_models

        replacement = copy.deepcopy(current)
        parent_changed = False
        for stage in _pending_stages(current_plan, current):
            prepared_item = prepared_map.get(_stage_token(stage))
            if stage["kind"] == "training":
                child_id = _training_child_id(current, stage)
                child = _try_resolve(
                    resolver,
                    kind=EntityKind.TRAINING_RESULT,
                    entity_id=str(child_id),
                )
                if prepared_item is not None:
                    accepted_training = prepared_item.accepted.get("training_result")
                    if not isinstance(accepted_training, Mapping):
                        raise _UnrecoverableProgressionError(
                            "training acceptance did not return TrainingResult"
                        )
                    try:
                        child = writer.create_immutable(
                            kind=EntityKind.TRAINING_RESULT,
                            entity_id=child_id,
                            document=accepted_training,
                        )
                    except ValueError as error:
                        raise _UnrecoverableProgressionError(
                            "accepted TrainingResult conflicts with deterministic canonical child"
                        ) from error
                if child is None:
                    continue
                validator.validate(
                    kind=EntityKind.TRAINING_RESULT,
                    entity_id=child_id,
                    document=child,
                )
                child_status = child["status"]
                if child_status not in {"completed", "failed", "cancelled"}:
                    raise _UnrecoverableProgressionError(
                        "canonical TrainingResult is not terminal"
                    )

                if child_status == "completed":
                    model_id = _model_child_id(current, stage)
                    model_document: Mapping[str, object] | None = None
                    if prepared_item is not None:
                        maybe_model = prepared_item.accepted.get("model")
                        if maybe_model is not None:
                            if not isinstance(maybe_model, Mapping):
                                raise _UnrecoverableProgressionError(
                                    "training acceptance Model is malformed"
                                )
                            model_document = maybe_model
                    if model_document is None:
                        result_payload = child.get("result")
                        if not isinstance(result_payload, Mapping):
                            raise _UnrecoverableProgressionError(
                                "completed TrainingResult has no Model lineage"
                            )
                        if result_payload.get("model") != model_id:
                            raise _UnrecoverableProgressionError(
                                "completed TrainingResult Model identity mismatch"
                            )
                        model_document = {
                            "schema": "mjtensu.mldb-v2/model/v1",
                            "id": model_id,
                            "training_result": child_id,
                        }
                    writer.create_immutable(
                        kind=EntityKind.MODEL,
                        entity_id=model_id,
                        document=model_document,
                    )
                    finalized_models.append(model_id)
                elif prepared_item is not None and prepared_item.accepted.get("model") is not None:
                    raise _UnrecoverableProgressionError(
                        "non-completed TrainingResult must not have Model"
                    )

                _set_slot_terminal(
                    replacement,
                    stage=stage,
                    disposition=str(child_status),
                    result_id=str(child_id),
                )
                finalized_results.append(child_id)
                parent_changed = True
                continue

            child_id = _evaluation_child_id(current, stage)
            child = _try_resolve(
                resolver,
                kind=EntityKind.EVALUATION_RESULT,
                entity_id=str(child_id),
            )
            if prepared_item is not None:
                accepted_evaluation = prepared_item.accepted.get("evaluation_result")
                if not isinstance(accepted_evaluation, Mapping):
                    raise _UnrecoverableProgressionError(
                        "evaluation acceptance did not return EvaluationResult"
                    )
                try:
                    child = writer.create_immutable(
                        kind=EntityKind.EVALUATION_RESULT,
                        entity_id=child_id,
                        document=accepted_evaluation,
                    )
                except ValueError as error:
                    raise _UnrecoverableProgressionError(
                        "accepted EvaluationResult conflicts with deterministic canonical child"
                    ) from error
            if child is None:
                continue
            validator.validate(
                kind=EntityKind.EVALUATION_RESULT,
                entity_id=child_id,
                document=child,
            )
            child_status = child["status"]
            if child_status not in {"completed", "failed", "cancelled"}:
                raise _UnrecoverableProgressionError(
                    "canonical EvaluationResult is not terminal"
                )
            _set_slot_terminal(
                replacement,
                stage=stage,
                disposition=str(child_status),
                result_id=str(child_id),
            )
            finalized_results.append(child_id)
            parent_changed = True

        if parent_changed:
            _finish_status_if_closed(replacement)
            writer.replace_nonterminal_study_result(
                entity_id=study_result_id,
                replacement=replacement,
            )
    return finalized_results, finalized_models


def _apply_skips(
    *,
    resolver: CanonicalRepositoryResolver,
    coordinator: StudyResultMutationCoordinator,
    writer: CanonicalRepositoryWriter,
    study_result_id: StudyResultId,
    ordered_skips: list[tuple[PlannedStageRef, str]],
) -> bool:
    if not ordered_skips:
        return False
    with coordinator.acquire(study_result_id=study_result_id):
        current, _plan = _load_context(resolver, study_result_id=study_result_id)
        if _terminal(current):
            return False
        replacement = copy.deepcopy(current)
        changed = False
        for stage, reason in ordered_skips:
            slot = _find_result_slot(replacement, stage)
            if slot["disposition"] != "pending":
                continue
            _set_slot_skipped(replacement, stage=stage, reason=reason)
            changed = True
        if not changed:
            return False
        _finish_status_if_closed(replacement)
        writer.replace_nonterminal_study_result(
            entity_id=study_result_id,
            replacement=replacement,
        )
        return True


def _mark_global_failure(
    *,
    resolver: CanonicalRepositoryResolver,
    coordinator: StudyResultMutationCoordinator,
    writer: CanonicalRepositoryWriter,
    study_result_id: StudyResultId,
) -> bool:
    with coordinator.acquire(study_result_id=study_result_id):
        current, plan = _load_context(resolver, study_result_id=study_result_id)
        if _terminal(current):
            return False
        if current["status"] == "cancelling":
            raise _UnrecoverableProgressionError(
                "cannot convert a cancelling StudyResult into failed"
            )
        replacement = copy.deepcopy(current)
        for stage in _pending_stages(plan, replacement):
            _set_slot_skipped(replacement, stage=stage, reason="global_failure")
        replacement["status"] = "failed"
        replacement["diagnostic"] = copy.deepcopy(_GLOBAL_FAILURE_DIAGNOSTIC)
        writer.replace_nonterminal_study_result(
            entity_id=study_result_id,
            replacement=replacement,
        )
        return True


def _cancellation_skips_and_request(
    *,
    backend: BackendPort,
    plan: StudyPlan,
    result: StudyResult,
    observations: Mapping[tuple[str, str, str | None], _ObservationState],
    upstream_skips: list[ReadinessSkip],
) -> list[tuple[PlannedStageRef, str]]:
    ordered: list[tuple[PlannedStageRef, str]] = [
        (item["stage"], item["reason"]) for item in upstream_skips
    ]
    upstream_tokens = {_stage_token(stage) for stage, _reason in ordered}
    if result["status"] != "cancelling":
        return ordered

    has_admitted_pending = False
    for stage in _pending_stages(plan, result):
        if _stage_token(stage) in upstream_tokens:
            continue
        state = observations.get(_stage_token(stage))
        if state is None or state.unavailable:
            continue
        if state.observation is None:
            ordered.append((stage, "study_cancelled"))
        else:
            has_admitted_pending = True

    if has_admitted_pending:
        try:
            backend.cancel_study(study_result=result["id"])
        except _BACKEND_TRANSIENT_ERRORS:
            pass
    return ordered


def _admit_ready(
    *,
    backend: BackendPort,
    plan: StudyPlan,
    result: StudyResult,
    mldb_data_root: Path,
    observations: Mapping[tuple[str, str, str | None], _ObservationState],
    ready: list[PlannedStageRef],
) -> list[StageKey]:
    admitted: list[StageKey] = []
    for stage in ready:
        state = observations.get(_stage_token(stage))
        if state is not None and (state.unavailable or state.observation is not None):
            continue
        stage_input = _build_stage_input(
            plan=plan,
            result=result,
            stage=stage,
            mldb_data_root=mldb_data_root,
        )
        expected_key = _stage_key(plan=plan, result=result, stage=stage)
        try:
            observation = backend.admit(stage_input=stage_input)
        except _BACKEND_TRANSIENT_ERRORS:
            continue
        _validate_backend_value(
            observation,
            expected_key=expected_key,
            label="backend admission",
        )
        admitted.append(copy.deepcopy(expected_key))
    return admitted


def _final_response(
    *,
    initial: StudyResult,
    final: StudyResult,
    admitted: list[StageKey],
    finalized_results: list[FinalizedResultId],
    finalized_models: list[ModelId],
    active: list[StageKey],
) -> StudyAdvanceResult:
    canonical_changed = _canonical_json_bytes(initial) != _canonical_json_bytes(final)
    return {
        "study_result": final["id"],
        "status": final["status"],
        "changed": canonical_changed or bool(admitted),
        "admitted": admitted,
        "finalized_results": finalized_results,
        "finalized_models": finalized_models,
        "active": active,
        "terminal": _terminal(final),
    }


def advance_study(
    *,
    repository_root: str | Path,
    study_result_id: StudyResultId | str,
    backend: BackendPort,
    object_bytes: _ObjectByteAccess,
) -> StudyAdvanceResult:
    """Perform exactly one resumable Study reconciliation/admission pass."""
    repository_root = Path(repository_root)
    mldb_data_root = repository_root / "mldb_data"
    validated_id = StudyResultId(_validate_typed_reference(study_result_id))
    resolver = CanonicalRepositoryResolver(mldb_data_root)
    validator = AcceptedResultRecordValidator(mldb_data_root=mldb_data_root)
    writer = CanonicalRepositoryWriter(repository_root, record_validator=validator)
    coordinator = StudyResultMutationCoordinator(repository_root)
    acceptor = ResultAcceptor(mldb_data_root=mldb_data_root, object_bytes=object_bytes)

    # 1. StudyResult / Plan / source-lineage validation.
    initial, plan = _load_context(resolver, study_result_id=validated_id)
    if _terminal(initial):
        return _final_response(
            initial=initial,
            final=initial,
            admitted=[],
            finalized_results=[],
            finalized_models=[],
            active=[],
        )

    finalized_results: list[FinalizedResultId] = []
    finalized_models: list[ModelId] = []
    admitted: list[StageKey] = []
    active: list[StageKey] = []

    try:
        # 2. Create/recover the backend-native Study execution container first.
        # Generic backends without this W011 capability remain stage-driven.
        _ensure_backend_study_execution(
            backend=backend,
            plan=plan,
            result=initial,
        )

        # 3. Observe/recover deterministic ownership for every pending StageKey.
        observations, active = _observe_pending(backend=backend, plan=plan, result=initial)

        # 3. Collect terminal work.
        candidates = _collect_candidates(
            backend=backend,
            plan=plan,
            result=initial,
            observations=observations,
        )

        # 4. Formal acceptance (including object verification) outside mutation locks.
        prepared = _prepare_terminal_acceptance(
            plan=plan,
            result=initial,
            candidates=candidates,
            acceptor=acceptor,
            mldb_data_root=mldb_data_root,
        )

        # 5-6. Child-first persistence, then stale-safe parent reconciliation.
        finalized_results, finalized_models = _reconcile_terminal_children(
            resolver=resolver,
            coordinator=coordinator,
            writer=writer,
            validator=validator,
            study_result_id=validated_id,
            plan=plan,
            prepared=prepared,
        )

        current, current_plan = _load_context(resolver, study_result_id=validated_id)
        _project_backend_summary_best_effort(
            backend=backend,
            resolver=resolver,
            result=current,
        )
        if _terminal(current):
            return _final_response(
                initial=initial,
                final=current,
                admitted=[],
                finalized_results=finalized_results,
                finalized_models=finalized_models,
                active=active,
            )

        # 7. Upstream skips, then cancellation ownership split.
        try:
            readiness_for_skips = ExecutionReadinessResolver(
                mldb_data_root=mldb_data_root
            ).derive(plan=current_plan, result=current)
        except (FileNotFoundError, ValueError) as error:
            raise _UnrecoverableProgressionError(
                "canonical execution-readiness setup is invalid"
            ) from error
        ordered_skips = _cancellation_skips_and_request(
            backend=backend,
            plan=current_plan,
            result=current,
            observations=observations,
            upstream_skips=readiness_for_skips["skipped"],
        )
        _apply_skips(
            resolver=resolver,
            coordinator=coordinator,
            writer=writer,
            study_result_id=validated_id,
            ordered_skips=ordered_skips,
        )

        current, current_plan = _load_context(resolver, study_result_id=validated_id)
        _project_backend_summary_best_effort(
            backend=backend,
            resolver=resolver,
            result=current,
        )
        if _terminal(current):
            return _final_response(
                initial=initial,
                final=current,
                admitted=[],
                finalized_results=finalized_results,
                finalized_models=finalized_models,
                active=active,
            )

        # 8. Re-derive readiness from the reconciled canonical state.
        try:
            readiness = ExecutionReadinessResolver(
                mldb_data_root=mldb_data_root
            ).derive(plan=current_plan, result=current)
        except (FileNotFoundError, ValueError) as error:
            raise _UnrecoverableProgressionError(
                "canonical execution-readiness setup is invalid"
            ) from error

        # 9. Admit only currently ready stages.
        admitted = _admit_ready(
            backend=backend,
            plan=current_plan,
            result=current,
            mldb_data_root=mldb_data_root,
            observations=observations,
            ready=readiness["ready"],
        )

        # 10. Re-resolve terminal closure after all canonical mutations.
        final, final_plan = _load_context(resolver, study_result_id=validated_id)
        if not _terminal(final) and not _pending_stages(final_plan, final):
            with coordinator.acquire(study_result_id=validated_id):
                latest, latest_plan = _load_context(resolver, study_result_id=validated_id)
                if not _terminal(latest) and not _pending_stages(latest_plan, latest):
                    replacement = copy.deepcopy(latest)
                    _finish_status_if_closed(replacement)
                    writer.replace_nonterminal_study_result(
                        entity_id=validated_id,
                        replacement=replacement,
                    )
            final, _final_plan = _load_context(resolver, study_result_id=validated_id)
    except _UnrecoverableProgressionError:
        _mark_global_failure(
            resolver=resolver,
            coordinator=coordinator,
            writer=writer,
            study_result_id=validated_id,
        )
        final, _final_plan = _load_context(resolver, study_result_id=validated_id)

    _project_backend_summary_best_effort(
        backend=backend,
        resolver=resolver,
        result=final,
    )
    return _final_response(
        initial=initial,
        final=final,
        admitted=admitted,
        finalized_results=finalized_results,
        finalized_models=finalized_models,
        active=active,
    )
