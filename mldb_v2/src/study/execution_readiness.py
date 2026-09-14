"""Pure semantic readiness and exact StageInput construction for Study Plans."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict, cast

from mldb_v2.src.backend.stage_input import RuntimeModel, StageInput
from mldb_v2.src.common.ids import (
    EvaluationCoordinateId,
    TrialId,
    _canonical_json_bytes,
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
)
from mldb_v2.src.results.study_result import StudyResult, StudyResultTrial, _validate_study_result
from mldb_v2.src.study._plan_build import _validate_study_plan
from mldb_v2.src.study.plan import PlanTrial, StudyPlan
from mldb_v2.src.training.model import _ResolvedModelLineage, _resolve_model_lineage


class TrainingStageRef(TypedDict):
    kind: Literal["training"]
    trial: TrialId


class EvaluationStageRef(TypedDict):
    kind: Literal["evaluation"]
    trial: TrialId
    coordinate: EvaluationCoordinateId


PlannedStageRef: TypeAlias = TrainingStageRef | EvaluationStageRef
UpstreamSkipReason: TypeAlias = Literal["upstream_failed", "upstream_cancelled"]


class ReadinessSkip(TypedDict):
    stage: PlannedStageRef
    reason: UpstreamSkipReason


class ExecutionReadiness(TypedDict):
    ready: list[PlannedStageRef]
    skipped: list[ReadinessSkip]


def _same_canonical_value(left: object, right: object) -> bool:
    try:
        return _canonical_json_bytes(left) == _canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _validate_result_plan_lineage(result: StudyResult, plan: StudyPlan) -> None:
    if result["plan"] != plan["id"]:
        raise ValueError("StudyResult plan does not match StudyPlan")
    if result["study"] != plan["study"]:
        raise ValueError("StudyResult study does not match StudyPlan")
    if result["source_commit"] != plan["source_commit"]:
        raise ValueError("StudyResult source_commit does not match StudyPlan")
    if len(result["trials"]) != len(plan["trials"]):
        raise ValueError("StudyResult trial topology does not match StudyPlan")

    for result_trial, plan_trial in zip(result["trials"], plan["trials"]):
        if result_trial["trial"] != plan_trial["trial"]:
            raise ValueError("StudyResult trial order does not match StudyPlan")
        expects_training = plan_trial["source"]["kind"] == "training"
        if (result_trial["training"] is not None) != expects_training:
            raise ValueError("StudyResult training topology does not match StudyPlan")
        if len(result_trial["evaluations"]) != len(plan_trial["evaluations"]):
            raise ValueError("StudyResult evaluation topology does not match StudyPlan")
        for result_slot, plan_coordinate in zip(
            result_trial["evaluations"], plan_trial["evaluations"]
        ):
            if result_slot["coordinate"] != plan_coordinate["coordinate"]:
                raise ValueError("StudyResult evaluation coordinate does not match StudyPlan")
            if result_slot["stage"] != plan_coordinate["stage"]:
                raise ValueError("StudyResult evaluation stage does not match StudyPlan")


def _validated_inputs(
    plan: Mapping[str, object] | StudyPlan,
    result: object,
) -> tuple[StudyPlan, StudyResult]:
    validated_plan = _validate_study_plan(plan)
    validated_result = _validate_study_result(result)
    _validate_result_plan_lineage(validated_result, validated_plan)
    return validated_plan, validated_result


def _find_plan_trial(plan: StudyPlan, trial_id: str) -> PlanTrial:
    matches = [trial for trial in plan["trials"] if trial["trial"] == trial_id]
    if len(matches) != 1:
        raise ValueError("planned stage trial does not name exactly one StudyPlan trial")
    return matches[0]


def _find_result_trial(result: StudyResult, trial_id: str) -> StudyResultTrial:
    matches = [trial for trial in result["trials"] if trial["trial"] == trial_id]
    if len(matches) != 1:
        raise ValueError("planned stage trial does not name exactly one StudyResult trial")
    return matches[0]


def _require_plan_pin(plan: StudyPlan, *, kind: str, entity_id: str) -> None:
    matches = [
        pin for pin in plan["pins"]
        if pin["kind"] == kind and pin["id"] == entity_id
    ]
    if len(matches) != 1:
        raise ValueError(f"StudyPlan requires exactly one {kind} pin for runtime Model lineage")


def _validate_runtime_training_lineage(
    *,
    plan: StudyPlan,
    result: StudyResult,
    plan_trial: PlanTrial,
    result_trial: StudyResultTrial,
    lineage: _ResolvedModelLineage,
) -> None:
    source = plan_trial["source"]
    if source["kind"] != "training":
        raise ValueError("runtime-produced Model requires a training-source trial")
    training_slot = result_trial["training"]
    if training_slot is None or training_slot["disposition"] != "completed":
        raise ValueError("runtime-produced Model requires completed canonical training")
    training_result = lineage.training_result
    expected_training_result = f"{result['id']}-{plan_trial['trial']}-train"
    expected_model = f"{result['id']}-{plan_trial['trial']}-model"
    if training_slot["result"] != expected_training_result:
        raise ValueError("completed training slot does not reference deterministic TrainingResult")
    if training_result["id"] != expected_training_result:
        raise ValueError("runtime Model TrainingResult identity does not match StudyResult slot")
    if lineage.model["id"] != expected_model:
        raise ValueError("runtime Model identity does not match deterministic StudyResult child")
    if training_result["study_result"] != result["id"]:
        raise ValueError("runtime TrainingResult StudyResult lineage mismatch")
    if training_result["plan"] != plan["id"]:
        raise ValueError("runtime TrainingResult Plan lineage mismatch")
    if training_result["trial"] != plan_trial["trial"]:
        raise ValueError("runtime TrainingResult trial lineage mismatch")
    if training_result["source_commit"] != plan["source_commit"]:
        raise ValueError("runtime TrainingResult source commit lineage mismatch")
    expected = {
        "task": source["task"],
        "corpus": source["corpus"],
        "architecture": source["architecture"],
        "train_protocol": source["train_protocol"],
        "parameters": source["parameters"],
        "seed": source["seed"],
    }
    actual = {
        "task": training_result["task"],
        "corpus": training_result["corpus"],
        "architecture": training_result["architecture"],
        "train_protocol": training_result["train_protocol"],
        "parameters": training_result["parameters"],
        "seed": training_result["seed"],
    }
    if not _same_canonical_value(actual, expected):
        raise ValueError("runtime TrainingResult does not exactly match StudyPlan training source")


def _validate_existing_model_lineage(
    *,
    plan: StudyPlan,
    plan_trial: PlanTrial,
    lineage: _ResolvedModelLineage,
) -> None:
    source = plan_trial["source"]
    if source["kind"] != "existing_model":
        raise ValueError("existing Model lineage requires an existing-model trial")
    if lineage.model["id"] != source["model"]:
        raise ValueError("existing Model identity does not match StudyPlan source")
    _require_plan_pin(plan, kind="model", entity_id=str(lineage.model["id"]))
    _require_plan_pin(
        plan,
        kind="training_result",
        entity_id=str(lineage.training_result["id"]),
    )


def _runtime_model_for_trial(
    *,
    plan: StudyPlan,
    result: StudyResult,
    plan_trial: PlanTrial,
    result_trial: StudyResultTrial,
    mldb_data_root: str | Path,
) -> RuntimeModel:
    source = plan_trial["source"]
    if source["kind"] == "training":
        model_id = f"{result['id']}-{plan_trial['trial']}-model"
    elif source["kind"] == "existing_model":
        model_id = str(source["model"])
    else:
        raise ValueError("unsupported StudyPlan trial source")

    lineage = _resolve_model_lineage(mldb_data_root, model_id)
    if source["kind"] == "training":
        _validate_runtime_training_lineage(
            plan=plan,
            result=result,
            plan_trial=plan_trial,
            result_trial=result_trial,
            lineage=lineage,
        )
    else:
        _validate_existing_model_lineage(
            plan=plan,
            plan_trial=plan_trial,
            lineage=lineage,
        )

    return {
        "model": lineage.model["id"],
        "training_result": lineage.training_result["id"],
        "task": lineage.training_result["task"],
        "architecture": lineage.training_result["architecture"],
        "weights": copy.deepcopy(lineage.weights),
    }


def _validate_runtime_model_for_evaluations(
    runtime_model: RuntimeModel,
    plan_trial: PlanTrial,
) -> None:
    for coordinate in plan_trial["evaluations"]:
        if coordinate["task"] != runtime_model["task"]:
            raise ValueError("runtime Model Task does not match planned Evaluation Task")


def _training_ref(trial_id: TrialId) -> TrainingStageRef:
    return {"kind": "training", "trial": trial_id}


def _evaluation_ref(
    trial_id: TrialId,
    coordinate: EvaluationCoordinateId,
) -> EvaluationStageRef:
    return {"kind": "evaluation", "trial": trial_id, "coordinate": coordinate}


class ExecutionReadinessResolver:
    """Derive Plan dependency readiness without consulting backend ownership or scheduling state."""

    def __init__(self, *, mldb_data_root: str | Path) -> None:
        self._mldb_data_root = Path(mldb_data_root)

    def derive(
        self,
        *,
        plan: Mapping[str, object] | StudyPlan,
        result: StudyResult,
    ) -> ExecutionReadiness:
        validated_plan, validated_result = _validated_inputs(plan, result)
        ready: list[PlannedStageRef] = []
        skipped: list[ReadinessSkip] = []
        cancelling = validated_result["status"] == "cancelling"

        for plan_trial, result_trial in zip(
            validated_plan["trials"], validated_result["trials"]
        ):
            trial_id = plan_trial["trial"]
            source = plan_trial["source"]
            pending_evaluations = [
                (coordinate, slot)
                for coordinate, slot in zip(
                    plan_trial["evaluations"], result_trial["evaluations"]
                )
                if slot["disposition"] == "pending"
            ]

            if source["kind"] == "training":
                training = result_trial["training"]
                if training is None:
                    raise ValueError("training-source trial is missing StudyResult training slot")
                disposition = training["disposition"]
                if disposition == "pending":
                    if not cancelling:
                        ready.append(_training_ref(trial_id))
                    continue
                if disposition in {"failed", "cancelled"}:
                    reason: UpstreamSkipReason = (
                        "upstream_failed"
                        if disposition == "failed"
                        else "upstream_cancelled"
                    )
                    for coordinate, _slot in pending_evaluations:
                        skipped.append(
                            {
                                "stage": _evaluation_ref(
                                    trial_id, coordinate["coordinate"]
                                ),
                                "reason": reason,
                            }
                        )
                    continue
                if disposition != "completed" or cancelling or not pending_evaluations:
                    continue
                runtime_model = _runtime_model_for_trial(
                    plan=validated_plan,
                    result=validated_result,
                    plan_trial=plan_trial,
                    result_trial=result_trial,
                    mldb_data_root=self._mldb_data_root,
                )
                _validate_runtime_model_for_evaluations(runtime_model, plan_trial)
                ready.extend(
                    _evaluation_ref(trial_id, coordinate["coordinate"])
                    for coordinate, _slot in pending_evaluations
                )
                continue

            if source["kind"] != "existing_model":
                raise ValueError("unsupported StudyPlan trial source")
            if cancelling or not pending_evaluations:
                continue
            runtime_model = _runtime_model_for_trial(
                plan=validated_plan,
                result=validated_result,
                plan_trial=plan_trial,
                result_trial=result_trial,
                mldb_data_root=self._mldb_data_root,
            )
            _validate_runtime_model_for_evaluations(runtime_model, plan_trial)
            ready.extend(
                _evaluation_ref(trial_id, coordinate["coordinate"])
                for coordinate, _slot in pending_evaluations
            )

        return {"ready": ready, "skipped": skipped}


def _validate_stage_ref(stage: object) -> PlannedStageRef:
    if type(stage) is not dict:
        raise ValueError("planned stage reference must be a mapping")
    kind = stage.get("kind")
    if kind == "training":
        if set(stage) != {"kind", "trial"}:
            raise ValueError("training stage reference fields do not match schema")
        return _training_ref(_validate_trial_id(stage["trial"]))
    if kind == "evaluation":
        if set(stage) != {"kind", "trial", "coordinate"}:
            raise ValueError("evaluation stage reference fields do not match schema")
        return _evaluation_ref(
            _validate_trial_id(stage["trial"]),
            _validate_evaluation_coordinate_id(stage["coordinate"]),
        )
    raise ValueError("planned stage reference kind must be training or evaluation")


def _materialize_stage_input(
    *,
    plan: Mapping[str, object] | StudyPlan,
    result: StudyResult,
    stage: PlannedStageRef,
    mldb_data_root: str | Path,
) -> StageInput:
    """Materialize one exact pending-stage input without consulting backend ownership."""
    validated_plan, validated_result = _validated_inputs(plan, result)
    stage_ref = _validate_stage_ref(stage)
    trial_id = str(stage_ref["trial"])
    plan_trial = _find_plan_trial(validated_plan, trial_id)
    result_trial = _find_result_trial(validated_result, trial_id)

    if stage_ref["kind"] == "training":
        slot = result_trial["training"]
        if slot is None or slot["disposition"] != "pending":
            raise ValueError("StageInput materialization requires a pending training slot")
    else:
        coordinate_id = stage_ref["coordinate"]
        result_slots = [
            slot for slot in result_trial["evaluations"]
            if slot["coordinate"] == coordinate_id
        ]
        if len(result_slots) != 1 or result_slots[0]["disposition"] != "pending":
            raise ValueError("StageInput materialization requires a pending evaluation slot")

    common = {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": validated_result["id"],
        "plan": validated_plan["id"],
        "plan_sha256": validated_plan["content_sha256"],
        "trial": plan_trial["trial"],
        "source_commit": validated_plan["source_commit"],
        "pins": copy.deepcopy(validated_plan["pins"]),
    }

    if stage_ref["kind"] == "training":
        source = plan_trial["source"]
        if source["kind"] != "training":
            raise ValueError("training stage reference does not match StudyPlan source")
        return cast(
            StageInput,
            {
                **common,
                "kind": "training",
                "coordinate": None,
                "stage": {
                    "task": source["task"],
                    "corpus": source["corpus"],
                    "architecture": source["architecture"],
                    "train_protocol": source["train_protocol"],
                    "parameters": copy.deepcopy(source["parameters"]),
                    "seed": source["seed"],
                },
                "runtime_model": None,
            },
        )

    coordinate_id = stage_ref["coordinate"]
    matches = [
        coordinate for coordinate in plan_trial["evaluations"]
        if coordinate["coordinate"] == coordinate_id
    ]
    if len(matches) != 1:
        raise ValueError("evaluation stage reference does not match StudyPlan coordinate")
    coordinate = matches[0]
    runtime_model = _runtime_model_for_trial(
        plan=validated_plan,
        result=validated_result,
        plan_trial=plan_trial,
        result_trial=result_trial,
        mldb_data_root=mldb_data_root,
    )
    if runtime_model["task"] != coordinate["task"]:
        raise ValueError("runtime Model Task does not match planned Evaluation Task")
    return cast(
        StageInput,
        {
            **common,
            "kind": "evaluation",
            "coordinate": coordinate["coordinate"],
            "stage": {
                "name": coordinate["stage"],
                "task": coordinate["task"],
                "corpus": coordinate["corpus"],
                "evaluation_protocol": coordinate["evaluation_protocol"],
                "parameters": copy.deepcopy(coordinate["parameters"]),
            },
            "runtime_model": runtime_model,
        },
    )


def _build_stage_input(
    *,
    plan: Mapping[str, object] | StudyPlan,
    result: StudyResult,
    stage: PlannedStageRef,
    mldb_data_root: str | Path,
) -> StageInput:
    """Build one exact backend-neutral StageInput, only for a currently ready stage."""
    validated_plan, validated_result = _validated_inputs(plan, result)
    stage_ref = _validate_stage_ref(stage)
    resolver = ExecutionReadinessResolver(mldb_data_root=mldb_data_root)
    readiness = resolver.derive(plan=validated_plan, result=validated_result)
    if stage_ref not in readiness["ready"]:
        raise ValueError("StageInput may only be constructed for a currently ready stage")
    return _materialize_stage_input(
        plan=validated_plan,
        result=validated_result,
        stage=stage_ref,
        mldb_data_root=mldb_data_root,
    )
