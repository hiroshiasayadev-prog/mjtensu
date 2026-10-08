"""Human-readable Study condition labels and display-name safety gates."""

from __future__ import annotations

import json
from collections.abc import Mapping

from mldb_v2.src.common.display_names import validate_user_facing_display_name
from mldb_v2.src.common.ids import EntityKind

def _stable_value(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if type(value) is float:
        return f"{value:.12g}"
    if type(value) in {int, str}:
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def build_condition_labels(
    *,
    base_labels: Mapping[str, str],
    conditions: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    """Build unique labels from visible model names plus actually varying conditions.

    Internal trial IDs are dictionary keys only and are never used as label fallbacks.
    Ambiguous duplicate conditions fail closed instead of leaking the internal ID.
    """

    if set(base_labels) != set(conditions):
        raise ValueError("display-label base/condition trial sets do not match")

    groups: dict[str, list[str]] = {}
    for trial, raw_base in base_labels.items():
        base = validate_user_facing_display_name(raw_base, field="base display label")
        groups.setdefault(base, []).append(trial)

    labels: dict[str, str] = {}
    used: set[str] = set()
    missing = object()
    for base, trials in groups.items():
        varying: list[str] = []
        if len(trials) > 1:
            keys = sorted({key for trial in trials for key in conditions[trial]})
            for key in keys:
                serialized = {
                    "<missing>" if (value := conditions[trial].get(key, missing)) is missing
                    else json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
                    for trial in trials
                }
                if len(serialized) > 1:
                    varying.append(key)
            if not varying:
                raise ValueError(
                    f"user-facing conditions for {base!r} are indistinguishable; "
                    "add an explicit varying parameter instead of falling back to an internal trial ID"
                )

        for trial in trials:
            if varying:
                fragments = [
                    f"{key}={_stable_value(conditions[trial].get(key, '<unset>'))}"
                    for key in varying
                ]
                candidate = f"{base} | {', '.join(fragments)}"
            else:
                candidate = base
            candidate = validate_user_facing_display_name(candidate)
            if candidate in used:
                raise ValueError(f"duplicate user-facing display name: {candidate!r}")
            labels[trial] = candidate
            used.add(candidate)
    return labels


def _local_reference_name(reference: object) -> str:
    text = str(reference)
    return text.split("/", 1)[1] if "/" in text else text


def build_plan_conditions(
    *,
    plan: Mapping[str, object],
    resolver: object | None,
) -> dict[str, dict[str, object]]:
    """Derive direct, human-readable condition snapshots from one StudyPlan."""

    raw_trials = plan.get("trials")
    if type(raw_trials) is not list or not raw_trials:
        raise ValueError("StudyPlan trials are missing")
    base_labels: dict[str, str] = {}
    conditions: dict[str, dict[str, object]] = {}
    resolve = getattr(resolver, "resolve", None) if resolver is not None else None

    for raw_trial in raw_trials:
        if not isinstance(raw_trial, Mapping) or type(raw_trial.get("trial")) is not str:
            raise ValueError("StudyPlan trial is malformed")
        trial_id = str(raw_trial["trial"])
        source = raw_trial.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("StudyPlan trial source is malformed")

        architecture_id: str | None = None
        model_id: str | None = None
        source_conditions: dict[str, object] = {}
        if source.get("kind") == "training":
            if type(source.get("architecture")) is not str:
                raise ValueError("training trial architecture is missing")
            architecture_id = str(source["architecture"])
            parameters = source.get("parameters")
            if isinstance(parameters, Mapping):
                source_conditions.update({str(key): value for key, value in parameters.items()})
            if "seed" in source:
                source_conditions["seed"] = source["seed"]
        elif source.get("kind") == "existing_model":
            if type(source.get("model")) is not str:
                raise ValueError("existing-model trial model is missing")
            model_id = str(source["model"])
            # The model identity itself is part of the exact source condition.
            source_conditions["model"] = model_id
            if callable(resolve):
                try:
                    model = resolve(kind=EntityKind.MODEL, entity_id=model_id)
                    training_result_id = model.get("training_result") if isinstance(model, Mapping) else None
                    if type(training_result_id) is str:
                        training_result = resolve(
                            kind=EntityKind.TRAINING_RESULT,
                            entity_id=training_result_id,
                        )
                        if isinstance(training_result, Mapping):
                            if type(training_result.get("architecture")) is str:
                                architecture_id = str(training_result["architecture"])
                            parameters = training_result.get("parameters")
                            if isinstance(parameters, Mapping):
                                source_conditions.update(
                                    {str(key): value for key, value in parameters.items()}
                                )
                            if "seed" in training_result:
                                source_conditions["seed"] = training_result["seed"]
                except (FileNotFoundError, ValueError):
                    pass
        else:
            raise ValueError("unsupported StudyPlan trial source")

        label: str | None = None
        if architecture_id is not None:
            if callable(resolve):
                try:
                    architecture = resolve(
                        kind=EntityKind.ARCHITECTURE,
                        entity_id=architecture_id,
                    )
                except (FileNotFoundError, ValueError):
                    architecture = None
                if (
                    isinstance(architecture, Mapping)
                    and type(architecture.get("name")) is str
                    and architecture["name"]
                ):
                    label = str(architecture["name"])
            if label is None:
                label = _local_reference_name(architecture_id)
        elif model_id is not None:
            # Never use an internal model ID containing trial- as a visible fallback.
            label = _local_reference_name(model_id)
        if label is None:
            raise ValueError("StudyPlan trial has no user-facing model label")
        base_labels[trial_id] = validate_user_facing_display_name(
            label, field="Study condition base label"
        )
        conditions[trial_id] = source_conditions

    labels = build_condition_labels(base_labels=base_labels, conditions=conditions)
    return {
        trial: {"label": labels[trial], "parameters": dict(conditions[trial])}
        for trial in labels
    }


def build_plan_condition_labels(
    *,
    plan: Mapping[str, object],
    resolver: object | None,
) -> dict[str, str]:
    return {
        trial: str(snapshot["label"])
        for trial, snapshot in build_plan_conditions(plan=plan, resolver=resolver).items()
    }
