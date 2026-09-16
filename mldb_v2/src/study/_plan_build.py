"""Private deterministic StudyPlan compilation, validation, and persistence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import cast

from mldb_v2.src.common.ids import (
    EntityKind,
    StudyPlanId,
    _canonical_json_bytes,
    _validate_evaluation_coordinate_id,
    _validate_namespace_id,
    _validate_trial_id,
    _validate_typed_reference,
)
from mldb_v2.src.common.parameters import (
    _validate_public_parameter_value,
    _validate_training_seed,
)
from mldb_v2.src.repository.canonical_writes import CanonicalRepositoryWriter
from mldb_v2.src.catalog._executable_definition_loading import _validate_source_path
from mldb_v2.src.study._grid_expansion import (
    _ExpandedExistingModelSource,
    _ExpandedTrainingSource,
    _ExpandedTrial,
    _expand_study_grid,
)
from mldb_v2.src.study._planning_preflight import _StudyPlanningInput
from mldb_v2.src.study._source_pinning import (
    _CommittedSourcePin,
    _CommittedSourcePinCollection,
    _collect_study_source_pins,
)
from mldb_v2.src.study.plan import StudyPlan

_PLAN_SCHEMA = "mjtensu.mldb-v2/study-plan/v1"
_TOP_LEVEL_FIELDS = {
    "schema",
    "id",
    "content_sha256",
    "study",
    "source_commit",
    "pins",
    "trials",
}
_PIN_FIELDS = {
    "kind",
    "id",
    "yaml_sha256",
    "companion_sha256",
    "sources",
    "manifest_sha256",
    "manifest_entries",
}
_SOURCE_FIELDS = {"path", "sha256"}
_TRIAL_FIELDS = {"trial", "source", "evaluations"}
_TRAINING_SOURCE_FIELDS = {
    "kind",
    "task",
    "corpus",
    "architecture",
    "train_protocol",
    "parameters",
    "seed",
}
_EXISTING_SOURCE_FIELDS = {"kind", "model"}
_EVALUATION_FIELDS = {
    "coordinate",
    "stage",
    "task",
    "corpus",
    "evaluation_protocol",
    "parameters",
}
_PIN_KIND_ORDER = (
    "namespace",
    "task",
    "corpus",
    "architecture",
    "train_protocol",
    "evaluation_protocol",
    "study",
    "model",
    "training_result",
)
_PIN_KIND_INDEX = {kind: index for index, kind in enumerate(_PIN_KIND_ORDER)}
_EXECUTABLE_PIN_KINDS = {"architecture", "train_protocol", "evaluation_protocol"}
_NO_COMPANION_PIN_KINDS = {"namespace", "task", "study", "model", "training_result"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)


class _StudyPlanError(ValueError):
    """Bounded private StudyPlan validation/build failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _require_exact_dict(value: object, fields: set[str], *, code: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise _StudyPlanError(code)
    return value


def _validate_sha256(value: object, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _StudyPlanError("invalid_sha256")
    return value


def _validate_source_commit(value: object) -> str:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        raise _StudyPlanError("invalid_source_commit")
    return value


def _validate_parameters(value: object) -> None:
    if not isinstance(value, Mapping):
        raise _StudyPlanError("invalid_parameters")
    for key, item in value.items():
        if type(key) is not str:
            raise _StudyPlanError("invalid_parameters")
        try:
            _validate_public_parameter_value(item)
        except ValueError as error:
            raise _StudyPlanError("invalid_parameters") from error


def _validate_pin_identity(kind: str, value: object) -> str:
    try:
        if kind == "namespace":
            return str(_validate_namespace_id(value))
        return _validate_typed_reference(value)
    except ValueError as error:
        raise _StudyPlanError("invalid_pin_id") from error


def _validate_pin(pin: object) -> tuple[str, str]:
    record = _require_exact_dict(pin, _PIN_FIELDS, code="invalid_pin")
    kind = record["kind"]
    if type(kind) is not str or kind not in _PIN_KIND_INDEX:
        raise _StudyPlanError("invalid_pin_kind")
    entity_id = _validate_pin_identity(kind, record["id"])
    _validate_sha256(record["yaml_sha256"])
    companion = _validate_sha256(record["companion_sha256"], nullable=True)

    sources = record["sources"]
    if type(sources) is not list:
        raise _StudyPlanError("invalid_pin_sources")
    source_paths: list[str] = []
    for raw_source in sources:
        source = _require_exact_dict(raw_source, _SOURCE_FIELDS, code="invalid_pin_source")
        try:
            path = _validate_source_path(
                source["path"], namespace=entity_id.split("/", 1)[0]
            )
        except ValueError as error:
            raise _StudyPlanError("invalid_pin_source_path") from error
        _validate_sha256(source["sha256"])
        source_paths.append(path)
    if source_paths != sorted(source_paths) or len(source_paths) != len(set(source_paths)):
        raise _StudyPlanError("invalid_pin_source_order")

    manifest_sha = _validate_sha256(record["manifest_sha256"], nullable=True)
    manifest_entries = record["manifest_entries"]
    if manifest_entries is not None and (type(manifest_entries) is not int or manifest_entries < 0):
        raise _StudyPlanError("invalid_manifest_entries")

    if kind in _NO_COMPANION_PIN_KINDS:
        if companion is not None or sources or manifest_sha is not None or manifest_entries is not None:
            raise _StudyPlanError("invalid_pin_structure")
    elif kind in _EXECUTABLE_PIN_KINDS:
        if companion is None or manifest_sha is not None or manifest_entries is not None:
            raise _StudyPlanError("invalid_pin_structure")
    elif kind == "corpus":
        if sources or manifest_sha is None or manifest_entries is None:
            raise _StudyPlanError("invalid_pin_structure")
    else:
        raise _StudyPlanError("invalid_pin_kind")
    return kind, entity_id


def _validate_pins(value: object) -> dict[str, set[str]]:
    if type(value) is not list:
        raise _StudyPlanError("invalid_pins")
    refs = {kind: set() for kind in _PIN_KIND_ORDER}
    previous_key: tuple[int, str] | None = None
    for raw_pin in value:
        kind, entity_id = _validate_pin(raw_pin)
        key = (_PIN_KIND_INDEX[kind], entity_id)
        if previous_key is not None and key <= previous_key:
            if key == previous_key:
                raise _StudyPlanError("duplicate_pin")
            raise _StudyPlanError("invalid_pin_order")
        previous_key = key
        refs[kind].add(entity_id)

    namespaces = refs["namespace"]
    for kind in _PIN_KIND_ORDER[1:]:
        for entity_id in refs[kind]:
            namespace = entity_id.split("/", 1)[0]
            if namespace not in namespaces:
                raise _StudyPlanError("missing_namespace_pin")
    return refs


def _require_ref(refs: Mapping[str, set[str]], kind: str, entity_id: str) -> None:
    if entity_id not in refs[kind]:
        raise _StudyPlanError("missing_referenced_pin")


def _validate_training_source(source: dict[str, object], refs: Mapping[str, set[str]]) -> None:
    if set(source) != _TRAINING_SOURCE_FIELDS:
        raise _StudyPlanError("invalid_trial_source")
    try:
        task = _validate_typed_reference(source["task"])
        corpus = _validate_typed_reference(source["corpus"])
        architecture = _validate_typed_reference(source["architecture"])
        train_protocol = _validate_typed_reference(source["train_protocol"])
        _validate_training_seed(source["seed"])
    except ValueError as error:
        raise _StudyPlanError("invalid_trial_source") from error
    _validate_parameters(source["parameters"])
    _require_ref(refs, "task", task)
    _require_ref(refs, "corpus", corpus)
    _require_ref(refs, "architecture", architecture)
    _require_ref(refs, "train_protocol", train_protocol)


def _validate_existing_source(source: dict[str, object], refs: Mapping[str, set[str]]) -> None:
    if set(source) != _EXISTING_SOURCE_FIELDS:
        raise _StudyPlanError("invalid_trial_source")
    try:
        model = _validate_typed_reference(source["model"])
    except ValueError as error:
        raise _StudyPlanError("invalid_trial_source") from error
    _require_ref(refs, "model", model)


def _validate_evaluation(value: object, expected_index: int, refs: Mapping[str, set[str]]) -> None:
    record = _require_exact_dict(value, _EVALUATION_FIELDS, code="invalid_evaluation")
    expected = f"eval-{expected_index:04d}"
    try:
        actual = str(_validate_evaluation_coordinate_id(record["coordinate"]))
        task = _validate_typed_reference(record["task"])
        corpus = _validate_typed_reference(record["corpus"])
        protocol = _validate_typed_reference(record["evaluation_protocol"])
    except ValueError as error:
        raise _StudyPlanError("invalid_evaluation") from error
    if actual != expected:
        raise _StudyPlanError("invalid_evaluation_sequence")
    if type(record["stage"]) is not str or not record["stage"]:
        raise _StudyPlanError("invalid_evaluation")
    _validate_parameters(record["parameters"])
    _require_ref(refs, "task", task)
    _require_ref(refs, "corpus", corpus)
    _require_ref(refs, "evaluation_protocol", protocol)


def _validate_trials(value: object, refs: Mapping[str, set[str]]) -> None:
    if type(value) is not list or not value:
        raise _StudyPlanError("invalid_trials")
    for trial_index, raw_trial in enumerate(value, start=1):
        trial = _require_exact_dict(raw_trial, _TRIAL_FIELDS, code="invalid_trial")
        expected = f"trial-{trial_index:04d}"
        try:
            actual = str(_validate_trial_id(trial["trial"]))
        except ValueError as error:
            raise _StudyPlanError("invalid_trial_sequence") from error
        if actual != expected:
            raise _StudyPlanError("invalid_trial_sequence")

        source = trial["source"]
        if type(source) is not dict:
            raise _StudyPlanError("invalid_trial_source")
        kind = source.get("kind")
        if kind == "training":
            _validate_training_source(source, refs)
        elif kind == "existing_model":
            _validate_existing_source(source, refs)
        else:
            raise _StudyPlanError("invalid_trial_source")

        evaluations = trial["evaluations"]
        if type(evaluations) is not list or not evaluations:
            raise _StudyPlanError("invalid_evaluations")
        for evaluation_index, evaluation in enumerate(evaluations, start=1):
            _validate_evaluation(evaluation, evaluation_index, refs)


def _digest_payload(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": record["schema"],
        "study": record["study"],
        "source_commit": record["source_commit"],
        "pins": record["pins"],
        "trials": record["trials"],
    }


def _content_digest(record: Mapping[str, object]) -> str:
    try:
        payload = _canonical_json_bytes(_digest_payload(record))
    except (KeyError, TypeError, ValueError) as error:
        raise _StudyPlanError("invalid_plan_content") from error
    return hashlib.sha256(payload).hexdigest()


def _plan_id(study_id: str, content_sha256: str) -> StudyPlanId:
    namespace, local_id = study_id.split("/", 1)
    return StudyPlanId(f"{namespace}/{local_id}-plan-{content_sha256[:16]}")


def _validate_study_plan(document: Mapping[str, object]) -> StudyPlan:
    if not isinstance(document, Mapping):
        raise _StudyPlanError("invalid_plan")
    record = dict(document)
    if set(record) != _TOP_LEVEL_FIELDS:
        raise _StudyPlanError("invalid_plan_fields")
    if record["schema"] != _PLAN_SCHEMA:
        raise _StudyPlanError("invalid_plan_schema")
    try:
        study = _validate_typed_reference(record["study"])
        plan_id = _validate_typed_reference(record["id"])
    except ValueError as error:
        raise _StudyPlanError("invalid_plan_identity") from error
    _validate_source_commit(record["source_commit"])
    supplied_digest = _validate_sha256(record["content_sha256"])

    refs = _validate_pins(record["pins"])
    if refs["study"] != {study}:
        raise _StudyPlanError("missing_study_pin")
    _validate_trials(record["trials"], refs)

    actual_digest = _content_digest(record)
    if supplied_digest != actual_digest:
        raise _StudyPlanError("content_digest_mismatch")
    expected_id = str(_plan_id(study, actual_digest))
    if plan_id != expected_id:
        raise _StudyPlanError("plan_id_mismatch")
    return cast(StudyPlan, record)


class _StudyPlanRecordValidator:
    """CanonicalRecordValidator adapter owned by the StudyPlan domain."""

    def validate(self, *, kind, entity_id, document) -> None:
        if kind != EntityKind.STUDY_PLAN:
            raise ValueError("StudyPlan validator only accepts study_plan")
        validated = _validate_study_plan(document)
        if validated["id"] != entity_id:
            raise ValueError("StudyPlan entity identity mismatch")


def _convert_pin(pin: _CommittedSourcePin) -> dict[str, object]:
    return {
        "kind": pin.kind,
        "id": pin.id,
        "yaml_sha256": pin.yaml_sha256,
        "companion_sha256": pin.companion_sha256,
        "sources": [
            {"path": source.path, "sha256": source.sha256}
            for source in pin.sources
        ],
        "manifest_sha256": pin.manifest_sha256,
        "manifest_entries": pin.manifest_entries,
    }


def _convert_trial(trial: _ExpandedTrial) -> dict[str, object]:
    source = trial.source
    if isinstance(source, _ExpandedTrainingSource):
        source_record: dict[str, object] = {
            "kind": "training",
            "task": source.task,
            "corpus": source.corpus,
            "architecture": source.architecture,
            "train_protocol": source.train_protocol,
            "parameters": deepcopy(dict(source.parameters)),
            "seed": source.seed,
        }
    elif isinstance(source, _ExpandedExistingModelSource):
        source_record = {"kind": "existing_model", "model": source.model}
    else:
        raise _StudyPlanError("invalid_expanded_trial")

    evaluations = [
        {
            "coordinate": evaluation.coordinate,
            "stage": evaluation.stage,
            "task": evaluation.task,
            "corpus": evaluation.corpus,
            "evaluation_protocol": evaluation.evaluation_protocol,
            "parameters": deepcopy(dict(evaluation.parameters)),
        }
        for evaluation in trial.evaluations
    ]
    return {"trial": trial.trial, "source": source_record, "evaluations": evaluations}


def _build_from_parts(
    *,
    planning: _StudyPlanningInput,
    expansion: tuple[_ExpandedTrial, ...],
    pin_collection: _CommittedSourcePinCollection,
) -> StudyPlan:
    try:
        study = _validate_typed_reference(planning.study["id"])
    except (KeyError, ValueError) as error:
        raise _StudyPlanError("invalid_planning_input") from error
    payload: dict[str, object] = {
        "schema": _PLAN_SCHEMA,
        "study": study,
        "source_commit": pin_collection.source_commit,
        "pins": [_convert_pin(pin) for pin in pin_collection.pins],
        "trials": [_convert_trial(trial) for trial in expansion],
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    plan: dict[str, object] = {
        "schema": _PLAN_SCHEMA,
        "id": _plan_id(study, digest),
        "content_sha256": digest,
        "study": study,
        "source_commit": pin_collection.source_commit,
        "pins": payload["pins"],
        "trials": payload["trials"],
    }
    return _validate_study_plan(plan)


def _build_study_plan(
    *,
    repository_root: str | Path,
    mldb_data_root: str | Path,
    planning: _StudyPlanningInput,
    selected_commit: str,
    grid_expander: Callable[[_StudyPlanningInput], tuple[_ExpandedTrial, ...]] = _expand_study_grid,
    source_pin_collector: Callable[..., _CommittedSourcePinCollection] = _collect_study_source_pins,
) -> StudyPlan:
    """Compile one planning boundary into one deterministic complete StudyPlan."""

    if not isinstance(planning, _StudyPlanningInput):
        raise _StudyPlanError("invalid_planning_input")
    expansion = grid_expander(planning)
    pin_collection = source_pin_collector(
        repository_root=repository_root,
        mldb_data_root=mldb_data_root,
        selected_commit=selected_commit,
        planning=planning,
    )
    return _build_from_parts(
        planning=planning,
        expansion=expansion,
        pin_collection=pin_collection,
    )


def _create_study_plan(*, repository_root: str | Path, plan: Mapping[str, object]) -> StudyPlan:
    """Create or replay one validated immutable StudyPlan through the W001 writer."""

    validated = _validate_study_plan(plan)
    writer = CanonicalRepositoryWriter(
        repository_root,
        record_validator=_StudyPlanRecordValidator(),
    )
    stored = writer.create_immutable(
        kind=EntityKind.STUDY_PLAN,
        entity_id=validated["id"],
        document=validated,
    )
    return _validate_study_plan(stored)
