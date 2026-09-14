"""Atomic canonical record creation and Study Result transition safety."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

from mldb_v2.src.common.diagnostic import _validate_diagnostic
from mldb_v2.src.common.ids import (
    EntityKind,
    EvaluationResultId,
    ModelId,
    StudyPlanId,
    StudyResultId,
    TrainingResultId,
    _canonical_json_bytes,
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
    _validate_typed_reference,
)
from mldb_v2.src.repository._process_lock import _process_file_lock
from mldb_v2.src.repository._yaml import _YamlError, _load_yaml
from mldb_v2.src.repository.resolution import _read_namespace_document

CanonicalDocument: TypeAlias = Mapping[str, object]
ImmutableCanonicalKind: TypeAlias = Literal[
    EntityKind.STUDY_PLAN,
    EntityKind.TRAINING_RESULT,
    EntityKind.MODEL,
    EntityKind.EVALUATION_RESULT,
]
ImmutableCanonicalId: TypeAlias = (
    StudyPlanId | TrainingResultId | ModelId | EvaluationResultId
)


class CanonicalRecordValidator(Protocol):
    """Owning-domain validation port for complete immutable canonical records."""

    def validate(
        self,
        *,
        kind: ImmutableCanonicalKind,
        entity_id: ImmutableCanonicalId,
        document: CanonicalDocument,
    ) -> None: ...

_DOMAINS = {
    EntityKind.STUDY_PLAN: "study_plans",
    EntityKind.TRAINING_RESULT: "training_results",
    EntityKind.MODEL: "models",
    EntityKind.EVALUATION_RESULT: "evaluation_results",
    EntityKind.STUDY_RESULT: "study_results",
}
_IMMUTABLE_KINDS = frozenset(
    {
        EntityKind.STUDY_PLAN,
        EntityKind.TRAINING_RESULT,
        EntityKind.MODEL,
        EntityKind.EVALUATION_RESULT,
    }
)
_STUDY_STATUSES = {
    "submitted",
    "cancelling",
    "completed",
    "completed_with_failures",
    "failed",
    "cancelled",
}
_TERMINAL_STUDY_STATUSES = {
    "completed",
    "completed_with_failures",
    "failed",
    "cancelled",
}
_STAGE_DISPOSITIONS = {"pending", "completed", "failed", "cancelled", "skipped"}
_SKIP_REASONS = {
    "upstream_failed",
    "upstream_cancelled",
    "study_cancelled",
    "global_failure",
}
_EXECUTION_KEY_RE = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
_RFC3339_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\Z",
    re.ASCII,
)


def _as_record(document: CanonicalDocument) -> dict[str, object]:
    if not isinstance(document, Mapping):
        raise ValueError("canonical document must be a mapping")
    record = dict(document)
    _canonical_json_bytes(record)
    return record


def _documents_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return _canonical_json_bytes(dict(left)) == _canonical_json_bytes(dict(right))


def _serialize_record(record: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(record),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _read_record(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"invalid canonical document: {path}") from error
    try:
        value = _load_yaml(text)
    except _YamlError as error:
        raise ValueError(f"invalid canonical document: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"canonical document must be an object: {path}")
    _canonical_json_bytes(value)
    return value


def _fsync_parent_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_replace_record(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = _serialize_record(record)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _entity_parts(entity_id: object) -> tuple[str, str]:
    validated = _validate_typed_reference(entity_id)
    namespace, local_id = validated.split("/", 1)
    return namespace, local_id


def _canonical_path(
    *,
    data_root: Path,
    kind: EntityKind,
    entity_id: object,
) -> Path:
    namespace, local_id = _entity_parts(entity_id)
    domain = _DOMAINS.get(kind)
    if domain is None:
        raise ValueError("unsupported canonical write kind")
    namespace_root = data_root / namespace
    try:
        _read_namespace_document(data_root, namespace)
    except FileNotFoundError as error:
        raise ValueError(f"namespace is not canonical: {namespace}") from error
    return namespace_root / domain / f"{local_id}.yaml"


def _validate_document_identity(record: Mapping[str, object], entity_id: object) -> None:
    expected = _validate_typed_reference(entity_id)
    actual = record.get("id")
    if actual != expected:
        raise ValueError("canonical document identity mismatch")
    _validate_typed_reference(actual)


def _validate_slot(
    slot: object,
    *,
    expected_result: str,
    evaluation: bool,
) -> dict[str, object]:
    if type(slot) is not dict:
        raise ValueError("stage slot must be an object")
    expected_keys = {"disposition", "result", "reason"}
    if evaluation:
        expected_keys |= {"coordinate", "stage"}
    if set(slot) != expected_keys:
        raise ValueError("stage slot fields do not match schema")

    disposition = slot["disposition"]
    result = slot["result"]
    reason = slot["reason"]
    if disposition not in _STAGE_DISPOSITIONS:
        raise ValueError("invalid stage disposition")
    if disposition == "pending":
        if result is not None or reason is not None:
            raise ValueError("pending stage must not contain result or reason")
    elif disposition == "skipped":
        if result is not None or reason not in _SKIP_REASONS:
            raise ValueError("skipped stage requires one stable skip reason")
    else:
        if result != expected_result or reason is not None:
            raise ValueError("terminal child stage must reference its deterministic result")
        _validate_typed_reference(result)
    return slot


def _validate_created_at(value: object) -> None:
    if type(value) is not str or _RFC3339_UTC_RE.fullmatch(value) is None:
        raise ValueError("created_at must be RFC3339 UTC using Z")
    from datetime import datetime

    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("created_at must be RFC3339 UTC using Z") from error


def _validate_study_result_record(
    record: Mapping[str, object],
    *,
    entity_id: StudyResultId,
) -> dict[str, object]:
    expected_fields = {
        "schema",
        "id",
        "execution_key",
        "plan",
        "study",
        "source_commit",
        "backend",
        "created_at",
        "status",
        "diagnostic",
        "trials",
    }
    if set(record) != expected_fields:
        raise ValueError("StudyResult fields do not match schema")
    if record["schema"] != "mjtensu.mldb-v2/study-result/v1":
        raise ValueError("invalid StudyResult schema")
    _validate_document_identity(record, entity_id)

    namespace, local_id = _entity_parts(entity_id)
    execution_key = record["execution_key"]
    if type(execution_key) is not str or _EXECUTION_KEY_RE.fullmatch(execution_key) is None:
        raise ValueError("invalid StudyResult execution_key")
    parsed_execution_key = uuid.UUID(hex=execution_key)
    if parsed_execution_key.version != 4 or parsed_execution_key.variant != uuid.RFC_4122:
        raise ValueError("StudyResult execution_key must encode UUID4")
    if local_id != f"run-{execution_key}":
        raise ValueError("StudyResult id does not match execution_key")
    plan_namespace, _ = _entity_parts(record["plan"])
    study_namespace, _ = _entity_parts(record["study"])
    if plan_namespace != namespace or study_namespace != namespace:
        raise ValueError("StudyResult, Study, and Plan namespaces must match")
    if type(record["source_commit"]) is not str or _COMMIT_RE.fullmatch(record["source_commit"]) is None:
        raise ValueError("source_commit must be a full Git object id")
    if type(record["backend"]) is not str or not record["backend"]:
        raise ValueError("backend must be a non-empty generic backend name")
    _validate_created_at(record["created_at"])

    status = record["status"]
    if status not in _STUDY_STATUSES:
        raise ValueError("invalid StudyResult status")
    diagnostic = _validate_diagnostic(record["diagnostic"])
    trials = record["trials"]
    if type(trials) is not list:
        raise ValueError("StudyResult trials must be a list")

    dispositions: list[str] = []
    for trial_index, trial in enumerate(trials, start=1):
        if type(trial) is not dict or set(trial) != {"trial", "training", "evaluations"}:
            raise ValueError("StudyResult trial fields do not match schema")
        actual_trial_id = _validate_trial_id(trial["trial"])
        trial_id = f"trial-{trial_index:04d}"
        if actual_trial_id != trial_id:
            raise ValueError("StudyResult trial IDs must be contiguous and ordered")

        training = trial["training"]
        if training is not None:
            expected_training = f"{namespace}/run-{execution_key}-{trial_id}-train"
            validated_training = _validate_slot(
                training,
                expected_result=expected_training,
                evaluation=False,
            )
            dispositions.append(validated_training["disposition"])  # type: ignore[arg-type]

        evaluations = trial["evaluations"]
        if type(evaluations) is not list:
            raise ValueError("StudyResult evaluations must be a list")
        for evaluation_index, evaluation_slot in enumerate(evaluations, start=1):
            coordinate = f"eval-{evaluation_index:04d}"
            if type(evaluation_slot) is not dict:
                raise ValueError("evaluation slot must be an object")
            actual_coordinate = _validate_evaluation_coordinate_id(
                evaluation_slot.get("coordinate")
            )
            if actual_coordinate != coordinate:
                raise ValueError("evaluation coordinates must be contiguous and ordered")
            stage = evaluation_slot.get("stage")
            if type(stage) is not str or not stage:
                raise ValueError("evaluation stage must be a non-empty string")
            expected_evaluation = f"{namespace}/run-{execution_key}-{trial_id}-{coordinate}"
            validated_evaluation = _validate_slot(
                evaluation_slot,
                expected_result=expected_evaluation,
                evaluation=True,
            )
            dispositions.append(validated_evaluation["disposition"])  # type: ignore[arg-type]

    has_pending_stage = "pending" in dispositions
    if status in _TERMINAL_STUDY_STATUSES and has_pending_stage:
        raise ValueError("terminal StudyResult cannot contain pending stages")
    if status in {"submitted", "cancelling"} and not has_pending_stage:
        raise ValueError("non-terminal StudyResult requires at least one pending planned stage")
    skip_reasons = [
        slot["reason"]
        for slot in _iter_stage_slots(record)
        if slot["disposition"] == "skipped"
    ]
    if status == "submitted" and diagnostic is not None:
        raise ValueError("submitted StudyResult diagnostic must be null")
    if status == "completed" and any(item != "completed" for item in dispositions):
        raise ValueError("completed StudyResult requires every planned stage completed")
    if status == "completed_with_failures":
        if all(item == "completed" for item in dispositions):
            raise ValueError("completed_with_failures requires a non-completed stage")
        if "global_failure" in skip_reasons or "study_cancelled" in skip_reasons:
            raise ValueError("completed_with_failures cannot represent global failure or Study cancellation")
    if status in {"completed", "completed_with_failures"} and diagnostic is not None:
        raise ValueError("completed StudyResult diagnostic must be null")
    if status == "failed":
        if diagnostic is None:
            raise ValueError("failed StudyResult requires a diagnostic")
        if "global_failure" not in skip_reasons:
            raise ValueError("failed StudyResult requires a global_failure skip")
        if "study_cancelled" in skip_reasons:
            raise ValueError("failed StudyResult cannot represent Study cancellation")
    return dict(record)


def _iter_stage_slots(record: Mapping[str, object]):
    for trial in record["trials"]:  # type: ignore[index]
        training = trial["training"]
        if training is not None:
            yield training
        yield from trial["evaluations"]


def _validate_initial_study_result(record: Mapping[str, object]) -> None:
    if record["status"] != "submitted" or record["diagnostic"] is not None:
        raise ValueError("initial StudyResult must be submitted with null diagnostic")
    if any(slot["disposition"] != "pending" for slot in _iter_stage_slots(record)):
        raise ValueError("initial StudyResult stages must all be pending")


def _validate_study_result_transition(
    current: Mapping[str, object],
    replacement: Mapping[str, object],
) -> None:
    immutable_top = (
        "schema",
        "id",
        "execution_key",
        "plan",
        "study",
        "source_commit",
        "backend",
        "created_at",
    )
    for field in immutable_top:
        if current[field] != replacement[field]:
            raise ValueError(f"StudyResult immutable field changed: {field}")

    current_status = current["status"]
    replacement_status = replacement["status"]
    allowed_status = {
        "submitted": {
            "submitted",
            "cancelling",
            "completed",
            "completed_with_failures",
            "failed",
        },
        "cancelling": {"cancelling", "cancelled"},
    }
    if replacement_status not in allowed_status.get(current_status, set()):
        raise ValueError("invalid StudyResult status transition")

    current_trials = current["trials"]
    replacement_trials = replacement["trials"]
    if len(current_trials) != len(replacement_trials):
        raise ValueError("StudyResult trial topology is immutable")

    for current_trial, replacement_trial in zip(current_trials, replacement_trials):
        if current_trial["trial"] != replacement_trial["trial"]:
            raise ValueError("StudyResult trial identity is immutable")
        if (current_trial["training"] is None) != (replacement_trial["training"] is None):
            raise ValueError("StudyResult training topology is immutable")
        current_evaluations = current_trial["evaluations"]
        replacement_evaluations = replacement_trial["evaluations"]
        if len(current_evaluations) != len(replacement_evaluations):
            raise ValueError("StudyResult evaluation topology is immutable")

        current_slots: list[Mapping[str, object]] = []
        replacement_slots: list[Mapping[str, object]] = []
        if current_trial["training"] is not None:
            current_slots.append(current_trial["training"])
            replacement_slots.append(replacement_trial["training"])

        for current_eval, replacement_eval in zip(
            current_evaluations,
            replacement_evaluations,
        ):
            if current_eval["coordinate"] != replacement_eval["coordinate"]:
                raise ValueError("StudyResult evaluation coordinate is immutable")
            if current_eval["stage"] != replacement_eval["stage"]:
                raise ValueError("StudyResult evaluation stage is immutable")
            current_slots.append(current_eval)
            replacement_slots.append(replacement_eval)

        for current_slot, replacement_slot in zip(current_slots, replacement_slots):
            current_disposition = current_slot["disposition"]
            if current_disposition == "pending":
                continue
            if not _documents_equal(current_slot, replacement_slot):
                raise ValueError("terminal stage disposition is immutable")


class CanonicalRepositoryWriter:
    """Validated atomic canonical writes with frozen idempotence/lifecycle semantics."""

    def __init__(
        self,
        repository_root: str | os.PathLike[str],
        *,
        record_validator: CanonicalRecordValidator,
    ) -> None:
        if record_validator is None:
            raise TypeError("record_validator is required")
        self._repository_root = Path(repository_root)
        self._data_root = self._repository_root / "mldb_data"
        self._record_validator = record_validator
        self._write_lock_root = (
            self._repository_root / ".local" / "mldb_v2" / "canonical_write_locks"
        )

    def create_immutable(
        self,
        *,
        kind: ImmutableCanonicalKind,
        entity_id: ImmutableCanonicalId,
        document: CanonicalDocument,
    ) -> CanonicalDocument:
        if kind not in _IMMUTABLE_KINDS:
            raise ValueError("kind is not an immutable canonical kind")
        record = _as_record(document)
        _validate_document_identity(record, entity_id)
        path = _canonical_path(data_root=self._data_root, kind=kind, entity_id=entity_id)
        self._record_validator.validate(
            kind=kind,
            entity_id=entity_id,
            document=document,
        )
        return self._create_idempotent(
            kind=kind,
            entity_id=entity_id,
            path=path,
            record=record,
        )

    def create_study_result(
        self,
        *,
        entity_id: StudyResultId,
        document: CanonicalDocument,
    ) -> CanonicalDocument:
        record = _as_record(document)
        validated = _validate_study_result_record(record, entity_id=entity_id)
        path = _canonical_path(
            data_root=self._data_root,
            kind=EntityKind.STUDY_RESULT,
            entity_id=entity_id,
        )
        key = str(path.relative_to(self._repository_root))
        with _process_file_lock(lock_root=self._write_lock_root, key=key):
            if path.exists():
                existing = _read_record(path)
                if _documents_equal(existing, validated):
                    return existing
                raise ValueError("lifecycle conflict: StudyResult identity already exists")
            _validate_initial_study_result(validated)
            _atomic_replace_record(path, validated)
            return validated

    def replace_nonterminal_study_result(
        self,
        *,
        entity_id: StudyResultId,
        replacement: CanonicalDocument,
    ) -> CanonicalDocument:
        replacement_record = _as_record(replacement)
        validated_replacement = _validate_study_result_record(
            replacement_record,
            entity_id=entity_id,
        )
        path = _canonical_path(
            data_root=self._data_root,
            kind=EntityKind.STUDY_RESULT,
            entity_id=entity_id,
        )
        key = str(path.relative_to(self._repository_root))
        with _process_file_lock(lock_root=self._write_lock_root, key=key):
            if not path.exists():
                raise FileNotFoundError(path)
            current = _read_record(path)
            _validate_study_result_record(current, entity_id=entity_id)

            if _documents_equal(current, validated_replacement):
                return current
            if current["status"] in _TERMINAL_STUDY_STATUSES:
                raise ValueError("lifecycle conflict: terminal StudyResult is immutable")

            _validate_study_result_transition(current, validated_replacement)
            _atomic_replace_record(path, validated_replacement)
            return validated_replacement

    def _create_idempotent(
        self,
        *,
        kind: ImmutableCanonicalKind,
        entity_id: ImmutableCanonicalId,
        path: Path,
        record: Mapping[str, object],
    ) -> CanonicalDocument:
        key = str(path.relative_to(self._repository_root))
        with _process_file_lock(lock_root=self._write_lock_root, key=key):
            if path.exists():
                existing = _read_record(path)
                _validate_document_identity(existing, entity_id)
                self._record_validator.validate(
                    kind=kind,
                    entity_id=entity_id,
                    document=existing,
                )
                if _documents_equal(existing, record):
                    return existing
                raise ValueError("lifecycle conflict: canonical identity already exists")
            _atomic_replace_record(path, record)
            return dict(record)
