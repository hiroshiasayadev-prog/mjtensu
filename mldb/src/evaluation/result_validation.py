"""Public Python signatures for MLDB Evaluation result acceptance.

Implementation of generic Evaluation Protocol result validation, registered v1
formal-artifact validation, artifact materialization, and accepted integrity metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from .interface import EvaluationResult, UnavailableOutput
from .protocol import EvaluationArtifactDeclaration, EvaluationOutputs

if TYPE_CHECKING:
    from ..runtime.catalog_handles import TaskHandle


@dataclass(frozen=True, slots=True)
class AcceptedEvaluationArtifact:
    """Persisted metadata for one accepted formal Evaluation Run artifact."""

    path: str
    format: Literal["jsonl", "csv", "json"]
    schema: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class EvaluationValidationIssue:
    """One non-fatal persisted Evaluation result-validation issue."""

    output: str
    type: str
    message: str


@dataclass(frozen=True, slots=True)
class AcceptedEvaluationResult:
    """Persisted-ready formal result accepted for one Evaluation Run."""

    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, AcceptedEvaluationArtifact]
    unavailable_outputs: Sequence[UnavailableOutput]
    validation_issues: Sequence[EvaluationValidationIssue]


@dataclass(frozen=True, slots=True)
class _ArtifactCandidate:
    key: str
    declaration: EvaluationArtifactDeclaration
    source: Path
    relative_path: Path


class _ArtifactInvalid(ValueError):
    def __init__(self, issue_type: str, message: str) -> None:
        super().__init__(message)
        self.issue_type = issue_type
        self.message = message


_CATEGORICAL_PREDICTIONS_SCHEMA = (
    "mjtensu.mldb/eval-artifact/categorical-predictions/v1"
)
_CONFUSION_MATRIX_SCHEMA = "mjtensu.mldb/eval-artifact/confusion-matrix/v1"
_SUPPORTED_FORMATS = frozenset({"jsonl", "csv", "json"})


def _fail(message: str) -> None:
    raise ValueError(message)


def _is_metric_scalar(value: object) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _parse_output_reference(reference: object) -> tuple[str, str]:
    if not isinstance(reference, str):
        _fail("Unavailable output reference must be a string.")
    namespace, separator, key = reference.partition(".")
    if separator != "." or namespace not in {"metrics", "artifacts"} or not key:
        _fail(
            "Unavailable output must use 'metrics.<key>' or 'artifacts.<key>'."
        )
    return namespace, key


def _validate_unavailable_outputs(
    outputs: EvaluationOutputs,
    result: EvaluationResult,
) -> tuple[UnavailableOutput, ...]:
    unavailable = result.unavailable_outputs
    if isinstance(unavailable, (str, bytes)) or not isinstance(unavailable, Sequence):
        _fail("EvaluationResult.unavailable_outputs must be a sequence.")

    seen: set[str] = set()
    accepted: list[UnavailableOutput] = []
    for item in unavailable:
        if not isinstance(item, UnavailableOutput):
            _fail("Every unavailable output must be an UnavailableOutput value.")
        if not isinstance(item.type, str) or not item.type.strip():
            _fail("Unavailable output type must be a non-empty string.")
        if not isinstance(item.message, str) or not item.message.strip():
            _fail("Unavailable output message must be a non-empty string.")

        namespace, key = _parse_output_reference(item.output)
        reference = f"{namespace}.{key}"
        if reference in seen:
            _fail(f"Unavailable output '{reference}' was reported more than once.")
        seen.add(reference)

        declarations = outputs.metrics if namespace == "metrics" else outputs.artifacts
        if key not in declarations:
            _fail(f"Unavailable output '{reference}' is not declared by the protocol.")

        returned = result.metrics if namespace == "metrics" else result.artifacts
        if isinstance(returned, Mapping) and key in returned:
            _fail(f"Output '{reference}' cannot be returned and unavailable together.")

        if namespace == "artifacts" and outputs.artifacts[key].required:
            _fail(f"Required artifact '{key}' cannot be reported unavailable.")

        accepted.append(item)

    return tuple(accepted)


def _validate_metrics(
    outputs: EvaluationOutputs,
    result: EvaluationResult,
    unavailable_refs: set[str],
) -> dict[str, int | float]:
    if not isinstance(result.metrics, Mapping):
        _fail("EvaluationResult.metrics must be a mapping.")

    accepted: dict[str, int | float] = {}
    for key, value in result.metrics.items():
        if not isinstance(key, str) or not key:
            _fail("Returned metric keys must be non-empty strings.")
        if key not in outputs.metrics:
            _fail(f"Returned metric '{key}' is not declared by the protocol.")
        if not _is_metric_scalar(value):
            _fail(f"Returned metric '{key}' must be a finite int or float, not bool.")
        accepted[key] = value

    for key in outputs.metrics:
        returned = key in result.metrics
        unavailable = f"metrics.{key}" in unavailable_refs
        if returned == unavailable:
            if returned:
                _fail(f"Metric '{key}' cannot be returned and unavailable together.")
            _fail(
                f"Declared metric '{key}' must be returned or explicitly unavailable."
            )

    return accepted


def _resolve_candidate_path(candidate: object, work_root: Path) -> tuple[Path, Path]:
    if not isinstance(candidate, Path):
        raise _ArtifactInvalid(
            "artifact-path-invalid",
            "Returned artifact path must be a pathlib.Path.",
        )

    unresolved = candidate if candidate.is_absolute() else work_root / candidate
    try:
        resolved = unresolved.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _ArtifactInvalid(
            "artifact-path-invalid",
            f"Returned artifact path cannot be resolved: {exc}",
        ) from exc

    try:
        relative = resolved.relative_to(work_root)
    except ValueError as exc:
        raise _ArtifactInvalid(
            "artifact-path-invalid",
            "Returned artifact path escapes the Evaluation Run work directory.",
        ) from exc

    if relative == Path(".") or not resolved.is_file():
        raise _ArtifactInvalid(
            "artifact-path-invalid",
            "Returned artifact path must identify a file beneath the work directory.",
        )

    return resolved, relative


def _validate_jsonl_format(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    raise _ArtifactInvalid(
                        "format-validation-failed",
                        f"Invalid JSONL at line {line_number}: {exc.msg}.",
                    ) from exc
    except UnicodeDecodeError as exc:
        raise _ArtifactInvalid(
            "format-validation-failed",
            "JSONL artifact is not valid UTF-8.",
        ) from exc


def _validate_csv_format(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            for _ in reader:
                pass
    except UnicodeDecodeError as exc:
        raise _ArtifactInvalid(
            "format-validation-failed",
            "CSV artifact is not valid UTF-8.",
        ) from exc
    except csv.Error as exc:
        raise _ArtifactInvalid(
            "format-validation-failed",
            f"CSV artifact is malformed: {exc}.",
        ) from exc


def _validate_json_format(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            json.load(stream)
    except UnicodeDecodeError as exc:
        raise _ArtifactInvalid(
            "format-validation-failed",
            "JSON artifact is not valid UTF-8.",
        ) from exc
    except json.JSONDecodeError as exc:
        raise _ArtifactInvalid(
            "format-validation-failed",
            f"JSON artifact is malformed: {exc.msg}.",
        ) from exc


def _validate_declared_format(path: Path, artifact_format: str) -> None:
    if artifact_format == "jsonl":
        _validate_jsonl_format(path)
    elif artifact_format == "csv":
        _validate_csv_format(path)
    elif artifact_format == "json":
        _validate_json_format(path)
    else:
        raise _ArtifactInvalid(
            "format-validation-failed",
            f"Unsupported formal artifact format '{artifact_format}'.",
        )


def _task_labels(task: TaskHandle) -> frozenset[str]:
    try:
        labels = task.metadata.target.labels
    except AttributeError as exc:
        raise _ArtifactInvalid(
            "schema-validation-failed",
            "Categorical artifact validation requires Task categorical labels.",
        ) from exc
    if not isinstance(labels, tuple) or not all(isinstance(label, str) for label in labels):
        raise _ArtifactInvalid(
            "schema-validation-failed",
            "Task categorical labels are not available in the expected validated shape.",
        )
    return frozenset(labels)


def _validate_categorical_predictions(path: Path, task: TaskHandle) -> None:
    labels = _task_labels(task)
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise _ArtifactInvalid(
                        "schema-validation-failed",
                        f"Categorical predictions line {line_number} is not valid JSON.",
                    ) from exc
                if not isinstance(row, dict):
                    raise _ArtifactInvalid(
                        "schema-validation-failed",
                        f"Categorical predictions line {line_number} must be an object.",
                    )
                for field in ("sample_id", "target", "prediction"):
                    if field not in row or not isinstance(row[field], str):
                        raise _ArtifactInvalid(
                            "schema-validation-failed",
                            f"Categorical predictions line {line_number} requires string '{field}'.",
                        )
                if row["target"] not in labels:
                    raise _ArtifactInvalid(
                        "schema-validation-failed",
                        f"Categorical predictions line {line_number} has unknown target label.",
                    )
                if row["prediction"] not in labels:
                    raise _ArtifactInvalid(
                        "schema-validation-failed",
                        f"Categorical predictions line {line_number} has unknown prediction label.",
                    )
    except UnicodeDecodeError as exc:
        raise _ArtifactInvalid(
            "schema-validation-failed",
            "Categorical predictions artifact is not valid UTF-8.",
        ) from exc


def _parse_non_negative_integer(raw: object) -> int:
    if not isinstance(raw, str):
        raise ValueError
    stripped = raw.strip()
    if not stripped or not stripped.isascii() or not stripped.isdigit():
        raise ValueError
    return int(stripped, 10)


def _validate_confusion_matrix(path: Path, task: TaskHandle) -> None:
    labels = _task_labels(task)
    seen_pairs: set[tuple[str, str]] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            fieldnames = reader.fieldnames
            if fieldnames is None or not {"target", "prediction", "count"}.issubset(
                fieldnames
            ):
                raise _ArtifactInvalid(
                    "schema-validation-failed",
                    "Confusion matrix requires target, prediction, and count columns.",
                )

            for row_number, row in enumerate(reader, start=2):
                target = row.get("target")
                prediction = row.get("prediction")
                count = row.get("count")
                if target not in labels or prediction not in labels:
                    raise _ArtifactInvalid(
                        "schema-validation-failed",
                        f"Confusion matrix row {row_number} uses a label outside the Task label set.",
                    )
                try:
                    _parse_non_negative_integer(count)
                except (TypeError, ValueError) as exc:
                    raise _ArtifactInvalid(
                        "schema-validation-failed",
                        f"Confusion matrix row {row_number} count must be a non-negative integer.",
                    ) from exc
                pair = (target, prediction)
                if pair in seen_pairs:
                    raise _ArtifactInvalid(
                        "schema-validation-failed",
                        f"Confusion matrix row {row_number} duplicates a target/prediction pair.",
                    )
                seen_pairs.add(pair)
    except UnicodeDecodeError as exc:
        raise _ArtifactInvalid(
            "schema-validation-failed",
            "Confusion matrix artifact is not valid UTF-8.",
        ) from exc
    except csv.Error as exc:
        raise _ArtifactInvalid(
            "schema-validation-failed",
            f"Confusion matrix CSV is malformed: {exc}.",
        ) from exc


def _validate_registered_schema(
    path: Path,
    declaration: EvaluationArtifactDeclaration,
    task: TaskHandle,
) -> None:
    if declaration.schema == _CATEGORICAL_PREDICTIONS_SCHEMA:
        if declaration.format != "jsonl":
            raise _ArtifactInvalid(
                "schema-validation-failed",
                "Categorical predictions v1 requires declared format jsonl.",
            )
        _validate_categorical_predictions(path, task)
        return

    if declaration.schema == _CONFUSION_MATRIX_SCHEMA:
        if declaration.format != "csv":
            raise _ArtifactInvalid(
                "schema-validation-failed",
                "Confusion matrix v1 requires declared format csv.",
            )
        _validate_confusion_matrix(path, task)
        return

    raise _ArtifactInvalid(
        "unsupported-artifact-schema",
        f"No registered formal validator exists for schema '{declaration.schema}'.",
    )


def _validate_artifact_candidate(
    key: str,
    declaration: EvaluationArtifactDeclaration,
    candidate: object,
    task: TaskHandle,
    work_root: Path,
) -> _ArtifactCandidate:
    if declaration.format not in _SUPPORTED_FORMATS:
        raise _ArtifactInvalid(
            "format-validation-failed",
            f"Declared format '{declaration.format}' is not supported by Evaluation v1.",
        )

    source, relative = _resolve_candidate_path(candidate, work_root)
    try:
        _validate_declared_format(source, declaration.format)
        _validate_registered_schema(source, declaration, task)
    except OSError as exc:
        raise _ArtifactInvalid(
            "artifact-read-failed",
            f"Artifact could not be read for validation: {exc}",
        ) from exc

    return _ArtifactCandidate(
        key=key,
        declaration=declaration,
        source=source,
        relative_path=relative,
    )


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _run_relative_artifact_path(relative: Path) -> str:
    return PurePosixPath("artifacts", *relative.parts).as_posix()


def _materialize_artifacts(
    candidates: list[_ArtifactCandidate],
    artifacts_root: Path,
) -> dict[str, AcceptedEvaluationArtifact]:
    accepted: dict[str, AcceptedEvaluationArtifact] = {}
    created: list[Path] = []
    materialized: dict[Path, tuple[Path, str, int]] = {}

    try:
        artifacts_root.mkdir(parents=True, exist_ok=True)
        artifacts_root = artifacts_root.resolve(strict=True)

        for candidate in candidates:
            source_digest, source_bytes = _hash_file(candidate.source)
            cached = materialized.get(candidate.source)
            if cached is None:
                destination = artifacts_root.joinpath(*candidate.relative_path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    _fail(
                        f"Formal artifact destination already exists: {destination}."
                    )
                created.append(destination)
                shutil.copyfile(candidate.source, destination)
                destination_digest, destination_bytes = _hash_file(destination)
                if (
                    destination_digest != source_digest
                    or destination_bytes != source_bytes
                ):
                    _fail(
                        f"Formal artifact materialization changed bytes for '{candidate.key}'."
                    )
                materialized[candidate.source] = (
                    destination,
                    destination_digest,
                    destination_bytes,
                )
            else:
                destination, destination_digest, destination_bytes = cached
                if destination_digest != source_digest or destination_bytes != source_bytes:
                    _fail(
                        f"Artifact source changed during materialization for '{candidate.key}'."
                    )

            accepted[candidate.key] = AcceptedEvaluationArtifact(
                path=_run_relative_artifact_path(candidate.relative_path),
                format=candidate.declaration.format,
                schema=candidate.declaration.schema,
                sha256=destination_digest,
                bytes=destination_bytes,
            )
    except Exception:
        for path in reversed(created):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    return accepted


def accept_evaluation_result(
    outputs: EvaluationOutputs,
    result: EvaluationResult,
    task: TaskHandle,
    work_dir: Path,
    artifacts_dir: Path,
) -> AcceptedEvaluationResult:
    """Validate and materialize one candidate Evaluation Protocol result."""

    if not isinstance(outputs.metrics, Mapping) or not isinstance(
        outputs.artifacts, Mapping
    ):
        _fail("Evaluation outputs must contain metric and artifact mappings.")
    if not isinstance(result, EvaluationResult):
        _fail("Evaluation entrypoint did not return EvaluationResult.")
    if not isinstance(result.artifacts, Mapping):
        _fail("EvaluationResult.artifacts must be a mapping.")

    try:
        work_root = work_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Evaluation work directory cannot be resolved: {exc}") from exc
    if not work_root.is_dir():
        _fail("Evaluation work_dir must identify a directory.")

    unavailable_outputs = _validate_unavailable_outputs(outputs, result)
    unavailable_refs = {item.output for item in unavailable_outputs}
    accepted_metrics = _validate_metrics(outputs, result, unavailable_refs)

    for key in result.artifacts:
        if not isinstance(key, str) or not key:
            _fail("Returned artifact keys must be non-empty strings.")
        if key not in outputs.artifacts:
            _fail(f"Returned artifact '{key}' is not declared by the protocol.")

    for key, declaration in outputs.artifacts.items():
        returned = key in result.artifacts
        unavailable = f"artifacts.{key}" in unavailable_refs
        if declaration.required and not returned:
            if unavailable:
                _fail(f"Required artifact '{key}' cannot be unavailable.")
            _fail(f"Required artifact '{key}' was not returned.")

    valid_candidates: list[_ArtifactCandidate] = []
    validation_issues: list[EvaluationValidationIssue] = []
    for key, candidate_path in result.artifacts.items():
        declaration = outputs.artifacts[key]
        try:
            candidate = _validate_artifact_candidate(
                key,
                declaration,
                candidate_path,
                task,
                work_root,
            )
        except _ArtifactInvalid as exc:
            if declaration.required:
                _fail(f"Required artifact '{key}' is invalid: {exc.message}")
            validation_issues.append(
                EvaluationValidationIssue(
                    output=f"artifacts.{key}",
                    type=exc.issue_type,
                    message=exc.message,
                )
            )
            continue
        valid_candidates.append(candidate)

    is_partial = bool(unavailable_outputs or validation_issues)
    if is_partial and not accepted_metrics and not valid_candidates:
        _fail("A partial Evaluation result must retain at least one trusted formal output.")

    accepted_artifacts = _materialize_artifacts(valid_candidates, artifacts_dir)

    return AcceptedEvaluationResult(
        metrics=dict(accepted_metrics),
        artifacts=accepted_artifacts,
        unavailable_outputs=tuple(unavailable_outputs),
        validation_issues=tuple(validation_issues),
    )
