"""Public Python signatures for MLDB Evaluation result acceptance.

This skeleton fixes the boundary that converts one candidate
:class:`EvaluationResult` returned by an Evaluation Protocol into persisted-ready
formal Evaluation Run result data. It owns generic declaration/result validation,
registered v1 artifact-schema validation, accepted-artifact materialization, and
integrity metadata production.

It intentionally does not define Evaluation Protocol metadata or execution,
Evaluation Run identity/lifecycle/persistence, Model or Corpus loading, Study
lineage, queue/worker behavior, repository scanning, or protocol/model-family-
specific evaluation correctness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..runtime.catalog_handles import TaskHandle
from .interface import EvaluationResult, UnavailableOutput
from .protocol import EvaluationOutputs


@dataclass(frozen=True, slots=True)
class AcceptedEvaluationArtifact:
    """Persisted metadata for one accepted formal Evaluation Run artifact.

    ``path`` is the canonical Evaluation Run-relative path beneath ``artifacts/``.
    It is persisted path text, not the candidate protocol-produced
    :class:`~pathlib.Path` and not an absolute filesystem destination.

    ``format`` and ``schema`` are copied from the matched sealed Evaluation Protocol
    declaration after generic validation. ``sha256`` and ``bytes`` describe the
    exact immutable bytes materialized into Evaluation Run-owned artifact storage.

    This value is the canonical ``result.artifacts.<key>`` metadata shape that the
    later Evaluation Run skeleton should reuse rather than redefining artifact
    path/format/schema/integrity fields.
    """

    path: str
    format: Literal["jsonl", "csv", "json"]
    schema: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class EvaluationValidationIssue:
    """One non-fatal persisted Evaluation result-validation issue.

    ``output`` identifies the affected declared formal output using the persisted
    ``metrics.<key>`` / ``artifacts.<key>`` reference grammar. ``type`` is a short
    machine-readable issue identifier and ``message`` is concise human-readable
    detail.

    This is intentionally distinct from ``common.errors.ValidationIssue``. The
    common type represents static definition validation as ``code/message/path``;
    this type is an Evaluation Run historical incompleteness record with the
    persisted ``output/type/message`` contract.

    In an accepted result, validation issues represent rejected returned optional
    formal artifacts. Completion-critical validation problems do not appear here;
    they make result acceptance fail instead.
    """

    output: str
    type: str
    message: str


@dataclass(frozen=True, slots=True)
class AcceptedEvaluationResult:
    """Persisted-ready formal result accepted for one Evaluation Run.

    ``metrics`` contains only declared finite ``int``/``float`` values; booleans are
    invalid even though ``bool`` is a Python ``int`` subtype. Explicitly unavailable
    metrics never receive ``None``, NaN, or another scalar placeholder.

    ``artifacts`` contains only materialized formal artifacts represented by
    :class:`AcceptedEvaluationArtifact`; candidate work-file paths never cross this
    accepted boundary.

    ``unavailable_outputs`` reuses the frozen evaluation-interface contract exactly.
    Only declared non-completion-critical unavailability can survive acceptance:
    unavailable metrics and unavailable optional artifacts. Required-artifact
    unavailability is an operation failure.

    ``validation_issues`` contains non-fatal rejected returned optional artifacts.

    Later Evaluation Run code can derive completion without another public
    dependency: the result is partial exactly when ``unavailable_outputs`` or
    ``validation_issues`` is non-empty, otherwise it is complete. Acceptance also
    enforces that a partial result retains at least one accepted metric or artifact;
    a partial reason with no accepted formal output is an operation failure.
    """

    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, AcceptedEvaluationArtifact]
    unavailable_outputs: Sequence[UnavailableOutput]
    validation_issues: Sequence[EvaluationValidationIssue]


def accept_evaluation_result(
    outputs: EvaluationOutputs,
    result: EvaluationResult,
    task: TaskHandle,
    work_dir: Path,
    artifacts_dir: Path,
) -> AcceptedEvaluationResult:
    """Validate and materialize one candidate Evaluation Protocol result.

    ``outputs`` is the sealed protocol's declared formal output surface and ``result``
    is the candidate value returned by its ``evaluate`` entrypoint. ``task`` supplies
    the selected Task semantic contract needed by the registered initial categorical
    artifact schemas, including validation of ``target`` and ``prediction`` labels.
    No Corpus handle is required by the current schemas: categorical-predictions v1
    treats ``sample_id`` as Corpus-local identity but does not require generic sample
    existence, uniqueness, ordering, or full-Corpus coverage validation.

    ``work_dir`` is the concrete Evaluation Run ``work/`` directory used to verify
    that every returned candidate artifact identifies a file within protocol-owned
    working space. ``artifacts_dir`` is the already-resolved Evaluation Run
    ``artifacts/`` destination into which accepted bytes are materialized. These two
    paths are sufficient for this boundary; repository objects, Model, Training Run,
    Study, and orchestration state are deliberately not inputs.

    Successful acceptance enforces the generic result contract, including declared
    key agreement, finite scalar values with booleans rejected, exactly-one declared
    metric outcome, unavailable-output reference/conflict rules, required/optional
    artifact semantics, candidate work-file containment, declared format checks, and
    registered formal schema validation. The initial registered schemas are
    ``mjtensu.mldb/eval-artifact/categorical-predictions/v1`` and
    ``mjtensu.mldb/eval-artifact/confusion-matrix/v1``; schema-specific validators
    remain private implementation details rather than public registry APIs.

    Every valid returned artifact is materialized into ``artifacts_dir`` and returned
    as :class:`AcceptedEvaluationArtifact` with a canonical Run-relative ``artifacts/``
    path plus SHA-256 and byte-size metadata for the exact accepted bytes. A rejected
    optional returned artifact is omitted and recorded as
    :class:`EvaluationValidationIssue`; an optional artifact silently omitted is valid
    and does not by itself make the result partial.

    Any completion-critical violation or materialization failure is an operation
    failure. This skeleton intentionally does not introduce a Result monad or a new
    public exception hierarchy. If partial reasons remain after otherwise successful
    validation, at least one accepted metric or artifact must remain or the operation
    fails instead of producing an empty partial result.
    """

    ...
