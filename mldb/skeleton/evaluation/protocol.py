"""Public Python signatures for MLDB Evaluation Protocol metadata.

This skeleton fixes the reusable Evaluation Protocol v1 definition, its declared
formal metric/artifact output surface, and draft/sealed metadata lifecycle. It
intentionally does not define EvaluationContext, EvaluationResult, the ``evaluate``
callable type, Model/Corpus loading, executable loading, sealing mutation, Evaluation
Run lifecycle, result acceptance, artifact-file validation, or queue/worker behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Mapping

from ..common.errors import ValidationReport
from ..common.ids import EvaluationProtocolId, TaskId
from ..common.parameters import PublicParameterDeclarations


class EvaluationProtocolStatus(str, Enum):
    """Persisted Evaluation Protocol lifecycle state."""

    DRAFT = "draft"
    SEALED = "sealed"


@dataclass(frozen=True, slots=True)
class EvaluationProtocolImplementation:
    """Evaluation Protocol v1 executable-implementation identity metadata.

    ``entrypoint`` is fixed to ``"evaluate"`` by the v1 contract. ``sha256``
    identifies the exact sibling Evaluation Protocol Python bytes. It is required
    when the containing protocol is ``SEALED`` and may be omitted while the protocol
    is ``DRAFT``.

    Resolving the sibling path, calculating or comparing hashes, importing the
    implementation, validating callable shape, and exposing the callable are runtime
    or sealing responsibilities rather than behavior of this metadata value.
    """

    entrypoint: Literal["evaluate"]
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationMetricDeclaration:
    """One protocol-local declared scalar metric.

    Evaluation Protocol v1 formal metrics use only ``type: number``. The metric key
    lives in :class:`EvaluationOutputs.metrics` and has meaning only within the
    containing protocol; this declaration does not introduce a universal metric
    ontology. ``description`` is optional human-readable metadata.

    Concrete Run values, finite-number validation, explicit unavailability, and
    result acceptance belong to the evaluation result-validation boundary rather
    than this declaration type.
    """

    type: Literal["number"]
    description: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationArtifactDeclaration:
    """One declared formal structured Evaluation Protocol artifact.

    ``format`` is restricted to the three Evaluation Protocol v1 structured formats.
    ``schema`` is the versioned formal artifact schema identifier. ``required``
    declares whether a successful concrete evaluation must provide a valid artifact
    under this protocol-local key.

    Candidate/accepted file paths, SHA-256 values, byte sizes, work-directory
    containment, file-format validation, schema-content validation, and import into
    Evaluation Run-owned storage are concrete Run/result concerns and are deliberately
    absent from this declaration.
    """

    format: Literal["jsonl", "csv", "json"]
    schema: str
    required: bool


@dataclass(frozen=True, slots=True)
class EvaluationOutputs:
    """Declared formal output surface of one Evaluation Protocol.

    Both mappings are required persisted fields but may be empty. Keys are local to
    the containing protocol. ``metrics`` contains only scalar metric declarations;
    structured information belongs under ``artifacts`` through explicit versioned
    artifact declarations.

    The wrapper mirrors the persisted ``outputs.metrics`` / ``outputs.artifacts``
    shape without introducing a generic output hierarchy.
    """

    metrics: Mapping[str, EvaluationMetricDeclaration]
    artifacts: Mapping[str, EvaluationArtifactDeclaration]


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """One reusable post-training evaluation definition for exactly one Task.

    The protocol is permanently bound only to ``task``. A concrete Model and Corpus
    are selected later for one Evaluation Run and are not properties of this reusable
    definition.

    ``parameters`` reuses the shared Wave 0 public-parameter declaration contract
    directly. Its normalized declaration retains the required executable ``default``
    value and does not add an Evaluation-Protocol-specific generic type/range schema.

    ``outputs`` declares the complete formal scalar-metric and structured-artifact
    surface consumed later by generic result validation.
    """

    schema: Literal["mjtensu.mldb/evaluation-protocol/v1"]
    id: EvaluationProtocolId
    status: EvaluationProtocolStatus
    task: TaskId
    name: str
    description: str
    implementation: EvaluationProtocolImplementation
    parameters: PublicParameterDeclarations
    outputs: EvaluationOutputs


def validate_evaluation_protocol_metadata(
    protocol: EvaluationProtocol,
) -> ValidationReport:
    """Validate static Evaluation Protocol metadata invariants without external I/O.

    This validation owns only rules decidable from one normalized in-memory protocol:
    the v1 schema declaration, terminal positive-integer ``-vN`` ID grammar, lifecycle
    value and sealed/hash conditional, fixed ``evaluate`` entrypoint, validity of each
    represented public-parameter default in the shared JSON-compatible value domain,
    output mapping shape, ``type: number`` metric declarations, and artifact format,
    schema-identifier string metadata, and boolean requiredness metadata.

    Raw YAML parsing/normalization must reject malformed authored shapes that cannot be
    represented by these valid declaration types, such as a public parameter missing
    ``default`` or an artifact declaration missing ``format``, ``schema``, or
    ``required``. Empty ``parameters``, ``outputs.metrics``, and ``outputs.artifacts``
    mappings are valid.

    This function does not compare ``id`` with a filesystem basename, resolve the
    referenced Task, read or hash sibling Python bytes, import or inspect the sibling
    implementation, run sealing pytest, mutate lifecycle state, resolve Model/Corpus
    selections, validate Model/Corpus/Task compatibility, resolve caller overrides,
    validate concrete metric values or unavailable-output reports, inspect artifact
    files, accept/import result artifacts, allocate/finalize an Evaluation Run, or
    perform queue/worker behavior. Those checks belong to repository/runtime,
    executable loading, sealing, launch preflight, result validation, or Run lifecycle
    boundaries as applicable.
    """

    ...
