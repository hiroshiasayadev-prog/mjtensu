"""Worker-local domain execution boundary for one concrete MLDB attempt.

This skeleton composes the frozen Worker assignment contract with the existing runtime
Handle, Train/Evaluation Context, executable-loader, Model-loader, and canonical-weight
boundaries. It starts only after all required assignment byte objects have already been
retrieved, integrity-verified, and materialized at Worker-local filesystem paths.

The module owns no Worker API communication, Queue interaction, lease/heartbeat logic,
Controller persistence, canonical Run mutation, candidate upload, retry decision, cache
policy, or Run-ID allocation. Worker-local paths represented here are execution
resources only and must never be interpreted as canonical Controller repository paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..evaluation.interface import EvaluationContext, EvaluationResult
from ..model.loading import ModelHandle
from ..runtime.catalog_handles import ArchitectureHandle, CorpusHandle, TaskHandle
from ..runtime.definition_handles import EvaluationProtocolHandle, TrainProtocolHandle
from ..runtime.executable_loader import (
    load_architecture_build,
    load_evaluation_entrypoint,
    load_train_entrypoint,
)
from ..training.protocol import TrainContext
from ..training.weights import (
    CanonicalWeightsArtifact,
    accept_trained_state,
    serialize_canonical_weights,
)
from .worker_api import EvaluationAssignment, TrainingAssignment


@dataclass(frozen=True, slots=True)
class TrainingExecutionFiles:
    """Already verified Worker-local files required for one Training attempt.

    ``corpus_artifact``, ``architecture_implementation``, and
    ``train_protocol_implementation`` are local materializations of the exact immutable
    bytes selected by the corresponding :class:`TrainingAssignment` descriptors after
    required integrity verification has already succeeded.

    ``work_dir`` is the protocol-owned execution working directory supplied through the
    existing ``TrainContext``. ``weights_candidate_path`` is the concrete Worker-local
    destination at which the accepted generic learned state is serialized for later
    candidate upload. Neither path is a canonical Training Run repository location.

    This value does not describe download, cache, repository, or remote-resource policy.
    Directory preparation and assignment-byte verification happen before this boundary.
    """

    corpus_artifact: Path
    architecture_implementation: Path
    train_protocol_implementation: Path
    work_dir: Path
    weights_candidate_path: Path


@dataclass(frozen=True, slots=True)
class EvaluationExecutionFiles:
    """Already verified Worker-local files required for one Evaluation attempt.

    ``corpus_artifact``, ``model_architecture_implementation``, ``model_weights``, and
    ``evaluation_protocol_implementation`` are local materializations of the exact
    immutable bytes selected by the corresponding :class:`EvaluationAssignment`
    descriptors after required integrity verification has already succeeded.

    ``work_dir`` is supplied unchanged through the existing ``EvaluationContext`` and
    owns protocol-local temporary, diagnostic, and candidate formal-result files. None
    of these paths is canonical Controller repository provenance.

    The Corpus builder is intentionally absent because distributed execution consumes
    the registered immutable Corpus artifact rather than its authoring/materialization
    source.
    """

    corpus_artifact: Path
    model_architecture_implementation: Path
    model_weights: Path
    evaluation_protocol_implementation: Path
    work_dir: Path


@dataclass(frozen=True, slots=True)
class TrainingExecutionCandidate:
    """Execution-local learned-weight candidate produced before Worker API upload.

    ``local_weights_path`` is the Worker-local file written by the frozen
    ``serialize_canonical_weights()`` operation. ``artifact`` is the exact
    :class:`CanonicalWeightsArtifact` metadata returned for those serialized bytes,
    including format, canonical Run-relative artifact name, SHA-256, and byte count.

    The presence of that Run-relative metadata does not make ``local_weights_path`` a
    canonical Training Run artifact. Controller acceptance, canonical placement,
    Training Run terminalization, and deterministic Model creation occur only after the
    later Worker API upload/report boundary.
    """

    local_weights_path: Path
    artifact: CanonicalWeightsArtifact


def execute_training_attempt(
    assignment: TrainingAssignment,
    files: TrainingExecutionFiles,
) -> TrainingExecutionCandidate:
    """Execute exactly one Training domain invocation from one frozen assignment.

    An implementation of this boundary reconstructs the existing runtime values exactly
    as follows, using no repository lookup and no synthetic metadata provenance paths:

    - ``TaskHandle(metadata=assignment.task, metadata_path=None)``;
    - ``CorpusHandle(metadata=assignment.corpus, metadata_path=None,
      artifact_path=files.corpus_artifact, builder_path=None)``;
    - ``ArchitectureHandle(metadata=assignment.architecture, metadata_path=None,
      implementation_path=files.architecture_implementation)``;
    - ``TrainProtocolHandle(metadata=assignment.protocol, metadata_path=None,
      implementation_path=files.train_protocol_implementation)``.

    Those handles form the frozen ``TrainContext`` with ``assignment.seed``, the
    complete ``assignment.parameters`` mapping, and ``files.work_dir``. The implementation
    then reuses ``load_train_entrypoint()`` and ``load_architecture_build()``; it invokes
    the loaded Train Protocol exactly once for this attempt, passes the returned module
    through ``accept_trained_state()``, and serializes that accepted state exactly once
    with ``serialize_canonical_weights(..., files.weights_candidate_path)``.

    Successful serialization returns :class:`TrainingExecutionCandidate` containing the
    local candidate path and the exact artifact metadata produced by the frozen canonical
    weight serializer. This operation does not upload the file and does not construct
    ``TrainingSuccessCandidate`` or ``CandidateArtifactRef`` from ``worker_api``.

    Loader failure, context/Handle construction failure, Architecture build/state
    acceptance failure, ``train(context)`` exception, invalid trained-state return, or
    candidate serialization failure is an attempt execution failure. The exception is
    allowed to escape this boundary for the Worker loop to translate into ``AttemptFailed``;
    this function must not rerun ``train()`` for the same attempt.

    Cancellation control is intentionally not a parameter here. A later Worker loop may
    check cooperative cancellation immediately before and after this call/invocation,
    but this boundary defines no token framework and no generic forced-stop mechanism.
    """

    task = TaskHandle(metadata=assignment.task, metadata_path=None)
    corpus = CorpusHandle(
        metadata=assignment.corpus,
        metadata_path=None,
        artifact_path=files.corpus_artifact,
        builder_path=None,
    )
    architecture = ArchitectureHandle(
        metadata=assignment.architecture,
        metadata_path=None,
        implementation_path=files.architecture_implementation,
    )
    protocol = TrainProtocolHandle(
        metadata=assignment.protocol,
        metadata_path=None,
        implementation_path=files.train_protocol_implementation,
    )
    context = TrainContext(
        task=task,
        corpus=corpus,
        architecture=architecture,
        seed=assignment.seed,
        parameters=assignment.parameters,
        work_dir=files.work_dir,
    )

    train = load_train_entrypoint(protocol)
    architecture_build = load_architecture_build(architecture)
    trained_module = train(context)
    accepted_state = accept_trained_state(trained_module, architecture_build)
    artifact = serialize_canonical_weights(
        accepted_state,
        files.weights_candidate_path,
    )
    return TrainingExecutionCandidate(
        local_weights_path=files.weights_candidate_path,
        artifact=artifact,
    )


def execute_evaluation_attempt(
    assignment: EvaluationAssignment,
    files: EvaluationExecutionFiles,
) -> EvaluationResult:
    """Execute exactly one Evaluation domain invocation from one frozen assignment.

    An implementation reconstructs the existing runtime values with no repository lookup
    and no synthetic provenance paths:

    - ``TaskHandle(metadata=assignment.task, metadata_path=None)``;
    - ``CorpusHandle(metadata=assignment.corpus, metadata_path=None,
      artifact_path=files.corpus_artifact, builder_path=None)``;
    - ``ArchitectureHandle(metadata=assignment.model_architecture, metadata_path=None,
      implementation_path=files.model_architecture_implementation)``;
    - ``ModelHandle(metadata=assignment.model, metadata_path=None,
      training_run=assignment.training_run, training_run_metadata_path=None,
      weights_path=files.model_weights, architecture=<worker ArchitectureHandle>)``;
    - ``EvaluationProtocolHandle(metadata=assignment.protocol, metadata_path=None,
      implementation_path=files.evaluation_protocol_implementation)``.

    The reconstructed Task, Corpus, and Model handles form the existing frozen
    ``EvaluationContext`` with the complete ``assignment.parameters`` mapping and
    ``files.work_dir``. The implementation reuses ``load_evaluation_entrypoint()`` and
    invokes the loaded Evaluation Protocol exactly once for this attempt.

    Model loading is deliberately not duplicated or preloaded by this executor. Protocol
    code retains the frozen contract of calling ``load_model(context.model)`` when it
    needs the learned ``torch.nn.Module``; that common loader therefore continues to own
    Architecture loading, canonical weight loading, fresh model construction, and strict
    state application. Any such failure raised during ``evaluate(context)`` is an
    execution failure of this attempt.

    After the single successful protocol invocation, this Worker boundary establishes
    the frozen Evaluation-interface path invariant *before* any Worker API projection:
    every represented ``result.artifacts`` :class:`~pathlib.Path` must identify a path
    beneath the exact ``files.work_dir`` supplied as ``EvaluationContext.work_dir`` for
    this invocation. A returned path outside that work directory is an execution-local
    contract failure and the result must not be handed to Worker API projection.

    This is intentionally only the execution-local containment check whose evidence
    would otherwise disappear when Worker API projection replaces each local ``Path``
    with ``CandidateArtifactRef``. It does not perform declared-key agreement, scalar
    validation, unavailable-output semantics, required/optional artifact decisions,
    structured format/schema validation, canonical naming, artifact commit, or terminal
    status selection; those remain Controller-owned through the frozen
    ``accept_evaluation_result()`` contract.

    The Worker boundary does not add a separate regular-file existence requirement here.
    The frozen formal-acceptance contract still owns the candidate-file validation it
    already defines, and a later Worker upload/open operation naturally cannot transfer
    a missing or non-file candidate. This local repair adds no second formal artifact
    validator.

    Once containment succeeds, the exact same frozen :class:`EvaluationResult` object
    returned by the protocol is returned unchanged as the execution-local candidate;
    no replacement ``EvaluationResult`` is constructed. Its artifact ``Path`` values
    remain Worker-local candidate files until a later Worker loop uploads them and
    projects them into ``EvaluationSuccessCandidate``. This boundary performs no
    Controller formal-result acceptance or canonical import.

    Loader failure, context/Handle construction failure, Model-loading failure reached
    through protocol execution, ``evaluate(context)`` exception, or a return value that
    does not satisfy the frozen ``EvaluationResult`` boundary is an attempt execution
    failure. The exception is allowed to escape for later translation into
    ``AttemptFailed`` and this function must not rerun ``evaluate()`` for the same
    attempt.

    Cancellation control remains outside this function for the same reason as Training:
    the Worker loop may provide cooperative pre/post-invocation checks without freezing a
    generic cancellation-token or forced-termination mechanism here.
    """

    task = TaskHandle(metadata=assignment.task, metadata_path=None)
    corpus = CorpusHandle(
        metadata=assignment.corpus,
        metadata_path=None,
        artifact_path=files.corpus_artifact,
        builder_path=None,
    )
    architecture = ArchitectureHandle(
        metadata=assignment.model_architecture,
        metadata_path=None,
        implementation_path=files.model_architecture_implementation,
    )
    model = ModelHandle(
        metadata=assignment.model,
        metadata_path=None,
        training_run=assignment.training_run,
        training_run_metadata_path=None,
        weights_path=files.model_weights,
        architecture=architecture,
    )
    protocol = EvaluationProtocolHandle(
        metadata=assignment.protocol,
        metadata_path=None,
        implementation_path=files.evaluation_protocol_implementation,
    )
    context = EvaluationContext(
        task=task,
        corpus=corpus,
        model=model,
        parameters=assignment.parameters,
        work_dir=files.work_dir,
    )

    evaluate = load_evaluation_entrypoint(protocol)
    result = evaluate(context)
    if not isinstance(result, EvaluationResult):
        raise TypeError("evaluate(context) must return EvaluationResult")

    work_dir = files.work_dir.resolve()
    for artifact_path in result.artifacts.values():
        if not isinstance(artifact_path, Path):
            raise TypeError("EvaluationResult artifact paths must be pathlib.Path values")
        resolved_artifact = artifact_path.resolve()
        if resolved_artifact == work_dir or not resolved_artifact.is_relative_to(work_dir):
            raise ValueError("EvaluationResult artifact path must be beneath work_dir")

    return result
