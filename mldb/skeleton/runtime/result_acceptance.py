"""Controller-side acceptance of successful Worker result candidates.

This module begins only after concrete preflight, canonical ``RUNNING`` child Run
allocation, Queue-attempt activation, Worker execution, and candidate-byte upload have
already occurred. It converts a Worker ``success`` candidate into canonical Training or
Evaluation result state while preserving the existing frozen domain and persistence
contracts.

The boundary deliberately owns no Queue closure/satisfaction, Worker outcome
acknowledgement, lease validation, retry/backoff decision, Study reconciliation, Worker
loop, acquire/dispatch, or generic artifact-management framework. ``attempt_id`` is used
only to select immutable candidate staging bytes; the later outcome handler remains
responsible for proving that the attempt/lease is authorized for the supplied Run.
"""

from __future__ import annotations

from ..catalog.architecture import ArchitectureStatus
from ..evaluation.interface import EvaluationResult
from ..evaluation.preflight import preflight_evaluation
from ..evaluation.result_validation import accept_evaluation_result
from ..evaluation.run import (
    EvaluationRun,
    EvaluationRunExecution,
    EvaluationRunResult,
    EvaluationRunStatus,
)
from ..model.persistence import ensure_model_for_completed_training_run
from ..orchestration.candidates import CandidateStore, read_verified_candidate_bytes
from ..orchestration.worker_api import EvaluationSuccessCandidate, TrainingSuccessCandidate
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from .executable_loader import load_architecture_build
from .resolution import resolve_architecture
from .run_persistence import (
    persist_evaluation_run_transition,
    persist_training_run_transition,
    read_evaluation_run,
    read_training_run,
)
from ..training.run import (
    TrainingRun,
    TrainingRunExecution,
    TrainingRunResult,
    TrainingRunStatus,
)
from ..training.weights import CanonicalWeightsArtifact, load_canonical_weights


def accept_training_success(
    run: TrainingRun,
    attempt_id: int,
    candidate: TrainingSuccessCandidate,
    finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    candidate_store: CandidateStore,
) -> TrainingRun:
    """Accept one authorized Training success candidate into canonical MLDB state.

    ``run`` must be the exact canonical ``RUNNING`` Training Run allocated for the
    authorized ``attempt_id``. The attempt/lease association is a precondition supplied
    by the later Worker outcome handler; this function does not query or mutate Queue
    state. Before result work it must read the canonical Run by ``run.id`` through the
    frozen typed persistence boundary and require exact agreement with the supplied
    ``RUNNING`` value. A terminal Run is never reopened or rewritten here.

    Acceptance uses ``candidate.weights`` only as an attempt-local reference. It first
    calls :func:`read_verified_candidate_bytes`, which retrieves the stored exact bytes
    by ``attempt_id``/``ref.key`` and independently requires both SHA-256 and byte-count
    agreement with the reference. Worker metadata alone is never trusted.

    The verified payload is then validated as canonical learned state without invoking
    the Train Protocol again. The implementation freshly resolves exactly
    ``run.architecture`` through the frozen Architecture resolver. Resolver success is
    not sufficient at this acceptance boundary because ``resolve_architecture()`` may
    legitimately return a draft definition for operations that permit draft use. Before
    exposing executable code, the freshly resolved Architecture must still satisfy
    ``architecture.metadata.status is ArchitectureStatus.SEALED`` and
    ``architecture.metadata.implementation.sha256 is not None``. A draft, malformed, or
    otherwise inconsistent current Architecture is therefore an acceptance failure even
    though the canonical ``RUNNING`` Run could only have been allocated after an earlier
    successful sealed Training preflight.

    After those current-metadata checks, the implementation obtains the selected
    ``build`` callable through ``load_architecture_build()``. That frozen loader remains
    the owner of comparing the recorded implementation SHA-256 with the actual selected
    Python bytes before callable exposure; this function must not duplicate the loader's
    SHA-256 implementation. The candidate bytes are written to one implementation-private
    staging file beneath this Training Run's canonical ``work/`` directory and that
    staged path plus the Architecture build callable are delegated to
    ``load_canonical_weights()``. This reuses restricted CPU ``torch.load``, plain
    string-to-tensor mapping validation, fresh Architecture construction, and strict
    ``load_state_dict`` compatibility. No second checkpoint convention, key repair,
    non-strict loading, or call to ``train()`` is permitted.

    Only after that semantic validation succeeds may the exact candidate bytes become
    canonical ``artifacts/weights.pt``. The commit must preserve byte identity rather
    than reserialize the loaded module/state. If the canonical weights path is absent,
    the implementation commits these exact bytes through the existing filesystem
    complete-replacement primitive. If bytes already exist there while the Run is still
    ``RUNNING`` after an interrupted earlier acceptance, they must be read and required
    to equal this exact verified candidate; conflicting existing canonical bytes are an
    operation failure and must never be overwritten. The committed bytes are re-read or
    otherwise verified before metadata construction so the resulting
    :class:`CanonicalWeightsArtifact` records the SHA-256 and byte count of the exact
    canonical file with ``format='pytorch-state-dict'`` and
    ``path='artifacts/weights.pt'``.

    The successful terminal value is a new :class:`TrainingRun` that preserves the
    canonical initial schema/ID/Corpus/Architecture/Train-Protocol/complete parameters,
    seed, start time, Study lineage, environment, and work facts, sets
    ``status=COMPLETED``, records ``finished_at``, carries exactly one
    :class:`TrainingRunResult` for the committed weights, and carries no failure. It is
    persisted only through ``persist_training_run_transition()``.

    Canonical ordering is therefore:

    ``candidate verification -> learned-state validation -> exact weights commit ->
    RUNNING-to-COMPLETED persistence -> deterministic Model ensure``.

    After the Run transition succeeds, the function calls
    ``ensure_model_for_completed_training_run()`` and returns the completed Run only
    after the deterministic Model exists exactly. Queue ``SATISFIED`` remains later.

    The ordering intentionally tolerates filesystem crash gaps without a transaction
    framework. If exact weights were committed but the Run remains ``RUNNING``, replay of
    this same candidate verifies the already-present identical bytes and can finish the
    Run. If the Run reached ``COMPLETED`` but Model creation was interrupted, the later
    outcome handler or reconciliation may invoke the independent idempotent Model ensure
    boundary without reopening the Run. In particular, an outcome handler catching an
    acceptance exception must inspect the fresh canonical Training Run before deciding
    that the attempt may be terminalized as ``FAILED``. It may do so only while that Run
    is still ``RUNNING``; it must never attempt ``COMPLETED -> FAILED``. When the Run is
    already ``COMPLETED`` and the deterministic Model is missing, replay/reconciliation
    repairs only that Model ensure gap.
    """

    ...


def accept_evaluation_success(
    run: EvaluationRun,
    attempt_id: int,
    candidate: EvaluationSuccessCandidate,
    finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    candidate_store: CandidateStore,
) -> EvaluationRun:
    """Accept one authorized Evaluation success candidate into canonical MLDB state.

    ``run`` must be the exact canonical ``RUNNING`` Evaluation Run allocated for the
    authorized attempt. As with Training, Queue/lease authorization is an upstream
    precondition. The implementation first reads the canonical Run through
    ``read_evaluation_run()`` and requires it to equal the supplied running value; no
    terminal Evaluation Run is reopened or rewritten by this boundary.

    Acceptance reconstructs the formal validation context from canonical Run identity,
    not from ephemeral Worker response state. It reuses ``preflight_evaluation()`` with
    ``run.model``, ``run.corpus``, ``run.evaluation_protocol``, and the Run's complete
    persisted ``parameters`` as the concrete selected values. Success must reproduce the
    exact same complete resolved parameter mapping. The resulting frozen preflight
    supplies the canonical resolved common ``TaskHandle`` and the selected sealed
    Evaluation Protocol whose ``metadata.outputs`` is the one formal declaration model
    consumed below. No second Evaluation-output declaration type is introduced.

    Every ``candidate.artifacts`` reference is independently retrieved through
    :func:`read_verified_candidate_bytes` before formal result validation. Thus the exact
    stored bytes must exist for ``attempt_id``/``ref.key`` and must match both the
    reference SHA-256 and byte count. Candidate refs are not trusted as content evidence.

    After all required candidate bytes are verified, the implementation materializes
    each returned formal artifact into an implementation-private Controller staging path
    beneath this Evaluation Run's canonical ``work/`` directory. Staging filenames must
    be path-safe, but their naming and placement grammar are not a public replay,
    candidate-store, or repository-layout contract; raw Worker candidate keys must not
    be interpreted as filesystem paths. The staged exact bytes are then used to
    reconstruct only the existing frozen value:

    ``EvaluationResult(metrics=candidate.metrics, artifacts=<formal-key: staged-path>,
    unavailable_outputs=candidate.unavailable_outputs)``.

    Controller-side staging does not replace the Worker-side original ``work_dir``
    containment check performed before transport. It exists only so the portable
    candidate can once again satisfy the path-bearing input shape required by the common
    formal validator.

    The function then calls exactly ``accept_evaluation_result(``
    ``preflight.protocol.metadata.outputs, reconstructed_result, preflight.task,``
    ``run_paths.work_dir, run_paths.artifacts_dir)``. Declaration agreement, scalar
    validation, unavailable-output rules, required/optional artifact behavior,
    structured format/schema checks, canonical artifact import, SHA-256 calculation,
    byte counts, and optional-artifact validation issues remain owned by that frozen
    boundary rather than being duplicated here.

    From the returned ``AcceptedEvaluationResult``, partiality is derived exactly as:
    ``bool(unavailable_outputs) or bool(validation_issues)``. The function constructs
    ``COMPLETED_PARTIAL`` when partial and ``COMPLETED`` otherwise, preserving all
    canonical initial identity/parameter/start/Study/environment facts, setting
    ``finished_at``, storing ``EvaluationRunResult(metrics, artifacts)``, copying the
    accepted unavailable outputs and validation issues, and carrying no failure. The
    terminal value is persisted only through ``persist_evaluation_run_transition()``.

    Replaying the same exact candidate while the canonical Run is still ``RUNNING``
    must be safe even if an earlier acceptance call partially materialized files before
    terminal metadata was committed. Such files do not by themselves redefine canonical
    Evaluation Run history; the replay may reconstruct staging and re-enter the frozen
    result-acceptance operation for the same still-running attempt. This boundary does
    not strengthen ``accept_evaluation_result()`` into a globally reusable artifact
    transaction abstraction and does not promise a public deterministic artifact-path or
    artifact-identity scheme beyond that frozen contract. Once a terminal Run exists,
    this function does not rewrite it. Recognition of an exact lost-acknowledgement
    replay and mapping that fact to Worker API ``ALREADY_FINALIZED`` belongs to the later
    outcome handler.

    Any missing/conflicting candidate, failed context reconstruction, invalid formal
    result, required artifact failure, staging/import failure, or terminal persistence
    failure is an operation failure. This function chooses no retry policy and no Queue
    consequence; the later outcome handler owns unsuccessful Run handling subject to the
    canonical state actually reached before the failure.
    """

    ...
