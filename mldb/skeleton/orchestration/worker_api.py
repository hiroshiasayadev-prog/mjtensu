"""Transport-independent public signatures for the MLDB Worker logical API.

This skeleton fixes the Controller/Worker boundary for one separately executable
Worker process. Worker pulls one concrete Training or Evaluation attempt, retrieves the
immutable bytes selected by Controller, maintains its lease, uploads execution-local
candidate artifacts, and reports one domain outcome.

The module deliberately does not import Queue value/port types. ``attempt_id`` remains
the Queue-local ``int`` exposed by this boundary and lease/acquire/retrieval identities
remain opaque strings, so Wave 7-2B does not assume any Wave 7-2A public symbol.

Resolved metadata is carried by value because canonical repository ``Path`` values are
not remote Worker locators. Large or executable immutable bytes are represented by
:class:`ImmutableAssetDescriptor` and obtained through the asset-retrieval operation.
Worker-local cache placement, HTTP/JSON/streaming, authentication, Controller dispatch,
Run allocation, canonical result acceptance, Queue scheduling/retry policy, and Worker
execution-loop behavior are intentionally outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol, TypeAlias

from ..catalog.architecture import Architecture
from ..catalog.corpus import Corpus
from ..catalog.task import Task
from ..common.ids import EvaluationRunId, TrainingRunId
from ..common.parameters import ResolvedPublicParameters
from ..evaluation.interface import UnavailableOutput
from ..evaluation.protocol import EvaluationProtocol
from ..model.identity import Model
from ..training.protocol import TrainProtocol
from ..training.run import TrainingRun


WorkerJobKind: TypeAlias = Literal["training", "evaluation"]
"""The complete v1 Worker assignment-kind vocabulary."""


@dataclass(frozen=True, slots=True)
class AcquireWorkRequest:
    """Request one compatible attempt from Controller.

    ``worker_id`` is a non-empty process identity. ``acquire_token`` is one opaque
    Worker-generated identity for this logical acquire operation and must be reused
    unchanged when communication is retried after no definitive response.

    ``accepts`` is a non-empty immutable set containing only ``training`` and/or
    ``evaluation``. Worker selects capabilities, not a concrete logical job; Study job
    objects and Queue identifiers therefore do not appear in this request.
    """

    worker_id: str
    acquire_token: str
    accepts: frozenset[WorkerJobKind]


@dataclass(frozen=True, slots=True)
class ImmutableAssetDescriptor:
    """Controller-selected identity for one immutable byte object required by Worker.

    ``key`` is a non-empty assignment-local logical name describing the selected
    object. Assignment construction must choose it replay-stably for the same selected
    object. It is not a filesystem path and does not define a global artifact ontology.

    ``retrieval_id`` is an opaque Controller-provided identity used to request the exact
    bytes. It must be replay-stable for the lifetime of the authorized assignment:
    Controller must be able to reconstruct the same value from the persisted attempt
    plus authoritative immutable input identity when replaying a lost acquire response,
    rather than generating a fresh response-local token that would require storing the
    Worker API response in Queue state. Its concrete derivation remains private.

    ``retrieval_id`` deliberately contains no URL, repository path, cache path,
    transport routing, authentication material, or storage implementation contract.

    ``sha256`` is the required digest of the exact immutable bytes. ``bytes`` carries
    the required byte count when the underlying MLDB asset contract records one and is
    otherwise ``None``. Worker must verify every supplied integrity fact before use and
    must never substitute a mutable upstream source for this object.
    """

    key: str
    retrieval_id: str
    sha256: str
    bytes: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingAssignment:
    """One authorized Training execution attempt and its complete resolved input.

    Controller has already completed Training preflight, allocated ``run_id``, created
    the canonical schema-valid ``running`` Training Run, and established the Queue
    attempt before exposing this value.

    Metadata is projected from that successful preflight by value instead of sending
    repository-bound runtime handles. ``task``, ``corpus``, ``architecture``, and
    ``protocol`` are the exact resolved definitions selected by Controller; ``seed`` is
    the validated concrete Training seed; and ``parameters`` is the complete resolved
    public-parameter mapping, never caller overrides alone.

    The three descriptors identify the exact immutable bytes needed to materialize the
    Worker-side Corpus and executable implementations. Their hashes must agree with the
    corresponding metadata integrity identities. No descriptor is a canonical
    repository ``Path`` or remote filesystem locator.
    """

    attempt_id: int
    run_id: TrainingRunId
    lease_token: str
    lease_until: str
    kind: Literal["training"]
    task: Task
    corpus: Corpus
    architecture: Architecture
    protocol: TrainProtocol
    seed: int
    parameters: ResolvedPublicParameters
    corpus_artifact: ImmutableAssetDescriptor
    architecture_implementation: ImmutableAssetDescriptor
    train_protocol_implementation: ImmutableAssetDescriptor


@dataclass(frozen=True, slots=True)
class EvaluationAssignment:
    """One authorized Evaluation execution attempt and its complete resolved input.

    Controller has already completed Evaluation preflight, allocated ``run_id``,
    created the canonical schema-valid ``running`` Evaluation Run, and established the
    Queue attempt before exposing this value.

    ``task``, ``corpus``, ``model``, ``training_run``, ``model_architecture``, and
    ``protocol`` are the exact resolved metadata selected by Controller.
    ``training_run`` is the validated completed Training Run referenced by ``model``;
    Worker must not try to reconstruct that Run metadata from the three-field Model
    record. ``model_architecture`` is the Architecture reached through that completed
    Training Run lineage and is supplied explicitly because Worker must reconstruct the
    learned module without dereferencing Controller-local repository paths.
    ``parameters`` is the complete resolved Evaluation Protocol public-parameter
    mapping.

    Controller guarantees ``training_run.id == model.training_run``, that
    ``training_run`` is completed with canonical weight-result metadata, and that
    ``training_run.architecture == model_architecture.id``. ``model_weights`` identifies
    exactly the learned-weight bytes recorded by ``training_run.result.weights`` and its
    SHA-256/byte count must agree with that metadata. Together with ``training_run``,
    ``model_architecture``, and its implementation descriptor this provides the
    immutable loading lineage Worker needs without turning a canonical weights path into
    a remote locator. Corpus, Architecture implementation, and Evaluation Protocol bytes
    follow the same descriptor/integrity-agreement rule.
    """

    attempt_id: int
    run_id: EvaluationRunId
    lease_token: str
    lease_until: str
    kind: Literal["evaluation"]
    task: Task
    corpus: Corpus
    model: Model
    training_run: TrainingRun
    model_architecture: Architecture
    protocol: EvaluationProtocol
    parameters: ResolvedPublicParameters
    corpus_artifact: ImmutableAssetDescriptor
    model_architecture_implementation: ImmutableAssetDescriptor
    model_weights: ImmutableAssetDescriptor
    evaluation_protocol_implementation: ImmutableAssetDescriptor


@dataclass(frozen=True, slots=True)
class NoWork:
    """Successful acquire response indicating no compatible work is currently ready."""


@dataclass(frozen=True, slots=True)
class AcquireRejection:
    """Definitive acquire rejection rather than a retryable communication failure."""

    type: str
    message: str


AcquireWorkResponse: TypeAlias = (
    TrainingAssignment | EvaluationAssignment | NoWork | AcquireRejection
)
"""Exactly one v1 acquire result: assignment, no work, or definitive rejection."""


@dataclass(frozen=True, slots=True)
class RetrieveAssetRequest:
    """Request the exact immutable bytes named by one assignment descriptor."""

    asset: ImmutableAssetDescriptor


@dataclass(frozen=True, slots=True)
class RetrievedAsset:
    """Exact immutable bytes returned for one successful asset retrieval."""

    content: bytes


@dataclass(frozen=True, slots=True)
class AssetRejection:
    """Definitive refusal to fulfill an asset descriptor."""

    type: str
    message: str


AssetRetrievalResponse: TypeAlias = RetrievedAsset | AssetRejection
"""Successful immutable bytes or one definitive asset-retrieval rejection."""


@dataclass(frozen=True, slots=True)
class HeartbeatRequest:
    """Maintain liveness for the currently authorized concrete attempt."""

    attempt_id: int
    lease_token: str


@dataclass(frozen=True, slots=True)
class HeartbeatAccepted:
    """Accepted heartbeat with the current lease deadline and control signal.

    Every accepted heartbeat resets the attempt's inactivity deadline to exactly one
    hour after the Controller acceptance time. ``lease_until`` is the resulting Queue
    timestamp encoded as UTC RFC3339 text with six fractional-second digits and ``Z``.
    ``cancel_requested`` asks Worker to cooperatively stop; it is not itself a terminal
    Run or Queue status.
    """

    lease_until: str
    cancel_requested: bool


@dataclass(frozen=True, slots=True)
class HeartbeatRejection:
    """Definitive heartbeat rejection such as a stale or superseded lease."""

    type: str
    message: str


HeartbeatResponse: TypeAlias = HeartbeatAccepted | HeartbeatRejection
"""Accepted lease/control state or one definitive lease rejection."""


@dataclass(frozen=True, slots=True)
class CandidateUploadRequest:
    """Upload one attempt-local candidate artifact as exact bytes.

    ``key`` is a non-empty attempt-local candidate key. ``content_identity`` is the
    integrity identity for the exact ``content``; whenever the applicable MLDB result
    contract requires SHA-256, this value is that exact SHA-256 digest.

    Replaying the same ``attempt_id``/``key`` with the same content identity may be
    acknowledged as already present. An already accepted candidate under the same
    attempt/key but a conflicting identity must not be overwritten.
    """

    attempt_id: int
    lease_token: str
    key: str
    content_identity: str
    content: bytes


class CandidateUploadAcknowledgement(str, Enum):
    """Definitive Controller result for one candidate upload operation."""

    ACCEPTED = "accepted"
    ALREADY_PRESENT = "already_present"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class CandidateArtifactRef:
    """Reference from a reported success outcome to a previously uploaded candidate.

    The reference is attempt-local: ``attempt_id`` and ``lease_token`` live on the
    enclosing outcome. ``key`` selects the uploaded candidate, ``content_identity``
    must equal the identity used for that upload, and ``bytes`` is the exact candidate
    byte count. It is not a canonical Run artifact path and does not grant Worker any
    result-acceptance authority.
    """

    key: str
    content_identity: str
    bytes: int


@dataclass(frozen=True, slots=True)
class TrainingSuccessCandidate:
    """Worker-side learned-weight candidate for Controller Training acceptance.

    ``weights`` identifies the uploaded generic canonical-state candidate. Its
    ``content_identity`` is the SHA-256 of the exact uploaded ``pytorch-state-dict``
    bytes and ``bytes`` is their exact length. The canonical Run-relative path,
    acceptance, final Training Run status, and deterministic Model creation remain
    Controller responsibilities and are intentionally absent here.
    """

    weights: CandidateArtifactRef


@dataclass(frozen=True, slots=True)
class EvaluationSuccessCandidate:
    """Worker-side EvaluationResult projection suitable for remote handoff.

    ``metrics`` and ``unavailable_outputs`` preserve the protocol-returned formal
    result surface. ``artifacts`` replaces each Worker-local ``Path`` from
    :class:`evaluation.interface.EvaluationResult` with a reference to candidate bytes
    already uploaded through this API. Controller later validates declarations,
    structured artifact contents, optional/required rules, and canonical
    materialization under the frozen Evaluation result-acceptance contract.
    """

    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, CandidateArtifactRef]
    unavailable_outputs: Sequence[UnavailableOutput]


@dataclass(frozen=True, slots=True)
class TrainingSucceeded:
    """Report successful completion of one Training domain invocation."""

    attempt_id: int
    lease_token: str
    candidate: TrainingSuccessCandidate


@dataclass(frozen=True, slots=True)
class EvaluationSucceeded:
    """Report successful completion of one Evaluation domain invocation."""

    attempt_id: int
    lease_token: str
    candidate: EvaluationSuccessCandidate


@dataclass(frozen=True, slots=True)
class AttemptFailed:
    """Report one domain or execution-local failure without local domain retry."""

    attempt_id: int
    lease_token: str
    type: str
    message: str


@dataclass(frozen=True, slots=True)
class AttemptCancelled:
    """Report cooperative termination of one attempt."""

    attempt_id: int
    lease_token: str


AttemptOutcome: TypeAlias = (
    TrainingSucceeded | EvaluationSucceeded | AttemptFailed | AttemptCancelled
)
"""Exactly one Worker-reported v1 attempt outcome."""


class OutcomeAcknowledgement(str, Enum):
    """Definitive Controller acknowledgement of an attempt-outcome report.

    ``ALREADY_FINALIZED`` is the idempotent replay result when Controller had already
    accepted/finalized that exact attempt before the original acknowledgement was lost.
    A stale lease whose attempt was not already accepted, a conflicting replay, or an
    invalid candidate receives ``REJECTED`` instead.
    """

    ACCEPTED = "accepted"
    ALREADY_FINALIZED = "already_finalized"
    REJECTED = "rejected"


class WorkerApi(Protocol):
    """Logical pull-oriented Controller API consumed by one separate Worker.

    Implementations may be in-process adapters or future remote clients/servers, but
    these methods themselves define no HTTP routes, JSON encoding, byte streaming,
    authentication, retry counters, backoff configuration, Worker execution loop, or
    cache manager.

    Retryable communication failure is intentionally not a response variant: it means
    no definitive valid response was obtained. Worker client behavior retries the same
    logical operation after exactly 10 seconds without a finite retry count, preserving
    the request identity and local candidate/outcome state. ``no_work`` and every
    represented rejection/acknowledgement are definitive responses, not communication
    failures.
    """

    def acquire_work(self, request: AcquireWorkRequest) -> AcquireWorkResponse:
        """Acquire at most one compatible concrete attempt.

        Repeating the same ``acquire_token`` after a lost response must recover the same
        still-authorized committed assignment, including replay-stable immutable-asset
        descriptor identities, and must not allocate another child Run. A token already
        consumed by a closed/incompatible attempt is definitively rejected rather than
        reused for new work.
        """

        ...

    def retrieve_asset(self, request: RetrieveAssetRequest) -> AssetRetrievalResponse:
        """Return exact immutable bytes for the Controller-provided descriptor.

        Worker verifies the returned bytes against ``request.asset.sha256`` and the
        optional/required byte count before use. Successful transfer with an integrity
        mismatch is an execution/assignment failure, not a communication failure and
        not permission to substitute another source.
        """

        ...

    def heartbeat(self, request: HeartbeatRequest) -> HeartbeatResponse:
        """Extend the current lease or definitively reject stale authorization.

        Successful acceptance establishes a new one-hour inactivity deadline and may
        carry a cooperative cancellation request. A stale, expired, closed, or
        superseded lease must not be revived by this operation.
        """

        ...

    def upload_candidate(
        self,
        request: CandidateUploadRequest,
    ) -> CandidateUploadAcknowledgement:
        """Store one attempt-local candidate without canonical result acceptance.

        The same attempt/key and same content identity may return ``ALREADY_PRESENT``.
        A conflicting candidate for the same attempt/key, stale authorization, or other
        definitive invalid request returns ``REJECTED`` rather than overwriting prior
        bytes. Successful upload alone never satisfies a Queue job or terminalizes a
        canonical Run.
        """

        ...

    def report_outcome(self, outcome: AttemptOutcome) -> OutcomeAcknowledgement:
        """Report one Training/Evaluation success, failure, or cancellation.

        Controller owns candidate validation, canonical artifact commit, Run terminal
        state, Model creation, Queue closure/satisfaction, and any later domain retry.
        Worker never reruns ``train()`` or ``evaluate()`` under the same attempt after
        a domain failure; communication failure while reporting only retries this same
        outcome operation.

        Replaying an outcome after a lost acknowledgement must not create another Run or
        duplicate accepted canonical state. If the exact attempt was already finalized
        by the earlier accepted report, Controller returns ``ALREADY_FINALIZED``.
        """

        ...
