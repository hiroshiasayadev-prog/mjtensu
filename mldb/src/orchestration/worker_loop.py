"""Worker-side orchestration boundary for one already-assigned MLDB attempt.

This module connects the frozen Worker API and Worker execution boundaries for exactly
one concrete Training or Evaluation assignment. It owns the in-process sequencing that
retrieves and verifies immutable inputs, keeps the attempt lease alive, materializes
Worker-local execution files, invokes the selected domain executor at most once, uploads
attempt-local result candidates, and retains/replays one outcome until Controller gives
a definitive acknowledgement.

The boundary starts after acquire. It does not poll for work, register a Worker, inspect
Study progression, allocate Runs, mutate Queue state, accept canonical results, choose a
domain retry, or recover a Worker process after crash. Controller repository
``FilesystemPort`` is intentionally absent: Worker-local scratch/cache/materialization
is not canonical MLDB repository persistence.

Communication retry here is the fixed Worker-API rule, not the Controller
:class:`~mldb.skeleton.orchestration.retry_policy.RetryPolicy`: when an API invocation
fails without a definitive Controller response, the same logical operation is retried
after exactly 10 seconds without a finite retry count. Transport-specific exception
classification remains caller-owned because the frozen :class:`WorkerApi` intentionally
defines response semantics rather than one network/HTTP exception hierarchy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from pathlib import Path
import threading
import time
from typing import Protocol, overload

from .worker_api import (
    AssetRejection,
    AttemptCancelled,
    AttemptFailed,
    CandidateArtifactRef,
    CandidateUploadAcknowledgement,
    CandidateUploadRequest,
    EvaluationAssignment,
    EvaluationSucceeded,
    EvaluationSuccessCandidate,
    HeartbeatAccepted,
    HeartbeatRejection,
    HeartbeatRequest,
    ImmutableAssetDescriptor,
    OutcomeAcknowledgement,
    RetrieveAssetRequest,
    RetrievedAsset,
    TrainingAssignment,
    TrainingSucceeded,
    TrainingSuccessCandidate,
    WorkerApi,
)
from .worker_execution import (
    EvaluationExecutionFiles,
    TrainingExecutionCandidate,
    TrainingExecutionFiles,
    execute_evaluation_attempt,
    execute_training_attempt,
)


class WorkerAttemptMaterializer(Protocol):
    """Prepare Worker-local execution files from already verified assignment bytes.

    The second argument contains every immutable object required by ``assignment`` keyed
    by the exact corresponding :class:`~mldb.skeleton.orchestration.worker_api.ImmutableAssetDescriptor.key`.
    ``run_worker_attempt()`` obtains those bytes only through ``WorkerApi.retrieve_asset``
    and verifies each descriptor SHA-256 plus every supplied byte count before this
    callable is entered.

    The implementation may choose Worker-local scratch paths, cache placement, directory
    creation, executable filenames, and cleanup policy. It may reuse cached content, but
    it must not substitute a mutable source or treat a Controller repository path as a
    Worker path. The returned paths are execution resources only.

    ``run_worker_attempt()`` re-reads and verifies every returned immutable input path
    against the assignment descriptor before domain execution, so cache/materialization
    does not weaken the Worker's independent integrity obligation. For Training the
    materializer additionally chooses the protocol ``work_dir`` and local
    ``weights_candidate_path``. For Evaluation it chooses the protocol ``work_dir``.
    Those output/work locations are not canonical Controller Run locations.

    This callable is deliberately the only Worker-local file-preparation port fixed by
    this Wave. It is not a cache manager, repository filesystem port, transfer session,
    or durable recovery database.
    """

    @overload
    def __call__(
        self,
        assignment: TrainingAssignment,
        assets: Mapping[str, bytes],
    ) -> TrainingExecutionFiles: ...

    @overload
    def __call__(
        self,
        assignment: EvaluationAssignment,
        assets: Mapping[str, bytes],
    ) -> EvaluationExecutionFiles: ...


def run_worker_attempt(
    assignment: TrainingAssignment | EvaluationAssignment,
    api: WorkerApi,
    materialize: WorkerAttemptMaterializer,
    is_retryable_communication_failure: Callable[[Exception], bool],
) -> None:
    """Run one already-assigned Worker attempt through definitive Controller handoff.

    ``assignment`` is exactly one frozen :class:`TrainingAssignment` or
    :class:`EvaluationAssignment` returned by acquire. One Worker-process caller enters
    this boundary once for that owned assignment; calling it concurrently or a second
    time for the same still-owned attempt would itself violate the frozen one-active-
    attempt Worker contract rather than define another local retry mechanism. Acquire
    itself, ``NoWork`` polling, Worker registration, and any outer forever-loop are
    outside this function.

    ``api`` is the frozen logical Worker API. ``materialize`` owns only Worker-local file
    placement as described by :class:`WorkerAttemptMaterializer`.
    ``is_retryable_communication_failure`` classifies an exception raised while invoking
    ``api`` as the Worker-API semantic case "no definitive valid Controller response was
    obtained". Returning ``True`` means this function waits exactly 10 seconds and
    retries the *same logical API operation* with the same request/candidate/outcome
    state. Returning ``False`` lets the exception escape as a fatal transport/protocol or
    software incompatibility; deterministic incompatibility must not be infinite-retried.
    Domain execution exceptions are never passed to this classifier.

    The classifier is the only transport-specific dependency exposed here. This boundary
    does not define a generic retry policy/engine, backoff object, retry count, clock,
    HTTP exception hierarchy, or public sleep abstraction. The fixed communication delay
    is exactly 10 seconds. Intentional process shutdown may terminate an in-progress
    communication retry outside normal completion of this function.

    **Heartbeat lifecycle.** Heartbeat begins while this assignment is Worker-owned and
    remains active through immutable-input retrieval, local materialization, synchronous
    domain execution, candidate preparation/upload, and waiting for a definitive outcome
    acknowledgement. Because the frozen Train/Evaluation callables are synchronous and
    expose no cancellation/liveness token, an implementation must maintain heartbeat
    independently of the blocked domain call, for example with one implementation-private
    lightweight heartbeat thread/task. This module exposes no thread/task manager or
    lifecycle class.

    Normal healthy-heartbeat cadence is intentionally not fixed by the Worker API/lease
    ADR and therefore is not added to this public signature. The implementation-private
    cadence must maintain the one-hour inactivity lease under healthy communication.
    For one heartbeat logical operation that obtains no definitive response, however,
    the retry interval is exactly 10 seconds and unbounded. A successful
    :class:`~mldb.skeleton.orchestration.worker_api.HeartbeatAccepted` completes that
    heartbeat operation, updates the Worker's observed control state, and later healthy
    heartbeat is a new logical heartbeat operation. A definitive
    :class:`~mldb.skeleton.orchestration.worker_api.HeartbeatRejection` stops ordinary
    heartbeat under this lease and marks the attempt no longer authoritative; it is not
    communication failure and must not be retried as one.

    Accepted heartbeat ``cancel_requested=True`` is remembered monotonically for this
    attempt. The Worker checks that state at feasible orchestration boundaries. Before
    domain invocation, an observed cancellation prevents ``execute_training_attempt()``
    or ``execute_evaluation_attempt()`` from being called and fixes the intended outcome
    to :class:`~mldb.skeleton.orchestration.worker_api.AttemptCancelled`. No thread kill,
    process kill, signal injection, or generic cancellation token is introduced.

    If cancellation becomes observable only while the synchronous domain callable is
    already running, this function cannot truthfully claim to have interrupted that
    invocation. The callable is allowed to return or raise normally. A successful return
    is retained as a truthful success and proceeds through candidate upload/outcome
    reporting; it is not unconditionally discarded merely because cancellation raced
    after the last feasible pre-invocation check. This matches the Controller contract,
    which may accept a racing truthful satisfying success. If the already-running domain
    invocation raises, that execution failure remains the intended failed outcome rather
    than being rewritten as a cancellation that did not cause the stop.

    Once one success/failure/cancelled outcome has been prepared, later heartbeat control
    does not replace it with a different outcome while communication is being retried.
    The exact prepared outcome and candidate state are retained until definitive outcome
    acknowledgement or process termination.

    **Immutable input retrieval and verification.** The required descriptor set is
    selected mechanically from assignment kind, without repository lookup or new asset
    naming:

    Training:

    - ``assignment.corpus_artifact``;
    - ``assignment.architecture_implementation``;
    - ``assignment.train_protocol_implementation``.

    Evaluation:

    - ``assignment.corpus_artifact``;
    - ``assignment.model_architecture_implementation``;
    - ``assignment.model_weights``;
    - ``assignment.evaluation_protocol_implementation``.

    Each descriptor is retrieved with
    ``api.retrieve_asset(RetrieveAssetRequest(asset=descriptor))``. Communication failure
    retries that same retrieval after exactly 10 seconds; it never causes domain
    execution to be retried. A definitive ``RetrievedAsset`` is independently verified
    by Worker: SHA-256 of the exact returned bytes must equal ``descriptor.sha256`` and,
    when ``descriptor.bytes`` is non-``None``, ``len(content)`` must equal it. Only after
    all such checks may the bytes be passed to ``materialize``. A successful transfer
    whose integrity facts disagree is an execution/assignment failure, not a
    communication retry.

    A definitive ``AssetRejection`` likewise means the assigned input cannot be used.
    The domain callable is not invoked; the Worker prepares ``AttemptFailed`` using the
    rejection's concise ``type`` and ``message`` and reports that outcome under the same
    attempt. For local retrieval-integrity/materialization failures, the minimal v1
    failure representation is the Python exception class name ``type(exc).__name__`` and
    concise ``str(exc)``; no ErrorClassifier or exception taxonomy is introduced.

    After ``materialize`` returns, the returned execution-file kind must agree with the
    assignment kind. Before passing it to the frozen executor, the Worker re-reads each
    immutable input path represented by those files and requires the exact same
    descriptor SHA-256/byte-count agreement. This permits implementation-private cache
    reuse while ensuring only verified exact bytes reach runtime Handles. Work/output
    paths are not interpreted as canonical repository paths.

    **Exactly-one domain invocation.** After a final pre-invocation cancellation and
    lease-authority check, Training calls exactly once::

        execute_training_attempt(assignment, files)

    and Evaluation calls exactly once::

        execute_evaluation_attempt(assignment, files)

    for this ``run_worker_attempt`` invocation. If that call raises, the function does
    not call it again under this attempt/Run identity. It fixes one ``AttemptFailed``
    with ``attempt_id=assignment.attempt_id``, ``lease_token=assignment.lease_token``,
    ``type=type(exc).__name__``, and concise ``message=str(exc)``, then proceeds only with
    outcome reporting. Controller/Queue alone may later create a new attempt/Run.

    **Training success candidate.** A successful Training executor returns one
    ``TrainingExecutionCandidate``. The Worker reads ``local_weights_path`` once for
    candidate preparation and requires the exact bytes to agree with the returned
    ``CanonicalWeightsArtifact.sha256`` and ``.bytes``. The upload request uses this
    already-frozen artifact's ``path`` value as the attempt-local candidate key, its
    SHA-256 as ``content_identity``, and the exact local bytes as ``content``. Reusing the
    existing frozen weight-artifact identifier avoids defining a second Worker-only
    weights naming vocabulary; the candidate key nevertheless remains attempt-local and
    is not interpreted as a Controller filesystem path.

    ``CandidateUploadAcknowledgement.ACCEPTED`` and ``ALREADY_PRESENT`` both establish
    the same ``CandidateArtifactRef`` from that exact key/hash/byte count and permit
    construction of ``TrainingSucceeded(candidate=TrainingSuccessCandidate(...))``.
    Communication failure retries the identical upload request after exactly 10 seconds
    and does not rerun Training.

    A definitive candidate-upload ``REJECTED`` cannot be retried as communication and
    cannot support a success reference. Because the frozen acknowledgement intentionally
    carries no rejection-reason subtype, this Worker boundary does not guess whether the
    rejection was candidate conflict, stale authority, or another Controller refusal.
    If the attempt is still known authoritative, the failed result handoff is represented
    minimally as ``AttemptFailed(type="CandidateUploadRejected", message=<concise key
    context>)`` and reported once; this single literal is only the execution-local
    handoff failure representation, not a new error taxonomy. If a definitive heartbeat
    has already established lease loss, no new failure outcome is forced under that
    stale authorization. In either case the completed Training domain call is never
    rerun.

    **Evaluation success candidate.** A successful Evaluation executor returns the exact
    same frozen ``EvaluationResult`` object produced by the protocol after its local
    containment check. ``metrics`` and ``unavailable_outputs`` are retained unchanged.
    For every ``result.artifacts`` entry, the exact formal artifact mapping key is also
    the candidate-upload ``key``; it is not renamed, normalized, prefixed, or projected
    through a new artifact ontology. The Worker reads that local path's exact bytes,
    computes SHA-256, uploads the attempt/key/identity/bytes, and on ``ACCEPTED`` or
    ``ALREADY_PRESENT`` creates the corresponding ``CandidateArtifactRef``. The final
    ``EvaluationSuccessCandidate.artifacts`` mapping preserves the same formal keys and
    pairs them with those refs.

    Candidate-file read/hash preparation failure is execution-local result preparation
    failure: it fixes ``AttemptFailed`` and never reruns Evaluation. A definitive upload
    ``REJECTED`` follows the same minimal result-handoff rule as Training. Communication
    failure for any artifact upload retries only that same upload every 10 seconds while
    retaining the already-completed ``EvaluationResult`` and any earlier uploaded refs.

    **Lease loss.** A definitive heartbeat rejection means this lease is stale,
    superseded, closed, or otherwise no longer authoritative. Before domain invocation it
    prevents the invocation and prevents a new candidate/outcome from being forced under
    that lease. During an uninterruptible synchronous domain call, the Worker cannot kill
    the call, but after it returns it must not begin fresh candidate acceptance work under
    authorization already known lost. During an already-ambiguous candidate or outcome
    communication operation, replay may continue only to obtain that operation's
    definitive Controller response; such replay preserves the exact original request and
    cannot revive authority. In particular, an outcome accepted before its response was
    lost may still resolve as ``ALREADY_FINALIZED``; an outcome not previously accepted
    may resolve as ``REJECTED``.

    **Outcome acknowledgement.** Reporting uses exactly one prepared immutable outcome.
    ``api.report_outcome(outcome)`` communication failure waits exactly 10 seconds and
    retries that same object without changing candidate references, failure information,
    or outcome kind. ``OutcomeAcknowledgement.ACCEPTED``, ``ALREADY_FINALIZED``, and
    ``REJECTED`` are all definitive and end this one-attempt operation. ``REJECTED`` does
    not cause local domain rerun, candidate replacement, alternate outcome reporting, or
    acquire of replacement work.

    Heartbeat remains active while outcome acknowledgement is unresolved and stops only
    after the attempt has reached one of the definitive end conditions above (including
    definitive lease loss where no ambiguous handoff remains). The function returns
    ``None`` because canonical Run status and Queue satisfaction remain Controller
    authority; this Worker boundary does not expose a second success/failure lifecycle.

    **Process crash.** Attempt-local retrieved bytes, materialized files, completed domain
    result, uploaded refs, and prepared outcome need only survive while this Worker
    process remains alive. This function introduces no checkpoint file, local database,
    restart journal, or candidate replay database. If the process crashes, Controller
    lease expiry/reconciliation/new-attempt semantics own recovery; a restarted Worker
    does not silently re-enter the same domain invocation from this boundary.
    """

    _run_worker_attempt_impl(
        assignment,
        api,
        materialize,
        is_retryable_communication_failure,
    )


_COMMUNICATION_RETRY_SECONDS = 10.0
_HEALTHY_HEARTBEAT_INTERVAL_SECONDS = 30.0


class _NonRetryableApiCall(Exception):
    def __init__(self, error: Exception) -> None:
        self.error = error
        super().__init__(str(error))


class _ProtocolResponseError(TypeError):
    pass


class _HeartbeatControl:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stop = threading.Event()
        self.first_attempt_done = threading.Event()
        self.cancel_requested = False
        self.lease_lost = False
        self.lease_until: str | None = None
        self.fatal_error: Exception | None = None

    def accept(self, response: HeartbeatAccepted) -> None:
        with self._lock:
            self.lease_until = response.lease_until
            self.cancel_requested = self.cancel_requested or response.cancel_requested

    def reject(self) -> None:
        with self._lock:
            self.lease_lost = True

    def fail(self, error: Exception) -> None:
        with self._lock:
            if self.fatal_error is None:
                self.fatal_error = error

    def snapshot(self) -> tuple[bool, bool, Exception | None]:
        with self._lock:
            return self.cancel_requested, self.lease_lost, self.fatal_error


def _run_worker_attempt_impl(
    assignment: TrainingAssignment | EvaluationAssignment,
    api: WorkerApi,
    materialize: WorkerAttemptMaterializer,
    classifier: Callable[[Exception], bool],
) -> None:
    control = _HeartbeatControl()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(assignment, api, classifier, control),
        name=f"mldb-heartbeat-{assignment.attempt_id}",
        daemon=True,
    )
    heartbeat.start()
    try:
        descriptors = _descriptors_for_assignment(assignment)
        assets: dict[str, bytes] = {}
        try:
            if len({descriptor.key for descriptor in descriptors}) != len(descriptors):
                raise ValueError("assignment asset descriptor keys must be unique")
            for descriptor in descriptors:
                prepared = _pre_domain_control_outcome(assignment, control)
                if prepared is not None:
                    _report_outcome(api, prepared, classifier)
                    return
                if control.snapshot()[1]:
                    return
                response = _call_api_with_retry(
                    lambda request=RetrieveAssetRequest(asset=descriptor): api.retrieve_asset(request),
                    classifier,
                )
                _raise_heartbeat_fatal(control)
                if control.snapshot()[1]:
                    return
                if isinstance(response, AssetRejection):
                    prepared = _prepare_failure_if_authorized(
                        assignment,
                        control,
                        response.type,
                        response.message,
                    )
                    if prepared is not None:
                        _report_outcome(api, prepared, classifier)
                    return
                if not isinstance(response, RetrievedAsset):
                    raise _ProtocolResponseError("retrieve_asset returned an unsupported response")
                _verify_descriptor_content(descriptor, response.content)
                assets[descriptor.key] = response.content

            prepared = _pre_domain_control_outcome(assignment, control)
            if prepared is not None:
                _report_outcome(api, prepared, classifier)
                return
            files = materialize(assignment, assets)
            _verify_materialized_files(assignment, files)
        except _NonRetryableApiCall:
            raise
        except _ProtocolResponseError:
            raise
        except Exception as error:
            if _is_heartbeat_fatal(error, control):
                raise
            prepared = _prepare_failure_if_authorized(
                assignment,
                control,
                type(error).__name__,
                str(error),
            )
            if prepared is not None:
                _report_outcome(api, prepared, classifier)
            return

        control.first_attempt_done.wait()
        _raise_heartbeat_fatal(control)
        cancelled, lease_lost, _ = control.snapshot()
        if lease_lost:
            return
        if cancelled:
            _report_outcome(
                api,
                AttemptCancelled(
                    attempt_id=assignment.attempt_id,
                    lease_token=assignment.lease_token,
                ),
                classifier,
            )
            return

        if isinstance(assignment, TrainingAssignment):
            assert isinstance(files, TrainingExecutionFiles)
            try:
                candidate = execute_training_attempt(assignment, files)
            except Exception as error:
                _, lease_lost, fatal = control.snapshot()
                if fatal is not None:
                    raise fatal
                if lease_lost:
                    return
                _report_outcome(api, _attempt_failed(assignment, error), classifier)
                return
            _raise_heartbeat_fatal(control)
            if control.snapshot()[1]:
                return
            outcome = _prepare_training_success(assignment, candidate, api, classifier, control)
        else:
            assert isinstance(assignment, EvaluationAssignment)
            assert isinstance(files, EvaluationExecutionFiles)
            try:
                result = execute_evaluation_attempt(assignment, files)
            except Exception as error:
                _, lease_lost, fatal = control.snapshot()
                if fatal is not None:
                    raise fatal
                if lease_lost:
                    return
                _report_outcome(api, _attempt_failed(assignment, error), classifier)
                return
            _raise_heartbeat_fatal(control)
            if control.snapshot()[1]:
                return
            outcome = _prepare_evaluation_success(assignment, result, api, classifier, control)

        if outcome is not None:
            _report_outcome(api, outcome, classifier)
    except _NonRetryableApiCall as fatal:
        raise fatal.error
    finally:
        control.stop.set()
        heartbeat.join()


def _heartbeat_loop(
    assignment: TrainingAssignment | EvaluationAssignment,
    api: WorkerApi,
    classifier: Callable[[Exception], bool],
    control: _HeartbeatControl,
) -> None:
    request = HeartbeatRequest(
        attempt_id=assignment.attempt_id,
        lease_token=assignment.lease_token,
    )
    while not control.stop.is_set():
        try:
            response = api.heartbeat(request)
        except Exception as error:
            control.first_attempt_done.set()
            try:
                retryable = classifier(error)
            except Exception as classifier_error:
                control.fail(classifier_error)
                return
            if not retryable:
                control.fail(error)
                return
            if control.stop.wait(_COMMUNICATION_RETRY_SECONDS):
                return
            continue

        control.first_attempt_done.set()
        if isinstance(response, HeartbeatAccepted):
            control.accept(response)
            if control.stop.wait(_HEALTHY_HEARTBEAT_INTERVAL_SECONDS):
                return
            continue
        if isinstance(response, HeartbeatRejection):
            control.reject()
            return
        control.fail(_ProtocolResponseError("heartbeat returned an unsupported response"))
        return


def _call_api_with_retry(operation: Callable[[], object], classifier: Callable[[Exception], bool]) -> object:
    while True:
        try:
            return operation()
        except Exception as error:
            try:
                retryable = classifier(error)
            except Exception as classifier_error:
                raise _NonRetryableApiCall(classifier_error) from classifier_error
            if not retryable:
                raise _NonRetryableApiCall(error) from error
            time.sleep(_COMMUNICATION_RETRY_SECONDS)


def _descriptors_for_assignment(
    assignment: TrainingAssignment | EvaluationAssignment,
) -> tuple[ImmutableAssetDescriptor, ...]:
    if isinstance(assignment, TrainingAssignment):
        return (
            assignment.corpus_artifact,
            assignment.architecture_implementation,
            assignment.train_protocol_implementation,
        )
    if isinstance(assignment, EvaluationAssignment):
        return (
            assignment.corpus_artifact,
            assignment.model_architecture_implementation,
            assignment.model_weights,
            assignment.evaluation_protocol_implementation,
        )
    raise TypeError("assignment must be TrainingAssignment or EvaluationAssignment")


def _verify_descriptor_content(descriptor: ImmutableAssetDescriptor, content: bytes) -> None:
    if not isinstance(content, bytes):
        raise TypeError("retrieved asset content must be bytes")
    if hashlib.sha256(content).hexdigest().lower() != descriptor.sha256.lower():
        raise ValueError(f"asset SHA-256 mismatch for {descriptor.key}")
    if descriptor.bytes is not None and len(content) != descriptor.bytes:
        raise ValueError(f"asset byte-count mismatch for {descriptor.key}")


def _verify_materialized_files(
    assignment: TrainingAssignment | EvaluationAssignment,
    files: TrainingExecutionFiles | EvaluationExecutionFiles,
) -> None:
    if isinstance(assignment, TrainingAssignment):
        if not isinstance(files, TrainingExecutionFiles):
            raise TypeError("training assignment requires TrainingExecutionFiles")
        pairs = (
            (assignment.corpus_artifact, files.corpus_artifact),
            (assignment.architecture_implementation, files.architecture_implementation),
            (assignment.train_protocol_implementation, files.train_protocol_implementation),
        )
    else:
        if not isinstance(files, EvaluationExecutionFiles):
            raise TypeError("evaluation assignment requires EvaluationExecutionFiles")
        pairs = (
            (assignment.corpus_artifact, files.corpus_artifact),
            (assignment.model_architecture_implementation, files.model_architecture_implementation),
            (assignment.model_weights, files.model_weights),
            (assignment.evaluation_protocol_implementation, files.evaluation_protocol_implementation),
        )
    for descriptor, path in pairs:
        if not isinstance(path, Path):
            raise TypeError("materialized immutable input must be a pathlib.Path")
        _verify_descriptor_content(descriptor, path.read_bytes())


def _pre_domain_control_outcome(
    assignment: TrainingAssignment | EvaluationAssignment,
    control: _HeartbeatControl,
):
    _raise_heartbeat_fatal(control)
    cancelled, lease_lost, _ = control.snapshot()
    if lease_lost:
        return None
    if cancelled:
        return AttemptCancelled(
            attempt_id=assignment.attempt_id,
            lease_token=assignment.lease_token,
        )
    return None


def _prepare_failure_if_authorized(
    assignment: TrainingAssignment | EvaluationAssignment,
    control: _HeartbeatControl,
    failure_type: str,
    message: str,
) -> AttemptFailed | None:
    _raise_heartbeat_fatal(control)
    _, lease_lost, _ = control.snapshot()
    if lease_lost:
        return None
    return AttemptFailed(
        attempt_id=assignment.attempt_id,
        lease_token=assignment.lease_token,
        type=failure_type,
        message=message,
    )


def _attempt_failed(
    assignment: TrainingAssignment | EvaluationAssignment,
    error: Exception,
) -> AttemptFailed:
    return AttemptFailed(
        attempt_id=assignment.attempt_id,
        lease_token=assignment.lease_token,
        type=type(error).__name__,
        message=str(error),
    )


def _prepare_training_success(
    assignment: TrainingAssignment,
    candidate: TrainingExecutionCandidate,
    api: WorkerApi,
    classifier: Callable[[Exception], bool],
    control: _HeartbeatControl,
):
    try:
        content = candidate.local_weights_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != candidate.artifact.sha256.lower():
            raise ValueError("training candidate SHA-256 disagrees with artifact metadata")
        if len(content) != candidate.artifact.bytes:
            raise ValueError("training candidate byte count disagrees with artifact metadata")
    except Exception as error:
        return _prepare_failure_if_authorized(
            assignment, control, type(error).__name__, str(error)
        )

    _raise_heartbeat_fatal(control)
    if control.snapshot()[1]:
        return None
    request = CandidateUploadRequest(
        attempt_id=assignment.attempt_id,
        lease_token=assignment.lease_token,
        key=candidate.artifact.path,
        content_identity=candidate.artifact.sha256,
        content=content,
    )
    acknowledgement = _call_api_with_retry(
        lambda: api.upload_candidate(request), classifier
    )
    if acknowledgement is CandidateUploadAcknowledgement.REJECTED:
        if control.snapshot()[1]:
            return None
        return AttemptFailed(
            attempt_id=assignment.attempt_id,
            lease_token=assignment.lease_token,
            type="CandidateUploadRejected",
            message=f"candidate upload rejected: {candidate.artifact.path}",
        )
    if acknowledgement not in {
        CandidateUploadAcknowledgement.ACCEPTED,
        CandidateUploadAcknowledgement.ALREADY_PRESENT,
    }:
        raise TypeError("upload_candidate returned an unsupported acknowledgement")
    _raise_heartbeat_fatal(control)
    if control.snapshot()[1]:
        return None
    ref = CandidateArtifactRef(
        key=candidate.artifact.path,
        content_identity=candidate.artifact.sha256,
        bytes=len(content),
    )
    return TrainingSucceeded(
        attempt_id=assignment.attempt_id,
        lease_token=assignment.lease_token,
        candidate=TrainingSuccessCandidate(weights=ref),
    )


def _prepare_evaluation_success(
    assignment: EvaluationAssignment,
    result,
    api: WorkerApi,
    classifier: Callable[[Exception], bool],
    control: _HeartbeatControl,
):
    refs: dict[str, CandidateArtifactRef] = {}
    for key, path in result.artifacts.items():
        _raise_heartbeat_fatal(control)
        if control.snapshot()[1]:
            return None
        try:
            content = path.read_bytes()
        except Exception as error:
            return _prepare_failure_if_authorized(
                assignment, control, type(error).__name__, str(error)
            )
        digest = hashlib.sha256(content).hexdigest()
        request = CandidateUploadRequest(
            attempt_id=assignment.attempt_id,
            lease_token=assignment.lease_token,
            key=key,
            content_identity=digest,
            content=content,
        )
        acknowledgement = _call_api_with_retry(
            lambda request=request: api.upload_candidate(request), classifier
        )
        if acknowledgement is CandidateUploadAcknowledgement.REJECTED:
            if control.snapshot()[1]:
                return None
            return AttemptFailed(
                attempt_id=assignment.attempt_id,
                lease_token=assignment.lease_token,
                type="CandidateUploadRejected",
                message=f"candidate upload rejected: {key}",
            )
        if acknowledgement not in {
            CandidateUploadAcknowledgement.ACCEPTED,
            CandidateUploadAcknowledgement.ALREADY_PRESENT,
        }:
            raise TypeError("upload_candidate returned an unsupported acknowledgement")
        _raise_heartbeat_fatal(control)
        if control.snapshot()[1]:
            return None
        refs[key] = CandidateArtifactRef(
            key=key,
            content_identity=digest,
            bytes=len(content),
        )
    return EvaluationSucceeded(
        attempt_id=assignment.attempt_id,
        lease_token=assignment.lease_token,
        candidate=EvaluationSuccessCandidate(
            metrics=result.metrics,
            artifacts=refs,
            unavailable_outputs=result.unavailable_outputs,
        ),
    )


def _report_outcome(api: WorkerApi, outcome, classifier: Callable[[Exception], bool]) -> None:
    acknowledgement = _call_api_with_retry(
        lambda: api.report_outcome(outcome), classifier
    )
    if acknowledgement not in {
        OutcomeAcknowledgement.ACCEPTED,
        OutcomeAcknowledgement.ALREADY_FINALIZED,
        OutcomeAcknowledgement.REJECTED,
    }:
        raise TypeError("report_outcome returned an unsupported acknowledgement")


def _raise_heartbeat_fatal(control: _HeartbeatControl) -> None:
    fatal = control.snapshot()[2]
    if fatal is not None:
        raise fatal


def _is_heartbeat_fatal(error: Exception, control: _HeartbeatControl) -> bool:
    fatal = control.snapshot()[2]
    return fatal is not None and error is fatal
