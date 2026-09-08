"""Minimal retry-disposition decision boundary for MLDB orchestration.

This module fixes only the policy seam needed after unsuccessful logical work: whether
another concrete attempt is permitted and, when it is, the exact Queue timestamp at
which that retry becomes eligible. Queue mutation remains owned by Controller through
the frozen :class:`QueuePort` operations.

The boundary deliberately defines no retry limit, attempt budget, backoff formula,
priority rule, failure taxonomy, Worker-specific rule, Model-family rule, scheduler,
or concrete default policy. Queue attempt history is operational and may be incomplete
after Queue database loss; it is never promoted to canonical Study or child-Run truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from .queue import QueueAttempt, QueueJob


UnsatisfiedAttemptOutcome: TypeAlias = Literal[
    "failed",
    "cancelled",
    "completed_partial",
]
"""Terminal child outcomes that do not satisfy one logical Queue job.

``failed`` and ``cancelled`` apply to both Training and Evaluation attempts.
``completed_partial`` applies only to Evaluation. A completed Training Run with its
required deterministic Model and a ``completed`` Evaluation Run are satisfying facts
and therefore never enter this policy boundary.
"""


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Minimal storage-ready retry disposition selected by orchestration policy.

    ``retry_not_before is not None`` permits another concrete attempt and supplies the
    exact value Controller passes to Queue ``RETRY_WAIT`` mutation. The timestamp uses
    the frozen Queue storage encoding: UTC RFC3339 text with exactly six fractional
    digits and a ``Z`` suffix (``YYYY-MM-DDTHH:MM:SS.ffffffZ``).

    ``retry_not_before is None`` means no further retry is permitted. It intentionally
    does not choose the caller's applicable terminal Queue state: in particular, a
    cancelled child may lead to Queue ``CANCELLED`` or ``FAILED`` according to the
    later orchestration stop reason.
    """

    retry_not_before: str | None


class RetryPolicy(Protocol):
    """Consumer-oriented policy for unsuccessful logical work disposition.

    Implementations return decisions only. They do not mutate Queue state, allocate or
    terminalize Runs, accept candidates, create Models, finalize Studies, acknowledge
    Worker operations, validate leases, create attempts, or decide whether a successful
    child result satisfies its logical job.

    ``at`` and every returned non-``None`` retry timestamp use the exact Queue storage
    timestamp encoding. This protocol does not require byte-identical replay or pure
    determinism beyond the explicit decision value contract.
    """

    def after_preflight_failure(
        self,
        job: QueueJob,
        prior_attempts: tuple[QueueAttempt, ...],
        *,
        at: str,
    ) -> RetryDecision:
        """Decide disposition after concrete pre-Run/preflight failure.

        This path occurs while the selected logical job is still ``READY`` and before
        child Run allocation, so the failing event has no child Run and no Queue
        attempt of its own. ``prior_attempts`` is the operational history returned by
        ``QueuePort.attempts_for_job(job.job_id)`` before this failure. It may be empty
        or incomplete after Queue reconstruction and must not be treated as canonical
        execution history.

        No exception object or failure-category hierarchy is part of this Wave's public
        policy input. ``RetryDecision(retry_not_before=...)`` maps mechanically to
        ``defer_ready_job()``, while ``RetryDecision(None)`` maps to
        ``fail_ready_job()`` in the later Controller acquire handler.
        """

        ...

    def after_unsatisfied_attempt(
        self,
        job: QueueJob,
        attempt_history: tuple[QueueAttempt, ...],
        outcome: UnsatisfiedAttemptOutcome,
        *,
        at: str,
    ) -> RetryDecision:
        """Decide disposition after canonical child work remains unsatisfied.

        Controller calls this only after canonical child history establishes the
        represented terminal unsatisfied outcome. ``attempt_history`` is retained
        operational history from ``QueuePort.attempts_for_job(job.job_id)`` when such
        history still exists; it may be empty or incomplete after Queue database loss
        or reconstruction and is not canonical child lineage truth. No current
        ``QueueAttempt`` is required merely to make the retry decision.

        In normal active-attempt handling, the caller keeps the separately authorized
        current attempt for lease validation and later ``close_attempt()`` while this
        policy receives the retained job attempt history. During reconciliation, the
        same policy method remains callable for an unsuccessfully terminal child Run
        even when the concrete attempt row was lost, or when a cross-store crash left a
        canonical ``RUNNING`` child Run before Queue attempt activation and Controller
        later terminalized that orphaned child unsuccessfully. Reconciliation must not
        fabricate attempt IDs, Worker/lease/acquire identities, or attempt numbers to
        call this boundary.

        ``outcome`` is deliberately restricted to ``failed``, ``cancelled``, or
        Evaluation-only ``completed_partial`` rather than accepting the complete
        Training/Evaluation lifecycle enums. A stale or expired lease whose canonical
        running child Run is first terminalized as ``FAILED`` reuses this same method
        with ``outcome="failed"``; no lease-expiry-specific policy method is needed.
        Queue loss or orphan recovery likewise describes recovery circumstances rather
        than new retry outcomes.

        A non-``None`` retry timestamp means another attempt is permitted and maps to
        ``close_attempt(..., target_status=RETRY_WAIT, retry_not_before=...)`` when a
        current attempt exists, or to the equivalent reconciliation repair of the
        logical job when it does not. ``None`` means only that no retry remains: the
        later Controller orchestration chooses the applicable terminal Queue operation.
        In normal retained-attempt handling that is normally ``close_attempt(...,
        target_status=FAILED)`` or, for an intentional logical stop, ``cancel_job(...)``;
        reconciliation applies the corresponding terminal job repair. This policy does
        not choose ``CANCELLED`` versus ``FAILED`` from a cancelled child alone.
        """

        ...
