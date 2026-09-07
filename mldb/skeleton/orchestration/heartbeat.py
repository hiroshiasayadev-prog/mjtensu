"""Controller-side Worker heartbeat / cooperative-control orchestration boundary.

This module connects the frozen Worker heartbeat request/response values, durable Queue
lease authority/update surface, and canonical Study Run status for one logical heartbeat
operation.

The boundary is deliberately one small standalone function. It introduces no Controller
class, heartbeat manager, cancellation persistence, child-Run terminalization, Queue job
cancellation, lease-expiry recovery, Worker polling cadence, communication-retry loop,
or distributed transaction. Study cancellation request handling remains a separate
Controller operation; heartbeat observes only the already-canonical
:class:`StudyRunStatus` reached from Queue-owned attempt/job identity.
"""

from __future__ import annotations

from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..runtime.run_persistence import read_study_run
from ..study.run import StudyRunStatus
from .queue_ports import QueuePort
from .worker_api import (
    HeartbeatAccepted,
    HeartbeatRejection,
    HeartbeatRequest,
    HeartbeatResponse,
)


def handle_heartbeat(
    request: HeartbeatRequest,
    accepted_at: str,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    queue: QueuePort,
) -> HeartbeatResponse:
    """Handle one Worker heartbeat under current Queue and canonical Study authority.

    ``request`` is the frozen Worker heartbeat identity. ``accepted_at`` is the exact v1
    Queue timestamp used both for the initial current-authorization check and, only on an
    accepted path, the lease heartbeat update. This operation does not derive a clock
    value or recalculate ``lease_until`` locally.

    **Authorization is first.** Before reading parent Study state or mutating liveness,
    the handler calls::

        queue.authorized_open_attempt(
            request.attempt_id,
            request.lease_token,
            as_of=accepted_at,
        )

    ``None`` is a definitive :class:`HeartbeatRejection`. A stale, expired, closed,
    invalidated, or superseded authorization is never revived and the handler must not
    call ``queue.heartbeat_attempt`` after this result. Worker-supplied identity never
    substitutes for Queue authority.

    **Parent Study identity comes only from Queue state.** For an authorized attempt the
    handler resolves exactly::

        attempt.job_id
          -> queue.job_by_id(attempt.job_id)
          -> job.logical.coordinate.study_run
          -> read_study_run(...)

    It never accepts a Study Run ID from Worker. A definitively missing parent Queue job,
    missing canonical Study Run, or established identity/state inconsistency is a
    definitive heartbeat rejection when that fact can be distinguished safely. A
    repository/Queue/infrastructure failure that does not establish a definitive
    semantic response must not be mislabeled as a Worker rejection; it remains an
    operation failure under the frozen Worker communication semantics.

    **RUNNING Study.** When the canonical parent has
    :attr:`StudyRunStatus.RUNNING`, the handler performs the normal lease update only
    after the Study read::

        updated = queue.heartbeat_attempt(
            attempt.attempt_id,
            request.lease_token,
            accepted_at=accepted_at,
        )

    and returns::

        HeartbeatAccepted(
            lease_until=updated.lease_until,
            cancel_requested=False,
        )

    The Queue-returned ``lease_until`` is authoritative. The handler must not compute
    one hour from ``accepted_at`` itself or echo the pre-update attempt deadline.

    **CANCELLED Study.** :attr:`StudyRunStatus.CANCELLED` is durable intentional-stop
    authority. A still-authorized active attempt may intentionally remain open so its
    Worker can observe cooperative cancellation. The handler therefore performs the
    same normal ``queue.heartbeat_attempt`` lease update and returns the Queue-returned
    deadline with ``cancel_requested=True``. It does not terminalize the Training or
    Evaluation Run, close the attempt, or cancel the Queue job. The Worker later reports
    the frozen ``AttemptCancelled`` outcome through the separate outcome handler.

    **Other terminal Study states are inconsistent with active execution.** Canonical
    :attr:`StudyRunStatus.COMPLETED`,
    :attr:`StudyRunStatus.COMPLETED_WITH_FAILURES`, or
    :attr:`StudyRunStatus.FAILED` while this Worker attempt is still authorized is an
    orchestration/crash inconsistency, not a generic cooperative-stop signal. The
    handler must not extend that work as normal execution and returns a definitive
    :class:`HeartbeatRejection`. In particular, it must not reinterpret these states as
    ``cancel_requested=True``. This module adds no new public rejection enum; the frozen
    ``HeartbeatRejection.type``/``message`` surface carries the concise definitive
    reason.

    **Lease-authority serialization and races.** The required order is current Queue
    authorization -> parent Queue job resolution -> canonical Study read -> heartbeat
    lease update for RUNNING/CANCELLED only -> response construction. That authority/use
    sequence must participate in the same implementation-private concrete-attempt
    exclusion used by expired-lease recovery. The exclusion is entered before the
    authorization check and retained through the lease update or definitive non-update
    decision, so recovery cannot observe this attempt as expired and terminalize its
    canonical child while this heartbeat is still entitled to extend the old lease.

    No public lock argument or Queue mutation is added. The mechanism may be a broader
    Controller orchestration serialization or a correctly composed keyed equivalent; it
    must remain compatible with the existing same-Study dispatch/cancellation and
    reconciliation maintenance exclusions. Queue SQLite and canonical Study persistence
    are still separate stores and no cross-store transaction is introduced.

    This gives the required expiry ordering even when a heartbeat began before the
    expiry sweep and carries an earlier ``accepted_at``. If heartbeat enters first, its
    authorization/update commits before it releases the attempt exclusion, so recovery's
    later fresh Queue recheck observes the extended lease. If recovery enters first and
    closes/cancels the expired attempt before releasing, this heartbeat enters later and
    its authorization/update is stale; it must not revive the lease.

    Cancellation may still race immediately after a heartbeat reads canonical RUNNING
    and that heartbeat may validly return ``cancel_requested=False``. A later heartbeat
    observes the durable CANCELLED state and must then return ``cancel_requested=True``
    after a successful lease update. Conversely, once this operation itself has observed
    canonical CANCELLED, it must never return false for that accepted heartbeat.

    The Queue remains final lease-mutation authority after the initial authorization
    read. If authorization/update cannot complete for the same still-open current lease,
    the heartbeat is not accepted. A deterministically identifiable
    stale/closed/superseded update failure is a definitive rejection; ambiguous
    storage/service failure remains an operation failure so Worker transport semantics
    can retry the same logical heartbeat rather than receiving a false acceptance.

    This module reads Study cancellation intent only. It never calls
    ``persist_study_run_transition`` and mutates no Queue state except the ordinary
    accepted heartbeat lease update. Study cancellation request/propagation, acquire,
    candidate upload, outcome handling, lease-expiry scanning/recovery, Worker polling
    cadence, and the Worker's exact 10-second communication retry loop remain outside
    this operation.
    """

    ...
