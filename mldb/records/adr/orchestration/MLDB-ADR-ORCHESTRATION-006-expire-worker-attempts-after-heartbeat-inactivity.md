# MLDB-ADR-ORCHESTRATION-006: Expire Worker attempts after heartbeat inactivity

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-ORCHESTRATION-003, MLDB-ADR-ORCHESTRATION-005
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

MLDB Worker communication is pull-oriented and retryable Controller communication failures are retried indefinitely by the Worker every 10 seconds.

Queue attempts nevertheless need a finite stale-Worker rule so Controller can eventually recover work when a Worker process disappears or remains partitioned from Controller.

A fixed expiry measured only from initial assignment would incorrectly expire healthy long-running training even while heartbeat communication continues. The expiry must therefore move forward with successful Worker liveness communication.

The project does not need a sophisticated distributed-failure detector. A conservative timeout is preferable because ML training can be expensive and unnecessary duplicate training should be avoided.

## Decision

An active Worker attempt becomes stale only after one continuous hour without a successfully accepted heartbeat for its current lease.

Each successfully accepted heartbeat resets the inactivity window to one hour from that heartbeat.

Conceptually:

```text
last accepted heartbeat
        + 1 hour
        = current lease expiry
```

The initial assignment establishes the first liveness reference for the attempt until the first heartbeat is accepted.

Worker communication failure does not locally fail the domain execution. Worker continues retrying the same heartbeat operation every 10 seconds according to MLDB-ADR-ORCHESTRATION-005.

Controller remains authoritative for lease expiry. When one hour elapses without an accepted heartbeat, Controller may invalidate the lease and handle the corresponding child Run as an interrupted unsuccessful attempt according to the Queue lifecycle contract.

A heartbeat or result arriving after Controller has definitively invalidated or superseded the lease does not restore that authorization.

The one-hour inactivity timeout is a v1 orchestration constant rather than a Study, Train Protocol, Evaluation Protocol, or Worker-supplied parameter.

This decision does not fix the normal heartbeat cadence while communication is healthy. It only fixes communication retry cadence at 10 seconds under MLDB-ADR-ORCHESTRATION-005 and stale-attempt detection at one hour without an accepted heartbeat.

## Rationale

Sliding expiry from the latest accepted heartbeat allows arbitrarily long healthy training or evaluation attempts without extending leases through prediction of total job duration.

One hour is deliberately conservative relative to transient network interruptions and Controller restart, reducing the chance that an expensive still-running training attempt is replaced unnecessarily.

Keeping the timeout at orchestration level avoids contaminating ML experiment definitions with distributed-execution tuning.

## Rejected alternatives

### Expire one hour after initial assignment regardless of heartbeat

Long-running healthy training would expire even while Worker liveness remained continuously observable.

### Treat any failed heartbeat request as Worker failure

Transient Controller or network failure would convert communication problems into false domain failures and could duplicate expensive training.

### Never expire active attempts

A crashed or permanently disconnected Worker could leave a Study Run blocked indefinitely with a canonical child Run stuck `running`.

### Make the timeout configurable per Study or Protocol

Worker liveness is operational orchestration behavior rather than experiment meaning.

## Consequences

Queue `lease_until` represents the current inactivity deadline for the active attempt.

Assignment initializes that deadline to one hour after assignment establishment.

Every accepted heartbeat replaces it with one hour after the accepted heartbeat time.

Worker continues 10-second communication retry during an outage until it receives a definitive response or is intentionally stopped.

A communication outage shorter than one hour does not by itself force a new child Run.

After one hour without accepted heartbeat, Controller may close the current attempt, terminalize the still-running child Run as interrupted, and schedule a later retry with a new Run ID when policy permits.

## Evidence

Training work is substantially more expensive than Queue bookkeeping, so conservative stale detection is preferable to eager duplicate execution.

The existing Queue lifecycle already separates Worker communication state from canonical child Run history and requires stale leases to be resolved before retry.
