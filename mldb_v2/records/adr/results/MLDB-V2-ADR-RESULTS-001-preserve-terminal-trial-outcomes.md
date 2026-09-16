# MLDB-V2-ADR-RESULTS-001: Preserve terminal trial outcomes including failures and cancellations

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
  - MLDB-V2-ADR-STUDY-001
  - MLDB-V2-ADR-STORAGE-001
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.results.formal_history`

## Context

If Git stores only successful results, a Study or Study Plan with no result is ambiguous: it may
never have been submitted, it may have failed, or it may have been cancelled. That ambiguity is
unacceptable for formal experiment history.

At the same time, backend retries, heartbeats, live logs, and transient task state should not be
reimplemented as an MLDB state machine.

## Decision

Every formally submitted Study Plan creates canonical execution-history identity before backend
work is considered owned by MLDB.

Terminal Training and Evaluation records preserve `completed`, `failed`, or `cancelled` outcome.
Successful records contain their formal result payload. Failed and cancelled records contain
diagnostic/provenance summaries but do not pretend to contain a successful result.

If a logical stage has multiple backend attempts, the formal record preserves an ordered summary
of those attempts and their terminal statuses. Backend logs remain backend-owned.

The Study execution record aggregates every planned trial and closes as one of:

- `completed`;
- `completed_with_failures`;
- `failed`;
- `cancelled`.

A submitted Study execution that has not yet been collected remains explicitly non-terminal rather
than disappearing from history.

## Rationale

Git can distinguish "never run" from "ran and failed/cancelled" without duplicating backend
operational machinery.

## Rejected alternatives

### Store only successful results in Git

This loses negative experiment history and makes missing results ambiguous.

### Mirror every backend state transition into Git

This recreates the v1 operational lifecycle and produces noisy mutable history.

## Consequences

Submission and collection have small canonical-history responsibilities even though scheduling and
attempt execution belong to the backend.
