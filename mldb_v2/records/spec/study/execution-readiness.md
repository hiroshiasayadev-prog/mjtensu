# Contract: Planned-stage readiness

- **id**: `spec:mldb.v2.study.execution_readiness`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.study`
- **contract_class**: `lifecycle`

## Purpose

MLDB owns semantic dependencies implied by the immutable Plan; the backend owns scheduling only
after a stage is admitted. This is dependency progression, not a second resource scheduler.

## Readiness

For a training-source trial, the training stage is initially ready. Its Evaluation coordinates are
blocked until the Training Result is canonically `completed` and the deterministic Model exists.
For an existing-Model trial, all Evaluation coordinates are initially ready after Model integrity is
validated.

Evaluation coordinates are siblings: failure of one does not block another.
## Terminal upstream handling

If training becomes `failed`, every still-unattempted dependent Evaluation coordinate becomes
`skipped: upstream_failed`. If training becomes `cancelled`, they become
`skipped: upstream_cancelled`.

When Study Result status is `cancelling`, no new stage is ready for admission. Distinguishing
never-admitted pending stages from backend-admitted pending stages is not part of semantic readiness,
because admission ownership is operational rather than canonical. The application Study driver first
observes deterministic backend ownership for pending stage keys: never-admitted stages become
`skipped: study_cancelled`, while admitted work receives cancellation through the backend port and is
later collected normally.

The readiness boundary derives dependency readiness and upstream skip consequences only from Plan
intent plus canonical accepted outcomes. Backend queue order, worker state, retry policy, and current
admission ownership never redefine those dependencies. Admission ownership is consulted separately by
the Study driver only for idempotent admission recovery and cancellation handling.
