# Contract: Formal execution history

- **id**: `spec:mldb.v2.results.formal_history`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.results`
- **contract_class**: `lifecycle`

## Submission boundary

Formal start of a Study Plan allocates and persists its Study Result before backend work is admitted.
A Study definition/Plan with no Study Result was never formally started through MLDB v2.

The Study Result starts with every planned stage represented as `pending`. Backend state transitions
are not mirrored continuously; `spec:mldb.v2.api.study_driver` changes only the canonical
dispositions required to explain the immutable Plan.

Every planned stage must be non-pending before terminal Study closure. A stage prevented from ever
running is explicitly `skipped` with a reason rather than silently absent.

## Attempt summaries

Each actually executed logical stage has an ordered list of backend attempts. Attempt summary shape
is defined by `spec:mldb.v2.results.attempt_summary`. A later successful retry never erases earlier
failed/cancelled attempts. Retry policy, heartbeat, liveness, and intermediate state remain backend
concerns.
## Child-result rule

An attempted terminal training/evaluation stage has exactly one canonical Training/Evaluation Result
for the logical stage; that result owns all terminal attempt summaries. A never-attempted skipped
stage has no synthetic child Result.

Terminal Training Results, Models, Evaluation Results, and terminal Study Results are immutable.
Backend deletion does not delete or rewrite canonical history.
