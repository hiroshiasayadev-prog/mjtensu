# MLDB-V2-ADR-RESULTS-002: Close every planned stage explicitly

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
  - MLDB-V2-ADR-RESULTS-001
  - MLDB-V2-ADR-STUDY-001
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.results.study_result_format`

## Context

Recording failed/cancelled attempted stages is insufficient when upstream failure prevents a
planned downstream Evaluation from ever receiving an attempt. A Plan coordinate would again exist
without a formal explanation of its outcome.

## Decision

Study Result represents every planned stage with an explicit disposition. Terminal dispositions
are `completed`, `failed`, `cancelled`, or `skipped`. `skipped` records why no child execution result
exists, such as upstream failure or Study cancellation.
A never-attempted skipped stage does not get a synthetic Training/Evaluation Result. The Study
Result itself closes that planned coordinate.

## Consequences

A terminal Study Result is a complete explanation of its immutable Plan. Missing child Result files
are no longer ambiguous when the corresponding planned stage is explicitly skipped.
