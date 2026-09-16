# MLDB-V2-ADR-STUDY-002: Expand evaluation parameters within planned trials

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
  - MLDB-V2-ADR-STUDY-001
  - MLDB-V2-ADR-BACKEND-001
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.study.grid_expansion`

## Context

A fixed Evaluation Protocol can expose public parameters such as thresholds or matching settings.
Forcing every value combination into a separate Study would fragment one comparison intent, while
turning evaluation parameters into model trials would incorrectly imply retraining.

ClearML can filter and compare selected Tasks and visualize parameter/metric relationships, so a
moderate number of evaluation coordinates does not require flattening them into separate Projects.
## Decision

Evaluation public parameters MAY be expressed as explicit value lists in Study evaluation stages.
They expand deterministically into Evaluation coordinates beneath each already-materialized model
trial. Evaluation expansion never creates additional training/model trials.

Every Study MUST contain at least one evaluation stage. Training-only and existing-model no-op
Studies are outside the initial formal Study contract.

## Consequences

Study Plan records every Evaluation coordinate explicitly. ClearML maps one executed Evaluation
coordinate attempt to one Task and exposes the coordinate parameters for filtering/comparison.
Dynamic or backend-generated evaluation search remains outside the initial contract.
