# Overview: MLDB v2 formal results

- **id**: `spec:mldb.v2.results`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2`

## What this is

Defines canonical execution history that survives backend loss.

Training and Evaluation Result payload formats are owned by their domain topics. This topic defines
their common history semantics and Study-level aggregation.

## Topics

`spec:mldb.v2.results.formal_history` defines submission/attempt history semantics.
`spec:mldb.v2.results.attempt_summary` defines the backend-neutral terminal attempt value.
`spec:mldb.v2.results.study_result_format` defines complete planned-stage disposition and Study
closure.
