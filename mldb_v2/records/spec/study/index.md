# Overview: MLDB v2 Study

- **id**: `spec:mldb.v2.study`
- **status**: draft
- **date**: 2026-09-17
- **parent**: `spec:mldb.v2`

## What this is

Study is human-authored experiment intent. Study Plan is the immutable fully materialized contract
used by the MLDB lifecycle boundary and execution backend.

Namespace remains the backend Project grouping boundary. One concrete Study Result execution may map
to a backend-native execution container such as a ClearML Pipeline Run without making that backend
container canonical.

## Topics

| ref | responsibility |
|---|---|
| `spec:mldb.v2.study.study_format` | Declarative model sources, training grids, and evaluation grids. |
| `spec:mldb.v2.study.grid_expansion` | Deterministic trial and Evaluation-coordinate expansion. |
| `spec:mldb.v2.study.plan_format` | Immutable materialized trial/evaluation-coordinate plan. |
| `spec:mldb.v2.study.source_pinning` | Git commit and scoped clean-input rules for formal planning. |
| `spec:mldb.v2.study.execution_readiness` | Plan-defined semantic dependency readiness and skip rules. |
