# Overview: MLDB v2 execution backend

- **id**: `spec:mldb.v2.backend`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2`

## What this is

Defines the minimal operational boundary required to execute ready coordinates from an immutable
Study Plan without making the backend canonical.

## Topics

| ref | responsibility |
|---|---|
| `spec:mldb.v2.backend.backend_port` | Backend-neutral ready-stage admission/observation/cancel/collection contract. |
| `spec:mldb.v2.backend.stage_input` | Immutable backend-neutral execution input for one ready Plan coordinate. |
| `spec:mldb.v2.backend.candidate_outcome` | Exact backend-neutral terminal candidate and stage-key contract. |
| `spec:mldb.v2.backend.execution_harness` | Backend-neutral process entrypoint for one planned stage coordinate. |
| `spec:mldb.v2.backend.clearml_mapping` | Initial ClearML Project/Task/parameter/artifact mapping. |
