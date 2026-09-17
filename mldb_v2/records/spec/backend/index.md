# Overview: MLDB v2 execution backend

- **id**: `spec:mldb.v2.backend`
- **status**: draft
- **date**: 2026-09-17
- **parent**: `spec:mldb.v2`

## What this is

Defines the operational boundary for realizing one immutable Study Plan as one backend-owned Study
execution without making backend state canonical. Native pipelines/controllers may own physical child
scheduling while MLDB retains semantic gates and formal result acceptance.

## Topics

| ref | responsibility |
|---|---|
| `spec:mldb.v2.backend.backend_port` | Backend-neutral Study execution, child observation/candidate, cancellation, and operational ownership contract. |
| `spec:mldb.v2.backend.stage_input` | Immutable backend-neutral execution input for one semantically-released Plan coordinate. |
| `spec:mldb.v2.backend.candidate_outcome` | Exact backend-neutral terminal candidate and stage-key contract. |
| `spec:mldb.v2.backend.execution_harness` | Backend-neutral process entrypoint for one planned child stage. |
| `spec:mldb.v2.backend.clearml_mapping` | ClearML Project/Pipeline/Task/telemetry/artifact mapping. |
