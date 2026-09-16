# Contract: Execution backend port

- **id**: `spec:mldb.v2.backend.backend_port`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.backend`
- **contract_class**: `port`

## Required capabilities

A backend adapter supports these semantic capabilities:

- idempotently admit one MLDB-ready `spec:mldb.v2.backend.stage_input`;
- return/recover opaque backend execution identity for its logical attempt(s);
- query enough state to distinguish active from terminal backend work;
- collect terminal candidates conforming to `spec:mldb.v2.backend.candidate_outcome`;
- request cancellation of admitted work for one Study execution.

The application Study driver, not the backend adapter, decides which Plan coordinates are ready.
The backend never receives authority to invent, omit, or reorder formal Plan semantics.

## Optional observational capabilities

A backend MAY expose live/retained attempt logs, richer telemetry, or UI-navigation metadata for
read-only application queries. Generic MLDB must capability-check these features; lack of them does
not make the backend unable to execute formal Studies.
## Admission identity

Logical admission is keyed by exact Study Result ID plus trial and stage coordinate. A recoverable
client failure or repeated `advance_study` call MUST NOT create duplicate logical work merely because
the caller did not receive the first admission response.

## Rules

- The port consumes canonical/compiled values, not raw backend-specific configuration dictionaries.
- Backend IDs are opaque strings to generic MLDB and are provenance only.
- Backend retry, resource scheduling, queueing, and liveness remain backend concerns.
- Backend-created attempts invoke `spec:mldb.v2.backend.execution_harness`; adapters do not invent a
  second model-family-specific execution path.
- Backend telemetry/logs are observational only unless accepted by a formal result contract.
- Backend `completed` is never sufficient for canonical MLDB completion; result acceptance still runs.
- The port exposes observations/candidates needed by `spec:mldb.v2.api.study_driver` but does not own
  Study Result mutation.
