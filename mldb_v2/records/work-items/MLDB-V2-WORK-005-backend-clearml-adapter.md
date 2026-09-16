# MLDB-V2-WORK-005: Backend and ClearML adapter

- **status**: completed
- **date**: 2026-09-09
- **depends_on**: [MLDB-V2-WORK-004]
- **source_refs**: `spec:mldb.v2.backend.backend_port`, `spec:mldb.v2.backend.stage_input`, `spec:mldb.v2.backend.candidate_outcome`, `spec:mldb.v2.backend.clearml_mapping`

## Goal
Implement the generic execution-backend contract and the first ClearML adapter for idempotent stage admission, observation, terminal candidate collection, cancellation, and optional logs without leaking ClearML semantics into canonical MLDB identity.

## Boundary
Own backend implementation under `mldb_v2/src/backend/`, including ClearML-specific adapter modules. The backend schedules admitted work only; it does not decide Plan readiness or mutate canonical Study history.

## Task candidates
| task | responsibility | dependency |
|---|---|---|
| MLDB-V2-TASK-005-01 | Implement backend registry/configuration and generic BackendPort integration around frozen StageKey/StageInput/CandidateOutcome values. | W004 |
| MLDB-V2-TASK-005-02 | Implement ClearML deterministic admission identity, project/task metadata mapping, and remote common-harness launch. | T01 |
| MLDB-V2-TASK-005-03 | Implement ClearML active/terminal observation and ordered attempt/candidate recovery without parsing human Task names. | T02 |
| MLDB-V2-TASK-005-04 | Implement Study cancellation and optional log-reading capability through ClearML. | T02,T03 |
| MLDB-V2-TASK-005-05 | Verify idempotent admission/recovery and adapter conformance with mocked plus available real ClearML integration. | T03,T04 |

## Completion condition
- MLDB Namespace maps to ClearML Project; Study/Plan identity is metadata, not a Project-per-Study convention.
- ClearML IDs remain opaque provenance only.
- Repeated admission of one StageKey cannot create duplicate logical work.
- Queueing, retry, heartbeat, worker/GPU scheduling remain ClearML responsibilities.
- Endpoint/credentials/queue configuration never appears in canonical Plan/Result values.

## Closure evidence — 2026-09-13
- T005-01 through T005-05 are completed.
- Frozen BackendPort/ClearML mapping completion conditions are satisfied under mocked coverage and bounded authenticated real ClearML verification.
- Final focused W005 verification: **85 passed**; full `mldb_v2/tests`: **1227 passed, 3 skipped**.
- Real ClearML admission/recovery/observe/cancel/collect passed with no duplicate logical work and no credential leakage into canonical values.
- One `default` queue exists but ClearML currently reports **0 workers**; remote harness consumption remains an operational deployment prerequisite for later real execution, not a W005 code blocker.
