# MLDB-V2-WORK-006: Result acceptance and Study driver

- **status**: completed
- **date**: 2026-09-09
- **depends_on**: [MLDB-V2-WORK-003, MLDB-V2-WORK-004, MLDB-V2-WORK-005]
- **source_refs**: `spec:mldb.v2.verification.result_acceptance`, `spec:mldb.v2.results.study_result_format`, `spec:mldb.v2.study.execution_readiness`, `spec:mldb.v2.api.study_driver`, `spec:mldb.v2.repository.mutation_coordination`

## Goal
Implement formal Training/Evaluation result acceptance and the resumable, idempotent Study progression engine that reconciles backend observations with immutable Plan semantics and canonical history.

## Boundary
Own result-acceptance implementation, readiness/progression internals, StageInput construction from canonical lineage, and driver-level reconciliation. Do not implement CLI presentation or backend scheduling policy.

## Task candidates
| task | responsibility | dependency |
|---|---|---|
| MLDB-V2-TASK-006-01 | Implement training terminal-candidate acceptance, canonical TrainingResult/Model construction, and rejection-to-failed-result behavior. | W003,W004 |
| MLDB-V2-TASK-006-02 | Implement evaluation terminal-candidate acceptance with exact declared metric/artifact validation and failed-result fallback. | W003,W004 |
| MLDB-V2-TASK-006-03 | Implement semantic readiness/skip derivation plus runtime Model StageInput construction for newly-ready coordinates. | W003,T01 |
| MLDB-V2-TASK-006-04 | Implement one idempotent `advance_study` reconciliation pass with child-before-parent persistence and mutation coordination. | T01,T02,T03,W005 |
| MLDB-V2-TASK-006-05 | Implement cancellation/recovery/concurrency verification covering interrupted writes, repeated callers, backend-unavailable, failed/cancelled training, and existing Models. | T04 |

## Completion condition
- Backend `completed` never directly establishes canonical success.
- Training acceptance persists TrainingResult, then Model, then parent disposition; Evaluation uses child-before-parent order.
- Semantic readiness depends only on Plan/canonical outcomes; backend ownership is consulted separately for recovery/cancellation.
- Every planned stage closes as completed/failed/cancelled/skipped and terminal Study status follows the frozen closure rules.
- Repeated/concurrent advancement cannot duplicate logical backend admission or lose canonical child/parent updates.


## Closure evidence — 2026-09-14
- T006-01 through T006-05 are completed.
- T006-05 adversarial closure found and repaired one same-ID/different-content deterministic child conflict in mldb_v2/src/study/study_driver.py; exact replay remains idempotent while conflicting valid content now fails boundedly.
- T006-05 focused closure: **15 passed**; T006-01..05 focused join: **126 passed**.
- Relevant W003/W004/W005 integration: **340 passed, 1 skipped**.
- Full python -m pytest mldb_v2/tests -q: **1254 passed, 3 skipped**.
- py_compile, exact-symbol import smoke, forbidden dependency/genericity scan, and whitespace checks: PASS.
- W006 completion conditions are satisfied: backend completion never bypasses acceptance, child-before-parent persistence/recovery holds, readiness remains canonical/Plan-driven, cancellation/recovery/concurrency semantics are verified, and repeated callers cannot lose canonical updates or duplicate logical backend admission.
- **MLDB-V2-WORK-006 is completed.**
