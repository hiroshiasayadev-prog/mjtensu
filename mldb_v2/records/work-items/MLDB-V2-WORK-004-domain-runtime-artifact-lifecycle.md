# MLDB-V2-WORK-004: Domain runtime and artifact lifecycle

- **status**: completed
- **date**: 2026-09-09
- **depends_on**: [MLDB-V2-WORK-001, MLDB-V2-WORK-002]
- **source_refs**: `spec:mldb.v2.training`, `spec:mldb.v2.evaluation`, `spec:mldb.v2.backend.execution_harness`, `spec:mldb.v2.backend.stage_input`

## Goal
Implement the backend-neutral ML execution runtime that materializes one StageInput into a provisional terminal candidate using the frozen Train/Evaluation callable contracts and canonical artifact rules.

## Boundary
Own runtime behavior under `mldb_v2/src/training/`, `src/evaluation/`, and execution/materialization helpers required by the common harness, excluding the reusable TrainProtocol/EvaluationProtocol definition and Train/Evaluation callable-interface modules established by W002. W004 reuses the W002 companion-loading/interface boundary instead of creating another import convention. Do not schedule work, mutate StudyResult progression, or decide result acceptance.

## Task candidates
| task | responsibility | dependency |
|---|---|---|
| MLDB-V2-TASK-004-01 | Implement runtime Architecture build invocation plus canonical PyTorch state-dict validation/serialization/loading and Model lineage loading, reusing the W002 companion-loader boundary. | W002 |
| MLDB-V2-TASK-004-02 | Implement sealed Corpus/artifact materialization, formal candidate artifact publication/integrity helpers, and the configured concrete S3-compatible object-byte transport adapter used by the execution harness. | W001,W002 |
| MLDB-V2-TASK-004-03 | Implement Training StageInput preflight, exact TrainContext construction, protocol invocation, strict returned-module compatibility, and training candidate creation. | T01,T02 |
| MLDB-V2-TASK-004-04 | Implement Evaluation StageInput preflight, canonical Model loading, exact EvaluationContext invocation, and evaluation candidate creation/publication. | T01,T02 |
| MLDB-V2-TASK-004-05 | Compose the common execution harness and verify classifier/detector/custom-evaluation execution paths with bounded fixtures. | T03,T04 |

## Completion condition
- One common harness handles both training and evaluation variants without model-family branches in MLDB core.
- Canonical weights are plain CPU Tensor state dicts and strict-load into a fresh Architecture.
- Protocols receive no backend IDs, credentials, raw object-store clients, or canonical output paths.
- Evaluation supports arbitrary declared scalar metrics and arbitrary declared file artifacts.
- W004 includes at least one production configured S3-compatible object-byte transport usable by the execution harness; fake/in-memory transport alone does not satisfy W004 completion.
- Harness success remains provisional candidate success; it never writes formal Training/Evaluation Results.


## Phase A join  E2026-09-13
- T004-01 is completed after focused verification and coordinator shared-tree join verification.
- T004-02 code is complete and focused-green; only real boto3 dependency activation/client-configuration smoke remains before T004-02/W004 closure.
- Shared-tree Phase A focused join: **40 passed, 1 skipped**; the skip is the Windows symlink-permission storage probe.
- T004-02's remaining operational activation does not block T004-03/T004-04 implementation because their consumed storage/runtime surface is already code-complete and verified.
- Phase B is released in parallel: T004-03 Training runtime and T004-04 Evaluation runtime. Full regression remains deferred to T004-05/W004 closure.
## Phase B join  E2026-09-13
- T004-03 and T004-04 are completed after their focused verification and coordinator shared-tree join.
- Shared-tree Phase B focused join: **74 passed, 1 skipped**; the skip is the known Windows symlink-permission probe.
- Training and Evaluation runtimes remain file-disjoint and both return only frozen domain candidate payloads; neither fabricates backend AttemptSummary/execution IDs or writes canonical Results.
- T004-05 common harness / W004 closure is released now. T004-02's real boto3 dependency activation/client-configuration smoke remains a final W004 closure gate, not an implementation blocker for T004-05 composition.

## Phase C / closure status  E2026-09-13
- T004-05 common harness implementation and W004 regression are green: focused **24 passed**; W004 integrated **219 passed, 2 skipped**; W001/W002/W003 smoke **321 passed**; full `mldb_v2/tests` **1056 passed, 3 skipped, 0 failed**.
- `py_compile`, import, genericity/forbidden-dependency scan, diff check, and bounded adversarial review are green with no blocking W004 code finding.
- T004-05 is completed. T004-02 remains the only non-completed W004 Task.
- Active-environment recheck confirms `boto3` absent and no usable AWS/S3/MinIO configuration available, so the required production S3-compatible activation/configuration smoke cannot run yet.
- Exact current state: **W004 implementation/regression complete; closure blocked only on production boto3 dependency activation/config smoke**.
- W005/W006 implementation may continue on the already-frozen runtime seams; W004-dependent Work Item completion gates must remain open until this final operational smoke is satisfied.


## Final closure — 2026-09-13
- The final T004-02 production transport gate is satisfied against live `https://mldb-s3.thebugrat.dev` backed by MinIO Community Edition (`GNU AGPLv3`).
- Actual W004 S3 transport smoke passed: bucket creation, immutable PUT, GET, idempotent same-byte replay, different-byte overwrite rejection, and cleanup; result **`W004_S3_SMOKE=PASS`**.
- No production credentials were written to canonical values or records; local `.env` remains ignored.
- W004 regression remains **1056 passed, 3 skipped, 0 failed** with bounded adversarial review finding no blocker.
- T004-01 through T004-05 are now completed. **MLDB-V2-WORK-004 is completed.**
- W005/W006 gates that depended on W004 completion are released.
