# MLDB-V2-TASK-007-05: Verify Application / Query API closure

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-007
- **task_type**: verification
- **depends_on**: [MLDB-V2-TASK-007-04, MLDB-V2-WORK-006]
- **outputs**: W007 integrated regression, W006 join verification, adversarial API verification, full mldb_v2 regression

## Closure gate clarification
W006 formal completion is required here. This is deliberately not an implementation-start dependency for T007-01/02/03.

## Exclusive ownership
May modify only:
- `mldb_v2/tests/test_api_conformance.py`
- this Task record
- `mldb_v2/records/work-items/MLDB-V2-WORK-007-application-query-api.md` only for coordinator-approved closure evidence/status

Do not modify implementation unless a BLOCK NOW finding is handed back to the owning T007 implementation Task. Do not modify `records/tasks/index.md`, W006, CLI, Specs, or Skeletons.

## Goal
Close W007 once public API shape, read/mutation boundaries, W006 integration, generic backend behavior, and stable error categories are proven together.

## Verification
- Run focused T007 suites first, then T007 integrated regression, relevant W001/W002/W003/W005/W006 joins, adversarial verification, and finally full `mldb_v2/tests`.
- Prove query/observe/log/diagnose never mutate canonical/backend execution state.
- Prove `start_study` never performs backend admission and `run`/`resume` progress only through the exact W006 one-pass primitive.
- Prove public error categories remain exact and bounded; unsupported optional logs map to `unsupported_capability`.
- Prove discovery-first behavior requires no caller filesystem search and filters/order match frozen contracts.
- Prove rerun uses the exact source Plan, not the current Study definition.
- Treat only frozen-shape violations, mutation-boundary breaches, W006 duplicate logic, genericity breaks, canonical corruption, or real dependency blockers as BLOCK NOW. Record nonsemantic polish as deferred findings without reopening completed Tasks.

## Completion condition
All W007 contract obligations pass against completed W006/W005 production seams and full `mldb_v2/tests` is green. Only then may the coordinator mark the W007 Work Item completed.

No commit/stage/stash/reset/clean/restore/push.
## Closure evidence — 2026-09-14
- Re-ran the repair-state T007 focused API join: `test_api_authoring.py`, `test_api_query.py`, `test_api_execution.py`, and `test_api_application_integration.py` -> **51 passed in 17.72s**.
- Added `mldb_v2/tests/test_api_conformance.py`; closure conformance -> **6 passed**. Combined T007 closure-focused API suites -> **57 passed in 17.85s**.
- Regression coverage fixes the adversarial category boundary: malformed caller request -> `invalid_request`; well-formed missing identity -> `not_found`; lifecycle conflict -> `lifecycle_conflict`; canonical/domain corruption -> `validation_failed`; internally generated invalid UUID is not relabeled as caller `invalid_request`.
- Conformance also proves exact Frozen Application/Query/Error mirrors, all 10 stable public error codes, read-only query behavior, optional-log capability handling, and Application cancellation request followed by W006-owned terminal closure.
- Relevant W001/W002/W003/W005/W006 integration join -> **977 passed, 1 skipped in 268.23s**.
- Adversarial/static verification PASS: Frozen mirrors, Python `py_compile`, no runtime Skeleton dependency, backend/CLI genericity, W006 progression non-duplication, identity/parsing scan, whitespace/BOM.
- Full `python -m pytest mldb_v2/tests -q` -> **1299 passed, 3 skipped in 326.82s**.
- The 3 full-suite skips are host capability skips for symlink creation in evaluation artifact runtime, executable asset directory verification, and storage artifact publication; no W007 regression is skipped.
- No implementation files changed during T007-05. No commit/add/stage/stash/reset/clean/restore/checkout/worktree/push performed.
