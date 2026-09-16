# MLDB-V2-WORK-007: Application and query API

- **status**: completed
- **date**: 2026-09-13
- **depends_on**: [MLDB-V2-WORK-006]
- **source_refs**: `spec:mldb.v2.api.application_interface`, `spec:mldb.v2.api.query_interface`, `spec:mldb.v2.api.errors`

## Dependency interpretation
`depends_on: W006` is the W007 integration/closure gate, not an implementation-start gate. Frozen Application/Query/Error contracts are already fixed, so contract-first work that depends only on W001/W002/W003/W005 boundaries may proceed before W006 closes. W007 never reimplements the W006 Study progression state machine.

## Goal
Implement the single transport-independent Application boundary used by CLI/Python/future adapters, including discovery, authoring checks, execution control, observation, logs, and diagnostics.

## Boundary
Own `mldb_v2/src/api/`. Compose lower layers only through their established boundaries. Query operations remain read-only; `run`/`resume` are foreground loops over the same W006 one-pass progression primitive. W008 owns `mldb_v2/src/cli/` independently.

## Parallel implementation DAG
```text
T007-01 authoring/planning ------------------+
T007-02 read-only query/observation ----------+-> T007-04 concrete Application integration -> T007-05 closure
T007-03 execution composition via fake seam --+                 ^                                  ^
                                                               |                                  |
T005-05/W005 production backend -------------+-----------------+                                  |
T006-04 actual one-pass primitive ------------+-----------------+                                  |
W006 formal closure -------------------------------------------------------------------------------+
```

## Tasks
| task | responsibility | start dependency |
|---|---|---|
| MLDB-V2-TASK-007-01 | Stable error boundary plus validate/verify/seal/plan composition. | W001,W002,W003 |
| MLDB-V2-TASK-007-02 | Runtime QueryInterface mirror plus read-only discovery/progress/observation/logs/diagnostics. | W001,W005 contract; production join at T005-05 |
| MLDB-V2-TASK-007-03 | start/advance/run/resume/rerun/cancel Application-side composition against injected W006 seam. | W001,W003,T006-01..03; fake seam before T006-04 |
| MLDB-V2-TASK-007-04 | Runtime ApplicationInterface mirror and concrete application integration. | T007-01..03,T005-05,T006-04 |
| MLDB-V2-TASK-007-05 | Integrated/adversarial/full regression and W007 closure. | T007-04,W006 closed |

## Completion condition
- Empty validate/verify scopes work; bulk sealing requires explicit bulk intent.
- `start_study` durably allocates history without initial backend admission.
- `run`/`resume` progress Studies only through the W006 one-pass primitive; observation/query methods never do.
- Run/model/result discovery never requires caller-side filesystem search.
- Optional backend logs return `unsupported_capability` when absent rather than synthesized data.
- W006 progression/acceptance/readiness/cancellation algorithms are not duplicated under `src/api`.
- W007 closure runs aggregate regression/adversarial verification/full `mldb_v2/tests`; per-Task implementation uses focused tests only.

## Closure evidence — 2026-09-14
- T007-01..04 focused API join after T007-03 adversarial repair: **51 passed**; T007-05 conformance: **6 passed**; combined closure-focused API verification: **57 passed**.
- Public execution request error categories are separated from canonical/domain failures: malformed request -> `invalid_request`, missing canonical identity -> `not_found`, lifecycle conflict -> `lifecycle_conflict`, canonical/domain validation -> `validation_failed`.
- Relevant W001/W002/W003/W005/W006 production integration join: **977 passed, 1 skipped**.
- Frozen mirror, `py_compile`, Skeleton dependency, genericity, identity/parsing, W006 non-duplication, and whitespace/BOM adversarial checks all PASS.
- Full `mldb_v2/tests`: **1299 passed, 3 skipped in 326.82s**; skips are host symlink-capability skips, not W007 semantic coverage gaps.
- Query remains read-only; start remains zero-admission; run/resume use W006 one-pass progression; rerun preserves the exact source Plan; cancellation request ownership remains Application-side with later progression/closure owned by W006.
- W007 completion conditions are satisfied against completed W005/W006 production seams. No implementation change was required in T007-05.

## Post-closure production composition repair evidence — 2026-09-14
- W008 closure verification exposed the missing production composition seam; `ApplicationComposition` / `compose_application()` have been added without changing the Frozen Application contract or W006 semantics.
- `default_backend` is `str | None`; `MLDB_V2_DEFAULT_BACKEND` is the only new selector. ClearML registration/configuration lives only in `src/api/composition.py`, with backend and S3 activation remaining lazy.
- Updated the two stale W007 static assertions to match the production architecture: exact 7-symbol public exports, all-API Skeleton/CLI prohibition, and ClearML prohibition on generic API modules with only `composition.py` excluded.
- Focused production-composition verification: **29 passed** total = Application integration **16**, API conformance **6**, composition **7**.
- Static verification PASS: `py_compile`, Skeleton scan, CLI scan, generic API ClearML scan, public export/import smoke, `git diff --check -- mldb_v2`, and owned-file trailing whitespace.
- W007 remains `completed`; this repair changed tests/records only and introduces no new production behavior.

## Post-closure object-byte wiring repair evidence ? 2026-09-14
- W008 adversarial closure exposed a narrow production wiring omission: `Application` owned `_ObjectByteAccess`, but `AuthoringPlanningService` did not pass it into its default `_RepositoryDefinitionVerifier`.
- Repair is composition-only: `Application._object_bytes -> AuthoringPlanningService(object_access=...) -> _RepositoryDefinitionVerifier(object_access=...)`. The existing sealer continues to share that same verifier. Frozen Application/Query/Error interfaces and the W002 verification algorithm are unchanged.
- Added explicit Application integration coverage asserting the owned object-byte access is the verifier's configured access. Removed the W008 concrete-integration private verifier substitution and replaced it with a production-valid fixture using exact Corpus bytes and real verifier/integrity/asset-test paths.
- Production gate: the original one-entry sealed reproduction no longer returns `Corpus object-byte access is not configured`; production CLI `verify` and `plan` both exit **0** on a production-valid sealed Study without monkeypatching.
- W007 regression after repair: focused API **41 passed**, composition **7 passed**, Application integration **17 passed**. Relevant W002 verification/Corpus tests: **175 passed, 1 skipped**.
- W008 concrete integration **13 passed**, aggregate **106 passed**, and full `mldb_v2/tests` **1343 passed, 3 skipped**.
- W007 remains `completed`; this repair changes only the missing production dependency wiring and associated regression evidence.
