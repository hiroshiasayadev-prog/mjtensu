# MLDB-V2-TASK-007-04: Integrate concrete Application API

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-007
- **task_type**: integration
- **depends_on**: [MLDB-V2-TASK-007-01, MLDB-V2-TASK-007-02, MLDB-V2-TASK-007-03, MLDB-V2-TASK-006-04, MLDB-V2-TASK-005-05]
- **outputs**: runtime ApplicationInterface mirror plus concrete application composition over authoring/query/execution components

## Join gate clarification
This is the first concrete join against the actual W006 one-pass primitive and completed W005 backend composition. It does not need T006-05/W006 formal closure to begin integration, but W006 closure remains mandatory before W007 closure.

## Exclusive ownership
May modify only:
- `mldb_v2/src/api/application_interface.py`
- `mldb_v2/src/api/application.py`
- `mldb_v2/src/api/__init__.py`
- `mldb_v2/tests/test_api_application_integration.py`
- this Task record

Do not modify T007-01/02/03-owned files, W005/W006 implementation, CLI, Specs/Skeletons, or `records/tasks/index.md`.

## Goal
Expose one concrete transport-independent Application boundary by wiring the completed authoring, query, and execution components to actual lower-layer production seams without moving domain logic into `src/api`.

## Work
- Mirror frozen `skeleton/api/application_interface.py` exactly in runtime `src/api/application_interface.py`; import runtime QueryInterface/value types only.
- Compose T007-01 authoring/planning, T007-02 query/observation, and T007-03 execution shell behind one concrete application object.
- Bind `advance_study`/cancellation only to the actual W006 internal primitive/seam established by T006-04; adapt shape only, never copy its algorithm.
- Bind configured backend resolution/log capability only through W005 production composition; no ClearML-specific branch in generic application code.
- Apply the T007-01 stable public error mapping consistently across all public methods.
- Preserve `start_study` zero-admission, query read-only, run/resume same-one-pass-loop, and exact Frozen response shapes.
- Keep W008 CLI completely outside this Task.

## Focused verification
Run exact public-shape comparison, cross-component integration with generic fake backend, actual W006 seam adapter tests, public error categories, zero-admission start, query no-mutation, run/resume/rerun/cancel delegation, and W005 optional logs. Do not run full suite yet.

## Completion condition
One concrete Application implementation satisfies the frozen ApplicationInterface/QueryInterface and uses W006 rather than duplicating it. No CLI code or transport policy exists in `src/api`.

No commit/stage/stash/reset/clean/restore/push.
## Completion evidence
- Added exact runtime `src/api/application_interface.py` mirror using runtime `mldb_v2.src.*` imports only.
- Added concrete `src/api/application.py` composition over T007-01 authoring/planning, T007-02 query/observation, and T007-03 execution shell.
- Bound one-pass progression directly to W006 `study_driver.advance_study()` and generic backend resolution to W005 `BackendRegistry`/`BackendConfig`; no ClearML-specific branch exists in generic API code.
- Implemented the Application-owned cancellation request boundary: under `StudyResultMutationCoordinator`, re-resolve the canonical StudyResult, return terminal/cancelling idempotent outcomes, and for `submitted` change only `status` to `cancelling` using `CanonicalRepositoryWriter.replace_nonterminal_study_result(...)`.
- Backend cancellation is requested only after the repository mutation lock is released; cancellation progression, stage skips, collection, acceptance, and terminal closure remain W006-owned.
- Added stable package exports in `src/api/__init__.py` and focused integration coverage in `tests/test_api_application_integration.py`.
- Static closure verification passed: exact Frozen runtime-mirror comparison, `py_compile`, no trailing whitespace, no `mldb_v2.skeleton` runtime dependency, no ClearML/CLI branching in `application.py`.
- Focused API join verification: `.\.venv\Scripts\python.exe -m pytest -q mldb_v2\tests\test_api_authoring.py mldb_v2\tests\test_api_query.py mldb_v2\tests\test_api_execution.py mldb_v2\tests\test_api_application_integration.py` -> `48 passed`.
- Focused W006 cancellation/lock verification: `.\.venv\Scripts\python.exe -m pytest -q mldb_v2\tests\test_study_driver.py -k "cancelling or backend_calls_are_outside_study_result_mutation_lock"` -> `4 passed, 24 deselected`.
- Focused W005 backend composition/log verification: `.\.venv\Scripts\python.exe -m pytest -q mldb_v2\tests\test_clearml_backend_conformance.py -k "concrete_cancel_and_optional_logs_remain_outside_required_port or registry_activation_is_explicit_generic_clearml_type"` -> `2 passed, 14 deselected`.
- No full `mldb_v2/tests` suite was run for this Task. No commit/stage/stash/reset/clean/restore/checkout/worktree/push was performed.

## Post-closure production composition repair evidence — 2026-09-14
- W008 closure verification exposed a missing production Application composition seam; `ApplicationComposition` and `compose_application()` are now present and publicly exported.
- `ApplicationComposition.default_backend` is `str | None`; `MLDB_V2_DEFAULT_BACKEND` is the only newly introduced backend selector.
- ClearML registration/configuration is confined to the production composition root; backend resolution and S3 transport activation remain lazy.
- Frozen `ApplicationInterface` remains unchanged apart from the established runtime import mirror; W006 progression/cancellation semantics were not changed.
- Repaired the two stale W007 closure assertions: public `__all__` now verifies the exact 7-symbol public surface, and ClearML genericity scanning excludes only `composition.py` while Skeleton/CLI scans still cover all `src/api` modules.
- Focused repair suite: `test_api_application_integration.py` + `test_api_conformance.py` + `test_api_composition.py` -> **29 passed**.
- Breakdown: Application integration **16 passed**; API conformance **6 passed**; composition **7 passed**.
- Static checks PASS: `py_compile` (13 files), whole-API Skeleton dependency scan, whole-API CLI dependency scan, generic-API ClearML scan (9 modules; `composition.py` excluded), exact public export/import smoke, `git diff --check -- mldb_v2`, and owned-file trailing-whitespace scan.
- T007-04 remains `completed`; no production implementation was changed by this static-test repair.

## Post-closure object-byte wiring repair evidence ? 2026-09-14
- W008 concrete planning exposed that the concrete `Application` retained `_ObjectByteAccess` for W006 advancement but omitted it when constructing `AuthoringPlanningService`; the latter therefore created a W002 verifier with `object_access=None`.
- Repaired only that composition join: `AuthoringPlanningService` accepts optional `_ObjectByteAccess`, passes it to the existing `_RepositoryDefinitionVerifier`, and `Application` supplies its already-owned `self._object_bytes`. No Frozen interface or verification/progression algorithm changed.
- Added focused integration coverage for the exact ownership/wiring identity. Production `verify`/`plan` pass on a production-valid sealed Study without replacing the verifier.
- Post-repair regression: API focused **41 passed**, composition **7 passed**, Application integration **17 passed**, relevant W002 verification/Corpus **175 passed, 1 skipped**, W008 concrete **13 passed**, W008 aggregate **106 passed**, full `mldb_v2/tests` **1343 passed, 3 skipped**.
- T007-04 remains `completed`.
