# MLDB-V2-TASK-007-01: Implement authoring and planning application composition

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-007
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-001, MLDB-V2-WORK-002, MLDB-V2-WORK-003]
- **outputs**: application error value/mapping plus validate/verify/seal/plan composition

## Start gate clarification
W006 is not an implementation-start dependency for this Task. W001/W002/W003 already expose the frozen repository, lifecycle, and StudyPlan boundaries required here. W006 remains a W007 integration/closure gate only.

## Exclusive ownership
May modify only:
- `mldb_v2/src/api/errors.py`
- `mldb_v2/src/api/_errors.py`
- `mldb_v2/src/api/_authoring.py`
- `mldb_v2/tests/test_api_authoring.py`
- this Task record

Do not modify `src/api/query_interface.py`, `src/api/application_interface.py`, `src/api/_query.py`, `src/api/_execution.py`, `src/api/application.py`, W006 files, CLI, Specs/Skeletons, or `records/tasks/index.md`.

## Goal
Compose frozen Application authoring/check behavior over W001/W002/W003 without duplicating validation, verification, sealing, repository traversal, Study compilation, or source-pinning semantics.

## Work
- Mirror `skeleton/api/errors.py` exactly in runtime `src/api/errors.py`; no Skeleton runtime imports.
- Implement one private bounded exception/error translation seam in `_errors.py`; preserve the ten frozen public categories and never expose backend exception objects, credentials, or stack traces.
- `validate_scope` uses W001 canonical inventory and W002 `DefinitionValidator`; empty scope includes repository structural issues plus every discoverable reusable definition in canonical order.
- `verify_scope` uses W001 inventory and W002 `DefinitionVerifier`; empty scope means every sealable definition. Continue through per-target failures.
- `seal_scope` resolves targets canonically, requires explicit `bulk=True` for more than one target, and delegates each exact target to W002 `DefinitionSealer` in canonical order. Do not add cross-file transaction semantics.
- `plan_study` composes W003 `_StudyPlanningPreflight`, current repository commit, `_build_study_plan`, and `_create_study_plan`; do not reproduce expansion/source-pinning logic.
- Map lower bounded failures to stable application error categories only at this application boundary.

## Focused verification
Cover empty/narrow/exact scopes, repository issues, canonical ordering, continue-through-failure reports, bulk seal guard, per-target seal result, sealed Study plan/replay, unsealed/source-dirty failures, and stable error values. Use focused tests plus cheapest W002/W003 integration only; no full suite.

## Completion condition
Authoring/check operations produce exact frozen report semantics, bulk mutation is guarded, planning is idempotent and source-pinned through W003, and no backend/W006/CLI behavior is introduced.

No commit/stage/stash/reset/clean/restore/push.

## Completion evidence
- Runtime `src/api/errors.py` mirrors Frozen `skeleton/api/errors.py` exactly.
- Added bounded private application-error translation in `src/api/_errors.py`; public values contain only `code` and `message`.
- Implemented `validate_scope`, `verify_scope`, `seal_scope`, and `plan_study` in `src/api/_authoring.py` by composing W001/W002/W003 boundaries only.
- Scope selection uses `CanonicalRepositoryListing`; no independent filesystem walk or kind inference from typed IDs.
- Bulk sealing requires explicit `bulk=True`, preserves canonical order, continues after per-target failure, and does not roll back prior success.
- Study planning delegates `_StudyPlanningPreflight`, current Git commit resolution, `_build_study_plan`, and `_create_study_plan`; replay is idempotent and dirty selected source maps to `source_not_pinned`.
- Focused verification: `mldb_v2/tests/test_api_authoring.py` ? **10 passed**. Full `mldb_v2/tests` intentionally not run.
- `py_compile`, BOM/trailing-whitespace checks, and forbidden backend/W006/CLI import checks: **PASS**.
- No commit/stage/stash/reset/clean/restore/checkout/worktree/push performed.
