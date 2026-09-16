# MLDB-V2-TASK-007-02: Implement read-only query and observation API

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-007
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-001, MLDB-V2-WORK-005]
- **outputs**: runtime QueryInterface mirror and read-only query/discovery/observation composition

## Start gate clarification
W006 is not required to start. W001 listing/resolution and the W005 generic BackendPort plus optional backend-log capability are sufficient. T005-05/W005 closure remains an integration gate for production backend wiring, not for fake-backed focused implementation.

## Exclusive ownership
May modify only:
- `mldb_v2/src/api/query_interface.py`
- `mldb_v2/src/api/_query.py`
- `mldb_v2/tests/test_api_query.py`
- this Task record

Do not modify `src/api/errors.py`, `src/api/_errors.py`, `src/api/application_interface.py`, `src/api/_authoring.py`, `src/api/_execution.py`, `src/api/application.py`, W005/W006 files, CLI, Specs/Skeletons, or `records/tasks/index.md`.

## Goal
Implement the frozen read-only discovery/query surface without filesystem-search logic above W001, without importing executable definitions for listing, and without any canonical/backend mutation.

## Work
- Mirror frozen `skeleton/api/query_interface.py` exactly in runtime `src/api/query_interface.py`; use runtime `mldb_v2.src` values only.
- `list_entities` and `get_entity` delegate to W001 listing/resolution; `definitions` means the aggregate reusable-definition view only, with lifecycle filtering applied only where valid.
- `list_study_results` delegates to W001 `CanonicalRepositoryListing.list_study_results` and preserves deterministic ordering/filter validation.
- `get_study_result` resolves and validates one canonical StudyResult, then derives exact training/evaluation/total counters only from canonical stage slots. Existing-Model trials contribute zero training stages.
- `observe_study` first derives the same canonical view, then performs backend `observe` calls only for exact planned/pending stage keys as allowed by canonical identity. Observation must never alter dispositions or canonical files.
- `read_backend_logs` capability-checks an injected/backend optional log seam. Unsupported capability becomes a bounded private unsupported result for later application-error mapping; never synthesize logs from canonical records.
- `diagnose` is read-only. Keep checks bounded/injectable and never enqueue, seal, plan, create results, or mutate configuration.
- Do not depend on W006 readiness/progression internals to compute canonical progress.

## Focused verification
Cover discovery/resource/filter validation, exact resolution, StudyResult progress math, existing-Model training count, backend observation ordering, zero mutation proof, optional logs supported/unsupported, and diagnose side-effect freedom. Use fakes where production W005 composition is still closing; no full suite.

## Completion condition
All query methods are read-only, discovery-first, deterministic, backend observation remains non-canonical, and no W006 state-machine logic is reproduced.

No commit/stage/stash/reset/clean/restore/push.
## Completion evidence
- Implemented runtime `src/api/query_interface.py` as an exact runtime mirror of the Frozen QueryInterface shape using only `mldb_v2.src.*` types; machine comparison against the Frozen source with import-prefix substitution matched exactly.
- Implemented `src/api/_query.py` as a read-only composition over W001 listing/resolution and the generic W005 backend observation boundary; no W006 progression primitive, canonical writer, sealing, planning, admission, collection, or cancellation dependency is used.
- `definitions` lists only Task, Corpus, Architecture, Train Protocol, Evaluation Protocol, and Study definitions; lifecycle filtering is bounded to reusable definitions and unsupported filter/resource combinations fail privately instead of being ignored.
- Study progress is derived only from runtime-validated canonical StudyResult stage slots. Counters contain exactly planned/pending/completed/failed/cancelled/skipped, and Existing-Model trials contribute zero training stages.
- Observation queries preserve canonical state, query backend state only for exact pending planned StageKeys in deterministic trial/stage order, and ignore missing backend observations without converting them into canonical failure.
- Backend logs are optional-capability checked without ClearML-specific generic branching; diagnostics use only read-only injected probes.
- Added focused verification in `tests/test_api_query.py` covering Frozen shape parity, discovery/filter behavior, exact resolution, StudyResult list filters, progress, Existing-Model behavior, observation ordering/non-mutation, logs supported/unsupported, diagnose read-only behavior, and absence of mutation/progression backend calls.
- Focused verification: `.\.venv\Scripts\python.exe -m pytest mldb_v2\tests\test_api_query.py -q` -> `11 passed`.
- No full `mldb_v2/tests` suite was run for this Task. No commit/stage/stash/reset/clean/restore/push was performed.