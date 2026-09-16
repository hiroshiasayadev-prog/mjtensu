# MLDB-V2-TASK-003-04: Implement canonical StudyPlan build and immutable write

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-003
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-003-02, MLDB-V2-TASK-003-03]
- **outputs**: StudyPlan public shape/validation, deterministic Plan construction, immutable canonical Plan creation, focused tests

## Goal
Compose T003-02 deterministic expansion with T003-03 committed source pins into the exact frozen StudyPlan representation, derive its canonical content digest/ID, validate the complete record, and persist it idempotently through the W001 immutable canonical writer.

## Work
- Mirror the frozen `study/plan.py` public shapes in `mldb_v2/src/study/plan.py` without Skeleton runtime imports.
- Build pins and trials mechanically from established private T003-02/T003-03 boundaries; do not re-expand grids, re-discover source pins, or re-run Study semantic validation.
- Compute `content_sha256` from canonical digest content after excluding only `id` and `content_sha256`, then derive `<study-namespace>/<study-local>-plan-<prefix16>` exactly.
- Implement the owning StudyPlan validator required by W001 `CanonicalRecordValidator`, covering exact schema/fields, identity/digest invariants, pin order/uniqueness, trial/evaluation sequence/order, and source-variant structure.
- Persist through `CanonicalRepositoryWriter.create_immutable(EntityKind.STUDY_PLAN, ...)`; preserve idempotent replay and same-ID different-content conflict behavior. No Git commit, backend state, timestamp, or execution work.

## Done condition
Identical validated planning inputs and selected committed source identity reproduce the same complete StudyPlan content/digest/ID and exact canonical record; a valid first write and exact replay succeed, while malformed or conflicting records are rejected without overwrite.

## Verification
Cover public-shape conformance, canonical digest/ID derivation, training/existing-model plans, PlanPin conversion/order, trial/evaluation conversion/order, validator rejection matrix, deterministic repeat build, immutable write/replay/conflict, no source/grid rediscovery, py_compile/import, and full regression.

## Completion Evidence - 2026-09-13
- Added exact Frozen Skeleton mirror `mldb_v2/src/study/plan.py` using only runtime `mldb_v2.src` types; AST public-shape comparison against the frozen Skeleton passes and there is no Skeleton runtime import.
- Added private `_build_study_plan` in `mldb_v2/src/study/_plan_build.py`; one `_StudyPlanningInput` instance is passed to both T003-02 `_expand_study_grid` and T003-03 `_collect_study_source_pins`, preventing caller-side expansion/pin mixing.
- T003-04 does not reload/revalidate Study semantics, rediscover references, execute companions, inspect weights, or recompute Corpus/source pin semantics.
- `_CommittedSourcePin` converts mechanically to exact `PlanPin`; fixed kind order, Unicode ID order, duplicate rejection, SHA/path/source ordering, namespace IDs, kind-specific companion/manifest structure, and namespace coverage are validated without normalization.
- `source_commit` is taken from the pin collection and requires a full lowercase 40/64-hex Git object ID.
- T003-02 expanded trials convert mechanically to exact frozen training/existing-model source variants and Evaluation coordinates; trial order is unchanged and Evaluation order remains trial-local.
- Plan parameter mappings are copied as ordinary JSON-compatible dictionaries and validated structurally with W001 public-parameter value validation; defaults are not recomputed and seed remains separate.
- Digest construction excludes only `id` and `content_sha256`, hashes W001 `_canonical_json_bytes`, persists the full lowercase SHA-256, then derives `<study-namespace>/<study-local>-plan-<digest[:16]>` before complete validation.
- The complete non-normalizing StudyPlan validator rejects wrong/missing/extra top-level fields, invalid schema/Study/source commit, malformed pins, wrong pin order/duplicates, missing namespace/reference pins, malformed source variants, invalid parameters/seeds, trial/Evaluation gaps or reordering, wrong digest, and wrong Plan ID.
- Private `CanonicalRecordValidator` adapter accepts only `EntityKind.STUDY_PLAN` and delegates to the owning Plan validator; persistence uses only `CanonicalRepositoryWriter.create_immutable`, never direct atomic replacement.
- Synthetic committed training-repository E2E covers T003-01 preflight -> T003-02 expansion -> T003-03 committed pins -> T003-04 Plan build -> W001 immutable write -> exact replay. It uses multiple Architectures, multiple seeds, Train/Evaluation grids, multiple Evaluation stages, resolved defaults, exact ordering, and deterministic digest/ID.
- Synthetic existing-Model E2E preserves authored Model order and verifies exact existing-model source shape plus Model, TrainingResult, and lineage Architecture pins; no weights/runtime loading occurs.
- Writer-boundary regressions cover first create, exact replay, invalid existing record rejection before replay, and controlled same-ID conflicting valid content without weakening production digest construction.
- Focused `test_study_plan.py`: **12 passed**. W003 T003-01..04: **69 passed**. Full `mldb_v2/tests`: **807 passed, 1 skipped** (baseline 795 passed, 1 skipped).
- Changed src/test py_compile PASS; import smoke for `mldb_v2.src.study.plan` and `mldb_v2.src.study._plan_build` PASS; deterministic repeated compilation PASS; Frozen public-shape comparison PASS.
- Dependency/mutation scan CLEAN: no mldb v1 runtime dependency, Skeleton runtime import, backend/ClearML, torch, random/time/uuid Plan identity, direct atomic write, or Git mutation path.
- Current real `mldb_data/tile-classifier` and `mldb_data/rotated-fcos` contain no StudyPlan created by T003-04 tests; canonical `mldb_data/**/__pycache__` count remains **0**; `git diff --check -- mldb_v2` PASS.
- Existing tracked-flat `mldb_data` deletions, namespace-first migration, v1/tools changes, and unrelated dirty state were preserved. No commit/stage/stash/restore/reset/clean/push was performed.
- No blocker remains for T003-04. W003 status, T003-05, `tasks/index.md`, W004, Specs, Skeleton, W001 writer, and T003-01/02/03 implementation/tests were not modified.
## Coordinator cross-review — W003-CL02 — 2026-09-13
- Fresh T003-04 verification is green: focused **12 passed**, T003-01..04 **69 passed**, full `mldb_v2/tests` **807 passed, 1 skipped**, changed-file py_compile/import PASS, canonical `mldb_data/**/__pycache__` count 0, and `git diff --check -- mldb_v2` PASS.
- Blocking PlanPin source-shape finding: `_validate_pin()` validates `sources[*].path` only with generic `_validate_safe_relative_path`, so it accepts paths forbidden by the frozen executable-integrity contract, including `tools/evil.py` and glob-like `product/*.py`, `product/x?.py`, `product/[x].py`.
- `spec:mldb.v2.study.plan_format` requires `PlanPin.sources` to be the exact source set from executable-integrity; `spec:mldb.v2.verification.executable_integrity` explicitly rejects `tools/` implementation paths, globs, directories, absolute paths, and non-repository-relative source paths.
- Coordinator probe re-signed otherwise-valid Plans with each forbidden source path above; `_validate_study_plan()` accepted all four. This means the owning StudyPlan validator can authorize a canonical Plan representation that violates the frozen source-entry contract.
- Repair must reuse the established W002 executable-source structural path rule (or an exactly shared private boundary) rather than invent a looser Plan-only source grammar. Add focused negative regressions for `tools/` and glob forms; preserve T003-03-generated valid source pins unchanged.
- Keep T003-04 `planned` until W003-CL02 repair and full verification are green. T003-05 remains blocked by T003-04.
## W003-CL02 Repair Completion Evidence - 2026-09-13
- Reused the established W002 executable-source path boundary directly: `mldb_v2.src.catalog._executable_definition_loading._validate_source_path`; T003-04 no longer uses the generic W001 safe-relative-path helper for `PlanPin.sources[*].path`.
- StudyPlan validation now rejects `tools/evil.py`, `tools/sub/evil.py`, `product/*.py`, `product/x?.py`, and `product/[x].py` as `invalid_pin_source_path` after digest/Plan ID recomputation.
- Valid executable-source regressions remain accepted for `product/a.py`, `product/recognition/models/rotated_fcos.py`, and `package/sub/module.py`; standalone Plan validation does not require source-file existence.
- Regression coverage exercises both `_validate_study_plan` and `_StudyPlanRecordValidator.validate`, so manually supplied canonical StudyPlans cannot bypass the executable-integrity source grammar at the W001 immutable-writer validator boundary.
- Focused `test_study_plan.py`: **20 passed**. W003 T003-01..04: **77 passed**. Full `mldb_v2/tests`: **815 passed, 1 skipped**.
- Changed src/test py_compile PASS; import smoke PASS; direct valid/tools/glob Plan source-path probes PASS; Frozen public Plan shape remains covered by focused tests.
- Dependency/Git mutation source scan CLEAN; canonical `mldb_data/**/__pycache__` count remains **0**; `git diff --check -- mldb_v2` PASS; staged diff remains empty. Existing unrelated repository dirty state and untracked `mldb_v2/` state were preserved; no commit/stage/stash/restore/reset/clean/push was performed.
- W003-CL02 is closed. No blocker remains for T003-04. T003-05, W003, `tasks/index.md`, W004, Specs, Skeleton, W001/W002 implementation, and T003-01/02/03 were not changed.
