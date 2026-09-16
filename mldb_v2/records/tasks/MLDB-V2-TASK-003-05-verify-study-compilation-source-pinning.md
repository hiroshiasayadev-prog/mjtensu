# MLDB-V2-TASK-003-05: Verify Study compilation and source pinning

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-003
- **task_type**: verification
- **depends_on**: [MLDB-V2-TASK-003-04]
- **outputs**: W003 end-to-end conformance/determinism verification, focused verification tests/evidence, Work Item closure recommendation

## Goal
Independently verify the completed W003 planning pipeline from sealed Study preflight through deterministic expansion, committed-source pinning, StudyPlan digest/identity, and immutable persistence against the frozen contracts.

## Work
- Exercise classifier-like and rotated-detector-like training Studies, Evaluation grids, cross-namespace pins, and existing-Model planning through the actual T003-01..04 boundaries without redefining them.
- Re-run determinism, raw committed-byte pinning, stale-input rejection, required-clean-set vs unrelated-dirty behavior, canonical Plan digest/ID, immutable create/replay/conflict, and public-shape conformance.
- Use committed synthetic repositories for formal success cases; current canonical example Studies remain read-only drafts and must not be modified or falsely treated as plan-ready.
- Do not repair W003 implementation inside this verification Task. A genuine frozen-contract failure remains a blocking finding assigned to its owning T003 Task.

## Done condition
Independent verification finds no W003 blocking contract violation and demonstrates that identical committed inputs reproduce the same Plan while relevant source drift is rejected and unrelated dirty work does not poison planning.

## Verification
Run focused W003 suites plus adversarial end-to-end probes, full `mldb_v2/tests`, public-shape/py_compile/import/dependency scans, current-repo no-mutation/listing hygiene, and `git diff --check -- mldb_v2`.

## Completion Evidence ? 2026-09-13
- Added independent `mldb_v2/tests/test_study_compilation_conformance.py` closure coverage across all required scenario classes: classifier-like training grids/defaults/evaluation grids/deterministic repeat; rotated-detector-like multiple Architectures and Evaluation stages with authored order; cross-namespace pins with unrelated dirty namespace/files allowed; existing-Model authored order plus Model/TrainingResult/Architecture lineage pins without weight/runtime loading; required-source and stale-planning-input rejection; and immutable first-create/exact-replay/malformed-existing/conflict behavior.
- Focused T003-05: **8 passed**. W003 T003-01..05 suites: **85 passed**. Full `mldb_v2/tests`: **823 passed, 1 skipped**.
- Study planning Python py_compile: **7 modules PASS**; import smoke for preflight/grid/pinning/Plan/build: **5 modules PASS**. Runtime import scan found **0** Skeleton, mldb v1, torch, ClearML, or backend imports in `mldb_v2/src/study/`. Frozen public StudyPlan shape remains covered by the W003 suite.
- `mldb_data` byte snapshot before/after W003/full verification is unchanged: **0 added, 0 removed, 0 changed**; canonical `mldb_data/**/__pycache__` count is **0**. `git diff --check -- mldb_v2` PASS and staged W003 diff is empty.
- No blocking finding was found. No W003 implementation source, Spec/Skeleton, `mldb_data`, `tasks/index.md`, or W004+ file was modified; no commit/stage/stash/restore/reset/clean/push was performed.
