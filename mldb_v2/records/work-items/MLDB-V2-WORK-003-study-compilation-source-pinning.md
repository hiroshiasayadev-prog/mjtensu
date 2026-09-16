# MLDB-V2-WORK-003: Study compilation and source pinning

- **status**: completed
- **date**: 2026-09-09
- **depends_on**: [MLDB-V2-WORK-001, MLDB-V2-WORK-002]
- **source_refs**: `spec:mldb.v2.study.study_format`, `spec:mldb.v2.study.grid_expansion`, `spec:mldb.v2.study.plan_format`, `spec:mldb.v2.study.source_pinning`

## Goal
Implement deterministic Study expansion and immutable StudyPlan creation from sealed definitions and committed source inputs.

## Boundary
Own planning-side `mldb_v2/src/study/` except the reusable `study.py` definition shape/validation established by W002. W003 consumes W002 definition validation/verification and W001 repository/source primitives; it does not redefine Study sealing semantics, execute domain code, create backend Tasks, accept results, or drive Study lifecycle.

## Task candidates
| task | responsibility | dependency |
|---|---|---|
| MLDB-V2-TASK-003-01 | Implement planning preflight that resolves an already-sealed Study and referenced canonical inputs, rechecks W002 verification/lifecycle usability, and prepares validated inputs for expansion without duplicating Study semantic validation. | W002 |
| MLDB-V2-TASK-003-02 | Implement deterministic training trial and Evaluation-coordinate expansion, including type-sensitive duplicate detection/default resolution. | T01 |
| MLDB-V2-TASK-003-03 | Implement committed-source pin collection for canonical YAML, companions, declared project sources, Corpus manifests, and existing Model lineage, consuming the validated planning-input boundary from T003-01 rather than rediscovering Study semantics. | W001,W002,T01 |
| MLDB-V2-TASK-003-04 | Implement canonical Plan digest/ID construction and immutable repository creation. | T02,T03 |
| MLDB-V2-TASK-003-05 | Verify byte/determinism behavior with classifier, rotated detector, Evaluation grids, and existing-Model planning fixtures. | T04 |

## Planning ownership clarification — 2026-09-11
Study model-source/evaluation-stage semantic compatibility is a W002 seal gate because `definition_lifecycle` requires complete planning validation before Study sealing. T003-01 therefore consumes that completed gate at planning time rather than implementing a second validator. Source-commit cleanliness/pin collection remains T003-03; grid expansion and Plan construction remain wholly W003.

## Completion condition
- Training axes create Trials; Evaluation axes only create trial-local coordinates.
- Authored ordering and Unicode-key ordering exactly match the Specification.
- Formal planning rejects dirty/unpinned required canonical/runtime sources without requiring the whole repository to be clean.
- Recompiling identical committed inputs produces the same Plan content digest and ID.
- Plans contain no backend/live-execution state or compilation timestamp.

## Closure Evidence ? 2026-09-13
- T003-01 through T003-05 are completed. Independent T003-05 closure verification exercised the full sealed-Study planning path: planning preflight -> deterministic expansion -> committed source pinning -> deterministic StudyPlan digest/ID -> immutable create/replay.
- Required classifier-like, rotated-detector-like, cross-namespace, existing-Model, source-drift, and persistence scenarios all passed with no blocking contract violation. Training axes create Trials; Evaluation axes remain trial-local; authored Architecture/Model/stage ordering and Unicode axis ordering are preserved; required canonical/runtime drift and stale planning inputs reject while unrelated dirty work remains allowed.
- Closure counts: T003-05 focused **8 passed**; W003 suites **85 passed**; full `mldb_v2/tests` **823 passed, 1 skipped**. py_compile/import/dependency scans PASS; public StudyPlan shape coverage PASS; `mldb_data` mutation check unchanged; canonical `__pycache__` **0**; `git diff --check -- mldb_v2` PASS.
- W003 has **no blocking finding** and is closed. `tasks/index.md` remains untouched for the Global Coordinator; W004+ was not modified.
