# MLDB-V2-TASK-003-03: Implement committed source pin collection

- **status**: completed
- **date**: 2026-09-12
- **work_item**: MLDB-V2-WORK-003
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-001, MLDB-V2-WORK-002, MLDB-V2-TASK-003-01]
- **outputs**: committed-source pin collection under `mldb_v2/src/study/`, focused tests

## Goal
Build the exact canonical Plan pin set for one selected Git commit and reject required working-tree inputs that do not match that commit without requiring whole-repository cleanliness.

## Work
- Consume T003-01 `_StudyPlanningInput` as the exact referenced-entity graph; reuse W001 Git snapshot/source primitives and W002 definition/integrity boundaries rather than rediscovering Study semantics or inventing another executable source convention.
- Pin Namespace/Task/Corpus/Architecture/TrainProtocol/EvaluationProtocol/Study plus referenced Model/TrainingResult lineage as applicable.
- Hash exact canonical YAML bytes from the selected commit; pin executable/builder companion SHA, exact declared project-source path/hash set, and Corpus manifest digest/count according to PlanPin.
- Enforce fixed pin-kind order and Unicode ID order, one `(kind,id)` each, while keeping unrelated dirty files/namespaces out of the cleanliness scope.
- Keep Study grid expansion, Plan ID/content digest construction, immutable Plan write, backend checkout/execution, and Git mutation outside this Task.

## Done condition
For a validated Study input and selected commit, the collector either returns the exact deterministic PlanPin set backed by committed bytes or fails because a required canonical/runtime source is dirty, missing, or inconsistent.

## Verification
Cover exact committed bytes, targeted dirty/missing required paths, unrelated dirty files allowed, executable declared sources, Corpus manifest/builder pins, existing Model lineage, pin ordering/deduplication, no Git mutation, py_compile/import, and full regression.

## Coordinator cross-review — W003-CL01 — 2026-09-12
- Fresh implementation verification is green: T003-03 focused **22 passed**, T003-01..03 **53 passed**, full `mldb_v2/tests` **791 passed, 1 skipped**, changed-file py_compile PASS, canonical `mldb_data/**/__pycache__` count 0, and `git diff --check -- mldb_v2` PASS.
- Blocking stale-input finding: the collector checks current required files against `selected_commit`, but does not bind the supplied T003-01 `_StudyPlanningInput` records back to those committed/current canonical documents.
- Reproduction: prepare planning input A from a sealed Study referencing `arch-a` + `arch-z`; then change the canonical Study to B referencing only `arch-a`, commit B, and collect using `selected_commit=B` with stale planning input A. Collection currently succeeds and pins both Architectures while the pinned committed Study YAML references only `arch-a`.
- This violates `spec:mldb.v2.study.source_pinning`: canonical v2 inputs consumed by the Plan must match the selected Git commit exactly. A later T003-04 could otherwise combine expansion from stale Study A with a Study pin whose raw bytes are committed Study B.
- Repair must keep T003-01 as the graph owner: exact-resolve each non-Namespace graph record only to verify that the supplied planning record is semantically/type-sensitively identical to the current canonical record whose bytes match the selected commit. Do not rediscover references or rebuild Study semantics.
- Add a regression for stale Study graph and, preferably, stale referenced record binding. Keep Task status `planned` until the repair and full verification are green.

## Completion Evidence — 2026-09-12
- Closed W003-CL01 by binding every non-Namespace pin-graph record back to the exact current canonical document after working-tree bytes are proven equal to `selected_commit`.
- Added exact `CanonicalRepositoryResolver` mapping for Task, Corpus, Architecture, TrainProtocol, EvaluationProtocol, Study, Model, and TrainingResult; Existing-Model lineage Architecture resolution also uses the exact resolver.
- Planning/current semantic equality uses W001 `_canonical_json_bytes`, so mapping key order is irrelevant while `true`, `1`, and `1.0` remain distinct; stale inputs fail with bounded private code `planning_input_stale`.
- T003-03 remains a binding/pinning layer only: it does not rebuild the Study graph, rediscover references, recompute compatibility/defaults/axes, create expansion rows, or add Plan metadata.
- Added regressions for stale Study A→B graph, stale TrainProtocol default 32→64, stale existing-model TrainingResult, and direct type-sensitive comparison.
- Existing pin kind order, Unicode ID order, deduplication, committed-byte hashes, clean-set behavior, non-HEAD selection, unrelated dirty allowance, and deterministic repeated collection remain unchanged.
- Focused `test_study_source_pinning.py`: **26 passed** (baseline 22). T003-01..03: **57 passed** (baseline 53). Full `mldb_v2/tests`: **795 passed, 1 skipped** (baseline 791 passed, 1 skipped).
- Changed src/test py_compile PASS; import smoke PASS; dependency scan CLEAN; Git-mutation literal scan CLEAN; canonical `mldb_data/**/__pycache__` count **0**; `git diff --check -- mldb_v2` PASS.
- No Spec/Skeleton/`mldb_data/**`/`mldb_tests/**`/T003-01/T003-02/W004/task-index changes were made. No Git commit/stage/checkout/reset/clean/stash/push was performed.
- No blocker remains for T003-03.
