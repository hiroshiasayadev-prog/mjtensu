# MLDB-V2-TASK-003-01: Implement Study planning preflight

- **status**: completed
- **date**: 2026-09-12
- **work_item**: MLDB-V2-WORK-003
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-001, MLDB-V2-WORK-002]
- **outputs**: planning-side Study preflight under `mldb_v2/src/study/`, focused tests

## Goal
Resolve one already-sealed Study and the canonical definitions needed for planning, reusing W002 validation/verification without reimplementing Study semantics, and return a private validated planning input boundary for later grid expansion.

## Work
- Require exact sealed Study resolution and successful W002 DefinitionVerifier result.
- Resolve training-source or existing-Model references needed by expansion through exact W001 canonical resolution while preserving authored order.
- Establish the selected Task and exact Train/Evaluation Protocol declarations required for later default resolution and axis expansion.
- Keep grid Cartesian expansion, Trial/Evaluation-coordinate allocation, Git source pin collection, Plan hashing/writing, companion execution, backend work, and canonical mutation outside this Task.
- Do not create a second Study validator; consume W002 `study.py`/DefinitionVerifier behavior.

## Done condition
A sealed, W002-verifiable Study can be converted into deterministic private planning inputs sufficient for T003-02/T003-03, while invalid/unsealed/unverifiable Studies fail before expansion or source pinning.

## Verification
Cover training and existing-Model Studies, cross-namespace references, sealed/reference failures, no semantic-validator duplication, no Plan generation/backend calls, py_compile/import, dependency scan, and full regression.

## Completion Evidence — 2026-09-12
- Added private `_StudyPlanningPreflight` and immutable planning-input dataclasses in `mldb_v2/src/study/_planning_preflight.py`; no public Skeleton/API shape was added.
- Planning entry requires exact local Study load, `status == sealed`, then injected W002 `DefinitionVerifier` success with `valid == True` and exactly empty diagnostics. Invalid results and verifier exceptions are converted to bounded private planning failures.
- Training-source preflight resolves exact Task, training Corpus, TrainProtocol, authored Architecture sequence, authored sparse axes/seeds, and authored Evaluation stage Corpus/Protocol records. Protocol declarations are retained; defaults are not materialized and no Cartesian expansion/Trial/Evaluation IDs are produced.
- Existing-Model preflight preserves authored Model order and resolves each exact Model plus exact TrainingResult through `CanonicalRepositoryResolver`; common Task and TrainingResult Architecture identities are retained. No W004 Model runtime, weights/object access, `torch`, or formal result acceptance is used.
- Authored Architecture/Model/seed/Evaluation-stage/axis-value/mapping order is preserved; no Unicode-key sorting occurs in T003-01. Cross-namespace references are supported through exact canonical IDs without namespace-equality logic.
- Current canonical Studies `tile-classifier/rotation-robustness-example-v1` and `rotated-fcos/spatial-screen-example-v1` both local-load successfully and are rejected by planning preflight because they remain `draft`; canonical YAML was not changed.
- Focused: **11 passed**. W002 regression: **148 passed**. Full `mldb_v2/tests`: **749 passed, 1 skipped**.
- Changed Python py_compile PASS; private module import smoke PASS; current-Study read-only smoke **1 passed**; dependency scan CLEAN (no v1/Skeleton/backend/ClearML/torch/Plan/Git runtime dependency); canonical `mldb_data/**/__pycache__` count **0**; `git diff --check -- mldb_v2` PASS.
- Existing tracked-flat `mldb_data` deletions and namespace-first untracked migration remain present and were not cleaned/reverted. No Git staging/commit/stash/restore/push was performed.
- No blocker remains for T003-01.

## Completion Evidence — 2026-09-12
- Implemented private `_StudyPlanningPreflight` / `_StudyPlanningInput` boundary in `mldb_v2/src/study/_planning_preflight.py`; no new public Skeleton/API contract was introduced.
- Subject Study is loaded through W002 `_load_study_definition`, must be `sealed`, then must receive exact W002 verifier success (`valid == True` and `diagnostics == []`). Verifier failures/exceptions are bounded before planning resolution.
- Training-source input retains exact Study/Task/Corpus/TrainProtocol, authored Architecture order, authored sparse axes/value order, authored seed order, Evaluation-stage order, exact Evaluation Corpus/Protocol records, and protocol parameter declarations. Defaults are not materialized.
- Existing-Model input uses only W001 `CanonicalRepositoryResolver` for Model/TrainingResult records, preserves authored Model order, and retains common Task plus TrainingResult Architecture identity. No weight/object access or torch/model-runtime loading is performed.
- Cross-namespace exact references are supported without namespace-equality compatibility logic; W002 remains the sole semantic compatibility gate.
- Current canonical examples both local-load successfully and planning preflight rejects both because they remain `draft`; no `mldb_data/**` mutation was performed.
- Focused: **11 passed**. W002 regression: **148 passed**. Full `mldb_v2/tests`: **749 passed, 1 skipped**.
- Changed Python py_compile PASS; import smoke PASS; dependency scan CLEAN (no v1/Skeleton/backend/ClearML/torch/Plan/Git mutation dependencies); canonical-domain `__pycache__`: **0**; `git diff --check -- mldb_v2` PASS.
- No blocker remains. No Git commit/stage/stash/restore/push was performed; existing flat deletions and namespace-first migration state were preserved.
