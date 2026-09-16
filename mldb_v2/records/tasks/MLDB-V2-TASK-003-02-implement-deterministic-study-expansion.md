# MLDB-V2-TASK-003-02: Implement deterministic Study expansion

- **status**: completed
- **date**: 2026-09-12
- **work_item**: MLDB-V2-WORK-003
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-003-01]
- **outputs**: deterministic training/existing-model trial and Evaluation-coordinate expansion under `mldb_v2/src/study/`, focused tests

## Goal
Expand one validated T003-01 planning input into deterministic private trial/Evaluation-coordinate materialization matching the frozen grid-order/default-resolution contract, without source pinning or Plan persistence.

## Work
- Training trials: Architecture authored order, then swept TrainProtocol keys in ascending Unicode code-point order, each axis value authored order, then seed authored order.
- Existing-Model trials preserve authored Model order and create no training parameter/seed axes.
- Resolve complete Train/Evaluation parameter mappings with W001 common parameter helpers; omitted published keys use protocol defaults, unknown/invalid overrides fail.
- Evaluation coordinates are trial-local and ordered by Evaluation stage authored order, swept EvaluationProtocol keys in Unicode order, and axis values in authored order; a stage with no axes yields exactly one coordinate.
- Allocate exact `trial-NNNN` / `eval-NNNN` IDs, preserving four-digit sequence semantics, and keep expansion side-effect free.
- Do not construct Plan pins, source commits, StudyPlan digest/ID, write canonical records, execute companions, or call backends.

## Done condition
Training and existing-Model planning inputs expand reproducibly into exact ordered private trial/coordinate structures with complete resolved parameters and no mutation/backend effects.

## Verification
Cover axis order, value order, seed order, defaults, zero-axis stages, multiple stages, type-sensitive values, existing-Model order, ID sequence/overflow boundary, no mutation/execution, py_compile/import, and full regression.

## Completion Evidence — 2026-09-12
- Added private deterministic expansion in `mldb_v2/src/study/_grid_expansion.py` with immutable `_ExpandedTrainingSource`, `_ExpandedExistingModelSource`, `_ExpandedEvaluationCoordinate`, and `_ExpandedTrial` structures only; no public `StudyPlan`/`PlanPin` shape was introduced.
- Training expansion order is exact: Architecture authored order -> swept TrainProtocol keys in ascending Python Unicode order -> each axis value authored order -> seed authored order. Zero explicit axes yields Architecture x seed trials without a synthetic axis.
- Complete Train/Evaluation parameter mappings are produced only through W001 `_resolve_public_parameters`; resolved mapping declaration order is preserved, omitted keys receive defaults, and training seed remains separate from protocol parameters.
- Expansion-time axis uniqueness reuses `_public_parameter_values_equal`, preserving type-sensitive semantics (`true`, `1`, `1.0` distinct) including recursive list/object equality. Defensive Architecture/Model/seed uniqueness checks are bounded private failures.
- Existing-Model expansion preserves authored Model order, emits one trial per Model with only `kind=existing_model` and exact Model ID, and performs no weights/runtime/lineage-based grid expansion.
- Evaluation expansion is trial-local: stages stay in authored order, swept keys use Unicode order, values stay authored order, zero-axis stages emit one default-only coordinate, IDs span stages and reset to `eval-0001` for each next trial.
- Trial IDs are exact sequential `trial-0001..trial-9999`; Evaluation IDs are exact `eval-0001..eval-9999`. Cardinality is checked before final expansion and deterministic bounded errors reject a 10000th Trial or per-trial Evaluation coordinate.
- Focused tests cover ordering, defaults, type sensitivity, Existing Model behavior, Evaluation reset/order, sequence limits, purity/determinism, and synthetic canonical repo -> T003-01 preflight -> T003-02 expansion. Focused: **20 passed**; T003-01 + T003-02 regression: **31 passed**.
- Full `mldb_v2/tests` regression before final record update: **769 passed, 1 skipped** (baseline 749 + 20 T003-02 tests).
- Changed src/test py_compile PASS; import smoke PASS; repeated-expansion determinism probe PASS; dependency scan CLEAN (no mldb v1, Skeleton runtime, backend/ClearML, torch, StudyPlan/PlanPin, Git/subprocess); canonical `mldb_data/**/__pycache__` count **0**.
- Expansion is side-effect free and synthetic canonical tree bytes remain unchanged; no companion invocation, backend call, Git mutation, Plan write, `mldb_data/**` write, or `mldb_tests/**` write occurs.
- Existing tracked-flat `mldb_data` deletions, namespace-first migration, and unrelated dirty state were preserved; no commit/stage/stash/restore/revert/push was performed. No blocker remains for T003-02.
