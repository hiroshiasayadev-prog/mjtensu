# MLDB-V2-WORK-002: Catalog verification and sealing

- **status**: completed
- **date**: 2026-09-09
- **depends_on**: [MLDB-V2-WORK-001]
- **source_refs**: `spec:mldb.v2.catalog`, `spec:mldb.v2.verification.definition_lifecycle`, `spec:mldb.v2.verification.executable_integrity`, `spec:mldb.v2.verification.executable_asset_tests`

## Goal
Implement reusable Namespace/Task/Corpus/Architecture/TrainProtocol/EvaluationProtocol/Study definition loading, validation, verification, executable integrity, asset-test gates, and sealing without introducing model-family-specific semantics.

## Boundary
Own `mldb_v2/src/catalog/` and definition-side `mldb_v2/src/verification/`. W002 also owns the reusable definition/interface shape modules required by the frozen Skeleton: `src/training/train_protocol.py`, `src/training/train_interface.py`, `src/evaluation/evaluation_protocol.py`, `src/evaluation/evaluate_interface.py`, and `src/study/study.py`. W003/W004 consume these modules and own planning/runtime behavior around them; they must not reimplement definition validation or callable-interface contracts.

## Task candidates
| task | responsibility | dependency |
|---|---|---|
| MLDB-V2-TASK-002-01 | Implement Namespace/Task/Corpus typed parsing and semantic validation, including manifest/builder structural constraints. | W001 |
| MLDB-V2-TASK-002-02 | Implement Architecture/TrainProtocol/EvaluationProtocol typed parsing plus same-basename companion loading and frozen callable-interface checks. | W001 |
| MLDB-V2-TASK-002-03 | Implement executable sibling/project-source integrity verification and Corpus builder integrity. | T01,T02 |
| MLDB-V2-TASK-002-04 | Implement derived executable-asset pytest gate through the injected PytestRunner boundary. | T02,T03 |
| MLDB-V2-TASK-002-05 | Implement Study definition validation and complete entity-level validate/verify composition, including sealed-reference compatibility and existing-Model lineage checks required by Study format. | T01-T04 |
| MLDB-V2-TASK-002-06 | Implement sealing lifecycle for Task/Corpus/Architecture/TrainProtocol/EvaluationProtocol/Study using the completed validation/verification gates. | T01-T05 |
| MLDB-V2-TASK-002-07 | Focused closure repair: enforce completeness of declared result-affecting project-owned executable sources. | T03,T06 |

## Planning ownership clarification — 2026-09-11
`spec:mldb.v2.verification.definition_lifecycle` requires Study sealing to perform complete planning validation of the declared model source/evaluation stages against referenced sealed definitions. That gate therefore belongs to W002 before sealing, not to downstream W003. W002 validation stops at semantic/reference/lifecycle compatibility: it does not expand grids, create a StudyPlan, pin a Git source commit, or execute companions.

W003 planning preflight may re-resolve the already-sealed Study and invoke W002 verification to establish planning-time usability, but it must not duplicate the Study semantic validator. This removes the prior circular ownership where W002 sealing depended conceptually on validation assigned to a Work Item that itself depends on W002.

## Completion condition
- Classifier, rotated-detector, and custom-evaluation examples parse without common-schema specialization.
- `validate`, `verify`, and `seal` remain distinct operations.
- Sealed executable definitions pin sibling and declared result-affecting project sources exactly.
- Required asset tests use derived canonical locations, never YAML-stored test paths.
- Study sealing validates its declared source/stages against referenced sealed compatible definitions without compiling a Plan.
- Namespace receives no seal lifecycle and `tools/` is never an executable-definition boundary.

## Coordinator closure finding ? W002-CL01 ? 2026-09-12
Independent closure review found one blocking executable-integrity gap. `spec:mldb.v2.verification.executable_integrity` states that an executable importing result-affecting project-owned source without declaring it is non-conforming. The current T002-03 verifier hashes only the declared set and returns `valid=True` for a synthetic companion containing `from product.behavior import VALUE` with no `implementation.sources`. W002 therefore remains `planned` pending T002-07. T002-01..06 remain completed; the repair is additive behind the existing frozen public shape.

## Final closure verification — 2026-09-12
- T002-01 through T002-07 are completed. Closure finding `W002-CL01` is closed by static direct/transitive repository-owned import completeness verification behind the frozen executable-integrity public shape.
- Focused T002-07 executable-integrity regression: **51 passed**. W002 regression: **342 passed, 1 skipped**. Full `mldb_v2/tests`: **738 passed, 1 skipped**.
- Current canonical executable examples remain **10/10 valid**; actual canonical repository listing remains **18 items / 0 issues** and canonical-domain `__pycache__` remains **0**.
- Frozen public executable-integrity shape comparison PASS. Changed Python py_compile/import PASS. Forbidden dependency scan CLEAN. `git diff --check -- mldb_v2` PASS.
- W002 completion conditions are satisfied: definition validation/verification/sealing remain distinct; executable sibling and project-source integrity are enforced; asset-test paths are derived; Study seal-time compatibility is verified without Plan compilation; Namespace has no seal lifecycle; `tools/` is not an executable boundary.
- Existing tracked-flat deletion plus namespace-first untracked migration state remains unchanged. No Git staging/commit/stash/restore/push was performed.

MLDB-V2-WORK-002 may now be used as a downstream dependency.
