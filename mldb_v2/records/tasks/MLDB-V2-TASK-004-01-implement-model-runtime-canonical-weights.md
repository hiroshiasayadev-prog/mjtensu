# MLDB-V2-TASK-004-01: Implement Model runtime and canonical weights

- **status**: completed
- **date**: 2026-09-12
- **work_item**: MLDB-V2-WORK-004
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-002]
- **outputs**: canonical-weight and Model runtime under `mldb_v2/src/training/`, focused tests

## Goal
Implement backend-neutral Architecture build invocation, canonical `pytorch-state-dict/v1` serialization/validation/loading, and immutable Model/TrainingResult lineage loading using the W002 executable-definition boundary.

## Work
- Mirror frozen `canonical_weights.py`, `model.py`, and required TrainingResult public shapes without Skeleton runtime imports.
- Build fresh Architecture modules through the existing W002 companion loader; do not invent another import convention.
- Validate plain string-to-Tensor state dicts, detach/move tensors to CPU, serialize directly with `torch.save`, and reject wrapped checkpoints/optimizer/scheduler/scaler/counters/config payloads.
- Load weights with `torch.load(..., map_location="cpu", weights_only=True)`, revalidate the plain mapping, build a fresh Architecture, and strict-load with no key repair/non-strict fallback.
- Resolve Model -> completed TrainingResult -> Architecture lineage exactly; keep object-byte transport/materialization, training invocation, evaluation invocation, formal result acceptance, and backend scheduling outside this Task.

## Done condition
Canonical learned-state bytes can be produced and strictly loaded into the exact fresh Architecture for a valid completed Model lineage with no model-family branches or backend coupling.

## Verification
Cover canonical mapping validity, CPU normalization, direct serialization, strict compatibility, wrapped/invalid checkpoint rejection, exact Model/TrainingResult lineage, no companion-loader duplication, py_compile/import, dependency scan, and full regression.

## Completion evidence — 2026-09-13
- Implemented canonical plain CPU Tensor state-dict serialization/loading, strict fresh-Architecture compatibility, and exact Model -> completed TrainingResult -> Architecture -> Task lineage resolution.
- T004-01 focused tests reported **22 passed**; W002 direct Architecture-loader smoke **7 passed**; changed-module py_compile/import and forbidden-dependency/genericity scans PASS.
- Coordinator Phase-A join reran T004-01 plus T004-02 focused tests in the shared working tree: **40 passed, 1 skipped**. The skip is the T004-02 Windows symlink-permission probe, not T004-01.
- No full `mldb_v2/tests` regression was required at Task completion; broad regression remains deferred to T004-05/W004 closure.
- No commit/stage/stash/worktree operation was performed.