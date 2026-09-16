# MLDB-V2-TASK-004-03: Implement Training runtime

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-004
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-004-01, MLDB-V2-TASK-004-02]
- **outputs**: Training StageInput preflight/invocation and provisional training candidate payload

## Goal
Implement backend-neutral training execution from one frozen TrainingStageInput through the W002 Train callable and W004 canonical-weight/artifact boundaries.

## Work
- Validate exact TrainingStageInput lineage and frozen value shape.
- Materialize Corpus, construct exact TrainContext with a fresh Architecture, invoke TrainProtocol, and validate returned module strictly.
- Publish canonical-weight candidate bytes and return the frozen `TrainingCandidateResult` payload shape.
- Consume `mldb_v2/src/backend/stage_input.py` and `candidate_outcome.py`; do not redefine those values.
- Do not invent backend AttemptSummary/execution IDs; full TerminalCandidate assembly is outside this stage-specific runtime.
- Do not schedule backend work, write formal Results, mutate StudyResult, or add model-family branches.

## Verification
Focused training-runtime tests, direct dependency smoke, py_compile/import, and genericity scan only. No full mldb_v2 regression.

## Completion evidence — 2026-09-13
- Added `mldb_v2/src/training/runtime.py` with `_execute_training_stage(...) -> TrainingCandidateResult`.
- Exact TrainingStageInput preflight, sealed Task/Corpus/Architecture/TrainProtocol lineage, complete parameter validation, manifest-driven Corpus materialization, exact TrainContext construction, Train callable invocation, fresh-Architecture strict learned-state compatibility, canonical state-dict serialization, and immutable weights publication are implemented without backend/result-acceptance coupling.
- Task-session focused verification: **27 passed**; direct T004-01/T004-02 smoke **24 passed, 1 skipped**; W002 Architecture/Train callable smoke **4 passed, 93 deselected**; common parameter smoke **31 passed**; py_compile/import/genericity/diff checks passed.
- Coordinator shared-tree Phase B join with T004-04: **74 passed, 1 skipped**. The skip is the known Windows symlink-permission probe.
- No blocker remains. Full W004 regression is deferred to T004-05 closure.
