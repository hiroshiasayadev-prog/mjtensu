# MLDB-V2-TASK-004-04: Implement Evaluation runtime

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-004
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-004-01, MLDB-V2-TASK-004-02]
- **outputs**: Evaluation StageInput preflight/invocation and provisional evaluation candidate payload

## Goal
Implement backend-neutral evaluation execution from one frozen EvaluationStageInput through canonical Model loading, the W002 Evaluation callable, and W004 artifact publication.

## Work
- Validate exact EvaluationStageInput/runtime Model lineage and frozen value shape.
- Materialize Corpus, load the canonical Model strictly, construct exact EvaluationContext, and invoke EvaluationProtocol.
- Validate declared candidate metrics/artifacts and publish produced artifact bytes with declared immutable metadata.
- Return the frozen `EvaluationCandidateResult` payload shape from `mldb_v2/src/backend/candidate_outcome.py`; do not redefine backend-neutral values.
- Do not invent backend AttemptSummary/execution IDs; full TerminalCandidate assembly is outside this stage-specific runtime.
- Do not write formal Results, mutate StudyResult, implement result acceptance, or add metric/model-family branches.

## Verification
Focused evaluation-runtime tests, direct dependency smoke, py_compile/import, and genericity scan only. No full mldb_v2 regression.

## Completion evidence — 2026-09-13
- Added `mldb_v2/src/evaluation/runtime.py` with `_execute_evaluation_stage(...) -> EvaluationCandidateResult`.
- Pinned definition root and runtime canonical Model/TrainingResult root remain distinct; exact runtime snapshot/lineage, canonical-weight integrity/strict load, sealed Task/Corpus/EvaluationProtocol/Architecture lineage, complete parameters, exact EvaluationContext, declaration-driven metrics, and declared artifact publication are implemented without backend/result-acceptance coupling.
- Task-session focused verification: **47 passed, 1 skipped**; direct T004-01/T004-02 smoke **38 passed, 1 skipped**; W002 Evaluation loader smoke **36 passed, 61 deselected**; py_compile/import/genericity/diff checks passed.
- Coordinator shared-tree Phase B join with T004-03: **74 passed, 1 skipped**. The skip is the known Windows symlink-permission probe.
- No blocker remains. Full W004 regression is deferred to T004-05 closure.
