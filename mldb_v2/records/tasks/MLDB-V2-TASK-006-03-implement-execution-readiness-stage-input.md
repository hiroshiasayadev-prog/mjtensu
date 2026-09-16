# MLDB-V2-TASK-006-03: Implement semantic readiness and runtime StageInput construction

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-006
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-003, MLDB-V2-TASK-006-01]
- **outputs**: pure execution-readiness resolver and ready-stage StageInput construction, focused tests

## Start gate clarification — 2026-09-13
W003 verification and T006-01 are completed. W004 has already established the StageInput/CandidateOutcome seam plus canonical Model/TrainingResult runtime required here, so T006-03 may start before full W004 closure. T005/W005 is not an implementation-start dependency because this Task performs no backend calls.

## Runtime value ownership
`mldb_v2/src/results/study_result.py` does not yet exist. T006-03 owns the runtime mirror of the frozen `StudyResult` public value shape needed by semantic readiness. Keep this module value/validation-only; persistence/mutation remains T006-04.

## Goal
Derive ready/skipped planned stages only from immutable Plan plus canonical outcomes, and construct exact runtime StageInput values for newly-ready training/evaluation coordinates including accepted Model lineage.

## Work
- Implement the frozen `ExecutionReadinessResolver` semantics: training initially ready; evaluations wait for accepted training+Model; existing-Model evaluations are ready after Model integrity; evaluation siblings are independent.
- Derive only `upstream_failed`/`upstream_cancelled` semantic skips here; cancelling admits nothing new and backend ownership remains outside this boundary.
- Build exact training/evaluation StageInput values from Plan content, StudyResult identity, source commit/pins, and canonically accepted Model/TrainingResult lineage; never from backend metadata.
- For runtime-produced and existing Models, use the same RuntimeModel shape and verified immutable weights lineage.
- Reuse actual W004 Model/TrainingResult/canonical-weight runtime; do not invent a competing loading or identity boundary.
- Do not observe/admit/cancel backend work, persist children, mutate StudyResult, or encode ClearML/resource policy.

## Done condition
Given only Plan + canonical result/model state, the resolver returns deterministic ready/skip sets and exact backend-neutral StageInputs for every newly-ready coordinate.

## Verification
Focused pure readiness/StageInput tests only: training/existing-model paths, upstream failure/cancel, sibling independence, cancelling, exact pins/digest/model lineage, and proof no backend calls occur. No broad regression; no commit/stage/push.

## Completion evidence — 2026-09-13
- Added runtime StudyResult mirror and exact validator: `mldb_v2.src.results.study_result._validate_study_result`.
- Added pure semantic resolver: `mldb_v2.src.study.execution_readiness.ExecutionReadinessResolver`.
- Added ready-only exact StageInput builder: `mldb_v2.src.study.execution_readiness._build_stage_input`.
- Runtime-produced and existing Models both resolve through W004 `_resolve_model_lineage` and materialize the same `RuntimeModel` shape; no BackendPort/ClearML/persistence/mutation dependency was introduced.
- Focused verification: `mldb_v2/tests/test_execution_readiness.py` -> **15 passed**. Changed Python `py_compile`, import smoke, dependency scan, and changed-path whitespace checks pass.
