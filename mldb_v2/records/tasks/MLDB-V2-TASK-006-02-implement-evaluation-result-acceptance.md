# MLDB-V2-TASK-006-02: Implement evaluation candidate acceptance

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-006
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-003, MLDB-V2-WORK-004]
- **outputs**: evaluation terminal-candidate acceptance boundary and focused tests

## Start gate clarification — 2026-09-13
W003 verification is completed. W004 has already established the StageInput/CandidateOutcome runtime seam, and T004-02 has completed the artifact byte-integrity runtime required by evaluation acceptance. T006-02 may therefore start before full W004 closure. `depends_on: W004` remains the W006 integration/completion gate, not an implementation-start serialization requirement. T004-04 candidate production is not a prerequisite for this persistence-free acceptance implementation.

## Goal
Convert one terminal evaluation candidate plus exact StudyResult/Plan/StageInput lineage into a complete canonical EvaluationResult, enforcing the sealed EvaluationProtocol formal metric/artifact declaration exactly.

## Work
- Consume frozen ResultAcceptance shapes, W003 Plan pins/coordinate lineage, W004 backend value seam, and actual T004-02 artifact byte-integrity runtime; do not guess or redefine them.
- Validate exact StudyResult/Plan/trial/coordinate/stage/source/Model/Corpus/Protocol lineage and ordered terminal attempts before accepting payload.
- Require every declared required metric/artifact, permit absent optional outputs, reject undeclared formal outputs, enforce metric integer/number finite rules, and enforce artifact format/schema plus URI/bytes/SHA integrity.
- Failed/cancelled backend candidates map directly to terminal formal results; invalid backend-completed candidates become failed EvaluationResults with bounded acceptance diagnostics.
- Keep this Task persistence-free and disjoint from T006-01; canonical child-before-parent writes belong to T006-04.
- Do not observe/admit backend work, derive readiness, mutate StudyResult, or consume extra backend telemetry as formal output.

## Done condition
Evaluation acceptance deterministically yields one complete valid EvaluationResult or failed/cancelled fallback without silently dropping required outputs or accepting undeclared formal values.

## Verification
Focused acceptance tests only: required/optional metrics/artifacts, type/finite checks, format/schema/hash/size failures, lineage mismatch, attempts, failed fallback, no writes/backend calls. No broad regression; no commit/stage/push.

## Completion evidence — 2026-09-13
- Implemented persistence-free `_accept_evaluation_result` in `mldb_v2/src/verification/_evaluation_result_acceptance.py`.
- Reuses W003 `_validate_study_plan` and exact Plan pins/coordinates; reuses W004 `EvaluationStageInput` / Evaluation candidate seams, `_resolve_model_lineage`, canonical weights ref validation, and `_ObjectByteAccess.read_verified`.
- Sealed EvaluationProtocol declarations are enforced exactly for required/optional/undeclared metrics and artifacts; integer/number bool and non-finite values reject; artifact format/schema/URI/bytes/SHA are verified without repair.
- Backend failed/cancelled map directly; malformed completed payloads/StageKeys become bounded `evaluation_acceptance_failed` formal failures. Request/StageInput lineage mismatches reject before candidate acceptance.
- Focused verification: `40 passed`; changed Python `py_compile` PASS; import smoke PASS; dependency scan PASS; no Skeleton/ClearML/persistence dependency.
- Full `mldb_v2/tests` intentionally not run. No commit/add/stash/reset/clean/restore/push performed.
