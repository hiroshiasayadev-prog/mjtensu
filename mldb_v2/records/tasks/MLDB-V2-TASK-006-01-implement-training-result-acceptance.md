# MLDB-V2-TASK-006-01: Implement training candidate acceptance

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-006
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-003, MLDB-V2-WORK-004]
- **outputs**: training terminal-candidate acceptance boundary and focused tests

## Start gate clarification — 2026-09-13
W003 verification is completed. W004 has already established the StageInput/CandidateOutcome runtime seam, and T004-01 has completed the canonical-weight/Model runtime symbols required by training acceptance. T006-01 may therefore start before full W004 closure. `depends_on: W004` remains the W006 integration/completion gate, not an implementation-start serialization requirement. T004-03 candidate production is not a prerequisite for this persistence-free acceptance implementation.

## Goal
Convert one terminal training candidate plus exact StudyResult/Plan/StageInput lineage into a complete canonical TrainingResult and optional deterministic Model, with invalid completed candidates converted to failed formal results.

## Work
- Consume the frozen ResultAcceptance request/result shapes, exact W003 Plan lineage, W004 backend value seam, and actual T004-01 canonical-weight/Model runtime symbols; do not guess or redefine them.
- Validate exact stage/source identity, attempts, definitions/pins, source commit, artifact URI/bytes/SHA, and canonical `pytorch-state-dict/v1` weights compatibility.
- For valid completed candidates, construct deterministic TrainingResult and Model identities exactly from StudyResult + trial; failed/cancelled candidates produce no Model.
- A backend-completed candidate failing acceptance becomes a failed TrainingResult with bounded acceptance diagnostic; never repair malformed success payloads.
- Keep this Task persistence-free so T006-01 and T006-02 can run safely in parallel; canonical child-before-parent writes belong to T006-04.
- Do not observe/admit backend work, derive readiness, mutate StudyResult, or implement ClearML-specific logic.

## Done condition
Training acceptance deterministically yields one complete valid TrainingResult plus Model only for accepted success, and a failed/cancelled TrainingResult otherwise, with no backend or persistence side effects.

## Verification
Focused acceptance tests only: success, backend failure/cancel, weights integrity/format/lineage mismatch, deterministic IDs, failed fallback, exact attempts, and no writes/backend calls. No broad regression; no commit/stage/push.

## Completion evidence — 2026-09-13
- Implemented persistence-free `_accept_training_candidate` in `mldb_v2/src/verification/_training_result_acceptance.py`; no public ResultAcceptor composition or Evaluation acceptance changes.
- Deterministic children follow the frozen StudyResult identity contract: `<StudyResult.id>-<trial>-train` and `<StudyResult.id>-<trial>-model`, with Model -> TrainingResult linkage validated by actual runtime validators.
- Consumes actual W003/W004 runtime boundaries: validated StudyPlan lineage/pins, TrainingStageInput/CandidateOutcome, canonical weights ArtifactRef/byte verification/loading, strict fresh-Architecture state loading, TrainingResult validation, Model validation, and canonical AttemptSummary validation. No Skeleton runtime imports or ClearML dependency.
- Backend failed/cancelled candidates preserve exact ordered attempts and diagnostic with no Model; malformed backend-completed candidates become canonical failed TrainingResults with bounded `training_acceptance_failed` diagnostic and no Model.
- Focused verification: 24 passed in 3.34s; changed Python py_compile PASS; import smoke PASS; forbidden dependency scan PASS; tracked git diff --check -- PASS and untracked changed-file git diff --no-index --check PASS. Full mldb_v2/tests intentionally not run.
- No canonical persistence, StudyResult mutation, backend observe/admit/cancel calls, commit, stage, stash, reset, clean, restore, or push performed. W004 closure remains the W006 integration/completion gate, not a blocker for this completed Task implementation.
