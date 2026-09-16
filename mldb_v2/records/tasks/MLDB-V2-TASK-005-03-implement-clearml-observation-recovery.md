# MLDB-V2-TASK-005-03: Implement ClearML observation and candidate recovery

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-005
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-005-02]
- **outputs**: ClearML active/terminal observation and ordered candidate recovery, focused tests

## Goal
Recover exact logical stage ownership and convert ClearML runtime state into frozen backend-neutral observations/terminal candidates without deriving identity from presentation fields.

## Work
- Query by deterministic ownership metadata established by T005-02 and validate exact StageKey/source ownership before returning anything.
- Distinguish active from terminal work and preserve opaque ClearML Task IDs in attempt order.
- Reconstruct complete ordered AttemptSummary values and terminal completed/failed/cancelled candidates from verified backend metadata/artifact projections.
- Reject ambiguous duplicate ownership, stage/source mismatch, malformed terminal payloads, or non-recoverable ownership instead of guessing.
- Do not parse human Task names; do not accept candidates canonically or mutate StudyResult/TrainingResult/EvaluationResult.
- Keep cancellation/logs in T005-04 and scheduling/retry/heartbeat/GPU policy in ClearML.

## Done condition
`observe`/`collect` recover one exact logical stage across process restarts and emit only frozen BackendObservation/TerminalCandidate values with deterministic attempt ordering.

## Verification
Focused mocked observation/recovery tests only: active/terminal transitions, retry ordering, ambiguous ownership, metadata mismatch, malformed candidate rejection, and no human-name parsing. No broad regression; no commit/stage/push.
## Completion Evidence — 2026-09-13
- Added `mldb_v2/src/backend/_clearml_observation.py` with `ClearMLObservationService.observe()` and `ClearMLObservationService.collect()` only; admission, cancellation/logs, result acceptance, canonical persistence, and Study mutation remain outside this Task.
- Ownership recovery derives project `mldb/<namespace>` and `_ownership_key_for_stage_key()` from canonical JSON of exactly StudyResult ID, trial ID, stage kind, and Evaluation coordinate. Focused parity tests compare it directly with T005-02 `_ownership_key()` for training and evaluation.
- Recovery searches searchable ownership metadata, returns `None` on zero matches, rejects duplicate ownership, restores canonical `mldb.stage_input`, and verifies exact requested StageKey, admission metadata lineage, ownership configuration, source commit, CommonExecutionHarness symbol, and public-parameter projection. Human Task names are never parsed.
- Added SDK-neutral `ClearMLObservationClient` plus `ClearMLRuntimeProjection`. The client seam supplies backend-declared ordered opaque execution IDs and stored CommonExecutionHarness terminal candidates; the core module imports no ClearML SDK.
- Active work maps to exact frozen `ActiveBackendObservation(state="active", backend="clearml")` with ordered opaque execution IDs preserved. Active retry history may contain prior failed/cancelled harness candidates but rejects a completed attempt followed by more active work.
- Terminal recovery validates one exact harness TerminalCandidate per terminal attempt, reconstructs the non-empty ordered AttemptSummary history, preserves failed/cancelled retries before later success, and rejects duplicate execution IDs, invalid timestamps/order, malformed diagnostics, StageKey mismatch, or status contradictions.
- Completed training candidates validate the exact canonical weights ArtifactRef. Completed evaluation candidates validate finite numeric metrics and exact candidate ArtifactRefs including format/schema. Failed/cancelled candidates require a diagnostic and `result=None`; backend completion remains provisional and no acceptance call occurs.
- Focused T005-03 pytest: **23 passed**. T005-02 + T005-03 focused join: **37 passed**. Full `mldb_v2/tests` was intentionally not run.
- Changed Python `py_compile`: PASS. Import smoke: PASS. AST dependency/genericity scan: PASS. Direct whitespace scan: PASS. `git diff --check --` on owned paths: PASS.
- Existing unrelated dirty/untracked state was preserved. Parallel T005-04 cancellation/log files were observed in the working tree but were not modified by this Task. No commit/add/stash/reset/clean/restore/push was performed.
- No T005-03 implementation blocker remains. T005-05 still owns real ClearML SDK activation/composition and adapter-level integration verification.
