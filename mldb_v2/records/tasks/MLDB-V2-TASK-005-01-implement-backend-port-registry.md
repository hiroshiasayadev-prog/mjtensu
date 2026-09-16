# MLDB-V2-TASK-005-01: Implement generic BackendPort registry and configuration

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-005
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-004]
- **outputs**: runtime BackendPort contract integration, backend registry/config boundary, focused tests

## Start gate clarification — 2026-09-13
W004 owner has already established the frozen backend-neutral runtime value seam under `mldb_v2/src/backend/`: `stage_input.py`, `candidate_outcome.py`, and `execution_harness.py`. T005-01 may therefore start before full W004 closure and must consume those files read-only. `depends_on: W004` remains the W005 completion/integration gate, not an implementation-start serialization requirement.

## Goal
Establish the backend-neutral runtime boundary that registers/configures execution backends and consumes the frozen StageKey, StageInput, BackendObservation, and TerminalCandidate contracts without backend-specific identity leaking into canonical values.

## Work
- Mirror the frozen `backend_port.py` contract in runtime `mldb_v2.src.backend.backend_port`; consume the already-established W004 runtime symbols from `mldb_v2.src.backend.stage_input` and `mldb_v2.src.backend.candidate_outcome` without redefining them.
- Add a small backend registry/configuration boundary keyed by generic backend type name; endpoint, credentials, queue, worker, and SDK objects remain operational only.
- Preserve exact StageKey/StageInput values across `admit`, `observe`, `collect`, and `cancel_study`; do not add scheduling, readiness, retry, heartbeat, GPU, or Study mutation policy.
- Keep ClearML-specific admission/observation/cancellation implementation outside this Task.
- Do not modify Frozen Specs/Skeletons, W004-owned `stage_input.py`/`candidate_outcome.py`/`execution_harness.py`, canonical Plan/Result formats, or public API/CLI.

## Done condition
A configured backend can be resolved by generic backend name through one runtime BackendPort boundary, with frozen value contracts unchanged and no backend configuration serialized into canonical MLDB records.

## Verification
Focused registry/config/port tests only: exact contract shape, unknown backend/config failures, no canonical config leakage, no Skeleton runtime imports, py_compile/import, and dependency scan. Do not run full `mldb_v2/tests`; no commit/stage/push.

## Completion Evidence - 2026-09-13
- Added `mldb_v2/src/backend/backend_port.py` as the runtime mirror of the frozen BackendPort Protocol. It imports actual W004 `StageInput`, `StageKey`, `BackendObservation`, and `TerminalCandidate` values from `mldb_v2.src.backend.*`; no Skeleton runtime values are copied or imported.
- Added operational-only `BackendConfig(backend_type, options)` in `_config.py`. Backend type/option-key validation is bounded; the top-level options mapping is copied and read-only, accepts opaque SDK/config objects, and is hidden from dataclass repr so credentials/config are not projected into canonical values.
- Added `BackendRegistry` in `_registry.py`, keyed only by generic backend type name. Registration rejects invalid/non-callable/duplicate factories; resolution reports unknown types and rejects factories that do not expose the four BackendPort methods. No scheduling/readiness/retry/result-acceptance policy is present.
- Added focused `test_backend_port_registry.py` covering exact Protocol signature/type surface, actual W004 runtime types, registration/resolution, duplicate/invalid/unknown cases, invalid operational config, StageInput/StageKey preservation, BackendObservation/TerminalCandidate integration, operational config non-leakage, and forbidden dependency checks.
- Focused pytest: **16 passed**. Changed Python `py_compile`: PASS. Import smoke for `BackendPort`, `BackendConfig`, and `BackendRegistry`: PASS.
- AST dependency scan: PASS. Runtime source imports contain no `mldb_v2.skeleton`, ClearML, scheduling, execution-readiness, or result-acceptance dependency.
- `git diff --check --` for the changed paths exits clean. Because the repository currently reports the whole `mldb_v2/` tree as untracked, a supplemental direct changed-file scan also confirmed final newlines and no trailing spaces/tabs.
- W004-owned `stage_input.py`, `candidate_outcome.py`, and `execution_harness.py`, Frozen Specs/Skeletons, Work Item status, `tasks/index.md`, and all scope-external files were left unchanged. No commit/add/stash/reset/clean/restore/push was performed.
- No T005-01 implementation blocker remains. W005 completion/integration remains gated by W004 as recorded by the Work Item dependency.
