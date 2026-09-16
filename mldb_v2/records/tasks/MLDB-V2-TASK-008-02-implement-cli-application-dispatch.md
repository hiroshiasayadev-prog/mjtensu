# MLDB-V2-TASK-008-02: Implement CLI Application dispatch

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-008
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-008-01]
- **outputs**: concrete `CliApplicationAdapter` delegation and fake-Application focused tests

## Start gate clarification — 2026-09-13
W007 completion is not required. Start once T008-01 is present and W007 has established the exact public `mldb_v2/src/api/application_interface.py` mirror seam. Do not runtime-import Frozen Skeleton as a substitute.

## Exclusive ownership
- `mldb_v2/src/cli/adapter.py`
- `mldb_v2/tests/test_cli_adapter.py`

## Work
- Implement one-shot normalized-request dispatch exclusively through `ApplicationInterface` methods.
- Cover discovery/inspect, validate/verify/seal/plan, run/resume/rerun/cancel/advance, watch/logs/doctor using fake ApplicationInterface values.
- Preserve exact API response values; do not create CLI domain replacements or reshape structured values.
- ID-less `ps`/`get`/`status`/`watch` remain discovery operations; `watch` calls query/observation only; `run`/`resume` delegate foreground driver methods; `advance` delegates exactly one pass.
- Map selectors only where semantically supported and reject impossible normalized combinations rather than ignoring them.
- No filesystem/YAML/repository resolver/BackendPort/ClearML imports and no W006/W007 private-helper dependency.

## Verification
Focused fake-Application delegation tests, mutation/read boundary tests, exact Skeleton adapter-shape check, and forbidden-import scan. No broad regression; no commit/stage/push.

## Completion evidence - 2026-09-14
- Confirmed the W007 public runtime mirror seam exists before implementation: `src/api/application_interface.py`, `query_interface.py`, and `errors.py` are all present; no Frozen Skeleton runtime fallback is used.
- Added `src/cli/adapter.py` as a one-shot normalized-request dispatcher over public `ApplicationInterface` methods only.
- Discovery behavior: `ps`, ID-less `status`, ID-less `watch`, and ID-less `get` delegate to public list/query operations; exact `status`/`watch` use read-only observation and never progression.
- Execution behavior: `run`, `resume`, `rerun`, `cancel`, and `advance` delegate to their Application-owned methods; `advance` performs exactly one Application call and watch never calls progression methods.
- Authoring behavior: `validate`, `verify`, `seal`, and `plan` delegate through public Application methods with explicit definition-scope mapping; exact/bulk seal intent is preserved.
- Logs build the exact public `BackendLogRequest` (`stage` -> `coordinate`) and delegate only to `read_backend_logs`; doctor delegates only to `diagnose`.
- API return objects are returned unchanged; the adapter defines no CLI-domain response replacement.
- Adapter-level validation rejects impossible normalized combinations instead of silently ignoring them, including exact get/watch with selectors and aggregate-definition exact lookup.
- Focused fake-Application verification: `.\.venv\Scripts\python.exe -m pytest mldb_v2\tests\test_cli_adapter.py -q` -> `33 passed in 0.21s`.
- `py_compile`, Skeleton `dispatch` public-shape mirror test, forbidden dependency/private-helper scan, and trailing-whitespace scan passed. No broad regression was run.

## W007 public-contract integration findings
- `RunRequest.backend` is optional in the Frozen CLI request contract, and CLI operations require configured default-backend behavior when `--backend` is omitted, but public `ApplicationInterface.run_study(*, study, backend)` requires a backend and exposes no default-backend delegation seam. The adapter therefore rejects backend-omitted `run` rather than inventing a backend or reaching into W007/private/runtime configuration.
- `ValidateRequest` / `VerifyRequest` expose `fail_fast`, but public `ApplicationInterface.validate_scope()` / `verify_scope()` expose no fail-fast argument or equivalent delegation seam. `fail_fast=False` maps to the normal complete-report API; explicit `fail_fast=True` is rejected rather than silently ignored or emulated by reshaping the returned report.
- These findings are not worked around in CLI code and remain for W007/W008 integration ownership.

No commit/stage/stash/reset/clean/restore/checkout/worktree/push was performed.
