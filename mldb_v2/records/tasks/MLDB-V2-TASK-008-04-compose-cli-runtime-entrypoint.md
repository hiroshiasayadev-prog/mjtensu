# MLDB-V2-TASK-008-04: Compose CLI runtime and entrypoint

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-008
- **task_type**: integration
- **depends_on**: [MLDB-V2-TASK-008-01, MLDB-V2-TASK-008-02, MLDB-V2-TASK-008-03]
- **outputs**: parser→adapter→renderer runtime composition, fake-Application CLI integration, repository-local `mldb` entrypoint wiring

## Start gate clarification — 2026-09-13
W007 completion is not required. Compose against an injected/fake `ApplicationInterface`; concrete W007 construction remains T008-05. The repository currently has no Python packaging/console-script configuration, so do not invent a broad packaging migration.

## Exclusive ownership
- `mldb_v2/src/cli/main.py`
- `mldb_v2/src/cli/__main__.py`
- `mldb_v2/tests/test_cli_main.py`
- `mldb.cmd` — exact minimal repository-local Windows command shim selected after confirming root `pyproject.toml` / `setup.py` / `setup.cfg` are absent; this path was recorded here before first edit.

## Work
- Compose argv parser -> normalized request -> `CliApplicationAdapter.dispatch()` -> rendering/exit behavior with injectable Application factory.
- Keep `watch` read-only; any refresh/polling belongs here, not adapter/application semantics, and must not invent progression calls.
- Preserve immediate durable identity/output behavior provided by Application methods; do not reimplement run/resume progression.
- Handle KeyboardInterrupt as local CLI interruption only; never translate it into cancellation.
- Provide fake-Application end-to-end command tests without repository/backend/ClearML access.

## Verification
Focused CLI main tests and repository-local entrypoint smoke with fake application. No W007 concrete integration, no broad regression, no commit/stage/push.

## Completion evidence — 2026-09-14
- Reconfirmed actual repo gates: W007 and T008-01/T008-02/T008-03 are `completed`; all three public API mirror files required by T008-02 are present.
- Reconfirmed repository root has no `pyproject.toml`, `setup.py`, or `setup.cfg`; no repository-wide Python packaging migration was introduced.
- Added `src/cli/main.py` composition only: argv parsing -> injected `ApplicationInterface` factory -> one `CliApplicationAdapter.dispatch()` -> renderer -> stdout/stderr/exit policy.
- Added `src/cli/__main__.py` as the Python module entrypoint; it contains no Application construction or domain semantics.
- Added root `mldb.cmd` as the recorded minimal repository-local Windows shim. From repository root, `mldb --help` resolves to `.venv\Scripts\python.exe -m mldb_v2.src.cli` when the venv exists, with `python` fallback.
- `watch` remains one read-only adapter dispatch; no polling interval, termination rule, progression call, repository read, backend call, or ClearML access was added in CLI main.
- `advance` is one dispatch only. `run`/`resume` progression remains Application-owned.
- `KeyboardInterrupt` returns local foreground exit 130 with `interrupted` on stderr and never invokes `cancel_study`.
- Public Application failures render only bounded `code: message` on stderr with non-zero exit; structured success output stays on stdout.
- `validate`/`verify --fail-fast` is accepted but deliberately evaluates the full selected scope: CLI runtime clears the optional short-circuit hint before dispatch. This is allowed because the CLI contract says fail-fast MAY stop early; validation semantics remain wholly Application-owned.
- Focused verification: `.\.venv\Scripts\python.exe -m pytest mldb_v2\tests\test_cli_main.py -q` -> `12 passed in 0.57s`.
- `py_compile` passed for `main.py`, `__main__.py`, and `test_cli_main.py`; owned-file trailing-whitespace scan passed.
- Repository-local entrypoint smoke: `cmd /c "mldb --help"` from repository root -> exit 0 and `usage: mldb ...`.

## Closure blockers handed to T008-05 / W008 closure
- CLI operations require a configured default backend when `run --backend` omits the name, but actual repo/spec exposes no default-backend configuration source and `ApplicationInterface.run_study` requires an explicit backend. T008-04 does not invent `clearml`, a new environment variable, or private configuration access; backend-omitted `run` therefore remains an exact integration blocker.
- The repository-local `mldb.cmd` provides the requested local command smoke without inventing packaging, but the repo still has no installation/console-script mechanism that makes `mldb` globally available outside the repository/path context. If W008 closure requires a truly installed command, T008-05 must choose an explicit existing configuration source or approve a minimal packaging/installation mechanism.
- Concrete production `ApplicationInterface` construction is intentionally not wired here; T008-05 owns that integration even though W007 is now completed.
- No full suite, concrete backend E2E, commit, stage, stash, reset, clean, restore, or push was performed.
