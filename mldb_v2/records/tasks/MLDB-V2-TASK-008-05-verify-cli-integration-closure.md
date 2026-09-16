# MLDB-V2-TASK-008-05: Verify concrete CLI integration and close W008

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-008
- **task_type**: verification
- **depends_on**: [MLDB-V2-TASK-008-01, MLDB-V2-TASK-008-02, MLDB-V2-TASK-008-03, MLDB-V2-TASK-008-04, MLDB-V2-WORK-007]
- **outputs**: concrete W007 integration, installed-command smoke, W008 integrated regression/adversarial evidence

## Goal
Join the completed CLI adapter to the concrete W007 Application without CLI-side workarounds, then close W008.

## Exclusive ownership
- `mldb_v2/tests/test_cli_integration.py`
- W008 Task/Work Item completion evidence
- only minimal edits to W008-owned `mldb_v2/src/cli/**` required by true integration findings; W007 files remain forbidden

## Work
- Construct/use the concrete W007 Application through its public boundary only and run representative discovery, authoring, execution-control, monitor/log/doctor CLI paths.
- Verify installed `mldb` entrypoint, ID-less discovery, safe `--all`, read-only `watch`, foreground run/resume delegation, one-pass advance, structured output shape, and unsupported backend-log capability surfacing.
- Run forbidden-dependency scans proving CLI has no filesystem/YAML/repository-core/BackendPort/ClearML/Study-driver-private access.
- If W007 public behavior differs from Frozen contract, record a W007 finding; do not absorb it in CLI.
- Leave production real-backend final E2E/conformance to W009.

## Verification
Run W008 integrated tests, concrete W007 integration, installed entrypoint smoke, full `mldb_v2/tests`, `git diff --check -- mldb_v2`, and one bounded adversarial CLI review. No commit/stage/push.

## Closure checkpoint — 2026-09-14
- Cheap W008 focused join passed: **93 passed in 0.82s** across parser/adapter/rendering/main tests.
- The original production-composition blocker was repaired in W007: public `compose_application()` now supplies `ApplicationComposition(application, default_backend)` and CLI production `main()` joins through that public seam only.
- Post-repair W008 join passes: concrete CLI integration **13 passed**, aggregate parser/adapter/rendering/main/integration **106 passed**, W007 composition regression **29 passed**, outside-cwd repository command smoke PASS, forbidden-import/static checks PASS, and full `mldb_v2/tests` **1342 passed, 3 skipped in 371.89s**.
- Adversarial review found a new W007-owned blocker that the initial concrete test fixture had hidden by replacing `application._authoring._verifier` with a test verifier. Production `Application` owns `_ObjectByteAccess`, but `Application -> AuthoringPlanningService -> _RepositoryDefinitionVerifier` does not pass it as `object_access`.
- Exact production reproduction on a sealed training Study: `verify_scope(...)` returns `valid: False` with `corpus_object_invalid: Corpus object-byte access is not configured`; `plan_study(...)` returns public `validation_failed`. Therefore production `mldb plan` / `mldb run` cannot satisfy the concrete W007 join for normal Corpus-backed Studies.
- W002 explicitly requires Corpus verification to use injected W001 object-byte access. This is a W007 Application-composition wiring defect, not a CLI policy/configuration defect. T008-05 must not repair it in `src/cli/**` or mutate W007-owned files.
- `--fail-fast` remains non-blocking: the CLI contract says early stopping MAY occur, so full-scope delegation remains conformant.
- T008-05 remains `planned`; W008 cannot close until the focused W007 object-byte verifier wiring is repaired and concrete `plan/run` are reverified without private test substitution.

## Closure completion ? 2026-09-14 after W007 object-byte wiring repair
- Focused W007 repair wires the Application-owned `_ObjectByteAccess` through `AuthoringPlanningService` into the existing `_RepositoryDefinitionVerifier(object_access=...)`; the W002 verifier algorithm, Frozen API interfaces, W006 progression, and CLI architecture are unchanged.
- First production-composition gate on the original one-entry sealed fixture no longer reports `Corpus object-byte access is not configured`; it reaches exact object verification instead, proving the configured object-byte seam is used.
- Production CLI on a production-valid sealed Study passed without verifier monkeypatching: `verify study study-ns/study-v1 --json` -> exit **0**, `valid: true`; `plan study-ns/study-v1` -> exit **0** and a canonical StudyPlan.
- Removed the closure-invalid `application._authoring._verifier = _ValidVerifier()` substitution from `test_cli_integration.py`. The concrete fixture now supplies exact Corpus object bytes, valid executable companions/hashes, canonical asset tests, and uses the real W007/W002 verifier path.
- W007 focused API regression: **41 passed**; `test_api_composition.py`: **7 passed**; `test_api_application_integration.py`: **17 passed** including explicit object-byte wiring coverage.
- Relevant W002 verification/Corpus/object-byte regression: **175 passed, 1 skipped**.
- Concrete CLI integration: **13 passed in 100.83s**. W008 aggregate parser/adapter/rendering/main/integration: **106 passed in 101.80s**.
- Adversarial/static checks PASS: production composition, configured/default/no-default behavior covered by W008 tests, outside-cwd `mldb.cmd --help` and `ps --json`, private-verifier replacement scan, CLI forbidden imports, `py_compile`, trailing whitespace, read-only watch, one-pass advance, unsupported logs, and structured-output paths. `--fail-fast` remains conformant under the Frozen optional-early-stop contract.
- Full `python -m pytest mldb_v2/tests -q` was run once after all production/W008 gates and passed **1343 passed, 3 skipped in 380.30s**.
- T008-05 is `completed`; no production real-backend training E2E was added because that remains W009 scope.
