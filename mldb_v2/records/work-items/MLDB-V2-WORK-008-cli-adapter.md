# MLDB-V2-WORK-008: CLI adapter

- **status**: completed
- **date**: 2026-09-13
- **depends_on**: [MLDB-V2-WORK-007]
- **source_refs**: `spec:mldb.v2.cli.operations`, `spec:mldb.v2.cli.selectors_output`, `spec:mldb.v2.api.application_interface`, `spec:mldb.v2.api.query_interface`

## Start gate clarification — 2026-09-13
W007 completion is the W008 concrete-integration/closure gate, not the W008 implementation start gate. Frozen CLI/API Skeletons already fix command/resource spellings, normalized request shapes, output selectors, `CliApplicationAdapter`, and `ApplicationInterface`. W008 may therefore implement parser/request mirrors, rendering, and fake-Application dispatch before W007 completion. Runtime `src` must not import `mldb_v2.skeleton`; adapter import/testing joins only on W007 establishing the exact public `mldb_v2/src/api/application_interface.py` mirror seam.

## Goal
Implement the installed `mldb` command as a thin user-facing adapter over the Application API with discovery-first defaults, shared selectors, deterministic structured output, and safe bulk mutation semantics.

## Boundary
Own `mldb_v2/src/cli/` plus packaging/entrypoint wiring required for the installed command. The CLI never reads repository files/YAML or talks to BackendPort/ClearML directly. W007 owns `mldb_v2/src/api/`.

## Task plan
| task | responsibility | implementation start dependency |
|---|---|---|
| MLDB-V2-TASK-008-01 | Exact CLI public value/request mirrors plus argv parser/normalization. | W001 completed; no W007 completion |
| MLDB-V2-TASK-008-02 | One-shot `CliApplicationAdapter` dispatch over all command families using fake ApplicationInterface tests. | T008-01 + W007 public API mirror seam; not W007 completion |
| MLDB-V2-TASK-008-03 | Deterministic table/wide/json/yaml rendering and exit/error presentation policy. | none beyond frozen contracts |
| MLDB-V2-TASK-008-04 | CLI runtime/main composition and repository-local installed-entrypoint wiring with injected/fake application. | T008-01,T008-02,T008-03; not W007 completion |
| MLDB-V2-TASK-008-05 | Concrete W007 integration, installed smoke, integrated/adversarial verification, W008 closure. | T008-01..04 + W007 completed |

## Completion condition
- `mldb ps`, ID-less `get`, `status`, and `watch` provide discovery without filesystem hunting.
- `validate`/`verify` allow global, Namespace, kind, and exact-target scopes; multi-target mutation requires explicit `--all`.
- `watch` is strictly read-only; `run`/`resume` delegate foreground progression and `advance` remains exactly one pass.
- Structured stdout is undecorated/deterministic and preserves API list/object semantic shape.
- Optional backend logs surface `unsupported_capability`; CLI never works around a W007 contract mismatch.

## Coordinator checkpoint — 2026-09-14
- W007 is completed; concrete integration/closure gate is open.
- T008-01, T008-02, and T008-03 are completed.
- T008-04 is the current implementation critical path; T008-05 follows for concrete W007 integration and W008 closure.
- T008-02 correctly leaves backend-omitted `run` unresolved at adapter level: default-backend selection belongs to CLI runtime configuration, not `CliApplicationAdapter` or W007 private helpers.
- `--fail-fast` has no ApplicationInterface short-circuit seam; the CLI spec marks early stopping as optional (`MAY`), so closure must verify acceptable behavior without changing validation semantics.

## Coordinator checkpoint — 2026-09-14 after T008-04
- T008-01/02/03/04 are completed; T008-05 is the sole remaining W008 closure task.
- W007 is completed, so concrete Application integration is unblocked.
- Actual repo has operational `.env` keys for ClearML and S3-compatible storage, but no declared default-backend selector; T008-05 must not invent one.
- Repository-local `mldb.cmd` smoke passes, but no global Python packaging/console-script mechanism exists.
- `main()` still requires an injected Application factory; T008-05 must either establish the minimal production composition from existing public seams/configuration or record the exact closure blocker without changing W007.

## Closure blocker — 2026-09-14
- T008-05 proved the repository command has no production `ApplicationInterface` construction join: `main()` requires an injected factory and the public API currently exposes only the dependency-heavy concrete `Application` constructor.
- W008 must not repair this by importing BackendRegistry/ClearML/S3/repository internals into `src/cli/**`; the missing seam belongs to W007/public application composition ownership.
- No authoritative configured default-backend selector was found in the current `mldb_v2/src`/runtime configuration surface, so backend-omitted `run` also remains blocked pending that public operational composition/configuration seam.
- W008 stays `active-integration`; T008-05 stays `planned` until the focused W007 repair lands, then closure verification resumes from concrete integration rather than restarting T008-01..04.

## Coordinator checkpoint — 2026-09-14 after W007 composition repair
- W007 production composition repair is verified; T007-04/W007 remain `completed`.
- Public `compose_application()` now returns `ApplicationComposition(application, default_backend)` without exposing backend/storage construction to CLI.
- `MLDB_V2_DEFAULT_BACKEND` is the sole new selector; absence remains explicit `None`, with no implicit `clearml`.
- T008-05 resumed and the CLI-side production join/default-backend resolution is implemented without importing backend/storage/repository internals into `src/cli/**`.
- Verification passed through concrete integration **13**, aggregate W008 **106**, W007 composition regression **29**, outside-cwd installed-command smoke, static/forbidden-import checks, and full `mldb_v2/tests` **1342 passed, 3 skipped**.

## Remaining closure blocker — 2026-09-14 adversarial review
- Concrete planning evidence exposed a second W007-owned composition defect: production `Application` receives `_ObjectByteAccess`, but its `AuthoringPlanningService` constructs `_RepositoryDefinitionVerifier` without `object_access`.
- W002 requires Corpus verification to use injected object-byte access. With an otherwise valid sealed Corpus-backed Study, production `verify_scope` reports `corpus_object_invalid: Corpus object-byte access is not configured`, and `plan_study` maps that to `validation_failed`.
- The first concrete CLI fixture had hidden this by replacing the private verifier; that substitution is not acceptable W008 closure evidence. W008 will not work around the defect in CLI.
- W008 therefore remains `active-integration`; T008-05 remains `planned`. Resume only after a focused W007 repair wires the existing Application object-byte access into authoring/verification, then rerun concrete `plan/run` plus closure verification.

## Closure complete ? 2026-09-14
- The W007 object-byte composition blocker is repaired at the Application boundary: the existing Application-owned `_ObjectByteAccess` now reaches the existing W002 `_RepositoryDefinitionVerifier`; no CLI workaround or verifier-algorithm change was introduced.
- Production `verify` and `plan` both pass on a production-valid sealed Study through `compose_application()` with no verifier monkeypatch. The original one-entry reproduction also advances past the former `object-byte access is not configured` failure into actual exact-object verification.
- The invalid private verifier replacement was removed from concrete CLI integration coverage; real W002 verification, exact Corpus bytes, executable integrity, asset tests, and callable loading are exercised instead.
- W008 concrete integration: **13 passed**. W008 aggregate: **106 passed**. Outside-cwd command smoke and CLI dependency/static checks PASS.
- Full `mldb_v2/tests`: **1343 passed, 3 skipped in 380.30s**, run once after production `verify/plan` and W008 integration passed.
- T008-05 is `completed`; all W008 completion conditions are satisfied. Production real-backend training E2E remains W009 scope.
