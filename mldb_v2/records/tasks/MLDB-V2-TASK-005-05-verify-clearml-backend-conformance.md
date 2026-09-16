# MLDB-V2-TASK-005-05: Verify ClearML backend conformance

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-005
- **task_type**: verification
- **depends_on**: [MLDB-V2-TASK-005-03, MLDB-V2-TASK-005-04]
- **outputs**: mocked BackendPort conformance plus configured real-ClearML verification evidence

## Start gate clarification — 2026-09-13
T005-01 through T005-04 and W004 are completed, so T005-05 may start immediately. This Task owns the first full ClearML BackendPort composition and production SDK activation seam. Current active Python environment has no `clearml` package and no `CLEARML*` environment variables; do not invent credentials. If repository-local ignored configuration or an explicit configured service is available, use it without printing/persisting secrets; otherwise record the exact real-integration blocker while completing all mocked conformance work.

## Goal
Close W005 by proving idempotent admission/recovery/cancellation and frozen backend-value conformance, including a real ClearML path when runtime configuration is available.

## Work
- Compose the completed T005-02 admission, T005-03 observation/collection, and T005-04 cancellation/log capabilities behind one concrete ClearML BackendPort implementation without changing the frozen generic port.
- Exercise the complete generic BackendPort through mocked ClearML SDK boundaries, including ambiguous admission response, process restart recovery, retries, terminal candidates, cancellation, and optional logs.
- Verify Namespace -> `mldb/<namespace>`, Study/Plan identity as metadata, opaque backend IDs, exact StageKey echo, and no human Task-name parsing.
- Add the bounded production ClearML SDK activation/configuration seam; when endpoint/credentials/queue configuration is available, run one real adapter conformance path using the common W004 harness. Do not hard-code or persist secrets.
- Verify endpoint/credentials remain non-canonical and queue/retry/heartbeat/GPU scheduling remain ClearML responsibilities.
- Do not create model-family runners, bypass result acceptance, alter W004 runtime, or implement Study progression.

## Done condition
W005 satisfies the frozen BackendPort/ClearML mapping contracts under mocked coverage and, when configured, one bounded real admission/observation/cancellation path using the generic harness.

## Verification
Run focused W005 tests first, then W005 closure regression including relevant W004 integration and broad `mldb_v2/tests`; record unavailable real-ClearML configuration explicitly rather than inventing credentials. No commit/stage/push.

## Completion Evidence — 2026-09-13
- Added concrete `mldb_v2.src.backend.clearml_backend.ClearMLBackend`, composing `ClearMLAdmissionService`, `ClearMLObservationService`, `ClearMLCancellationService`, and optional `ClearMLLogService`; no T005-02/03/04 logic is copied into the composition layer.
- Added production activation seam `mldb_v2.src.backend._clearml_sdk.ClearMLSDKAdapter` with lazy ClearML SDK import. Generic module import remains valid without the `clearml` package installed.
- Added explicit registry helper `register_clearml_backend(registry)`, registering generic backend type `clearml`; no global import-side-effect registration was introduced.
- BackendPort conformance is preserved exactly for `admit(stage_input=...)`, `observe(stage_key=...)`, `collect(stage_key=...)`, and `cancel_study(study_result=...)`. Optional logs remain outside the required port surface.
- Mocked W005 focused verification: **85 passed**. This covers first/repeated admission, ambiguous-create recovery, restart recovery, duplicate ownership rejection, active/terminal observation, retries, training/evaluation completed/failed/cancelled collection, cancellation, optional logs, registry resolution, opaque IDs, and genericity boundaries.
- W004 + W005 relevant integration verification: **303 passed, 2 skipped**.
- Full `python -m pytest mldb_v2/tests -q`: **1227 passed, 3 skipped**.
- Production SDK activation used ClearML SDK **2.1.12** installed only under ignored `.local/t00505-clearml-sdk`; the project `.venv`, global Python, and repository dependency files were not modified.
- Repository `.env` supplied all five ClearML connection variables. Secret values were never printed, recorded, or persisted into tracked files; direct credential-leak scan over owned implementation/test files passed.
- Authenticated real ClearML API query: **PASS**.
- Real bounded backend smoke: Namespace -> `mldb/<namespace>` **PASS**; first admission **PASS**; repeated admission/no duplicate **PASS**; ownership recovery **PASS**; active observe **PASS**; optional logs capability check **PASS (no retained log payload)**; cancellation **PASS**; cancelled candidate collection **PASS**.
- Real SDK exposed configuration objects as `collections.OrderedDict`; T005-05 repaired only `_clearml_sdk._configuration()` to accept the declared `Mapping` shape and copy to `dict`. Mock coverage now exercises this real SDK return shape.
- Live ClearML queue inspection found one queue, `default`, and **0 registered workers**. Therefore a real remote `CommonExecutionHarness` attempt could not be consumed by an agent; launch mapping remains covered by mocked production-adapter tests. This is an operational agent-availability limitation, not a W005 BackendPort conformance blocker.
- Adversarial closure review found one active-state mismatch: ClearML `publishing` was classified active but cancellation only requested stop for `in_progress`. `_clearml_sdk.py` now treats both `in_progress` and `publishing` as stoppable active work; dedicated regression coverage passes.
- Genericity scan: PASS. Runtime modules import no Skeleton, Study driver, result acceptance, or model-family-specific implementation; task names and opaque backend IDs are not parsed for MLDB identity.
- `py_compile` and SDK-free import smoke: PASS.
- Smoke-created queued work was stopped after verification; no live test work was left active.
- No T006/W007 implementation file was modified. No commit/add/stash/reset/clean/restore/push was performed.
- T005-05 completion criteria are satisfied. W005 backend/ClearML completion criteria are satisfied; no remaining W005 code blocker is known. A ClearML agent is still required only for actual remote harness execution/first remote experiment.
