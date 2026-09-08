# MLDB v1 Spec Conformance Audit

Date: 2026-09-09
Scope: `mldb/records/spec/**/*.md` (49 specs), `mldb/skeleton`, `mldb/src`, `mldb/tests`, and the current real SSH experiment workflow under `tools/mldb` / `mldb_data`.

This is an audit only. Existing implementation/spec/skeleton files were not modified by this audit.

## Verdict meanings

- **IMPLEMENTED**: the current implementation and tests materially satisfy the spec contract inspected.
- **PARTIAL**: substantial implementation exists, but the current usable system does not satisfy the whole contract/boundary.
- **VIOLATED**: current skeleton/source/registered asset/workflow contradicts an explicit spec rule.
- **MISSING-TEST**: implementation appears present but a normative behavior lacks meaningful verification.
- **SPEC-GAP**: the intended MLDB capability cannot be implemented from the current specs without inventing a contract.

A passing skeleton signature guard is not sufficient evidence of spec conformance. It only checks `skeleton -> src` public-surface equality. It does not prove that the skeleton itself matches the specs.

## Executive result

The MLDB core is not "nothing": Study planning, Run persistence/lifecycle, Model identity, Evaluation result acceptance, Queue/reconciliation, Worker API values, Controller operations, and executable sealing machinery are substantially implemented and tested.

The audited contracts are now implemented across the catalog/runtime/orchestration path. The remaining acceptance item is execution evidence for a second materially different model family (rotated detection) through the same generic SSH Study runner; classifier evidence is complete.
## Conformance matrix

### Root / API / Catalog

| Spec | Skeleton | Source | Tests | Verdict | Evidence / reason |
|---|---|---|---|---|---|
| `spec:mldb` | root | broad surfaces exist | full suite + three real E2E acceptances | **IMPLEMENTED** | Generic classifier, full-feature classifier v3, and rotated-detector Studies all completed through the same generic SSH Study runner with sealed immutable assets. |
| `spec:mldb.api` | controller surface | controller exists | `test_api_controller.py` | **IMPLEMENTED** | Public API scope correctly excludes HTTP and direct Training/Evaluation launches. |
| `spec:mldb.api.controller` | 7 operations | 7 operations delegate to lower domain/runtime services | `test_api_controller.py` | **IMPLEMENTED** | `validate_definition`, `seal_definition`, `execute_study`, progress, cancel, get/list entity all exist. |
| `spec:mldb.catalog` | generic catalog surfaces | generic catalog surfaces | catalog/runtime tests | **IMPLEMENTED** | Generic Task/Corpus boundaries are preserved and both categorical classification and rotated detection have concrete contracts. |
| `spec:mldb.catalog.task_format` | open Task target + concrete typed targets | same | `test_catalog_task.py` | **IMPLEMENTED** | Categorical validation is conditional; rotated detection is explicit; unknown future non-categorical mappings remain open rather than globally rejected. |
| `spec:mldb.catalog.corpus_format` | generic outer metadata | generic outer metadata | `test_catalog_corpus.py`, `test_runtime_i2a.py` | **IMPLEMENTED** | Outer `corpus/v1`, SQLite identity/hash, `data.schema`, table, representation, builder params and splits are implemented. |
| `spec:mldb.catalog.image_classification_corpus` | concrete-schema behavior documented in resolver | concrete SQLite validation | `test_runtime_i2a.py` | **IMPLEMENTED** | Core columns, target/class-index agreement, split inventory and uint8 BLOB shape checks exist. |
| `spec:mldb.catalog.rotated_object_detection_task` | concrete typed target | same | `test_catalog_task.py` | **IMPLEMENTED** | Labels plus `cx-cy-w-h-angle-deg` geometry and 180-degree period are validated without closing the generic target boundary. |
| `spec:mldb.catalog.rotated_object_detection_corpus` | concrete-schema contract | SQLite schema validator | `test_runtime_i2a.py` + real Corpus resolution | **IMPLEMENTED** | Self-contained RGB BLOB samples, JSON OBB annotations, labels, geometry, split counts and payload sizes are validated. |
| `spec:mldb.catalog.architecture_format` | generic interface mappings | generic interface mappings | `test_catalog_architecture.py`, runtime verification tests | **IMPLEMENTED** | No classifier-only forward type is imposed; arbitrary non-empty `interface.output.kind` is accepted. |
| `spec:mldb.catalog.architecture_build` | `build() -> nn.Module` | loader + model/training consumers | `test_runtime_i2a.py`, worker/model tests | **IMPLEMENTED** | Zero-argument build boundary and fresh Architecture construction are present without model-family branching. |
### Runtime / Training / Model

| Spec | Skeleton | Source | Tests | Verdict | Evidence / reason |
|---|---|---|---|---|---|
| `spec:mldb.runtime` | runtime surfaces | runtime surfaces | runtime/domain tests + classifier E2E | **IMPLEMENTED** | Runtime resolution/loading/materialization is model-family neutral and the SSH path consumes assigned immutable assets only. |
| `spec:mldb.runtime.component_model` | resolver/loader/executor/materializer separation | same logical boundaries | runtime/worker/study tests | **IMPLEMENTED** | Practical SSH execution no longer injects undeclared project support files. |
| `spec:mldb.runtime.asset_resolution` | generic typed resolution | generic typed resolution + data-schema dispatch | `test_runtime_i2a.py` | **IMPLEMENTED** | Task resolution preserves non-categorical structures; Corpus validation dispatches by concrete `data.schema`. |
| `spec:mldb.runtime.public_parameters` | common parameter model | `common.parameters` resolution | `test_common_parameters.py`, Study/preflight tests | **IMPLEMENTED** | Exact published keys, defaults, override rejection and JSON-compatible value preservation are covered. |
| `spec:mldb.training` | training surfaces | training surfaces | training/runtime/asset tests | **IMPLEMENTED** | Current classifier and rotated-detector Train Protocols are self-contained sealed executable assets. |
| `spec:mldb.training.train_protocol_format` | generic protocol | generic validator/loader | verification + `mldb_tests/train_protocols/*` | **IMPLEMENTED** | Active Train Protocols keep result-affecting project logic inside their hashed sibling `.py` and were sealed through the public pytest gate. |
| `spec:mldb.training.train_interface` | `TrainContext`, module return | `TrainContext`, worker execution, state acceptance | `test_training_core.py`, `test_orchestration_worker_execution.py` | **IMPLEMENTED** | Resolved inputs/seed/parameters/work dir are passed and returned module is accepted against a fresh Architecture before serialization. |
| `spec:mldb.training.training_run_format` | run dataclasses | validation + persistence | `test_training_core.py`, `test_runtime_run_persistence.py` | **IMPLEMENTED** | Complete resolved inputs, seed, timestamps, lineage and weight result metadata are persisted. |
| `spec:mldb.training.training_run_lifecycle` | state/transition surfaces | transition/finalization/orchestration | training + dispatch/outcome/recovery tests | **IMPLEMENTED** | Preflight-before-allocation, immutable terminal attempts, retry-new-ID and Model-on-success semantics are implemented. |
| `spec:mldb.training.canonical_weights` | weight boundary | strict state acceptance, CPU tensor state, direct `torch.save`, hash/bytes | `test_training_core.py`, model/worker tests | **IMPLEMENTED** | Canonical learned bytes remain Training-Run-owned and are integrity tracked. |
| `spec:mldb.model` | model surfaces | identity/loading/persistence | runtime/model-related tests | **IMPLEMENTED** | Minimal Model identity over completed Training Run is present. |
| `spec:mldb.model.identity` | deterministic mapping | deterministic mapping + ensure | training/recovery/runtime tests | **IMPLEMENTED** | `tr-... -> mdl-...` one-to-one creation and completed-run requirement are enforced. |
| `spec:mldb.model.model_format` | minimal 3-field record | strict normalizer/validator | runtime/model tests | **IMPLEMENTED** | Extra Model fields are rejected; weights are resolved through Training Run rather than duplicated. |
### Evaluation

| Spec | Skeleton | Source | Tests | Verdict | Evidence / reason |
|---|---|---|---|---|---|
| `spec:mldb.evaluation` | evaluation surfaces | evaluation surfaces | evaluation/runtime/asset tests | **IMPLEMENTED** | Current classifier and rotated-detector Evaluation Protocols are self-contained sealed executable assets. |
| `spec:mldb.evaluation.evaluate_interface` | `EvaluationContext/Result` | same + Worker execution | `test_evaluation_core.py`, `test_orchestration_worker_execution.py` | **IMPLEMENTED** | Generic runtime does not implement classifier/detector algorithms; protocol owns them. |
| `spec:mldb.evaluation.evaluation_protocol_format` | generic parameters/metrics/artifacts | generic validator | verification + `mldb_tests/evaluation_protocols/*` | **IMPLEMENTED** | Active Evaluation Protocols contain result-affecting project logic in their hashed sibling `.py` and were sealed through the public gate. |
| `spec:mldb.evaluation.evaluation_run_format` | run surfaces | validation/persistence | `test_evaluation_core.py`, `test_runtime_run_persistence.py` | **IMPLEMENTED** | Exact Model/Corpus/Protocol/parameters, lineage, metrics/artifacts and terminal facts are represented. |
| `spec:mldb.evaluation.evaluation_run_lifecycle` | lifecycle surfaces | transitions + result acceptance/finalization | evaluation/outcome/reconciliation tests | **IMPLEMENTED** | `completed`, `completed_partial`, `failed`, `cancelled`, retry and failure isolation are covered. |
| `spec:mldb.evaluation.result_validation` | acceptance boundary | scalar + artifact validation/import | `test_evaluation_core.py` | **IMPLEMENTED** | Finite scalar rules, unavailable outputs, required/optional artifacts, partial status and immutable import are implemented. |
| `spec:mldb.evaluation.artifacts` | registered artifact schemas | dispatcher in result validation | `test_evaluation_core.py` | **IMPLEMENTED** | The two artifact schemas actually defined by v1 are registered. |
| `spec:mldb.evaluation.artifacts.categorical_predictions` | schema-specific validation surface | JSONL content validator | `test_evaluation_core.py` | **IMPLEMENTED** | sample/target/prediction and Task-label validation exist. |
| `spec:mldb.evaluation.artifacts.confusion_matrix` | schema-specific validation surface | CSV content validator | `test_evaluation_core.py` | **IMPLEMENTED** | target/prediction/count schema and categorical label validation exist. |
### Study

| Spec | Skeleton | Source | Tests | Verdict | Evidence / reason |
|---|---|---|---|---|---|
| `spec:mldb.study` | Study/model-source surfaces | full Study materialization/orchestration support | Study + orchestration tests | **IMPLEMENTED** | Both training-grid and existing-Model source modes are represented; no direct-run alternative is invented. |
| `spec:mldb.study.study_format` | definition surfaces | static validation + preflight | `test_study_core.py`, `test_study_preflight.py` | **IMPLEMENTED** | Exclusive source, uniqueness, public parameter axes, fixed eval stages and sealed execution preflight are covered. |
| `spec:mldb.study.grid_expansion` | expansion surface | deterministic Cartesian expansion + existing-model order | `test_study_core.py` | **IMPLEMENTED** | Defaults are resolved into complete coordinates; ordering/trial numbering are tested. |
| `spec:mldb.study.plan_format` | plan value/validation | JSONL plan persistence/validation | `test_study_core.py`, persistence/launch tests | **IMPLEMENTED** | Complete resolved parameters, immutable intent, no Queue/child Run identity in rows. |
| `spec:mldb.study.study_run_format` | run surface | validation/persistence | Study core/progress/persistence tests | **IMPLEMENTED** | Plan hash/bytes/trial/evaluation counts and terminal requirements exist. |
| `spec:mldb.study.study_run_lifecycle` | lifecycle surface | launch/reconcile/cancel/progress | extensive orchestration tests | **IMPLEMENTED** | Retry lineage, blocked evaluations, completed-with-failures, resume/reconcile and terminality are present. |
### Repository / Orchestration / Verification

| Spec | Skeleton | Source | Tests | Verdict | Evidence / reason |
|---|---|---|---|---|---|
| `spec:mldb.repository` | layout surface | repository layout/filesystem | `test_repository_primitive.py` | **IMPLEMENTED** | Canonical `mldb_data`, asset test root and non-canonical Queue separation exist. |
| `spec:mldb.repository.layout` | complete path API | complete path API | `test_repository_primitive.py` + signature guard | **IMPLEMENTED** | Entity-specific locations, Run dirs, weights/plan/test paths and Queue separation are implemented. |
| `spec:mldb.orchestration` | orchestration surfaces | Queue/Controller/Worker core + generic SSH bridge | orchestration suite + classifier E2E | **IMPLEMENTED** | Generic Study execution uses Controller/Queue/Worker boundaries without model-family IDs in the runner. |
| `spec:mldb.orchestration.responsibility_model` | Controller/Queue/Worker surfaces | same + generic SSH bridge | orchestration tests + worker cache path | **IMPLEMENTED** | Worker consumes only Controller-selected immutable descriptors; hard-coded `SUPPORT_FILES` injection was removed. |
| `spec:mldb.orchestration.job_model` | Training/Evaluation job values | derivation + dependencies | `test_orchestration_queue_contracts.py` | **IMPLEMENTED** | Training-derived and existing-Model coordinates/dependencies are represented separately from Run attempts. |
| `spec:mldb.orchestration.queue_lifecycle` | QueuePort/lifecycle surfaces | SQLite Queue + recovery/reconciliation/outcome | queue/recovery/reconciliation/restart tests | **IMPLEMENTED** | blocked/ready/active/retry_wait/satisfied/failed/cancelled and canonical-history reconciliation are extensively covered. |
| `spec:mldb.orchestration.queue_storage_format` | QueuePort contract | `_sqlite_queue.py` | `test_orchestration_sqlite_queue.py`, queue contracts | **IMPLEMENTED** | Controller-local SQLite jobs/attempts, uniqueness, leases, timestamps and transaction behavior are implemented. |
| `spec:mldb.orchestration.worker_api` | full logical Worker API | handlers + generic `worker_loop.py` + SSH bridge | worker contracts/data/loop tests + classifier E2E | **IMPLEMENTED** | SSH materialization verifies descriptor SHA/bytes and reuses the SHA-addressed remote cache without extra domain files. |
| `spec:mldb.verification` | verification/sealing surfaces | verification lifecycle + public tooling | `test_verification_i2e.py` + real asset sealing | **IMPLEMENTED** | Current active classifier and detector executable assets were sealed through `seal_definition` after canonical pytest gates. |
| `spec:mldb.verification.executable_asset_tests` | test-dir/verifier/seal gates | fully implemented sealing gate | `test_verification_i2e.py` + `mldb_tests/` | **IMPLEMENTED** | Canonical asset-test directories exist for active Architecture/Train/Evaluation assets and each collected/passed before sealing. |
## Matrix count

All 49 spec files are represented exactly once in the matrix.

- **IMPLEMENTED: 49**
- **PARTIAL: 0**
- **VIOLATED: 0**
- **MISSING-TEST: 0** as a primary row verdict.
- **SPEC-GAP: 0** as a primary existing-spec verdict, because a missing contract is by definition not one of the 47 existing rows. Confirmed gaps are listed separately below.

The count must not be read as "32/47 done". A small number of red boundary contracts can make the whole system unusable for a model family even when many lifecycle/storage contracts are green.

## Resolved spec gaps

The two blocking catalog gaps found on 2026-09-08 are now explicit contracts:

- `spec:mldb.catalog.rotated_object_detection_task` defines the rotated-detection target structure.
- `spec:mldb.catalog.rotated_object_detection_corpus` defines the self-contained SQLite RGB/OBB physical schema.

The generic Task boundary remains open for future non-categorical target structures not yet standardized.

Detector structured prediction artifacts remain optional: current detector Evaluation Protocols can return formal scalar metrics under the existing generic evaluation contract. A future detector-prediction artifact schema is only needed if those structured predictions must themselves become validated formal artifacts.
## Repaired real-workflow findings

The seven system-level violations identified on 2026-09-08 have been repaired:

1. Task target handling is no longer globally categorical-fixed.
2. Corpus validation dispatches by concrete `data.schema` and includes rotated detection.
3. Active classifier and detector Train Protocols are self-contained hashed siblings.
4. Active classifier and detector Evaluation Protocols are self-contained hashed siblings.
5. `mldb_tests/` contains canonical executable-asset tests and active assets were sealed through the public gate.
6. `run_ssh_worker.py` no longer injects hard-coded recognition support files outside assignment descriptors.
7. The tile-specific Study runner was removed in favor of `run_study_ssh.py --study-id ...`.

The invalid 2026-09-08 smoke history was quarantined under `.local/mldb-invalid-history-20260908`; it is not part of the current canonical catalog. The successful compliant classifier Study is `sr-20260909-001`.
## Generic Study execution entrypoint

The generic launcher is now the MLDB operating surface.

The supported shape is:

```powershell
.\.venv\Scripts\python.exe tools\mldb\run_study_ssh.py `
  --host 192.168.11.22 `
  --study-id rotated-fcos-spatial-screen-v1
```

`run_study_ssh.py` now:

- know no Task, Corpus, Architecture, Protocol, or model-family ID;
- not import a domain bootstrap module;
- not create or mutate reusable definitions as a side effect of Study execution;
- call the existing public `execute_study` boundary for the exact authored Study ID;
- drive generic Queue/Worker execution until the Study Run becomes terminal;
- let normal validation/preflight reject missing, draft, incompatible, or corrupt inputs;
- use only Controller-selected immutable asset descriptors for Worker execution bytes;
- preserve SHA-addressed Worker cache reuse for those descriptors.

A Study YAML should be the experiment-specific execution input. A new Study must not require a new Python runner.
## Signature-guard finding

Current `test_signature_guard.py` intentionally treats a skeleton module with no corresponding implementation module as non-failure.

At audit time there are **zero skeleton-only Python modules**, so this hole is not presently hiding an entire missing module. It is still a weak completion criterion and must not be used as evidence that all frozen skeleton work is implemented.

More importantly, the guard compares skeleton to source; it cannot detect a skeleton that already diverged from spec. `catalog/task.py` demonstrates exactly that failure mode: skeleton and source agree with each other while both narrow the Task target contract.

## Repair status

The repair sequence is complete through implementation and classifier acceptance:

1. Rotated-detection Task and Corpus contracts added.
2. Skeleton/source Task genericity repaired and covered by tests.
3. Runtime Corpus `data.schema` dispatch repaired with classification and rotated-detection fixtures.
4. Self-contained classifier v2 and rotated-FCOS executable assets created.
5. Canonical `mldb_tests/` created and all active executable assets sealed through the public Controller gate.
6. Generic `run_study_ssh.py --study-id ...` established; tile-specific runner removed; Worker support-file injection removed.
7. Classifier acceptance completed as `sr-20260909-001` through the generic runner.
8. Rotated-detector acceptance completed as `sr-20260909-002` through the same generic runner; `ev-20260909-002` was accepted and the Study reached `completed`.
9. Full-feature classifier parity acceptance completed as `sr-20260909-003`; its immutable plan records nonzero rotation/perspective/shear/stretch/projective augmentation plus cache controls, and `ev-20260909-003` was accepted.

## Manual/tooling status

`mldb/AGENTS.md` must describe only the repaired workflow: authored Study YAML, public sealing for executable definitions, and generic `run_study_ssh.py --study-id ...`. It must not direct agents to the removed tile-specific runner or the quarantined v1 executable definitions.
## Test-suite observation

Final local verification after the repair:

```text
533 passed, 9 warnings, 174 subtests passed
```

The suite includes generic Task regression coverage, rotated Task/Corpus resolution, all active executable-asset tests, and the v3 full classifier parameter surface.

## Real GPU acceptance evidence

All acceptance paths use the same generic `tools/mldb/run_study_ssh.py --study-id ...` entrypoint:

- `sr-20260909-001`: compliant classifier v2 Study, `completed`.
- `sr-20260909-002`: rotated FCOS detector Study, `completed`; `ev-20260909-002` accepted.
- `sr-20260909-003`: full-feature classifier v3 Study, `completed`; its immutable plan records nonzero rotation, perspective, shear, stretch, projective augmentation and cache controls, and `ev-20260909-003` was accepted.

The classifier and detector paths therefore exercise materially different Task/Corpus/model families through the same Controller/Queue/Worker execution flow. No remaining spec row is partial or violated.
