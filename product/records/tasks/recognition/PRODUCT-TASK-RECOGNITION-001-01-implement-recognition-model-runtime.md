# PRODUCT-TASK-RECOGNITION-001-01: Implement recognition model runtime

- **status**: in_progress
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-RECOGNITION-001
- **task_type**: implementation
- **estimate**: 1.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **outputs**:
  - production recognition model-runtime implementation
  - PRODUCT-TASK-RECOGNITION-001-01

## Goal

Implement production recognition model asset acquisition, manifest validation, ONNX session initialization, provider fallback, retry, and app-lifetime session ownership for all three recognition roles.

## Work

- Implement the model-set manifest/runtime-spec dispatch boundary.
- Implement asset prefetch/deduplication without constructing sessions during prefetch.
- Validate model integrity/version/runtime-spec inputs required by the model-runtime contract.
- Construct one app-lifetime ONNX session per model role on initialization.
- Apply the configured per-model provider preference/fallback sequence.
- Preserve successful initialization idempotence and concurrent-call deduplication.
- Make later initialization retryable after a failed attempt without permanently caching a rejection.
- Keep ONNX Runtime types private to Recognition infrastructure.
- Add focused tests for manifest validation, prefetch/init deduplication, provider fallback, failure retry, and disposal/lifecycle semantics.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| model asset layer | Implement build-pinned model-set prefetch/integrity handling for detector, tile-classifier, and red-five-classifier roles. | Concurrent prefetch requests deduplicate; prefetch does not create inference sessions; invalid/incompatible manifests fail with normalized runtime errors. | Vitest contract tests with deterministic asset/runtime fakes. |
| RecognitionRuntime initialization | Construct and retain one production session per required model role using known runtime specs. | Successful `initialize()` is idempotent and concurrent calls share one in-flight initialization. | Focused runtime tests. |
| retry behavior | Permit a later initialization attempt after an earlier failure. | A rejected initialization attempt is not permanently cached; a subsequent healthy attempt can become ready. | Failure-then-success test. |
| provider fallback | Use the accepted role preference order `wasm-simd -> wasm-threaded -> webgl` and normalize unavailable-provider failure. | Each fallback branch is deterministic and the selected provider is inspectable for diagnostics/verification without leaking ORT objects publicly. | Provider-fake matrix tests plus later real-artifact verification in R06. |
| lifecycle isolation | Keep sessions app-lifetime and independent of Recognition route leave; dispose only at owning runtime/application lifecycle end. | Route-level stop does not recreate or dispose healthy shared sessions. | Lifecycle-focused unit/contract tests. |

## Done condition

The three-role production model runtime satisfies the model-runtime contract, includes focused automated coverage for all initialization/provider/lifecycle branches, and exposes no ONNX Runtime type through public Recognition contracts.

## Verification

- Run focused model-runtime Vitest tests.
- Run strict typecheck and architecture/static checks for the Recognition module.
- Record configured provider preference and each covered fallback branch.
- Defer actual production ONNX artifact loading to PRODUCT-TASK-RECOGNITION-001-06.

## Evidence

- `spec:product.system.contracts.model_runtime` is the implementation authority.
- `spec:product.system.contracts.runtime_errors` defines normalized recognition-runtime failures.
- `spec:product.system.contracts.testing_strategy` requires focused lifecycle/provider tests plus a later actual-artifact gate.
- Implementation added under `product/frontend/src/recognition/model-runtime/`: runtime-spec dispatch/manifest validation, content-addressed Cache API asset acquisition with SHA-256 verification and in-flight deduplication, app-lifetime three-role session ownership, provider fallback diagnostics, failed-initialization cleanup/retry, and private ONNX Runtime Web session adaptation.
- `product/frontend/package.json` now declares `onnxruntime-web` `1.27.0`, matching the already-used recognition PWA runtime tooling rather than changing runtime-library behavior during this Task.
- Focused coverage is added in `product/frontend/test/recognition-model-runtime.test.ts` for manifest incompatibility, prefetch/runtime acquisition deduplication, integrity and unavailable-asset normalization, initialization idempotence/concurrency, per-role provider fallback, provider exhaustion, failure-then-success retry, public ORT-type isolation, and runtime-owned disposal.
- Actual production ONNX artifact loading remains deferred to PRODUCT-TASK-RECOGNITION-001-06 as required.
- Verification is pending local dependency installation/lockfile update and execution of the declared Vitest/typecheck/lint/build commands; do not mark this Task `done` until those results are recorded.
- The recorded SYSTEM T05 review verdict is still `NEEDS REVISION`; T06 and T07 record both named corrections as `done`, but the bootstrap review gate must be re-established as PASS before R01 completion is treated as dependency-clean.
