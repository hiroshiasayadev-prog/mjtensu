# PRODUCT-TASK-RECOGNITION-001-01: Implement recognition model runtime

- **status**: not_started
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
- Execution results are recorded here when the Task is performed.
