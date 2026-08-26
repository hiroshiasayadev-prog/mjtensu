# PRODUCT-TASK-RECOGNITION-001-05: Compose production recognition services

- **status**: completed
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-RECOGNITION-001
- **task_type**: implementation
- **estimate**: 1.5d
- **depends_on**:
  - PRODUCT-TASK-RECOGNITION-001-01
  - PRODUCT-TASK-RECOGNITION-001-04
- **outputs**:
  - production RecognitionService/RealtimeRecognizer composition
  - PRODUCT-TASK-RECOGNITION-001-05

## Goal

Compose the model runtime and semantic pipeline into the public one-frame/realtime Recognition API with the accepted scheduler, lifecycle, and normalized-error behavior.

## Work

- Implement the public one-frame pipeline boundary from camera frame/semantic region input to live observations plus frame recognition snapshot.
- Implement the realtime recognition controller against the public camera-frame acquisition boundary without embedding UI navigation.
- Request recognition work at the accepted 100 ms cadence while allowing at most one acceptance-owning evaluation at a time.
- Drop/replace stale camera frames rather than accumulating a required inference queue.
- Emit realtime preparation/live/stabilization/commit/failure updates required by the public Recognition API.
- Stop route-owned realtime work on leave while preserving app-lifetime model sessions.
- Normalize inference/runtime failures through the defined runtime-error taxonomy.
- Add focused scheduler/concurrency/lifecycle/public-contract tests with deterministic fake frame/model dependencies.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| one-frame pipeline | Compose detector, classifiers, postprocess, semantic reconstruction, and frame snapshot under the public Recognition boundary. | Public results contain only canonical/product observation and structure types and preserve the stage semantics fixed by the pipeline spec. | Contract tests with deterministic stage fakes plus real-artifact R06. |
| realtime scheduler | Drive recognition requests at the 100 ms target cadence without overlapping acceptance-owning evaluations or a growing stale-frame queue. | Scheduler tests prove one active evaluation maximum and latest/stale-frame behavior under slow inference. | Fake-clock/concurrency tests. |
| lifecycle | Start/stop realtime work independently of app-lifetime model-session ownership. | Stopping Recognition cancels route-owned scheduling but does not dispose healthy shared model sessions. | Lifecycle tests with fake runtime/session counters. |
| error normalization | Surface normalized recognition runtime/inference failures without leaking ORT/browser-internal exceptions into UI-facing contracts. | Known failure injections map to the required `RecognitionRuntimeError` variants. | Error-mapping contract tests. |

## Done condition

The production public Recognition services compose all accepted pipeline stages, scheduler/lifecycle semantics, and normalized errors, with focused contract tests passing before actual-artifact verification.

## Verification

- Run public Recognition API contract tests.
- Run fake-clock scheduler/concurrency tests for 100 ms request cadence, one-active-evaluation, and stale-frame handling.
- Run route-stop/app-session lifecycle tests.
- Run normalized-error mapping tests.
- Run strict typecheck/lint/architecture checks.

## Evidence

- `spec:product.system.contracts.recognition_api` defines the public service boundary.
- `spec:product.recognition.runtime_recognition` defines cadence and commit behavior.
- `spec:product.system.contracts.model_runtime` defines app-lifetime session ownership.
- Production public contracts were added in `product/frontend/src/recognition/contracts.ts` for `RecognitionRuntime`, one-frame `RecognitionPipeline`, `RecognitionFrameSource`, `RealtimeRecognizer`, run/listener/update types, and normalized source geometry.
- `product/frontend/src/recognition/production-pipeline.ts` now composes the initialized detector/base/red-five sessions with fixed-composite preprocessing, detector postprocess/duplicate suppression, crop classification, source-normalized observations, and the R04 semantic snapshot builder. ORT/session objects remain below the Recognition public boundary.
- `product/frontend/src/recognition/production-runtime.ts` now implements the model-runtime contract's app-lifetime `initialize() -> createPipeline() -> dispose()` composition. Pipeline disposal waits route-owned in-flight work but does not dispose shared model sessions; runtime disposal closes tracked pipelines before releasing the R01 model runtime.
- `product/frontend/src/recognition/realtime-recognizer.ts` now requests immediately and then at the accepted `100 ms` cadence, enforces recognizer-wide single-flight evaluation, avoids capturing/queueing frames while inference is active, drops stale results across stop/reset/run replacement, emits `scanning`/`stabilizing`/single `confirmed` updates, materializes fresh `TileInstanceId` values only at confirmation, and reports normalized runtime failures through `onError`.
- `product/frontend/src/recognition/index.ts` exposes the public R05 runtime/realtime contracts and factories without exporting ONNX Runtime session types.
- Focused deterministic coverage was authored in `product/frontend/test/recognition-services.test.ts` for one-frame stage composition, normalized preview geometry, conditional red-five invocation, detector/base/red-five failure normalization, incompatible outputs, runtime/pipeline lifecycle isolation, immediate/100 ms cadence, slow-inference backpressure, run replacement, ineligible scanning, three-result confirmation, reset/stop stale-result suppression, and fatal error delivery.
- Classifier normalization is no longer a public Recognition runtime/factory input. Production composition resolves classifier preprocessing ownership from each initialized model's `runtimeSpec`; `product/frontend/src/recognition/model-runtime/runtime-specs.ts` is the code-owned location for the exact normalization contract.
- The exact classifier normalization values were materialized on 2026-08-27 from the selected local compact training databases using `tools/recognition/materialize_classifier_runtime_normalization.py`, which mirrors the training-time statistics definitions. `c8-tile-35-v1` is fixed to mean `[0.68306223733377514]`, std `[0.27237886485683077]`; `c8-red-five-v1` is fixed to mean `[0.66025093606229934, 0.69172744263865471, 0.6489080530422624]`, std `[0.30491469480493394, 0.24924454491506576, 0.27107025824445752]`. Production callers cannot override these values.
- Focused R05 verification was executed on 2026-08-27 from `product/frontend`:
  - `npm test -- recognition-services.test.ts recognition-semantics.test.ts recognition-stabilization.test.ts recognition-model-runtime.test.ts recognition-c8-classifier.test.ts recognition-detection-postprocessor.test.ts` -> PASS: 6 test files, 50 tests.
  - `npm run typecheck` -> PASS with no TypeScript diagnostics.
  - `npm run lint` -> PASS: `Architecture import boundaries: OK (51 source files checked).`
- The R05 implementation/focused verification gate is therefore PASS. R01 was rechecked and closed as `completed` on 2026-08-27 after its model-runtime tests, typecheck, architecture lint, and corrected SYSTEM bootstrap findings were confirmed PASS.
- `tools/recognition/materialize_classifier_runtime_normalization.py` deterministically derives the base C8 values from `gray35_jp500_seed42_v2.sqlite` and the warm-augmented RGB red-five values from `rgb64_binary_jp5000_seed42.sqlite` using the same training-time statistics definitions, then materializes them into `src/recognition/model-runtime/runtime-specs.ts`; this materialization completed successfully on 2026-08-27.
- `recognition-services.test.ts` now exercises the production runtime-spec normalization path without the private override and asserts the resulting base/red-five tensor normalization values.
- Post-materialization verification was rerun on 2026-08-27 from `product/frontend` and passed completely:
  - `npm test -- recognition-services.test.ts recognition-semantics.test.ts recognition-stabilization.test.ts recognition-model-runtime.test.ts recognition-c8-classifier.test.ts recognition-detection-postprocessor.test.ts` -> PASS: 6 test files, 50 tests.
  - `npm run typecheck` -> PASS with no TypeScript diagnostics.
  - `npm run lint` -> PASS: `Architecture import boundaries: OK (51 source files checked).`
- All declared R05 dependencies are completed, the code-owned classifier normalization contract is materialized, and the focused post-materialization gate is PASS. R05 is complete; actual production ONNX artifact loading/parity remains the separate R06 verification scope.
