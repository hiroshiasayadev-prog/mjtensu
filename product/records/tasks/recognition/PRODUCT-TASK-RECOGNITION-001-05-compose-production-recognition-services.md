# PRODUCT-TASK-RECOGNITION-001-05: Compose production recognition services

- **status**: not_started
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
- Execution results are recorded here when the Task is performed.
