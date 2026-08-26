# Contract: Recognition API

- **id**: `spec:product.system.contracts.recognition_api`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Public implementation contract separating one-frame recognition from realtime scheduling/stabilization.
The Recognition page consumes the realtime facade rather than directly orchestrating detector/classifier/pipeline stages.

## Recognition frame input

```ts
export interface RecognitionFrame {
  readonly source: CanvasImageSource;
  readonly sourceSize: Size;
  readonly regions: Readonly<Record<RecognitionRegion, NormalizedRect>>;
  readonly capturedAtMs: number;
}
```

Requirements:

- `source` is one immutable logical camera frame for the duration of `evaluate`;
- `sourceSize` describes that frame's source coordinate space;
- `regions` are exactly the visible semantic Recognition-page regions for that frame;
- `regions` and returned observation boxes use `spec:product.system.concepts.coordinate_system`;
- no `320 x 320` detector-composite coordinates cross this boundary.

## One-frame pipeline

```ts
export interface RecognitionPipeline {
  evaluate(frame: RecognitionFrame): Promise<FrameRecognitionSnapshot>;
  dispose(): Promise<void>;
}
```

`RecognitionPipeline.evaluate` owns exactly one frame's semantic evaluation:

```text
visible semantic regions
  -> fixed detector composite
  -> detector
  -> duplicate suppression
  -> candidate crops
  -> 35-class base classification
  -> red-five refinement
  -> observations
  -> hand/dora ordering
  -> meld grouping/reconstruction
  -> FrameRecognitionSnapshot
```

The one-frame pipeline does not own:

- cadence/scheduling;
- previous-frame history;
- consecutive-result counting;
- automatic confirmation;
- Application session state;
- score validity.

Calling `evaluate` twice with separate frames does not itself establish temporal continuity.

## Frame source boundary

Realtime recognition requests the latest frame through a minimal source contract:

```ts
export interface RecognitionFrameSource {
  captureLatest(): RecognitionFrame | null;
}
```

The camera module provides this boundary through the adapter described by `spec:product.system.contracts.camera_api`. Recognition does not own browser camera permission/navigation behavior merely because it consumes frames.

`null` means that no usable latest frame is currently available; it is not a recognition runtime error.

## Realtime updates

```ts
export type RealtimeRecognitionUpdate =
  | {
      readonly kind: 'scanning';
      readonly snapshot: FrameRecognitionSnapshot;
    }
  | {
      readonly kind: 'stabilizing';
      readonly snapshot: FrameRecognitionSnapshot;
    }
  | {
      readonly kind: 'confirmed';
      readonly result: RecognizedStructure;
    };
```

`scanning` and `stabilizing` expose the latest snapshot so the UI can render live candidate boxes, tile icons, unresolved candidates, and meld-group connectors without invoking recognition internals.

The consecutive stabilization count is intentionally not part of the public update contract.

`confirmed` exposes the committed `RecognizedStructure`. Live frame observations are not carried into Application as scoring state.

## Listener and run lifetime

```ts
export interface RealtimeRecognitionListener {
  onUpdate(update: RealtimeRecognitionUpdate): void;
  onError(error: RecognitionRuntimeError): void;
}

export interface RecognitionRun {
  stop(): void;
}

export interface RealtimeRecognizer {
  start(
    source: RecognitionFrameSource,
    listener: RealtimeRecognitionListener,
  ): RecognitionRun;

  reset(): void;
  dispose(): Promise<void>;
}
```

The discriminated variants of `RecognitionRuntimeError` are owned by `spec:product.system.contracts.runtime_errors`. Runtime/infrastructure failures must use `onError`; ordinary ineligible camera content must use normal `onUpdate` state.

## Realtime recognizer behavior

`RealtimeRecognizer` owns:

- the target `100 ms` request cadence;
- single-in-flight backpressure;
- latest-frame acquisition;
- invocation of `RecognitionPipeline.evaluate`;
- commit-eligibility handling;
- three-consecutive semantic-result stabilization;
- materialization of committed `TileInstanceId` values;
- delivery of at most one `confirmed` result between `start()`/`reset()` stabilization boundaries.

The intended execution model is:

```text
cadence point
    │
    ├─ evaluation already in flight
    │      └─ skip; do not queue stale frame
    │
    └─ idle
           ↓
      captureLatest()
           ↓
      pipeline.evaluate()
           ↓
      eligibility + stabilizer
           ↓
      scanning / stabilizing / confirmed
```

## Confirmation behavior

The realtime facade implements `spec:product.system.concepts.recognition_state`:

- fewer than 10 valid visible non-dora tile observations cannot participate in stabilization;
- fewer than 2 valid `completed-hand` observations cannot participate in stabilization;
- an ineligible frame resets the current stabilization run;
- an eligible changed semantic draft becomes consecutive result 1;
- the same semantic draft for three consecutive completed evaluations confirms;
- bounding-box jitter alone does not break equality;
- no startup wait is inserted before recognition merely to create a fixed delay;
- no winning-shape/yaku/scoring check is used as a recognition gate.

## Reset

`reset()` clears realtime candidate/confirmation state without requiring the UI to reconstruct the recognizer or call internal stabilizer APIs.

If a live run remains active, subsequent frames begin a new stabilization boundary and may later produce one new `confirmed` result.

## Stop and dispose

`RecognitionRun.stop()` stops that live run from acquiring/evaluating new frames and from delivering a later confirmation.

`RealtimeRecognizer.dispose()` stops active realtime work and releases resources owned by the realtime recognizer itself. It does **not** dispose the app-lifetime ONNX model sessions owned by `RecognitionRuntime`.

Likewise, `RecognitionPipeline.dispose()` releases only pipeline-owned transient resources. Shared detector/classifier sessions are owned and disposed by `spec:product.system.contracts.model_runtime`.

## UI access rule

Production UI code may consume `RealtimeRecognizer` and the semantic types exported by the recognition public entry point.
It must not directly import or call:

- `RecognitionPipeline` implementation internals;
- NanoDet decode/post-processing helpers;
- duplicate-suppression helpers;
- tile/red-five classifier implementations;
- ONNX Runtime sessions;
- the temporal stabilizer.

`RecognitionPipeline` is a public architectural seam for composition/integration testing, not the normal page-level API.

## Test seams

The contract must permit at least:

- realtime-state tests using a fake `RecognitionPipeline` and fake `RecognitionFrameSource` without loading ONNX models;
- one-frame pipeline integration tests without rendering a page;
- Recognition-page tests using a fake `RealtimeRecognizer` without camera/model execution.

This separation is required so page behavior, stabilization behavior, and ML runtime integration do not need to be tested as one inseparable unit.
