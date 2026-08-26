# Contract: Camera API

- **id**: `spec:product.system.contracts.camera_api`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Public implementation contract for browser-camera ownership, preview attachment, latest-frame acquisition, and camera-session teardown.

The camera module owns browser media APIs but does not own recognition regions, tile semantics, recognition scheduling, or scoring/application state.

## Public API

```ts
export interface CameraService {
  open(request: CameraOpenRequest): Promise<CameraSession>;
}

export interface CameraOpenRequest {
  readonly facingMode: 'environment';
}

export interface CameraSession {
  readonly preview: CameraPreview;

  captureLatest(): CameraFrame | null;
  stop(): Promise<void>;
}

export interface CameraPreview {
  attach(video: HTMLVideoElement): void;
  detach(): void;
}

export interface CameraFrame {
  readonly image: CanvasImageSource;
  readonly size: Size;
  readonly capturedAtMs: number;
}
```

`Size` is the library-independent system geometry type used by the recognition contract.

## Camera ownership

Only the camera implementation owns direct interaction with browser camera APIs such as:

- `navigator.mediaDevices.getUserMedia`;
- `MediaStream`;
- `MediaStreamTrack`;
- assignment of the owned camera stream to a preview element;
- camera-track teardown.

The UI must not acquire or stop `MediaStreamTrack` objects directly.
The recognition module must not request camera permission or construct browser media streams.

## Open behavior

`CameraService.open()` requests the rear/environment-facing camera and resolves only when a usable `CameraSession` has been established.

The initial browser capture request uses an implementation preference equivalent to:

```ts
{
  video: {
    facingMode: { ideal: 'environment' },
    width: { ideal: 1280 },
    height: { ideal: 720 },
  },
}
```

`1280 x 720` is an **ideal capture preference**, not a hard camera contract.
A browser/device may provide a different resolution and the session remains valid when the returned stream is otherwise usable.

The actual source size is reported by each `CameraFrame.size` and must not be inferred from the requested ideal constraint.

The system does not require the camera source itself to be `320 x 320`. Recognition owns normalization into the fixed detector composite.

## Why camera resolution is not fixed

The detector ultimately evaluates the fixed `320 x 320` recognition composite, so detector/classifier tensor cost is not reduced merely by changing camera capture resolution.

A lower camera resolution may reduce upstream video transfer, preview, crop, and resize work, but an unnecessarily low source can discard tile detail before recognition normalization.

The current contract therefore selects `1280 x 720` as the default ideal balance while leaving the actual resolution adaptive to the browser/device. A later measured performance decision may revise the ideal without changing the semantic camera API.

## Preview boundary

`CameraPreview.attach(video)` attaches the session-owned camera preview to the supplied `HTMLVideoElement`.

Requirements:

- the UI provides the display element but does not receive the underlying `MediaStream` as its public camera contract;
- calling `detach()` removes the session's preview attachment without transferring stream ownership to the UI;
- preview attachment does not define recognition crop regions;
- the fixed recognition-region overlay is owned by recognition/UI contracts, not by the camera module.

## Latest-frame acquisition

`CameraSession.captureLatest()` returns the latest currently usable camera image or `null` when no usable frame is currently available.

Requirements:

- the call does not queue frames;
- the call does not wait for a future frame;
- returned `size` describes the source coordinate space of `image`;
- `capturedAtMs` is the capture/snapshot timestamp used by the recognition frame boundary;
- the returned image must behave as one logical frame for the consumer's evaluation and must not expose a partially updated backing image during one `RecognitionPipeline.evaluate()` call.

The concrete snapshot/copy mechanism is implementation-owned so long as this logical-frame guarantee holds.

## Recognition adapter

The camera module does not add recognition-region semantics to `CameraFrame`.

The recognition-facing adapter combines the raw camera frame with the current fixed recognition-region layout:

```ts
export class CameraRecognitionFrameSource
  implements RecognitionFrameSource {
  constructor(
    camera: CameraSession,
    regions: RecognitionRegionProvider,
  );

  captureLatest(): RecognitionFrame | null;
}
```

The exact concrete class name is not part of the public camera API, but the dependency boundary is normative:

```text
CameraSession
    │ raw CameraFrame
    ▼
recognition-side frame-source adapter
    │ + RecognitionRegion layout
    ▼
RecognitionFrameSource
    ▼
RealtimeRecognizer
```

Camera itself must not know the meanings `completed-hand`, `dora-indicators`, or `melds`.

## Session lifetime

The intended page lifetime is:

```text
Recognition page enter
        ↓
CameraService.open()
        ↓
CameraSession
        ↓
preview.attach()
        ↓
RealtimeRecognizer.start()
        ↓
confirmed / page leave / fatal runtime error
        ↓
RecognitionRun.stop()
        ↓
preview.detach()
        ↓
CameraSession.stop()
```

`CameraSession.stop()`:

- is idempotent;
- stops every camera `MediaStreamTrack` owned by the session;
- prevents later frame acquisition from that stopped session;
- causes later `captureLatest()` calls to return `null`;
- leaves no active camera preview owned by the session.

Calling page teardown more than once must therefore not leak the camera or fail merely because the session was already stopped.

## Camera state

The public API does not expose a second mutable camera-state machine.
The lifecycle is represented by the operations themselves:

```text
before open resolves     -> opening
open resolves session    -> usable session
open rejects             -> open failure
session.stop() completes -> stopped
```

The UI/application may derive presentation state from these operations without synchronizing against an independent camera-owned `ready/opening/stopped` enum.

## Errors

Camera open/runtime failures must be normalized at the camera boundary rather than leaking browser-specific `DOMException` names through the rest of the application.

Camera failures use `CameraRuntimeError` from `spec:product.system.contracts.runtime_errors`, which distinguishes permission denial, missing hardware, temporary device unavailability, unsupported capability, and unexpected runtime failure.

Ordinary `captureLatest() === null` is not itself a runtime error.

## Test seams

The contract must permit:

- camera-owning browser integration tests without loading recognition models;
- realtime-recognizer tests using a fake `RecognitionFrameSource` without opening a camera;
- Recognition-page tests using fake camera/realtime services without `getUserMedia`;
- teardown tests proving repeated `stop()` does not retain active camera tracks.

## Boundary

| concern | owner |
|---|---|
| Browser camera stream and preview lifetime | Camera module / this contract. |
| Actual capture resolution | Browser/device, reported through `CameraFrame.size`. |
| Default `1280 x 720` ideal request | Camera implementation constrained by this contract. |
| Recognition semantic regions | Recognition system/product specs. |
| Fixed `320 x 320` detector composite | Recognition pipeline. |
| Realtime 100 ms scheduling/backpressure | `spec:product.system.contracts.recognition_api`. |
| Recognition-page overlays and masks | UI product specs. |
