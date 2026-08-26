# Contract: Runtime error taxonomy

- **id**: `spec:product.system.contracts.runtime_errors`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Implementation-facing taxonomy for operational failures that cross camera, recognition-runtime, or scoring boundaries.

The taxonomy deliberately excludes ordinary semantic states that are already represented by normal feature contracts.
It also avoids one catch-all `SystemError` union: each owning module exposes only the failures relevant to its public boundary.

## What is not a runtime error

The following are normal states/results and must not be surfaced through this runtime-error taxonomy:

- fewer than the recognition commit-eligibility minimums;
- recognition still scanning or stabilizing;
- an `unresolved` recognition meld awaiting correction;
- incomplete Conditions input;
- contradictory draft scoring input represented by `ScoringPreview.invalid-input`;
- an invalid winning shape;
- no-yaku;
- `CameraSession.captureLatest() === null` when no usable latest frame is currently available;
- a preferred execution provider failing when a later declared provider successfully initializes.

These conditions are expected parts of the product flow rather than infrastructure/runtime failures.

## Camera runtime errors

```ts
export type CameraRuntimeError =
  | {
      readonly kind: 'permission-denied';
    }
  | {
      readonly kind: 'device-not-found';
    }
  | {
      readonly kind: 'device-unavailable';
    }
  | {
      readonly kind: 'unsupported';
    }
  | {
      readonly kind: 'runtime-failure';
      readonly cause: unknown;
    };
```

Semantics:

| kind | meaning |
|---|---|
| `permission-denied` | Camera access was denied by user/browser/site policy or is otherwise not permitted for the current origin. |
| `device-not-found` | No usable camera device matching the request exists. |
| `device-unavailable` | A camera exists but cannot currently be opened/used, for example because the device is busy or otherwise temporarily unavailable. |
| `unsupported` | The current browser/runtime lacks a camera capability required by the camera contract. |
| `runtime-failure` | An unexpected camera failure occurred after or during normal camera operation and does not fit a more specific recoverable category. |

`CameraService.open()` rejects with a normalized `CameraRuntimeError` rather than requiring callers to inspect browser-specific `DOMException.name` values.
Unexpected browser/platform error objects may be retained as `cause` only on the generic runtime-failure path or in diagnostics; callers must branch on `kind`, not on browser exception internals.

## Recognition runtime errors

```ts
export type RecognitionRuntimeError =
  | {
      readonly kind: 'model-asset-unavailable';
      readonly model: RecognitionModelRole;
    }
  | {
      readonly kind: 'model-integrity-failure';
      readonly model: RecognitionModelRole;
    }
  | {
      readonly kind: 'model-incompatible';
      readonly model: RecognitionModelRole;
    }
  | {
      readonly kind: 'execution-provider-unavailable';
      readonly model: RecognitionModelRole;
    }
  | {
      readonly kind: 'model-initialization-failure';
      readonly model: RecognitionModelRole;
      readonly cause: unknown;
    }
  | {
      readonly kind: 'inference-failure';
      readonly model: RecognitionModelRole;
      readonly cause: unknown;
    };
```

Semantics:

| kind | meaning |
|---|---|
| `model-asset-unavailable` | A required model artifact cannot be resolved from network/cache for runtime initialization. |
| `model-integrity-failure` | Resolved model bytes do not match the manifest identity/integrity contract and cannot be safely used. |
| `model-incompatible` | The model-set manifest/schema/runtime-spec is not supported by this application/runtime implementation. |
| `execution-provider-unavailable` | Every declared provider candidate for the required model failed or was unavailable. |
| `model-initialization-failure` | A required model session could not be constructed for a reason not normalized as provider exhaustion, integrity failure, or incompatibility. |
| `inference-failure` | An already initialized detector/classifier session failed during inference. |

Provider fallback failures before a later provider succeeds are diagnostics, not public runtime errors.
The public error exposes the affected semantic `RecognitionModelRole`; provider-specific stack traces, HTTP details, ORT errors, URLs, and benchmark diagnostics must not become UI decision inputs.

`RecognitionRuntime.initialize()` rejects with a `RecognitionRuntimeError` when initialization cannot complete.
`RealtimeRecognizer` reports fatal runtime/inference failures through `RealtimeRecognitionListener.onError(error)`.

A background Top-page model prefetch failure does not by itself enter a user-visible fatal state. It becomes a runtime error only if Recognition initialization later cannot resolve the required artifact or otherwise cannot initialize.

## Scoring errors

```ts
export type ScoringError =
  | {
      readonly kind: 'input-contract-violation';
      readonly cause?: unknown;
    }
  | {
      readonly kind: 'adapter-failure';
      readonly cause: unknown;
    };
```

Semantics:

| kind | meaning |
|---|---|
| `input-contract-violation` | The strict `ScoringInput`/rule-profile boundary was called with a state that violates its declared invariants. This normally indicates an implementation defect because ordinary editable/invalid states belong to `ScoringPreview`. |
| `adapter-failure` | Translation to/from the concrete scoring library or an unexpected scoring-library execution path failed even though the public input contract was coherent. |

The following must remain `ScoringPreview` results rather than `ScoringError` values:

- `incomplete`;
- `invalid-input`;
- `invalid-winning-shape`;
- `no-yaku`.

`ScoringService.calculate()` has only `ScoringCalculation` as its normal return value and reports these exceptional failures by throwing a normalized `ScoringError`.

## Application invariant failures

No general `ApplicationRuntimeError` taxonomy is defined for invalid application commands or impossible application state.

Examples such as:

- selecting a winning-tile ID that is not in the current completed hand;
- invoking a scoring-session operation without an active scoring session;
- constructing an internally impossible application state;

are programming/invariant defects unless a later product requirement identifies a recoverable user-visible case.
They should fail fast in development/tests rather than being converted into a generic recoverable application error bucket.

## Public information versus diagnostics

Public error discriminants contain only information required for caller recovery and presentation decisions.

Implementation diagnostics may additionally record:

- original browser/ORT/library exceptions;
- failed execution-provider attempts;
- URLs and HTTP statuses;
- stack traces;
- selected provider/runtime versions;
- model-set/version/hash details.

Those diagnostics must not be required by UI/Application logic.
Callers branch on stable semantic `kind` values and, for recognition, the affected model role.

## Top-level unexpected-error boundary

The application may retain one final UI/bootstrap error boundary that accepts `unknown` and presents a generic unexpected-failure recovery surface.
This is a fallback for defects/unclassified failures and is not a substitute for merging feature-owned errors into one `SystemError` union.

## Boundary

| concern | owner |
|---|---|
| Camera operational failures | Camera module / `CameraRuntimeError`. |
| Model/runtime/inference failures | Recognition module / `RecognitionRuntimeError`. |
| Strict scoring/adapter exceptional failures | Scoring module / `ScoringError`. |
| Normal recognition/scoring semantic states | Their existing state/result contracts, not this taxonomy. |
| Human-readable error copy and recovery controls | UI product/implementation layer. |
| Low-level diagnostic logging | Owning infrastructure implementation. |
