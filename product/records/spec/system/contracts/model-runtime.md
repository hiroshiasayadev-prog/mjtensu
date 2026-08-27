# Contract: Recognition model assets and runtime

- **id**: `spec:product.system.contracts.model_runtime`
- **status**: draft
- **date**: 2026-08-27
- **parent**: `spec:product.system`

## What this is

Implementation-facing contract for recognition-model asset discovery, background prefetch, ONNX Runtime session initialization, execution-provider fallback, reuse, and disposal.

The contract deliberately separates **model asset availability** from **inference-session readiness**.
Downloading/caching ONNX artifacts is not equivalent to constructing `InferenceSession` objects.

## Runtime model set

The production recognition runtime uses exactly three model roles:

```ts
export type RecognitionModelRole =
  | 'detector'
  | 'tile-classifier'
  | 'red-five-classifier';
```

Current responsibilities are:

| role | responsibility |
|---|---|
| `detector` | Detect tile candidate regions from the fixed `320 x 320` composite. |
| `tile-classifier` | Classify candidate crops independently into the 34 base riichi tile identities or invalid/background; the runtime may evaluate multiple crops in one bounded batch. |
| `red-five-classifier` | Refine base `5m`, `5p`, or `5s` candidates independently into ordinary-five versus red-five identity; the runtime may evaluate multiple applicable crops in one bounded batch. |

## Model-set manifest

The application uses one versioned recognition-model-set manifest rather than scattering model URLs, hashes, runtime contracts, or provider policy through feature code.

```ts
export type ExecutionProvider =
  | 'wasm-simd'
  | 'wasm-threaded'
  | 'webgl';

export type RecognitionModelRuntimeSpec =
  | 'nanodet-plus-m-320-v1'
  | 'c8-tile-35-v1'
  | 'c8-red-five-v1';

export interface RecognitionModelArtifactManifest {
  readonly role: RecognitionModelRole;
  readonly url: string;
  readonly sha256: string;
  readonly runtimeSpec: RecognitionModelRuntimeSpec;
  readonly providerPreference: readonly ExecutionProvider[];
}

export interface RecognitionModelSetManifest {
  readonly schemaVersion: 1;
  readonly modelSetVersion: string;
  readonly models: Readonly<
    Record<RecognitionModelRole, RecognitionModelArtifactManifest>
  >;
}
```

The manifest is a declaration of which artifact implements each known runtime contract. It is not a general-purpose configuration language for arbitrary ONNX graphs.

The runtime must reject an unknown `runtimeSpec` rather than guessing input/output or preprocessing semantics.

## Runtime-spec ownership

A `RecognitionModelRuntimeSpec` identifies a known code-owned model contract.
The implementation behind each spec owns the exact runtime behavior required for that artifact, including where applicable:

- tensor input name, shape, batch-axis behavior, and dtype;
- preprocessing and normalization;
- semantic class/output ordering;
- detector decoding and post-processing;
- classifier output interpretation.

These semantics must not be reconstructed from filename conventions or UI code.

Exporter-produced metadata may be used by build/release tooling to verify or generate manifest entries, but the full training/export metadata document is not itself the production runtime contract.

## Asset prefetch

Model artifact acquisition is represented separately from inference runtime initialization.

```ts
export interface RecognitionModelAssets {
  prefetch(manifest: RecognitionModelSetManifest): Promise<void>;
}
```

The application may start `prefetch()` asynchronously after Top is available.
Prefetch must not block initial Top-page presentation.

Prefetch behavior:

- requests/caches all three manifest-referenced ONNX artifacts;
- deduplicates concurrent requests for the same model-set/artifact rather than downloading the same artifact twice;
- permits the Recognition runtime to await the same in-flight acquisition when the user enters Recognition before prefetch completes;
- does not construct ONNX Runtime `InferenceSession` objects;
- does not allocate long-lived inference backend/GPU resources merely because Top was opened;
- a background prefetch failure does not make Top unusable; Recognition initialization may retry/resolve the required artifact later.

The concrete browser/PWA cache mechanism is implementation-owned. Cache correctness must be based on versioned/content-addressable asset identity rather than assuming that a stable URL always contains unchanged bytes.

`sha256` is the artifact identity/integrity value exposed by the manifest. Exact verification timing may be optimized by implementation, but a model artifact known not to match its manifest identity must not be used for inference.

## Recognition runtime lifecycle

Inference sessions are created lazily when Recognition is first needed and are retained for the application lifetime.

```ts
export interface RecognitionEvaluationTiming {
  readonly totalMs: number;
  readonly candidateCount: number;
  readonly redFiveCandidateCount: number;
  readonly detectorPreprocessingMs: number;
  readonly detectorInferenceMs: number;
  readonly detectorPostprocessingMs: number;
  readonly cropExtractionMs: number;
  readonly baseClassifierPreprocessingMs: number;
  readonly baseClassifierInferenceMs: number;
  readonly redFiveClassifierPreprocessingMs: number;
  readonly redFiveClassifierInferenceMs: number;
}

export interface RecognitionRuntimeModelDiagnostic {
  readonly role: RecognitionModelRole;
  readonly runtimeSpec: RecognitionModelRuntimeSpec;
  readonly selectedProvider?: ExecutionProvider;
  readonly failedProviders: readonly ExecutionProvider[];
}

export interface RecognitionRuntimeDiagnostics {
  readonly models: readonly RecognitionRuntimeModelDiagnostic[];
  readonly recentEvaluations: readonly RecognitionEvaluationTiming[];
}

export interface RecognitionRuntime {
  initialize(): Promise<void>;
  createPipeline(): RecognitionPipeline;
  getDiagnostics?(): RecognitionRuntimeDiagnostics;
  dispose(): Promise<void>;
}
```

Lifecycle:

```text
App / Top available
    ↓
background model-asset prefetch may run
    ↓
first Recognition entry
    ↓
RecognitionRuntime.initialize()
    ↓
resolve model artifacts
    ↓
create detector/classifier InferenceSessions
    ↓
READY
    ↓
Recognition page may be entered/exited repeatedly
    ↓
reuse the same initialized model sessions
    ↓
application teardown
    ↓
RecognitionRuntime.dispose()
```

`initialize()`:

- is idempotent after successful initialization;
- deduplicates concurrent initialization calls onto the same initialization work;
- may wait for an already-running asset prefetch rather than starting duplicate artifact retrieval;
- creates the three inference sessions and makes them available to pipelines;
- does not recreate sessions merely because the Recognition page was left and later re-entered;
- after a failed initialization attempt, permits a later `initialize()` call to retry acquisition/session construction rather than permanently caching the rejected initialization promise.

`createPipeline()` may succeed only after runtime initialization is ready. Each created pipeline uses the runtime-owned model sessions rather than loading its own copies of the models.

`getDiagnostics?()` is an optional inspection surface for development, acceptance, and performance investigation. When implemented, it may report selected/failed providers and recent stage timings, including bounded classifier batch work. It is not required for recognition correctness and must not be used as an application/UI decision input.

## Session ownership

The long-lived detector and classifier `InferenceSession` objects are owned by `RecognitionRuntime`, not by individual Recognition pages or pipeline instances.

A page/run lifetime therefore releases only page/run/transient resources.
It must not dispose the shared model sessions when the user leaves Recognition for Conditions or Result.

`RecognitionRuntime.dispose()` is the owner operation that releases the long-lived model/runtime resources.
After disposal, the current runtime instance must not be reused without constructing a new runtime instance.

## Provider selection

Execution-provider selection is performed independently for each model session according to that model's `providerPreference` list.
There is no requirement that detector, base classifier, and red-five classifier use the same provider.

The current production preference for **all three model roles** is:

```ts
[
  'wasm-simd',
  'wasm-threaded',
  'webgl',
]
```

That order reflects current measured runtime behavior: WASM SIMD is preferred; WASM threaded is retained as the next fallback; WebGL remains supported as a lower-priority fallback.
A later dedicated end-to-end benchmark may change the per-model ordering without changing the manifest/API shape.

Provider initialization behavior for one model is:

```text
for provider in model.providerPreference
    try create session
    if successful -> select and stop
    if unavailable/initialization failure -> try next provider

if every provider fails -> runtime initialization failure
```

Provider fallback is an infrastructure concern and must not be implemented separately in pages or model-specific pipeline stages.

### WASM threaded availability

`wasm-threaded` may require browser capabilities such as cross-origin isolation. When those prerequisites are unavailable, that provider is treated as unavailable and fallback continues to the next manifest entry.

Unavailable provider prerequisites are not a reason to reject the entire runtime while another declared provider can initialize successfully.

## Provider diagnostics

The runtime may expose/log implementation diagnostics identifying the selected provider and failed provider attempts for development/performance investigation. The optional public `RecognitionRuntime.getDiagnostics()` inspection surface may carry these provider diagnostics together with recent evaluation timings.
These diagnostics are not recognition semantics and must not become UI/application decision inputs.

A development-only explicit provider override may be retained for benchmarking, but production automatic selection follows the manifest order.

## Model-set consistency

One initialized `RecognitionRuntime` uses one coherent `RecognitionModelSetManifest` version.
The detector and classifiers from different manifest versions must not be mixed ad hoc during one initialized runtime.

A newly deployed model-set version is picked up on a later runtime/application lifecycle according to PWA update/cache behavior; the current contract does not hot-swap model sessions in the middle of an active initialized runtime.

## Error boundary

The runtime must distinguish ordinary provider fallback from fatal model-runtime initialization failure.
A failed first-choice provider is not fatal when a later declared provider succeeds.

Fatal cases include at least:

- required manifest/model artifact unavailable after applicable retry/cache resolution;
- artifact integrity/identity mismatch that cannot be resolved;
- unknown/incompatible runtime spec;
- all declared execution providers fail for a required model;
- unrecoverable ONNX Runtime/session initialization failure.

Exact public error variants belong to `spec:product.system.contracts.runtime_errors`.

## Test seams

The design must permit:

- manifest validation tests without creating ONNX sessions;
- asset-prefetch tests without entering Recognition;
- runtime-lifecycle tests with fake model/session factories;
- provider fallback tests per model without loading the other model roles;
- pipeline tests with runtime-owned fake model sessions;
- repeated Recognition-page enter/leave tests proving that shared sessions are not recreated or disposed each time.

## Boundary

| concern | owner |
|---|---|
| Model-set manifest and asset/runtime lifecycle | This contract. |
| Model-training/export evidence | Investigations/export metadata/tooling. |
| Concrete ONNX preprocessing/decoding implementations | Recognition runtime-spec implementations. |
| Per-frame recognition semantics | `spec:product.system.contracts.recognition_api`. |
| Page/camera lifetime | Camera/Recognition page contracts. |
| Fatal runtime error taxonomy | `spec:product.system.contracts.runtime_errors`. |
