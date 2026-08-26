# Contract: PWA cache and update lifecycle

- **id**: `spec:product.system.contracts.pwa_cache_update`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Implementation-facing contract for the production PWA's application-shell caching, recognition-model asset caching, model-manifest/version consistency, offline behavior, and service-worker update lifecycle.

The goal is to keep one running application instance internally coherent while avoiding a large mandatory ONNX download during PWA installation or initial Top rendering.

## Cache classes

The application distinguishes two cache classes with different acquisition policies.

### Application shell

The application shell includes the static assets required to start the current frontend build, including the build-pinned recognition-model-set manifest.

The shell is precached as part of the PWA/service-worker build so an installed build can start without requiring a fresh network fetch for its own JavaScript/CSS/manifest resources.

The exact Workbox/vite-plugin-pwa generated file list is implementation-owned, but the manifest consumed by one application build must be part of that build's coherent static asset set.

### Recognition ONNX artifacts

The three ONNX artifacts are not mandatory install-time precache assets.

They use the asset lifecycle from `spec:product.system.contracts.model_runtime`:

```text
initial UI available
    ↓
application-owned background prefetch
    ↓
content/version-addressed runtime cache
    ↓
RecognitionRuntime.initialize()
    ↓
use cached/in-flight artifact or fetch if still absent
```

This prevents PWA installation or initial Top presentation from being blocked by the full model download.

## Model artifact identity

Recognition-model artifact URLs must be versioned or content-addressed so an unchanged URL is not relied upon to identify changed ONNX bytes.

The model-set manifest also supplies `sha256` identity/integrity information as defined by `spec:product.system.contracts.model_runtime`.

Cached model lookup therefore corresponds to one declared artifact identity, not merely to a semantic role such as `detector`.

A cached artifact known not to match the active manifest must not be used for inference.

## Build and manifest consistency

A production frontend build is paired with one recognition-model-set manifest that the build understands.

The application must not treat the model manifest as an independently hot-updating network configuration source while an older JavaScript build remains active.

In particular, the following combination must not be created by ordinary runtime refresh behavior:

```text
old application JavaScript
+
new independently fetched model manifest
+
unknown/new runtimeSpec
```

A new model-set manifest is delivered as part of a new coherent application/PWA build unless a later compatibility design explicitly introduces independently updatable model manifests.

One initialized `RecognitionRuntime` continues to use the one manifest/model-set version selected for that application lifecycle and does not hot-swap model sessions midway through an active run.

## Runtime cache strategy for model assets

For the active build's declared ONNX URLs, cached valid artifacts are preferred over re-downloading identical bytes.

Conceptually:

```text
request required artifact
    ↓
valid cached artifact exists?
    ├─ yes -> use cached artifact
    └─ no  -> fetch -> verify/identify -> cache -> use
```

Background prefetch and Recognition initialization must deduplicate the same in-flight acquisition as required by the model-runtime contract.

A network failure does not invalidate a valid cached artifact for the active manifest.

## Offline behavior

When the current application shell and all required recognition artifacts for the active manifest are already cached, the core product flow can run without network access because camera capture, recognition inference, correction, and scoring are local browser operations.

When the shell is available but one or more required recognition artifacts have never been cached:

- Top and other shell-only UI remain available;
- Recognition initialization may attempt normal asset acquisition;
- if the artifact cannot be acquired, Recognition surfaces the normalized `model-asset-unavailable` runtime failure;
- the application must not fabricate a partially initialized recognition pipeline.

The first product contract does not promise that a never-before-loaded recognition model is usable offline.

## PWA update policy

A newly available application/service-worker build must not forcibly reload an active application instance or destroy an active scoring session merely to apply the update.

The current running application continues with its existing JavaScript, manifest, model-set identity, and initialized runtime until that application lifecycle ends.

The new build becomes the active user experience on a later application restart/reload according to the normal service-worker lifecycle.

The production update policy therefore does not require `skipWaiting`-style forced activation followed by an automatic page reload while an old client is active.

If update availability is exposed to the user in the future, accepting an update must still respect the product's active-session lifecycle rather than silently discarding work.

## Cache transition across application versions

Old and new model-set assets may coexist temporarily while a new PWA build is waiting to become active.

Cache cleanup must not remove assets still required by an active old application client merely because a newer service worker has been downloaded.

Once the newer build is active and no older application client depends on the previous model set, stale application/model caches may be cleaned up according to implementation-owned storage policy.

The current contract does not require retaining historical model sets indefinitely.

## Failure behavior

Cache/service-worker implementation failures that prevent required model acquisition are normalized through the recognition/model-runtime error boundary rather than exposed to feature UI as raw Cache API, Fetch API, or Workbox exceptions.

A failed background model prefetch remains non-fatal to Top presentation. It becomes user-relevant only when Recognition requires an unavailable artifact.

## Test seams

The implementation must permit verification that:

- Top can render without waiting for ONNX prefetch completion;
- ONNX artifacts are not mandatory install-time precache resources;
- background prefetch survives route changes;
- concurrent prefetch and Recognition initialization do not download the same artifact twice;
- an already cached active model set supports Recognition while offline;
- shell-only UI remains usable when offline before model acquisition;
- an old application client does not begin consuming a newly deployed incompatible manifest/model set;
- a PWA update does not force-reload an active scoring session;
- stale caches are cleaned only after they are no longer required by an active old build.

## Boundary

| concern | owner |
|---|---|
| Application-shell/model cache and update lifecycle | This contract. |
| Model-set manifest schema, artifact integrity, provider/session lifecycle | `spec:product.system.contracts.model_runtime`. |
| Runtime error semantics | `spec:product.system.contracts.runtime_errors`. |
| Frontend/PWA technology selection | `spec:product.system.architecture` and PRODUCT-ADR-SYSTEM-001. |
| Concrete Workbox/cache names and generated service-worker code | Implementation. |
