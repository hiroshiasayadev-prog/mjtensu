# PRODUCT-ADR-SYSTEM-002: Use build-pinned manifests and deferred model caching for PWA updates

- **status**: accepted
- **date**: 2026-08-26
- **depends_on**: PRODUCT-ADR-SYSTEM-001
- **migrated_to_spec**: `spec:product.system.contracts.pwa_cache_update`

## Context

The production application is a smartphone PWA with a conventional frontend shell plus three comparatively large ONNX recognition artifacts.
The shell should become available quickly and remain usable offline after installation, while recognition-model acquisition should not make PWA installation or initial Top rendering wait for the full model payload.

The recognition model runtime already separates model-asset acquisition from ONNX `InferenceSession` initialization and keeps one coherent model-set manifest for an initialized application runtime.
The PWA/service-worker layer therefore needs a cache/update policy that preserves that model-set consistency across deployments.

An independently refreshed model manifest would create a compatibility risk: an older JavaScript application could observe a newer manifest containing model artifacts or runtime-spec identifiers that it does not understand.
A forced service-worker activation/reload could also destroy an active scoring session merely because a deployment became available.

## Decision

Treat the application shell and recognition ONNX artifacts differently.

### Application shell and manifest

Precache the application shell required to start one frontend build.
Pair that build with one recognition-model-set manifest that is delivered as part of the same coherent application build.

Do not treat the production model manifest as an independently hot-updating network configuration source for an already running older frontend build.

### ONNX model artifacts

Do not require the three ONNX files as install-time precache assets.
After the initial UI is available, application bootstrap may begin the background model prefetch defined by the recognition model-runtime contract.

Use versioned/content-addressed model URLs together with the manifest's SHA-256 identity so cached artifacts are reusable without relying on a stable mutable URL.

Recognition initialization reuses a valid cached artifact or the same in-flight background acquisition before fetching a missing artifact.

### Offline behavior

Once the shell and all model artifacts for the active manifest are cached, the core capture/recognition/correction/scoring flow is expected to work without network access.

If the shell is cached but the model assets have never been acquired, shell-only UI remains usable, but Recognition cannot become ready offline and reports the normalized model-asset-unavailable failure.

### PWA updates

Do not force-reload an active application instance merely because a newer service worker/application build is available.
The running client continues using its existing JavaScript, manifest, model set, and inference sessions for that application lifecycle.

A newer build is adopted on a later restart/reload through the normal service-worker lifecycle.
Do not require forced `skipWaiting` plus automatic reload while an old client is active.

Old and new caches may coexist while the new build waits. Cleanup must not remove assets still needed by an active old client.
Stale caches may be removed after the newer build is active and the old application lifecycle is no longer using them.

## Rationale

The frontend shell is small enough and important enough for startup/offline availability that precaching is appropriate.
The ONNX payload is different: making it install-time mandatory would delay installation and initial UI for a resource that is only required when Recognition is used.

Background prefetch hides much of the later network latency without confusing downloaded bytes with initialized inference sessions.
Content/version-addressed model identity allows cache reuse and reliable deployment changes.

Build-pinning the model manifest keeps the JavaScript implementation and the model runtime contracts it understands coherent.
This avoids accidental combinations of old code with newly published incompatible runtime specs.

Allowing a running application to finish its current lifecycle also preserves the active scoring session and the app-lifetime inference-session ownership already selected elsewhere in the architecture.

## Rejected alternatives

### Precache every ONNX model during service-worker installation

This would make the large model payload a prerequisite for installation/update completion even when the user only opens Top or Help.
Background acquisition after initial UI availability provides a better startup tradeoff.

### Always fetch the latest model manifest from the network

This would permit an older frontend build to observe a manifest it was not built to understand.
Independent model-manifest updates may be introduced later only with an explicit compatibility/version-negotiation design.

### Network-first ONNX loading on every Recognition entry

The model artifacts are immutable/version-addressed within one manifest and should be reused from cache once acquired.
Repeated network-first loading would add latency and make offline recognition unnecessarily fragile.

### Force service-worker activation and reload immediately

An automatic reload can discard an active session and can replace the frontend while the user is mid-flow.
The current product has no requirement strong enough to justify that disruption.
