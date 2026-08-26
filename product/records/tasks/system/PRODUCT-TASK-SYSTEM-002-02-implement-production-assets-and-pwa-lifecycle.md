# PRODUCT-TASK-SYSTEM-002-02: Implement production assets and PWA lifecycle

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: implementation
- **estimate**: 1.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
  - PRODUCT-TASK-RECOGNITION-001-07
  - PRODUCT-TASK-SCORING-001-07
  - PRODUCT-TASK-APPLICATION-001-06
  - PRODUCT-TASK-UI-001-07
- **outputs**:
  - production model/WASM asset manifest and packaging
  - production PWA cache/update implementation
  - PRODUCT-TASK-SYSTEM-002-02

## Goal

Implement reproducible production asset pinning and the build-pinned PWA cache/update lifecycle for application shell, recognition models, and Agari WASM.

## Work

- Materialize the concrete production recognition model-set manifest with version, artifact URLs/paths, hashes, runtime specs, and provider preferences.
- Bind the exact reviewed Agari fork/WASM provenance selected by the Scoring Work Item into the production build.
- Precache the application shell and build-pinned manifest without making service-worker installation wait for all ONNX model payloads.
- Implement deferred/content-versioned recognition-model acquisition and offline reuse after assets are cached.
- Keep one running frontend build paired with its own compatible model manifest and prevent old-JavaScript/new-manifest mixing.
- Implement non-disruptive update availability so an active scoring session is not forcibly destroyed merely because a new service worker/build is available.
- Include Agari WASM in the coherent production asset/update story according to its build ownership while preserving scoring availability expectations.
- Add focused asset-manifest/cache/update tests at the deterministic/browser-support layer.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| recognition model manifest | Materialize one build-pinned manifest for detector/base/red-five artifacts with integrity/runtime/provider metadata. | Production runtime can identify exact model-set version and hashes; unknown/mismatched runtime specs are not silently accepted. | Manifest schema/integrity tests plus production build inspection. |
| Agari artifact pin | Tie the exact reviewed fork/WASM artifact to the production build reproducibly. | Build Evidence identifies the fork revision/artifact consumed by Scoring. | Build metadata inspection and S06 provenance cross-check. |
| service-worker install | Precache shell/build manifest without blocking install on full ONNX acquisition. | Initial PWA install/start can complete before recognition models are fully fetched. | Service-worker/cache integration tests. |
| deferred/offline model cache | Cache versioned ONNX assets after/deferred from shell startup and reuse them offline when complete. | After successful acquisition, an offline production build can initialize Recognition from cached shell/model assets. | Browser/PWA cache tests, with final release verification in I03. |
| update coherence | Preserve build/manifest compatibility and avoid forced active-session reload. | One running build cannot observe a newer incompatible manifest; update availability does not silently destroy the active session. | Controlled old/new build cache/update tests. |

## Done condition

The production build reproducibly pins the model set and Agari WASM, implements the accepted shell/deferred-model/offline/update cache lifecycle, and passes focused manifest/cache/update tests.

## Verification

- Validate production model manifest shape, hashes, and runtime specs.
- Build the production PWA and inspect emitted shell/manifest/model/WASM asset relationships.
- Run deterministic service-worker/cache/update tests.
- Run strict typecheck/lint/architecture checks where applicable.
- Defer full browser offline/update acceptance to PRODUCT-TASK-SYSTEM-002-03.

## Evidence

- PRODUCT-ADR-SYSTEM-002 and `spec:product.system.contracts.pwa_cache_update` define the cache/update contract.
- `spec:product.system.contracts.model_runtime` defines recognition manifest/runtime semantics.
- The reviewed Scoring Work Item supplies exact Agari provenance consumed here.
- Execution results are recorded here when the Task is performed.
