# PRODUCT-TASK-SYSTEM-002-02: Implement production assets and PWA lifecycle

- **status**: done
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

### Implementation: 2026-08-27

- Added `src/recognition/model-runtime/production-model-set.json` as the single machine-readable production model-set source consumed by the runtime. Each ONNX URL now carries its declared SHA-256 as a content identity query while preserving the exact reviewed artifact filename, runtime spec, and provider order.
- `production-model-set.ts` now materializes and validates the runtime manifest from that JSON source, so unknown/mismatched runtime specs still fail through the existing manifest validation boundary rather than being inferred from filenames.
- Added `build/production-assets.ts`, which emits one content-versioned `production-assets-<sha>.json` build artifact from the exact recognition model-set source plus the canonical `vendor/agari-wasm/provenance.json`. The emitted manifest records the exact reviewed Agari module/artifact provenance consumed by the production build.
- Enabled `vite-plugin-pwa` production `generateSW` wiring. The install precache includes the application shell, generated build manifest, and build-owned WASM assets, explicitly excludes `*.onnx`, keeps navigation fallback/offline shell behavior, and raises the precache size ceiling enough for the current ORT WASM runtime asset.
- Kept service-worker update adoption non-disruptive with `skipWaiting: false` and `clientsClaim: false`. `src/pwa/lifecycle.ts` registers the production worker, exposes update-available subscriptions/callbacks, and never posts an activation message or reloads the active client.
- Recognition ONNX acquisition remains deferred through the existing SHA-256-verified browser Cache API resolver. Added a browser-cache test proving a fully acquired model set is reused by a fresh resolver while network fetch is unavailable.
- Added focused deterministic tests for production asset provenance/build pinning, ONNX-vs-WASM precache policy, first-install versus waiting-update lifecycle behavior, and offline recognition-model cache reuse.

### Verification: 2026-08-27

First local execution reached all gates except one deterministic asset-manifest suite:

- `test/pwa-lifecycle.test.ts`: **PASS**, 3/3.
- `test/recognition-browser-model-cache.test.ts`: **PASS**, 1/1.
- `test/scoring-production-wasm-artifact.test.ts`: **PASS**, 2/2.
- `test/recognition-production-model-artifacts.test.ts`: **PASS**, 2/2.
- `test/pwa-production-assets.test.ts`: **FAILED BEFORE TEST COLLECTION** because `build/production-assets.ts` used `fileURLToPath(new URL(..., import.meta.url))`; under Vitest's transformed module environment that URL was not `file:`. This was a test/build-tool path-resolution defect, not an asset-integrity or PWA contract failure.
- `npm run typecheck`: **PASS**.
- `npm run lint`: **PASS**, `Architecture import boundaries: OK (58 source files checked)`.
- `npm run build`: **PASS** with Vite 8.2.2 and PWA generateSW. The build emitted `production-assets-e15bf73e46ef0d48.json`, Agari WASM, ORT WASM, `sw.js`, and Workbox runtime files. Chunk-size and future native-config-loader messages were warnings only.
- Inspected generated `dist/sw.js`: the precache contains `production-assets-e15bf73e46ef0d48.json`, the application shell, `agari_wasm_bg-CaQzJWDG.wasm`, and `ort-wasm-simd-threaded-Cpm-ox6i.wasm`; no `.onnx` model payload appears in the install precache.
- Inspected `dist/production-assets-e15bf73e46ef0d48.json`: it records recognition model set `recognition-v1-2026-08-27`, all three exact SHA-256/runtime/provider declarations, and Agari fork commit `fb362b6db416e67984cdb36f704d8ebf6657662e` with WASM SHA-256 `0e3297ed5f6807eac4d7369eb5846bc17e5ea4851470bf9d40c78ec6030e277c` and 200739-byte identity.

The Vitest-only path-resolution failure was corrected by resolving production build inputs from the frontend working root instead of `import.meta.url`, which is not guaranteed to remain a `file:` URL after Vitest transformation.

Final corrected rerun:

- `npx vitest run test/pwa-production-assets.test.ts`: **PASS**, 1 file / 4 tests / 0 failures.
- The suite verifies the content-versioned build asset manifest, exact Recognition/Agari production pins, SHA-256-addressed model URLs, ONNX exclusion from install-time precache, and non-forced service-worker update policy.

All deterministic S02 acceptance checks are now PASS. Full browser offline/update acceptance remains intentionally deferred to PRODUCT-TASK-SYSTEM-002-03.
