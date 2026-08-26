# PRODUCT-TASK-SYSTEM-002-03: Verify real-service browser PWA

- **status**: completed
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: verification
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-01
  - PRODUCT-TASK-SYSTEM-002-02
- **outputs**:
  - PRODUCT-TASK-SYSTEM-002-03

## Goal

Execute the browser-level production integration gate using real composed services and production PWA assets, including offline/update/build-coherence behavior.

## Work

- Run the production browser build with real Scoring/Recognition runtime composition where browser automation can provide controlled camera/media input.
- Verify real Agari WASM initialization and score calculation through the production UI/Application path.
- Verify production Recognition model manifest/session initialization and bounded fixture/media recognition where supported by the browser test host.
- Verify application-shell startup and build-pinned manifest coherence.
- Verify deferred ONNX acquisition rather than service-worker-install blocking.
- Verify offline startup/Recognition availability after shell/model assets have been cached.
- Verify update availability without forced active-session destruction and without old-JavaScript/new-manifest mixing.
- Execute required real-service route/history integration checks.
- Record expected/observed outcomes and one PASS, FAIL, or validly BLOCKED verdict.

## Done condition

Every predefined real-service browser/PWA integration check has an observed result and the overall verification verdict is PASS, FAIL, or validly BLOCKED.

## Verification

| check | expected result | observed result |
|---|---|---|
| production build with real service graph | PASS | PASS — browser-verification build and final production build completed. |
| real Agari WASM initialization/scoring path | PASS | PASS — production Application/UI path calculated the bounded winning hand through real Agari WASM and rendered 2,600 points. |
| real recognition model manifest/runtime initialization | PASS | PASS — model set `recognition-v1-2026-08-27`; detector/base/red-five sessions all initialized with `wasm-simd`, no provider failures. |
| shell startup before full ONNX acquisition | PASS | PASS — Top rendered and production service worker activated while ONNX requests were deliberately held. |
| deferred model cache acquisition | PASS | PASS — after releasing held requests, all three production ONNX artifacts populated the browser model cache. |
| offline startup after required caches are complete | PASS | PASS — Top reloaded successfully with the browser offline after production caches were complete. |
| offline Recognition after required model caches are complete | PASS | PASS — the real Recognition artifact/runtime harness completed offline from cached shell/model assets. |
| active session survives update availability until user/app chooses update route | PASS | PASS — a waiting update probe was installed while the existing scored Result and active controller remained intact. |
| no old-JavaScript/new-manifest mixing | PASS | PASS — cached build identity remained unchanged while the new worker was waiting. |
| real-service route/history checks | PASS | PASS — production scoring Conditions/Result history round-trip passed; complete route/history browser suite passed. |
| strict typecheck/lint/architecture gate | PASS | PASS — typecheck passed; architecture import boundaries passed across 58 source files. |

**Overall verdict: PASS.**

## Evidence

- This Task owns the L3 real-service/PWA verification required by `spec:product.system.contracts.testing_strategy`.
- Browser/version, build/model-set/WASM identities, cache states, commands, observed results, and final verdict are recorded here when executed.
- Real physical camera/device acceptance remains owned by I04/I05.

### Verification harness: 2026-08-27

- Added a `browser-verification` Vite build mode that keeps the production PWA/service worker enabled while also emitting the controlled browser harness entries required by this Task. `npm run test:e2e` now runs that build instead of overwriting the production output with the PWA-disabled `e2e` mode before Playwright.
- Added `test/e2e/production-scoring.html` / `production-scoring-main.tsx` and `production-scoring.spec.ts`. The harness constructs the real production service graph, seeds one bounded known winning structure through the production Application store, executes calculation from the real Conditions UI through the real Agari WASM service, verifies Result/history behavior, inspects the browser-cached build manifest identity, and introduces a normal waiting service-worker update without activating/reloading the running scored session.
- Added `test/e2e/production-pwa.spec.ts`. It holds production ONNX requests while asserting shell/service-worker startup, releases them and verifies the three-model deferred Cache API population, then reruns the real Recognition artifact/runtime harness offline under the production service worker before verifying offline Top startup.
- Added `test/e2e/update-probe-sw.js` as a browser-verification-only update candidate with normal waiting semantics; it exists only to exercise the active-client update boundary and does not opt into `skipWaiting`.
- Existing `recognition-production-artifacts.spec.ts` remains the bounded real-model execution check and existing fake-service browser flow remains the route/history behavioral coverage; the new production-scoring harness adds the corresponding real Scoring/Application/UI path.

### Execution results: 2026-08-27

User-executed final verification:

- `npm run test:e2e` — **PASS**, Playwright Chromium 19/19 tests passed in 21.2 s. The supplied output did not print the concrete Chromium binary version; the project uses `@playwright/test` 1.62.1.
- Browser-verification Vite build — **PASS**, Vite 8.2.2 emitted the production PWA plus browser harnesses, `production-assets-e15bf73e46ef0d48.json`, Agari WASM, ORT WASM, `sw.js`, and Workbox runtime files.
- Real Recognition diagnostics — **PASS**: model set `recognition-v1-2026-08-27`; detector `nanodet-plus-m-320-v1`, tile classifier `c8-tile-35-v1`, and red-five classifier `c8-red-five-v1` all selected `wasm-simd` with no failed providers. The bounded classifier fixtures and blank-frame pipeline matched expected output.
- `production-pwa.spec.ts` — **PASS**: shell/service-worker startup completed while ONNX acquisition was held; all three model artifacts were then cached; cached Recognition and Top startup both remained available offline.
- `production-scoring.spec.ts` — **PASS**: the real production service graph initialized Agari WASM, calculated through the production Conditions/Result UI path, preserved the 2,600-point Result through browser history, kept the original active worker/controller while an update worker waited, and retained the same cached build-manifest identity.
- `npm run typecheck` — **PASS**.
- `npm run lint` — **PASS**, `Architecture import boundaries: OK (58 source files checked)`.
- `npm run build` — **PASS**, production PWA emitted 9 precache entries and generated `dist/sw.js` / `dist/workbox-2fbc6a65.js`.
- Build warnings about future Vite native config-loader extension requirements and >500 kB minified chunks are non-blocking and do not violate this Task's browser/PWA acceptance criteria.

Every predefined browser/PWA integration check has an observed PASS result. **PRODUCT-TASK-SYSTEM-002-03 is complete with overall verdict PASS.**

