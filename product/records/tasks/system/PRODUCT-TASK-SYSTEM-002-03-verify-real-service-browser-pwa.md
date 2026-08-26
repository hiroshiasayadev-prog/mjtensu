# PRODUCT-TASK-SYSTEM-002-03: Verify real-service browser PWA

- **status**: not_started
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

| check | expected result |
|---|---|
| production build with real service graph | PASS |
| real Agari WASM initialization/scoring path | PASS |
| real recognition model manifest/runtime initialization | PASS |
| shell startup before full ONNX acquisition | PASS |
| deferred model cache acquisition | PASS |
| offline startup after required caches are complete | PASS |
| offline Recognition after required model caches are complete | PASS |
| active session survives update availability until user/app chooses update route | PASS |
| no old-JavaScript/new-manifest mixing | PASS |
| real-service route/history checks | PASS |
| strict typecheck/lint/architecture gate | PASS |

The overall result is PASS only when every required check is PASS.

## Evidence

- This Task owns the L3 real-service/PWA verification required by `spec:product.system.contracts.testing_strategy`.
- Browser/version, build/model-set/WASM identities, cache states, commands, observed results, and final verdict are recorded here when executed.
- Real physical camera/device acceptance remains owned by I04/I05.
