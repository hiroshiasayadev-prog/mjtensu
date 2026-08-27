# PRODUCT-TASK-SYSTEM-002-04: Verify iPhone 13 functional acceptance

- **status**: in_progress
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: verification
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-01
  - PRODUCT-TASK-SYSTEM-002-02
- **outputs**:
  - PRODUCT-TASK-SYSTEM-002-04

## Goal

Execute target-device functional acceptance of the installed production PWA on iPhone 13 using the real camera, production recognition models, production Agari WASM, and full scoring flow.

## Work

- Install/open the production PWA on iPhone 13 Safari/PWA mode and record environment/build identities.
- Verify camera permission/startup and the landscape Recognition capture layout.
- Verify actual model loading/provider selection and live recognition overlays.
- Verify a stable recognized structure automatically transitions to Conditions without a shutter or extra confirmation.
- Verify winning-tile/condition correction and successful score calculation through real Agari WASM.
- Verify Result presentation and at least one condition correction/recalculation path.
- Verify installed/offline behavior after required shell/model assets have been cached.
- Exercise a recoverable camera/runtime retry path when it can be induced safely and reproducibly; record BLOCKED for only that subcheck if the environment cannot induce it without changing the production build.
- Record expected/observed outcomes and one overall PASS, FAIL, or validly BLOCKED verdict.

## Done condition

Every predefined target-device functional check has an observed result and the overall verification verdict is PASS, FAIL, or validly BLOCKED under the production testing strategy.

## Verification

| check | expected result |
|---|---|
| iPhone 13 Safari/PWA production startup | PASS |
| camera permission/startup | PASS |
| landscape fixed capture regions | PASS |
| production model initialization/provider selection | PASS |
| live boxes/identity/meld feedback | PASS |
| three-stable-result automatic Conditions transition | PASS |
| Conditions selection/condition edit + real score calculation | PASS |
| Result display + recalculation path | PASS |
| cached offline application/Recognition availability | PASS |
| recoverable retry path | PASS or explicitly BLOCKED only when safe induction is unavailable |

The overall result follows the predefined acceptance-gate rules; an unexecuted required check is not silently treated as PASS.

## Evidence

- `spec:product.system.contracts.testing_strategy` selects iPhone 13 Safari/PWA as the initial real-device release acceptance environment.
- Device OS/browser/PWA mode, build/model-set/WASM identities, selected execution providers, screenshots/log notes where useful, observed results, and final verdict are recorded here when executed.
- Timing/performance acceptance is deliberately separate in I05.

### Acceptance session: 2026-08-27

Production identities fixed before target-device execution:

- production build asset version: `e15bf73e46ef0d48`
- production asset manifest: `production-assets-e15bf73e46ef0d48.json`
- recognition model set: `recognition-v1-2026-08-27`
- detector runtime spec: `nanodet-plus-m-320-v1`
- tile-classifier runtime spec: `c8-tile-35-v1`
- red-five-classifier runtime spec: `c8-red-five-v1`
- Agari upstream commit: `a0a9ce15cdf1bea6e7e158bbac1adb4e7a33a547`
- Agari fork commit: `fb362b6db416e67984cdb36f704d8ebf6657662e`
- Agari WASM SHA-256: `0e3297ed5f6807eac4d7369eb5846bc17e5ea4851470bf9d40c78ec6030e277c`

Target-device observations remain pending. Do not infer device OS, installed-PWA mode, selected ONNX execution providers, camera behavior, recognition success, offline behavior, or retry behavior from desktop/browser verification.

The current production UI proves runtime readiness but does not surface the selected provider identity. Therefore the provider-selection row cannot be marked PASS from UI observation alone; target-device provider evidence must be obtained through a device-observable diagnostic mechanism rather than inferred from manifest preference order.

#### Target-device observation record

| check | observed result |
|---|---|
| iPhone 13 / iOS version / Safari or installed-PWA mode recorded | pending |
| iPhone 13 Safari/PWA production startup | pending |
| camera permission/startup | pending |
| landscape fixed capture regions | pending |
| production model initialization/provider selection | pending — provider identity must be observed, not inferred |
| live boxes/identity/meld feedback | pending |
| three-stable-result automatic Conditions transition | pending |
| Conditions selection/condition edit + real score calculation | pending |
| Result display + recalculation path | pending |
| cached offline application/Recognition availability | pending |
| recoverable retry path | pending |

No overall verdict is recorded until the target-device observations above are executed.

### Acceptance usability adjustment verification: 2026-08-27

During target-device acceptance, two observability/usability issues were corrected before continuing the device gate:

- Recognition now surfaces whether the current frame is below commit eligibility, has unresolved meld geometry, or is in stable-result confirmation instead of presenting every non-confirmed state only as `認識しています`.
- Conditions-page tile correction now automatically installs each valid corrected structure; the extra `牌姿を反映` action is no longer required there. Result-origin recognition correction retains its explicit commit because that action owns recalculation/navigation behavior.

User-executed regression/build verification after these changes:

- `npx vitest run test/recognition-page.test.tsx test/tile-correction-ui.test.tsx` — **PASS**, 26/26 tests.
- `npm run typecheck` — **PASS**.
- `npm run build` — **PASS**, Vite 8.2.2 production PWA build completed and generated `sw.js` / Workbox assets.
- application bundle for this acceptance build: `assets/index-D8t1Pnba.js`
- production asset-manifest identity remains `e15bf73e46ef0d48`; that identity pins Recognition/Agari production artifacts and is not by itself a unique source/UI bundle identifier.
- Vite native-config-loader extension notices and the >500 kB chunk notice are non-blocking warnings for this functional acceptance task.

Because the production PWA deliberately leaves an update worker waiting while an existing controlled client is active, target-device continuation must ensure the updated application bundle is actually loaded rather than assuming an already-open installed PWA activated the new worker.
