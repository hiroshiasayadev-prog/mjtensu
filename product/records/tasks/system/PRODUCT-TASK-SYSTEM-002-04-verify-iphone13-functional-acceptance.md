# PRODUCT-TASK-SYSTEM-002-04: Verify iPhone 13 functional acceptance

- **status**: not_started
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
