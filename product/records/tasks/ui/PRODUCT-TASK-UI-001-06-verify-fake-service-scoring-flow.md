# PRODUCT-TASK-UI-001-06: Verify fake-service scoring flow

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: verification
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-UI-001-01
  - PRODUCT-TASK-UI-001-02
  - PRODUCT-TASK-UI-001-03
  - PRODUCT-TASK-UI-001-04
  - PRODUCT-TASK-UI-001-05
- **outputs**:
  - PRODUCT-TASK-UI-001-06

## Goal

Execute one deterministic browser-level acceptance gate for the complete visible scoring flow and recovery paths using public-contract fake Camera, Recognition, Scoring, and Application dependencies.

## Work

- Run Playwright through Top -> Recognition -> Conditions -> Result using deterministic recognition/scoring results.
- Verify Recognition preparation, camera/runtime failure, retry, and stable auto-transition behavior.
- Verify Recognition -> Conditions history replacement and guarded-route behavior.
- Verify Conditions winning-tile selection, condition editing, preview/readiness, calculation, and structural correction entry.
- Verify Result condition correction, recognition correction, immediate recalculation, Conditions fallback after confirmed repair-needed correction, and explicit new Recognition.
- Verify stale pre-correction Result is not restored after a confirmed structural correction.
- Verify Help round-trip and session preservation behavior.
- Record expected/observed outcomes and one PASS, FAIL, or validly BLOCKED verdict.

## Done condition

Every predefined fake-service browser-flow and recovery check has an observed result and the overall verification verdict is PASS, FAIL, or validly BLOCKED.

## Verification

| check | expected result |
|---|---|
| Top -> Recognition -> Conditions -> Result | PASS |
| Recognition startup/preparation state matrix | PASS |
| camera/runtime retry ownership | PASS |
| Recognition -> Conditions history replacement | PASS |
| winning-tile/condition interaction | PASS |
| no-yaku / invalid-input / invalid-shape visible recovery | PASS |
| Result -> Conditions -> Result condition correction | PASS |
| Result -> recognition correction -> immediate Result | PASS |
| confirmed recognition correction -> Conditions fallback | PASS |
| pre-confirm correction cancel preserves old Result | PASS |
| confirmed correction never restores stale old Result | PASS |
| explicit new Recognition replaces prior session | PASS |
| no-session Conditions/Result route guards | PASS |
| Help navigation/session preservation | PASS |
| strict typecheck/lint/architecture gate | PASS |

The overall result is PASS only when every required check is PASS.

## Evidence

- `spec:product.system.contracts.testing_strategy` defines this L3 fake-service browser acceptance layer.
- The fake services emit only public product contract values; real ONNX/WASM compatibility remains covered by feature L2 and final integration verification.
- Exact Playwright project/browser version, fixture identities, observed results, and final verdict are recorded here when executed.
