# PRODUCT-WORK-UI-001: Production scoring-flow UI

- **status**: completed
- **date**: 2026-08-26
- **source_refs**:
  - `spec:product.ui.screen_flow`
  - `spec:product.ui.pages.recognition`
  - `spec:product.ui.pages.conditions`
  - `spec:product.ui.pages.result`
  - `spec:product.ui.pages.recognition_correction`
  - `spec:product.ui.components.condition_controls`
  - `spec:product.ui.components.tile_correction_editor`
  - `spec:product.system.contracts.testing_strategy`
- **impact_refs**: []
- **tasks**:
  - PRODUCT-TASK-UI-001-01
  - PRODUCT-TASK-UI-001-02
  - PRODUCT-TASK-UI-001-03
  - PRODUCT-TASK-UI-001-04
  - PRODUCT-TASK-UI-001-05
  - PRODUCT-TASK-UI-001-06
  - PRODUCT-TASK-UI-001-07
  - PRODUCT-TASK-UI-001-08
  - PRODUCT-TASK-UI-001-09
  - PRODUCT-TASK-UI-001-10
  - PRODUCT-TASK-UI-001-11
  - PRODUCT-TASK-UI-001-12

## Goal

Implement the production Top, Recognition, Conditions, correction, Result, and Help surfaces against public service/application contracts, with deterministic component and browser-flow tests that do not wait for real model/WASM completion.

## Boundary

This Work Item owns production React pages/components, route-visible behavior, fake-service-driven UI testing, and presentation mapping from product semantic types.

It does not own camera/model/scoring engine internals, Application domain/state semantics, real ONNX/WASM integration, service-worker caching, or final real-device acceptance.

## Impact Scope

| target | impact |
|---|---|
| production `ui` module | Implement pages, visible components, camera overlay presentation, correction UI, and navigation behavior. |
| router/page guards | Implement visible route behavior against Application session existence and screen-flow semantics. |
| UI tests | Add Testing Library component coverage and Playwright fake-service flow coverage. |

## Task flow

```text
SYSTEM T05 bootstrap review PASS
   +-> U01 shell / Top / Help / routes
   +-> U02 Recognition page with public Camera/Recognition fakes
   +-> U03 Conditions and condition controls
   +-> U04 shared tile-correction editor and Recognition correction page
   +-> U05 Result/yaku/fu/payment presentation

U01 + U02 + U03 + U04 + U05 -> U06 fake-service browser E2E verification -> U07 independent integrated review
                                                                                  |
                                                                                  | NEEDS REVISION
                                                                                  v
                                               U08 condition-correction transaction
                                               U09 Recognition public contracts
                                               U10 scoring-flow dependency seam
                                                        \   |   /
                                                         U11 correction verification
                                                                  |
                                                                  v
                                                         U12 independent re-review
```

U01 through U05 may proceed in parallel where they do not write the same shared component/router surface. Shared-writer order must be serialized locally when implementation reveals an actual common-file conflict.

## Task Candidates

| task | task type | responsibility | dependency |
|---|---|---|---|
| PRODUCT-TASK-UI-001-01 | implementation | Implement application shell, Top, Help, route guards, and baseline screen-flow routing. | SYSTEM T05 |
| PRODUCT-TASK-UI-001-02 | implementation | Implement Recognition page startup/preparation/failure/live-overlay/auto-transition presentation against public service fakes. | SYSTEM T05 |
| PRODUCT-TASK-UI-001-03 | implementation | Implement Conditions page, winning-tile selection, condition controls, preview/readiness presentation, and calculate action. | SYSTEM T05 |
| PRODUCT-TASK-UI-001-04 | implementation | Implement the shared tile-correction editor and Result-origin Recognition correction surface. | SYSTEM T05 |
| PRODUCT-TASK-UI-001-05 | implementation | Implement Result page score summary, yaku list, fu detail, payment presentation, and correction/restart actions. | SYSTEM T05 |
| PRODUCT-TASK-UI-001-06 | verification | Execute the deterministic fake-service Playwright scoring-flow and recovery suite. | U01, U02, U03, U04, U05 |
| PRODUCT-TASK-UI-001-07 | review | Independently review the complete production UI behavior against the UI Specifications. | U06 |
| PRODUCT-TASK-UI-001-08 | correction | Make Result-origin condition correction cancellable/transactional and consume the seat-wind focus shortcut. | U07 |
| PRODUCT-TASK-UI-001-09 | correction | Align Recognition UI and fakes with the public Camera/Recognition service contracts. | U07 |
| PRODUCT-TASK-UI-001-10 | correction | Remove fabricated scoring fallback semantics from the production UI dependency context. | U07 |
| PRODUCT-TASK-UI-001-11 | verification | Verify all U07 corrections with focused, browser-flow, architecture, type, and build gates. | U08, U09, U10 |
| PRODUCT-TASK-UI-001-12 | review | Independently re-review the corrected production UI and close or re-open findings. | U11 |

## Completion Condition

- All primary-flow pages and correction/recovery surfaces defined by the UI Specifications exist.
- The UI consumes public product/Application contracts and does not import ONNX Runtime or Agari bindings directly.
- Conditions and correction behavior uses the shared policy/editor contracts rather than duplicating semantic rules in page components.
- Focused component tests pass.
- Fake-service browser E2E proves the required navigation, recovery, correction, and stale-result behavior.
- The final independent integrated review (U12 after U07 corrections) is PASS with no unresolved findings.

## Evidence

- The UI page/component Specifications define the production visible behavior.
- `spec:product.ui.screen_flow` defines route/navigation semantics.
- The production testing strategy permits deterministic fakes specifically so UI implementation and browser-flow verification can proceed before real Recognition/Scoring integration.
- U07 independent review recorded **NEEDS REVISION** with four findings covering Result-origin correction transaction semantics, Recognition public-contract divergence, fabricated scoring fallback semantics, and seat-wind shortcut focus.
- U08 through U10 corrected those findings; U11 verified the corrected current candidate with strict typecheck/lint/architecture/build gates, 29/29 Vitest files / 306/306 tests, and 14/14 focused fake-service Playwright cases.
- U12 independently re-reviewed the corrected production UI, closed all four U07 findings, found no new production UI boundary finding, and recorded final integrated verdict **PASS**.
- PRODUCT-WORK-UI-001 completion conditions are therefore satisfied and the Work Item is complete.
