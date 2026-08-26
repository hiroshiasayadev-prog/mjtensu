# PRODUCT-TASK-UI-001-02: Implement Recognition page

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: implementation
- **estimate**: 2d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **outputs**:
  - production Recognition page implementation
  - PRODUCT-TASK-UI-001-02

## Goal

Implement the production Recognition page startup, camera/model preparation states, visible capture regions, live observations, failure recovery, and automatic stable-result transition using public Camera/Recognition service contracts.

## Work

- Implement the landscape camera layout with completed-hand, dora, and meld recognition frames and outside-region masking.
- Start camera and RecognitionRuntime initialization in parallel on page entry.
- Show preview as soon as camera is ready even while recognition runtime is still preparing.
- Start realtime recognition only when both camera and runtime are ready.
- Render retained candidate boxes, recognized identity overlays/unresolved states, and meld-group connector/preview feedback from public realtime updates.
- Render concealed-kan logical preview without inventing boxes for hidden members.
- Implement camera-specific and recognition-runtime-specific startup/fatal failure recovery with independent retry ownership.
- Stop route-owned realtime/camera work on leave while preserving app-lifetime recognition model initialization/session ownership.
- On stable recognition commit, create/replace the Application scoring session and navigate to Conditions with history replacement.
- Add focused component tests against deterministic Camera/Recognition fakes.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| capture surface | Render the three fixed semantic capture frames as the actual visible recognition boundary in landscape orientation. | No hidden inner crop/progress counter/region toggle is introduced; empty dora/meld regions remain valid. | Component/layout semantic tests. |
| preparation lifecycle | Start camera/runtime preparation independently and expose the accepted camera-ready/model-ready combinations. | Preview appears once camera is ready; recognition starts only after both dependencies are ready. | Deferred-promise fake-service tests. |
| live feedback | Render boxes/identities/unresolved/meld grouping from public recognition updates without interpreting model tensors/confidence internally. | Overlay state follows provided semantic observations and concealed-kan preview rules. | Fake realtime-update component tests. |
| recovery | Provide camera-owned and recognition-runtime-owned retry/Top actions without tearing down the healthy counterpart unnecessarily. | Each injected normalized error shows the specified recovery path and retry calls only the failing resource owner. | Error/retry matrix tests. |
| auto transition | On committed recognition, create/replace the active session and replace Recognition history with Conditions. | No shutter/extra OK is required; normal Back from initial Conditions does not reopen the completed camera run. | Fake commit + router/history test. |

## Done condition

The Recognition page matches its accepted startup/live/recovery/commit contract and passes focused deterministic tests without depending on actual ONNX model completion.

## Verification

- Run preparation-state matrix tests.
- Run live observation/meld/concealed-kan overlay tests.
- Run camera/runtime failure and retry matrix tests.
- Run stable-commit history-replacement test.
- Run strict typecheck/lint/architecture checks.

## Evidence

- `spec:product.ui.pages.recognition` defines the visible page behavior.
- Camera/model runtime/public Recognition contracts define service lifecycle/error inputs.
- `spec:product.system.contracts.testing_strategy` permits deterministic service fakes for this page implementation.
- Execution results are recorded here when the Task is performed.
