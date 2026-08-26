# PRODUCT-TASK-UI-001-09: Align Recognition UI with public service contracts

- **status**: not_started
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: correction
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-UI-001-07
- **finding_refs**:
  - PRODUCT-TASK-UI-001-07/F-MAJ-02
- **outputs**:
  - Recognition UI bound to public Camera/Recognition contracts
  - PRODUCT-TASK-UI-001-09

## Goal

Correct F-MAJ-02 by removing the page-private duplicate Camera/Recognition service contract and wiring Recognition UI against the accepted top-level public feature APIs used by production integration.

## Work

- Expose/complete the production Camera public API required by `spec:product.system.contracts.camera_api` through the camera top-level entry point.
- Replace UI-owned Camera service/session/frame interfaces with the public Camera contract types.
- Replace UI-owned realtime Recognition service/update/snapshot types with the public `RealtimeRecognizer`, `RealtimeRecognitionUpdate`, and semantic snapshot types exported from the Recognition top-level entry point.
- Adapt presentation rendering to the existing public semantic fields such as observation bbox/classification and meld interpretation without creating a second semantic service contract.
- Keep the camera-to-recognition frame-source projection at the accepted boundary while avoiding detector/composite/private Recognition imports from UI.
- Keep preparation, retry ownership, landscape gating, live overlay, confirmation, and teardown behavior unchanged semantically.
- Update Recognition component tests and fake-flow service fixtures to implement the same public contracts used by production.
- Ensure the production composition root can supply the real Camera/Recognition implementations without a UI-specific semantic translation adapter.

## Done condition

Recognition page production dependencies and test fakes implement the same public Camera/Recognition contracts, with no duplicate page-level service/update contract and no private Recognition implementation imports in UI.

## Verification

- Type-check a production-compatible service bundle using the public Camera and Recognition top-level types.
- Verify Recognition preparation order, owner-specific retry, landscape gating, live observation/meld rendering, confirmation, and teardown tests remain green.
- Verify fake-flow E2E Recognition services implement the public contracts rather than UI-private aliases.
- Confirm UI imports only public `@/camera` and `@/recognition` entry points for cross-feature dependencies.
- Confirm no detector, classifier, stabilizer, ONNX, or private Recognition implementation import is introduced into UI.
- Run `npm run typecheck`, `npm run lint`, focused Recognition tests, and affected fake-service Playwright tests.

## Evidence

- U07 F-MAJ-02 found duplicate UI-owned Camera/Recognition service and update types in `src/ui/recognition-page.tsx`.
- The current public Recognition snapshot shape differs from the UI-private snapshot shape, so the fake-service E2E does not currently prove production-service compatibility.
- `spec:product.system.contracts.camera_api` and `spec:product.system.contracts.recognition_api` define the accepted page-facing service boundary.
- `spec:product.system.architecture` requires cross-feature consumption through public top-level entries and reserves composition to the app boundary.
