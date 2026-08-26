# PRODUCT-TASK-UI-001-09: Align Recognition UI with public service contracts

- **status**: done
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
- `src/camera/contracts.ts` now defines the Camera public `CameraService`, `CameraSession`, `CameraPreview`, `CameraFrame`, `CameraOpenRequest`, `CameraRuntimeError`, and `Size` contract types, and `src/camera/index.ts` exposes them through the Camera top-level entry point.
- `src/ui/recognition-page.tsx` now consumes `CameraService` / `CameraSession` only from `@/camera` and `RecognitionRuntime` / `RealtimeRecognizer` / `FrameRecognitionSnapshot` / semantic observation and meld types only from `@/recognition`; the page-private Camera/Recognition service, update, snapshot, observation, and meld contract declarations were removed.
- Live presentation now reads public `TileObservation.bbox`, `TileObservation.classification`, and `MeldGroupObservation.interpretation.kind` directly while preserving the existing visible overlay and concealed-kan preview behavior.
- The camera-to-recognition projection remains page-local and returns the public `RecognitionFrameSource`, adding only the fixed visible `RECOGNITION_CAPTURE_REGIONS` to the raw public `CameraFrame`.
- `test/recognition-page.test.tsx` now uses public Camera/Recognition types and constructs the public `FrameRecognitionSnapshot` shape, including `draft` and `commitEligibility`.
- `test/e2e/fake-flow-main.tsx` now declares fake `CameraService`, `RecognitionRuntime`, and `RealtimeRecognizer` implementations directly and returns them as `RecognitionPageServices`; the runtime retry fixture also uses the public `RecognitionRuntimeError` field name `model` rather than the prior divergent `role` field.
- `src/ui/index.ts` no longer re-exports the removed page-private Camera/Recognition aliases.
- `spec:product.system.contracts.camera_api` and `spec:product.system.contracts.recognition_api` define the accepted page-facing service boundary.
- `spec:product.system.architecture` requires cross-feature consumption through public top-level entries and reserves composition to the app boundary.
- Focused Vitest verification passed: `recognition-page.test.tsx`, `architecture-boundaries.test.ts`, and `public-entry-points.test.ts` completed with 3/3 files and 24/24 tests passing.
- `npm run typecheck` completed successfully for both application and test TypeScript projects.
- `npm run lint` completed successfully with `Architecture import boundaries: OK (52 source files checked)`.
- Affected fake-service Playwright verification passed with 14/14 Chromium tests, including camera/runtime preparation ordering, owner-specific retry, Result correction flows, recognition correction flows, route guards, and session replacement.
- Verification therefore confirms the Recognition UI and fake-flow seam now consume the accepted public Camera/Recognition contracts without UI-private semantic translation aliases; PRODUCT-TASK-UI-001-07/F-MAJ-02 is corrected.
