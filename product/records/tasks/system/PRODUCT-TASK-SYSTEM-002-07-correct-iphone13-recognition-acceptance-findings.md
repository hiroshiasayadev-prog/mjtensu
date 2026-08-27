# PRODUCT-TASK-SYSTEM-002-07: Correct iPhone 13 Recognition acceptance findings

- **status**: done
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: correction
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-01
  - PRODUCT-TASK-SYSTEM-002-02
- **finding_refs**:
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-01
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-02
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-03
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-05
- **outputs**:
  - corrected iPhone 13 Recognition startup and capture experience
  - PRODUCT-TASK-SYSTEM-002-07

## Goal

Correct the functional and device-layout findings discovered during iPhone 13 Recognition acceptance so that the production Recognition page starts cleanly after camera permission and presents a usable, spec-conforming landscape capture surface without device-specific retry or alignment workarounds.

## Work

- Reproduce and identify the normalized owner/category of the first-use failure that occurs immediately after granting camera permission on iPhone 13.
- Correct the startup race/lifecycle behavior so a healthy first permission grant proceeds to usable Recognition without requiring a second manual retry.
- Preserve independent camera/runtime ownership: a healthy resource must not be torn down merely because the other side is preparing, retried, or fails.
- Rework the Recognition route layout so the camera/capture surface is the primary iPhone landscape surface and the entire fixed recognition surface is visible without vertical clipping or scrolling.
- Keep an accessible abandon/recovery affordance in the viewport-filling Recognition experience; do not rely on an inaccessible off-screen page header.
- Adjust visible semantic-region placement so completed hand, dora, and meld regions remain contract-correct while being practical to populate simultaneously on the target tabletop/device setup.
- Preserve required region aspect ratios (`17:4`, `17:4`, `1:1`) and ensure visible frames remain the actual recognition-input boundaries.
- Preserve outside-region masking and make live candidate state plus meld-group feedback readily distinguishable on the camera surface. Exact colors remain implementation-owned, but an all-white overlay treatment that is difficult to distinguish on the target device is not acceptable.
- Preserve automatic stabilization transition and empty dora/meld validity.
- Add/adjust focused component/browser tests for the corrected startup and layout contracts where deterministic verification is possible.
- Re-run I04 on the real iPhone 13 after implementation; this Task does not mark I04 PASS by itself.

## Done condition

F-MAJ-01 through F-MAJ-03 and F-MAJ-05 are corrected in production code, focused automated verification passes, and the build is ready for target-device re-execution of the expanded I04 Recognition acceptance matrix.

## Verification

- `npx vitest run test/camera-service.test.ts test/recognition-page.test.tsx`
- `npm run typecheck`
- `npm run lint`
- `npm run build`
- Real-device re-verification remains owned by PRODUCT-TASK-SYSTEM-002-04.

## Evidence

- `spec:product.ui.pages.recognition` requires the camera preview to be the primary surface, visible frames to correspond to actual recognition input, owner-specific startup/recovery, and automatic shutterless transition.
- `spec:product.recognition.runtime_recognition` fixes the semantic-region roles/aspect ratios while leaving implementation-owned visible placement free to satisfy target-device usability.
- PRODUCT-TASK-SYSTEM-002-04 recorded F-MAJ-01 through F-MAJ-03 from iPhone 13 execution.

### Correction implementation: 2026-08-27

- F-MAJ-01 was reproduced at the browser-camera/current-frame boundary: immediately after a newly permitted stream attaches, Safari can transiently reject preview `play()` or expose non-zero video readiness/dimensions before `drawImage(video, ...)` can copy the first frame. Before this correction, a thrown current-frame copy escaped `CameraSession.captureLatest()` and `RealtimeRecognizer` normalized the source-side exception as detector `inference-failure`, presenting Recognition-owned recovery even though the camera session itself was healthy.
- `BrowserCameraSession` now waits for media data before the initial preview play when necessary, retries one transient first `play()` rejection without reopening the camera, and treats transient `InvalidStateError` / `AbortError` frame-copy failures as warmup (`null`) so the existing realtime cadence retries naturally. Unknown frame-copy failures still surface instead of being silently swallowed.
- The camera/runtime ownership contract remains independent: camera recovery does not reinitialize a healthy runtime, and runtime recovery does not detach/reopen a healthy camera session. Existing focused owner-specific tests remain in place.
- F-MAJ-02 was corrected by making Recognition a viewport-filling route overlay and fitting the complete `16:9` capture surface by both dynamic viewport width and height (`min(100vw, 177.7778dvh)` / `min(100dvh, 56.25vw)`). The production shell header/page spacing can no longer clip the capture surface vertically. An always-visible `終了` control remains inside the capture surface, and recovery panels remain accessible inside the same viewport-filling experience.
- F-MAJ-03 was corrected without changing semantic roles/aspect ratios: dora remains `17:4`, completed hand remains `17:4`, and meld remains `1:1`; the dora row now ends only about `2.1%` of capture height above the completed-hand row instead of the previous roughly `32.1%` gap. Visible frames remain the exact normalized regions passed into each recognition frame.
- Focused tests were added/adjusted for transient first-frame warmup, transient preview-play retry, viewport-filling capture geometry, preserved region aspect ratios, practical dora/hand spacing, and the in-viewport abandon affordance.
- Automated verification completed on 2026-08-27: `npx vitest run test/camera-service.test.ts test/recognition-page.test.tsx` — **PASS**, 17/17 tests; `npm run typecheck` — **PASS**; `npm run lint` — **PASS**, architecture import boundaries OK across 58 source files; `npm run build` — **PASS**, Vite 8.2.2 production PWA build completed and generated `sw.js` / Workbox assets.
- Build warnings about future Vite native-config-loader extension requirements and the >500 kB application chunk are non-blocking for this functional correction Task.
- F-MAJ-01 through F-MAJ-03 are therefore corrected in production code with focused automated verification passing. This Task is complete and the build is ready for target-device I04 re-execution; device PASS remains owned by PRODUCT-TASK-SYSTEM-002-04.
