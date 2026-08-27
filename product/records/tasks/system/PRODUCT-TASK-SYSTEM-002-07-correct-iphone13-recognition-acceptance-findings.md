# PRODUCT-TASK-SYSTEM-002-07: Correct iPhone 13 Recognition acceptance findings

- **status**: in_progress
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
- Preserve outside-region masking, live candidate overlays, meld feedback, automatic stabilization transition, and empty dora/meld validity.
- Add/adjust focused component/browser tests for the corrected startup and layout contracts where deterministic verification is possible.
- Re-run I04 on the real iPhone 13 after implementation; this Task does not mark I04 PASS by itself.

## Done condition

F-MAJ-01 through F-MAJ-03 are corrected in production code, focused automated verification passes, and the build is ready for target-device re-execution of the expanded I04 Recognition acceptance matrix.

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
- The first target-device re-execution showed that the UI/layout changes were loaded, but F-MAJ-01 was **not** corrected: rotating into landscape immediately produced `認識処理を続行できませんでした`, and the recovery actions were not tappable. This invalidates the earlier completion verdict, so the Task returned to `in_progress`.
- Source inspection identified a concrete orientation-race path consistent with the device failure: semantic-region source rectangles are valid only against a `16:9` Recognition frame, while Safari can still expose portrait-oriented `videoWidth` / `videoHeight` immediately after rotation. Starting realtime on that first landscape boundary can therefore feed portrait frame dimensions into fixed-composite aspect-ratio validation, which is normalized as detector `inference-failure`.
- The follow-up correction normalizes every camera capture to the same `16:9` geometry used by the visible `object-fit: fill` Recognition surface before semantic regions are applied. This preserves visible/input-boundary correspondence even while the underlying MediaStream dimensions lag orientation.
- Recovery interaction was hardened independently: preview video and decorative SVG overlays no longer participate in pointer hit-testing, recovery panels explicitly receive pointer events above the capture layers, and the exit control remains above recovery/visual overlays.
- Recognition recovery now also renders a concise model/kind/cause diagnostic on-device so any remaining target-device failure can be classified from the acceptance screen rather than guessed from the generic message.
- Follow-up automated verification completed on 2026-08-27: `npx vitest run test/camera-service.test.ts test/recognition-page.test.tsx` — **PASS**, 18/18 tests; `npm run typecheck` — **PASS**; `npm run lint` — **PASS**, architecture import boundaries OK across 58 source files; `npm run build` — **PASS**, Vite 8.2.2 production PWA build completed and generated `sw.js` / Workbox assets. Follow-up application bundle: `assets/index-Dea4oSG0.js`.
- Vite native-config-loader extension notices and the >500 kB chunk notice remain non-blocking for this functional correction.
- A subsequent iPhone 13 re-execution exposed two additional UI defects: the portrait guidance was duplicated during the initial entry because both the centered orientation note and the generic status surface could render the same guidance, and the recovery `再試行` / `トップへ` actions remained untappable despite panel-local z-index/pointer-event hardening.
- The portrait state now renders only the centered orientation note; the generic status surface is suppressed until landscape, eliminating duplicate guidance deterministically.
- Recovery ownership was moved to a dedicated full-capture interaction layer (`recognition-recovery-layer`) above all camera/decorative content with explicit `pointer-events: auto` and `touch-action: manipulation`; visual video/SVG layers remain non-interactive. The always-available `終了` control stays above that layer. This avoids relying on panel-local stacking behavior on iOS.
- Focused tests were extended to require a single portrait guidance surface and to assert the dedicated recovery interaction layer semantics.
- Automated verification completed on 2026-08-27 for this UI follow-up: `npx vitest run test/recognition-page.test.tsx` — **PASS**, 11/11 tests; `npm run typecheck` — **PASS**; `npm run build` — **PASS**, Vite 8.2.2 production PWA build completed. Follow-up application bundle: `assets/index-DqAoXi9U.js`.
- Another iPhone 13 re-execution is still required before this Task can return to `done`, specifically to confirm the portrait guidance appears only once and the `再試行` / `トップへ` recovery actions are tappable on-device.
- Device follow-up then confirmed the duplicate orientation guidance was removed. The remaining UX requirement was reconsidered: Recognition must not require the browser viewport itself to become landscape, because users may keep iOS orientation lock enabled while physically holding the device for landscape capture.
- Recognition startup is now independent of viewport orientation. The previous `matchMedia('(orientation: landscape)')` gate and orientation-blocking guidance were removed; camera/runtime readiness alone governs realtime startup.
- Layout adaptation is based on the actual viewport aspect ratio (`innerHeight > innerWidth`). A portrait viewport renders the same complete `16:9` Recognition surface rotated `90deg` clockwise and swaps the width/height fit formulas so the rotated surface remains fully visible. The semantic region coordinate system, mask, observations, recovery surface, and Recognition frame source remain unchanged because the entire surface rotates as one unit.
- Portrait viewport mode provides an in-surface `反対向き` action that toggles the presentation between `90deg` and `-90deg`, covering either physical landscape holding direction without depending on device-orientation APIs that may be constrained by orientation lock.
- Focused component coverage now requires realtime Recognition to start in a portrait viewport, the rotated surface to fit using the portrait formulas, no orientation-blocking guidance to render, and display-direction reversal not to restart Recognition.
- Automated verification completed on 2026-08-27 for the portrait-viewport adaptation: `npx vitest run test/recognition-page.test.tsx` — **PASS**, 11/11 tests; `npm run typecheck` — **PASS**; `npm run build` — **PASS**, Vite 8.2.2 production PWA build completed. Follow-up application bundle: `assets/index-CR-Z-1QG.js`.
- Target-device re-execution then showed the first portrait-viewport adaptation was visually incorrect: rotating the entire surface also rotated the camera preview, while the portrait stream was merely stretched into `16:9`. Camera motion therefore felt approximately 90 degrees out of phase with physical device motion. The layout adaptation cannot treat camera presentation as ordinary rotated UI content.
- The correction now separates camera orientation from UI orientation. In portrait viewport mode the `16:9` UI/overlay surface still rotates into the physical landscape presentation, but a portrait camera preview receives the opposite quarter-turn inside that surface, cancelling the UI rotation for the camera image itself. The preview uses aspect-preserving `object-fit: cover` instead of stretching.
- `CameraSession.captureLatest()` now accepts an optional quarter-turn and builds the canonical `16:9` Recognition frame with the same camera counter-rotation used by the preview. Portrait source frames are center-cropped at the reciprocal `9:16` source aspect, quarter-turned, and scaled into `16:9`; landscape frames are center-cropped directly to `16:9`. This keeps visible preview content, semantic overlay coordinates, and Recognition input in the same geometry without distortion.
- The current camera counter-rotation is read through a ref by the active Recognition frame source, so `反対向き` and viewport aspect changes update preview/capture orientation without restarting the Recognition run.
- Focused tests were extended to verify portrait camera counter-rotation, non-stretched canonical capture, dynamic `-90` / `90` Recognition capture rotation, and return to rotation `0` in a landscape viewport.
- Automated verification completed on 2026-08-27 for the camera/UI orientation correction: `npx vitest run test/camera-service.test.ts test/recognition-page.test.tsx` — **PASS**, 18/18 tests; `npm run typecheck` — **PASS**; `npm run lint` — **PASS**, architecture import boundaries OK across 58 source files; `npm run build` — **PASS**, Vite 8.2.2 production PWA build completed. Follow-up application bundle: `assets/index-l5C5e3Kw.js`.
- Target-device re-execution then confirmed the camera motion direction was corrected, but the rotation-based layout still changed the apparent camera/bbox sizing and produced vertically stretched-looking geometry between portrait and landscape. The remaining problem is therefore layout scaling, not camera orientation.
- The rotation-based portrait presentation has been removed entirely, including the `反対向き` control and camera quarter-turn API. Portrait and landscape now have separate layout definitions but share the exact same unrotated `16:9` capture-surface geometry, `object-fit: cover` preview behavior, semantic-region percentages, mask, and bbox coordinate system.
- To make the camera surface and bbox pixel scale stable across portrait/landscape on the same device, both layout definitions size the capture surface from the viewport short edge: width `min(100vw, 100dvh)`, height `min(56.25vw, 56.25dvh)`. Landscape no longer enlarges the Recognition surface independently; the tradeoff is intentional because an enlarged landscape surface cannot remain the same fully-visible size in portrait.
- Browser camera capture now mirrors the visible surface without any orientation transform: every source is center-cropped to `16:9` and scaled to a canonical `16:9` frame. A portrait `720×1280` source therefore becomes a centered `720×405` crop rather than being stretched or quarter-turned.
- This separation matches observed WebKit behavior where camera-track dimensions may change independently during orientation transitions; viewport layout and camera-stream orientation are therefore treated as independent inputs rather than inferred from one another.
- Focused tests now require identical capture-surface CSS sizing and identical semantic bbox percentage geometry across portrait/landscape resize, plus aspect-preserving center crop for portrait camera input.
- iOS/WebKit background checked during the correction: Safari viewport dimensions can change layout when orientation changes, dynamic viewport units intentionally follow the changing visible viewport, and WebKit camera-track dimensions have historically changed independently across orientation transitions. The implementation therefore does not derive camera transforms from viewport orientation; the viewport only selects a layout while the camera surface/input remain one stable `16:9` geometry.
- Automated verification completed on 2026-08-27 for the stable portrait/landscape geometry correction: `npx vitest run test/camera-service.test.ts test/recognition-page.test.tsx` — **PASS**, 18/18 tests; `npm run typecheck` — **PASS**; `npm run lint` — **PASS**, architecture import boundaries OK across 58 source files; `npm run build` — **PASS**, Vite 8.2.2 production PWA build completed. Follow-up application bundle: `assets/index-BZvBJVtA.js`.
- Target-device re-execution is still required before this Task can return to `done`: verify that portrait and landscape show the same apparent camera-surface size and bbox pixel scale, with no stretch or aspect-ratio change, and that camera motion remains natural in both viewport orientations.
