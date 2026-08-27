# PRODUCT-TASK-SYSTEM-002-09: Correct iPhone 13 Recognition overlay feedback

- **status**: done
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: correction
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-07
- **finding_refs**:
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-05
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-06
- **outputs**:
  - corrected target-device Recognition overlay differentiation
  - PRODUCT-TASK-SYSTEM-002-09

## Goal

Correct the live Recognition overlay findings discovered during iPhone 13 acceptance so detector candidates, unresolved recognition state, meld-group feedback, and recognized tile identity are immediately understandable while aligning physical tiles on the live camera surface.

## Work

- Replace the current effectively all-white live overlay treatment with a visually distinguishable treatment for recognized detector candidates, unresolved candidates, and meld-group connectors/previews.
- Replace raw canonical labels such as `7z` with a user-facing tile identity treatment. Prefer a compact tile-face/icon representation over exposing internal suit/honor codes when the implementation remains lightweight.
- Keep overlay tile visuals local/static and browser-cacheable; do not generate/decode new image data per recognition update. A shared sprite or equivalently bounded static asset approach is acceptable.
- Preserve red-five distinction in the user-facing tile treatment.
- Do not introduce model confidence, provider, or other debug-only concepts into the production surface.
- Keep exact colors and tile artwork implementation-owned; satisfy the specification requirement that detector boxes and meld-group connectors are visually distinguishable on the target device.
- Preserve current observation geometry, outside-region mask, semantic-region frames, meld reconstruction, and stabilization behavior.
- Add focused Recognition-page tests that assert distinct semantic visual treatments without over-specifying exact decorative styling where unnecessary.
- Return to I04 for target-device verification after implementation; this Task does not mark the device acceptance row PASS by itself.

## Done condition

F-MAJ-05 and F-MAJ-06 are corrected in production code, focused automated verification passes, and the updated build is ready for iPhone 13 re-verification under I04.

## Verification

- `npx vitest run test/recognition-page.test.tsx`
- `npm run typecheck`
- `npm run lint`
- `npm run build`
- Real-device visual acceptance remains owned by PRODUCT-TASK-SYSTEM-002-04.

## Evidence

- `spec:product.ui.pages.recognition` requires detector boxes and meld-group connectors to be visually distinguishable while leaving exact overlay colors implementation-owned.
- PRODUCT-TASK-SYSTEM-002-04 records F-MAJ-05 from iPhone 13 execution: retained candidate boxes and meld connectors appear white and are difficult to distinguish during live alignment.
- PRODUCT-TASK-SYSTEM-002-07 was already completed for F-MAJ-01 through F-MAJ-03 before F-MAJ-05 was recorded; this finding therefore remains a separate corrective Task rather than retroactively changing I07 completion scope.

### Correction implementation: 2026-08-27

- F-MAJ-05 is corrected in the Recognition overlay implementation by giving recognized candidates and unresolved candidates separate semantic treatments instead of the previous all-white box treatment. Recognized observations use a solid recognized-state treatment; unresolved observations use a distinct dashed warning treatment.
- Meld-group feedback now has its own connector/preview treatment, visually separated from both candidate states. Existing observation geometry, semantic-region frames, outside-region mask, meld membership reconstruction, and stabilization flow are unchanged.
- F-MAJ-06 is corrected by removing canonical tile codes from the visible live feedback. Recognized observations and meld previews now render compact CSS tile faces using rank + `萬` / `筒` / `索` or the honor glyph (`東南西北白發中`) rather than labels such as `7z`.
- Tile-face rendering is local/static CSS and text only; the correction adds no runtime image generation, decoding, network lookup, model/provider confidence, or other debug-only production UI.
- Red-five identity remains explicit: red-five tile faces receive their own red visual treatment and user-facing accessible labels such as `赤五索`.
- Focused Recognition-page coverage now asserts recognized/unresolved visual differentiation, non-white meld connector semantics, tile-face rendering, concealed-kan backs, red-five preservation, and absence of raw canonical codes in the visible feedback.
- Automated verification completed on 2026-08-27: `npx vitest run test/recognition-page.test.tsx` — **PASS**, 11/11 tests; `npm run typecheck` — **PASS**; `npm run lint` — **PASS**, architecture import boundaries OK across 60 source files; `npm run build` — **PASS**, Vite 8.2.2 production PWA build completed and generated `sw.js` / Workbox assets.
- Verification build application bundle: `assets/index-xIfSdrbB.js`.
- Vite native-config-loader extension notices and the >500 kB application chunk notice are non-blocking for this overlay correction Task.
- F-MAJ-05 and F-MAJ-06 are therefore corrected in production code and automated verification is complete. Real-device visual acceptance remains owned by PRODUCT-TASK-SYSTEM-002-04.
