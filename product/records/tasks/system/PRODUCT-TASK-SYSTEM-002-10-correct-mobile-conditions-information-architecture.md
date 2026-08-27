# PRODUCT-TASK-SYSTEM-002-10: Correct mobile Conditions information architecture

- **status**: done
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: correction
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-09
- **finding_refs**:
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-07
- **outputs**:
  - corrected target-device Conditions mobile presentation
  - reusable scoring-flow tile presentation primitives where required
  - PRODUCT-TASK-SYSTEM-002-10

## Goal

Correct the Conditions-page mobile information architecture discovered during iPhone 13 acceptance so the page works as a continuous scoring-verification surface: the user can understand the recognized hand, select the winning tile, adjust conditions, immediately see how awarded yaku change, correct the hand when needed, and calculate without decoding internal identifiers or navigating an undifferentiated wall of controls.

## Work

- Replace the generic scoring-flow shell presentation on Conditions with a mobile app-bar treatment that uses `条件入力` as the route title and provides an accessible leading back action whose destination follows `spec:product.ui.screen_flow` semantics for initial Conditions versus Result-origin correction.
- Remove user-facing internal instance identifiers such as `recognition:43`; winning-tile state must be communicated by the selected tile instance itself.
- Rename/restructure the current ambiguous `認識牌姿` area around its actual primary task: `和了牌を選択` or an equivalent user-facing winning-tile selection concept.
- Render completed-hand, dora, and meld evidence using the lightweight reusable tile-face presentation established by I09 rather than canonical text codes such as `4p` / `7z`; preserve red-five distinction and duplicate tile-instance selection semantics.
- Consolidate recognized-structure verification and correction entry into one coherent hand card/surface rather than presenting `認識牌姿` and the full correction editor as two visually competing peer sections.
- Keep the completed hand prominent and selectable; show dora and meld evidence compactly below/adjacent as supporting evidence; expose a clear secondary `牌を修正` action that opens or reveals the shared correction interaction without making the full editor permanently compete with ordinary condition entry.
- Keep ordinary condition input in one vertically scrollable flow instead of splitting hand and condition entry across tabs. Preserve the existing semantic order from `spec:product.ui.pages.conditions` and retain `その他の条件` as a secondary disclosure.
- Introduce clear mobile visual grouping for the major semantic areas (hand/winning-tile verification, ordinary conditions, additional conditions) using bounded card/section separation rather than an undifferentiated white page. Avoid excessive nested-card chrome.
- Replace the ordinary in-document `現在の役` section plus separate calculation footer with a persistent bottom yaku/action dock that remains visible while the central Conditions content scrolls.
- The persistent yaku dock must provide immediate feedback after winning-tile, structure, or condition changes. It must represent at least `ready`, `no-yaku`, `invalid-winning-shape` / invalid-input, and incomplete states without exposing scoring-internal IDs.
- Keep the default dock compact enough not to dominate the viewport; summarize multiple yaku when necessary and allow an expanded/detail presentation for the full current-yaku list without requiring navigation away from Conditions.
- Place the calculation action in the same persistent bottom dock and enable/disable it from the existing scoring-readiness semantics. Ensure the scrollable content reserves enough bottom space that controls are not obscured by the dock.
- Account for iPhone safe areas in the fixed/persistent bottom surface and top app-bar layout.
- Preserve current-yaku semantics as feedback only: UI must continue delegating yaku/validity calculation to the existing scoring/application services rather than reimplementing scoring judgments.
- Preserve Result-origin condition-correction transaction/cancel semantics and the post-recognition-repair stale-result boundary established by the completed UI correction/review tasks.
- Add focused component/browser-flow coverage for back behavior, winning-tile visual selection, absence of internal IDs/codes, correction-entry hierarchy, persistent yaku feedback, dock state transitions, calculation availability, and Result-origin correction semantics.
- Return to I04 for target-device verification after implementation; this Task does not mark device acceptance PASS by itself.

## Done condition

F-MAJ-07 is corrected in production code, the Conditions page presents a coherent mobile scoring-verification hierarchy with persistent current-yaku feedback and calculation action, internal tile/session identifiers are absent from ordinary user-facing presentation, focused automated verification passes, and the updated build is ready for iPhone 13 re-verification under I04.

## Verification

- `npx vitest run test/conditions-page.test.tsx test/tile-correction-ui.test.tsx test/navigation-history.test.ts test/result-page.test.tsx`
- Run the focused fake-service scoring-flow Playwright cases covering Conditions and Result-origin correction.
- `npm run typecheck`
- `npm run lint`
- `npm run build`
- Real-device visual/interaction acceptance remains owned by PRODUCT-TASK-SYSTEM-002-04.

## Evidence

- `spec:product.ui.pages.conditions` defines Conditions as the post-recognition surface for recognized-hand verification, winning-tile selection, scoring-condition entry, correction, current-yaku feedback, and calculation in one semantic flow.
- `spec:product.ui.components.tile_presentation` requires reusable visible tile identity, red-five distinction, completed-hand order, winning-tile instance distinction, dora evidence, and meld grouping without exposing recognition-model details.
- `spec:product.ui.screen_flow` owns back behavior for initial Conditions, Result-origin condition correction, and post-recognition repair continuation.
- PRODUCT-TASK-SYSTEM-002-04 records F-MAJ-07 from iPhone 13 execution: internal winning-tile IDs/raw tile codes, ambiguous labels, duplicate/flat structure presentation, missing ordinary back affordance, and non-persistent current-yaku feedback make target-device verification unnecessarily difficult.
- PRODUCT-TASK-SYSTEM-002-09 owns the preceding Recognition overlay/tile-face correction so this Task reuses the same lightweight user-facing tile presentation rather than creating a second tile-art system.
- Implementation on 2026-08-27 introduces reusable `MobileScoringPageShell` and `PersistentBottomBar` primitives shared with the I11 Result correction. Conditions owns its route-title app bar and semantic Back action instead of the generic production header.
- Conditions presentation is implemented as one hand/winning-tile verification card, one ordinary-condition card, and one `その他の条件` disclosure card. The shared correction editor is hidden behind `牌を修正` and revealed inside the hand surface rather than permanently competing with ordinary condition entry.
- Completed-hand, dora, meld, and correction-editor tile controls use the reusable `TileFace` / `formatTileIdentity` presentation, preserving red-five and tile-instance semantics while removing ordinary visible canonical codes and winning-tile/session instance IDs.
- The Conditions bottom dock is fixed, safe-area-aware, reserves scroll clearance, reports `ready`, `no-yaku`, `invalid-winning-shape`, `invalid-input`, and `incomplete` previews from the existing scoring service, supports an expandable full-yaku list, and co-locates the readiness-gated `計算する` action.
- Focused component/navigation/Result-origin/E2E assertions cover app-bar Back behavior, visual winning-tile selection, hidden correction hierarchy, persistent preview transitions, safe-area/content-clearance hooks, calculation readiness, and absence of ordinary raw tile/session identifiers.
- Dependency PRODUCT-TASK-SYSTEM-002-09 is now `done`; its Recognition overlay correction establishes the user-facing tile identity direction consumed by this Task.
- Automated verification completed on 2026-08-27: `npx vitest run test/conditions-page.test.tsx test/tile-correction-ui.test.tsx test/navigation-history.test.ts test/result-page.test.tsx test/shell-routing.test.tsx` — **PASS**, 5/5 files and 74/74 tests.
- `npm run build:e2e` — **PASS**; the E2E-mode bundle and fake-flow harness were generated successfully.
- `npx playwright test test/e2e/fake-service-scoring-flow.spec.ts` — **PASS**, 15/15 Chromium cases, including initial Conditions app-bar Back, winning-tile/preview recovery, correction entry, Result-origin condition recalculation/cancel, seat-wind focus, stale-Result prevention, session replacement, and route guards.
- `npm run typecheck` — **PASS**.
- `npm run lint` — **PASS**, `Architecture import boundaries: OK (60 source files checked)`.
- `npm run build` — **PASS**, production PWA build completed with generated service worker/Workbox assets.
- Vite native-config-loader extension notices and the >500 kB application chunk warning are non-blocking warnings for this correction Task.
- F-MAJ-07 was initially closed after the automated gate above; follow-up review found that Conditions/correction presentation still used CSS/text placeholder tile faces instead of the static SVG tile assets already established under `public/tiles` by I09.
- Follow-up correction on 2026-08-27 switches the shared Conditions/Result `TileFace` presentation to the existing `tile-assets` SVG mapping, including dedicated red-five assets. The correction keyboard is restructured into four fixed semantic rows: 萬子 / 筒子 / 索子 each render `1 2 3 4 5 赤5 6 7 8 9`, and 字牌 renders `東 南 西 北 白 發 中`; keyboard/editor tile faces use a larger touch-oriented SVG size while the 14-tile Conditions hand remains compact enough for iPhone width.
- Focused tests now also assert SVG asset usage, red-five asset selection, keyboard row grouping/order, and keyboard-size tile presentation.
- Follow-up verification completed on 2026-08-27 after the SVG/keyboard correction: `npx vitest run test/conditions-page.test.tsx test/tile-correction-ui.test.tsx test/result-page.test.tsx` — **PASS**, 3/3 files and 52/52 tests.
- `npm run typecheck` — **PASS** after the follow-up correction.
- `npm run build` — **PASS** after the follow-up correction. The production output includes the complete tile asset set, including normal suit/honor SVGs, `5m-red.svg` / `5p-red.svg` / `5s-red.svg`, and `back.svg`, confirming that the static tile artwork is now part of the built application rather than only present under `public/tiles`.
- The production PWA precache increased to 47 entries and includes the emitted tile assets; existing Vite native-config-loader and >500 kB chunk notices remain non-blocking warnings.
- F-MAJ-07 is therefore corrected with the intended SVG tile presentation and keyboard hierarchy, automated verification is complete, and this Task is returned to `done`.
- Real-device visual/interaction acceptance remains owned by PRODUCT-TASK-SYSTEM-002-04.
