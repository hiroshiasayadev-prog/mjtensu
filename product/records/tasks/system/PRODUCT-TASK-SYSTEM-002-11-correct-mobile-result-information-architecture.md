# PRODUCT-TASK-SYSTEM-002-11: Correct mobile Result information architecture

- **status**: done
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: correction
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-09
- **finding_refs**:
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-08
- **outputs**:
  - corrected target-device Result mobile presentation
  - compact reusable result tile/yaku/score presentation where required
  - PRODUCT-TASK-SYSTEM-002-11

## Goal

Correct the Result-page mobile information architecture discovered during iPhone 13 acceptance so the final scoring result is readable at a glance as a compact, high-density result surface: tile evidence first, awarded yaku second, score/value third, with correction and restart actions always available but visually secondary to the result itself.

## Work

- Rework Result into three primary visual groups/cards: `牌`, `役`, and `点数`. Keep the groups visually distinct while maintaining compact internal spacing appropriate for a read-only result screen; avoid the large tap-target spacing required by Conditions.
- Reuse the lightweight user-facing tile-face presentation established by I09. Do not expose canonical tile codes such as `4p` / `7z` or recognition/session instance identifiers in ordinary Result presentation.
- In the tile card, place supplied dora indicators above the hand. Render a compact `ドラ` label at the left and the indicator tile faces immediately to its right; do not introduce visible/ura/kan-dora source categories that the current product does not own.
- Give completed hand and meld evidence separate rows for readability.
- Render the completed-hand row with ordinary hand tiles left-aligned and the selected winning-tile instance isolated at the right edge as a clearly distinct result-only winning-tile presentation. Preserve the selected tile instance semantically and preserve the relative ordering of the remaining completed-hand evidence.
- Render the current win method (`ロン` / `ツモ`) immediately below the isolated winning tile so the relation between winning tile and win method is visible without a separate explanatory sentence.
- Render melds on their own row, right-aligned, with each meld kept as a visually separate block. Use tile faces smaller than the completed-hand result tiles, preserving meld-group boundaries and red-five distinction. Avoid unnecessary `ポン` / `チー` text where the grouped tile presentation is sufficient.
- Keep Result tile faces compact enough that the tile evidence card consumes materially less vertical space than an editable Conditions/correction surface. The exact pixel sizes are implementation-owned and must be validated on iPhone 13 rather than copied from desktop defaults.
- Rework the yaku card into a compact aligned list rather than stretching yaku names and han values to opposite viewport edges. Use a bounded/fixed yaku-name column plus a compact awarded-value column so entries scan as one unit.
- Keep textual awarded value (`1翻`, `2翻`, etc.) visible; color must never be the sole carrier of han information.
- Add a thick underline/accent to each yaku row using the accepted awarded-value bands: `1翻` blue, `2翻` green, `3–5翻` yellow, `6翻以上` red, and yakuman-class entries a clearly distinct multicolor/rainbow treatment. Bonus entries such as dora may use the same awarded-han banding rather than requiring a separate neutral scheme.
- Ensure long yaku names remain readable without forcing the awarded-value column to the far viewport edge. Exact truncation/wrapping behavior is implementation-owned but must not hide the awarded yaku identity or value.
- Rework the score card so final points are the strongest numeric element. Show supporting final fu, total han, and limit classification nearby but at lower visual priority.
- Place `符の詳細` adjacent to the displayed fu value when ordinary fu detail is meaningful. Do not expose a fu-detail action for yakuman/non-applicable results or invent a breakdown that the scoring result does not provide.
- Present fu detail as an on-demand mobile bottom sheet (or equivalently compact mobile modal surface) rather than expanding the default Result card. Preserve the current Result/session while open and close back to the unchanged Result.
- The fu-detail surface must continue to present the aggregate categories supplied by scoring: base, closed-ron, tsumo, meld, pair, wait, raw total, rounding, and final fu where applicable; Chiitoitsu must show its fixed 25-fu rule directly.
- Keep the ron/tsumo payment breakdown visible in the score card below or adjacent to the primary point result, including payer distinction for tsumo as required by the score-summary contract.
- Keep the `親` / `子` seat-wind correction shortcut associated with the payment/score area and preserve its existing semantic route to Conditions rather than introducing a separate dealer flag.
- Replace the ordinary lower document action area with a persistent bottom action bar that remains accessible while the result content scrolls. Account for `env(safe-area-inset-bottom)` and reserve content padding so the bar does not obscure the score card.
- Keep `認識結果を修正` and `条件を修正` as secondary actions and `もう一度判定` as the clearest new-hand action. Do not let the persistent action bar visually compete with the primary point result.
- Preserve current Result correction/recalculation semantics: recognition correction, condition correction, seat-wind focus, new-recognition session replacement, atomic score replacement after recalculation, and stale-result protection must remain unchanged.
- Add focused component/browser-flow coverage for tile-card row hierarchy, winning-tile/win-method presentation, meld grouping/size variant, no raw tile codes, yaku alignment and han-band semantics, score priority hooks, fu-detail applicability/open-close behavior, payment presentation, seat-wind shortcut, sticky actions, safe-area/content clearance, and correction/restart routing.
- Return to I04 for target-device verification after implementation; this Task does not mark device acceptance PASS by itself.

## Done condition

F-MAJ-08 is corrected in production code, Result presents a compact mobile hierarchy of tile evidence -> yaku -> score with a dominant final point value, fu detail is available on demand from the fu value without occupying the default layout, correction/restart actions remain persistently accessible, focused automated verification passes, and the updated build is ready for iPhone 13 re-verification under I04.

## Verification

- `npx vitest run test/result-page.test.tsx test/navigation-history.test.ts test/conditions-page.test.tsx`
- Run focused fake-service Playwright cases covering Result, condition correction, recognition correction, seat-wind shortcut, and new-recognition restart.
- `npm run typecheck`
- `npm run lint`
- `npm run build`
- Real-device visual/interaction acceptance remains owned by PRODUCT-TASK-SYSTEM-002-04.

## Evidence

- `spec:product.ui.pages.result` defines Result information priority as recognized hand/dora evidence -> yaku/han -> fu/total han/limit -> final point result -> payment -> correction/restart actions, with final points the primary outcome.
- `spec:product.ui.components.tile_presentation` allows a compact Result-specific tile presentation while requiring winning-tile distinction, dora evidence, red-five identity, and visually preserved meld grouping.
- `spec:product.ui.components.score_summary` requires the final point value to be the strongest numeric element, payment detail to remain visible, the `親` / `子` shortcut to derive from seat wind, and `符の詳細` to be associated with the fu value when applicable.
- `spec:product.ui.components.fu_detail_dialog` requires on-demand fu explanation without occupying the default Result layout and preserves the unchanged Result/session while open.
- PRODUCT-TASK-SYSTEM-002-04 records F-MAJ-08 from target-device review: the current Result does not yet provide the compact result-screen hierarchy needed for efficient visual verification on iPhone 13.
- The accepted mobile direction takes information-density/layout cues from a mahjong game result-screen pattern without requiring pixel-level reproduction: compact tile evidence, dense yaku/value rows, and a strongly emphasized final score.
- PRODUCT-TASK-SYSTEM-002-09 owns the reusable lightweight tile-face representation; I11 consumes it so Result does not create an independent tile-art system. Because I10 and I11 both depend only on I09, Conditions and Result mobile corrections may proceed in parallel after I09.

### Correction implementation: 2026-08-27

- Result presentation has been reworked without changing `src/ui/pages.tsx`, allowing I11 component work to proceed in parallel with I10 route/shell integration. `ResultPage` keeps its existing callback and navigation contract while `ResultPresentation` owns the mobile information architecture.
- Result now consumes the shared `MobileScoringPageShell`, `PersistentBottomBar`, and `TileFace` presentation primitives being shared with I10 instead of creating a separate Result-only app-bar or tile-art implementation.
- The default Result content is now three compact cards in semantic order: `牌`, `役`, `点数`. Dora indicators are shown above the completed hand, the selected winning-tile instance is isolated at the right edge with `ロン` / `ツモ` immediately below it, and meld groups are rendered on a separate right-aligned row using the compact tile-face size.
- Ordinary Result presentation no longer renders canonical tile codes. Remaining hand tiles preserve their relative order after the selected winning instance is separated, meld-group boundaries remain explicit, and red-five identity continues through the shared tile-face primitive. After I10's follow-up tile correction, that shared primitive now renders the static SVG assets from `public/tiles` (including dedicated red-five assets); Result coverage verifies the expected SVG asset mapping.
- Yaku rows now use a bounded name/value grid with textual awarded values and semantic underline bands for 1 han, 2 han, 3-5 han, 6+ han, and yakuman-class entries. Dora and aka-dora bonus rows reuse the same awarded-han bands.
- The score card makes final points the dominant numeric element while keeping fu, total han, limit classification, payment breakdown, and the existing `親` / `子` correction shortcut nearby at lower priority.
- `符の詳細` is adjacent to the displayed fu value only when fu exists. Fu details are now shown in an on-demand bottom sheet with safe-area bottom padding; standard aggregate categories and the fixed 25-fu Chiitoitsu rule are preserved, and closing returns to the unchanged Result.
- Result correction/restart controls now live in a persistent safe-area-aware bottom action bar with secondary correction actions and a clearer `もう一度判定` action. Main content reserves bottom clearance so the bar does not cover the score card.
- Focused Result coverage has been extended for tile-card hierarchy, isolated winning tile and win method, absence of visible raw tile codes, meld grouping, all han-band classes, score-priority hook, fu-sheet close behavior, fixed action bar, safe-area-aware content clearance, and the existing correction/restart routes.
- Focused component verification passed on 2026-08-27: `npx vitest run test/result-page.test.tsx test/navigation-history.test.ts test/conditions-page.test.tsx` -> 3 files / 42 tests passed.
- Static/build gates passed on 2026-08-27: `npm run typecheck`, `npm run lint` (`Architecture import boundaries: OK (60 source files checked)`), and `npm run build`. The build emitted only the existing Vite native-config-loader extension warnings and chunk-size advisory; production build and PWA generation completed successfully.
- Focused fake-service browser verification passed on 2026-08-27: `npm run build:e2e` completed successfully and `npx playwright test test/e2e/fake-service-scoring-flow.spec.ts` -> 15/15 passed. The passing flows include Result condition correction/recalculation and cancel, seat-wind shortcut focus, recognition correction cancel/recalculate/repair fallback, stale-result protection, and explicit new-recognition session replacement.
- I10 is now complete, so I11 owns the final shared-shell integration. `ProductionShell` now suppresses its generic header/padding for both `appRoutePaths.conditions` and `appRoutePaths.result`, preventing the scored Result mobile app bar from being wrapped by the legacy `mjtensu` header/container.
- The unscored Result fallback now also renders inside `MobileScoringPageShell`, preserves the existing route back to non-cancellable Conditions, and no longer exposes `winningTileId` as user-facing text. Shell-routing coverage asserts both Conditions and Result replace the generic production header, while scored Result coverage asserts the legacy `mjtensu` link remains absent.
- Post-I10 verification on 2026-08-27 passed for the focused component/shell suite: `npx vitest run test/result-page.test.tsx test/shell-routing.test.tsx test/navigation-history.test.ts test/conditions-page.test.tsx` -> 4 files / 60 tests passed. `npm run typecheck`, `npm run lint` (`Architecture import boundaries: OK (61 source files checked)`), `npm run build`, and `npm run build:e2e` also passed; the production build emits and precaches the full SVG tile asset set.
- The first post-I10 Playwright rerun passed 14/15 cases. The sole failure was the pre-I11 Help round-trip test waiting for the removed legacy Result `mjtensu` header link; the Result shell behavior itself was therefore correct. That E2E was updated to assert the legacy link is absent and to exercise Help/session preservation through explicit in-harness SPA route transitions instead of relying on the removed header.
- Final post-I10 browser verification passed on 2026-08-27: `npx playwright test test/e2e/fake-service-scoring-flow.spec.ts` -> 15/15 Chromium cases passed, including the updated Help round-trip/session-preservation case without the legacy Result header.
- F-MAJ-08 is corrected in production code, all focused automated verification is green, and PRODUCT-TASK-SYSTEM-002-11 is complete. Real-device visual/interaction acceptance now returns to PRODUCT-TASK-SYSTEM-002-04 for iPhone 13 re-verification.
