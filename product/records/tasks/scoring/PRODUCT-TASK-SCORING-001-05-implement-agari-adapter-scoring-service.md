# PRODUCT-TASK-SCORING-001-05: Implement Agari adapter and ScoringService

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SCORING-001
- **task_type**: implementation
- **estimate**: 2d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
  - PRODUCT-TASK-SCORING-001-03
- **outputs**:
  - production TypeScript scoring implementation
  - PRODUCT-TASK-SCORING-001-05

## Goal

Implement the production TypeScript WASM loader, Agari input/result adapter, and public synchronous ScoringService without leaking concrete Agari types outside the scoring module.

## Work

- Load/instantiate the production Agari WASM according to the build-management decision from S01.
- Serialize canonical product tiles, red fives, melds, dora indicators, conditions, and explicit rule profile into the stable fork request.
- Implement scoring-independent winning-structure validation through the dedicated fork API.
- Normalize stable fork yaku codes into product `RegularYakuId` / `YakumanYakuId` values.
- Preserve awarded regular-yaku han, combine duplicate same-ID yakuhai entries by summing awarded han, and keep yakuman multiplier authority in `LimitClassification.units`.
- Normalize aggregate fu, chiitoitsu fixed 25 fu, limit classifications, dora/aka counts, and ron/tsumo payments exactly as the scoring contracts specify.
- Map not-winning-shape/no-yaku/invalid-request/internal failures into the product preview/error boundary without display-string parsing.
- Keep `preview()` and `calculate()` on one shared concrete evaluation path after strict input construction.
- Add focused TypeScript adapter/service tests using stable WASM response fixtures in addition to later real-WASM S06 verification.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| WASM loader boundary | Initialize the generated Agari WASM and expose a ready synchronous scoring dependency to ScoringService. | ScoringService public operations remain synchronous after construction; raw WASM types stay private. | Loader/service contract tests and architecture checks. |
| input adapter | Serialize all product tile/meld/condition/rule semantics exactly as `agari-adapter` specifies, including one indicator set -> ordinary Agari indicators and empty ura list. | Exhaustive mapping tests cover tile kinds/red fives, all meld kinds, all conditions, and every rule-profile field. | Table-driven adapter tests. |
| structure validation | Use the dedicated fork shape API after product-owned structural validation. | Ordinary/chiitoitsu/kokushi/non-winning outcomes map to the declared `WinningStructureValidation` states without fake conditions. | Focused validation tests. |
| yaku/fu/limit/dora normalization | Convert stable machine results into product-owned `YakuEntry`, `FuCalculation`, `LimitClassification`, and `DoraContribution`. | No display-name parsing, kuisagari recomputation, or han-derived yakuman multiplier inference occurs in TypeScript. | Exhaustive result-adapter fixture tests. |
| payment/result normalization | Map Agari payment totals to product ron/dealer-tsumo/non-dealer-tsumo results without recalculation. | Known response fixtures produce exact product payment and totalPoints values. | Payment mapping tests. |
| preview/calculate path | Share one concrete evaluation path for scoring-ready preview and final calculation. | The same strict input/profile cannot produce divergent yaku semantics between preview and calculate. | Service-level parity tests. |

## Done condition

The production TypeScript scoring module fully satisfies the Scoring API and Agari adapter contracts with focused mapping/service tests passing and no concrete Agari/WASM types exposed through public cross-module APIs.

## Verification

- Run exhaustive input/result mapping tests.
- Run winning-structure validation tests.
- Run preview/calculate parity tests.
- Run scoring runtime-error mapping tests.
- Run strict typecheck/lint/architecture checks.
- Defer full real-WASM golden corpus execution to S06.

## Evidence

- `src/scoring/agari/` now owns the private stable V1 ABI model, product-to-Agari input adapter, Agari-to-product result adapter, synchronous ScoringService implementation, and asynchronous WASM initialization boundary.
- `src/scoring/index.ts` exposes only the product-owned Scoring API plus `loadProductionScoringService()`; no concrete `Agari*` / raw WASM type is exported through the public scoring entry point.
- Input mapping covers canonical tiles/red fives, chi/pon/open-kan/concealed-kan notation, the one product dora-indicator set -> ordinary Agari indicators plus empty ura list, every scoring condition, and every explicit fork rule field.
- Product-owned validation now rejects invalid meld composition, invalid red-five identity, impossible tile multiplicity, contradictory conditions, and invalid winning-tile selection before a deterministic score request can reach Agari.
- Stable yaku codes are normalized without display-string parsing; awarded regular-yaku han is preserved, duplicate same-ID regular entries are aggregated, and actual-yakuman multiplier authority remains `score_level.units`.
- Fu, chiitoitsu fixed 25 fu, dora/aka contribution, structured limits, and raw Agari ron/tsumo payment fields are normalized without TypeScript score/payment recalculation.
- Focused fixture tests were added in `test/agari-input-adapter.test.ts`, `test/agari-result-adapter.test.ts`, and `test/agari-scoring-service.test.ts` with shared fixtures in `test/agari-test-fixtures.ts`.
- During implementation, `DEFAULT_RULE_PROFILE` was corrected to match `spec:product.scoring.input`: kazoe yakuman disabled, double-yakuman variants disabled, and double-wind pair fu set to 2.
- Focused verification on 2026-08-27 passed: `npm test -- agari-input-adapter.test.ts agari-result-adapter.test.ts agari-scoring-service.test.ts correction-draft-service.test.ts tile-correction-ui.test.tsx` completed 5 test files / 48 tests with 0 failures.
- `npm run typecheck` passed after fixing Vitest mock literal widening in `agari-scoring-service.test.ts`.
- `npm run lint` passed with `Architecture import boundaries: OK (47 source files checked)`.
- The committed production `vendor/agari-wasm/` package required by S01 is not currently present in the mjtensu checkout, so real production-loader/WASM execution is intentionally deferred to S06 together with the full golden corpus compatibility gate.
- `spec:product.system.contracts.agari_adapter` is the concrete translation authority.
- `spec:product.system.contracts.scoring_api` is the public library-independent authority.
- S03 supplies the stable machine ABI consumed here.
