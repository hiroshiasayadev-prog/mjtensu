# PRODUCT-TASK-SCORING-001-05: Implement Agari adapter and ScoringService

- **status**: not_started
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

- `spec:product.system.contracts.agari_adapter` is the concrete translation authority.
- `spec:product.system.contracts.scoring_api` is the public library-independent authority.
- S03 supplies the stable machine ABI consumed here.
- Execution results are recorded here when the Task is performed.
