# PRODUCT-TASK-UI-001-05: Implement Result page

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: implementation
- **estimate**: 1.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **outputs**:
  - production Result page implementation
  - PRODUCT-TASK-UI-001-05

## Goal

Implement the production Result page score hierarchy, yaku/han, fu detail, payment, evidence tiles, and correction/restart actions from product-owned scoring result types.

## Work

- Render recognized winning structure and mark the selected winning tile.
- Render yaku names from product `YakuId` presentation mapping and awarded regular-yaku han from ScoringCalculation.
- Render aggregate indicator dora and aka dora separately from ordinary yaku.
- Render final fu/han/limit hierarchy and final points without inventing han/fu for actual yakuman cases.
- Render ron, dealer-tsumo, and non-dealer-tsumo payment breakdown from `ScoringPayment`.
- Implement the fu-detail dialog using aggregate engine-provided categories and the explicit chiitoitsu fixed-25 case.
- Implement compact 親/子 status action by routing to Conditions focused on seat wind rather than creating a second dealer state.
- Implement recognition correction, condition correction, and explicit new-recognition actions.
- Ensure recalculation atomically replaces stale result presentation.
- Add focused result/presentation/action tests using deterministic ScoringCalculation fixtures.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| yaku/dora presentation | Render product yaku identities with awarded regular-yaku han and separate dora/aka contribution. | UI uses product presentation mapping, not Agari display strings; it does not recalculate kuisagari or yakuman multipliers. | Semantic result fixture tests. |
| score summary | Render fu/han/limit/final points and dealer/child status according to result semantics. | Non-limit, kiriage mangan, other limits, counted yakuman, and actual yakuman fixtures display without fake values. | Score-summary matrix tests. |
| payment | Render ron and both tsumo payment shapes directly from ScoringPayment. | Displayed payer amounts equal the supplied product result fields. | Payment fixture tests. |
| fu detail | Render aggregate base/menzen-ron/tsumo/meld/pair/wait/raw/rounded fields or chiitoitsu fixed 25. | No per-meld fu reconstruction is invented; yakuman-class result does not expose misleading fu detail. | Fu-dialog fixture tests. |
| correction/restart actions | Route to condition correction, recognition correction, or explicit new recognition while preserving/discarding session state as specified. | No stale score remains visible after accepted recalculation or new-recognition reset. | Action/router tests. |

## Done condition

The Result page and reusable score/yaku/fu/payment components present all accepted product result variants and correction/restart actions correctly with focused deterministic tests passing.

## Verification

- Run yaku/dora presentation tests.
- Run score-limit/payment fixture matrix tests.
- Run fu-detail ordinary/chiitoitsu/yakuman tests.
- Run correction/status/new-recognition action tests.
- Run strict typecheck/lint/architecture checks.

## Evidence

- `spec:product.ui.pages.result`, yaku-list, score-summary, and fu-detail Specifications define the visible result contract.
- `spec:product.scoring.result` and `spec:product.system.contracts.scoring_api` define the semantic data consumed here.
- Execution results are recorded here when the Task is performed.
