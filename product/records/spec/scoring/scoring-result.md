# Contract: Scoring result

- **id**: `spec:product.scoring.result`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.scoring`

## What this is

Semantic result returned after one valid scoring input is calculated.
The contract contains enough detail for the result page to explain both the awarded yaku and the fu calculation without depending on a scoring-library-specific object shape.

## Result content

A successful result contains:

| concept | requirement |
|---|---|
| Yaku | Ordered or otherwise presentation-ready product yaku identities. Regular yaku carry awarded han; yakuman entries carry detected yakuman identity. Display names are presentation-owned. |
| Dora contribution | Dora from the supplied indicator set and aka-dora contribution must be distinguishable from ordinary yaku. Distinguishing visible, kan, ura, or kan-ura indicator source is not required. |
| Total han | Total regular-yaku plus bonus-dora han for non-actual-yakuman scoring, including counted-yakuman cases when that rule is enabled. |
| Fu | Final rounded fu value when fu is meaningful for the scored hand. |
| Fu detail | Aggregate fu categories from the scoring engine plus the unrounded/rounding relationship needed to explain the final fu value. Seven pairs is represented explicitly as fixed 25 fu. |
| Limit classification | Mangan with kiriage distinction, haneman, baiman, sanbaiman, or yakuman with final unit count and counted/actual distinction when applicable. |
| Winner role | Dealer or non-dealer, derived from seat wind. |
| Win method | Ron or tsumo. |
| Payment detail | The amount paid by the discarder for ron, or dealer/non-dealer payer amounts for tsumo. |
| Hand point total | The score value of the winning hand before excluded table-settlement additions such as honba and riichi sticks. |

## Payment semantics

### Ron

The result must expose the single payment owed by the discarder to the winner.
The result page may present this as the primary point total and need not invent a separate table-settlement total.

### Tsumo

The result must expose the payer breakdown needed to distinguish dealer and non-dealer payments.
Examples of the semantic shape are:

```text
non-dealer winner: non-dealer payment / dealer payment
 dealer winner:    each opponent payment
```

The result must also expose the corresponding winning-hand point total used by the product presentation.
Honba and riichi-stick awards are not included.

## Fu detail semantics

When ordinary fu is meaningful, the result preserves the aggregate categories exposed by the scoring engine:

```text
base fu
closed-ron fu
tsumo fu
meld fu total
pair fu total
wait fu total
raw total
rounded total
```

The result does not require per-meld or per-tile fu contributor reconstruction when the engine exposes only aggregate category totals.

Seven pairs is represented as the semantic fixed-25-fu case rather than ordinary `base = 25` presentation.

For actual or counted yakuman-class outcomes, `fu` is not applicable and must not be presented as though it affected payment. Counted yakuman may still expose its total han; actual yakuman does not use han as the score authority.

## Yaku identity and limit semantics

Yaku identity is stable product semantics rather than a scoring-engine display string. The UI derives Japanese display names from the product yaku identity.

For regular yaku, the result preserves the awarded han after hand-openness and active-rule effects. The UI must not independently apply kuisagari or other han adjustments.

Yakuman yaku entries preserve which yakuman were detected, while the final score multiplier is represented separately by the yakuman limit's `units`. This allows detected identities to remain stable when rule policy changes double-yakuman variants or disables multiple-yakuman stacking.

A counted yakuman, when enabled by the active rule profile, is distinguishable from an actual yakuman. When counted yakuman is disabled, high non-yakuman han remains represented by the applicable ordinary limit such as sanbaiman.

## Dora and indicator traceability

The result presentation must be able to show the supplied dora indicators alongside the calculated result.
Those indicators remain input evidence. Their visible/kan/ura/kan-ura source is intentionally not preserved by the scoring contract.

## Failure boundary

A scoring failure must remain distinct from a successful zero-value result.
Invalid or contradictory scoring input, unsupported scoring structure, and internal scoring-adapter failure must not be presented as a normal calculated result.
The concrete failure taxonomy is deferred until implementation requires separate user-visible recovery paths.

## Non-goals

- Honba and riichi-stick settlement.
- Whole-table score movement and post-hand player totals.
- Rule-engine implementation details.
- Exact Japanese display copy or typography.

## Boundary

| concern | owner |
|---|---|
| Semantic scoring result | This contract. |
| Result-page hierarchy and visible placement | `spec:product.ui.pages.result`. |
| Fu-detail modal presentation | `spec:product.ui.components.fu_detail_dialog`. |
| Recalculation after edits | `spec:product.application.scoring_session`. |
| Concrete scoring-library return mapping | Implementation. |
