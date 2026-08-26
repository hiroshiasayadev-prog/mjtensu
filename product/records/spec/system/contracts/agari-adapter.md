# Contract: Agari scoring adapter

- **id**: `spec:product.system.contracts.agari_adapter`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Internal scoring-infrastructure contract for translating mjtensu's library-independent scoring model to and from the mjtensu-maintained Agari fork described by `spec:product.system.contracts.agari_fork`.

The adapter is private to the `scoring` module. Application, UI, recognition, and domain code consume only `spec:product.system.contracts.scoring_api` and must not depend on Agari Rust/WASM request types, result types, yaku codes, score-level codes, parser notation, or errors.

## Isolation rule

Only private scoring infrastructure may import or instantiate the generated Agari WASM module.

No Agari type may appear in:

- `ScoringService`;
- `ScoringDraft`;
- `ScoringInput`;
- `ScoringPreview`;
- `ScoringCalculation`;
- canonical tile/domain types;
- Application state;
- UI props/state.

The scoring module's public entry point must not export Agari-only request, response, parser, or WASM types.

## Runtime boundary

Agari's score and shape-validation calls are synchronous after the WASM module is initialized.

Any asynchronous WASM asset loading/instantiation is a construction/bootstrap concern. A usable `ScoringService` instance is exposed only after its Agari WASM dependency is ready; `ScoringService.preview()`, `calculate()`, and `validateWinningStructure()` remain synchronous as defined by `spec:product.system.contracts.scoring_api`.

## Evaluation pipeline

Preview and calculation share one scoring-engine evaluation path:

```text
ScoringDraft / ScoringInput
       ↓
product-owned contract validation
       ↓
strict ScoringInput
       ↓
Agari request serialization
       ↓
Agari fork WASM scoring
       ↓
stable Agari response-code normalization
       ↓
ScoringPreview / ScoringCalculation
```

`preview()` may stop before Agari evaluation when required product input is incomplete or contradictory.

`calculate()` accepts strict scoring input and must use the same Agari scoring semantics used by a ready preview.

## Product-owned validation before Agari

The scoring implementation rejects product-contract defects before treating them as Agari scoring outcomes. This includes at least:

- `winningTileId` not identifying a tile in `completedHand`;
- invalid red-five identity (`red === true` on a non-five);
- unresolved or malformed melds reaching strict scoring translation;
- chi/pon/kan member composition that violates the strict meld contract;
- completed-hand/meld counts that cannot represent the supplied logical structure;
- impossible product condition combinations defined by `spec:product.scoring.input`;
- impossible physical tile multiplicity in the supplied scoring structure.

Agari must not become the application's generic input sanitizer.

## Tile serialization

Agari uses tile-kind notation rather than a 136-physical-tile identity model. `TileInstanceId` is therefore never translated into or persisted as an Agari tile identifier.

The adapter serializes each `TileIdentity` as:

| mjtensu | Agari notation |
|---|---|
| `1m..9m` | `1m..9m` |
| `1p..9p` | `1p..9p` |
| `1s..9s` | `1s..9s` |
| `1z..7z` | `1z..7z` |
| red `5m` | `0m` |
| red `5p` | `0p` |
| red `5s` | `0s` |

The selected `winningTileId` is first resolved to its `TileInstance` in `completedHand`; its tile kind is then supplied as Agari's winning tile.

Agari scoring needs the winning tile kind for decomposition/wait semantics, not a persistent physical-copy identifier. Selecting one of two equal tile identities therefore does not require a concrete-library instance allocation. Red-five count remains determined from the serialized whole hand/meld structure, so choosing a red versus ordinary copy of the same five does not alter wait geometry.

## Meld serialization

The strict product meld structure maps to Agari notation as follows:

| mjtensu meld | Agari representation |
|---|---|
| `chi` | open three-tile sequence `(123m)` style |
| `pon` | open three-tile triplet `(555p)` style |
| `open-kan` | open four-tile kan `(5555p)` style |
| `concealed-kan` | concealed four-tile kan `[5555p]` style |

A concealed kan reaches this adapter as four logical identities even when camera recognition originally observed only two face-up members.

The current product does not distinguish daiminkan from kakan. Both serialize as an open kan because current scoring semantics require open-versus-concealed kan status but not call history. Chankan is supplied independently through scoring conditions.

## Dora-indicator mapping

The product intentionally stores one ordered set of all indicator tiles that should count for the winning hand and does not preserve visible/kan/ura/kan-ura source.

Therefore the adapter maps:

```text
ScoringInput.doraIndicators
    -> Agari dora_indicators

Agari ura_dora_indicators
    -> []
```

This is deliberate. Supplying product indicators through Agari's ordinary indicator input prevents the concrete engine from re-applying an ura-dora eligibility rule that the product cannot represent separately. Agari's `regular` and `ura` source distinction is not exposed as product semantics; the normalized product result combines all indicator-derived contribution into `DoraContribution.dora` and keeps only aka dora separate.

## Condition mapping

The adapter maps every strict scoring condition explicitly:

| product | Agari fork request |
|---|---|
| `winMethod === 'tsumo'` | `is_tsumo = true` |
| `winMethod === 'ron'` | `is_tsumo = false` |
| round wind | `round_wind` |
| seat wind | `seat_wind` |
| `riichi` | `is_riichi` / `is_double_riichi` |
| ippatsu | `is_ippatsu` |
| rinshan | `is_rinshan` |
| chankan | `is_chankan` |
| haitei or houtei | `is_last_tile` |
| tenhou | `is_tenhou` |
| chiihou | `is_chiihou` |

Haitei versus houtei remains unambiguous because `winMethod` distinguishes tsumo from ron.

Derived facts such as dealer status and hand openness are not supplied as duplicate product state. Dealer status derives from seat wind. Open/closed state derives from serialized meld structure.

## Rule-profile mapping

Every non-literal field of `ScoringRuleProfile` maps explicitly to the fork's rule config defined by `spec:product.system.contracts.agari_fork`:

```text
openTanyao             -> open_tanyao
akaDora                -> aka_dora
dora                   -> dora
ippatsu                -> ippatsu
kiriageMangan          -> kiriage_mangan
kazoeYakuman           -> kazoe_yakuman
multipleYakuman        -> multiple_yakuman
doubleYakumanVariants  -> double_yakuman_variants
doubleWindPairFu       -> double_wind_pair_fu
```

The current literal product invariants remain asserted by compatibility tests rather than caller-configurable fork fields:

- chiitoitsu = 25 fu;
- pinfu tsumo = 20 fu;
- open pinfu-shaped ron minimum = 30 fu.

The adapter must not rely on an implicit fork default for any mapped rule field.

## Winning-structure validation

`ScoringService.validateWinningStructure()` uses the fork's dedicated winning-shape WASM entry point.

The validation request contains only the logical completed hand and melds. It does not invent win method, winds, riichi state, yaku, dora, or a fake winning tile merely to drive the full score API.

The adapter maps the stable fork result as:

- winning shape -> `valid`;
- coherent but non-winning shape -> `not-winning-shape`;
- adapter-generated malformed request -> scoring adapter failure, because product structural validation should have prevented it.

Application/UI must not maintain a second ordinary/chiitoitsu/kokushi solver.

## Stable response codes only

Adapter control flow must use stable discriminants supplied by the fork, not English error-message matching or display-name parsing.

In particular:

- `not-winning-shape` must not be recognized by matching `"No valid hand structure found"`;
- `no-yaku` must not be recognized by matching `"No valid yaku found"`;
- score level must not be inferred by parsing `"Mangan"`, `"Double Yakuman"`, or similar display strings;
- yaku identity must not use Japanese/English display names as the stable product identity.

Unknown fork discriminants are adapter incompatibility failures.

## Yaku normalization

The fork exposes a stable yaku code for each awarded yaku. The adapter converts those codes to mjtensu-owned `RegularYakuId` / `YakumanYakuId` values.

For regular yaku, the adapter preserves the fork-provided awarded han after open/closed and rule effects. It must not recalculate kuisagari or infer han from the product yaku ID.

Agari's parameterized yakuhai output may occur more than once for a double-value wind. The adapter combines repeated entries with the same normalized regular-yaku ID into one product entry whose `han` is the sum of the awarded han values. This is presentation-safe because the semantic identity is the same while preserving the engine's total awarded han.

Yakuman result entries preserve detected identity only. The adapter does not attach an inferred per-yaku multiplier; the authoritative final multiplier is the normalized `LimitClassification` yakuman `units` value from the fork's structured score level after active rule policy has been applied.

Product-visible yaku names are derived from product yaku IDs by presentation policy rather than copied from Agari display strings.

Dora and aka dora are normalized separately and must not become ordinary `YakuEntry` values.

## Fu normalization

Agari already exposes aggregate fu detail suitable for the current Result UI:

```text
base
menzen_ron
tsumo
melds
pair
wait
raw_total
rounded
```

For an ordinary fu-scored hand, the adapter maps these values directly to `FuCalculation.kind === 'standard'` using `base`, `menzenRon`, `tsumo`, `melds`, `pair`, `wait`, `rawTotal`, and `rounded`. It must not reconstruct per-meld fu contributors that Agari does not expose as separate result entries.

When the awarded hand is Chiitoitsu, the adapter normalizes the engine's fixed 25-fu result to `{ kind: 'chiitoitsu', fixed: 25 }` rather than presenting 25 as an ordinary base-fu contribution.

For yakuman-class outcomes, product output uses `fu: null` even if Agari internally returns a placeholder fu result; product presentation must not imply that ordinary fu affected yakuman payment.

## Limit and payment normalization

The fork exposes structured score-level semantics, including yakuman unit count, counted-yakuman status, and whether mangan was reached through kiriage.

The adapter normalizes this to product `LimitClassification` rather than parsing Agari display text:

- `normal` -> `null`;
- ordinary mangan -> `{ kind: 'mangan', kiriage: false }`;
- kiriage mangan -> `{ kind: 'mangan', kiriage: true }`;
- haneman / baiman / sanbaiman -> the corresponding product discriminant;
- actual yakuman -> `{ kind: 'yakuman', units, counted: false }`;
- counted yakuman -> `{ kind: 'yakuman', units: 1, counted: true }`.

The adapter must not derive yakuman units by dividing han or by counting `YakuEntry` values.

Payments map directly:

- ron -> `from_discarder`;
- dealer tsumo -> equal `from_non_dealer` payment for each opponent;
- non-dealer tsumo -> `from_dealer` and `from_non_dealer`;
- `payment.total` -> `ScoringCalculation.totalPoints`.

mjtensu must not recompute payments from han/fu after Agari has produced the result.

## Preview normalization

After product-owned validation and strict translation:

- fork `not-winning-shape` -> `invalid-winning-shape`;
- fork `no-yaku` -> `no-yaku`;
- scored result with at least one scoring yaku -> `ready` plus normalized yaku.

Dora or aka dora alone never make a no-yaku hand ready.

## Adapter failure boundary

Invalid fork ABI/result shape, unknown yaku codes, unknown score-level codes, impossible response combinations, WASM invocation failure, and other adapter incompatibilities use the scoring runtime-error boundary rather than normal preview states.

Concrete Agari diagnostic strings are retained only for diagnostics and are never presentation-branch inputs.

## Suggested implementation placement

A private implementation may use a shape similar to:

```text
scoring/
  scoring-service.ts
  agari/
    agari-wasm-loader.ts
    agari-scoring-service.ts
    agari-input-adapter.ts
    agari-result-adapter.ts
    agari-yaku-map.ts
```

Exact filenames are implementation-owned.

## Required compatibility tests

The adapter/fork integration test suite must cover at least:

- every tile kind and all three red fives;
- red-five count across completed hand and melds;
- selected winning-tile mapping;
- chi, pon, open-kan, and concealed-kan serialization;
- ordinary, chiitoitsu, and kokushi winning-shape validation;
- each scoring condition;
- all product rule-profile fields;
- all product dora indicators mapped through Agari ordinary-indicator input with Agari ura input empty;
- no-yaku distinct from dora-only bonus;
- every supported fork yaku code mapped to a product yaku identity with awarded regular-yaku han preserved;
- duplicate same-ID yakuhai entries combined by summing awarded han;
- double-value wind behavior;
- fu breakdown normalization;
- normal, kiriage-mangan, haneman, baiman, sanbaiman, counted-yakuman, and actual-yakuman limit normalization as applicable to the supplied profile;
- single, multiple, and double-variant yakuman policy;
- ron, dealer-tsumo, and non-dealer-tsumo payment normalization;
- explicit failure on an unknown fork result/yaku code.

## Boundary

| concern | owner |
|---|---|
| Library-independent scoring API | `spec:product.system.contracts.scoring_api`. |
| Product score input/result semantics | `spec:product.scoring`. |
| Required modifications to upstream Agari | `spec:product.system.contracts.agari_fork`. |
| Concrete Agari translation and result normalization | This contract / scoring infrastructure. |
| Scoring-session orchestration | `spec:product.system.contracts.application_session_api`. |
| Condition normalization for ordinary UI interaction | `spec:product.system.contracts.scoring_condition_policy`. |
