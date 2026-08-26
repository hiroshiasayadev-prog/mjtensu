# Contract: mjtensu Agari fork

- **id**: `spec:product.system.contracts.agari_fork`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Implementation-facing contract for the mjtensu-maintained fork of `agari-industries/agari` used as the production riichi scoring engine.

The upstream project already provides the required hand decomposition, ordinary/situational yaku detection, red-five counting, dora counting, fu calculation, yakuman stacking representation, payment calculation, and browser-targetable WASM wrapper. The fork exists only to make product rule policy and the browser ABI explicit where upstream behavior is currently fixed or stringly typed.

This contract defines the required semantic delta from upstream. It does not authorize unrelated rewrites of Agari's decomposition or scoring algorithms.

## Source and artifact management

The canonical mjtensu Agari fork source lives in a separate Git repository derived from `agari-industries/agari`.

mjtensu does not vendor the fork's Rust source tree and does not use a Git submodule for the fork. The fork must retain upstream MIT licensing and attribution.

### Revision authority

Every production Agari artifact records two exact full Git commit SHAs:

- the upstream Agari commit that forms the selected fork base;
- the mjtensu fork commit from which the production WASM package was built.

A branch name, moving tag, release name, or upstream `main` is not a production revision authority. Human-facing tags may exist, but the full commit SHAs remain authoritative.

Production builds must not silently track upstream changes.

### Generated WASM consumption

The fork's browser package is built explicitly from the selected fork revision in release mode and the generated package is committed into mjtensu under `vendor/agari-wasm/`.

The committed package contains the generated JavaScript glue, WASM binary, generated type declarations required by the frontend, and its provenance manifest.

Ordinary mjtensu frontend development, test, and production-build commands consume this committed package directly. They must not require Rust, Cargo, `wasm-pack`, a checkout of the Agari fork, or network access to rebuild the scoring engine.

Refreshing the Agari artifact is a separate explicit maintenance workflow. The concrete command name is implementation-owned, but the workflow must:

1. select and check out the exact fork commit;
2. verify the recorded upstream-base commit relationship;
3. run the upstream-compatible and mjtensu fork regression tests required by this contract;
4. build the browser package in release mode;
5. replace the committed generated package and provenance manifest together;
6. run the mjtensu Agari ABI/golden compatibility gate before the new artifact is accepted as production input.

### Provenance manifest

`vendor/agari-wasm/` includes one machine-readable provenance manifest equivalent to:

```ts
interface AgariArtifactProvenanceV1 {
  readonly schemaVersion: 1;
  readonly upstreamRepository: 'https://github.com/agari-industries/agari';
  readonly upstreamCommit: string; // exact full Git SHA
  readonly forkRepository: string;
  readonly forkCommit: string; // exact full Git SHA
  readonly abiVersion: string;
  readonly wasmSha256: string;
  readonly wasmBytes: number;
  readonly rustcVersion: string;
  readonly wasmPackVersion: string;
  readonly buildProfile: 'release';
  readonly generatedAt?: string;
}
```

The concrete file name is implementation-owned. `upstreamCommit` and `forkCommit` must be full commit SHAs rather than branch/tag names.

The production scoring artifact identity is determined by the fork commit, stable ABI version, and WASM SHA-256. `generatedAt`, when retained, is evidence only and must not make an otherwise identical artifact a different semantic version.

The frontend/runtime must not infer product rule semantics from this provenance manifest. It is build and integrity evidence, not the scoring request contract.

### Upstream upgrade policy

Upstream updates are explicit maintenance changes, never automatic production upgrades.

An upstream merge/rebase or other base-revision change is acceptable only after the selected fork revision passes:

- the applicable upstream Agari tests;
- the mjtensu fork regression tests required below;
- the stable WASM ABI checks;
- the mjtensu scoring golden corpus through the production adapter path.

A failed compatibility gate leaves the previous pinned fork revision and committed generated package as the production authority.

## Keep upstream scoring responsibilities

The fork continues to delegate these responsibilities to Agari core rather than reimplementing them in TypeScript:

- ordinary 4-meld-1-pair decomposition;
- chiitoitsu and kokushi decomposition;
- winning-tile wait interpretation;
- ordinary yaku detection;
- riichi/double-riichi/ippatsu/menzen-tsumo and situational yaku detection;
- open/closed meld handling;
- red-five counting;
- indicator-derived dora counting;
- fu calculation;
- score-limit/base-point calculation;
- ron/tsumo payment calculation;
- selection of the highest-paying valid interpretation.

The fork must preserve upstream behavior for these areas unless a product rule field below explicitly changes it.

## Fork rule configuration

Introduce one explicit rule configuration consumed by yaku, dora, fu, limit, and payment evaluation where applicable.

Conceptually:

```rust
pub struct RuleConfig {
    pub open_tanyao: bool,
    pub aka_dora: bool,
    pub dora: bool,
    pub ippatsu: bool,
    pub kiriage_mangan: bool,
    pub kazoe_yakuman: bool,
    pub multiple_yakuman: bool,
    pub double_yakuman_variants: bool,
    pub double_wind_pair_fu: u8, // exactly 2 or 4
}
```

The concrete Rust location/name may differ, but one coherent config object must reach every affected calculation. Rule switches must not be duplicated as unrelated globals or WASM-only post-processing.

The WASM score request includes this rule config explicitly. No product-significant rule relies on an implicit fork default when called from mjtensu.

## Rule semantics

### Open tanyao

When `open_tanyao == false`, an otherwise-open hand must not receive Tanyao. Closed Tanyao remains valid.

When `open_tanyao == true`, preserve upstream open-Tanyao behavior.

### Aka dora

When `aka_dora == false`, recognized red fives remain valid tile identities for parsing/shape purposes but contribute zero aka-dora han.

When `aka_dora == true`, preserve upstream aka-dora counting.

### Indicator dora

When `dora == false`, ordinary and ura indicator-derived dora contribute zero bonus han.

When `dora == true`, preserve upstream indicator successor/counting behavior.

mjtensu currently maps its one undifferentiated indicator set to the ordinary Agari indicator input and supplies no Agari ura indicators; this product mapping is owned by `spec:product.system.contracts.agari_adapter`.

### Ippatsu rule

When `ippatsu == false`, `is_ippatsu` in game context must not award the Ippatsu yaku.

When `ippatsu == true`, preserve upstream Ippatsu behavior and prerequisites.

### Kiriage mangan

When `kiriage_mangan == true`, the following otherwise-non-mangan scores are treated as mangan:

- 4 han 30 fu;
- 3 han 60 fu.

The structured score result must indicate that mangan was reached by kiriage rather than ordinary mangan thresholds.

When `kiriage_mangan == false`, preserve upstream ordinary thresholds.

### Kazoe yakuman

For non-yakuman hands:

- when `kazoe_yakuman == true`, 13+ total han scores as one counted yakuman;
- when `kazoe_yakuman == false`, 13+ total han is capped at sanbaiman.

Counted yakuman never stacks merely because total non-yakuman han reaches 26 or more.

Actual yakuman yaku are unaffected by this switch.

### Double-yakuman variants

Upstream represents these as double-yakuman equivalents:

- Daisuushii;
- Kokushi 13-sided wait;
- Suuankou tanki;
- Junsei Chuuren Poutou.

The fork preserves their distinct detected yaku identity but makes their score contribution rule-aware:

- `double_yakuman_variants == true` -> each contributes 2 yakuman units;
- `double_yakuman_variants == false` -> each contributes 1 yakuman unit.

The yaku identity must not be collapsed merely because its multiplier is configured to one.

### Multiple yakuman

For actual yakuman yaku:

- when `multiple_yakuman == true`, sum the configured yakuman units of all simultaneously awarded yakuman;
- when `multiple_yakuman == false`, score at one yakuman unit total while still permitting the result to report all detected yakuman identities.

`multiple_yakuman` is independent of `double_yakuman_variants`.

### Double-wind pair fu

For a pair of the same wind that is both round wind and seat wind:

- `double_wind_pair_fu == 2` -> award 2 pair fu total;
- `double_wind_pair_fu == 4` -> award 4 pair fu total.

A wind matching only one of round or seat remains 2 fu. Dragon pairs remain 2 fu.

Any other value for `double_wind_pair_fu` is an invalid rule configuration.

## Yakuman-unit calculation

The fork must not derive final actual-yakuman multiplier solely from a convention such as `total_han / 13` after rule application.

Introduce rule-aware yakuman-unit evaluation so that detected identity and score multiplier remain separable. Conceptually:

```text
Yaku identity
  -> base yakuman variant identity
  -> configured unit contribution (1 or 2)
  -> multiple-yakuman aggregation policy
  -> final actual yakuman units
```

This is required because upstream currently encodes some double variants as 26 han equivalents, while mjtensu may intentionally score the same detected variant as one yakuman.

Normal-yaku han and counted-yakuman logic remain separate from actual-yakuman unit aggregation.

## Preserve fixed product-compatible fu rules

The current product contract fixes these values and does not require additional fork switches:

- chiitoitsu: exactly 25 fu;
- pinfu tsumo: exactly 20 fu;
- open pinfu-shaped ron/no-extra-fu hand: 30 fu minimum.

The fork must preserve the upstream behavior that matches these values and include regression tests for them.

If the product later makes any of these values configurable, that requires an explicit extension of this fork contract and rule config.

## Stable WASM scoring ABI

The production adapter must not branch on Agari display strings. The fork therefore exposes a stable tagged WASM scoring API in addition to or instead of the upstream stringly response.

Conceptually:

```ts
interface AgariScoreRequestV1 {
  readonly hand: string;
  readonly winning_tile: string;
  readonly is_tsumo: boolean;
  readonly is_riichi: boolean;
  readonly is_double_riichi: boolean;
  readonly is_ippatsu: boolean;
  readonly round_wind: string;
  readonly seat_wind: string;
  readonly dora_indicators: readonly string[];
  readonly ura_dora_indicators: readonly string[];
  readonly is_last_tile: boolean;
  readonly is_rinshan: boolean;
  readonly is_chankan: boolean;
  readonly is_tenhou: boolean;
  readonly is_chiihou: boolean;
  readonly rules: AgariRuleConfigV1;
}
```

The exact generated TypeScript shape is implementation-owned, but the following semantic guarantees are required.

### Tagged score outcome

The score function returns stable outcome discriminants equivalent to:

```text
scored
not-winning-shape
no-yaku
invalid-request
internal-error
```

Human-readable diagnostics may accompany failures, but product/application control flow must never require matching those strings.

### Stable yaku code and awarded han

Every awarded yaku entry includes a stable machine code separate from its display name.

Examples are conceptually:

```text
riichi
double-riichi
ippatsu
menzen-tsumo
tanyao
pinfu
...
kokushi-musou
kokushi-13-wait
suuankou
suuankou-tanki
...
```

Parameterized Yakuhai must preserve the honor identity in its code or structured payload. The fork may report duplicate yakuhai entries for a double-value wind, matching the two awarded han.

Every **regular** yaku entry also exposes the awarded han after hand-openness and active-rule effects have been applied. The TypeScript adapter must not need to recalculate kuisagari or otherwise infer the awarded han from yaku identity.

Yakuman entries expose their detected stable identity but do not need to carry the final product yakuman multiplier individually. Rule-aware actual-yakuman aggregation remains authoritative in the structured score level.

Display strings are optional presentation/debug fields and are not stable identifiers.

### Structured score level

The successful result exposes a structured score-level representation, not only `ScoreLevel::name()` text.

It must distinguish at least:

```text
normal
mangan { kiriage: boolean }
haneman
baiman
sanbaiman
yakuman { units: positive integer, counted: boolean }
```

A counted yakuman, when enabled, is distinguishable from actual yakuman.

### Fu breakdown

Preserve the existing aggregate fu breakdown across WASM:

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

Do not invent per-meld contributor detail solely for mjtensu.

### Payment

Preserve structured payment values:

```text
total
from_discarder
from_dealer
from_non_dealer
```

The adapter must not need to recalculate payment from han/fu.

## Winning-shape WASM API

Add a dedicated scoring-independent shape-validation entry point for correction use.

Conceptually:

```text
validate_winning_shape_v1(hand)
  -> winning
   | not-winning-shape
   | invalid-request
   | internal-error
```

This function parses the supplied closed tiles/meld notation and uses Agari decomposition to determine whether it represents a supported completed hand shape.

It must not require:

- win method;
- round/seat wind;
- winning tile;
- riichi/situational conditions;
- dora indicators;
- rule profile;
- existence of a scoring yaku.

Ordinary, chiitoitsu, and kokushi completed shapes are supported according to the same decomposition implementation used by scoring.

## Error taxonomy at the fork boundary

Expected semantic outcomes such as `not-winning-shape` and `no-yaku` are stable result variants, not generic Rust/WASM failures.

Malformed serialized requests and parser failures return `invalid-request` with a diagnostic code/message suitable for infrastructure logging.
Unexpected internal failures return `internal-error`.

The TypeScript adapter converts `invalid-request` from an adapter-generated request into an adapter/runtime failure because product validation and deterministic serialization should prevent it during normal operation.

## Upstream compatibility discipline

Changes to upstream Agari are kept narrow and isolated around:

- rule configuration;
- rule-aware yaku/dora/fu/limit behavior;
- actual-yakuman unit aggregation;
- stable WASM request/result codes;
- dedicated winning-shape validation.

Do not fork the hand decomposition algorithm or duplicate a TypeScript scoring engine unless later evidence demonstrates an upstream correctness problem that cannot be fixed locally.

Every upstream merge/rebase must pass both upstream tests and the mjtensu compatibility suite.

## Required fork tests

The fork must add regression tests for at least:

- open Tanyao enabled and disabled;
- aka dora enabled and disabled;
- indicator dora enabled and disabled;
- Ippatsu enabled and disabled;
- 4 han 30 fu kiriage on/off;
- 3 han 60 fu kiriage on/off;
- 13+ non-yakuman han with kazoe on/off;
- counted yakuman remaining one unit even above 26 non-yakuman han;
- every upstream double-yakuman variant under both double-variant policies;
- multiple simultaneous actual yakuman with multiple-yakuman on/off;
- interaction of multiple-yakuman and double-variant policy;
- double-wind pair 2/4 fu;
- chiitoitsu 25 fu;
- pinfu tsumo 20 fu;
- open no-extra-fu minimum 30 fu;
- stable structured score-level result;
- stable yaku codes and awarded regular-yaku han;
- stable `not-winning-shape` versus `no-yaku` outcomes;
- winning-shape validation for ordinary, chiitoitsu, kokushi, and non-winning hands;
- red-five and dora breakdown preservation;
- ron/dealer-tsumo/non-dealer-tsumo payment preservation.

## Boundary

| concern | owner |
|---|---|
| Product rule semantics | `spec:product.scoring.input` and `spec:product.system.contracts.scoring_api`. |
| Required Agari fork semantic delta | This contract. |
| Product-to-fork serialization and normalization | `spec:product.system.contracts.agari_adapter`. |
| Fork source code | Separate mjtensu-maintained Agari Git repository. |
| Source revision pinning, generated WASM package, provenance, and upgrade discipline | This contract. |
| Concrete artifact-refresh script/command implementation | Scoring infrastructure / production build. |
| UI presentation | Product UI specs. |
