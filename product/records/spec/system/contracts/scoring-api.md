# Contract: Scoring API

- **id**: `spec:product.system.contracts.scoring_api`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Library-independent implementation contract between Application and the riichi-mahjong scoring module.

The scoring module owns translation to/from the concrete scoring library. Application and UI consume only the types in this contract and the canonical system/domain model.

## Service boundary

```ts
export interface ScoringService {
  validateWinningStructure(
    structure: RecognizedStructure,
  ): WinningStructureValidation;

  preview(
    draft: ScoringDraft,
    ruleProfile: ScoringRuleProfile,
  ): ScoringPreview;

  calculate(
    input: ScoringInput,
    ruleProfile: ScoringRuleProfile,
  ): ScoringCalculation;
}
```

All operations are synchronous. The current scoring engine is local pure computation; the public contract must not introduce asynchronous orchestration without a later system decision.

`validateWinningStructure()` is deliberately independent of scoring conditions, winning-tile selection, yaku, fu, and rule-profile configuration. It exists so the correction editor can determine whether the edited completed-hand/meld structure is a supported completed winning hand before allowing the user to leave structural correction.

`ruleProfile` is an explicit argument to scoring operations even though the current product flow always supplies `DEFAULT_RULE_PROFILE`. This keeps scoring policy visible at the call boundary and permits later rule-profile selection without changing the service signature.

## Winning-structure validation

```ts
export type WinningStructureValidation =
  | {
      readonly kind: 'invalid-structure';
      readonly issues: readonly WinningStructureIssue[];
    }
  | {
      readonly kind: 'not-winning-shape';
    }
  | {
      readonly kind: 'valid';
    };
```

`invalid-structure` covers product-owned structural problems that prevent a meaningful winning-shape query, such as an unresolved or malformed meld group or a completed-hand count inconsistent with the logical meld count. `WinningStructureIssue` must retain enough semantic location information for the correction UI to associate an issue with the completed-hand region or a specific meld group without exposing concrete scoring-library errors.

`not-winning-shape` means the tile/meld structure is structurally coherent but does not form a supported completed winning hand. `valid` means the structure is a supported completed winning shape, including supported closed special forms such as seven pairs or thirteen orphans.

Dora indicators do not participate in winning-shape validity. Lack of yaku does not make a winning structure invalid.

The scoring adapter owns the actual mahjong winning-shape/decomposition query through the production Agari fork. Application and UI must not maintain a second four-meld/seven-pairs/thirteen-orphans solver.

## Draft input

`ScoringDraft` represents the current Conditions/application state before it is necessarily calculation-ready.

```ts
export interface ScoringDraft {
  readonly structure: RecognizedStructure;
  readonly winningTileId: TileInstanceId;
  readonly conditions: ScoringConditionsDraft;
}
```

`RecognizedStructure` is intentionally permissive and may still contain `unresolved` recognition meld groups that require correction.

`winningTileId` identifies the currently selected completed-hand tile instance. How Application initializes or changes that selection is outside the scoring module.

### Draft conditions

```ts
export type Wind = 'east' | 'south' | 'west' | 'north';
export type WinMethod = 'ron' | 'tsumo';
export type RiichiState = 'none' | 'riichi' | 'double-riichi';

export interface ScoringConditionsDraft {
  readonly winMethod: WinMethod | null;
  readonly roundWind: Wind | null;
  readonly seatWind: Wind | null;

  readonly riichi: RiichiState;
  readonly ippatsu: boolean;

  readonly rinshan: boolean;
  readonly chankan: boolean;
  readonly haitei: boolean;
  readonly houtei: boolean;
  readonly tenhou: boolean;
  readonly chiihou: boolean;
}
```

Only values that can genuinely be absent while Conditions input is incomplete are nullable. Boolean situational conditions use `false` for not selected rather than absence, and the contract does not use a blanket `Partial<>` representation.

## Preview result

```ts
export type ScoringPreview =
  | {
      readonly kind: 'incomplete';
      readonly missing: readonly ScoringRequiredField[];
    }
  | {
      readonly kind: 'invalid-input';
      readonly issues: readonly ScoringInputIssue[];
    }
  | {
      readonly kind: 'invalid-winning-shape';
    }
  | {
      readonly kind: 'no-yaku';
    }
  | {
      readonly kind: 'ready';
      readonly yaku: readonly YakuEntry[];
    };
```

Preview normalizes two different classes of validation without exposing concrete scoring-library types.

### Product/input-owned outcomes

The scoring boundary returns `incomplete` or `invalid-input` before asking the concrete scoring engine to solve a mahjong scoring problem when the draft cannot be converted into a coherent strict scoring input.

Examples include:

- a required ordinary condition is not yet supplied;
- a recognition meld remains `unresolved`;
- `ippatsu` is selected without riichi/double-riichi;
- `rinshan` is selected for ron;
- `chankan` is selected for tsumo;
- `haitei`/`houtei`, `tenhou`/`chiihou`, or other situational facts contradict the supplied win/dealer context;
- the selected `winningTileId` does not refer to a completed-hand tile in the supplied structure;
- the draft cannot be translated into the strict logical meld/input model required by scoring.

The exact discriminants of `ScoringRequiredField` and `ScoringInputIssue` must remain product-semantic values rather than concrete library error codes.

### Scoring-engine-owned outcomes

Once the draft can be translated into a coherent strict scoring input, the concrete scoring engine owns mahjong-rule evaluation including:

- whether the supplied tiles form a supported winning shape;
- decomposition of ambiguous ordinary hands;
- winning-tile placement within candidate decompositions and corresponding wait interpretation;
- yaku evaluation;
- fu evaluation;
- han/limit/point/payment calculation;
- dora and aka-dora contribution under the supplied rule profile.

The adapter normalizes those engine results into:

- `invalid-winning-shape` when no supported winning decomposition exists;
- `no-yaku` when a winning shape exists but no scoring yaku is awarded;
- `ready` when the current draft can be scored and has at least one scoring yaku.

Dora/aka-dora alone must not convert a no-yaku hand into `ready`. The adapter must distinguish ordinary/scoring yaku from bonus-dora contribution using the concrete engine result rather than reimplementing yaku detection in Application or UI.

## Strict scoring input

`calculate()` accepts a strict scoring representation rather than the permissive recognition draft.

```ts
export interface ScoringInput {
  readonly completedHand: readonly TileInstance[];
  readonly melds: readonly ScoringMeld[];
  readonly doraIndicators: readonly TileIdentity[];
  readonly winningTileId: TileInstanceId;
  readonly conditions: ScoringConditions;
}
```

### Strict conditions

```ts
export interface ScoringConditions {
  readonly winMethod: WinMethod;
  readonly roundWind: Wind;
  readonly seatWind: Wind;

  readonly riichi: RiichiState;
  readonly ippatsu: boolean;

  readonly rinshan: boolean;
  readonly chankan: boolean;
  readonly haitei: boolean;
  readonly houtei: boolean;
  readonly tenhou: boolean;
  readonly chiihou: boolean;
}
```

Derived facts such as dealer status, closed-hand status, menzen-tsumo status, ordinary composition yaku, dora count, and aka-dora count are not duplicated as independent input fields.

### Strict melds

```ts
export type ScoringMeld =
  | {
      readonly kind: 'chi';
      readonly tiles: readonly [
        TileIdentity,
        TileIdentity,
        TileIdentity,
      ];
    }
  | {
      readonly kind: 'pon';
      readonly tiles: readonly [
        TileIdentity,
        TileIdentity,
        TileIdentity,
      ];
    }
  | {
      readonly kind: 'open-kan';
      readonly tiles: readonly [
        TileIdentity,
        TileIdentity,
        TileIdentity,
        TileIdentity,
      ];
    }
  | {
      readonly kind: 'concealed-kan';
      readonly tiles: readonly [
        TileIdentity,
        TileIdentity,
        TileIdentity,
        TileIdentity,
      ];
    };
```

`unresolved` is not representable in `ScoringMeld`.
A concealed kan is already represented as its full four-member logical meld at this boundary even when recognition originally observed only two face-up tiles.

## Rule profile

```ts
export interface ScoringRuleProfile {
  readonly openTanyao: boolean;
  readonly akaDora: boolean;
  readonly dora: boolean;
  readonly ippatsu: boolean;
  readonly kiriageMangan: boolean;
  readonly kazoeYakuman: boolean;
  readonly multipleYakuman: boolean;
  readonly doubleYakumanVariants: boolean;
  readonly doubleWindPairFu: 2 | 4;
  readonly chiitoitsuFu: 25;
  readonly pinfuTsumoFu: 20;
  readonly openPinfuRonMinimumFu: 30;
}

export const DEFAULT_RULE_PROFILE: ScoringRuleProfile;
```

The concrete adapter maps every profile field explicitly and must not silently inherit concrete-library defaults for product-significant rule semantics.

## Successful calculation

```ts
export interface ScoringCalculation {
  readonly yaku: readonly YakuEntry[];
  readonly dora: DoraContribution;

  readonly han: number | null;
  readonly fu: FuCalculation | null;
  readonly limit: LimitClassification | null;

  readonly winnerRole: 'dealer' | 'non-dealer';
  readonly winMethod: WinMethod;

  readonly payment: ScoringPayment;
  readonly totalPoints: number;
}

export type RegularYakuId =
  | 'riichi'
  | 'double-riichi'
  | 'ippatsu'
  | 'menzen-tsumo'
  | 'tanyao'
  | 'pinfu'
  | 'iipeikou'
  | 'yakuhai-east'
  | 'yakuhai-south'
  | 'yakuhai-west'
  | 'yakuhai-north'
  | 'yakuhai-white'
  | 'yakuhai-green'
  | 'yakuhai-red'
  | 'rinshan-kaihou'
  | 'chankan'
  | 'haitei'
  | 'houtei'
  | 'toitoi'
  | 'sanshoku-doujun'
  | 'sanshoku-doukou'
  | 'ittsu'
  | 'chiitoitsu'
  | 'chanta'
  | 'sanankou'
  | 'sankantsu'
  | 'honroutou'
  | 'shousangen'
  | 'honitsu'
  | 'junchan'
  | 'ryanpeikou'
  | 'chinitsu';

export type YakumanYakuId =
  | 'tenhou'
  | 'chiihou'
  | 'kokushi-musou'
  | 'kokushi-13-wait'
  | 'suuankou'
  | 'suuankou-tanki'
  | 'daisangen'
  | 'shousuushii'
  | 'daisuushii'
  | 'tsuuiisou'
  | 'chinroutou'
  | 'ryuuiisou'
  | 'chuuren-poutou'
  | 'junsei-chuuren-poutou'
  | 'suukantsu';

export type YakuId = RegularYakuId | YakumanYakuId;

export type YakuEntry =
  | {
      readonly kind: 'regular';
      readonly id: RegularYakuId;
      readonly han: number;
    }
  | {
      readonly kind: 'yakuman';
      readonly id: YakumanYakuId;
    };

export interface DoraContribution {
  readonly dora: number;
  readonly akaDora: number;
}

export type FuCalculation =
  | {
      readonly kind: 'standard';
      readonly base: number;
      readonly menzenRon: number;
      readonly tsumo: number;
      readonly melds: number;
      readonly pair: number;
      readonly wait: number;
      readonly rawTotal: number;
      readonly rounded: number;
    }
  | {
      readonly kind: 'chiitoitsu';
      readonly fixed: 25;
    };

export type LimitClassification =
  | {
      readonly kind: 'mangan';
      readonly kiriage: boolean;
    }
  | {
      readonly kind: 'haneman';
    }
  | {
      readonly kind: 'baiman';
    }
  | {
      readonly kind: 'sanbaiman';
    }
  | {
      readonly kind: 'yakuman';
      readonly units: number;
      readonly counted: boolean;
    };

export type ScoringPayment =
  | {
      readonly kind: 'ron';
      readonly amount: number;
    }
  | {
      readonly kind: 'tsumo-dealer';
      readonly eachOpponent: number;
    }
  | {
      readonly kind: 'tsumo-non-dealer';
      readonly dealerPays: number;
      readonly nonDealerPays: number;
    };
```

`YakuId`, `FuCalculation`, and `LimitClassification` are mjtensu-owned result semantics. They are not aliases of Agari enums or WASM response types.

Regular-yaku `han` is the awarded value after open/closed and active-rule effects have been applied by the scoring engine. Application/UI must not recalculate kuisagari or other yaku han adjustments.

Yakuman entries preserve detected identity only. Per-yaku score multipliers are intentionally not exposed on `YakuEntry`; the authoritative final yakuman multiplier is `LimitClassification` with `kind: 'yakuman'` and its `units` field after `doubleYakumanVariants` and `multipleYakuman` have been applied.

`FuCalculation.kind === 'standard'` mirrors Agari's aggregate fu breakdown. It does not fabricate per-meld contributor records that the engine does not expose. `kind === 'chiitoitsu'` preserves the semantic meaning of fixed 25-fu seven pairs instead of presenting it as ordinary 25 base fu. Yakuman-class calculations use `fu: null` because fu does not determine their payment.

`LimitClassification` is `null` for non-limit ordinary scores. Kiriage is represented only on mangan; counted versus actual yakuman is represented explicitly while `units` remains the authoritative final yakuman-unit count.

`ScoringCalculation.han` is the total regular-yaku plus bonus-dora han for non-actual-yakuman results, including a counted-yakuman result when that rule is enabled. It is `null` for actual-yakuman scoring, where han is not the score authority. `fu` is `null` for both actual and counted yakuman-class limits because fu does not affect their payment.

Product-visible yaku names are presentation data derived from `YakuId`; display strings from Agari are not part of this API.

## Calculate failure boundary

`calculate()` is the strict calculation operation. Its normal success return is only `ScoringCalculation`; preview states such as `incomplete`, `invalid-winning-shape`, and `no-yaku` are not mixed into the successful calculation return type.

Application is expected to call `calculate()` only after the current state has been normalized into a strict `ScoringInput` and is calculation-ready.

The scoring implementation must still reject contract violations or adapter/runtime failures defensively rather than fabricate a result. Such failures use `ScoringError` from `spec:product.system.contracts.runtime_errors`.

## Concrete scoring-library isolation

Only scoring infrastructure/adapter code may import the concrete riichi-mahjong scoring library.

Application and UI must not:

- construct concrete library hand/mentsu objects;
- use concrete library tile IDs as `TileInstanceId`;
- inspect concrete library error codes to determine presentation state;
- reimplement winning-hand decomposition, yaku detection, fu, or point calculation for preview purposes.

The adapter owns translation between this contract and the concrete library.

## Boundary

| concern | owner |
|---|---|
| Winning-structure validation, draft/strict scoring API, and normalized preview/calculation result | This contract. |
| Product scoring semantics | `spec:product.scoring`. |
| Canonical tile/recognition model | `spec:product.system.concepts.canonical_tile_model`. |
| Selection/correction/session transitions | Application contracts. |
| Concrete Agari mapping | `spec:product.system.contracts.agari_adapter`. |
| Required production Agari fork behavior | `spec:product.system.contracts.agari_fork`. |
| Error taxonomy | `spec:product.system.contracts.runtime_errors`. |
