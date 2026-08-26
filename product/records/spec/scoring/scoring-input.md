# Contract: Scoring input

- **id**: `spec:product.scoring.input`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.scoring`

## What this is

Semantic input required to calculate one riichi-mahjong winning hand.
The contract is independent of the concrete scoring-library data model.

## Tile input

| concept | requirement |
|---|---|
| Completed-hand tiles | Ordered tile instances recognized in the completed-hand region. The winning tile is included in this collection. |
| Winning tile | Exactly one tile instance from the completed-hand tiles. A meld tile cannot be the winning tile. |
| Melds | Zero or more logical score-relevant meld groups with their tile identities and open/closed/kan semantics. A concealed kan is supplied as a full concealed-kan semantic meld even when camera recognition observed only its two face-up tiles. |
| Dora indicators | Zero or more indicator tiles that should count for this winning hand. The input does not distinguish visible, kan, ura, or kan-ura source; the user is responsible for supplying every applicable indicator and omitting inapplicable ones. |
| Red fives | Red-five identity is preserved and contributes as aka dora according to the active riichi scoring profile. |

Tile-instance identity must be retained separately from tile kind so that one physical tile can remain selected as the winning tile even when the same tile kind appears multiple times.

## Non-image conditions

The scoring input accepts the following conditions because they cannot be determined reliably from tile faces alone.

| condition | values / meaning |
|---|---|
| Win method | `ron` or `tsumo`. |
| Round wind | East, South, West, or North. |
| Seat wind | East, South, West, or North. East means the winner is dealer. |
| Riichi state | none, riichi, or double riichi. |
| Ippatsu | Whether ippatsu is active. |
| Rinshan kaihou | Whether the win is by rinshan draw. |
| Chankan | Whether the win is by robbing a kan. |
| Haitei | Whether the tsumo win is on the last drawable tile. |
| Houtei | Whether the ron win is on the last discard. |
| Tenhou | Whether the dealer wins by the initial hand condition. |
| Chiihou | Whether a non-dealer wins by the initial uninterrupted draw condition. |

## Derived conditions

The application must not ask for a condition that can be derived from the supplied structure without ambiguity.
Examples include:

- dealer status from `seat wind = East`;
- menzen/closed-hand status from the completed-hand and meld structure;
- menzen tsumo from closed-hand status plus `win method = tsumo`;
- ordinary tile-composition yaku from tile identities and meld structure;
- dora count from the supplied dora indicators;
- aka-dora count from red-five identity.

## Condition consistency

Before score calculation, the input must represent one internally consistent winning situation.
At minimum:

- exactly one winning tile must be selected from completed-hand tiles;
- riichi or double riichi requires a closed hand;
- ippatsu requires riichi or double riichi;
- rinshan kaihou requires tsumo and at least one logical kan in the winner's meld structure;
- chankan requires ron;
- haitei requires tsumo;
- houtei requires ron;
- rinshan kaihou and haitei cannot both describe the same win;
- chankan and houtei cannot both describe the same win;
- tenhou requires dealer status, tsumo, a closed no-meld structure, no riichi/double-riichi, and no other situational condition;
- chiihou requires non-dealer status, tsumo, a closed no-meld structure, no riichi/double-riichi, and no other situational condition;
- tenhou and chiihou are mutually exclusive;
- mutually exclusive situational conditions must not be presented simultaneously as one scoring fact.

Ordinary UI interaction uses `spec:product.system.contracts.scoring_condition_policy` to prevent or clear contradictions that depend only on condition fields. Structure-dependent requirements remain part of this scoring-input validation.
The scoring boundary must still reject an internally contradictory input rather than silently rewriting it.

## Explicitly excluded input

The current scoring contract does not accept:

- kyoku number within the round;
- honba count;
- riichi-stick pool / number of deposited riichi sticks;
- other players' riichi state;
- player scores before the win;
- exhaustive-draw settlement;
- nagashi mangan settlement;
- chombo, penalty, or tournament settlement state;
- responsibility-payment / pao settlement.

These exclusions mean the product calculates the winning hand's score and payment structure, not the complete table settlement.

## Scoring-rule profile boundary

The PWA does not expose arbitrary house-rule configuration in the current flow.
The current product uses one fixed `DEFAULT_RULE_PROFILE` representing common modern competitive Japanese riichi scoring.

| rule | default |
|---|---|
| Open tanyao | Enabled. |
| Aka dora | Enabled; recognized red-five identities count as aka dora. |
| Dora indicators | Enabled; every indicator supplied by the user contributes normally. |
| Ippatsu | Enabled. |
| Kiriage mangan | Enabled. |
| Kazoe yakuman | Disabled; non-yakuman hands are capped at sanbaiman. |
| Multiple yakuman | Enabled. |
| Double-yakuman variants such as kokushi 13-sided wait or suuankou tanki | Disabled; score as one yakuman each. |
| Double-wind pair fu | 2 fu. |
| Chiitoitsu | 25 fu. |
| Pinfu tsumo | 20 fu. |
| Open pinfu-shaped ron hand | 30 fu minimum. |

The fixed profile is a product-owned semantic policy rather than a branded tournament preset. The scoring-library adapter must map it explicitly to the concrete library configuration and must not inherit changing library defaults silently.

Rule-profile choices that change point semantics require an explicit product decision before they become configurable or change from `DEFAULT_RULE_PROFILE`.

## Boundary

| concern | owner |
|---|---|
| Semantic score input | This contract. |
| Camera-derived semantic tile structure | `spec:product.recognition.runtime_recognition`. |
| Default winning-tile selection and edits | `spec:product.application.scoring_session`. |
| Conditions-page controls | `spec:product.ui.pages.conditions`. |
| Concrete library input conversion | Implementation. |
